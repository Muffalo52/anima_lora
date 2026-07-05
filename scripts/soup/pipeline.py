#!/usr/bin/env python3
"""`make soup` orchestrator: uncond inter-train → seeded fine-tunes → SVD soup.

Ships the recipe validated in ``bench/memorization/report.md`` (the uncond-init
ladder, 2026-07-04/05) as one pipeline:

1. **Uncond inter-train** (Self-Soupervision, arXiv:2602.02890): a short
   ``caption_dropout_rate 1.0`` run on a *diluted* artist pool. Dose rule from
   the report: uncond member exposure = epochs × target share — 2 epochs at a
   ~minority share is neutral on memorization while buying the quality win;
   same-frame full-dose uncond is destructive. Default pool = the round-robin
   artist shard containing the target (``--pool_shard_n``), at
   ``--sample_ratio 0.5`` (the Act-3 operating point). The checkpoint name is
   deterministic in (pool, ratio, epochs), so it is trained **once** and
   reused across targets in the same shard.
2. **Seeded fine-tunes**: N normal captioned runs on the target's images,
   ``--network_weights``-initialized from the uncond checkpoint. Souping
   absorbs the training-seed lottery (a catastrophic draw is invisible to
   loss/AUC — report Act 5).
3. **ΔW soup, SVD-truncated to the ingredient rank** (``scripts/soup/build.py``)
   so the artifact stays single-adapter-sized (~99.9% retained energy on
   shared-init ingredients). The first ingredient's ``.snapshot.toml`` is
   copied next to the soup — the memorization probes replay membership from it.

Runs ``train.py`` as direct subprocesses (single daemon command job — never
submit nested daemon jobs from here; that deadlocks the serial queue).
Unknown CLI args are forwarded to the fine-tune runs (that's how
``make soup ARGS="--network_dim 32"`` reaches Phase 2).
"""

from __future__ import annotations

import argparse
import glob as _glob
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # allow `python scripts/soup/pipeline.py`

from scripts.tasks._common import build_launch_cmd, build_method_args, run  # noqa: E402

UNCOND_SEED = 1000  # fixed: the uncond init is a shared, reusable artifact


def _soup_defaults() -> dict:
    """The ``[soup]`` table from configs/soup/soup.toml — pipeline knob defaults
    (pool / dose / seeds / rank). Stripped from the train.py merge (registered as
    a metadata section), so it lives here purely to seed this argparser. Missing
    file / table → ``{}`` (argparse hard-coded fallbacks below still apply)."""
    import toml

    path = ROOT / "configs" / "soup" / "soup.toml"
    if not path.exists():
        return {}
    return toml.load(path).get("soup", {})


def resolve_pool(
    target: str,
    pool: list[str] | None,
    shard_n: int,
    resized_dir: str | None = None,
) -> list[str]:
    """The uncond pool artist list: explicit ``pool`` (target auto-added), else
    the round-robin shard (of ``shard_n``) that contains ``target``."""
    from library.datasets.artist_shard import list_artists, shard_of

    if pool:
        return sorted(set(pool) | {target})

    if resized_dir is None:
        from library.config.io import load_path_overrides
        from library.env import resolve_under_home

        resized_dir = load_path_overrides().get(
            "resized_image_dir", "post_image_dataset/resized"
        )
        resized_dir = str(resolve_under_home(resized_dir))
    resized = resized_dir
    artists = list_artists(resized)
    if target not in artists:
        raise SystemExit(
            f"--target {target!r} is not an artist directory under {resized} "
            f"({len(artists)} artists found)."
        )
    n = min(shard_n, len(artists))
    k = artists.index(target) % n + 1  # the 1-indexed shard target falls in
    return shard_of(artists, k, n)


def pool_pattern(pool: list[str]) -> str:
    """``|``-joined ``<artist>/*`` globs (the path_pattern alternation syntax)."""
    return "|".join(f"{_glob.escape(a)}/*" for a in pool)


def uncond_name(pool: list[str], ratio: float, epochs: int) -> str:
    """Deterministic uncond checkpoint name — same pool+dose → same artifact."""
    digest = hashlib.sha1("|".join(sorted(pool)).encode()).hexdigest()[:8]
    ratio_tag = f"{ratio:g}".replace(".", "p")
    return f"anima_uncond_{digest}_r{ratio_tag}_e{epochs}"


def _train(args_list: list[str], dry_run: bool) -> None:
    cmd = build_launch_cmd(*args_list)
    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return
    run(cmd)


def main() -> None:
    d = _soup_defaults()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--target", required=True, help="artist dir under resized/")
    ap.add_argument(
        "--pool",
        nargs="+",
        default=None,
        help="explicit uncond pool artists (target auto-added); default = the "
        "artist shard containing the target",
    )
    ap.add_argument("--pool_shard_n", type=int, default=d.get("pool_shard_n", 16))
    ap.add_argument("--uncond_ratio", type=float, default=d.get("uncond_ratio", 0.5))
    ap.add_argument("--uncond_epochs", type=int, default=d.get("uncond_epochs", 2))
    ap.add_argument(
        "--seeds", nargs="+", type=int, default=d.get("seeds", [1001, 1002, 1003])
    )
    ap.add_argument(
        "--rank",
        type=int,
        default=d.get("rank"),
        help="SVD truncation rank (default: the method config's network_dim)",
    )
    ap.add_argument("--method", default="soup")
    ap.add_argument("--preset", default="default")
    ap.add_argument("--dry_run", action="store_true")
    args, ft_extra = ap.parse_known_args()

    ckpt_dir = ROOT / "output" / "ckpt"
    pool = resolve_pool(args.target, args.pool, args.pool_shard_n)
    print(f"[soup] pool ({len(pool)} artists): {', '.join(pool)}")

    # Phase 1 — uncond inter-train (skipped when the init already exists).
    uncond = uncond_name(pool, args.uncond_ratio, args.uncond_epochs)
    uncond_path = ckpt_dir / f"{uncond}.safetensors"
    if uncond_path.exists():
        print(f"[soup] phase 1: reusing existing uncond init {uncond_path}")
    else:
        print(f"[soup] phase 1: training uncond init {uncond}")
        _train(
            build_method_args(
                args.method,
                preset=args.preset,
                extra=[
                    "--path_pattern",
                    pool_pattern(pool),
                    "--caption_dropout_rate",
                    "1.0",
                    "--sample_ratio",
                    str(args.uncond_ratio),
                    "--seed",
                    str(UNCOND_SEED),
                    "--max_train_epochs",
                    str(args.uncond_epochs),
                    "--save_every_n_epochs",
                    str(args.uncond_epochs),
                    "--checkpointing_epochs",
                    str(args.uncond_epochs),
                    "--output_name",
                    uncond,
                ],
            ),
            args.dry_run,
        )

    # Phase 2 — captioned fine-tunes from the shared init, one per seed.
    ingredients: list[Path] = []
    for seed in args.seeds:
        name = f"anima_soup_{args.target}_s{seed}"
        ingredients.append(ckpt_dir / f"{name}.safetensors")
        print(f"[soup] phase 2: fine-tune seed {seed} -> {name}")
        _train(
            build_method_args(
                args.method,
                preset=args.preset,
                extra=[
                    "--path_pattern",
                    f"{_glob.escape(args.target)}/*",
                    "--network_weights",
                    str(uncond_path),
                    "--seed",
                    str(seed),
                    "--output_name",
                    name,
                    *ft_extra,
                ],
            ),
            args.dry_run,
        )

    # Phase 3 — ΔW soup, SVD-truncated back to the single-adapter rank.
    if args.rank is not None:
        rank = args.rank
    else:
        from library.config.io import load_method_preset

        rank = int(load_method_preset(args.method, args.preset).get("network_dim", 16))
    soup_path = ckpt_dir / f"anima_soup_{args.target}.safetensors"
    print(f"[soup] phase 3: rank-{rank} SVD soup of {len(ingredients)} -> {soup_path}")
    if args.dry_run:
        print(
            f"  [dry-run] soup {', '.join(p.name for p in ingredients)} "
            f"--rank {rank} --out {soup_path}"
        )
        return
    from scripts.soup.build import build_soup_file

    build_soup_file([str(p) for p in ingredients], str(soup_path), rank=rank)
    # Probes (bench/memorization/*) replay training membership from the
    # snapshot; the ingredients share data/config, so the first one stands in.
    snapshot = ingredients[0].with_suffix("").with_suffix(".snapshot.toml")
    if snapshot.exists():
        shutil.copy(snapshot, soup_path.with_suffix("").with_suffix(".snapshot.toml"))
    print(f"[soup] done: {soup_path}")


if __name__ == "__main__":
    main()
