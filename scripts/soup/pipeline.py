#!/usr/bin/env python3
"""`make soup` orchestrator: uncond inter-train → seeded fine-tunes → SVD soup.

Ships the recipe validated in ``bench/memorization/report.md`` (the uncond-init
ladder, 2026-07-04/05) as one pipeline:

1. **Uncond inter-train** (Self-Soupervision, arXiv:2602.02890): a short
   ``caption_dropout_rate 1.0`` run on a *diluted* pool. The pool is selected by
   ``--pool_path_pattern`` (default ``*`` = the whole dataset) and diluted by
   ``--uncond_ratio`` (train.py's ``--sample_ratio``); the fine-tune selection
   (``--path_pattern``) is always unioned into it. Dose rule from the report:
   uncond member exposure = epochs × target share — 2 epochs at a minority share
   is memorization-neutral while buying the quality win; same-frame full-dose
   uncond is destructive, which is why the pool must be *broader* than the
   fine-tune set. The checkpoint name is deterministic in (pool, ratio, epochs),
   so it is trained **once** and reused across any soup drawing the same pool.
2. **Seeded fine-tunes**: N normal captioned runs on the ``--path_pattern``
   images, ``--network_weights``-initialized from the uncond checkpoint. Souping
   absorbs the training-seed lottery (a catastrophic draw is invisible to
   loss/AUC — report Act 5).
3. **ΔW soup, SVD-truncated to the ingredient rank** (``scripts/soup/build.py``)
   so the artifact stays single-adapter-sized (~99.9% retained energy on
   shared-init ingredients). The first ingredient's ``.snapshot.toml`` is
   copied next to the soup — the memorization probes replay membership from it.

Selection is by fnmatch **path_pattern** (``|``-separated alternatives, matched
against each image's path relative to its subset ``image_dir``) rather than a
single artist directory — so a soup can span any glob-expressible slice of the
dataset, not just one artist. The output slug derives from ``--path_pattern``
(or an explicit ``--name``).

Runs ``train.py`` as direct subprocesses (single daemon command job — never
submit nested daemon jobs from here; that deadlocks the serial queue).
Unknown CLI args are forwarded to the fine-tune runs (that's how
``make soup ARGS="--network_dim 32"`` reaches Phase 2).
"""

from __future__ import annotations

import argparse
import hashlib
import re
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


def pool_glob(pool_pattern: str, ft_pattern: str) -> str:
    """The Phase-1 uncond pool glob: ``pool_pattern`` with the fine-tune
    selection unioned in (duplicate ``|`` alternatives dropped so an image
    matched by both patterns is still kept once). ``*`` (match everything)
    already covers the fine-tune set, so it is returned as-is."""
    pool_pattern = (pool_pattern or "*").strip()
    if pool_pattern in ("", "*"):
        return "*"
    alts = [a for a in pool_pattern.split("|") if a]
    for a in ft_pattern.split("|"):
        if a and a not in alts:
            alts.append(a)
    return "|".join(alts)


def slug_for_pattern(pattern: str) -> str:
    """A filename-safe slug for a path_pattern. A plain ``<dir>/*`` (single
    alternative, no other glob metachars) → the dir's basename, so the common
    per-artist case stays ``anima_soup_<artist>``; anything richer → a sanitized
    prefix plus a short hash of the full pattern for uniqueness."""
    pattern = pattern.strip()
    m = re.fullmatch(r"([^|*?\[\]]+)/\*", pattern)
    if m:
        base = m.group(1).rstrip("/").split("/")[-1]
        slug = re.sub(r"[^0-9A-Za-z._-]+", "_", base).strip("_")
        if slug:
            return slug
    slug = re.sub(r"[^0-9A-Za-z]+", "_", pattern).strip("_")[:32]
    digest = hashlib.sha1(pattern.encode()).hexdigest()[:6]
    return f"{slug}_{digest}" if slug else f"pat_{digest}"


def uncond_name(pool: str, ratio: float, epochs: int) -> str:
    """Deterministic uncond checkpoint name — same pool+dose → same artifact."""
    digest = hashlib.sha1(pool.encode()).hexdigest()[:8]
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
    ap.add_argument(
        "--path_pattern",
        required=True,
        help="fnmatch glob (| = alternatives) selecting the fine-tune images, "
        "matched against each image's path relative to its subset image_dir. "
        "Replaces the old --target artist dir.",
    )
    ap.add_argument(
        "--pool_path_pattern",
        default=d.get("pool_path_pattern", "*"),
        help="Phase-1 uncond pool selection glob; default '*' (whole dataset), "
        "diluted by --uncond_ratio. The fine-tune set is always unioned in.",
    )
    ap.add_argument(
        "--name",
        default=None,
        help="output slug → output/ckpt/anima_soup_<name>.safetensors "
        "(default: derived from --path_pattern).",
    )
    ap.add_argument("--uncond_ratio", type=float, default=d.get("uncond_ratio", 0.5))
    ap.add_argument("--uncond_epochs", type=int, default=d.get("uncond_epochs", 2))
    ap.add_argument(
        "--num_soup",
        type=int,
        default=d.get("num_soup", 3),
        help="number of seeded fine-tunes to soup (seeds are UNCOND_SEED+1..+N, "
        "i.e. 1001..1000+N). Must be >= 2.",
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

    if args.num_soup < 2:
        raise SystemExit(
            f"--num_soup must be >= 2 (a soup needs at least 2 ingredients); "
            f"got {args.num_soup}."
        )
    # Seeds are derived from the count: UNCOND_SEED+1 .. UNCOND_SEED+num_soup
    # (1001..1000+N), deterministic so re-runs reproduce the same ingredients.
    seeds = [UNCOND_SEED + i for i in range(1, args.num_soup + 1)]

    ckpt_dir = ROOT / "output" / "ckpt"
    name = args.name or slug_for_pattern(args.path_pattern)
    pool = pool_glob(args.pool_path_pattern, args.path_pattern)
    print(f"[soup] fine-tune pattern {args.path_pattern!r} -> slug {name!r}")
    print(f"[soup] uncond pool pattern {pool!r}")

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
                    pool,
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
    for seed in seeds:
        iname = f"anima_soup_{name}_s{seed}"
        ingredients.append(ckpt_dir / f"{iname}.safetensors")
        print(f"[soup] phase 2: fine-tune seed {seed} -> {iname}")
        _train(
            build_method_args(
                args.method,
                preset=args.preset,
                extra=[
                    "--path_pattern",
                    args.path_pattern,
                    "--network_weights",
                    str(uncond_path),
                    "--seed",
                    str(seed),
                    "--output_name",
                    iname,
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
    soup_path = ckpt_dir / f"anima_soup_{name}.safetensors"
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
