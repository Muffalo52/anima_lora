#!/usr/bin/env python
"""Stratified tagger eval — the bench the KB-groups work never had.

Runs a trained anima-tagger checkpoint over its held-out split (cached PE
features, no re-encoding) and scores EVERY tag through the real inference
decision rule — per-tag calibrated thresholds + softmax-group argmax
refinement gated on *predicted* solo/escape — then stratifies the result:

  * by danbooru-KB taxonomy slice (eye_color, pose, clothing_details, …,
    plus cat:artist / cat:character / … for non-general categories),
  * by train-frequency tier (head ≥ --freq_head_min, mid ≥ --freq_mid_min,
    tail below).

Fixes the two measurement gaps that made prior group-loss A/Bs unreadable:
the trainer's macro-F1 EXCLUDES softmax-group tags (argmax-only at
inference), and nothing reported where — which taxonomy family, which
frequency tier — a change actually moved. ``macro_f1_all`` here includes
softmax tags via the same code path inference uses; ``mean AP`` is
threshold-free so it compares heads (e.g. linear vs label-embedding)
without threshold-calibration confounds.

READOUT: bench/tagger_eval/results/<ts>[-label]/{per_tag.csv,
         per_slice_kb.csv, per_slice_freq.csv, summary.md, result.json}

Run from anima_lora/::

    uv run python bench/tagger_eval/run_bench.py --label v2-baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--model_dir",
        default="models/captioners/anima-tagger-v2",
        help="Checkpoint dir (model.safetensors + config.json + vocab.json + "
        "dataset.json [+ thresholds.safetensors + rules.yaml]).",
    )
    p.add_argument(
        "--split",
        choices=["val", "train"],
        default="val",
        help="Manifest split to score (default: val).",
    )
    p.add_argument(
        "--tag_cache",
        default="models/danbooru_tags_classified.csv",
        help="Danbooru taxonomy KB CSV — drives the KB slice assignment. "
        "Use the KR CSV (its descriptions carry the [category path] prefix); "
        "the .en.csv has no taxonomy paths.",
    )
    p.add_argument(
        "--feature_cache_dir",
        default=None,
        help="Feature-cache root override (default: post_image_dataset/anima_tagger).",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--feature_cache_workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", default=None, help="Run-dir label suffix.")
    p.add_argument(
        "--freq_head_min",
        type=int,
        default=1000,
        help="Train-freq floor for the 'head' tier (default: 1000).",
    )
    p.add_argument(
        "--freq_mid_min",
        type=int,
        default=200,
        help="Train-freq floor for the 'mid' tier (default: 200; below = tail).",
    )
    return p.parse_args()


def collect_logits(args, model_dir: Path, cfg_d: dict, split_stems: List[str]):
    """Forward the split through the dual-encoder loader.

    Returns ``(tag_logits [N, n_tags] fp32 cpu, multi_hot [N, n_tags] cpu)``
    in loader-emission order (mh comes from the same batches as the logits,
    so the two always align).
    """
    from torch.utils.data import DataLoader

    from library.captioning.anima_tagger_data import (
        BucketBatchSampler,
        CachedDualDataset,
        TaggerManifest,
        collate_dual_token_batch,
    )
    from library.captioning.anima_tagger_model import AnimaTaggerConfig, AnimaTaggerHead
    from library.vision.encoders import get_encoder_info
    from safetensors.torch import load_file as st_load
    from scripts.anima_tagger.caches import cache_dir_for, feature_cache_root

    cfg = AnimaTaggerConfig.from_dict(cfg_d["model"])
    model = AnimaTaggerHead(cfg)
    model.load_state_dict(st_load(str(model_dir / "model.safetensors")))
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device).eval()

    pool_kind = str(cfg_d.get("pool_kind", cfg.pool_kind))
    pool_kind_aux = str(cfg_d.get("pool_kind_aux", cfg.pool_kind_aux))
    encoder = cfg_d.get("encoder", "pe")
    aux_encoder = cfg_d.get("aux_encoder", "pe_spatial")

    feature_root = feature_cache_root(args)
    cache_dir = cache_dir_for(feature_root, pool_kind, encoder)
    cache_dir_aux = cache_dir_for(feature_root, pool_kind_aux, aux_encoder)
    for d in (cache_dir, cache_dir_aux):
        if not d.exists():
            raise SystemExit(
                f"missing feature cache {d} — run `--mode build_features` with "
                f"the same encoder/pool flags the trainer used."
            )
    spec = get_encoder_info(encoder).bucket_spec if pool_kind == "map" else None
    spec_aux = (
        get_encoder_info(aux_encoder).bucket_spec if pool_kind_aux == "map" else None
    )
    manifest = TaggerManifest.from_path(model_dir / "dataset.json")
    ds = CachedDualDataset(
        manifest,
        cache_dir,
        pool_kind,
        spec,
        cache_dir_aux,
        pool_kind_aux,
        spec_aux,
        stems_subset=split_stems,
    )
    sampler = BucketBatchSampler(
        ds.buckets, batch_size=args.batch_size, seed=args.seed, shuffle=False
    )
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=args.feature_cache_workers,
        collate_fn=collate_dual_token_batch,
        pin_memory=True,
    )
    logit_chunks: List[torch.Tensor] = []
    mh_chunks: List[torch.Tensor] = []
    autocast = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    with torch.no_grad(), autocast:
        for batch in loader:
            tokens, tokens_aux, mh, _rate, _people, _bucket = batch
            tl, _rl, _pl = model(
                tokens.to(device, non_blocking=True),
                tokens_aux.to(device, non_blocking=True),
            )
            logit_chunks.append(tl.float().cpu())
            mh_chunks.append(mh)
    return torch.cat(logit_chunks, dim=0), torch.cat(mh_chunks, dim=0), device


def _fmt(x: float) -> str:
    return "nan" if (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def write_slice_csv(path: Path, rows: List[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    with open(model_dir / "config.json") as f:
        cfg_d = json.load(f)
    with open(model_dir / "vocab.json") as f:
        vocab = json.load(f)
    with open(model_dir / "dataset.json") as f:
        split_stems = json.load(f)["split"][args.split]

    from library.captioning import tag_rules as tr
    from library.captioning.correction import load_tag_knowledge_base
    from scripts.anima_tagger.eval_metrics import (
        aggregate_slices,
        assign_slices,
        micro_f1,
        per_tag_average_precision,
        per_tag_prf,
        predict_with_inference_rule,
    )
    from scripts.anima_tagger.train_common import GroupRouter

    tag_logits, multi_hot, device = collect_logits(args, model_dir, cfg_d, split_stems)
    n_samples, n_tags = tag_logits.shape
    log.info(
        "collected logits: N=%d n_tags=%d (split=%s)", n_samples, n_tags, args.split
    )

    # Thresholds: calibrated when present, flat 0.5 otherwise.
    thresh_path = model_dir / "thresholds.safetensors"
    if thresh_path.exists():
        from safetensors.torch import load_file as st_load

        thresholds = st_load(str(thresh_path))["thresholds"].float()
        thresh_source = "calibrated"
    else:
        thresholds = torch.full((n_tags,), 0.5)
        thresh_source = "flat-0.5"
    log.info("thresholds: %s", thresh_source)

    # Router (index sets only — the split multi_hot just feeds its pos-weight).
    router = GroupRouter.from_vocab(vocab, multi_hot, device=torch.device("cpu"))

    # KB slices, with the same rename-recovery derive_groups uses.
    kb = load_tag_knowledge_base(args.tag_cache)
    rules_path = model_dir / "rules.yaml"
    rename_recovery: Dict[str, str] = {}
    if rules_path.exists():
        rules = tr.load_rules(rules_path)
        rename_recovery = {tgt: src for src, tgt in rules.replacements}
    slices = assign_slices(
        vocab,
        kb,
        rename_recovery,
        freq_cuts=(args.freq_head_min, args.freq_mid_min),
    )

    # ---- Per-tag metrics through the real inference rule. ----
    pred = predict_with_inference_rule(tag_logits, thresholds, router)
    prec, rec, f1, support = per_tag_prf(pred, multi_hot)
    ap = per_tag_average_precision(tag_logits.sigmoid(), multi_hot)
    supported = support > 0
    n_supported = int(supported.sum().item())

    # Continuity metric: the trainer's residual macro-F1 (flat 0.5 threshold,
    # softmax-group tags excluded) so runs stay comparable with train_history.
    keep = torch.ones(n_tags, dtype=torch.bool)
    if router.softmax_member_indices is not None:
        keep[router.softmax_member_indices] = False
    pred_05 = tag_logits.sigmoid() > 0.5
    _, _, f1_res, sup_res = per_tag_prf(pred_05[:, keep], multi_hot[:, keep])
    macro_f1_residual = float(f1_res[sup_res > 0].mean().item())

    headline = {
        "split": args.split,
        "n_samples": n_samples,
        "n_tags": n_tags,
        "n_supported": n_supported,
        "thresholds": thresh_source,
        "macro_f1_all": float(f1[supported].mean().item()),
        "mean_ap_all": float(ap[supported].nanmean().item()),
        "micro_f1_all": micro_f1(pred, multi_hot),
        "macro_f1_residual_at_0p5": macro_f1_residual,
    }

    kb_rows = aggregate_slices(
        slices.indices_by("kb"), pred, multi_hot, f1, ap, support
    )
    freq_rows = aggregate_slices(
        slices.indices_by("freq"), pred, multi_hot, f1, ap, support
    )

    # ---- Artifacts. ----
    run_dir = make_run_dir("tagger_eval", label=args.label)
    per_tag_path = run_dir / "per_tag.csv"
    with open(per_tag_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "index",
                "name",
                "category",
                "kb_slice",
                "freq_tier",
                "freq",
                "support",
                "threshold",
                "precision",
                "recall",
                "f1",
                "ap",
            ]
        )
        for t in vocab["tags"]:
            i = int(t["index"])
            w.writerow(
                [
                    i,
                    t["name"],
                    t["category"],
                    slices.kb_slice[i],
                    slices.freq_tier[i],
                    int(t.get("freq", 0)),
                    int(support[i].item()),
                    f"{float(thresholds[i].item()):.2f}",
                    _fmt(float(prec[i].item())),
                    _fmt(float(rec[i].item())),
                    _fmt(float(f1[i].item())),
                    _fmt(float(ap[i].item())),
                ]
            )
    kb_csv = run_dir / "per_slice_kb.csv"
    freq_csv = run_dir / "per_slice_freq.csv"
    write_slice_csv(kb_csv, kb_rows)
    write_slice_csv(freq_csv, freq_rows)

    lines = [
        f"# tagger_eval — {args.label or model_dir.name} ({args.split})",
        "",
        f"N={n_samples} samples, {n_tags} tags ({n_supported} with split support), "
        f"thresholds={thresh_source}.",
        "",
        "## Headline",
        "",
        f"- **macro-F1 (all tags, inference rule)**: {_fmt(headline['macro_f1_all'])}",
        f"- **mean AP (all tags, threshold-free)**: {_fmt(headline['mean_ap_all'])}",
        f"- micro-F1 (all cells): {_fmt(headline['micro_f1_all'])}",
        f"- residual macro-F1 @0.5 (trainer-comparable): {_fmt(macro_f1_residual)}",
        "",
        "## Frequency tiers",
        "",
        "| tier | tags | supported | macro-F1 | mean AP | micro-F1 |",
        "|---|---|---|---|---|---|",
    ]
    for r in freq_rows:
        lines.append(
            f"| {r['slice']} | {r['n_tags']} | {r['n_supported']} | "
            f"{_fmt(r['macro_f1'])} | {_fmt(r['mean_ap'])} | {_fmt(r['micro_f1'])} |"
        )
    lines += [
        "",
        "## KB taxonomy slices",
        "",
        "| slice | tags | supported | macro-F1 | mean AP | micro-F1 |",
        "|---|---|---|---|---|---|",
    ]
    for r in kb_rows:
        lines.append(
            f"| {r['slice']} | {r['n_tags']} | {r['n_supported']} | "
            f"{_fmt(r['macro_f1'])} | {_fmt(r['mean_ap'])} | {_fmt(r['micro_f1'])} |"
        )
    summary_path = run_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")

    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics={
            **headline,
            "slices_kb": kb_rows,
            "slices_freq": freq_rows,
        },
        artifacts=[per_tag_path, kb_csv, freq_csv, summary_path],
        device=device,
    )
    print(f"\nwrote {run_dir}/")
    print("\n".join(lines[4:14]))


if __name__ == "__main__":
    main()
