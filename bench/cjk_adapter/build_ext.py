#!/usr/bin/env python3
"""Build the CJK extended-vocab sidecar for the LLM Adapter (Phase 1, no training).

Reads only two tensors (no model instantiation):
  - ``net.llm_adapter.embed.weight``  [32128, 1024] from the DiT checkpoint
  - ``model.embed_tokens.weight``     [151936, 1024] from the Qwen3 encoder

Fits the anchor ridge map W on surface-identical tokens, then writes
``assets/ext_embed.safetensors`` (rows, fp32) + ``assets/ext_embed.json``
(id mapping + build stats). CPU-only, no GPU needed.

    uv run python bench/cjk_adapter/build_ext.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from anima_lora import default_checkpoints  # noqa: E402
from bench.cjk_adapter import ext_vocab  # noqa: E402
from library.anima.weights import (  # noqa: E402
    load_qwen3_tokenizer,
    load_t5_tokenizer,
)
from library.env import resolve_under_home  # noqa: E402

ASSET_PREFIX = Path(__file__).resolve().parent / "assets" / "ext_embed"


def read_tensor(path: str, key: str) -> torch.Tensor:
    with safe_open(str(resolve_under_home(path)), framework="pt") as f:
        return f.get_tensor(key)


def main() -> None:
    ckpt = default_checkpoints()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dit", default=ckpt.dit)
    parser.add_argument("--text_encoder", default=ckpt.text_encoder)
    parser.add_argument("--ridge", type=float, default=1e-2)
    opts = parser.parse_args()

    t5_embed = read_tensor(opts.dit, "net.llm_adapter.embed.weight")
    qwen_embed = read_tensor(opts.text_encoder, "model.embed_tokens.weight")
    print(f"t5 table {tuple(t5_embed.shape)}, qwen table {tuple(qwen_embed.shape)}")

    qwen_tok = load_qwen3_tokenizer(str(resolve_under_home(opts.text_encoder)))
    t5_tok = load_t5_tokenizer()

    print("collecting clean CJK tokens + anchors …")
    clean = ext_vocab.collect_clean_qwen_tokens(qwen_tok)
    t5_ids, qwen_ids = ext_vocab.build_anchor_pairs(t5_tok, qwen_tok)
    print(f"clean CJK tokens: {len(clean)}, anchors: {len(t5_ids)}")

    W, holdout_cos = ext_vocab.fit_anchor_map(
        qwen_embed, t5_embed, t5_ids, qwen_ids, ridge=opts.ridge
    )
    print(f"anchor map fit: held-out cosine {holdout_cos:.4f}")

    print("building extended table (incl. per-char fallback rows) …")
    table, mapping = ext_vocab.build_ext_table(qwen_tok, qwen_embed, W, clean)
    n_char = len(mapping["char"])
    print(f"ext rows: {table.shape[0]} ({len(clean)} qwen tokens + {n_char} chars)")

    # Ridge regression shrinks toward the mean — rescale so new rows live at
    # the T5 table's typical row norm (the adapter downstream is scale-aware).
    t5_norm = t5_embed.float().norm(dim=-1).mean()
    ext_norm = table.norm(dim=-1).mean()
    scale = float(t5_norm / ext_norm)
    table = table * scale
    print(
        f"row-norm: t5 {t5_norm:.3f}, ext pre-scale {ext_norm:.3f}, scale {scale:.3f}"
    )

    ASSET_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    save_file({"ext_embed": table.contiguous()}, str(ASSET_PREFIX) + ".safetensors")
    mapping["stats"] = {
        "anchors": len(t5_ids),
        "holdout_cos": holdout_cos,
        "ridge": opts.ridge,
        "t5_row_norm": float(t5_norm),
        "ext_row_norm_prescale": float(ext_norm),
        "norm_scale": scale,
    }
    Path(str(ASSET_PREFIX) + ".json").write_text(
        json.dumps(mapping, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {ASSET_PREFIX}.safetensors / .json")


if __name__ == "__main__":
    main()
