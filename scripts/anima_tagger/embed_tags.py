"""Build per-tag text-description embeddings for the label-embed tag head.

Reads the kept ``vocab.json``, looks each tag up in the English danbooru KB
(``models/danbooru_tags_classified.en.csv`` — wiki descriptions, no taxonomy
prefix), and embeds ``"<name> — <kind> tag. <description>"`` with a text
embedder (default Qwen3-Embedding-0.6B, last-token pooling, unit-normalized).
Tags missing from the KB fall back to embedding a templated name-only string,
so every vocab row gets a real vector.

Name matching honors the snapshotted ``rules.yaml`` the same way
``derive_groups`` does: vocab names are post-rule, so any tag the rules
renamed away from its danbooru form is recovered via the inverse replacement
map before it's counted as missing.

Output: ``<out_dir>/tag_text_emb.safetensors`` — ``emb [n_tags, d]`` fp32 in
vocab order (the trainer slices it by the routing partition) — plus a
``tag_text_emb.meta.json`` coverage report. Usage::

    python -m scripts.anima_tagger.cli --mode embed_tags \\
        --out_dir models/captioners/anima-tagger-v2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)

# Description tail beyond this many characters is boilerplate ("See also…",
# tagging etiquette) more often than signal; also bounds tokenizer cost.
_MAX_DESC_CHARS = 1200


def build_tag_texts(
    vocab: dict,
    kb,
    rename_recovery: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    """Return ``(texts in vocab order, names that fell back to template)``."""
    texts: List[str] = []
    fallbacks: List[str] = []
    for t in vocab["tags"]:
        name = t["name"]
        cat = str(t.get("category", "general"))
        info = kb.describe(name)
        if info is None and name in rename_recovery:
            info = kb.describe(rename_recovery[name])
        if info is not None and info.description:
            desc = info.description[:_MAX_DESC_CHARS]
            texts.append(f"{name} — {info.kind} tag. {desc}")
        else:
            texts.append(f"{name} — {cat} tag on a danbooru-style imageboard.")
            fallbacks.append(name)
    return texts, fallbacks


@torch.no_grad()
def embed_texts(
    texts: List[str],
    model_id: str,
    device: str,
    batch_size: int = 32,
    max_tokens: int = 256,
) -> torch.Tensor:
    """Embed with a Qwen3-Embedding-style model: left padding → last-token
    pool → L2 normalize. Returns ``[len(texts), d] fp32`` on CPU.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = (
        AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    chunks: List[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        ).to(device)
        out = model(**enc)
        # Left padding puts every sequence's last real token at position -1.
        pooled = out.last_hidden_state[:, -1]
        chunks.append(torch.nn.functional.normalize(pooled, dim=-1).float().cpu())
        if (start // batch_size) % 10 == 0:
            logger.info(
                "embedded %d/%d", min(start + batch_size, len(texts)), len(texts)
            )
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def cmd_embed_tags(args: argparse.Namespace) -> None:
    from safetensors.torch import save_file as st_save

    from library.captioning import tag_rules as tr
    from library.captioning.correction import load_tag_knowledge_base

    out_dir = Path(args.out_dir)
    vocab_path = out_dir / "vocab.json"
    if not vocab_path.exists():
        raise SystemExit(f"need {vocab_path} — run --mode build_vocab first")
    vocab = json.loads(vocab_path.read_text())

    kb = load_tag_knowledge_base(args.tag_desc_csv)
    rules_path = out_dir / "rules.yaml"
    rename_recovery: Dict[str, str] = {}
    if rules_path.exists():
        rules = tr.load_rules(rules_path)
        rename_recovery = {tgt: src for src, tgt in rules.replacements}

    texts, fallbacks = build_tag_texts(vocab, kb, rename_recovery)
    n = len(texts)
    logger.info(
        "embedding %d tag descriptions (%d matched KB, %d name-only fallback) with %s",
        n,
        n - len(fallbacks),
        len(fallbacks),
        args.embed_model,
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    emb = embed_texts(
        texts,
        args.embed_model,
        device,
        batch_size=args.embed_batch_size,
        max_tokens=args.embed_max_tokens,
    )

    dest = out_dir / "tag_text_emb.safetensors"
    st_save({"emb": emb}, str(dest), metadata={"model": args.embed_model})
    meta = {
        "model": args.embed_model,
        "d_emb": int(emb.shape[1]),
        "n_tags": n,
        "n_kb_matched": n - len(fallbacks),
        "fallback_tags": fallbacks,
        "desc_csv": str(args.tag_desc_csv),
        "max_desc_chars": _MAX_DESC_CHARS,
    }
    (out_dir / "tag_text_emb.meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(
        "wrote %s  [%d, %d]  (+ meta.json; %d fallback tags)",
        dest,
        emb.shape[0],
        emb.shape[1],
        len(fallbacks),
    )
