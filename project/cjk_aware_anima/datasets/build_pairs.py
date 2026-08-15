"""Assemble the Phase 2a distillation corpus: (EN teacher, JA student) pairs.

Each pair feeds the objective in ``../plan.md`` — teacher reads the **EN**
string on the T5 side, student reads the **JA** string through the ext vocab,
and both share the Qwen side. So a pair is correct as long as the two strings
mean the same thing; it does *not* need to be a human-quality translation.
What it does need is for the JA side to use the tokens a JA user would type,
because ext rows only train where the corpus visits them.

D1 (our caption master) is composed, not translated wholesale: captions are tag
strings, so the JA side is assembled from ``tag_glossary_ja.json`` — which puts
community idiom and pinned proper nouns in by construction, instead of hoping
an MT pass reproduces them. Registers, because users prompt in all three:

* ``tags``    — JA tag string, glossary primary wording
* ``tags_alt``— same caption through the alternate wordings (ツインテ vs
  ツインテール): real variation users type, and it visits more ext rows
* ``natural`` — the caption rewritten as JA prose (``--mt``, GPU)

D6 quoted-text templates ride along, carrying the verbatim-string construction
the OCR/text-render line will need (``「X」と書かれた看板``), with the teacher
side emitted **both** fully translated and quote-preserved.

Coverage is measured in **ext rows**, not characters: the JA side is encoded
with the very ``HybridT5Encoder`` training will use, and the histogram over row
ids is what says which rows the run can actually move.

    python project/cjk_aware_anima/datasets/build_pairs.py          # CPU
    make daemon-run ARGS="project/cjk_aware_anima/datasets/build_pairs.py --mt"
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

DEFAULT_OUT = REPO / "post_image_dataset" / "cjk_distill"
EXT_PREFIX = REPO / "bench" / "cjk_adapter" / "assets" / "ext_embed"
T5_VOCAB = 32128  # ids at or above this index are ext rows

# D6: constructions that carry a verbatim quoted string. The point is the
# quote, so the teacher side gets both renderings (translated / preserved).
QUOTE_TEMPLATES = [
    (
        "「{x}」と書かれた看板",
        'a sign that says "{t}"',
        "a sign with 「{x}」 written on it",
    ),
    (
        "「{x}」と書かれたTシャツ",
        'a t-shirt that says "{t}"',
        "a t-shirt with 「{x}」 on it",
    ),
    (
        "「{x}」と書かれた黒板",
        'a blackboard that says "{t}"',
        "a blackboard with 「{x}」 written on it",
    ),
    (
        "「{x}」と書かれた紙",
        'a piece of paper that says "{t}"',
        "a piece of paper with 「{x}」 on it",
    ),
    (
        "吹き出しに「{x}」と書かれている",
        'a speech bubble saying "{t}"',
        "a speech bubble with 「{x}」 in it",
    ),
]


def load_captions(caption_dir: Path) -> list[tuple[str, str]]:
    out = []
    for p in sorted(caption_dir.rglob("*.txt")):
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            out.append((str(p.relative_to(caption_dir).with_suffix("")), text))
    return out


def split_caption(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def alt_pool(entry: dict, min_f1: float) -> list[str]:
    """Alternate wordings safe to swap in — synonyms, not merely co-occurring tags.

    The raw wiki ``other_names`` pool is not synonymous (``small breasts`` lists
    ガキ巨乳; ``1girl`` lists 女スパイ), so sampling it would inject wrong
    meaning into a whole register. For general tags the pool is therefore the
    back-translation-verified candidates only. Character/copyright alternates
    are kept as-is: those are real nicknames users type (ホシノ for 小鳥遊ホシノ).
    """
    if entry.get("axis") != "general":
        return list(entry.get("alts") or [])
    # Verified is not enough on its own: 汗液 and 短袖 both back-translate
    # perfectly and are Chinese, and they were being swapped into `tags_alt`.
    # Kana is the proof of Japaneseness, so an alternate must either carry kana
    # or be the wording already chosen as primary.
    return [
        c["ja"]
        for c in entry.get("candidates") or []
        if c.get("f1", 0) >= min_f1 and (c.get("kana") or c["ja"] == entry.get("ja"))
    ]


def seg_provenance(entry: dict | None, ja: str) -> tuple[str, float]:
    """(via, back-translation f1) of the wording actually chosen for a segment.

    Consumed by the distillation side as a per-span trust weight: the composed
    caption is the *student's* input, so a mistranslated tag teaches the ext
    rows of the wrong word (``colored inner hair`` → 色付きの陰毛). Span-level
    supervision can down-weight those instead of dropping the pair.
    """
    if not entry:
        return "unmapped", 0.0
    f1 = 0.0
    for c in entry.get("candidates") or []:
        if c.get("ja") == ja:
            f1 = float(c.get("f1") or 0.0)
            break
    return str(entry.get("via") or "unknown"), f1


def compose(
    segments: list[str], glossary: dict, *, alt: bool, rng: random.Random, min_f1: float
):
    """Map each caption segment to JA; report what could not be mapped.

    Returns ``(ja_segments, missing, spans)`` — ``spans`` carries the EN↔JA
    segment alignment that composition makes free (each EN tag maps to exactly
    one JA tag, in order), plus that wording's provenance.
    """
    out, missing, spans = [], [], []
    for seg in segments:
        e = glossary.get(seg)
        if not e or not e.get("ja"):
            out.append(seg)  # latin passthrough — trains no ext rows, counted below
            missing.append(seg)
            spans.append({"en": seg, "ja": seg, "via": "unmapped", "f1": 0.0})
            continue
        ja = e["ja"]
        if alt:
            pool = [p for p in alt_pool(e, min_f1) if p != ja]
            if pool:
                ja = rng.choice([ja, *pool])
        out.append(ja)
        via, f1 = seg_provenance(e, ja)
        spans.append({"en": seg, "ja": ja, "via": via, "f1": f1})
    return out, missing, spans


def build_d1(captions, glossary, args, rng) -> list[dict]:
    pairs = []
    for image_id, en in captions:
        segs = split_caption(en)
        for register, alt in (("tags", False), ("tags_alt", True)):
            if register == "tags_alt" and not args.alt_register:
                continue
            ja, missing, spans = compose(
                segs, glossary, alt=alt, rng=rng, min_f1=args.alt_min_f1
            )
            pairs.append(
                {
                    "id": f"D1/{image_id}/{register}",
                    "source": "D1",
                    "register": register,
                    "en": en,
                    "ja": "、".join(ja),
                    "n_missing": len(missing),
                    "spans": spans,
                }
            )
    return pairs


def build_d6(glossary, args, rng) -> list[dict]:
    """Quoted-text constructions, X drawn from native manga lines (D7)."""
    lovehina = json.loads((ASSETS / "lovehina_ja.json").read_text())
    lines = lovehina["lines"] if isinstance(lovehina, dict) else lovehina
    texts = []
    for row in lines:
        s = row["ja"] if isinstance(row, dict) else row
        s = re.sub(r"[「」『』]", "", str(s)).strip()
        if 2 <= len(s) <= 12:
            texts.append(s)
    rng.shuffle(texts)

    pairs = []
    for i in range(min(args.n_quotes, len(texts))):
        x = texts[i]
        ja_t, en_translated, en_preserved = QUOTE_TEMPLATES[i % len(QUOTE_TEMPLATES)]
        # `t` would be the translated quote; we do not have one per line, so the
        # translated arm keeps the JA string romanised-free and simply marks it
        # as text — the preserved arm is the one that matters for the line.
        pairs.append(
            {
                "id": f"D6/{i}/preserved",
                "source": "D6",
                "register": "quote_preserved",
                "en": en_preserved.format(x=x),
                "ja": ja_t.format(x=x),
                "n_missing": 0,
            }
        )
        pairs.append(
            {
                "id": f"D6/{i}/translated",
                "source": "D6",
                "register": "quote_translated",
                "en": en_translated.format(t="[TEXT]"),
                "ja": ja_t.format(x=x),
                "n_missing": 0,
            }
        )
    return pairs


def build_natural(captions, glossary, args) -> list[dict]:  # noqa: C901
    """Prose register: MT rewrites the caption, with our wording pinned (GPU).

    This is the one register MT is actually the right tool for — turning a tag
    list into a sentence is generation, not lookup. The glossary still governs
    the vocabulary: every character/copyright tag in the caption is passed as a
    terminology constraint, so 黄泉 survives the rewrite instead of coming back
    as アケロン.
    """
    sys.path.insert(0, str(ROOT))
    from mt import MTEngine, Request  # noqa: PLC0415

    # Prose costs hours of GPU and does not depend on the tag glossary, so it is
    # cached: a glossary fix should not re-buy it.
    cache_path = Path(args.natural_cache)
    if cache_path.exists():
        cached = [
            json.loads(line)
            for line in cache_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(f"  [pairs] reusing {len(cached)} cached prose rows from {cache_path}")
        return cached

    rows = captions[: args.mt_limit] if args.mt_limit else captions
    reqs = []
    for _image_id, en in rows:
        terms = []
        for seg in split_caption(en):
            e = glossary.get(seg)
            if e and e.get("ja") and e.get("axis") in {"character", "copyright"}:
                terms.append((seg.split(" (")[0], e["ja"]))
        reqs.append(
            Request(
                text=en,
                terms=terms[: args.max_terms],
                style=(
                    "a natural Japanese sentence describing the illustration, "
                    "as a Japanese user would write a prompt — not a list of "
                    "comma-separated keywords"
                ),
            )
        )

    engine = MTEngine(args.model, greedy=True, gpu_budget=args.gpu_budget)
    got = engine.translate(
        reqs,
        batch_size=args.batch_size,
        progress_every=20,
        max_new_tokens=args.max_new_tokens,
        cache=args.mt_cache / "natural_en2ja.jsonl",
    )

    # The teacher side must *say the same thing* as the student side, and prose
    # compresses: a rewrite covers maybe 70% of a 30-tag caption. Pairing the
    # full EN caption against it would hand the teacher content the student's
    # text does not contain, which the ext rows could only match by inventing
    # it. So the teacher text for this register is the back-translation of the
    # JA — which is exactly the construction the objective calls for ("teacher
    # = EN translation of the JA text"). The original caption is kept for
    # provenance only.
    keep = [
        (image_id, en, ja.strip())
        for (image_id, en), ja in zip(rows, got)
        if ja.strip()
    ]
    print(
        f"  [pairs] back-translating {len(keep)} prose rows for the teacher side",
        flush=True,
    )
    back = engine.translate(
        [Request(text=ja) for _, _, ja in keep],
        target_lang="en",
        batch_size=args.batch_size,
        progress_every=20,
        max_new_tokens=args.max_new_tokens,
        cache=args.mt_cache / "natural_ja2en.jsonl",
    )
    out = [
        {
            "id": f"D1/{image_id}/natural",
            "source": "D1",
            "register": "natural",
            "en": teacher.strip(),
            "ja": ja,
            "source_en": src_en,
            "n_missing": 0,
        }
        for (image_id, src_en, ja), teacher in zip(keep, back)
        if teacher.strip()
    ]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n",
        encoding="utf-8",
    )
    print(f"  [pairs] cached {len(out)} prose rows to {cache_path}")
    return out


def ext_coverage(pairs: list[dict], args) -> dict:
    """Histogram ext-row visits using the encoder training will use."""
    from bench.cjk_adapter import ext_vocab  # noqa: PLC0415
    from library.anima.weights import (  # noqa: PLC0415
        load_qwen3_tokenizer,
        load_t5_tokenizer,
    )
    from library.env import resolve_under_home  # noqa: PLC0415

    from anima_lora import default_checkpoints  # noqa: PLC0415

    mapping = json.loads(EXT_PREFIX.with_suffix(".json").read_text(encoding="utf-8"))
    qwen_tok = load_qwen3_tokenizer(
        str(resolve_under_home(default_checkpoints().text_encoder))
    )
    enc = ext_vocab.HybridT5Encoder.from_mapping(load_t5_tokenizer(), qwen_tok, mapping)

    visits: collections.Counter = collections.Counter()
    for p in pairs:
        ids, mask = enc.encode(p["ja"], max_length=args.max_length)
        visits.update(i for i, m in zip(ids, mask) if m and i >= T5_VOCAB)

    n_rows = mapping["rows"]
    bands = {"1-4": 0, "5-49": 0, "50-499": 0, "500+": 0}
    for c in visits.values():
        key = "1-4" if c < 5 else "5-49" if c < 50 else "50-499" if c < 500 else "500+"
        bands[key] += 1

    # The row histogram alone cannot answer the phase gate, which is written in
    # terms of script classes ("kana saturated, train the visited kanji set").
    # The ext table spans all of CJK including zh/ko, so its 58,968 rows are the
    # wrong denominator for a ja-only corpus — codepoint classes are the honest
    # readout.
    kana_total = {chr(c) for c in range(0x3041, 0x3097)} | {
        chr(c) for c in range(0x30A1, 0x30FB)
    }
    seen: collections.Counter = collections.Counter()
    for p in pairs:
        seen.update(p["ja"])
    kana_seen = {c for c in seen if c in kana_total}
    kanji_seen = {c for c in seen if "一" <= c <= "鿿"}

    return {
        "ext_rows_total": n_rows,
        "ext_rows_visited": len(visits),
        "ext_rows_visited_pct": round(100 * len(visits) / n_rows, 2),
        "visits_total": sum(visits.values()),
        "rows_by_visit_band": bands,
        "kana_codepoints_seen": len(kana_seen),
        "kana_codepoints_total": len(kana_total),
        "kana_saturation_pct": round(100 * len(kana_seen) / len(kana_total), 1),
        "kana_unseen": "".join(sorted(kana_total - kana_seen)),
        "kanji_codepoints_seen": len(kanji_seen),
        "kanji_seen_once_only": sum(1 for c in kanji_seen if seen[c] == 1),
        "top_rows": visits.most_common(20),
    }


def write_spotcheck(pairs: list[dict], path: Path, n: int, rng: random.Random) -> None:
    """Sample for the user sign-off gate, biased to what the corpus leans on."""
    by_source: dict[str, list[dict]] = collections.defaultdict(list)
    for p in pairs:
        by_source[p["source"]].append(p)
    picked: list[dict] = []
    for source, rows in sorted(by_source.items()):
        share = max(1, round(n * len(rows) / len(pairs)))
        picked += rng.sample(rows, min(share, len(rows)))
    lines = [
        "# Phase 2a translation spot-check",
        "",
        f"{len(picked)} sampled pairs. **EN is what the teacher reads; JA is what",
        "the student reads.** They only have to mean the same thing — flag rows",
        "where the JA says something different, or uses wording no Japanese user",
        "would type. Tag-wording fixes go in `tag_overrides.json`.",
        "",
    ]
    for p in picked:
        lines += [
            f"### {p['id']}  ({p['register']})",
            f"- EN: {p['en']}",
            f"- JA: {p['ja']}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--glossary", type=Path, default=ASSETS / "tag_glossary_ja.json")
    ap.add_argument("--captions", type=Path, default=REPO / "image_dataset")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--alt-register", action="store_true", default=True)
    ap.add_argument(
        "--alt-min-f1",
        type=float,
        default=0.75,
        help="back-translation score an alternate wording must clear to be swapped in",
    )
    ap.add_argument("--n-quotes", type=int, default=2000)
    ap.add_argument("--spotcheck", type=int, default=200)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-coverage", action="store_true", help="skip the ext histogram")
    ap.add_argument("--mt", action="store_true", help="add the prose register (GPU)")
    ap.add_argument("--model", default="tencent/Hy-MT2-7B")
    ap.add_argument("--gpu-budget", default="13GiB")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--mt-limit", type=int, default=0, help="prose for the first N only"
    )
    ap.add_argument("--max-terms", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--natural-cache", type=Path, default=ASSETS / "natural_ja.jsonl")
    ap.add_argument("--mt-cache", type=Path, default=ASSETS / ".mtcache")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    glossary = json.loads(args.glossary.read_text())["tags"]
    captions = load_captions(args.captions)

    pairs = build_d1(captions, glossary, args, rng) + build_d6(glossary, args, rng)
    if args.mt:
        pairs += build_natural(captions, glossary, args)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    report = {
        "n_pairs": len(pairs),
        "by_source": dict(collections.Counter(p["source"] for p in pairs)),
        "by_register": dict(collections.Counter(p["register"] for p in pairs)),
        "untranslated_segments": sum(p["n_missing"] for p in pairs),
        "glossary": str(args.glossary),
    }
    if not args.no_coverage:
        report["ext_coverage"] = ext_coverage(pairs, args)
    (args.out / "coverage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    write_spotcheck(pairs, args.out / "spotcheck.md", args.spotcheck, rng)

    print(f"wrote {args.out}/pairs.jsonl  ({len(pairs)} pairs)")
    for k, v in report["by_register"].items():
        print(f"    {k:18s} {v}")
    if "ext_coverage" in report:
        c = report["ext_coverage"]
        print(
            f"  ext rows visited: {c['ext_rows_visited']}/{c['ext_rows_total']} "
            f"({c['ext_rows_visited_pct']}%), {c['visits_total']} visits"
        )
        print(f"  rows by visit band: {c['rows_by_visit_band']}")
    print(f"  untranslated segments: {report['untranslated_segments']}")


if __name__ == "__main__":
    main()
