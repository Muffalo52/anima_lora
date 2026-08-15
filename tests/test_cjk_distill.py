"""Phase-2b unit gates for the CJK vocab-pack distillation.

Four properties the loop's correctness rests on, none of which shows up as a
crash when it breaks:

* **G1 — EN is bit-identical.** The whole line ships as a sidecar rather than a
  checkpoint fork because appending rows cannot move the pretrained 32,128.
  Guarded twice: the hybrid encoder must emit stock spiece ids for pure-EN
  text, and the split embedding must return the stock rows bit-for-bit.
* **Identity at init.** Every parameterization must start exactly at the
  zero-shot table, or the "trained vs zero-shot" comparison is meaningless.
* **Trimming invariant.** The cache stores adapter inputs trimmed to non-pad.
  That is only sound because target pads are masked *inside* the adapter, so a
  non-pad output cannot depend on how many pads follow it.
* **Attention sink accounting.** The DiT sees ``512 - N`` zero rows and never
  masks them (CLAUDE.md padding invariant); the readout folds them into the
  softmax denominator analytically, which must match materializing them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.distill_cjk import attn_bank, data, ext_table

REPO = Path(__file__).resolve().parents[1]
EXT_PREFIX = REPO / "bench" / "cjk_adapter" / "assets" / "ext_embed"

T5_TABLE_SIZE = 32128


# ---------------------------------------------------------------------------
# G1 — EN bit-exactness
# ---------------------------------------------------------------------------


def _hybrid_encoder():
    """The real encoder, or a skip when the tokenizers/assets aren't present."""
    if not EXT_PREFIX.with_suffix(".json").exists():
        pytest.skip("ext_embed assets not built (bench/cjk_adapter/build_ext.py)")
    from anima_lora import default_checkpoints
    from bench.cjk_adapter import ext_vocab
    from library.anima import strategy as strategy_anima

    ckpt = default_checkpoints()
    if not Path(ckpt.text_encoder).exists():
        pytest.skip("Qwen3 text encoder not downloaded")
    tok = strategy_anima.AnimaTokenizeStrategy(qwen3_path=ckpt.text_encoder)
    _, mapping = ext_vocab.load_ext_assets(EXT_PREFIX)
    enc = ext_vocab.HybridT5Encoder.from_mapping(
        tok.t5_tokenizer, tok.qwen3_tokenizer, mapping
    )
    return tok, enc


def test_pure_en_text_tokenizes_exactly_as_stock_spiece():
    tok, enc = _hybrid_encoder()
    text = "1girl, black hair, school uniform, classroom, anime style"
    ids, mask = enc.encode(text, 512)
    stock = tok.t5_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    assert ids == stock["input_ids"][0].tolist()
    assert mask == stock["attention_mask"][0].tolist()


def test_offsets_do_not_change_the_ids():
    """`encode_aligned` is the implementation `encode` delegates to."""
    _, enc = _hybrid_encoder()
    text = "1girl, 黒髪, holding a sign reading 「おはよう」"
    ids_a, mask_a = enc.encode(text, 512)
    ids_b, mask_b, offs = enc.encode_aligned(text, 512)
    assert ids_a == ids_b and mask_a == mask_b
    assert len(offs) == sum(mask_b)
    for a, b in offs[:-1]:  # last entry is the zero-width EOS marker
        assert 0 <= a <= b <= len(text)


def test_split_embedding_returns_stock_rows_bitwise():
    base = torch.randn(T5_TABLE_SIZE, 16)
    init = torch.randn(64, 16)
    table = ext_table.ExtTable(init, mode="global_row", rank=4)
    split = ext_table.SplitEmbedding(base.clone(), table)

    ids = torch.randint(0, T5_TABLE_SIZE, (2, 11))
    assert torch.equal(split(ids), torch.nn.functional.embedding(ids, base))

    # …and stays bit-identical after the ext parameters move.
    with torch.no_grad():
        table.up.add_(torch.randn_like(table.up))
        table.residual.add_(torch.randn_like(table.residual))
        table.log_gain.add_(0.5)
    assert torch.equal(split(ids), torch.nn.functional.embedding(ids, base))


def test_base_rows_are_not_trainable_leaves():
    base = torch.randn(T5_TABLE_SIZE, 8)
    table = ext_table.ExtTable(torch.randn(16, 8), mode="global")
    split = ext_table.SplitEmbedding(base, table)
    names = {n for n, _ in split.named_parameters()}
    assert names and all(n.startswith("table.") for n in names)


# ---------------------------------------------------------------------------
# Parameterization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["global", "row", "global_row"])
def test_every_parameterization_starts_at_the_zero_shot_table(mode):
    init = torch.randn(32, 12)
    table = ext_table.ExtTable(init, mode=mode, rank=4)
    idx = torch.arange(32)
    assert torch.allclose(table.rows(idx), init, atol=1e-6)
    assert torch.allclose(table.materialize(), init, atol=1e-6)


def test_row_residuals_only_reach_rows_over_the_visit_floor():
    init = torch.zeros(8, 4)
    tunable = torch.tensor([1, 5])
    table = ext_table.ExtTable(init, mode="row", tunable_rows=tunable)
    with torch.no_grad():
        table.residual.add_(1.0)
    out = table.rows(torch.arange(8))
    assert out[1].abs().sum() > 0 and out[5].abs().sum() > 0
    assert out[[0, 2, 3, 4, 6, 7]].abs().sum() == 0


def test_global_mode_promotes_unvisited_rows_to_mapped():
    """The tier the plan lacked: a row the corpus never saw still moves."""
    init = torch.randn(4, 4)
    visits = torch.tensor([3, 0, 0, 0])
    glob = ext_table.ExtTable(init, mode="global", rank=2)
    assert glob.provenance(visits) == ["mapped"] * 4
    rows = ext_table.ExtTable(init, mode="row", tunable_rows=torch.tensor([0]))
    assert rows.provenance(visits) == ["tuned", "zero-shot", "zero-shot", "zero-shot"]


def test_visit_counts_ignore_base_vocab_ids():
    counts = ext_table.visit_counts(
        4, [torch.tensor([5, T5_TABLE_SIZE + 1, T5_TABLE_SIZE + 1, T5_TABLE_SIZE + 3])]
    )
    assert counts.tolist() == [0, 2, 0, 1]


# ---------------------------------------------------------------------------
# Trimming invariant
# ---------------------------------------------------------------------------


def test_non_pad_outputs_do_not_depend_on_trailing_pads():
    """Why the cache may store trimmed rows.

    Target pads are masked inside the adapter (they gate both the self- and
    the cross-attention), so appending pads must not perturb a single non-pad
    output. If this ever breaks, every cached teacher row is subtly wrong and
    nothing else in the loop notices.
    """
    from library.anima.models import LLMAdapter

    torch.manual_seed(0)
    adapter = LLMAdapter(
        source_dim=16, target_dim=16, model_dim=16, num_layers=2, self_attn=True
    ).eval()
    adapter.embed = torch.nn.Embedding(64, 16)

    n = 7
    ids = torch.randint(0, 64, (1, n))
    src = torch.randn(1, 5, 16)
    src_mask = torch.ones(1, 5, dtype=torch.long)

    with torch.no_grad():
        short = adapter(
            source_hidden_states=src,
            target_input_ids=ids,
            target_attention_mask=torch.ones(1, n, dtype=torch.long),
            source_attention_mask=src_mask,
        )
        padded_ids = torch.nn.functional.pad(ids, (0, 25))
        mask = torch.zeros(1, n + 25, dtype=torch.long)
        mask[:, :n] = 1
        long = adapter(
            source_hidden_states=src,
            target_input_ids=padded_ids,
            target_attention_mask=mask,
            source_attention_mask=src_mask,
        )
    assert torch.allclose(short, long[:, :n], atol=1e-5)


# ---------------------------------------------------------------------------
# Attention sink accounting
# ---------------------------------------------------------------------------


def _naive_readout(x, mask, probe, seq_total):
    """Materialize the zero pads the DiT actually sees and attend over them."""
    B, L, D = x.shape
    full = torch.zeros(B, seq_total, D)
    full[:, :L] = x * mask.unsqueeze(-1)
    head_dim = probe.k_gain.shape[0]
    k = (full @ probe.k_weight.T).view(B, seq_total, -1, head_dim)
    k = attn_bank._rms_norm(k, probe.k_gain)
    v = (full @ probe.v_weight.T).view(B, seq_total, -1, head_dim)
    logits = torch.einsum("hmd,blhd->bhml", probe.queries, k) / (head_dim**0.5)
    w = torch.softmax(logits, dim=-1)
    return torch.einsum("bhml,blhd->bhmd", w, v)


def test_analytic_sink_matches_materialized_zero_pads():
    torch.manual_seed(0)
    probe = attn_bank.BlockProbe(
        block=0,
        k_weight=torch.randn(8, 6) * 0.1,
        v_weight=torch.randn(8, 6) * 0.1,
        k_gain=torch.ones(4),
        queries=torch.randn(2, 3, 4),
    )
    x = torch.randn(2, 9, 6)
    mask = torch.ones(2, 9, dtype=torch.long)
    mask[1, 6:] = 0
    got = attn_bank.readout(x, mask, probe, seq_total=32)
    want = _naive_readout(x, mask, probe, seq_total=32)
    assert torch.allclose(got, want, atol=1e-5)


def test_sink_mass_makes_token_count_observable():
    """Two arms with identical vectors but different lengths must differ."""
    torch.manual_seed(0)
    probe = attn_bank.BlockProbe(
        block=0,
        k_weight=torch.randn(8, 6),
        v_weight=torch.randn(8, 6),
        k_gain=torch.ones(4),
        queries=torch.randn(2, 3, 4),
    )
    x = torch.randn(1, 4, 6)
    mask = torch.ones(1, 4, dtype=torch.long)
    a = attn_bank.readout(x, mask, probe, seq_total=512)
    b = attn_bank.readout(x, mask, probe, seq_total=16)
    assert not torch.allclose(a, b, atol=1e-4)


# ---------------------------------------------------------------------------
# Span alignment
# ---------------------------------------------------------------------------


def test_vectorized_span_pooling_matches_the_naive_loop():
    """The batched index_add form must equal per-span mean-pool + cosine.

    The loop is launch-bound, not compute-bound, so the span loss pools every
    span in the batch at once. That rewrite is only safe if it is numerically
    the same thing.
    """
    import torch.nn.functional as F

    from scripts.distill_cjk import losses

    torch.manual_seed(0)
    B, L, D = 2, 7, 5
    student = torch.randn(B, L, D)
    teacher = torch.randn(B, L, D)
    spans = [  # (sample, student idx, teacher idx, weight)
        (0, [0, 1], [2], 1.0),
        (0, [3], [4, 5], 0.3),
        (1, [2, 4, 5], [0, 1], 0.8),
    ]
    pack = {
        "s_flat": torch.tensor(
            [i * L + j for k, (i, s, _, _) in enumerate(spans) for j in s]
        ),
        "s_seg": torch.tensor([k for k, (_, s, _, _) in enumerate(spans) for _ in s]),
        "t_flat": torch.tensor(
            [i * L + j for k, (i, _, t, _) in enumerate(spans) for j in t]
        ),
        "t_seg": torch.tensor([k for k, (_, _, t, _) in enumerate(spans) for _ in t]),
        "w": torch.tensor([w for *_, w in spans]),
        "n_spans": len(spans),
    }
    got = losses.loss_span(student, teacher, pack)

    total, weight = 0.0, 0.0
    for i, s_idx, t_idx, w in spans:
        s = student[i, s_idx].mean(0)
        t = teacher[i, t_idx].mean(0)
        total += w * (1.0 - F.cosine_similarity(s, t, dim=-1))
        weight += w
    assert torch.allclose(got, total / weight, atol=1e-6)


def test_span_loss_is_inert_without_a_pack():
    from scripts.distill_cjk import losses

    assert (
        float(losses.loss_span(torch.randn(1, 3, 4), torch.randn(1, 3, 4), None)) == 0
    )


def test_ja_span_chars_are_exact_under_the_join():
    segs = ["黒髪", "女子学生", "教室"]
    joined = "、".join(segs)
    for seg, (a, b) in zip(segs, data._ja_span_chars(segs)):
        assert joined[a:b] == seg


def test_en_span_chars_survive_repeated_tags():
    text = "1girl, solo, 1girl"
    spans = data._en_span_chars(text, ["1girl", "solo", "1girl"])
    assert [text[a:b] for a, b in spans] == ["1girl", "solo", "1girl"]
    assert spans[0][0] != spans[2][0]  # the scan advances, it does not re-find


def test_tokens_in_span_uses_overlap_not_containment():
    offsets = [(0, 3), (3, 7), (7, 9)]
    mask = [1, 1, 1]
    assert data._tokens_in_span(offsets, mask, (2, 8)) == [0, 1, 2]
    assert data._tokens_in_span(offsets, mask, (-1, -1)) == []
