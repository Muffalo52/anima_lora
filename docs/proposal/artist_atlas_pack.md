# Artist atlas — pack N trained artist LoRAs into one tag-hard-routed checkpoint

Status: **PROPOSED 2026-07-06 (Phase 0 not run).**

## Premise sources

- `bench/merge_interference/` Phase-0: independently trained artist ΔWs are
  **near-orthogonal already** (the finding that killed OrthoReg-for-merge — the
  regularizer had nothing left to buy). That result was banked as a negative for
  *merging*; this proposal is the positive it implies for *packing*: if the
  subspaces barely overlap, N adapters can coexist in one checkpoint with
  per-request selection and — the actual experiment — additive *mixing*.
- `networks/lora_modules/stacked_experts.py`: independent-A multi-expert module,
  each expert owns its own `(lora_down, lora_up)`, gates arrive via the
  `_routing_weights` broadcast buffer, **no router of its own**. This is exactly
  the substrate: packing = filling expert slots with pre-trained weights;
  routing = writing a one-hot into the buffer. No new module math.
- `scripts/soup/` SVD rank-truncation: normalizing heterogeneous ranks to a
  common `network_dim` at ΔW level is ~free (99.9% energy retained at rank 16,
  measured in the soup bench).
- `make caption-index` → `caption_index.json`: the artist-tag vocabulary, for
  free (tag → expert map derivation).

## The idea

Offline assembly, **no training**: `make pack ADAPTERS="a.safetensors b.safetensors …"`
takes N plain artist LoRAs (soup outputs are natural ingredients), SVD-normalizes
each ΔW to a common rank, and writes one checkpoint whose per-Linear layout is the
StackedExperts `(E, r, in)/(E, out, r)` stack, with metadata
`ss_atlas_tags = {artist_tag: expert_idx}`.

At inference the gate is **per-request, not per-step**: scan the prompt for known
artist tags before sampling, write the matching one-hot (or zero vector — pure
base model) into `_routing_weights`, sample normally. No learned router, no
σ-dependence, no per-step compute. Adapter family rides in checkpoint metadata as
usual, so `--lora_weight atlas.safetensors` auto-discovers.

What this buys over "just load a different file":

1. One distributable artifact for a whole artist roster (incl. the ComfyUI story —
   the hydralora loader node already speaks multi-expert checkpoints).
2. Per-item routing inside a batch (different prompts → different experts), which
   merge-on-load can never do.
3. **Multi-tag mixes**: two artist tags → weights `[α, β]` instead of one-hot.
   Near-orthogonality predicts the styles superpose with little interference —
   this is the scientific payload, and nothing in the merge-interference bench
   tested it at inference-compose level.

## Phase 0 — exactness + isolation gate (cheap, mostly CPU)

Pack 3 artists. Gates are correctness checks, not quality judgments:

- **G1 (exactness)**: with expert *i* hard-selected, the packed forward must
  reproduce the solo LoRA *i* forward. Same seed, same prompt → assert final
  latents match within bf16 tolerance. Any real mismatch is a fuse-spec bug
  (`match_fused_spec` on qkv/kv fusions is where it would hide), not a judgment
  call.
- **G2 (isolation)**: prompt with no known tag → bit-identical to base model
  (zero gate ⇒ expert contribution exactly zero by construction; assert anyway).
- **G3 (unit)**: pack two toy LoRAs, assert per-Linear reconstructed ΔW equals
  each ingredient's ΔW expert-wise after rank normalization (tolerance from the
  SVD truncation energy). Lives in `tests/`, no GPU.

Phase 0 failing G1 in a way not traceable to fusion bookkeeping kills the cheap
version (fall back to per-request merge-on-load, which is strictly less
interesting but always works).

## Phase 1 — scale + the mixing experiment

- Pack 8–16 artists; measure VRAM/latency delta vs solo (expected: negligible
  compute — one extra `(…, E, r)` einsum boundary — and N× LoRA params resident;
  if resident size matters, experts can live on CPU and swap per request since
  the gate is request-constant).
- **Mixing**: pairs of artist tags at `[0.5, 0.5]` and asymmetric blends,
  rendered at the standard 28-step/CFG-4 grid protocol, judged by eyeball
  side-by-sides (does the blend read as a coherent hybrid or as artifacts?).
  Orthogonality says interference should be low; whether *superposed styles*
  are perceptually coherent is exactly what we don't know.
- `make pack` wiring (daemon command job, snapshot sidecar, same conventions as
  `make soup`).

## Non-goals / guards

- **Not merge.** No weight fusion into the DiT, no OrthoReg revival (settled
  negative), no joint retraining. Ingredients stay plain-LoRA only — hydra/
  chimera checkpoints refused, same rule as soup.
- **Not a learned router.** If mixing turns out to need per-step or per-layer
  gating to look good, that is a *new* proposal with training in it — stop here
  and write it up rather than scope-creeping.

## Kill criteria

- G1 exactness unreachable after fuse-spec debugging → close, keep the unit
  test as documentation of why.
- Phase-1 mixes incoherent at every blend ratio → the atlas survives as a
  packaging/UX feature (gates 1–2 still hold) but the mixing claim is banked as
  a negative alongside the merge-interference finding.
