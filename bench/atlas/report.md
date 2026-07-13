# Artist atlas — Phase 0 report (2026-07-12)

Proposal: `docs/proposal/artist_atlas_pack.md`. Packer: `scripts/atlas/pack.py`
(`python -m scripts.atlas.pack`). Unit gates: `tests/test_atlas_pack.py`.
Full-model gates: `bench/atlas/run_bench.py` → `results/20260712-*-phase0*`.

## Verdict

**G1 PASS, G2 PASS — the cheap version survives.** Packing is exact; the
one-hot expert path is indistinguishable from the solo checkpoint up to the
measured bf16 kernel-path noise floor, and the zero gate is bit-identical to
the base model.

| Gate | Result |
|---|---|
| G2 zero-gate = base | max abs diff **0.0** (bitwise), both per-forward and after 10 sampling steps |
| G1 one-hot = solo (per-forward, σ∈{0.9,0.5,0.1}) | hews 0.078–0.114, sincos 0.071–0.079 of the adapter effect — vs **control floor 0.078–0.110** |
| G3 unit (ΔW roundtrip, fused qkv both save shapes, factory load, forwards) | green, CPU, fp32 to 1e-5 |

The control is a duplicate-ingredient atlas (same soup packed twice): its
one-hot is *mathematically identical* to the solo checkpoint, so its deviation
is pure numerics (stacked `(E,r)` einsum vs plain fused GEMM under bf16). The
real atlas's deviations match the control to the third decimal → no
bookkeeping error anywhere in the fuse/pack/load path. Trajectory-level RMSE
ratios (~4–6% after 10 Euler steps) are chaos-amplified versions of the same
floor — use the per-forward probes, not final latents, for exactness claims.

## What Phase 0 surfaced beyond the gates

1. **Soup q/k/v downs are not clones.** The soup SVD runs per split component,
   so the stacked-experts loader's cloned-down assumption
   (`down_fused = downs[0]`) would silently corrupt k/v had we packed
   per-component. The packer therefore reproduces the plain loader's
   **block-diagonal** fuse per expert (rank n·r on fused attn Linears) and
   zero-pads ranks across experts. This was exactly the "fuse-spec bug hides
   here" the proposal predicted — caught at pack design time.
2. **Two live loader bugs fixed** (`networks/lora_anima/`): `from_weights`
   defaulted stacked-experts checkpoints to `num_experts=4` regardless of the
   E axis, and `for_inference` downgraded the stacked spec to plain LoRA
   (`factory.py` if-chain). The FeRA cell's load path had evidently never been
   exercised.
3. **Gate default is uniform 1/E, not zero.** With `router_source="none"`
   nothing sets gates per step, and the placeholder is 1/E — a caller that
   forgets to gate gets the *ingredient average* (a real adapter-strength
   effect: RMSE 0.137 vs base), not the base model. Pinned in tests; any
   inference plumbing must write gates per request (zero vector for "no tag").
4. **Premise correction (measured, see the proposal's 2026-07-12 note and
   memory `project_soup_deltas_share_uncond_component`):** soup-line artist
   ΔWs are ~98% one shared uncond component (raw pairwise cos +0.96..+0.98,
   rank-32 truncation of the 5-way soup retains 99.99%); near-orthogonality
   holds only for the 2–7%-energy style residuals. Consequences: the
   rank-truncated merge baseline is NOT handicapped for these ingredients
   (averaging, not truncation, is what washes styles); atlas mixing should use
   **convex** gates (weights summing to 1) so the shared component stays at 1×
   while styles interpolate in the near-orthogonal residual space.

## Artifacts

- `output/ckpt/anima_atlas5_moe.safetensors` — 5 artists (kat_(bu-kunn),
  ama_mitsuki, hews, suujiniku, sincos), expert order in `ss_atlas_tags`.
- Cross-expert separation: RMSE(one-hot hews, one-hot sincos) = 0.088 ≈ 60% of
  a solo adapter's effect size — experts are genuinely distinct.

## Next (Phase 1, per proposal)

- Prompt-scan → gate wiring at inference (the per-request one-hot / convex
  blend), `make pack` target, VRAM/latency at 8–16 experts.
- The mixing experiment (28-step/CFG-4 grids, eyeball): with the premise
  correction, the interesting comparison is convex atlas blends vs the
  uniform merge — both keep the shared component at 1×; the atlas keeps
  styles at α instead of 1/N.
