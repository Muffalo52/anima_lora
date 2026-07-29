# Pad-sink collapse — verdict (Phase 0, 2026-07-22)

Run: `results/20260722-1508-phase0/` (base DiT, 1024², 28-step ER-SDE, CFG 4,
prompts n ∈ {17, 56, 150} of 512).

Prompted by arXiv 2607.19139 ("Text Template Tokens Are Implicit Semantic
Registers in DiTs", Qwen-Image/MMDiT). Anima has no template span and its pad
positions are zeroed before the DiT (same in upstream diffusion-pipe
`cosmos_predict2.py`), so the paper's semantic-register mechanism has no
substrate here — the pad tail is analytically pure null attention
(K=0 → logit 0, V=0), i.e. `(512−n)` added to every cross-attn softmax
denominator and nothing else.

## Findings

1. **Null-key collapse CONFIRMED.** Replacing the pad tail with one zero key
   biased by `log(512−n)` (`_ctx_k_bias`) matches the 512-pad forward at the
   kernel noise floor: relL2 vs the kernel-matched reference 4.0e-3–7.4e-3,
   floor (flash↔torch on identical math) 3.5e-3–7.0e-3. Pre-registered gate
   `nullkey_within_2x_floor` = **True**. 28-step trajectories drift 0.09–0.10
   final relL2 — indistinguishable from base_torch's own 0.05–0.09 (chaotic
   amplification of kernel-level noise; images visually identical).
2. **Trim without the null key destroys generation** (noise fields;
   final drift ~1.9–2.2) — the documented black-image invariant is exactly the
   lost softmax denominator, now demonstrated as the positive control.
3. **Real (un-zeroed) pad states destroy generation too** (drift 1.1–2.5;
   dark/moiré fields). Qwen3 pad positions are causal-LM EOS states that DO
   read the caption, but the DiT — trained only on zero pads — treats 360+
   loud keys as content and collapses. **Pads cannot be repurposed as
   registers training-free.** Any register-style slot would have to be
   trained in (and the register line is closed — RQ1/RQ3).
4. **No wall-clock win at 1024².** 245.5 ms (nullkey, L_ctx=151) vs 245.8 ms
   (base flash, L_ctx=512): the bias path forfeits flash for cross-attn
   (~10 ms penalty, see base_torch 255.9 ms) and the shorter context only wins
   that back. Cross-attn is a small slice at 4096 image tokens.

## Implications

- The mechanistic question is settled without a head-mass scan: Anima's
  cross-attn "sinks" carry zero semantics, provably and now empirically.
  Per-query pad mass acts as a soft OFF-gate for each head (output shrinks
  toward 0 as content logits weaken), nothing more.
- If the L_ctx cut is ever wanted for real (512-tier, memory-bound regimes),
  the flash-compatible exact rewrite is: run flash over the n content keys
  only, then rescale the output by `σ = exp(lse) / (exp(lse) + (512−n))`
  using flash's returned per-query logsumexp — no null key, no additive-mask
  detour. Not pursued: no demonstrated need.
- The 512-token TE cache shape stays authoritative; nothing here motivates a
  cache change.
