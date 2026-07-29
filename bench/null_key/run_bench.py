#!/usr/bin/env python3
"""Pad-sink collapse — 512-pad vs one-null-key equivalence bench.

Context (prompted by arXiv 2607.19139 "Text Template Tokens Are Implicit
Semantic Registers in DiTs"): Anima has no chat-template span — the caption is
tokenized raw and max-padded to 512, and every pad position's embedding is
**zeroed** before the DiT sees it (library/anima/strategy.py:135,
library/inference/text.py `_preprocess_text_embeds` zeroing; identical in the
upstream diffusion-pipe cosmos_predict2 pipeline). Because cross-attn
``kv_proj`` is bias-free and RMSNorm(0)=0, every pad key has K=0 → logit 0 →
contributes exactly exp(0)=1 to each softmax denominator, and V=0 → nothing to
the numerator. The whole (512−n)-token pad tail is therefore analytically ONE
null key carrying a logit bias of log(512−n):

    Σ_content exp(l_i) + (512−n)·exp(0)  ==  Σ_content exp(l_i) + exp(0 + log(512−n))

This bench verifies that collapse end-to-end and probes the two neighbouring
questions:

  * **base**       — standard 512-pad inference (default attn backend).
  * **base_torch** — base + an all-zero per-key logit bias. Mathematically
                     identical to base, but the bias routes cross-attn through
                     SDPA's additive-mask path (attn_mode="torch"), i.e. the
                     SAME kernel the nullkey arm uses. nullkey is judged
                     against THIS arm so kernel noise is separated from the
                     trim+bias rewrite; base-vs-base_torch is the noise floor.
  * **nullkey**    — content rows + one zero row, `_ctx_k_bias` =
                     [0]*n + [log(512−n)]. Expected: equal to base_torch at
                     the kernel noise floor. Exact identity in exact
                     arithmetic — any drift beyond floor is a bug.
  * **trim**       — content rows only, no null key. The known-broken arm
                     (denominator loses its 512−n mass → cross-attn outputs
                     blow up → the documented black/burned images). Kept as
                     the invariant's positive control.
  * **realpad**    — pads NOT zeroed: Qwen3 last_hidden kept at pad positions
                     and LLM-adapter output kept at T5 pad positions. The
                     "what if real pad tokens?" arm — pads are causal-LM EOS
                     states that DO read the caption, i.e. a poor man's
                     register. The DiT never saw them in training (both this
                     repo and upstream zero them), so this is an OOD probe:
                     eyeball + drift metrics tell us how much the model cares.

Phases: (1) single-forward velocity equivalence at high/mid/low σ,
(2) full 28-step CFG trajectories with matched ER-SDE noise → per-step drift +
decoded images + contact sheet, (3) cond-forward timing (base vs base_torch vs
nullkey; the nullkey speed win is the shorter L_ctx, honestly reported against
the torch-kernel baseline since a set bias forfeits flash for cross-attn).

Run eager (no --compile): varying L_ctx across arms would churn recompiles.

Run from anima_lora/::

    uv run python bench/null_key/run_bench.py
    uv run python bench/null_key/run_bench.py --skip_traj      # fast: phase 1+3 only
    uv run python bench/null_key/run_bench.py --prompts "1girl, sunset" ...
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from anima_lora import decode_to_pil, load_vae  # noqa: E402
from bench._anima import (  # noqa: E402
    DEFAULT_NEG,
    add_common_args,
    add_model_args,
    build_anima,
    resolve_dtype,
)
from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima import models as anima_models  # noqa: E402
from library.inference.adapters import set_xattn_kbias  # noqa: E402
from library.inference.models import load_text_encoder  # noqa: E402
from library.inference.sampling import (  # noqa: E402
    ERSDESampler,
    get_timesteps_sigmas,
)
from library.inference.text import (  # noqa: E402
    MAX_CROSSATTN_TOKENS,
    prepare_text_inputs,
)
from library.runtime.device import clean_memory_on_device  # noqa: E402

log = logging.getLogger("bench.null_key")
logging.basicConfig(level=logging.INFO, format="%(message)s")

ARMS = ("base", "base_torch", "nullkey", "trim", "realpad")

# Short / medium / long captions — n varies so the collapsed bias log(512−n)
# and the nullkey speedup are exercised across the realistic length range.
DEFAULT_PROMPTS = [
    "1girl, solo, silver hair, red eyes, smile, looking at viewer",
    "2girls, holding hands, school uniform, cherry blossoms, petals, wind, "
    "outdoors, blue sky, clouds, depth of field, from below, happy, blush, "
    "black hair, twintails, brown hair, short hair, pleated skirt, loafers",
    "A sprawling cyberpunk market street at night in heavy rain, neon signs "
    "in half-legible katakana reflecting off wet asphalt, a lone courier in "
    "a transparent raincoat weaving a bicycle between noodle stalls, steam "
    "rising from vents, holographic koi swimming between power lines, "
    "cluttered storefronts stacked three stories high with hand-painted "
    "banners, distant arcology towers dissolving into fog, cinematic "
    "lighting, extreme detail, wide angle, 1girl, cyberpunk, night, rain, "
    "neon lights, bicycle, market, steam, hologram, fish, crowd, umbrella, "
    "reflection, scenery, science fiction",
]


def _velocity(anima, x, sigma: float, emb, pad) -> torch.Tensor:
    t_b = torch.full((1,), float(sigma), device=x.device, dtype=x.dtype)
    return anima(x, t_b, emb, padding_mask=pad).float()


def _encode_realpad(anima, text_encoder, prompt: str, device) -> torch.Tensor:
    """(1, 512, D) crossattn embedding with pad positions left REAL.

    Same tokenize → Qwen3 → LLM-adapter pipeline as
    library/inference/text.py, minus the two zeroing steps
    (`prompt_embeds[~attn_mask]=0` and `crossattn_emb[~t5_mask]=0`).
    Encoder + adapter attention masks stay standard — the arm isolates
    output zeroing, not mask semantics.
    """
    from library.anima import text_strategies

    tok = text_strategies.TokenizeStrategy.get_strategy()
    q_ids, q_mask, t5_ids, t5_mask = tok.tokenize(prompt)
    with torch.no_grad():
        hs = text_encoder(
            input_ids=q_ids.to(text_encoder.device),
            attention_mask=q_mask.to(text_encoder.device),
        ).last_hidden_state  # NOT zeroed
        ctx = anima.llm_adapter(
            hs.to(device),
            t5_ids.to(device),
            target_attention_mask=t5_mask.to(device),
            source_attention_mask=q_mask.to(device),
        )  # NOT zeroed
    return ctx


def _split_content(emb512: torch.Tensor, t5_mask: torch.Tensor):
    """(n, trimmed content rows) from a zero-padded (1, 512, D) embedding.

    Asserts the layout assumptions the collapse relies on: right-padded mask
    (content = first n rows) and genuinely zero pad rows.
    """
    n = int(t5_mask.sum().item())
    assert bool(t5_mask[0, :n].all()), "t5 mask is not right-padded"
    assert n == 0 or not bool(t5_mask[0, n:].any()), (
        "t5 mask has content after the pad boundary"
    )
    pad_abs = emb512[0, n:].abs()
    assert pad_abs.numel() == 0 or float(pad_abs.max()) == 0.0, (
        "pad rows are not exactly zero — the analytic collapse does not apply"
    )
    return n, emb512[:, :n]


def _arm_embeds(emb512: torch.Tensor, t5_mask: torch.Tensor, device):
    """Per-arm (embedding, bias) pairs for one prompt side (cond or uncond)."""
    n, content = _split_content(emb512, t5_mask)
    L = emb512.shape[1]
    null_row = torch.zeros(
        1, 1, emb512.shape[2], device=emb512.device, dtype=emb512.dtype
    )
    n_pad = L - n
    assert n_pad > 0, f"prompt fills all {L} tokens — nothing to collapse"
    bias = torch.zeros(n + 1, device=device, dtype=torch.float32)
    bias[-1] = math.log(n_pad)
    return {
        "n": n,
        "base": (emb512, None),
        "base_torch": (emb512, torch.zeros(L, device=device, dtype=torch.float32)),
        "nullkey": (torch.cat([content, null_row], dim=1), bias),
        "trim": (content, None),
    }


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def _panel(images, labels, caption: str, thumb: int) -> Image.Image:
    imgs = [im.resize((thumb, thumb), Image.LANCZOS) for im in images]
    w, h = imgs[0].size
    head = 56
    panel = Image.new("RGB", (w * len(imgs), h + head), "white")
    d = ImageDraw.Draw(panel)
    d.text((6, 4), caption[:140], fill="black")
    for k, (im, lab) in enumerate(zip(imgs, labels)):
        panel.paste(im, (k * w, head))
        d.text((k * w + 6, head - 20), lab, fill="black")
    return panel


@torch.no_grad()
def _forward_arm(anima, arm, x, sigma, embeds, pad):
    emb, bias = embeds[arm]
    if bias is not None:
        set_xattn_kbias(anima, bias)
    try:
        return _velocity(anima, x, sigma, emb, pad)
    finally:
        if bias is not None:
            set_xattn_kbias(anima, None)


@torch.no_grad()
def _trajectory(
    anima, arm, x0, sig, sigmas_t, enc_c, enc_u, pad, guidance, er_seed, device, dtype
):
    """One full CFG denoise; returns (final latent, list of per-step latents)."""
    x = x0.clone()
    er = ERSDESampler(sigmas_t, seed=er_seed, device=device)
    states = []
    for i in range(len(sig) - 1):
        vc = _forward_arm(anima, arm, x, sig[i], enc_c, pad)
        vu = _forward_arm(anima, arm, x, sig[i], enc_u, pad)
        noise_pred = vu + guidance * (vc - vu)
        denoised = x.float() - sig[i] * noise_pred
        x = er.step(x, denoised, i).to(dtype)
        states.append(x.detach().clone())
    return x, states


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(ap)
    add_common_args(ap)
    ap.add_argument("--adapter", default=None, help="Optional LoRA checkpoint.")
    ap.add_argument("--prompts", nargs="+", default=None, help="Override prompts.")
    ap.add_argument("--neg_prompt", default=DEFAULT_NEG)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--infer_steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--flow_shift", type=float, default=3.0)
    ap.add_argument("--thumb", type=int, default=384)
    ap.add_argument(
        "--skip_traj",
        action="store_true",
        help="Skip phase 2 (trajectories+images): forward equivalence + timing only.",
    )
    ap.add_argument("--timing_iters", type=int, default=15)
    args = ap.parse_args()

    if args.compile:
        log.warning(
            "--compile with varying L_ctx across arms will recompile per arm; "
            "this bench is meant to run eager."
        )

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    prompts = args.prompts or DEFAULT_PROMPTS

    bundle = build_anima(args, adapter=args.adapter, train_mode=False)
    anima = bundle.anima
    assert getattr(anima, "use_llm_adapter", False) and hasattr(anima, "llm_adapter"), (
        "realpad arm needs the DiT's llm_adapter"
    )
    vae = None
    if not args.skip_traj:
        vae = load_vae(
            args.vae, device="cpu", dtype=torch.bfloat16, eval=True, vae_2d=True
        )

    _, sigmas = get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)
    sig = [float(s) for s in sigmas]

    # ── text encoding (one shared TE, freed before the denoise loops) ──────
    text_encoder = load_text_encoder(
        text_encoder=args.text_encoder, dtype=torch.bfloat16, device=device
    )
    text_encoder.eval()
    shared = {"text_encoder": text_encoder}
    per_prompt = []
    for p in prompts:
        ctx, ctx_null = prepare_text_inputs(
            device=device,
            anima=anima,
            prompt=p,
            negative_prompt=args.neg_prompt,
            text_encoder_path=args.text_encoder,
            shared_models=shared,
        )
        emb_c = ctx["embed"][0].to(device, dtype)
        t5_mask_c = ctx["embed"][3].to(device)
        emb_u = ctx_null["embed"][0].to(device, dtype)
        t5_mask_u = ctx_null["embed"][3].to(device)
        enc_c = _arm_embeds(emb_c, t5_mask_c, device)
        enc_u = _arm_embeds(emb_u, t5_mask_u, device)
        enc_c["realpad"] = (
            _encode_realpad(anima, text_encoder, p, device).to(dtype),
            None,
        )
        enc_u["realpad"] = (
            _encode_realpad(anima, text_encoder, args.neg_prompt, device).to(dtype),
            None,
        )
        per_prompt.append({"prompt": p, "cond": enc_c, "uncond": enc_u})
        log.info(
            f"[encode] n_cond={enc_c['n']:3d} n_uncond={enc_u['n']:3d} "
            f"bias=log({MAX_CROSSATTN_TOKENS - enc_c['n']})="
            f"{math.log(MAX_CROSSATTN_TOKENS - enc_c['n']):.4f}  «{p[:60]}»"
        )
    del text_encoder, shared
    gc.collect()
    clean_memory_on_device(device)

    C = anima_models.Anima.LATENT_CHANNELS
    hl, wl = args.height // 8, args.width // 8
    pad = torch.zeros(1, 1, hl, wl, dtype=dtype, device=device)

    run_dir = make_run_dir("null_key", label=args.label)
    artifacts: list[str] = []

    # ── phase 1: single-forward velocity equivalence ───────────────────────
    probe_idx = [0, len(sig) // 2, len(sig) - 2]
    fwd_metrics = []
    log.info("\n== phase 1: single-forward equivalence (ref = base_torch) ==")
    log.info(f"{'prompt':6s} {'σ':>6s} {'arm':10s} {'max|Δv|':>10s} {'relL2':>10s}")
    for pi, rec in enumerate(per_prompt):
        g = torch.Generator(device=device).manual_seed(args.seed + pi)
        x = torch.randn((1, C, 1, hl, wl), generator=g, device=device, dtype=dtype)
        for si in probe_idx:
            s = sig[si]
            ref = _forward_arm(anima, "base_torch", x, s, rec["cond"], pad)
            for arm in ARMS:
                if arm == "base_torch":
                    continue
                v = _forward_arm(anima, arm, x, s, rec["cond"], pad)
                m = {
                    "prompt_idx": pi,
                    "sigma": s,
                    "arm": arm,
                    "max_abs": _max_abs(v, ref),
                    "rel_l2": _rel_l2(v, ref),
                }
                fwd_metrics.append(m)
                log.info(
                    f"p{pi:<5d} {s:6.3f} {arm:10s} {m['max_abs']:10.3e} "
                    f"{m['rel_l2']:10.3e}"
                )

    # ── phase 2: full trajectories + images ────────────────────────────────
    traj_metrics = []
    if not args.skip_traj:
        log.info("\n== phase 2: 28-step CFG trajectories ==")
        for pi, rec in enumerate(per_prompt):
            g = torch.Generator(device=device).manual_seed(args.seed + 1000 + pi)
            x0 = torch.randn((1, C, 1, hl, wl), generator=g, device=device, dtype=dtype)
            er_seed = args.seed + 1000 + pi
            base_states = None
            imgs, labels = [], []
            for arm in ARMS:
                lat, states = _trajectory(
                    anima,
                    arm,
                    x0,
                    sig,
                    sigmas,
                    rec["cond"],
                    rec["uncond"],
                    pad,
                    args.guidance,
                    er_seed,
                    device,
                    dtype,
                )
                if arm == "base":
                    base_states = states
                drift = [
                    _rel_l2(s.float(), b.float()) for s, b in zip(states, base_states)
                ]
                im = decode_to_pil(vae, lat, device)
                fname = f"p{pi}_{arm}.png"
                im.save(run_dir / fname)
                artifacts.append(fname)
                imgs.append(im)
                labels.append(f"{arm} d{drift[-1]:.2e}")
                traj_metrics.append(
                    {
                        "prompt_idx": pi,
                        "arm": arm,
                        "final_drift_rel_l2": drift[-1],
                        "drift_per_step": drift,
                    }
                )
                log.info(f"p{pi} {arm:10s} final drift relL2 {drift[-1]:.3e}")
            sheet = f"sheet_p{pi}.png"
            _panel(imgs, labels, prompts[pi], args.thumb).save(run_dir / sheet)
            artifacts.append(sheet)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ── phase 3: cond-forward timing (longest prompt) ──────────────────────
    log.info("\n== phase 3: cond-forward timing ==")
    rec = max(per_prompt, key=lambda r: r["cond"]["n"])
    g = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.randn((1, C, 1, hl, wl), generator=g, device=device, dtype=dtype)
    timing = {}
    with torch.no_grad():
        for arm in ("base", "base_torch", "nullkey"):
            for _ in range(3):
                _forward_arm(anima, arm, x, sig[0], rec["cond"], pad)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts = []
            for _ in range(args.timing_iters):
                t0 = time.perf_counter()
                _forward_arm(anima, arm, x, sig[0], rec["cond"], pad)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            L = rec["cond"][arm][0].shape[1]
            timing[arm] = {"median_ms": float(np.median(ts) * 1e3), "L_ctx": L}
            log.info(f"{arm:10s} L_ctx={L:3d}  {timing[arm]['median_ms']:8.2f} ms")
    if "base_torch" in timing and "nullkey" in timing:
        timing["speedup_vs_base_torch"] = (
            timing["base_torch"]["median_ms"] / timing["nullkey"]["median_ms"]
        )
        timing["speedup_vs_base"] = (
            timing["base"]["median_ms"] / timing["nullkey"]["median_ms"]
        )
        log.info(
            f"nullkey speedup: {timing['speedup_vs_base_torch']:.3f}x vs "
            f"base_torch (kernel-matched), {timing['speedup_vs_base']:.3f}x vs "
            f"base (flash)"
        )

    # ── verdict helper: nullkey drift vs the kernel noise floor ────────────
    floors = [m["rel_l2"] for m in fwd_metrics if m["arm"] == "base"]
    nulls = [m["rel_l2"] for m in fwd_metrics if m["arm"] == "nullkey"]
    verdict = {
        "kernel_floor_rel_l2_max": max(floors) if floors else None,
        "nullkey_rel_l2_max": max(nulls) if nulls else None,
        "nullkey_within_2x_floor": (
            bool(max(nulls) <= 2.0 * max(floors)) if floors and nulls else None
        ),
    }
    log.info(
        f"\nverdict: kernel floor max relL2 {verdict['kernel_floor_rel_l2_max']:.3e}, "
        f"nullkey max relL2 {verdict['nullkey_rel_l2_max']:.3e}, "
        f"within 2x floor: {verdict['nullkey_within_2x_floor']}"
    )

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "arms": list(ARMS),
            "prompts": prompts,
            "n_tokens": [
                {"cond": r["cond"]["n"], "uncond": r["uncond"]["n"]} for r in per_prompt
            ],
            "forward_equivalence": fwd_metrics,
            "trajectories": traj_metrics,
            "timing": timing,
            "verdict": verdict,
            "schedule": {
                "infer_steps": args.infer_steps,
                "flow_shift": args.flow_shift,
                "guidance": args.guidance,
                "probe_sigmas": [sig[i] for i in probe_idx],
            },
        },
        label=args.label,
        artifacts=artifacts,
        device=device,
    )
    log.info(f"\n→ {run_dir}/")


if __name__ == "__main__":
    main()
