"""Phase-0.5 eval: did the trained register adapter relocate the self-attn sink,
and does arm B (learnable) beat arm A (fixed-zero) — i.e. does the sink need
learnable *content* (scratchpad, Theory 2) or just a routing target (Theory 1)?

Measures three conditions on matched seeds — **base** (no registers), **armA**,
**armB** — with one unified mid-block forward hook (the RQ1 probe localized the
~26× sink to intermediate layers). Per generation it accumulates, over the whole
denoising trajectory:

  * ``patch_sink_ratio`` = top-0.2% patch ‖x‖ / median patch ‖x‖ — the DSR
    high-norm outlier on the IMAGE tokens. **Relocation ⇒ this falls** vs base.
  * ``reg_ratio`` = max register ‖x‖ / median patch ‖x‖ — did the sink move ONTO
    a register? **Absorption ⇒ this is high.** Arm A can route but carries no
    content; if only arm B lifts reg_ratio while dropping patch_sink_ratio, the
    sink is a scratchpad (Theory 2).
  * ``border_score`` — unprompted-border rate (RQ1 was falsified for borders, so
    this should be flat; it's the no-regression guard, not the target).

Not the full RQ2 gate — the load-bearing sensitivity-drop (clamp) test reuses
`sink_intervention.py` on a trained adapter and is a follow-up. This nails the
RQ3 crossover + the A/B adjudication + a no-regression eyeball grid.

Run:
  uv run python bench/headroom/register_eval.py \
      --adapter_a <runA>/adapter.pt --adapter_b <runB>/adapter.pt \
      --seeds 4 --save_images
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anima_lora import GenerationRequest, load_vae  # noqa: E402
from bench._anima import add_common_args, add_model_args  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from bench.headroom.register_adapter import RegisterAdapter  # noqa: E402
from library.inference.generation import generate_body  # noqa: E402
from library.inference.models import load_dit_model, load_text_encoder  # noqa: E402
from library.inference.output import decode_latent, pixels_to_pil  # noqa: E402
from library.inference.text import ensure_text_strategies, prepare_text_inputs  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402

_PROMPTS = [
    ("plain", "1girl, solo, long hair, looking at viewer, standing, simple background, "
              "detailed face, masterpiece, best quality"),
    ("scene", "1girl, cafe interior, sitting at a table, window light, cup of coffee, "
              "detailed background, cinematic, masterpiece"),
    ("portrait", "portrait of a woman, close-up, freckles, soft lighting, bokeh, "
                 "highly detailed, photorealistic"),
]


def _border_score(img_chw: torch.Tensor, frac: float = 0.04, flat: float = 0.03,
                  extreme: float = 0.85) -> float:
    c, h, w = img_chw.shape
    x = img_chw.float()
    bh, bw = max(1, int(h * frac)), max(1, int(w * frac))
    edges = torch.cat([
        x[:, :bh, :].reshape(c, -1), x[:, -bh:, :].reshape(c, -1),
        x[:, :, :bw].reshape(c, -1), x[:, :, -bw:].reshape(c, -1),
    ], dim=1)
    luma = edges.mean(0)
    return float(((luma.abs() > extreme) & (edges.std(0) < flat)).float().mean())


class MidBlockProbe:
    """Forward hook on one mid block; splits patch vs register tokens and
    accumulates outlier ratios over a generation's trajectory."""

    def __init__(self, K: int):
        self.K = K
        self.reset()

    def reset(self):
        self.patch_sink = []
        self.reg_ratio = []

    def __call__(self, module, inputs, output):
        # output: (B, 1, seq(+K), 1, D). native-flatten eager path.
        out = output.detach()
        norms = out.float().norm(dim=-1).flatten()  # (seq+K,)
        n = norms.numel()
        seq = n - self.K
        patch = norms[:seq]
        med = patch.median().clamp_min(1e-6)
        k = max(1, int(0.002 * seq))
        self.patch_sink.append((patch.topk(k).values.mean() / med).item())
        if self.K > 0:
            reg = norms[seq:]
            self.reg_ratio.append((reg.max() / med).item())

    def summary(self):
        ps = float(np.mean(self.patch_sink)) if self.patch_sink else float("nan")
        rr = float(np.mean(self.reg_ratio)) if self.reg_ratio else 0.0
        return ps, rr


def load_adapter(anima, path: str, device) -> RegisterAdapter:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob["meta"]["config"]
    adapter = RegisterAdapter(
        anima,
        num_registers=cfg["num_registers"],
        arm=cfg["arm"],
        lora_rank=cfg["lora_rank"],
        target_blocks=cfg["target_blocks"],
    ).to(device=device, dtype=torch.bfloat16)
    missing, unexpected = adapter.load_state_dict(blob["state_dict"], strict=False)
    if missing or unexpected:
        print(f"  load {Path(path).parent.name}: missing={len(missing)} unexpected={len(unexpected)}")
    return adapter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    add_model_args(p)
    p.add_argument("--adapter_a", required=True, help="arm-A adapter.pt")
    p.add_argument("--adapter_b", required=True, help="arm-B adapter.pt")
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--infer_steps", type=int, default=24)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--mid_block", type=int, default=-1, help="block to probe; -1 = len//2")
    p.add_argument("--save_images", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    req = GenerationRequest(
        prompt="x", dit=args.dit, vae=args.vae, text_encoder=args.text_encoder,
        negative_prompt="", image_size=(args.size, args.size), infer_steps=args.infer_steps,
        guidance_scale=args.cfg, flow_shift=args.flow_shift, attn_mode=args.attn_mode,
        device=args.device, seed=0,
    )
    gen_args = req.to_args()
    gen_args.compile_blocks = False

    ensure_text_strategies(args.text_encoder)
    text_encoder = load_text_encoder(gen_args, dtype=torch.bfloat16, device=device)
    text_encoder.eval()
    shared = {"text_encoder": text_encoder, "conds_cache": {}}
    anima = load_dit_model(gen_args, device, torch.bfloat16)
    anima.requires_grad_(False)
    shared["model"] = anima
    vae = load_vae(args.vae, device="cpu", vae_2d=True)
    vae.to(torch.bfloat16).eval()

    mid = args.mid_block if args.mid_block >= 0 else len(anima.blocks) // 2

    conditions = [("base", None), ("armA", args.adapter_a), ("armB", args.adapter_b)]
    out_dir = make_run_dir("headroom", args.label or "reg_eval")
    img_dir = out_dir / "images"
    if args.save_images:
        img_dir.mkdir(exist_ok=True)

    rows = []
    for name, apath in conditions:
        adapter = load_adapter(anima, apath, device) if apath else None
        K = adapter.K if adapter else 0
        if adapter:
            adapter.apply_to()
        probe = MidBlockProbe(K)
        handle = anima.blocks[mid].register_forward_hook(probe)
        try:
            for pname, ptext in _PROMPTS:
                ctx, ctx_null = prepare_text_inputs(gen_args, device, anima, shared, prompt=ptext)
                for s in range(args.seeds):
                    probe.reset()
                    with torch.no_grad():
                        latent = generate_body(gen_args, anima, ctx, ctx_null, device, seed=s)
                    img = decode_latent(vae, latent, device)
                    clean_memory_on_device(device)
                    ps, rr = probe.summary()
                    bs = _border_score(img)
                    rows.append({
                        "cond": name, "prompt": pname, "seed": s,
                        "patch_sink_ratio": round(ps, 4),
                        "reg_ratio": round(rr, 4),
                        "border_score": round(bs, 4),
                    })
                    if args.save_images:
                        pixels_to_pil(img).save(img_dir / f"{name}_{pname}_s{s}.png")
        finally:
            handle.remove()
            if adapter:
                adapter.remove()

    # aggregate per condition
    def agg(name, key):
        vals = [r[key] for r in rows if r["cond"] == name]
        return round(float(np.mean(vals)), 4) if vals else float("nan")

    summary = {}
    for name, _ in conditions:
        summary[name] = {
            "patch_sink_ratio": agg(name, "patch_sink_ratio"),
            "reg_ratio": agg(name, "reg_ratio"),
            "border_score": agg(name, "border_score"),
        }

    base_ps = summary["base"]["patch_sink_ratio"]
    verdict = {
        "mid_block": mid,
        "patch_sink_drop_armA": round(base_ps - summary["armA"]["patch_sink_ratio"], 4),
        "patch_sink_drop_armB": round(base_ps - summary["armB"]["patch_sink_ratio"], 4),
        "reg_ratio_armA": summary["armA"]["reg_ratio"],
        "reg_ratio_armB": summary["armB"]["reg_ratio"],
    }

    write_result(out_dir, script=__file__, args=args,
                 metrics={"summary": summary, "verdict": verdict, "rows": rows},
                 label=args.label)

    print("\n=== RELOCATION CROSSOVER (mid block %d) ===" % mid)
    print(f"{'cond':6} {'patch_sink_ratio':>17} {'reg_ratio':>10} {'border':>8}")
    for name, _ in conditions:
        s = summary[name]
        print(f"{name:6} {s['patch_sink_ratio']:>17} {s['reg_ratio']:>10} {s['border_score']:>8}")
    print(f"\npatch-sink drop vs base:  armA {verdict['patch_sink_drop_armA']:+.4f}   "
          f"armB {verdict['patch_sink_drop_armB']:+.4f}")
    print(f"register absorption:      armA {verdict['reg_ratio_armA']:.3f}   "
          f"armB {verdict['reg_ratio_armB']:.3f}")
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
