#!/usr/bin/env python3
"""Phase 0 — DPCache paper-check: does DP-planned global placement beat the
shipped Spectrum schedules (window / SEA) at matched compute, on paper?

Proposal: docs/proposal/dpcache_dp_schedule.md (PR #74). No inference-path code
is touched — everything here is offline analysis over recorded full-step
trajectories, plus cheap head replays (t_embedder → final_layer → unpatchify).

Pipeline (per the proposal's Phase 0, with the stop_caching_step discipline from
the SEA P1 refutation — all arms share the SAME stop rule, live default n−3):

  1. Record full-step trajectories (final_layer features both CFG branches,
     model timesteps, x_t, guided noise_pred) for the prompt set at the shipped
     geometry. Cached to disk so analysis re-runs skip generation.
  2. Build open-loop PACT per the paper: segment cost C[i,j,k] = cumulative
     relative-L1 feature-forecast error over skipped steps j+1..k-1, with the
     predictor conditioned on the warmup prefix + keys {i, j} (a good match for
     Spectrum's forecaster: at w=0.3 the prediction is 70% two-point Taylor,
     which depends on exactly the last two keys). Pooled over prompts/branches.
  3. DP-select schedules at K matched to the window's realized decision-region
     refresh count. Two arms: constrained (warmup+stop forced, live invariants)
     and tail-free (only warmup forced, matched TOTAL compute — the tail check).
  4. Score every arm under the open-loop schedule replay: predictors update only
     at that schedule's actual steps; cached steps get forecast + head replay +
     CFG combine; cost = SEA-filtered x̂₀ deviation vs the full-compute x̂₀
     (the phase-1 gt_cf convention). Reference arms: greedy_c1 (cheapest steps
     by single-step counterfactual cost — the no-segment-model control) and an
     optional local-search oracle (upper bound for ANY open-loop planner).

Gate (proposal §Phase 0): dp must beat window and at least tie sea on summed
replay cost at matched K. The oracle arm strengthens the falsifier: if even
oracle placement can't beat the window, global planning is dead at Anima's
step counts regardless of cost model.

Usage::

    python bench/spectrum_dp/run_bench.py --label phase0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402

import anima_lora  # noqa: E402
import library.inference.sampling as _sampling  # noqa: E402
from library.env import default_checkpoints  # noqa: E402
from library.runtime.harness import build_inference_bundle  # noqa: E402
from networks.spectrum import _spectrum_fast_forward  # noqa: E402
from networks.spectrum_forecast import DTYPE as CHEB_DTYPE  # noqa: E402
from networks.spectrum_forecast import SpectrumPredictor  # noqa: E402
from networks.spectrum_sea import l1rel, sea_filter  # noqa: E402

# Same prompt set as the archived SEA bench (_archive/bench/spectrum_sea/).
DEFAULT_PROMPTS = [
    "1girl, long silver hair, red eyes, school uniform, classroom, afternoon light",
    "1girl, kimono, holding a paper umbrella, cherry blossoms, full body, detailed",
    "1boy, hoodie, city street at night, neon signs, rain, cinematic lighting",
    "2girls, sitting on a rooftop, summer, blue sky, clouds, wide shot",
    "1girl, witch hat, casting a spell, glowing particles, dark forest background",
    "1girl, swimsuit, beach, ocean waves, sunset, looking back at viewer",
    "1girl, maid outfit, serving tea, cozy cafe interior, warm lighting",
    "1boy, samurai armor, holding a katana, snowy mountain, dramatic sky",
    "1girl, mecha pilot suit, cockpit, holographic displays, sci-fi",
    "1girl, gothic lolita dress, sitting in a library, candlelight, books",
    "2boys, sparring, dojo, motion blur, dynamic pose, wide shot",
    "1girl, angel wings, floating in clouds, soft sunlight, ethereal",
]

TAIL_SPLIT = 0.45  # σ below this = resolve tail (x̂₀ settled by ~0.45 on Anima)

# Spectrum's shipped defaults (spectrum_denoise signature).
WARMUP, WINDOW, FLEX = 6, 2.0, 0.25
CHEB_M, CHEB_LAM, CHEB_W = 3, 0.1, 0.3
_EPS = 1e-8


# --------------------------------------------------------------------------- #
# capture — both CFG branches + the model timestep (for the head replay)
# --------------------------------------------------------------------------- #
class Capture:
    def __init__(self) -> None:
        self.feat_buf: list[torch.Tensor] = []
        self.t_buf: list[torch.Tensor] = []
        self.steps: list[dict] = []

    def final_layer_pre_hook(self, module, args):
        self.feat_buf.append(args[0].detach().to("cpu", torch.float16))

    def t_embedder_pre_hook(self, module, args):
        self.t_buf.append(args[0].detach().clone())

    @contextmanager
    def patched_step(self):
        orig = _sampling.step

        def patched(latents, noise_pred, sigmas, step_i):
            feats = list(self.feat_buf)
            t_model = self.t_buf[-1] if self.t_buf else None
            self.feat_buf, self.t_buf = [], []
            # cond fires first, uncond second; take the LAST two firings so any
            # pre-step compile-warm pollution is ignored. A single batched pair
            # (B=2) splits into rows.
            if len(feats) >= 2:
                fc, fu = feats[-2], feats[-1]
            elif len(feats) == 1 and feats[0].shape[0] == 2:
                fc, fu = feats[0][0:1], feats[0][1:2]
            else:
                fc = fu = feats[0] if feats else None
            sig = float(sigmas[step_i])
            x_t = latents[0, :, 0].detach().to("cpu", torch.float32)  # (C,H,W)
            v = noise_pred[0, :, 0].detach().to("cpu", torch.float32)
            self.steps.append(
                {
                    "i": int(step_i),
                    "sigma": sig,
                    "x_t": x_t,
                    "x0_true": x_t - sig * v,
                    "feat_c": fc,
                    "feat_u": fu,
                    "t_model": t_model,
                }
            )
            return orig(latents, noise_pred, sigmas, step_i)

        _sampling.step = patched
        try:
            yield
        finally:
            _sampling.step = orig


def make_args(ckpt, prompt, seed, a):
    req = anima_lora.GenerationRequest(
        dit=ckpt.dit,
        vae=ckpt.vae,
        text_encoder=ckpt.text_encoder,
        prompt=prompt,
        image_size=(a.size, a.size),
        infer_steps=a.steps,
        guidance_scale=a.cfg,
        flow_shift=a.flow_shift,
        sampler="euler",
        seed=seed,
        extra_argv=("--compile_blocks",) if not a.no_compile else (),
    )
    return req.to_args()


def traj_path(traj_dir: Path, prompt: str, a) -> Path:
    key = hashlib.sha1(prompt.encode()).hexdigest()[:10]
    geo = f"s{a.steps}_cfg{a.cfg:g}_fs{a.flow_shift:g}_sz{a.size}_seed{a.seed}"
    return traj_dir / geo / f"{key}.pt"


def capture_trajectory(bundle, ckpt, prompt, a) -> list[dict]:
    cap = Capture()
    hf = bundle.model.final_layer.register_forward_pre_hook(cap.final_layer_pre_hook)
    ht = bundle.model.t_embedder.register_forward_pre_hook(cap.t_embedder_pre_hook)
    try:
        with cap.patched_step():
            bundle.generate(make_args(ckpt, prompt, a.seed, a))
    finally:
        hf.remove()
        ht.remove()
    steps = sorted(cap.steps, key=lambda r: r["i"])
    assert all(s["feat_c"] is not None for s in steps), "feature capture failed"
    assert all(s["t_model"] is not None for s in steps), "timestep capture failed"
    return steps


# --------------------------------------------------------------------------- #
# schedules — all arms share the live stop rule (stop_at = n − 3)
# --------------------------------------------------------------------------- #
def window_schedule(
    n: int, warmup: int, stop_at: int, window: float, flex: float
) -> list[bool]:
    """Replicate the live growing-window rule (spectrum.py decision branch)."""
    curr_ws, consec = window, 0
    modes: list[bool] = []
    for i in range(n):
        actual = (i < warmup or i >= stop_at) or (
            (consec + 1) % max(1, math.floor(curr_ws)) == 0
        )
        modes.append(actual)
        if actual:
            if i >= warmup:
                curr_ws = round(curr_ws + flex, 3)
            consec = 0
        else:
            consec += 1
    return modes


def sea_schedule(
    dists: list[float], n: int, warmup: int, stop_at: int, delta: float
) -> list[bool]:
    """Exact live SEA accumulate-and-fire replay (spectrum.py:414-435).

    ``dists[i]`` is the SEA-filtered l1rel between x_t[i] and x_t[i-1] (NaN/0 at
    i=0). Accumulates every step, decides only in the decision region, resets on
    every actual forward (forced ones included) — matching the live loop.
    """
    accum = 0.0
    modes: list[bool] = []
    for i in range(n):
        if i >= 1:
            accum += dists[i]
        actual = (i < warmup or i >= stop_at) or accum >= delta
        modes.append(actual)
        if actual:
            accum = 0.0
    return modes


def solve_sea_delta(
    dists: list[float], n: int, warmup: int, stop_at: int, target: int
) -> float:
    """δ whose exact live replay yields ≈``target`` decision-region refreshes."""

    def dec_count(delta: float) -> int:
        m = sea_schedule(dists, n, warmup, stop_at, delta)
        return sum(m[warmup:stop_at])

    lo, hi = 0.0, max(sum(d for d in dists if np.isfinite(d)), _EPS)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if dec_count(mid) > target:
            lo = mid
        else:
            hi = mid
    return max(hi, _EPS)


def mask_str(mask: list[bool]) -> str:
    return "".join("A" if m else "." for m in mask)


# --------------------------------------------------------------------------- #
# predictor helpers
# --------------------------------------------------------------------------- #
def new_pred(n: int, device) -> SpectrumPredictor:
    return SpectrumPredictor(CHEB_M, CHEB_LAM, CHEB_W, torch.device(device), None, n)


def clone_pred(p: SpectrumPredictor, n: int) -> SpectrumPredictor:
    q = new_pred(n, p.cheb.device or "cpu")
    q.cheb.t_buf = p.cheb.t_buf.clone()
    q.cheb._H_buf = None if p.cheb._H_buf is None else p.cheb._H_buf.clone()
    q.cheb._shape = p.cheb._shape
    return q


@torch.no_grad()
def predict_many(p: SpectrumPredictor, t_stars: list[int]) -> torch.Tensor:
    """Batched predict() — identical math, one design matmul for all t_stars.

    Returns ``(S, F)`` flattened predictions in the blend dtype.
    """
    cheb = p.cheb
    cheb._fit_if_needed()
    device = cheb.t_buf.device
    ts = torch.as_tensor([float(t) for t in t_stars], dtype=CHEB_DTYPE, device=device)
    x_star = cheb._build_design(cheb._taus(ts))  # (S, P)
    h_cheb = x_star @ cheb._coef  # (S, F)
    if p.w >= 1.0 or cheb.t_buf.numel() < 2:
        return h_cheb
    H, t = cheb._H_buf, cheb.t_buf
    dt = (t[-1] - t[-2]).clamp_min(1e-8)
    k = ((ts - t[-1]) / dt).to(H.dtype).reshape(-1, 1)  # (S, 1)
    h_taylor = H[-1].unsqueeze(0) + k * (H[-1] - H[-2]).unsqueeze(0)
    return (1 - p.w) * h_taylor + p.w * h_cheb


def rel_l1(pred_flat: torch.Tensor, true_flat: torch.Tensor) -> float:
    return float(
        (pred_flat.float() - true_flat.float()).abs().sum()
        / (true_flat.float().abs().sum() + _EPS)
    )


# --------------------------------------------------------------------------- #
# PACT — open-loop Path-Aware Cost Tensor, pooled over prompts and branches
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_pact(trajs: list[list[dict]], n: int, warmup: int, device) -> dict:
    """err[(i, j)] = per-step forecast rel-L1 at steps j+1..n-1, predictor
    conditioned on the warmup prefix + keys {i, j}. Pooled mean over
    prompts × branches. C[i,j,k] follows by prefix-summing err[(i,j)]."""
    pairs = [(i, j) for j in range(warmup - 1, n) for i in range(warmup - 2, j)]
    pooled: dict[tuple[int, int], np.ndarray] = {
        p: np.zeros(n, dtype=np.float64) for p in pairs
    }
    denom = 0
    for traj in trajs:
        for key in ("feat_c", "feat_u"):
            feats = [s[key].to(device) for s in traj]
            denom += 1
            prefix = new_pred(n, device)
            for s in range(warmup):
                prefix.update(float(s), feats[s].float())
            for i in range(warmup - 2, n - 2):
                pi = clone_pred(prefix, n) if i >= warmup else prefix
                if i >= warmup:
                    pi.update(float(i), feats[i].float())
                for j in range(max(i + 1, warmup - 1), n):
                    if (i, j) not in pooled:
                        continue
                    steps_ahead = list(range(j + 1, n))
                    if not steps_ahead:
                        continue  # j = n-1: nothing to skip, cost stays 0
                    pj = clone_pred(pi, n)
                    if j >= warmup:
                        pj.update(float(j), feats[j].float())
                    preds = predict_many(pj, steps_ahead)
                    for si, s in enumerate(steps_ahead):
                        pooled[(i, j)][s] += rel_l1(preds[si], feats[s].reshape(-1))
            del feats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for p in pooled:
        pooled[p] /= max(1, denom)
    return pooled


def dp_plan(
    err: dict,
    n: int,
    warmup: int,
    domain_end: int,
    budget: int,
) -> list[bool]:
    """DPCache DP over segment costs derived from ``err``.

    Places ``budget`` keys inside [warmup, domain_end); steps < warmup and
    >= domain_end are forced actual. Segment cost of choosing next key k after
    keys (i, j) = sum of err[(i,j)][j+1..k-1]; the terminal segment connects the
    last key to domain_end the same way. State = (prev_key, cur_key).
    """
    cum = {p: np.concatenate([[0.0], np.cumsum(v)]) for p, v in err.items()}

    def seg(i: int, j: int, k: int) -> float:
        # sum err[(i,j)][s] for s in j+1..k-1
        return float(cum[(i, j)][k] - cum[(i, j)][j + 1])

    start = (warmup - 2, warmup - 1)
    dp: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, tuple[int, int]], tuple[int, int] | None] = {}
    for b in range(budget):
        ndp: dict[tuple[int, int], float] = {}
        for (i, j), c in dp.items():
            for k in range(max(j + 1, warmup), domain_end):
                nc = c + seg(i, j, k)
                if nc < ndp.get((j, k), float("inf")):
                    ndp[(j, k)] = nc
                    parent[(b, (j, k))] = (i, j)
        dp = ndp
    best_cost, best_state = float("inf"), None
    for (i, j), c in dp.items():
        total = c + seg(i, j, domain_end)
        if total < best_cost:
            best_cost, best_state = total, (i, j)
    assert best_state is not None, "DP infeasible: budget too large for domain"
    keys = []
    state, b = best_state, budget - 1
    while b >= 0:
        keys.append(state[1])
        state = parent[(b, state)]
        b -= 1
    mask = [i < warmup or i >= domain_end for i in range(n)]
    for k in keys:
        mask[k] = True
    return mask


# --------------------------------------------------------------------------- #
# scoring — open-loop schedule replay with real head replays
# --------------------------------------------------------------------------- #
@torch.no_grad()
def replay_cost(model, traj, mask, cfg, beta, device, n, pred_device="cpu") -> dict:
    """Score a schedule: predictors update only at actual steps; cached steps
    get forecast + head replay + CFG combine; cost = SEA-filtered x̂₀ rel-L1 vs
    the recorded full-compute x̂₀ (phase-1 gt_cf convention)."""
    fc_c, fc_u = new_pred(n, pred_device), new_pred(n, pred_device)
    per_step = [0.0] * n
    per_step_raw = [0.0] * n
    for i, s in enumerate(traj):
        if mask[i]:
            fc_c.update(float(i), s["feat_c"].float())
            fc_u.update(float(i), s["feat_u"].float())
            continue
        t_m = s["t_model"].to(device)
        cf = fc_c.predict(float(i)).to(device, torch.bfloat16)
        nc = _spectrum_fast_forward(model, t_m, cf)[0, :, 0].float().cpu()
        uf = fc_u.predict(float(i)).to(device, torch.bfloat16)
        nu = _spectrum_fast_forward(model, t_m, uf)[0, :, 0].float().cpu()
        noise_cached = nu + cfg * (nc - nu)
        x0_cached = s["x_t"] - s["sigma"] * noise_cached
        per_step[i] = l1rel(
            sea_filter(s["x0_true"], s["sigma"], beta),
            sea_filter(x0_cached, s["sigma"], beta),
        )
        per_step_raw[i] = l1rel(s["x0_true"], x0_cached)
    tail = sum(c for i, c in enumerate(per_step) if traj[i]["sigma"] < TAIL_SPLIT)
    return {
        "total": float(sum(per_step)),
        "total_raw": float(sum(per_step_raw)),
        "tail": float(tail),
        "per_step": per_step,
        "n_actual": int(sum(mask)),
    }


@torch.no_grad()
def single_step_costs(
    model, traj, cfg, beta, device, n, warmup, pred_device="cpu"
) -> np.ndarray:
    """Phase-1-style per-step counterfactual cost c1[i]: full-history predictor,
    cache exactly step i, measure injected x̂₀ error."""
    fc_c, fc_u = new_pred(n, pred_device), new_pred(n, pred_device)
    c1 = np.full(n, np.nan)
    for i, s in enumerate(traj):
        if i >= warmup and fc_c.cheb.t_buf.numel() >= max(2, CHEB_M + 2):
            t_m = s["t_model"].to(device)
            cf = fc_c.predict(float(i)).to(device, torch.bfloat16)
            nc = _spectrum_fast_forward(model, t_m, cf)[0, :, 0].float().cpu()
            uf = fc_u.predict(float(i)).to(device, torch.bfloat16)
            nu = _spectrum_fast_forward(model, t_m, uf)[0, :, 0].float().cpu()
            noise_cached = nu + cfg * (nc - nu)
            x0_cached = s["x_t"] - s["sigma"] * noise_cached
            c1[i] = l1rel(
                sea_filter(s["x0_true"], s["sigma"], beta),
                sea_filter(x0_cached, s["sigma"], beta),
            )
        fc_c.update(float(i), s["feat_c"].float())
        fc_u.update(float(i), s["feat_u"].float())
    return c1


def local_search(
    model,
    trajs,
    mask0,
    cfg,
    beta,
    device,
    n,
    warmup,
    stop_at,
    passes,
    max_evals,
    pred_device,
):
    """First-improvement hill climb on mean replay cost: move one decision-region
    actual to a decision-region cached slot. Upper bound for open-loop planners."""

    def objective(mask):
        return float(
            np.mean(
                [
                    replay_cost(model, t, mask, cfg, beta, device, n, pred_device)[
                        "total"
                    ]
                    for t in trajs
                ]
            )
        )

    mask = list(mask0)
    best = objective(mask)
    evals = 1
    for _ in range(passes):
        improved = False
        movable = [i for i in range(warmup, stop_at) if mask[i]]
        empty = [i for i in range(warmup, stop_at) if not mask[i]]
        for src in movable:
            for dst in empty:
                if evals >= max_evals:
                    return mask, best, evals
                cand = list(mask)
                cand[src], cand[dst] = False, True
                c = objective(cand)
                evals += 1
                if c < best - 1e-9:
                    mask, best, improved = cand, c, True
                    break
            if improved:
                break
        if not improved:
            break
    return mask, best, evals


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--flow-shift", dest="flow_shift", type=float, default=3.0)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompts-file", default=None)
    p.add_argument("--limit-prompts", type=int, default=0)
    p.add_argument("--label", default=None)
    p.add_argument("--no-compile", dest="no_compile", action="store_true")
    p.add_argument("--traj-dir", default=str(REPO_ROOT / "output" / "spectrum_dp_traj"))
    p.add_argument("--analysis-device", default="cuda")
    p.add_argument("--oracle-passes", type=int, default=2)
    p.add_argument("--oracle-max-evals", type=int, default=250)
    a = p.parse_args()

    if a.cfg == 1.0:
        sys.exit("Phase 0 needs CFG>1 to exercise the cond/uncond cache replay.")
    prompts = (
        [
            ln.strip()
            for ln in Path(a.prompts_file).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        if a.prompts_file
        else DEFAULT_PROMPTS
    )
    if a.limit_prompts > 0:
        prompts = prompts[: a.limit_prompts]

    n, warmup = a.steps, WARMUP
    stop_at = n - 3  # live default (spectrum.py: num_steps - 3); pinned for ALL arms
    ckpt = default_checkpoints()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = make_run_dir("spectrum_dp", label=a.label)
    traj_dir = Path(a.traj_dir)
    print(f"run dir: {run_dir}")
    print(
        f"{len(prompts)} prompts, {n} steps, cfg {a.cfg}, stop_at {stop_at}, β {a.beta}"
    )

    bundle = build_inference_bundle(
        make_args(ckpt, prompts[0], a.seed, a), device=device, with_vae=False
    )

    # ---- 1. trajectories (disk-cached; ~1.4GB/prompt at 1024²) -------------- #
    trajs: list[list[dict]] = []
    for pi, prompt in enumerate(prompts):
        tp = traj_path(traj_dir, prompt, a)
        if tp.exists():
            print(f"[{pi + 1}/{len(prompts)}] cached: {prompt[:58]}")
            trajs.append(torch.load(tp, weights_only=False))
            continue
        print(f"[{pi + 1}/{len(prompts)}] capture: {prompt[:58]}")
        steps = capture_trajectory(bundle, ckpt, prompt, a)
        tp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(steps, tp)
        trajs.append(steps)
    assert all(len(t) == n for t in trajs), "trajectory length mismatch"
    sigmas = [s["sigma"] for s in trajs[0]]

    # ---- 2. baseline schedules at matched K --------------------------------- #
    win_mask = window_schedule(n, warmup, stop_at, WINDOW, FLEX)
    k_dec = sum(win_mask[warmup:stop_at])
    print(
        f"window decision-region refreshes K = {k_dec} "
        f"(total actual {sum(win_mask)}/{n})\n  window: {mask_str(win_mask)}"
    )

    sea_masks, sea_deltas = [], []
    for traj in trajs:
        dists = [float("nan")] + [
            l1rel(
                sea_filter(traj[i]["x_t"], traj[i]["sigma"], a.beta),
                sea_filter(traj[i - 1]["x_t"], traj[i - 1]["sigma"], a.beta),
            )
            for i in range(1, n)
        ]
        dists = [0.0 if not np.isfinite(d) else d for d in dists]
        delta = solve_sea_delta(dists, n, warmup, stop_at, k_dec)
        sea_masks.append(sea_schedule(dists, n, warmup, stop_at, delta))
        sea_deltas.append(delta)
    sea_ks = [sum(m[warmup:stop_at]) for m in sea_masks]
    print(
        f"sea per-prompt decision refreshes: {sea_ks} (δ cv "
        f"{np.std(sea_deltas) / max(np.mean(sea_deltas), _EPS):.3f})"
    )

    # ---- 3. PACT + DP -------------------------------------------------------- #
    print("building open-loop PACT ...")
    analysis_dev = a.analysis_device
    try:
        err = build_pact(trajs, n, warmup, analysis_dev)
    except torch.cuda.OutOfMemoryError:
        print("  OOM on analysis device — falling back to cpu (slower)")
        torch.cuda.empty_cache()
        analysis_dev = "cpu"
        err = build_pact(trajs, n, warmup, analysis_dev)

    dp_mask = dp_plan(err, n, warmup, stop_at, k_dec)
    budget_free = k_dec + (n - stop_at)  # matched TOTAL post-warmup compute
    dp_free_mask = dp_plan(err, n, warmup, n, budget_free)
    print(f"  dp      : {mask_str(dp_mask)}\n  dp_free : {mask_str(dp_free_mask)}")

    # ---- 4. single-step counterfactual costs → greedy control ---------------- #
    print("single-step counterfactual sweep ...")
    c1_all = np.stack(
        [
            single_step_costs(
                bundle.model, t, a.cfg, a.beta, device, n, warmup, analysis_dev
            )
            for t in trajs
        ]
    )
    c1_mean = np.nanmean(c1_all, axis=0)
    order = sorted(range(warmup, stop_at), key=lambda i: -c1_mean[i])
    greedy_mask = [i < warmup or i >= stop_at for i in range(n)]
    for i in order[:k_dec]:
        greedy_mask[i] = True
    print(f"  greedy  : {mask_str(greedy_mask)}")

    # ---- 5. score all arms under the open-loop replay ------------------------ #
    print("scoring arms under open-loop schedule replay ...")
    arms: dict[str, list[dict]] = {}

    def score_arm(name, mask_or_masks):
        masks = (
            mask_or_masks
            if isinstance(mask_or_masks[0], list)
            else [mask_or_masks] * len(trajs)
        )
        arms[name] = [
            replay_cost(bundle.model, t, m, a.cfg, a.beta, device, n, analysis_dev)
            for t, m in zip(trajs, masks)
        ]
        tot = [r["total"] for r in arms[name]]
        print(f"  {name:10s} S = {np.mean(tot):.4f} ± {np.std(tot):.4f}")

    score_arm("window", win_mask)
    score_arm("sea", sea_masks)
    score_arm("dp", dp_mask)
    score_arm("dp_tailfree", dp_free_mask)
    score_arm("greedy_c1", greedy_mask)

    oracle_mask, oracle_evals = None, 0
    if a.oracle_passes > 0:
        print("oracle local search ...")
        max_evals = (
            a.oracle_max_evals if analysis_dev != "cpu" else min(a.oracle_max_evals, 60)
        )
        oracle_mask, _, oracle_evals = local_search(
            bundle.model,
            trajs,
            dp_mask,
            a.cfg,
            a.beta,
            device,
            n,
            warmup,
            stop_at,
            a.oracle_passes,
            max_evals,
            analysis_dev,
        )
        score_arm("oracle", oracle_mask)
        print(f"  oracle  : {mask_str(oracle_mask)}  ({oracle_evals} evals)")

    # ---- 6. verdict ---------------------------------------------------------- #
    def means(name):
        return float(np.mean([r["total"] for r in arms[name]]))

    def wins(x, y):
        return int(sum(rx["total"] < ry["total"] for rx, ry in zip(arms[x], arms[y])))

    s_win, s_sea, s_dp = means("window"), means("sea"), means("dp")
    gate_beats_window = s_dp < s_win
    gate_ties_sea = s_dp <= s_sea * 1.02
    tail_steps_free = [
        i for i in range(warmup, n) if dp_free_mask[i] and sigmas[i] < TAIL_SPLIT
    ]
    tail_steps_window = [
        i for i in range(warmup, n) if win_mask[i] and sigmas[i] < TAIL_SPLIT
    ]

    metrics = {
        "geometry": {
            "steps": n,
            "warmup": warmup,
            "stop_at": stop_at,
            "cfg": a.cfg,
            "flow_shift": a.flow_shift,
            "size": a.size,
            "beta": a.beta,
            "seed": a.seed,
            "k_dec": k_dec,
            "budget_tailfree": budget_free,
            "n_prompts": len(prompts),
            "sigmas": sigmas,
        },
        "schedules": {
            "window": mask_str(win_mask),
            "sea_per_prompt": [mask_str(m) for m in sea_masks],
            "sea_deltas": sea_deltas,
            "sea_k_dec": sea_ks,
            "dp": mask_str(dp_mask),
            "dp_tailfree": mask_str(dp_free_mask),
            "greedy_c1": mask_str(greedy_mask),
            "oracle": mask_str(oracle_mask) if oracle_mask else None,
        },
        "replay_cost": {
            name: {
                "mean": means(name),
                "std": float(np.std([r["total"] for r in rows])),
                "per_prompt": [r["total"] for r in rows],
                "mean_raw": float(np.mean([r["total_raw"] for r in rows])),
                "tail_mean": float(np.mean([r["tail"] for r in rows])),
                "n_actual": rows[0]["n_actual"],
            }
            for name, rows in arms.items()
        },
        "c1_mean_per_step": [None if not np.isfinite(v) else float(v) for v in c1_mean],
        "gate": {
            "dp_beats_window": gate_beats_window,
            "dp_ties_sea": gate_ties_sea,
            "pass": bool(gate_beats_window and gate_ties_sea),
            "dp_vs_window_win_rate": wins("dp", "window") / len(prompts),
            "dp_vs_sea_win_rate": wins("dp", "sea") / len(prompts),
        },
        "tail_check": {
            "tailfree_actual_steps_below_split": tail_steps_free,
            "window_actual_steps_below_split": tail_steps_window,
            "planner_abandons_tail": len(tail_steps_free) < len(tail_steps_window),
        },
        "oracle_evals": oracle_evals,
    }

    # per-step cost CSV (window arm per-step curves + c1)
    with (run_dir / "per_step.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["step", "sigma", "c1_mean"] + [f"S_{k}" for k in arms])
        for i in range(n):
            wr.writerow(
                [i, sigmas[i], c1_mean[i]]
                + [float(np.mean([r["per_step"][i] for r in arms[k]])) for k in arms]
            )

    write_result(
        run_dir,
        script=__file__,
        args=a,
        metrics=metrics,
        label=a.label,
        artifacts=["per_step.csv"],
        device=device,
    )

    print("\n=== Phase 0 verdict ===")
    for name in arms:
        print(f"  {name:12s} S = {means(name):.4f}")
    print(
        f"  dp beats window : {gate_beats_window} "
        f"(win rate {metrics['gate']['dp_vs_window_win_rate']:.2f})"
    )
    print(
        f"  dp ties sea     : {gate_ties_sea} "
        f"(win rate {metrics['gate']['dp_vs_sea_win_rate']:.2f})"
    )
    print(f"  GATE            : {'PASS' if metrics['gate']['pass'] else 'FAIL'}")
    print(
        f"  tail check      : tail-free planner keeps "
        f"{len(tail_steps_free)} σ<{TAIL_SPLIT} forwards "
        f"(window forces {len(tail_steps_window)})"
    )
    print(f"\nresult.json + per_step.csv in: {run_dir}")


if __name__ == "__main__":
    main()
