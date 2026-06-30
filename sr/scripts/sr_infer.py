"""ResShift inference — basicsr-free, over the vendored source.

A self-contained reimplementation of ResShift's ``inference_resshift.py`` + the core
of ``sampler.py``, using only the vendored ``sr/resshift`` tree (models / ldm /
util_image) — no basicsr, no datapipe, no external clone. Handles the released realsr
x4 versions (v1/v2 = 15-step, v3 = 4-step) AND our locally-trained ``x2`` art model
(version "x2" — config sr/configs/realsr_x2_art.yaml, checkpoint from output/sr/x2 or
--ckpt). The scale factor is read from the config (NOT hardcoded), so the chop/tiling
math generalizes. Tiled inference for large images + weight auto-download (torch.hub)
for the released x4 versions.

    python sr_infer.py -i <img|dir> -o <out_dir> [--version v3] [--chop_size 512]
    python sr_infer.py -i <img|dir> --version x2 [--ckpt output/sr/x2/resshift_x2_final.pth]
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# vendored ResShift source + rsd_models builders
HERE = Path(__file__).resolve().parent          # sr/scripts
SR = HERE.parent                                 # sr/
sys.path.insert(0, str(SR / "distill_rsd"))
import rsd_models as M  # noqa: E402

from utils import util_common, util_image, util_net  # noqa: E402  (vendored ResShift)
from utils.util_image import ImageSpliterTh  # noqa: E402

# version -> (config filename, checkpoint filename, release URL)
_REL = "https://github.com/zsyOAOA/ResShift/releases/download/v2.0"
_VQGAN = ("autoencoder_vq_f4.pth", f"{_REL}/autoencoder_vq_f4.pth")
_VERSIONS = {
    "v1": ("realsr_swinunet_realesrgan256.yaml", "resshift_realsrx4_s15_v1.pth",
           f"{_REL}/resshift_realsrx4_s15_v1.pth"),
    "v2": ("realsr_swinunet_realesrgan256.yaml", "resshift_realsrx4_s15_v2.pth",
           f"{_REL}/resshift_realsrx4_s15_v2.pth"),
    "v3": ("realsr_swinunet_realesrgan256_journal.yaml", "resshift_realsrx4_s4_v3.pth",
           f"{_REL}/resshift_realsrx4_s4_v3.pth"),
}
# our locally-trained art models (no release URL — checkpoint is produced by make sr-train)
_LOCAL = {
    "x2": (SR / "configs" / "realsr_x2_art.yaml", SR.parent / "output" / "sr" / "x2"),
}


def _latest_local_ckpt(ckpt_dir):
    """Newest resshift_x2_*.pth in a make-sr-train output dir (prefer *_final)."""
    cands = sorted(Path(ckpt_dir).glob("resshift_x2_*.pth"))
    if not cands:
        raise SystemExit(
            f"no resshift_x2_*.pth in {ckpt_dir} — train one first (make sr-train) "
            f"or pass --ckpt <path>.")
    final = [c for c in cands if c.stem.endswith("final")]
    return final[0] if final else cands[-1]


def _ensure_weight(name, url):
    """Return sr/weights/<name>, downloading from the release if absent."""
    dst = M.WEIGHTS / name
    if not dst.exists():
        M.WEIGHTS.mkdir(parents=True, exist_ok=True)
        print(f"[sr_infer] downloading {name} -> {dst}")
        torch.hub.download_url_to_file(url, str(dst), progress=True)
    return dst


class ResShiftInfer:
    """Released ResShift x4 sampler built from the vendored tree."""

    def __init__(self, version="v3", chop_size=512, chop_stride=-1, chop_bs=1,
                 use_amp=True, seed=12345, device=M.DEVICE, ckpt_path=None):
        self.device = device
        self.use_amp = use_amp
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if version in _LOCAL:
            # locally-trained art model (e.g. x2): config + checkpoint live in-repo,
            # no release URL. Our make-sr-train ckpts are {"ema","model","step"} dicts.
            cfg_path, ckpt_dir = _LOCAL[version]
            cfg = M.load_configs(cfg_path)
            ckpt = Path(ckpt_path) if ckpt_path else _latest_local_ckpt(ckpt_dir)
            print(f"[sr_infer] {version}: cfg={Path(cfg_path).name} ckpt={ckpt.name}")
            sd = torch.load(str(ckpt), map_location=device)
            sd = sd.get("ema") or sd.get("model") or sd.get("state_dict") or sd
        else:
            cfg_name, ckpt_name, ckpt_url = _VERSIONS[version]
            cfg = M.load_configs(M.RESSHIFT / "configs" / cfg_name)
            ckpt = _ensure_weight(ckpt_name, ckpt_url)
            sd = torch.load(str(ckpt), map_location=device)
            sd = sd["state_dict"] if "state_dict" in sd else sd
        cfg.autoencoder.ckpt_path = str(_ensure_weight(*_VQGAN))

        # scale factor is config-driven (x4 released = 4, our art model = 2), so all the
        # chop/tiling math below generalizes instead of assuming x4.
        self.sf = int(cfg.diffusion.params.sf)

        # released ckpts carry DDP/compile prefixes (module. / _orig_mod.) — reload_model
        # reconstructs the target keys, unlike build_teacher's strict load (distill-only).
        model = util_common.instantiate_from_config(cfg.model).to(device)
        util_net.reload_model(model, sd)
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model.eval()
        self.autoencoder = M.build_autoencoder(cfg, device)
        self.diffusion = M.build_diffusion(cfg)
        self.cond_lq = cfg.model.params.cond_lq
        # reflect-pad the LR so the encoded latent is divisible by the Swin alignment at
        # every level. latent = y0_px * sf / 4 (VQ-f4); deepest level downsamples by
        # 2^(#levels-1), each needing size % window == 0 -> latent % (window*2^(L-1)) == 0.
        # Solve for the pixel multiple: offset = latent_align * 4 / sf  (x4 -> 64, x2 -> 128).
        # The old hardcoded lq_size=64 only happened to be right for x4 (latent == pixel).
        latent_align = cfg.model.params.window_size * 2 ** (len(cfg.model.params.channel_mult) - 1)
        self.offset = latent_align * 4 // self.sf

        self.chop_size = chop_size * (4 // self.sf)
        if chop_stride < 0:
            self.chop_stride = (chop_size - {512: 64, 256: 32, 64: 16}[chop_size]) * (4 // self.sf)
        else:
            self.chop_stride = chop_stride * (4 // self.sf)
        self.chop_bs = chop_bs

    @torch.no_grad()
    def _sample(self, y0):
        """y0: 1xCxHxW in [-1,1] RGB -> SR tensor in [-1,1], reflect-padded to offset.

        We encode the LR to a LATENT (z_y) and run the reverse loop on latents, rather
        than diffusion.p_sample_loop. p_sample_loop forwards model_kwargs['lq'] through
        UN-encoded, so it only works when the pixel y0 and its latent z_y share spatial
        dims — true at x4 (x4 bicubic-up then /4 VAE is size-preserving) but NOT at x2
        (/2), where pixel-lq vs latent-x_t mismatch (e.g. 512 vs 256). Encoding here is
        scale-agnostic. clip_denoised stays False — latents aren't bounded to [-1,1].
        """
        ori_h, ori_w = y0.shape[2:]
        pad_h = (-ori_h) % self.offset
        pad_w = (-ori_w) % self.offset
        if pad_h or pad_w:
            y0 = F.pad(y0, (0, pad_w, 0, pad_h), mode="reflect")
        z_y = self.diffusion.encode_first_stage(y0, self.autoencoder, up_sample=True)
        z = self.diffusion.prior_sample(z_y)
        kw = {"lq": z_y} if self.cond_lq else None
        for i in reversed(range(self.diffusion.num_timesteps)):
            t = torch.full((z_y.shape[0],), i, device=z_y.device, dtype=torch.long)
            z = self.diffusion.p_sample(self.model, z, z_y, t, clip_denoised=False,
                                        model_kwargs=kw)["sample"]
        out = self.diffusion.decode_first_stage(z.float(), first_stage_model=self.autoencoder)
        if pad_h or pad_w:
            out = out[:, :, : ori_h * self.sf, : ori_w * self.sf]
        return out.clamp_(-1.0, 1.0)

    @torch.no_grad()
    def _process(self, im_lq):
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if self.use_amp else torch.no_grad()
        if im_lq.shape[2] > self.chop_size or im_lq.shape[3] > self.chop_size:
            spliter = ImageSpliterTh(im_lq, self.chop_size, stride=self.chop_stride,
                                     sf=self.sf, extra_bs=self.chop_bs)
            for pch, info in spliter:
                with ctx:
                    spliter.update(self._sample(pch), info)
            sr = spliter.gather()
        else:
            with ctx:
                sr = self._sample(im_lq)
        return sr * 0.5 + 0.5

    def run(self, in_path, out_path):
        in_path, out_path = Path(in_path), Path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in in_path.rglob("*")
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}) \
            if in_path.is_dir() else [in_path]
        for f in files:
            im = util_image.imread(f, chn="rgb", dtype="float32")          # h x w x c
            t = util_image.img2tensor(im).to(self.device)                  # 1 x c x h x w, [0,1]
            sr = self._process((t - 0.5) / 0.5)
            out = util_image.tensor2img(sr, rgb2bgr=True, min_max=(0.0, 1.0))
            util_image.imwrite(out, out_path / f"{f.stem}.png", chn="bgr", dtype_in="uint8")
            print(f"[sr_infer] {f.name} -> {out_path / (f.stem + '.png')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--in_path", required=True, help="input image or directory")
    ap.add_argument("-o", "--out_path", default=str(SR / "data" / "results"))
    ap.add_argument("-v", "--version", default="v3", choices=list(_VERSIONS) + list(_LOCAL),
                    help="released x4: v1/v2/v3 — or locally-trained x2 (make sr-train).")
    ap.add_argument("--ckpt", default=None,
                    help="explicit checkpoint (x2 only; default = newest in output/sr/x2).")
    ap.add_argument("--chop_size", type=int, default=512, choices=[512, 256, 64])
    ap.add_argument("--chop_stride", type=int, default=-1)
    ap.add_argument("--no_amp", action="store_true", help="disable bf16 autocast")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    sampler = ResShiftInfer(version=args.version, chop_size=args.chop_size,
                            chop_stride=args.chop_stride, use_amp=not args.no_amp,
                            seed=args.seed, ckpt_path=args.ckpt)
    sampler.run(args.in_path, args.out_path)
    print(f"[sr_infer] done -> {args.out_path}")


if __name__ == "__main__":
    main()
