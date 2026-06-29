"""RSD network builders: teacher / student / fake-critic / discriminator.

All three SR nets are ResShift's UNetModelSwin (174M). The student & fake are made
STOCHASTIC one-step generators by adding a zero-initialized conv on the noise eps
whose output is injected after input_blocks[0] (RSD App. C) — zero-init => identical
to the frozen teacher at step 0. The discriminator taps the fake critic's bottleneck
([B,640,8,8]) per RSD Fig 2 / Eq 12.

Imports the ResShift clone by path; resolves its relative weight paths under RESSHIFT.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

SR = Path(__file__).resolve().parents[1]   # sr/
REPO = SR.parent                           # repo root (for output/ + data/ paths)
RESSHIFT = SR / "resshift"                 # vendored ResShift source (no external clone)
WEIGHTS = SR / "weights"                   # model checkpoints (gitignored, see sr/README.md)
if str(RESSHIFT) not in sys.path:
    sys.path.insert(0, str(RESSHIFT))

from utils import util_common  # noqa: E402  (vendored ResShift)
from models.unet import UNetModelSwin  # noqa: E402  (vendored ResShift)

CONFIG = RESSHIFT / "configs" / "realsr_swinunet_realesrgan256.yaml"
TEACHER_CKPT = WEIGHTS / "resshift_realsrx4_s15_v2.pth"
DEVICE = "cuda"


def predict_x0(diff, model, z_t, z_y, t, eps=None):
    """ResShift x0 prediction (predict_type=xstart): scale input, forward, output IS x0.

    eps is passed only to the stochastic student/fake nets; the teacher is deterministic.
    """
    kw = {"lq": z_y}
    if eps is not None:
        kw["eps"] = eps
    return model(diff._scale_input(z_t, t), t, **kw)


def load_configs(config_path=CONFIG):
    cfg = OmegaConf.load(config_path)
    # autoencoder ckpt is config-relative ("weights/autoencoder_vq_f4.pth"); point it at sr/weights/
    cfg.autoencoder.ckpt_path = str(WEIGHTS / Path(cfg.autoencoder.ckpt_path).name)
    return cfg


def _strip_module(sd):
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}


class StochasticUNet(UNetModelSwin):
    """UNetModelSwin + a zero-init noise branch -> stochastic one-step generator.

    forward(x, timesteps, lq, eps): adds noise_proj(eps) after the first input conv.
    encode_features(x, lq, t): returns the bottleneck feature map for the GAN head.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_lat = self.input_blocks[0][0].out_channels  # post-conv width (160)
        # eps has the latent's channel count (3); project to the post-conv width.
        self.noise_proj = nn.Conv2d(3, in_lat, 3, padding=1)
        nn.init.zeros_(self.noise_proj.weight)
        nn.init.zeros_(self.noise_proj.bias)

    def _embed_and_concat(self, x, timesteps, lq):
        emb = self.time_embed(
            self._timestep_embedding(timesteps)
        ).type(self.dtype)
        if lq is not None:
            assert self.cond_lq
            lq = self.feature_extractor(lq.type(self.dtype))
            x = torch.cat([x, lq], dim=1)
        return x.type(self.dtype), emb

    def _timestep_embedding(self, timesteps):
        from models.unet import timestep_embedding
        return timestep_embedding(timesteps, self.model_channels)

    def forward(self, x, timesteps, lq=None, eps=None, mask=None):
        h, emb = self._embed_and_concat(x, timesteps, lq)
        hs = []
        for ii, module in enumerate(self.input_blocks):
            h = module(h, emb)
            if ii == 0 and eps is not None:
                h = h + self.noise_proj(eps.type(self.dtype))
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)

    def encode_features(self, x, lq=None, timesteps=None, eps=None):
        """Bottleneck features for the GAN head: input_blocks + middle ResBlock."""
        if timesteps is None:
            timesteps = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        h, emb = self._embed_and_concat(x, timesteps, lq)
        for ii, module in enumerate(self.input_blocks):
            h = module(h, emb)
            if ii == 0 and eps is not None:
                h = h + self.noise_proj(eps.type(self.dtype))
        # first sub-module of middle_block is the ResBlock -> bottleneck feats
        h = self.middle_block[0](h, emb)
        return h  # [B, 640, 8, 8]


class DiscHead(nn.Module):
    """Small DMD2-style discriminator on the fake-critic bottleneck [B,640,8,8]->[B,1]."""

    def __init__(self, in_ch=640):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 256, 4, 2, 1), nn.GroupNorm(32, 256), nn.SiLU(),   # 8->4
            nn.Conv2d(256, 256, 4, 2, 1), nn.GroupNorm(32, 256), nn.SiLU(),     # 4->2
            nn.Conv2d(256, 1, 2, 1, 0),                                          # 2->1
        )

    def forward(self, feats):
        return self.net(feats).flatten(1).mean(1)  # [B]


def _build_unet(cfg, cls):
    params = OmegaConf.to_container(cfg.model.params, resolve=True)
    return cls(**params)


def build_teacher(cfg, ckpt_path, device, dtype=torch.float32):
    model = _build_unet(cfg, UNetModelSwin)
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
    model.load_state_dict(_strip_module(sd), strict=True)
    model = model.to(device, dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_generator(cfg, teacher_ckpt, device, dtype=torch.float32):
    """Student or fake: teacher weights + zero-init noise branch (strict=False)."""
    model = _build_unet(cfg, StochasticUNet)
    sd = torch.load(teacher_ckpt, map_location="cpu")
    sd = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
    missing, unexpected = model.load_state_dict(_strip_module(sd), strict=False)
    # only noise_proj.* should be missing from the ckpt
    assert all("noise_proj" in m for m in missing), f"unexpected missing: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    return model.to(device, dtype)


def build_autoencoder(cfg, device):
    ae = util_common.get_obj_from_str(cfg.autoencoder.target)(
        **OmegaConf.to_container(cfg.autoencoder.params, resolve=True)
    )
    sd = torch.load(cfg.autoencoder.ckpt_path, map_location="cpu")
    sd = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
    ae.load_state_dict(_strip_module(sd), strict=False)
    ae = ae.to(device).eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


def build_diffusion(cfg):
    """ResShift GaussianDiffusion (residual-shift forward). Used for q_sample + x0-pred."""
    return util_common.get_obj_from_str(cfg.diffusion.target)(
        **OmegaConf.to_container(cfg.diffusion.params, resolve=True)
    )
