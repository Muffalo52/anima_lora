"""sigma_lowres Phase 1b invariants (σ>0.5 → 896 sibling latent).

Pins the load-bearing contracts of the trainer wiring:
  - the demote grid is a pure function shared by preprocess emit and trainer
    fetch (same inputs → same bucket), and off-route shapes return None;
  - the demoted npz key can never collide with the ``latents_*`` namespace
    (several readers grab the FIRST ``latents_*`` key);
  - ``draw_flat_sigmas`` is bit-identical to the in-body draw it was split
    from, and the σ-first two-step path reproduces the draw-inside path
    exactly (same seed → same noisy input / timesteps);
  - the preprocess emit appends the demoted key in-place, preserves every
    native key, is idempotent, and the dataset-side loader reads it back.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from library.datasets.buckets import (
    SIGMA_DEMOTE_ROUTE,
    demote_bucket_for,
    demoted_token_counts,
    freefit_band_for_edge,
)
from library.io.cache_names import demoted_latents_key
from library.runtime.noise import draw_flat_sigmas, get_noisy_model_input_and_timesteps


def _args(**kw):
    base = dict(
        timestep_sampling="sigmoid",
        sigmoid_scale=1.0,
        discrete_flow_shift=3.0,
        ip_noise_gamma=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


_SCHED = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))


class TestDemoteBucket:
    def test_route_shapes_land_in_demote_band(self):
        lo, hi = freefit_band_for_edge(896)
        # The frozen top-5 1024-tier shapes (all 4032/4200-token).
        for w, h in [(896, 1200), (800, 1344), (1200, 896), (768, 1344), (896, 1152)]:
            bucket = demote_bucket_for(w, h, *SIGMA_DEMOTE_ROUTE)
            assert bucket is not None
            bw, bh = bucket
            tok = (bw // 16) * (bh // 16)
            assert lo <= tok <= hi
            # Aspect preserved to free-fit tolerance (sub-patch residual).
            assert abs(bw / bh - w / h) < 0.1

    def test_off_route_returns_none(self):
        # A native-896 shape (3024 tokens): trains as-is, no demote.
        assert demote_bucket_for(1008, 768, *SIGMA_DEMOTE_ROUTE) is None
        # 768-tier and 1280-tier shapes are off this route too.
        assert demote_bucket_for(768, 720, *SIGMA_DEMOTE_ROUTE) is None
        assert demote_bucket_for(1280, 1260, *SIGMA_DEMOTE_ROUTE) is None

    def test_deterministic(self):
        a = demote_bucket_for(896, 1200, *SIGMA_DEMOTE_ROUTE)
        b = demote_bucket_for(896, 1200, *SIGMA_DEMOTE_ROUTE)
        assert a == b

    def test_demoted_token_counts_only_route_members(self):
        resos = {(896, 1200), (1008, 768)}  # one 1024-tier, one 896-tier
        counts = demoted_token_counts(resos, *SIGMA_DEMOTE_ROUTE)
        lo, hi = freefit_band_for_edge(896)
        assert counts  # the 1024-tier member contributes
        assert all(lo <= c <= hi for c in counts)


class TestDemotedKey:
    def test_never_in_latents_namespace(self):
        key = demoted_latents_key(880, 1184)
        assert not key.startswith("latents")
        assert key == "demoted_148x110"  # H//8 x W//8, native key convention


class TestSigmaDraw:
    @pytest.mark.parametrize("mode", ["sigmoid", "uniform", "shift"])
    def test_flat_draw_matches_inline_formula(self, mode):
        args = _args(timestep_sampling=mode)
        torch.manual_seed(7)
        got = draw_flat_sigmas(args, 4, 148, 110, torch.device("cpu"))
        torch.manual_seed(7)
        if mode == "sigmoid":
            want = torch.sigmoid(1.0 * torch.randn((4,)) + 0.0)
        elif mode == "uniform":
            want = torch.rand((4,))
        else:
            s = torch.sigmoid(torch.randn(4) * 1.0 + 0.0)
            want = (s * 3.0) / (1 + (3.0 - 1) * s)
        assert torch.equal(got, want)

    def test_density_modes_return_none(self):
        args = _args(timestep_sampling="something_else")
        assert draw_flat_sigmas(args, 4, 148, 110, torch.device("cpu")) is None

    def test_sigma_first_path_is_bit_exact(self):
        """draw σ → pass in ≡ draw-inside, given the same seed (RNG order is
        preserved because the split-out helper is the body's first RNG use)."""
        args = _args()
        latents = torch.randn(2, 16, 148, 110)
        noise = torch.randn_like(latents)

        torch.manual_seed(11)
        noisy_a, t_a, sig_a = get_noisy_model_input_and_timesteps(
            args, _SCHED, latents, noise, torch.device("cpu"), torch.float32
        )
        torch.manual_seed(11)
        pre = draw_flat_sigmas(args, 2, 148, 110, torch.device("cpu"))
        noisy_b, t_b, sig_b = get_noisy_model_input_and_timesteps(
            args,
            _SCHED,
            latents,
            noise,
            torch.device("cpu"),
            torch.float32,
            sigmas=pre,
        )
        assert torch.equal(noisy_a, noisy_b)
        assert torch.equal(t_a, t_b)
        assert torch.equal(sig_a, sig_b)


class TestEmitAndLoad:
    @pytest.fixture()
    def corpus(self, tmp_path: Path):
        """One 1024-tier resized PNG + its native npz (with extra keys)."""
        from PIL import Image

        w, h = 896, 1200  # 4200 tokens — 1024 tier
        img_dir = tmp_path / "resized" / "artist"
        img_dir.mkdir(parents=True)
        Image.new("RGB", (w, h), (128, 64, 32)).save(img_dir / "img1.png")

        cache_dir = tmp_path / "lora"
        npz_dir = cache_dir / "artist"
        npz_dir.mkdir(parents=True)
        npz_path = npz_dir / f"img1_{w:04d}x{h:04d}_anima.npz"
        native = {
            f"latents_{h // 8}x{w // 8}": np.zeros((16, h // 8, w // 8), np.float32),
            f"original_size_{h // 8}x{w // 8}": np.array([w, h]),
            f"crop_ltrb_{h // 8}x{w // 8}": np.array([0, 0, w, h]),
        }
        np.savez(npz_path, **native)
        return SimpleNamespace(
            data_dir=tmp_path / "resized",
            cache_dir=cache_dir,
            npz_path=npz_path,
            native_keys=set(native),
            wh=(w, h),
        )

    @pytest.fixture()
    def stub_vae(self):
        class _V:
            device = torch.device("cpu")
            dtype = torch.float32

            def encode_pixels_to_latents(self, px):
                return torch.ones(px.shape[0], 16, px.shape[-2] // 8, px.shape[-1] // 8)

        return _V()

    def test_emit_appends_preserves_and_idempotent(self, corpus, stub_vae):
        from library.preprocess.latents import (
            cache_demoted_latents,
            count_pending_demoted,
        )

        pending, eligible = count_pending_demoted(
            corpus.data_dir,
            native_edge=SIGMA_DEMOTE_ROUTE[0],
            demote_edge=SIGMA_DEMOTE_ROUTE[1],
            cache_dir=corpus.cache_dir,
            recursive=True,
        )
        assert (pending, eligible) == (1, 1)

        stats = cache_demoted_latents(
            corpus.data_dir,
            stub_vae,
            native_edge=SIGMA_DEMOTE_ROUTE[0],
            demote_edge=SIGMA_DEMOTE_ROUTE[1],
            cache_dir=corpus.cache_dir,
            recursive=True,
        )
        assert stats.written == 1 and stats.failed == 0

        bucket = demote_bucket_for(*corpus.wh, *SIGMA_DEMOTE_ROUTE)
        key = demoted_latents_key(*bucket)
        with np.load(corpus.npz_path) as npz:
            assert set(npz.files) == corpus.native_keys | {key}
            assert npz[key].shape == (16, bucket[1] // 8, bucket[0] // 8)

        # Idempotent: second pass skips.
        stats2 = cache_demoted_latents(
            corpus.data_dir,
            stub_vae,
            native_edge=SIGMA_DEMOTE_ROUTE[0],
            demote_edge=SIGMA_DEMOTE_ROUTE[1],
            cache_dir=corpus.cache_dir,
            recursive=True,
        )
        assert stats2.written == 0 and stats2.skipped == 1

    def test_dataset_loader_roundtrip(self, corpus, stub_vae):
        from library.datasets.base import BaseDataset
        from library.preprocess.latents import cache_demoted_latents

        cache_demoted_latents(
            corpus.data_dir,
            stub_vae,
            native_edge=SIGMA_DEMOTE_ROUTE[0],
            demote_edge=SIGMA_DEMOTE_ROUTE[1],
            cache_dir=corpus.cache_dir,
            recursive=True,
        )
        ds = BaseDataset(network_multiplier=1.0, debug_dataset=False)
        info = SimpleNamespace(latents_npz=str(corpus.npz_path), bucket_reso=corpus.wh)

        # Disabled → None (sidecar inert).
        assert ds._try_load_demoted_latent(info) is None

        ds.enable_sigma_demote(*SIGMA_DEMOTE_ROUTE)
        lat = ds._try_load_demoted_latent(info)
        bucket = demote_bucket_for(*corpus.wh, *SIGMA_DEMOTE_ROUTE)
        assert lat is not None and lat.dtype == torch.float32
        assert lat.shape == (16, bucket[1] // 8, bucket[0] // 8)

        # Off-route image → None even when enabled.
        off = SimpleNamespace(latents_npz=str(corpus.npz_path), bucket_reso=(1008, 768))
        assert ds._try_load_demoted_latent(off) is None


class TestPairedStepRng:
    """--paired_step_rng (CRN): σ/noise decoupled from the global stream so
    A/B arms with the same seed stay noise-locked."""

    def test_generator_draw_ignores_global_stream(self):
        args = _args()
        g1 = torch.Generator().manual_seed(123)
        torch.manual_seed(0)
        a = draw_flat_sigmas(args, 4, 148, 110, torch.device("cpu"), generator=g1)
        g2 = torch.Generator().manual_seed(123)
        torch.manual_seed(999)  # different global state must not matter
        torch.randn(1000)  # ...nor global consumption
        b = draw_flat_sigmas(args, 4, 148, 110, torch.device("cpu"), generator=g2)
        assert torch.equal(a, b)

    def test_two_arms_share_sigma_and_noise_sequences(self):
        """Simulate two arms: same (seed, counter) derivation → identical σ
        per step and identical native-shape noise, regardless of what each
        arm did to the global stream in between."""
        args = _args()

        def arm_step(counter, global_junk):
            torch.randn(global_junk)  # arm-specific global-stream consumption
            base = (42 * 1_000_003 + counter) * 2
            mask = (1 << 62) - 1
            g_s = torch.Generator().manual_seed(base & mask)
            g_n = torch.Generator().manual_seed((base + 1) & mask)
            sig = draw_flat_sigmas(
                args, 1, 148, 110, torch.device("cpu"), generator=g_s
            )
            noise = torch.randn((1, 16, 152, 108), generator=g_n)
            return sig, noise

        for step in (1, 2, 3):
            s_a, n_a = arm_step(step, global_junk=7)
            s_b, n_b = arm_step(step, global_junk=3001)
            assert torch.equal(s_a, s_b)
            assert torch.equal(n_a, n_b)
