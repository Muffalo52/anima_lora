import math
import random
from typing import NamedTuple, Tuple

# NB: numpy is imported lazily inside BucketManager so the free-fit helpers
# below (used by the GUI via library.preprocess.resize_preview) stay numpy-free.

# Per-tier token-count bands — the free-fit search range for each tier edge.
# Free-fit (the only resize mode; see freefit_bucket / freefit_band_for_edge,
# _archive/proposals/free_aspect_token_band_resize.md) preserves native aspect ratio
# and lands the patch-grid token count anywhere inside its tier's band, so every
# forward runs at its true token count with zero padding under compile_dynamic_seq
# (one block graph per band). Token count = (W//16)*(H//16), capped at 256/axis
# (rope). ``--target_res`` selects active tiers; ``choose_edge`` picks the one
# that resizes an image least.
EDGE_TOKEN_BANDS: dict = {
    512: (1008, 1024),
    768: (2160, 2160),
    896: (3000, 3024),
    1024: (4032, 4200),
    1280: (6300, 6300),
    1536: (8640, 8640),
}
ALLOWED_TARGET_RES = tuple(sorted(EDGE_TOKEN_BANDS))
DEFAULT_TARGET_RES = (1024,)


def _band(edge: int) -> tuple[int, int]:
    """The (lo, hi) token-count band for a tier edge, or a clear error."""
    try:
        return EDGE_TOKEN_BANDS[edge]
    except KeyError:
        raise ValueError(
            f"target_res {edge} not in allowed tiers {ALLOWED_TARGET_RES}"
        ) from None


def token_count_families(target_res) -> int:
    """Number of distinct band-endpoint token counts across the active tiers
    (deduped) — drives the dynamo cache-size budget in ``compile_blocks``."""
    counts: set = set()
    for edge in target_res:
        lo, hi = _band(edge)
        counts.add(lo)
        counts.add(hi)
    return len(counts)


def token_count_range(target_res) -> tuple[int, int]:
    """(min, max) token count across the active tiers — bounds the
    ``mark_dynamic`` seq-length hint in ``compile_blocks``."""
    los = [_band(edge)[0] for edge in target_res]
    his = [_band(edge)[1] for edge in target_res]
    if not los:
        raise ValueError("token_count_range requires at least one tier")
    return min(los), max(his)


def token_counts_for_resos(resos) -> set:
    """Distinct token counts ``(W//16)*(H//16)`` over a set of (W, H) resolutions."""
    return {(w // 16) * (h // 16) for w, h in resos}


def snap_sample_size(width: int, height: int) -> Tuple[int, int]:
    """Snap a requested sample (W, H) to the DiT's 16px pixel grid — the same
    snap ``_sample_image_inference`` applies, shared so the compile token
    budget agrees with the seq len a sample prompt actually runs at."""
    return max(64, width - width % 16), max(64, height - height % 16)


def token_counts_for_sample_prompts(prompts) -> set:
    """Distinct DiT token counts the training sample prompts will request
    (``prompts`` = ``train_util.load_prompts`` dicts, width/height default 512).
    Folded into the compile token budget so a sample resolution outside the
    training buckets widens the compiled range instead of raising a
    dynamic-seq ConstraintViolationError mid-training (issue #42)."""
    counts: set = set()
    for prompt_dict in prompts:
        w, h = snap_sample_size(
            int(prompt_dict.get("width", 512)),
            int(prompt_dict.get("height", 512)),
        )
        counts.add((w // 16) * (h // 16))
    return counts

def edge_for_token_count(n_tokens: int, edges=ALLOWED_TARGET_RES) -> int:
    """Map a patch-token count back to the tier edge it belongs to.

    The inverse of the preprocess tier assignment: given a cached latent's token
    count ``(W//16)*(H//16)``, return which tier (512…1536) emitted it. Used by
    the resolution-curriculum schedule (``autoscale_mode``) to rank a dataset's
    populated buckets into a low→high ladder. A count inside a tier's free-fit
    band maps to that tier; anything off-band (snap-era caches) falls back to the
    nearest tier by ``|log(nominal / n)|`` (same scale-symmetric metric as
    ``choose_edge``), so legacy data still ranks sensibly.
    """
    for edge in edges:
        lo, hi = freefit_band_for_edge(edge)
        if lo <= n_tokens <= hi:
            return edge
    n = max(int(n_tokens), 1)
    return min(
        edges,
        key=lambda e: abs(math.log(((_band(e)[0] + _band(e)[1]) / 2.0) / n)),
    )

# The single measured-safe σ-demote route (1024→896). Other candidate routes
# (896→768, 1280→1024) failed/differ per their own gradient probe — do not add
# routes without one (project/sigma_lowres/bench/run_sigma_probe.py).
SIGMA_DEMOTE_ROUTE: Tuple[int, int] = (1024, 896)


def demote_bucket_for(
    width: int,
    height: int,
    native_edge: int,
    demote_edge: int,
    max_ratio: "float | None" = None,
) -> "tuple[int, int] | None":
    """σ-demote sibling grid ``(W', H')`` for a native free-fit bucket, or None
    if ``(width, height)`` isn't native to ``native_edge``'s token band. Pure
    and deterministic — preprocess emit and trainer fetch both call this so
    they derive the identical demoted grid.
    """
    if max_ratio is None:
        max_ratio = DEFAULT_FREEFIT_MAX_RATIO
    lo, hi = freefit_band_for_edge(native_edge)
    tok = (width // 16) * (height // 16)
    if not (lo <= tok <= hi):
        return None
    return freefit_bucket(width, height, freefit_band_for_edge(demote_edge), max_ratio)


def demoted_token_counts(resos, native_edge: int, demote_edge: int) -> set:
    """Distinct token counts the σ-demote siblings of ``resos`` will run at.
    Unioned into the compile token budget (``train.py::_derive_token_budget``)
    when sigma_lowres is on, so demoted forwards stay inside the compiled range.
    """
    counts: set = set()
    for w, h in resos:
        bucket = demote_bucket_for(w, h, native_edge, demote_edge)
        if bucket is not None:
            counts.add((bucket[0] // 16) * (bucket[1] // 16))
    return counts


def choose_edge(width: int, height: int, target_res) -> int:
    """Assign an image to the tier that resizes it the *least*.

    Minimizes ``|log(nominal_tokens / native_tokens)|`` (nominal = each tier's
    band midpoint) — scale-symmetric, so a 0.95MP image stays at 1024 (small
    upscale) rather than being downscaled to 768. Single-element ``target_res``
    is a no-op.
    """
    if len(target_res) == 1:
        return target_res[0]
    native_tokens = (width / 16.0) * (height / 16.0)
    best_edge: int | None = None
    best_cost = float("inf")
    for edge in target_res:
        lo, hi = _band(edge)
        nominal = (lo + hi) / 2.0
        cost = abs(math.log(nominal / native_tokens))
        if cost < best_cost:
            best_cost, best_edge = cost, edge
    return best_edge


# Free-fit ("free-aspect token-band") solver — see
# _archive/proposals/free_aspect_token_band_resize.md. Pure, deterministic, no I/O.

DEFAULT_FREEFIT_MAX_RATIO = 4.0

# Single-family tiers (lo == hi, e.g. 768/1280/1536) and near-degenerate 512
# leave the solver no aspect freedom without widening — without this a tier's
# one exact token count forces a coarse divisor grid and crops like the old
# snap (#53 Phase 0: 0.866-aspect image on 768 tier → 0.9375, ~7.7% crop).
# Widening is free at compile time (whole [lo, hi] stays one compile_dynamic_seq
# graph).
FREEFIT_BAND_TOLERANCE = 0.025  # +/-2.5% -> ~5% interval around the tier's nominal

# 1024 stays frozen at its natural (4032, 4200): DCW_ASPECT_BUCKETS (CNS
# calibration + mod-distill) is drawn from this tier's band.
FREEFIT_FROZEN_EDGES: Tuple[int, ...] = (1024,)

# Bumped whenever the band derivation changes, so free-fit resized PNGs cached
# under an older band re-resize (folded into the resize metadata signature).
FREEFIT_BAND_VERSION = 2


def freefit_band_for_edge(
    edge: int, tol: float = FREEFIT_BAND_TOLERANCE
) -> tuple[int, int]:
    """Token-count band ``(lo, hi)`` for a single tier — the free-fit search range.

    Widens the tier's natural band (``EDGE_TOKEN_BANDS``) symmetrically by ``tol``
    (a wider band gives the solver more aspect freedom, less crop), except for
    ``FREEFIT_FROZEN_EDGES`` (1024, kept at its natural (4032, 4200)).
    """
    lo, hi = token_count_range((edge,))
    if edge in FREEFIT_FROZEN_EDGES:
        return lo, hi
    return round(lo * (1.0 - tol)), round(hi * (1.0 + tol))


def freefit_bucket(
    width: int,
    height: int,
    band: tuple[int, int],
    max_ratio: float = DEFAULT_FREEFIT_MAX_RATIO,
    patch: int = 16,
    rope_cap: int = 256,
) -> tuple[int, int]:
    """Native-aspect resize target whose patch grid fills the token ``band``.

    Returns pixel ``(W, H)`` (multiples of ``patch``) whose patch grid
    ``(W//patch)*(H//patch)`` lies in ``[lo, hi]`` and whose aspect is as close
    as possible to the image's — clamped to ``[1/max_ratio, max_ratio]`` —
    subject to ``max(W//patch, H//patch) <= rope_cap``. Crop is zero unless the
    ratio clamp fired. Exhaustive search over the (narrow, rope-capped) band, so
    the result is the global aspect-error minimum, tie-broken toward least resize.
    """
    lo, hi = int(band[0]), int(band[1])
    if lo <= 0 or hi < lo:
        raise ValueError(f"invalid free-fit band {band}")
    a = width / height
    a_clamped = min(max(a, 1.0 / max_ratio), float(max_ratio))

    best: tuple | None = None
    hp_max = min(rope_cap, hi)
    for hp in range(1, hp_max + 1):
        wp_lo = max(1, -(-lo // hp))  # ceil(lo / hp)
        wp_hi = min(rope_cap, hi // hp)  # floor(hi / hp)
        for wp in range(wp_lo, wp_hi + 1):
            aspect_err = abs(wp / hp - a_clamped)
            cover_scale = max(wp * patch / width, hp * patch / height)
            # (aspect first, then least rescale, then a deterministic shape key).
            key = (aspect_err, abs(math.log(cover_scale)), hp, wp)
            if best is None or key < best:
                best = key
    if best is None:
        raise ValueError(
            f"free-fit band {band} admits no grid under rope_cap={rope_cap}"
        )
    _, _, hp, wp = best
    return wp * patch, hp * patch


# Frozen literal: dataset's top-5 (H, W) resolutions by frequency from the old
# discrete 1024-tier bucket pool (pre-free-fit). Kept for CNS calibration and
# mod-guidance distillation which still key off it (DCW name only because those
# consumers import the symbol; the DCW line itself is retired, _archive/dcw/).
DCW_ASPECT_BUCKETS: Tuple[Tuple[int, int], ...] = (
    (1200, 896),  # 896x1200 portrait, most common, 4200-tok
    (1344, 800),  # 800x1344 tall portrait, 4200-tok
    (896, 1200),  # 1200x896 landscape, 4200-tok
    (1344, 768),  # 768x1344 tall portrait, 4032-tok
    (1152, 896),  # 896x1152 portrait, 4032-tok
)
DCW_ASPECT_NAMES: Tuple[str, ...] = tuple(f"{h}x{w}" for h, w in DCW_ASPECT_BUCKETS)
DCW_ASPECT_TABLE: dict = {hw: i for i, hw in enumerate(DCW_ASPECT_BUCKETS)}
N_DCW_ASPECTS: int = len(DCW_ASPECT_BUCKETS)


def make_bucket_resolutions(max_reso, min_size=256, max_size=1024, divisible=64):
    """Generate bucket resolutions for multi-aspect-ratio training.
    Moved from model_util.py to avoid dependency."""
    max_width, max_height = max_reso
    max_area = max_width * max_height

    resos = set()

    width = int(math.sqrt(max_area) // divisible) * divisible
    resos.add((width, width))

    width = min_size
    while width <= max_size:
        height = min(max_size, int((max_area // width) // divisible) * divisible)
        if height >= min_size:
            resos.add((width, height))
            resos.add((height, width))

        width += divisible

    resos = list(resos)
    resos.sort()
    return resos


class BucketManager:
    def __init__(
        self, max_reso=None, min_size=None, max_size=None, reso_steps=None
    ) -> None:
        if max_size is not None:
            if max_reso is not None:
                assert max_size >= max_reso[0], (
                    "the max_size should be larger than the width of max_reso"
                )
                assert max_size >= max_reso[1], (
                    "the max_size should be larger than the height of max_reso"
                )
            if min_size is not None:
                assert max_size >= min_size, (
                    "the max_size should be larger than the min_size"
                )

        if max_reso is None:
            self.max_reso = None
            self.max_area = None
        else:
            self.max_reso = max_reso
            self.max_area = max_reso[0] * max_reso[1]
        self.min_size = min_size
        self.max_size = max_size
        self.reso_steps = reso_steps

        self.resos = []
        self.reso_to_id = {}
        self.buckets = []

    def add_image(self, reso, image_or_info):
        bucket_id = self.reso_to_id[reso]
        self.buckets[bucket_id].append(image_or_info)

    def shuffle(self):
        for bucket in self.buckets:
            random.shuffle(bucket)

    def sort(self):
        sorted_resos = self.resos.copy()
        sorted_resos.sort()

        sorted_buckets = []
        sorted_reso_to_id = {}
        for i, reso in enumerate(sorted_resos):
            bucket_id = self.reso_to_id[reso]
            sorted_buckets.append(self.buckets[bucket_id])
            sorted_reso_to_id[reso] = i

        self.resos = sorted_resos
        self.buckets = sorted_buckets
        self.reso_to_id = sorted_reso_to_id

    def make_buckets(
        self,
        freefit_resos=None,
        target_res=None,
    ):
        if freefit_resos is not None:
            # Free-fit: the on-disk cached (W, H) set IS the source of truth for
            # buckets — every cached latent exact-matches in select_bucket, nothing
            # AR-snaps at load. target_res is preprocess-only and inert here.
            resos = sorted(set(tuple(r) for r in freefit_resos))
        else:
            resos = make_bucket_resolutions(
                self.max_reso, self.min_size, self.max_size, self.reso_steps
            )
        self.set_predefined_resos(resos)

    def set_predefined_resos(self, resos):
        import numpy as np

        self.predefined_resos = resos.copy()
        self.predefined_resos_set = set(resos)
        self.predefined_aspect_ratios = np.array([w / h for w, h in resos])

    def add_if_new_reso(self, reso):
        if reso not in self.reso_to_id:
            bucket_id = len(self.resos)
            self.reso_to_id[reso] = bucket_id
            self.resos.append(reso)
            self.buckets.append([])

    def select_bucket(self, image_width, image_height):
        aspect_ratio = image_width / image_height
        reso = (image_width, image_height)
        if reso in self.predefined_resos_set:
            pass
        else:
            import numpy as np

            ar_errors = self.predefined_aspect_ratios - aspect_ratio
            predefined_bucket_id = np.abs(ar_errors).argmin()
            reso = self.predefined_resos[predefined_bucket_id]

        ar_reso = reso[0] / reso[1]
        if aspect_ratio > ar_reso:
            scale = reso[1] / image_height
        else:
            scale = reso[0] / image_width

        resized_size = (
            int(image_width * scale + 0.5),
            int(image_height * scale + 0.5),
        )

        self.add_if_new_reso(reso)

        ar_error = (reso[0] / reso[1]) - aspect_ratio
        return reso, resized_size, ar_error

    @staticmethod
    def get_crop_ltrb(bucket_reso: Tuple[int, int], image_size: Tuple[int, int]):
        # Calculate crop left/top according to the preprocessing of Stability AI. Crop right is calculated for flip augmentation.

        bucket_ar = bucket_reso[0] / bucket_reso[1]
        image_ar = image_size[0] / image_size[1]
        if bucket_ar > image_ar:
            resized_width = bucket_reso[1] * image_ar
            resized_height = bucket_reso[1]
        else:
            resized_width = bucket_reso[0]
            resized_height = bucket_reso[0] / image_ar
        crop_left = (bucket_reso[0] - resized_width) // 2
        crop_top = (bucket_reso[1] - resized_height) // 2
        crop_right = crop_left + resized_width
        crop_bottom = crop_top + resized_height
        return crop_left, crop_top, crop_right, crop_bottom


class BucketBatchIndex(NamedTuple):
    bucket_index: int
    bucket_batch_size: int
    batch_index: int
