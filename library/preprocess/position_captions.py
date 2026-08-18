"""Position-aware caption rewrite (v2) — detect subjects, bind tags to sides.

Orchestration for ``make caption-position``: detect the ``girl`` instances in a
multi-subject image, order them, tag each mask-blanked crop, and rewrite the
caption in the dataset's hand-written convention::

    <flat tag bag>. On the left, akita neru, yellow eyes. On the right, ...

v2 *moves* an attributable tag out of the flat bag into its clause (each
attribute asserted exactly once) rather than the additive v1 (``rewrite=False``).
Reversible via :func:`library.captioning.position_clauses.flatten_caption`;
lands only on the derived caption (``post_image_dataset/resized/<rel>.txt``),
never the hand-written master under ``image_dataset/``.

Takes its two models as injected callables (``detect_fn``/``tag_fn``), staying
import-free of SAM3/the tagger; ``scripts/preprocess/position_captions.py``
owns argparse + model loading.

Per-rule evidence and the knob table live in
``docs/experimental/position_captions.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from library.captioning.position_clauses import (
    PositionClause,
    assign_positions,
    compose_caption,
    flatten_caption,
    has_clauses,
    ordered_indices,
    parse_caption,
)
from library.captioning.taxonomy import is_artist_tag, is_count_tag, is_rating_tag

# ---------------------------------------------------------------------------
# Clause vocabulary
# ---------------------------------------------------------------------------

# Tag groups that describe *one subject* and therefore bind to a position;
# everything else (lighting, background, framing, medium, interaction, …)
# stays in the flat bag. Drawn from the tagger's own ``groups.yaml`` rather
# than substring heuristics so the two can't drift.
SUBJECT_GROUPS = frozenset(
    {
        # identity / face
        "eye_color",
        "eye_shape",
        "face_features",
        "expression",
        "emotion",
        "age",
        "gender",
        "skin",
        "body_shape",
        "body_parts",
        "species_nonhuman",
        "animal_parts",
        # hair
        "hair_color",
        "hair_length",
        "hairstyle",
        "hair_accessory",
        # clothing
        "top_garment",
        "top_clothing",
        "bottom_clothing",
        "dress_onepiece",
        "underwear",
        "swimwear",
        "school_uniform",
        "traditional_clothing",
        "costume",
        "clothing_details",
        "fashion_style",
        "footwear",
        "socks_stockings",
        "gloves_acc",
        "earrings_necklace",
        "hat",
        "eyewear",
        "bag",
        # what the subject is doing
        "pose",
        "gesture",
        "daily_action",
    }
)

# Emitted first, in this order, when present — the attributes that actually
# disambiguate a subject. Everything else follows, ranked.
_PRIORITY_GROUPS = ("hair_color", "eye_color", "hair_length", "hairstyle")

# The groups you read off a **face**. A crop without a head in it has no
# evidence for any of them, so they are suppressed on a body-part crop
# (``allow_identity=False``).
_IDENTITY_GROUPS = frozenset({"hair_color", "eye_color", "hair_length", "hairstyle"})

# Groups where the flat bag outranks the crop tagger: once the caption names a
# hair color, a clause may only pick from the colors it named, else a crop
# hallucinating an uncontradicted value would ride discriminative-only straight
# into a clause (measured: 520/1600 identity clause tags contradicted the bag
# in the first full-corpus dry run). The hand-picked three plus, in
# `gated_groups()`, every EXCLUSIVE subject group the vocabulary declares —
# an exclusive group holds one value by construction, so a second is always a
# contradiction, not extra detail (103 `body_shape` + 39 `fashion_style` +
# others caught in the 2026-08-17 dry run). `hair_length` is not itself
# exclusive in groups.yaml, hence it stays hand-listed. Multi-value groups
# (`hairstyle`) stay ungated on purpose: a second value there is additive
# detail, not a contradiction.
_BAG_GATED_GROUPS = frozenset({"hair_color", "eye_color", "hair_length"})

# Groups whose value belongs to a **character**, not to a view of one — on a
# `1girl, multiple views` sheet, binding `aqua hair` to one view and removing it
# from the bag would claim the other views aren't aqua-haired. Outfit / pose /
# expression / framing carry no such implication, so those move freely.
#
# Corroboration rule: a tag in one of these groups may leave the bag only when
# the bag names >=2 distinct values of that group (evidence-based rather than
# count-based, since most proposals carry no reliable girls-count tag).
_CHARACTER_INVARIANT_GROUPS = frozenset(
    {
        "hair_color",
        "hair_length",
        "hairstyle",
        "eye_color",
        "eye_shape",
        "face_features",
        "age",
        "gender",
        "skin",
        "body_shape",
        "species_nonhuman",
        "animal_parts",
    }
)

# On a repeated-subject layout (`_LAYOUT_TAGS`) the subjects are one character
# drawn several times, so no clause may carry a trait the character owns — a
# stricter rule than the corroboration gate above (that one governs whether a
# tag may LEAVE the bag; this one blocks it from ENTERING a clause at all).
# `body_parts` joins the character-invariant set for this rule only: anatomy is
# owned by the character, but its *visibility* genuinely varies by view, so it
# stays bindable on a real multi-character image. Measured 45% of `multiple
# views` clause tags were view-invariant pre-gate (33% for comic panels); see
# docs/experimental/position_captions.md for the breakdown.
_VIEW_INVARIANT_GROUPS = _CHARACTER_INVARIANT_GROUPS | {"body_parts"}

# Exception to the corroboration rule: booru tags a single two-toned-hair
# character with two hair-color tags, so ">=2 values" doesn't imply two
# subjects. Ungrouped in groups.yaml, hence a plain name set rather than group
# membership — when one is in the bag, that group is pinned flat.
_MULTI_VALUE_MARKERS: Mapping[str, frozenset[str]] = {
    "hair_color": frozenset(
        {
            "multicolored hair",
            "two-tone hair",
            "gradient hair",
            "streaked hair",
            "colored inner hair",
            "split-color hair",
            "rainbow hair",
            "colored tips",
            "multicolored bangs",
            "alternate hair color",
        }
    ),
    "eye_color": frozenset(
        {
            "heterochromia",
            "multicolored eyes",
            "gradient eyes",
            "two-tone eyes",
        }
    ),
}

_GIRLS_COUNT_RE = re.compile(r"^(\d+)girls?$")
_BOYS_COUNT_RE = re.compile(r"^(\d+)boys?$")
# ``6+girls`` is an open-ended crowd tag, not the number six — matching it
# exactly is impossible, so it is treated like ``multiple girls``: count
# unknown, trust detection.
_OPEN_GIRLS_RE = re.compile(r"^\d+\+girls?$")
_OPEN_BOYS_RE = re.compile(r"^\d+\+boys?$")
# ``2koma`` / ``4koma`` name the panel count. Deliberately anchored, so the
# open-ended ``multiple 4koma`` does not match and stays unbounded.
_KOMA_COUNT_RE = re.compile(r"^(\d+)koma$")
_MULTI_VIEW_TAGS = frozenset({"multiple views", "multiple_views"})

# Panel layouts: a comic page draws the same character once per panel, so like
# `multiple views` its girls-count counts *characters*, not bindable subjects —
# `1girl, 2koma` is routinely two. Without this, comic pages fail the candidate
# prefilter as `single-subject`.
#
# `page number` is deliberately EXCLUDED: it marks a scanned art-book page, not
# a layout (checked — every image it catches is a single illustration with a
# margin number), so it's a false signal, not a weak one.
_PANEL_LAYOUT_TAGS = frozenset(
    {
        "comic",
        "silent comic",
        "silent_comic",
        "sequential",
        "2koma",
        "3koma",
        "4koma",
        "multiple 4koma",
        "multiple_4koma",
    }
)

# Every layout tag that decouples the girls-count from the bindable-subject
# count. Both branches of the prefilter and the count check read this.
_LAYOUT_TAGS = _MULTI_VIEW_TAGS | _PANEL_LAYOUT_TAGS


@dataclass(frozen=True)
class ClauseVocabulary:
    """Which tags may enter a clause, and in what order.

    ``tag_to_group`` from the tagger checkpoint's ``groups.yaml``;
    ``characters``/``excluded`` from its ``vocab.json``. Ungrouped tags are
    admitted only via the attributable + in-the-caption path (see
    :meth:`select`) — lets a curated compound like ``pink jacket`` bind while
    keeping ungrouped scene tags out. ``excluded`` = image-level categories
    (copyright/artist/metadata/deprecated) that would otherwise ride the ranked
    path in. ``exclusive_groups`` = softmax groups where only one member may
    enter a clause.
    """

    characters: frozenset[str] = frozenset()
    excluded: frozenset[str] = frozenset()
    exclusive_groups: frozenset[str] = frozenset()
    tag_to_group: Mapping[str, str] = field(default_factory=dict)

    def group_of(self, tag: str) -> str | None:
        return self.tag_to_group.get(tag)

    def gated_groups(self) -> frozenset[str]:
        """Groups where a clause may only pick a value the flat bag already named.

        The hand-picked ``_BAG_GATED_GROUPS`` plus every **exclusive** subject
        group the checkpoint declares. Derived rather than listed so the gate
        cannot drift from ``groups.yaml``: whatever the tagger models as a
        softmax over one subject is, by construction, a group where a second
        value is a contradiction rather than extra detail.
        """
        return _BAG_GATED_GROUPS | (self.exclusive_groups & SUBJECT_GROUPS)

    def is_subject_tag(self, tag: str) -> bool:
        return self.group_of(tag) in SUBJECT_GROUPS

    def is_scene_tag(self, tag: str) -> bool:
        """Grouped, but into a group that describes the scene, not a subject."""
        group = self.group_of(tag)
        return group is not None and group not in SUBJECT_GROUPS

    def select(
        self,
        kept: Mapping[str, float],
        groups: Mapping[str, str | None],
        *,
        flat_bag: frozenset[str],
        attributable: frozenset[str],
        shared: frozenset[str],
        max_tags: int,
        name_confidence: float,
        allow_unlisted_names: bool,
        discriminative_only: bool = True,
        allow_identity: bool = True,
        bag_gated_identity: bool = True,
        view_invariant: bool = False,
        max_novel_tags: int = 1,
    ) -> list[str]:
        """Clause tags for one crop, ordered most-disambiguating first.

        ``kept``/``groups`` = crop tagger output; ``flat_bag`` = curated caption
        (what's in the image; crop only decides *where*); ``attributable`` =
        tags only this crop kept; ``shared`` = tags every crop kept.

        Candidates are ranked once, admitted bag-first then up to
        ``max_novel_tags`` novel (caption never had) — a novel tag can never
        later be *moved* by :func:`plan_bag_removals`, so this bounds
        dead-weight invention without ever rescuing a crowded-out bag tag.

        Suppression knobs, weakest to strongest: ``discriminative_only``
        (default) drops ``shared`` tags — a `multiple views` sheet repeats the
        same character/hair on every view, crowding out the outfit that
        differs. ``allow_identity=False`` (body-part crops) drops
        hair/eye/hairstyle outright — no head, no evidence. ``bag_gated_identity``
        (default) makes the flat bag outrank the tagger for any
        :meth:`gated_groups` member, once the bag has spoken for it.
        ``view_invariant`` (repeated-subject layout) is strongest: drops the
        name and every ``_VIEW_INVARIANT_GROUPS`` trait, keeping only what a
        view/panel can differ in.
        """
        out: list[str] = []
        seen: set[str] = set()
        taken_groups: set[str] = set()
        blocked = shared if discriminative_only else frozenset()
        # Gated groups the caption has already spoken for — see the
        # ``bag_members`` test in ``add``.
        bag_members = (
            {
                group: {t for t in flat_bag if self.group_of(t) == group}
                for group in self.gated_groups()
            }
            if bag_gated_identity
            else {}
        )

        def add(tag: str) -> bool:
            if not tag or tag in seen or tag in blocked:
                return False
            # Checked here (not only on the ranked path below) because an
            # excluded tag can still be grouped, e.g. a deprecated alias filed
            # under hair_color — it must not ride the priority path in.
            if tag in self.excluded:
                return False
            group = self.group_of(tag)
            if group in self.exclusive_groups and group in taken_groups:
                return False  # one hair color / one eye color per subject
            if not allow_identity and group in _IDENTITY_GROUPS:
                return False  # no head in this crop — nothing to read it off
            if view_invariant and group in _VIEW_INVARIANT_GROUPS:
                return False  # same girl in every view/panel — the bag owns this
            if bag_members.get(group) and tag not in flat_bag:
                return False  # the caption named this attribute; it wins
            seen.add(tag)
            if group:
                taken_groups.add(group)
            out.append(tag)
            return True

        # ---- Rank the candidates (this is the *emitted* order) --------------
        candidates: list[str] = []

        # 1. Character name. A name the caption never claimed is a crop
        #    hallucination, so by default it must appear in the flat bag.
        #    Skipped entirely on a repeated-subject layout: every view is the
        #    same girl, so a bound name would claim the other views are someone
        #    else.
        names = (
            []
            if view_invariant
            else sorted(
                (
                    t
                    for t in kept
                    if t in self.characters and kept[t] >= name_confidence
                ),
                key=lambda t: -kept[t],
            )
        )
        for name in names:
            if allow_unlisted_names or name in flat_bag:
                candidates.append(name)
                break  # one identity per subject

        # 2. Exclusive-group winners (hair color, eye color, …). These are the
        #    softmax_when_solo groups, and a single-subject crop is exactly the
        #    condition under which they fire — the whole point of cropping.
        for group in _PRIORITY_GROUPS:
            members = sorted(
                (t for t in kept if self.group_of(t) == group),
                key=lambda t: -kept[t],
            )
            # A kept member the caption already named outranks the softmax
            # winner. Without this the gate and the winner fight each other: on
            # a gated group a novel winner is rejected by ``add`` and the group
            # then emits nothing, even though the crop also kept the very value
            # the bag listed. Reuse is the whole point, so it wins the slot.
            winner = next((t for t in members if t in flat_bag), None)
            if winner is None:
                # Group didn't fire (contaminated / multi-person crop): fall
                # back to the highest-probability kept member of that group.
                winner = groups.get(group) or (members[0] if members else None)
            if winner:
                candidates.append(winner)

        # 3. Everything else, ranking tags the caption already curated first.
        rest = [
            t
            for t in kept
            if t not in candidates
            and not is_count_tag(t)
            and not is_rating_tag(t)
            and not is_artist_tag(t)
            and t not in self.characters
            and t not in self.excluded
            and not self.is_scene_tag(t)
            and (self.is_subject_tag(t) or (t in flat_bag and t in attributable))
        ]
        rest.sort(key=lambda t: (t not in flat_bag, -kept[t]))
        candidates.extend(rest)

        # ---- Admit: the bag first, then a bounded number of novel tags ------
        rank = {tag: i for i, tag in enumerate(candidates)}
        novel_budget = max(0, max_novel_tags)
        for reuse_pass in (True, False):
            for tag in candidates:
                if len(out) >= max_tags:
                    break
                if (tag in flat_bag) is not reuse_pass:
                    continue
                if not reuse_pass and novel_budget <= 0:
                    break
                if add(tag) and not reuse_pass:
                    novel_budget -= 1
        out.sort(key=lambda t: rank[t])
        return out[:max_tags]


def load_clause_vocabulary(ckpt_dir: str | Path) -> ClauseVocabulary:
    """Build a :class:`ClauseVocabulary` from a tagger checkpoint directory."""
    import json

    from library.captioning import tag_groups as tg

    ckpt = Path(ckpt_dir)
    with open(ckpt / "vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    characters = frozenset(
        t["name"] for t in vocab["tags"] if t.get("category") == "character"
    )
    excluded = frozenset(
        t["name"]
        for t in vocab["tags"]
        if t.get("category") in {"copyright", "artist", "metadata", "deprecated"}
    )
    groups_path = ckpt / "groups.yaml"
    tag_to_group: Mapping[str, str] = {}
    exclusive: frozenset[str] = frozenset()
    if groups_path.exists():
        groups = tg.load_groups(groups_path)
        tag_to_group = dict(groups.tag_to_group)
        exclusive = frozenset(
            g.name for g in groups.groups if g.mode in {"softmax", "softmax_when_solo"}
        )
    return ClauseVocabulary(
        characters=characters,
        excluded=excluded,
        exclusive_groups=exclusive,
        tag_to_group=tag_to_group,
    )


# ---------------------------------------------------------------------------
# Candidate prefilter
# ---------------------------------------------------------------------------


def caption_subject_count(caption: str) -> int | None:
    """How many bindable subjects the caption itself claims, if it says.

    ``Ngirls`` gives a number; ``None`` means "more than one, count unknown"
    (the count-consistency check then trusts detection instead of skipping).

    A layout tag (``_LAYOUT_TAGS``) always forces ``None`` even alongside a
    girls-count, because that count tags *characters* while each view/panel is
    its own bindable subject (``1girl, multiple views`` is routinely four).
    ``multiple girls`` / open-ended ``N+girls`` are ``None`` too — an exact
    match against "six or more" can only fail.
    """
    tags = {t.strip().lower() for t in parse_caption(caption).flat_tags}
    if tags & _LAYOUT_TAGS:
        return None
    counts = [int(m.group(1)) for t in tags if (m := _GIRLS_COUNT_RE.match(t))]
    if not counts and (
        "multiple girls" in tags or any(map(_OPEN_GIRLS_RE.match, tags))
    ):
        return None
    return max(counts) if counts else 0


def caption_panel_ceiling(caption: str) -> int | None:
    """Most bindable subjects an ``Nkoma`` page can hold, or ``None`` if unbounded.

    A layout tag makes :func:`caption_subject_count` return ``None``, waiving
    the count check entirely — this restores a backstop for a comic page so a
    subject detected twice (e.g. by a shredded overlapping-mask split) still
    gets caught. ``Nkoma`` names the panel count, so the ceiling is
    ``panels x (girls + boys)`` (generous by construction: every panel drawing
    every character at once). Plain ``comic`` / ``multiple views`` carry no
    panel count and stay unbounded.

    ``None`` whenever any term is unknown — an unbounded check can only produce
    false skips.
    """
    tags = {t.strip().lower() for t in parse_caption(caption).flat_tags}
    panels = [int(m.group(1)) for t in tags if (m := _KOMA_COUNT_RE.match(t))]
    if not panels:
        return None
    girls = [int(m.group(1)) for t in tags if (m := _GIRLS_COUNT_RE.match(t))]
    if not girls and ("multiple girls" in tags or any(map(_OPEN_GIRLS_RE.match, tags))):
        return None
    boys = caption_boy_count(caption)
    if boys is None:
        return None
    # A page with no counted character at all still draws somebody per panel.
    per_panel = max(max(girls, default=0) + boys, 1)
    return max(panels) * per_panel


def caption_boy_count(caption: str) -> int | None:
    """How many *male* subjects the caption claims — the count check's slack.

    The SAM3 ``girl`` prompt does not reliably exclude males, so the count gate
    accepts the range ``girls .. girls + boys`` rather than equality. ``None`` =
    "some boys, count unknown", which drops the upper bound entirely.
    """
    tags = {t.strip().lower() for t in parse_caption(caption).flat_tags}
    counts = [int(m.group(1)) for t in tags if (m := _BOYS_COUNT_RE.match(t))]
    if not counts and ("multiple boys" in tags or any(map(_OPEN_BOYS_RE.match, tags))):
        return None
    return max(counts) if counts else 0


def is_repeated_subject_layout(caption: str) -> bool:
    """Is this one character drawn several times, rather than several characters?

    Any ``_LAYOUT_TAGS`` member says yes — an ``Nkoma``/``comic`` page is the
    same situation as ``multiple views`` panel-by-panel: the girl in panel 3 is
    the girl in panel 1, so whatever belongs to *her* discriminates nothing
    between panels. :meth:`ClauseVocabulary.select` drops the whole class
    (``view_invariant``). A comic can introduce a new character mid-page, which
    only makes a bound trait *sometimes* wrong instead of never — the bag keeps
    every suppressed tag regardless, so only the per-panel binding is lost.
    """
    tags = {t.strip().lower() for t in parse_caption(caption).flat_tags}
    return bool(tags & _LAYOUT_TAGS)


def is_candidate(caption: str) -> tuple[bool, str]:
    """Should this caption go through detection? Returns ``(ok, reason)``."""
    if has_clauses(caption):
        return False, "already-has-clauses"
    tags = {t.strip().lower() for t in parse_caption(caption).flat_tags}
    if tags & _MULTI_VIEW_TAGS:
        return True, "multiple-views"
    if tags & _PANEL_LAYOUT_TAGS:
        return True, "panel-layout"
    expected = caption_subject_count(caption)
    if expected is None or expected > 1:
        return True, "multi-girl"
    return False, "single-subject"


# ---------------------------------------------------------------------------
# Detection plumbing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """One detected subject: box in pixels, score, and optional instance mask.

    ``source`` records which detector pass produced the box — ``"subject"`` for
    the ``girl`` prompt, the part prompt itself for a body-part fallback box.
    Carried into the report so a reviewer can tell the two apart.
    """

    box: tuple[float, float, float, float]
    score: float
    mask: np.ndarray | None = None
    source: str = "subject"


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def box_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_containment(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over the *smaller* box — how nested the pair is.

    IoU is blind to nesting: a box wholly inside another scores tiny (`area_small
    / area_large`), so an inset icon and a group box spanning every subject both
    hide from it while scoring ~1.0 here.

    GOTCHA: suppressing on this is off by default — a *real* second subject
    (one girl in front of another) is just as nested as a group box, and
    ablation showed far more of the former in this corpus than the latter.
    Kept as an opt-in knob; :func:`drop_small_boxes` handles the inset case
    instead, and a surviving group box only costs one `count-mismatch` skip.
    """
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smallest = min(box_area(a), box_area(b))
    return (ix * iy) / smallest if smallest > 0 else 0.0


def mask_box_fill(det: Detection) -> float | None:
    """Fraction of the detection's own box that its mask actually claims.

    ``None`` when the detection carries no mask (stub tests, part boxes) so
    callers can fall back to score-only behaviour.
    """
    if det.mask is None:
        return None
    mask = np.asarray(det.mask)
    if mask.ndim == 3:
        mask = mask[0]
    height, width = mask.shape
    x1, y1, x2, y2 = det.box
    window = (
        mask[
            max(0, int(y1)) : min(height, int(y2)),
            max(0, int(x1)) : min(width, int(x2)),
        ]
        > 0.5
    )
    return float(window.mean()) if window.size else 0.0


def dedupe_detections(
    detections: Iterable[Detection],
    iou_threshold: float,
    containment_threshold: float = 1.01,
    fill_ratio_threshold: float = 0.0,
) -> list[Detection]:
    """Greedy IoU + containment suppression, highest score first.

    A threshold above 1.0 disables the containment rule — a box can never be
    more than fully inside another — leaving plain-IoU behaviour.

    ``fill_ratio_threshold`` > 0 enables the mask-quality tie-break
    (``docs/proposal/dedupe_mask_quality.md``, Phase 0 fixed the default at
    2.0): when a candidate collides with a kept box — the pair already judged
    to be the same object — and the candidate's :func:`mask_box_fill` beats the
    kept box's by at least this ratio, the candidate *replaces* the kept box
    instead of being dropped. SAM3's score is box-level confidence and says
    nothing about mask coherence, so a near-empty duplicate can outscore the
    clean mask by a hair and hand every downstream consumer a blank crop.
    Instance count is invariant by construction — the swap only changes which
    of two matched duplicates represents the object. This is deliberately NOT
    an absolute fill gate (settled negative: clean figures live at fill ~0.27);
    the ratio only ever compares the two halves of one matched pair. When
    either mask is missing, the pair falls back to score-only suppression.
    Single-pass: the swapped-in geometry is not re-checked against other kept
    boxes (bounded, order-stable; no cascade observed over the full corpus).
    """
    ranked = sorted(detections, key=lambda d: -d.score)
    keep: list[Detection] = []
    for det in ranked:
        hit = next(
            (
                i
                for i, k in enumerate(keep)
                if box_iou(det.box, k.box) >= iou_threshold
                or box_containment(det.box, k.box) >= containment_threshold
            ),
            None,
        )
        if hit is None:
            keep.append(det)
            continue
        if fill_ratio_threshold > 0:
            kept_fill = mask_box_fill(keep[hit])
            det_fill = mask_box_fill(det)
            if (
                kept_fill is not None
                and det_fill is not None
                and det_fill / max(kept_fill, 1e-9) >= fill_ratio_threshold
            ):
                keep[hit] = det
    return keep


def merge_part_detections(
    subjects: Sequence[Detection],
    parts: Iterable[Detection],
    *,
    iou_threshold: float,
    containment_threshold: float,
) -> list[Detection]:
    """Add body-part boxes that the subject prompt missed, never displacing one.

    Recovers a sheet whose panels are headless close-ups (hip/crotch/backside)
    that the `girl` prompt can't see.

    Containment is applied here even though :func:`dedupe_detections` leaves it
    off by default — the asymmetry is deliberate. A *part* nested in a subject
    is never a real second subject (unlike two subjects nested in each other),
    it's that subject's own body, so typing the rule to the part pass gets
    duplicate suppression without the false positives the global rule costs.

    Subjects are kept unconditionally; parts are considered highest-score first
    against everything kept so far, so duplicate part boxes on one panel
    collapse to one.
    """
    keep = list(subjects)
    for det in sorted(parts, key=lambda d: -d.score):
        if any(
            box_iou(det.box, k.box) >= iou_threshold
            or box_containment(det.box, k.box) >= containment_threshold
            for k in keep
        ):
            continue
        keep.append(det)
    return keep


def crop_instance(
    image: Image.Image,
    det: Detection,
    *,
    pad: float = 0.06,
    blank: bool = True,
    blank_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Padded bbox crop with every non-instance pixel blanked out.

    GOTCHA if skipped: a neighbor standing inside the padded box contributes
    their hair/outfit to this subject's tags. Falls back to a plain crop when
    the detector supplied no mask.
    """
    width, height = image.size
    x1, y1, x2, y2 = det.box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    box = (
        max(0, int(x1 - px)),
        max(0, int(y1 - py)),
        min(width, int(x2 + px)),
        min(height, int(y2 + py)),
    )
    if det.mask is None or not blank:
        return image.crop(box)
    mask = np.asarray(det.mask)
    if mask.ndim == 3:
        mask = mask[0]
    keep = mask[box[1] : box[3], box[0] : box[2]] > 0.5
    pixels = np.asarray(image.crop(box).convert("RGB")).copy()
    pixels[~keep] = blank_color
    return Image.fromarray(pixels)


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MovedTag:
    """One flat-bag tag the rewrite bound to a position and removed from the bag.

    ``margin`` is the relative slack the move cleared (``1 - rival/winner``),
    same scale as ``attribution_margin``.
    """

    tag: str
    position: str
    margin: float


@dataclass(frozen=True)
class RemovalPlan:
    """What the rewrite moves, and why it declined to move the rest.

    ``blocked`` maps a bag tag that *reached* a clause but stays flat to the rule
    that kept it there — the review artifact for tuning the two safety rules.
    """

    moved: tuple[MovedTag, ...] = ()
    blocked: Mapping[str, str] = field(default_factory=dict)


def _score_of(
    scores: Mapping[str, float], kept: Mapping[str, float], tag: str
) -> float:
    """This crop's probability for ``tag``.

    ``scores`` (whole-vocabulary) is what the margin needs, since the runner-up's
    probability matters even below its keep threshold. Falls back to ``kept``
    (0.0 if absent) when a caller supplies no ``scores`` (unit-test stubs only).
    """
    if tag in scores:
        return float(scores[tag])
    return float(kept.get(tag, 0.0))


def plan_bag_removals(
    flat_tags: Sequence[str],
    clause_tags: Sequence[Sequence[str]],
    positions: Sequence[str],
    kept_sets: Sequence[Mapping[str, float]],
    score_sets: Sequence[Mapping[str, float]],
    *,
    vocabulary: ClauseVocabulary,
    margin: float,
) -> RemovalPlan:
    """Decide which flat-bag tags the clauses have earned the right to take.

    A tag moves out of the bag when all five hold: (1) not a character name —
    the cast list stays flat *and* bound; (2) reached exactly one clause — two
    means shared, so it belongs to the bag; (3) corroboration, for a
    character-invariant group — the bag names >=2 values of that group with no
    two-tone marker explaining them away (see ``_CHARACTER_INVARIANT_GROUPS``);
    (4) exclusive keep — no *other* crop kept the tag, else it's a selection
    artifact, not an attribution; (5) relative margin — the runner-up's
    probability is below ``(1 - margin)`` of the winner's (relative, not
    absolute, since per-tag thresholds span ~0.05-0.85).

    Failing any rule is not an error — the tag stays in the bag *and* in its
    clause, i.e. v1's additive behaviour for that one tag.
    """
    bag: dict[str, str] = {}
    for tag in flat_tags:
        bag.setdefault(tag.strip().lower(), tag)

    where: dict[str, list[int]] = {}
    for i, tags in enumerate(clause_tags):
        for tag in tags:
            where.setdefault(tag.strip().lower(), []).append(i)

    # Census of the bag: which tags are characters (rule 1) and how many values
    # of each invariant group it names (rule 3).
    values_per_group: dict[str, set[str]] = {}
    names_in_bag: set[str] = set()
    for key in bag:
        group = vocabulary.group_of(key)
        if group in _CHARACTER_INVARIANT_GROUPS:
            values_per_group.setdefault(group, set()).add(key)
        if key in vocabulary.characters:
            names_in_bag.add(key)
    pinned_groups = {
        group for group, markers in _MULTI_VALUE_MARKERS.items() if markers & bag.keys()
    }

    moved: list[MovedTag] = []
    blocked: dict[str, str] = {}
    for key, indices in sorted(where.items()):
        if key not in bag:
            continue  # the clause tag was never in the bag — nothing to move
        if len(indices) != 1:
            blocked[key] = "multi-clause"
            continue
        group = vocabulary.group_of(key)
        if key in names_in_bag:
            # The cast list stays flat: the bag answers "who is in this image"
            # (and is how a prompt summons them), the clause "which one is where".
            blocked[key] = "character-name"
            continue
        if group in _CHARACTER_INVARIANT_GROUPS:
            if group in pinned_groups:
                blocked[key] = "two-tone-marker"
                continue
            if len(values_per_group.get(group, ())) < 2:
                blocked[key] = "sole-value"
                continue
        winner = indices[0]
        others = [j for j in range(len(clause_tags)) if j != winner]
        if any(key in kept_sets[j] for j in others):
            blocked[key] = "multi-kept"
            continue
        mine = _score_of(score_sets[winner], kept_sets[winner], key)
        rival = max(
            (_score_of(score_sets[j], kept_sets[j], key) for j in others),
            default=0.0,
        )
        slack = 1.0 - rival / mine if mine > 0.0 else 0.0
        if slack < margin:
            blocked[key] = "margin"
            continue
        moved.append(
            MovedTag(
                tag=bag[key],
                position=positions[winner],
                margin=round(slack, 3),
            )
        )
    return RemovalPlan(moved=tuple(moved), blocked=blocked)


@dataclass
class InstanceProposal:
    position: str
    box: list[int]
    score: float
    tags: list[str]
    crop: str | None = None
    source: str = "subject"
    # How many of ``tags`` the flat bag did not already contain.
    novel: int = 0


@dataclass
class ImageProposal:
    image: str
    caption_path: str
    status: str
    detected: int = 0
    expected: int | None = None
    original: str = ""
    proposed: str | None = None
    instances: list[InstanceProposal] = field(default_factory=list)
    # Boxes as detected, recorded even when a gate rejects the image (for
    # reviewer evidence); ``instances`` only populates once every gate passes.
    detections: list[dict] = field(default_factory=list)
    tokens: int | None = None
    # v2: which bag tags the clauses took, and which reached a clause but stayed
    # flat (tag -> the rule that pinned it). Both empty under ``rewrite=False``.
    moved: list[dict] = field(default_factory=list)
    pinned: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "proposed"


@dataclass
class PositionCaptionStats:
    seen: int = 0
    candidates: int = 0
    proposed: int = 0
    written: int = 0
    rewritten: int = 0
    moved_tags: int = 0
    # Clause tags in total, and how many were novel (not in the caption).
    # ``clause_tags - novel_tags`` is reuse.
    clause_tags: int = 0
    novel_tags: int = 0
    pinned_tags: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def pin(self, reason: str) -> None:
        self.pinned_tags[reason] = self.pinned_tags.get(reason, 0) + 1


@dataclass(frozen=True)
class PositionCaptionOptions:
    """Knobs for one pass. Defaults are the shipped v2 recipe."""

    prompt: str = "girl"
    score_threshold: float = 0.5
    retry_score_threshold: float = 0.35
    # Body-part fallback: extra SAM3 prompts run only when the subject prompt
    # undershoots. Off by default (empty tuple) — see ``merge_part_detections``.
    part_prompts: tuple[str, ...] = ()
    part_score_threshold: float = 0.5
    part_containment_threshold: float = 0.7
    iou_threshold: float = 0.65
    # Off by default — see ``box_containment``.
    containment_threshold: float = 1.01
    # Mask-quality tie-break inside an NMS-matched pair — see
    # ``dedupe_detections``. 2.0 sits in the measured empty ratio band
    # (multiview_audit.md §5.4); 0 disables (score-only survivor).
    dedupe_fill_ratio: float = 2.0
    min_area_frac: float = 0.005
    pad: float = 0.06
    blank_crops: bool = True
    row_tol: float = 0.25
    max_clause_tags: int = 8
    # How many tags a clause may introduce that the caption never contained;
    # the rest is filled from the flat bag first, since only a bag tag can
    # actually *move*. 0 = never invent; ``max_clause_tags`` = bag-blind.
    max_novel_tags: int = 1
    name_confidence: float = 0.5
    allow_unlisted_names: bool = False
    min_instances: int = 2
    max_instances: int = 8
    strict_count: bool = True
    discriminative_only: bool = True
    bag_gated_identity: bool = True
    # On a repeated-subject layout (``multiple views`` / comic panels), keep the
    # character's own traits — and her name — out of every clause: they belong
    # to the girl, not to a view of her.
    multi_view_gate: bool = True
    # v2: move an attributable tag out of the flat bag into its clause. False is
    # the additive v1 behaviour (bag untouched), kept for the training A/B.
    rewrite: bool = True
    # How far the winning crop must clear every other crop, *relative to its own
    # probability* (``1 - rival/winner``), before a tag may leave the bag — on
    # top of the hard rule that no other crop kept it. Only the removal is gated
    # — a tag that fails still enters its clause, so the caption degrades to v1
    # for it. 0.0 = trust the tagger's thresholds alone.
    attribution_margin: float = 0.25


def detect_subjects(
    image: Image.Image,
    detect_fn: Callable[[Image.Image, float], list[Detection]],
    options: PositionCaptionOptions,
    expected: int | None,
    part_detect_fn: Callable[[Image.Image, str, float], list[Detection]] | None = None,
) -> list[Detection]:
    """Detect + dedupe, with two escalations when the count falls short.

    ``detect_fn(image, score_threshold)`` returns raw detections. Both
    escalations fire only when detection undershoots what we have reason to
    expect (on a resolved image they'd only add duplicates):

    1. **Lower the score threshold** — recovers an extreme close-up.
    2. **Body-part prompts** (``part_detect_fn``, when supplied) — recovers a
       headless panel the subject prompt can't see at any threshold. Merged via
       :func:`merge_part_detections`, never displacing a subject box.

    GOTCHA: target is ``expected or min_instances``, NOT ``expected`` alone — a
    ``multiple views`` sheet reports ``expected=None`` on purpose (count tags
    characters, not views); gating on truthiness would skip the retry for that
    whole population.
    """

    def run(threshold: float) -> list[Detection]:
        dets = dedupe_detections(
            detect_fn(image, threshold),
            options.iou_threshold,
            options.containment_threshold,
            options.dedupe_fill_ratio,
        )
        return drop_small_boxes(dets, image.size, options.min_area_frac)

    dets = run(options.score_threshold)
    target = expected or options.min_instances
    if len(dets) < target and options.retry_score_threshold < options.score_threshold:
        retry = run(options.retry_score_threshold)
        if len(retry) > len(dets):
            dets = retry

    if len(dets) >= target or part_detect_fn is None or not options.part_prompts:
        return dets

    parts: list[Detection] = []
    for prompt in options.part_prompts:
        parts.extend(part_detect_fn(image, prompt, options.part_score_threshold))
    parts = drop_small_boxes(parts, image.size, options.min_area_frac)
    merged = merge_part_detections(
        dets,
        parts,
        iou_threshold=options.iou_threshold,
        containment_threshold=options.part_containment_threshold,
    )
    # Top up to the target, no further — a part prompt is a looser concept than
    # ``girl`` and can fragment into more boxes than there are real panels.
    return merged[: max(target, len(dets))]


def drop_small_boxes(
    detections: Iterable[Detection],
    image_size: tuple[int, int],
    min_area_frac: float,
) -> list[Detection]:
    """Discard boxes too small to be a bindable subject.

    A detection covering 0.3% of the canvas is an inset — a character drawn on a
    phone screen, a poster, a chibi in a corner — not a subject a position clause
    can meaningfully describe.
    """
    if min_area_frac <= 0:
        return list(detections)
    floor = min_area_frac * image_size[0] * image_size[1]
    return [d for d in detections if box_area(d.box) >= floor]


def propose_for_image(
    image: Image.Image,
    caption: str,
    *,
    detect_fn: Callable[[Image.Image, float], list[Detection]],
    tag_fn: Callable[[Image.Image], Mapping[str, object]],
    vocabulary: ClauseVocabulary,
    options: PositionCaptionOptions,
    crop_sink: Callable[[int, str, Image.Image], str] | None = None,
    part_detect_fn: Callable[[Image.Image, str, float], list[Detection]] | None = None,
) -> ImageProposal:
    """Build the clause proposal for one image. Never writes any caption."""
    parsed = parse_caption(caption)
    flat_bag = frozenset(t.strip().lower() for t in parsed.flat_tags)
    expected = caption_subject_count(caption)

    proposal = ImageProposal(
        image="",
        caption_path="",
        status="proposed",
        expected=expected,
        original=caption,
    )

    dets = detect_subjects(image, detect_fn, options, expected, part_detect_fn)
    proposal.detected = len(dets)
    proposal.detections = [
        {
            "box": [int(v) for v in d.box],
            "score": round(float(d.score), 3),
            "source": d.source,
        }
        for d in dets
    ]
    if len(dets) < options.min_instances:
        proposal.status = "skip:too-few-instances"
        return proposal
    if len(dets) > options.max_instances:
        proposal.status = "skip:too-many-instances"
        return proposal
    # Detection and the caption's own count must agree, else we'd write clauses
    # we can't ground. "Agree" is a range, not equality: the ``girl`` prompt
    # picks up males inconsistently, so girls..girls+boys both count as
    # consistent with the caption.
    if options.strict_count and expected:
        boys = caption_boy_count(caption)
        upper = None if boys is None else expected + boys
        if len(dets) < expected or (upper is not None and len(dets) > upper):
            proposal.status = "skip:count-mismatch"
            return proposal
    # A layout tag waives the check above (``expected`` is None by design); an
    # ``Nkoma`` tag restores a generous ceiling so a subject detected twice
    # still has a backstop.
    if options.strict_count and not expected:
        ceiling = caption_panel_ceiling(caption)
        if ceiling is not None and len(dets) > ceiling:
            proposal.status = "skip:count-mismatch"
            return proposal

    order = ordered_indices([d.box for d in dets], image.size, row_tol=options.row_tol)
    dets = [dets[i] for i in order]
    positions = assign_positions(
        [d.box for d in dets], image.size, row_tol=options.row_tol
    )

    # GOTCHA: mask-blanking is a subject-crop fix. On a part box the mask IS the
    # part, so blanking would delete the very content (torn jeans, pantyhose)
    # the part pass exists to recover. Part crops take the plain padded bbox.
    crops = [
        crop_instance(
            image,
            d,
            pad=options.pad,
            blank=options.blank_crops and d.source == "subject",
        )
        for d in dets
    ]
    predictions = [tag_fn(crop) for crop in crops]
    kept_sets = [dict(p.get("kept") or {}) for p in predictions]
    score_sets = [dict(p.get("scores") or {}) for p in predictions]
    # A tag only *this* crop keeps is attributable to it; one every crop keeps
    # discriminates nothing, so it stays in the flat bag instead of padding
    # every clause identically.
    counts: dict[str, int] = {}
    for kept in kept_sets:
        for tag in kept:
            counts[tag] = counts.get(tag, 0) + 1
    attributable = frozenset(t for t, n in counts.items() if n == 1)
    shared = frozenset(t for t, n in counts.items() if n == len(kept_sets))
    view_invariant = options.multi_view_gate and is_repeated_subject_layout(caption)

    for i, (det, kept, pred) in enumerate(zip(dets, kept_sets, predictions)):
        tags = vocabulary.select(
            kept,
            dict(pred.get("groups") or {}),
            flat_bag=flat_bag,
            attributable=attributable,
            shared=shared,
            max_tags=options.max_clause_tags,
            name_confidence=options.name_confidence,
            allow_unlisted_names=options.allow_unlisted_names,
            discriminative_only=options.discriminative_only,
            allow_identity=det.source == "subject",
            bag_gated_identity=options.bag_gated_identity,
            view_invariant=view_invariant,
            max_novel_tags=options.max_novel_tags,
        )
        crop_name = crop_sink(i, positions[i], crops[i]) if crop_sink else None
        proposal.instances.append(
            InstanceProposal(
                position=positions[i],
                box=[int(v) for v in det.box],
                score=round(float(det.score), 3),
                tags=tags,
                crop=crop_name,
                source=det.source,
                novel=sum(1 for t in tags if t.strip().lower() not in flat_bag),
            )
        )

    clauses = [
        PositionClause(position=inst.position, tags=tuple(inst.tags))
        for inst in proposal.instances
        if inst.tags
    ]
    if len(clauses) < options.min_instances:
        # Every crop tagged identically — the subjects are genuinely
        # indistinguishable to the tagger, so there is nothing to bind.
        proposal.status = "skip:no-discriminative-tags"
        return proposal

    flat = list(parsed.flat_tags)
    if options.rewrite:
        plan = plan_bag_removals(
            parsed.flat_tags,
            [inst.tags for inst in proposal.instances],
            [inst.position for inst in proposal.instances],
            kept_sets,
            score_sets,
            vocabulary=vocabulary,
            margin=options.attribution_margin,
        )
        proposal.pinned = dict(plan.blocked)
        taken = {m.tag.strip().lower() for m in plan.moved}
        remaining = [t for t in flat if t.strip().lower() not in taken]
        # Guard against emptying the bag entirely (unreachable in practice, but
        # the rewrite removes text so this is asserted, not assumed).
        if remaining:
            flat = remaining
            proposal.moved = [
                {"tag": m.tag, "position": m.position, "margin": m.margin}
                for m in plan.moved
            ]

    proposal.proposed = compose_caption(flat, clauses)
    return proposal


def _crop_sink(crops_dir: Path, rel: Path) -> Callable[[int, str, Image.Image], str]:
    """Save each crop under ``crops_dir`` mirroring the dataset layout.

    The dry-run review artifact: the reviewer reads a proposed clause next to
    the exact pixels the tagger saw, which is the only way to tell a detection
    miss from a tagging miss.
    """
    target = crops_dir / rel.parent

    def sink(index: int, position: str, crop: Image.Image) -> str:
        target.mkdir(parents=True, exist_ok=True)
        name = f"{rel.stem}_{index}_{position.replace(' ', '-')}.png"
        crop.save(target / name)
        return str((target / name).relative_to(crops_dir))

    return sink


def _save_skip_overlay(
    crops_dir: Path, rel: Path, image: Image.Image, proposal: ImageProposal
) -> None:
    """Draw the detected boxes over a skipped image, under ``_skipped/``.

    A skip produces no crops (``crop_sink`` runs only once every gate passes),
    which left the dry-run report with zero visual evidence for exactly the rows
    a reviewer has to adjudicate — is this an over-detection, a missing subject,
    or a wrong caption count? The overlay answers that at a glance.
    """
    from PIL import ImageDraw

    target = crops_dir / "_skipped" / rel.parent
    target.mkdir(parents=True, exist_ok=True)
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for i, det in enumerate(proposal.detections):
        box = det["box"]
        draw.rectangle(box, outline=(255, 0, 0), width=4)
        draw.text(
            (box[0] + 6, box[1] + 6), f"{i}:{det['score']:.2f}", fill=(255, 255, 0)
        )
    status = proposal.status.removeprefix("skip:")
    canvas.save(target / f"{rel.stem}_{status}.png")


def run_position_captions(
    *,
    resized_dir: Path,
    source_dir: Path,
    detect_fn: Callable[[Image.Image, float], list[Detection]],
    tag_fn: Callable[[Image.Image], Mapping[str, object]],
    vocabulary: ClauseVocabulary,
    options: PositionCaptionOptions | None = None,
    path_pattern: str | None = None,
    apply: bool = False,
    crops_dir: Path | None = None,
    token_count_fn: Callable[[str], int] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    part_detect_fn: Callable[[Image.Image, str, float], list[Detection]] | None = None,
) -> tuple[list[ImageProposal], PositionCaptionStats]:
    """Walk the resized tree, propose clauses, and (with ``apply``) write them.

    GOTCHA: the caption master (``source_dir``/``image_dataset/``) is NEVER
    written — the rewrite lands at ``resized_dir/<rel>`` (what the TE step
    encodes); the master is only the read fallback for an unmirrored image.
    Two things make that safe: the mirror pass
    (:func:`library.captioning.preprocess.write_corrected_preprocess_captions`)
    re-attaches clauses found on a destination whose master has none, and the
    write bumps mtime so the TE cache re-encodes (a stale
    ``{stem}.variants.txt`` sidecar, which wins over ``{stem}.txt`` at encode
    time, is dropped here too).

    v2's write is a rewrite, not an append. Recoverable
    (:func:`flatten_captions`) but not a no-op, hence ``apply`` defaults off.
    """
    from library.preprocess._dataset import walk_images

    options = options or PositionCaptionOptions()
    stats = PositionCaptionStats()
    rows: list[ImageProposal] = []

    images = walk_images(resized_dir, recursive=True, pattern=path_pattern)
    stats.seen = len(images)

    for index, image_path in enumerate(images, 1):
        rel = image_path.relative_to(resized_dir).with_suffix(".txt")
        dst_caption = resized_dir / rel
        # Prefer the derived caption (already order-corrected, and carrying an
        # earlier run's clauses so `is_candidate` skips it) and fall back to the
        # master for an image the caption step has not mirrored yet.
        caption_path = dst_caption if dst_caption.exists() else source_dir / rel
        if progress is not None:
            progress(index, len(images), str(rel))
        if not caption_path.exists():
            stats.skip("no-caption")
            continue
        caption = caption_path.read_text(encoding="utf-8").strip()
        ok, reason = is_candidate(caption)
        if not ok:
            stats.skip(reason)
            continue
        stats.candidates += 1

        crop_sink = _crop_sink(crops_dir, rel) if crops_dir is not None else None
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        proposal = propose_for_image(
            image,
            caption,
            detect_fn=detect_fn,
            tag_fn=tag_fn,
            vocabulary=vocabulary,
            options=options,
            crop_sink=crop_sink,
            part_detect_fn=part_detect_fn,
        )
        proposal.image = str(image_path.relative_to(resized_dir))
        proposal.caption_path = str(rel)
        rows.append(proposal)

        if not proposal.ok:
            stats.skip(proposal.status.removeprefix("skip:"))
            if crops_dir is not None:
                _save_skip_overlay(crops_dir, rel, image, proposal)
            continue
        stats.proposed += 1
        stats.clause_tags += sum(len(i.tags) for i in proposal.instances)
        stats.novel_tags += sum(i.novel for i in proposal.instances)
        if proposal.moved:
            stats.rewritten += 1
            stats.moved_tags += len(proposal.moved)
        for reason in proposal.pinned.values():
            stats.pin(reason)
        if token_count_fn is not None and proposal.proposed:
            proposal.tokens = token_count_fn(proposal.proposed)
        if apply:
            _write_derived_caption(dst_caption, proposal.proposed)
            stats.written += 1

    return rows, stats


def _write_derived_caption(dst_caption: Path, text: str) -> None:
    """Write a caption into the resized tree and drop its variant sidecar.

    GOTCHA: the sidecar wins over ``{stem}.txt`` at encode time, so leaving a
    pre-clause one behind would keep training the pre-clause caption regardless
    of how fresh ``{stem}.txt`` is.
    """
    from library.preprocess.caption_variants import variants_sidecar_path

    dst_caption.parent.mkdir(parents=True, exist_ok=True)
    dst_caption.write_text(text, encoding="utf-8")
    sidecar = variants_sidecar_path(dst_caption)
    if sidecar.exists():
        sidecar.unlink()


def flatten_captions(
    *,
    resized_dir: Path,
    source_dir: Path,
    path_pattern: str | None = None,
    apply: bool = False,
) -> tuple[list[dict], PositionCaptionStats]:
    """Undo a rewrite: merge every caption's clauses back into its flat bag.

    Text-only (no SAM3, no tagger, no pixels) since v2 moves tags rather than
    deleting them. Two uses: backing out an ``--apply`` run, and building the
    clause-free control corpus for a training A/B.

    Reads/writes the same derived caption as :func:`run_position_captions`
    (``resized_dir/<rel>``, master only as read fallback).

    GOTCHA: hand-written clauses are flattened too — the pass can't tell them
    from generated ones. Recoverable in the derived layer (master still holds
    them), but a real loss of curation if the clauses only ever existed here —
    hence the dry-run default.
    """
    from library.preprocess._dataset import walk_images

    stats = PositionCaptionStats()
    rows: list[dict] = []
    images = walk_images(resized_dir, recursive=True, pattern=path_pattern)
    stats.seen = len(images)
    for image_path in images:
        rel = image_path.relative_to(resized_dir).with_suffix(".txt")
        dst_caption = resized_dir / rel
        caption_path = dst_caption if dst_caption.exists() else source_dir / rel
        if not caption_path.exists():
            stats.skip("no-caption")
            continue
        original = caption_path.read_text(encoding="utf-8").strip()
        if not has_clauses(original):
            stats.skip("no-clauses")
            continue
        stats.candidates += 1
        flattened = flatten_caption(original)
        if flattened == original:
            stats.skip("unchanged")
            continue
        stats.proposed += 1
        rows.append(
            {
                "caption_path": str(rel),
                "original": original,
                "proposed": flattened,
            }
        )
        if apply:
            _write_derived_caption(dst_caption, flattened)
            stats.written += 1
    return rows, stats
