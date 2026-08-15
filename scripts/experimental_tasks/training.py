"""Experimental training entry-points: chimera, byg.

These are wired up under ``make exp-*`` / ``python tasks.py exp-*`` to keep
the unstable methods visually separate from the shipped ones (lora family,
modulation guidance, hydra, EasyControl). Each ``cmd_*`` is a thin shim that
translates env vars + extra argv into the right ``train.py`` (via
``accelerate launch``) or ``scripts/preprocess/*.py`` call.

(EasyControl and Turbo graduated to the shipped ``make easycontrol*`` /
``make turbo`` targets — see ``scripts/tasks/training.py``. The SPD distillation
LoRA — "Case B" — was archived 2026-07-05 to ``_archive/spd/``; SPD now ships
only as the training-free ``--spd`` / ``SPD=1`` inference stack.)
"""

from __future__ import annotations

from scripts.tasks._common import (
    PY,
    run,
    train,
)


def cmd_soft_tokens(extra):
    train("soft_tokens", extra)


def cmd_chimera(extra):
    """ChimeraHydra (dual-pool additive routing — docs/proposal/chimera_hydra.md).

    Drives ``configs/methods/chimera.toml``: OrthoHydra split into a content
    pool (K_c=3, per-layer rank-R router on pooled lx) and a freq pool
    (K_f=3, network-level FreqRouter on concat(FEI, σ-features)). Pool
    outputs are added (no multiplicative gate, no σ-band overlap mask).
    Single-phase co-training; per-pool balance loss; T-LoRA mask on the
    content branch only.
    """
    train("chimera", extra)


def cmd_byg(extra):
    """BYG — Bootstrap Your Generator unpaired instruction editing.

    Plain rank-64 LoRA trained with a multi-forward unpaired objective (bootstrap
    rollout + DDS prior + cycle + identity), conditioned on a parameter-free
    token-concat source latent. Reads ``configs/methods/byg.toml``.

    Run ``exp-byg-data`` first to build the per-image edit-tuple sidecars under
    ``post_image_dataset/byg/``. The image VAE/TE caches are the standard
    ``preprocess`` ones (the source image IS the training image).
    """
    train("byg", extra)


def cmd_cjk_cache(extra):
    """Stage the CJK distillation cache (Qwen hidden states + teacher output).

    One pass over ``post_image_dataset/cjk_distill/pairs.jsonl``. The teacher
    is frozen and both arms share the Qwen side, so this is computed once and
    reused by every ``exp-distill-cjk`` arm — rebuild only when the corpus, the
    tokenizer, or the ext *mapping* changes (not when the ext *values* do).
    The holdout split additionally caches the all-EN reference and the
    unk-wall arms, which is what makes the recovery-fraction metric possible
    without a text encoder at train time.
    """
    run([PY, "-m", "scripts.distill_cjk.cache", *extra])


def cmd_distill_cjk(extra):
    """Distill the extended T5-side vocab rows (project/cjk_aware_anima 2b).

    Run the gates in order — ``--mode oracle`` (loop + trimming invariant),
    ``--mode capacity`` (can the ext rows express the teacher at all?), then
    ``--mode train``. Output is a vocab pack (``ext_embed.safetensors`` +
    ``.json``), not a LoRA.
    """
    run([PY, "-m", "scripts.distill_cjk.distill", *extra])


def cmd_byg_data(extra):
    """Build BYG edit-tuple sidecars (tag-swap) into ``post_image_dataset/byg/``.

    One offline pass over the captioned corpus emitting
    ``<stem>_byg.safetensors`` (4 encoded role conditionings) per image. Pass
    ``--limit N`` for a quick smoke subset, ``--overwrite`` to rebuild.
    """
    run(
        [
            PY,
            "scripts/byg/build_edit_tuples.py",
            "--dir",
            "image_dataset",
            "--cache_dir",
            "post_image_dataset/byg",
            "--qwen3",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "--dit",
            "models/diffusion_models/anima-base-v1.0.safetensors",
            *extra,
        ]
    )
