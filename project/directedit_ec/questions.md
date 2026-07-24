# directedit_ec — open questions

## Q1 — Is the cliff-shaped operating point data-owned or architectural?

The inpaint prior's usable band is narrow because it was trained "cond is
authoritative" (`b_cond_init=-6`, `drop_p=0`, aligned pairs). Phase 2's
cross-image subject descriptor is the falsifier: if sweet-spot width does
NOT improve over inpaint's (~1 b_offset unit), the pairing wasn't the binding
constraint — the cliff is architectural (gate granularity), and the next
lever is per-block/per-σ gate schedules (a different, smaller proposal).

## Q2 — Can a trained prior do associative (position-free) retrieval?

The geometry row proved the inpaint prior is position-locked: with a
full-frame hole it produces the pose but keeps nothing. Phase 2 trains
retrieval that positional copying cannot satisfy (cond = image A, target =
image B of the same character). Falsifiable target: parity with vinj_t6 on
the 1b geometry edit.

## Q3 — What owns the hard-image ceiling?

10473210's in-place edits (halo removal, white→black hair) fail for every
method — the limit is the teacher (base model + inversion), not preservation.
Is it caption-attractor strength, inversion quality at CFG 4, or model prior?
Matters for Phase 3: a feed-forward editor can only be as good as the data
the teacher can label.

## Q4 — The hole-style artifact

Flat, saturated regeneration inside the hole on simple flat-background images
(7538087; all EC arms, mask-independent). Inpaint-prior-owned. Does it
persist under the Phase-2 subject prior, or is it an artifact of the inpaint
training data specifically? Watch it in the Phase-2 gate re-run.

## Q5 — Automatic mask source

The recipe needs a hole box. Manual today; the cfgdelta subject localizer
(foveation line's reusable artifact, `project_foveated_denoise_p0`) is the
planned automatic source. Open: does a loose auto-box degrade the recipe
(the anchor mask drops Δz everywhere inside it), i.e. how tight must masks be?

## Q6 — Edit-success metric beyond renders

Render judging caps phase gates at small n. Tag-readback
(`docs/proposal/tag_readback_reward.md`, Phase 0a passed) would give a
scalable edit-lands metric — blocked on a trained tagger checkpoint at bench
time. Wire it into `run_bench.py` when available.

## Q7 — Paper bar (decide after Phase 2)

The novel claim: *a pretrained image-conditioning adapter's attention gate is
a continuous preservation dial for flow-inversion editing, composing exactly
with residual-anchored inversion.* Missing for a paper: matched-NFE external
baselines (RF-Inversion / RF-Solver / FireFlow / FlowEdit), PIE-Bench,
quantitative edit-success + identity metrics, and the Phase-2 adapter so the
story isn't one off-label inpaint checkpoint. Per the FSG lesson
(`project_fsg_golden_path_phase0`): no free-quality claims without the
matched-NFE table.
