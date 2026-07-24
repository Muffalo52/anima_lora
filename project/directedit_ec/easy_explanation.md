# directedit_ec — an easy explanation

A plain-language walkthrough of what this line does and *why* it works.
`methods.md` tells you where the code is; this doc tells you what the code
is doing and the reasoning behind the recipe.

## The problem: editing an image without destroying it

You have a finished image and you want one local change — add glasses,
remove a halo, recolor the hair. Two forces fight each other:

- **Edit leverage** — the model must be free enough to actually make the
  change.
- **Preservation** — everything you did *not* ask to change must survive.

Every editing method is a different way of trading these off. This line is
about getting a better trade at **zero training cost**.

## Background 1: DirectEdit (what we start from)

DirectEdit works by "rewinding" the image back into noise and replaying it
with a new prompt:

1. **Inversion.** Run the flow-matching sampler *backwards* from the image
   to noise, using the source caption ψ_src. At each step, record the small
   error the model makes: `Δz_i = z_inv[i+1] − z_inv[i]`. These recorded
   residuals are called the **Δz anchor**.
2. **Edit pass.** Run the sampler *forward* again from that noise, but with
   the edit caption ψ_tar (e.g. same caption + ", glasses"). At every step,
   add back the recorded `Δz_i` before evaluating the model.

The anchor is the preservation mechanism: if ψ_tar equals ψ_src, adding the
residuals back makes the replay **bit-exact** — you get the original image
back perfectly. Change the caption a little, and the trajectory mostly
follows the original while the cross-attention pulls it toward the edit.

**The catch:** at real guidance strength (CFG 4) the anchor is overpowered.
CFG amplifies the prompt's pull, the trajectory drifts, and the background /
composition gets rewritten (Phase 0 measured this: pure-anchor edits lose
the composition badly).

The paper's patch for this is **V-injection**: for the first `t_inj` steps,
run a *parallel* forward pass on the unedited source and inject its
self-attention V (values) into the edit pass. That hard-copies source
appearance positionally. It works, but it's clumsy — you must tune "how many
steps" × "which blocks", it costs a second forward pass per injected step,
and it breaks `compile_blocks`.

## Background 2: EasyControl (the tool we repurpose)

EasyControl is our image-conditioning adapter family. Mechanically:

- The conditioning image is encoded and its tokens are appended to the
  self-attention **as extra keys/values** ("extended self-attention"). The
  target's queries can attend to the cond image's K/V and copy appearance
  from it.
- A learned per-block scalar **`b_cond`** biases the softmax logits of the
  cond positions. It is a *gate*: it decides how much attention mass flows
  to the cond tokens. Because it's a logit bias, each −1 on it means roughly
  e× less cond influence — a smooth, interpretable dial.
- The cond K/V are computed **once** and cached ("KV prefill"); after that,
  conditioning is nearly free per step.

The **inpaint adapter** is one trained EasyControl checkpoint: cond = the
image with a gray hole, target = the full image. Its learned behavior is
literally *"copy everything from the cond image, except the hole — generate
freely there."*

## The key insight

Look at what V-injection is: a *hand-tuned, hard, positional* copy of source
appearance into the edit pass.

Look at what EasyControl's extended self-attention is: a *learned, gated,
soft* copy of source appearance from a cond image.

**They are the same mechanism.** EasyControl is the trained generalization
of V-injection. So instead of hand-tuning attention injection, load an
already-trained EasyControl adapter onto the DirectEdit edit pass, feed it
the source image as cond, and let its learned gate do the preservation.

Phase 0 confirmed this composes cleanly:

- With the EC cond stream active through **both** the inversion and the edit
  pass, the Δz anchor's exact-reconstruction property still holds (recon
  quality ~unchanged). The two mechanisms don't fight.
- At the right gate setting, EC beat V-injection on preservation *while the
  edit still landed* — e.g. glasses land and the full source background
  survives, where vinj_t6 landed the glasses but invented a fireworks
  background.
- Tuning surface collapsed from "`t_inj` steps × block set" to **one
  scalar** (`--easycontrol_b_offset`, added on top of the trained `b_cond`),
  and the per-step parallel source forward became one cached KV prefill.

One important negative result: EC and V-injection **cannot be combined**.
The EC-patched block forward routes attention through its own extended
attention path, which bypasses the code V-injection patches. It's a
replacement, not an addition.

## Why one scalar wasn't enough — and the hole trick

The inpaint adapter was trained with "cond is authoritative" settings, so
its operating point is cliff-shaped: `cond_scale` turned out to be
near-binary (0.5 ≈ adapter off, 1.0 ≈ total clamp — the edit is suppressed
and you just get the source back). `b_offset` *is* a usable dial, but the
sweet spot was narrow (about −1 to −2) and **image-dependent** — one image
needed −1, another −2. Per-image tuning is exactly what we wanted to
eliminate.

The fix (Phase 1a) is to stop fighting the adapter and use it *as trained*.
The inpaint prior already knows one exception rule: **"copy everything
except the hole."** So: punch a gray hole in the cond image over the region
you want to edit. Outside the hole, the prior clamps to the source (its
trained behavior — no gate fiddling needed, `b_offset` stays at 0). Inside
the hole, it generates freely, steered by the edit caption.

Two practical details matter:

- The hole is drawn on the cond **image before VAE encoding** (gray fill),
  because that's what the adapter saw in training. Zeroing the latent
  instead would be out-of-distribution.
- This turned out to be half the answer.

## The amendment: you must punch the hole twice

With only the cond hole, the edit landed on just 1 of 3 test images. The
reason: DirectEdit's **Δz anchor is global**. Even with the EC prior
generating freely inside the hole, the anchor residuals — recorded from the
*unedited* source trajectory — keep pulling the hole region back toward the
source every step. One preservation mechanism was released; the other was
still holding on.

So the same mask is applied to the anchor too: inside the edit region, the
`Δz_i` residuals are dropped (this is the anchor-side half of the paper's
Eq. 12 mask blending, which had been a stub until this line). Now both
preservation mechanisms agree on where the "free zone" is:

| Region | EC inpaint prior | Δz anchor |
|---|---|---|
| Outside the mask | clamps to source (trained copy) | full residuals → near-exact trajectory |
| Inside the mask | generates freely | no anchor pull |

With both holes punched, the recipe lands **3/3** — including the hard
image — at `b_offset 0` (no per-image tuning at all), with outside-mask
preservation 2.6–17× better than the best alternative (V-injection or
gate-tuned EC).

That is the entire shipped recipe:

```bash
python scripts/edit.py <image> --edit "<caption + edit>" \
  --easycontrol_weight output/ckpt/methods/anima_inpaint.safetensors \
  --easycontrol_mask mask.png \
  --mask mask.png            # SAME file for both flags
```

Phase 1b then checked edit-type generalization: **ADD, REMOVE, REPLACE**
(the in-place edits) all work — REMOVE and REPLACE in fact *only* worked
under this recipe; V-injection landed neither anywhere.

## The whole thing in one picture

The pipeline — one mask file feeds both preservation mechanisms:

```
                       mask.png  (1 = edit region)
                           │
             ┌─────────────┴───────────────┐
             │ hole in the COND            │ hole in the ANCHOR
             ▼                             ▼
      source image                  Δz_i residuals
      gray-fill the hole            (recorded during inversion)
             │                      zeroed inside the edit region
             ▼                             │
      VAE encode → EC cond K/V             │
      prefilled ONCE                       │
      (inpaint adapter, b_offset 0)        │
             │                             │
   ══════════╪═════════════════════════════╪══════════════════
             │        THE TWO PASSES       │
             │                             │
      ① INVERSION  (ψ_src, CFG 1, EC cond stream ON)
         image ──backwards──▶ noise,  record Δz_i each step
             │                             │
      ② EDIT PASS  (ψ_tar, CFG 4, EC cond stream ON)
         noise ──forward──▶ edited image
         each step:                        │
           · z += Δz_i·(1−mask)  ◀─────────┘   anchor pins the
                                               trajectory outside
           · extended self-attn ◀── cond K/V   the mask only
             target queries copy source
             appearance — except the hole,
             where cond is just gray
```

And what each region of the latent experiences during the edit pass:

```
   ┌─────────────────────────────────────────────────┐
   │  OUTSIDE the mask                               │
   │   · EC inpaint prior clamps to source (trained  │
   │     "copy everything" behavior)                 │
   │   · full Δz anchor → near-exact trajectory      │
   │   ⇒ the original image survives                 │
   │                                                 │
   │            ┌───────────────────────┐            │
   │            │  INSIDE the mask      │            │
   │            │   · cond is gray      │            │
   │            │     ⇒ EC prior lets go│            │
   │            │   · Δz_i dropped      │            │
   │            │     ⇒ anchor lets go  │            │
   │            │   ⇒ ψ_tar steers      │            │
   │            │     freely — the edit │            │
   │            │     lands HERE        │            │
   │            └───────────────────────┘            │
   │                                                 │
   └─────────────────────────────────────────────────┘
```

The Phase-1a failure mode is visible in this picture: with only the cond
hole, the inner box still had the anchor arrow pointing into it — one hand
released, the other still holding. Punching both holes is what makes the
inner box genuinely free.

## What it can't do (and why)

- **Geometry / pose edits fail — by design of the prior.** The inpaint
  adapter copies *positionally* ("the pixel at (x,y) should look like cond's
  (x,y)"). A pose change needs content to move to different positions, which
  positional copying can't express. Masking the full frame does let the pose
  land — but then nothing is preserved, which just proves the suppression
  was preservation-owned. Fixing this is Phase 2's goal: train a **subject
  descriptor** adapter on cross-image pairs of the same character (image A
  as cond, image B as target), so the network is *forced* to learn
  content-based retrieval instead of positional copying — same
  EasyControlNetwork, only the data pairing changes.
- **Inpaint-style artifact:** on simple flat-background images, the hole
  sometimes regenerates in a flat, saturated style — the inpaint prior's own
  aesthetic leaking through. Mask-independent, adapter-owned.
- **The mask is manual today** (a drawn box). The planned automatic upgrade
  is the cfgdelta subject localizer inherited from the foveation line.

## The one-sentence summary

DirectEdit's preservation used to come from a hand-tuned attention hack
(V-injection); this line replaces it with a *pretrained* inpaint adapter's
learned "copy everything except the hole" behavior, punches the same hole in
both the adapter's cond image and DirectEdit's Δz anchor, and gets
better-than-V-injection editing with **zero training and zero per-image
tuning**.
