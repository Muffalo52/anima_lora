# project/ — active promoted lines

One subdir per research line that has graduated past "proposal + bench report"
into an ongoing project with open phases. Each subdir is the line's home page:

| File | Contents |
|---|---|
| `methods.md` | The implementation — what code exists, where it lives, how to run it |
| `bench.md` | Digest of measured results — omitted when the line's bench lives in-tree (its own `report.md` serves directly) |
| `questions.md` | Open questions the line has not answered |
| `roadmap.md` | Remaining phases, gates, and kill criteria |
| `outcomes.md` | Shippable/practical artifacts the line produced (optional — appears once something is ship-shaped) |

Canonical sources these digest (never duplicated wholesale):
the line's proposal(s) (frozen designs) and its bench (`report.md` = raw
verdicts + full tables, `results/` = run envelopes). A promoted line may
adopt these into its home — `project/<line>/bench/` for the bench and e.g.
`initial_proposal.md` for the founding proposal (directedit_ec and
sigma_lowres do both); lines that haven't keep them in `bench/<line>/`
and `docs/proposal/<line>*.md`.

Active projects:

- [`sigma_lowres/`](sigma_lowres/) — σ-conditional low-res gradient routing.
  Spectral mechanism refuted; one measured-safe route (1024→896 @ σ>0.5);
  discriminating probe (1280→1024) pending.
- [`directedit_ec/`](directedit_ec/) — EasyControl cond stream as a learned
  preservation prior for DirectEdit. Phases 0–1b passed zero-training;
  Phase 2.5 (delta-caption instruction editor) probe PASSED at the trained
  point → EasyEdit ship proposal + paper prep are the owed write-ups
  (neither the line's `outcomes.md` nor an `easyedit_comfy_node` proposal
  exists yet).
- [`cjk_aware_anima/`](cjk_aware_anima/) — native JA/CJK prompt conditioning
  via an extended T5-side vocab distilled against the EN-translation teacher.
  Probe + zero-shot ext vocab measured (`bench/cjk_adapter/`); the Phase 2a
  data assets MT cannot produce are built ([`datasets/`](cjk_aware_anima/datasets/)
  — Wikidata proper-noun lexicon, native-register manga eval set); the
  distillation loop itself is unstarted. Split three ways instead of a single
  founding proposal: [`motivation.md`](cjk_aware_anima/motivation.md) (why,
  incl. the directions already ruled out),
  [`done.md`](cjk_aware_anima/done.md) (completed-item checklist),
  [`plan.md`](cjk_aware_anima/plan.md) (Phase 2 design, gates, deployment,
  risks). Measured numbers stay with the code:
  [`datasets/README.md`](cjk_aware_anima/datasets/README.md) and
  `bench/cjk_adapter/results/`.
