# project/ — active promoted lines

One subdir per research line that has graduated past "proposal + bench report"
into an ongoing project with open phases. Each subdir is the line's home page:

| File | Contents |
|---|---|
| `methods.md` | The implementation — what code exists, where it lives, how to run it |
| `bench.md` | Digest of measured results (canonical data stays in `bench/<line>/`) |
| `questions.md` | Open questions the line has not answered |
| `roadmap.md` | Remaining phases, gates, and kill criteria |

Canonical sources these digest (never duplicated wholesale):
`docs/proposal/<line>*.md` (frozen designs), `bench/<line>/report.md`
(raw verdicts + full tables), `bench/<line>/results/` (run envelopes).

Active projects:

- [`sigma_lowres/`](sigma_lowres/) — σ-conditional low-res gradient routing.
  Spectral mechanism refuted; one measured-safe route (1024→896 @ σ>0.5);
  discriminating probe (1280→1024) pending.
- [`directedit_ec/`](directedit_ec/) — EasyControl cond stream as a learned
  preservation prior for DirectEdit. Phases 0–1b passed zero-training;
  Phase 2 (cross-image subject descriptor) unblocked.
