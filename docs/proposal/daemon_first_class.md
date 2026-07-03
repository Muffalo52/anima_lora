# Daemon as the first-class run manager — disposable daemon, attach-by-default, bench as jobs

Status: **Phase 0 + repo relocation + Phase 1a + `make gen` (1c command-job chokepoint) + Phase 2a (pause/resume tree-freeze) shipped (2026-07-03); Phases 1b / 1c-rest / 2b / 2c remain proposals.**
Motivation: the daemon is architecturally
sound (serial GPU queue, disk-backed state, self-describing surface) but
socially opt-in — training grew `--queue`, preprocess/mask submit, and
everything else (`bench/**/run_bench.py`, `test-*`, `exp-*` loops, ad-hoc
inference) runs inline. The gap isn't the queue; it's that routing through the
daemon today costs terminal UX and carries two real staleness vectors, so
nobody routes anything they iterate on. This proposal removes the costs and
then migrates the remaining surfaces. Related: the ledger side of "job →
config → checkpoint → result in one record" is
[`sqlite_run_ledger.md`](sqlite_run_ledger.md) — this proposal produces the
`result_path` facts that ledger would index; neither depends on the other.

## What's already solved (don't rebuild)

The stale-*job* problem is handled, and it's the load-bearing enabler for
everything below:

- Jobs are identified as `(pid, create_time)`, never a bare pid
  (`scripts/daemon/jobs.py:70`), and monitored by liveness polling, not
  `Popen.wait`.
- On boot, `manager._reconcile()` (`scripts/daemon/manager.py:778`) walks
  `jobs/`, **re-attaches still-alive orphans** and marks dead ones
  `orphaned`. A daemon can die or be killed mid-run and the next daemon
  adopts the run losslessly.
- All state is flat files under `output/daemon/` — anything that can read
  files can observe a run with the HTTP port down (`gui/daemon.py` does).
- The daemon and its client are **stdlib-only** — no `library.*`, no torch.
  Boot is ~1s.

Consequence: **the daemon is safe to restart at any time.** That is the fact
this proposal leans on; the pieces below are corollaries.

## The two staleness vectors that remain

1. **Stale daemon code.** A daemon started last week serves last week's
   `scripts/daemon/*` after an edit or pull. During active development this
   is disqualifying — you can't trust a resident process while you're
   changing its code — so people bypass the daemon entirely.
2. **Stale environment.** `_build_cmd` spawns jobs with
   `env = os.environ.copy()` — the **daemon's boot env**
   (`scripts/daemon/manager.py:715`) — layering only the job's explicit
   `extra_env` (`manager.py:735`). A job submitted from a shell with a
   different `CUDA_VISIBLE_DEVICES` / `ANIMA_DIT` / `HF_TOKEN` silently runs
   with week-old values.

Plus two adoption blockers that aren't staleness:

3. **UX regression.** `--queue` means losing the terminal: no live tqdm, no
   Ctrl-C, no exit code for a wrapping script. `daemon-attach` exists
   (`scripts/tasks/daemon.py:113`) but is a separate manual step and doesn't
   propagate job outcome.
4. **No result awareness.** The daemon knows exit codes and
   `progress.jsonl`; it knows nothing about the bench envelope
   (`bench/_common.py::write_result` → `bench/<m>/results/<ts>/result.json`),
   so it can't serve as an experiment ledger and bench gains nothing by
   routing through it.

## The one load-bearing principle

**The daemon is disposable, not durable.** It is a throwaway view over disk
state, restarted eagerly whenever its code is stale. We never ask "is the
resident daemon trustworthy?" — we make the answer irrelevant by restarting
on mismatch, which reconcile/adopt already makes lossless. Corollary
constraint: the daemon stays stdlib-only forever. The moment it imports
`library.*` or holds a model, restarts stop being ~1s and staleness becomes
real again.

## Phase 0 — trust — ✅ SHIPPED 2026-07-03

Implemented as described below. Landed surface: `config.source_fingerprint()` /
`config.capture_env()`; `Job.captured_env` + `Job.returncode`; `/health` +
pidfile carry `fingerprint`; `ensure_daemon()` eager-restarts on a stale
fingerprint; `daemon-status` reports `stale_code`; `_build_cmd` layers
daemon-env ← captured_env ← extra_env; the `run_gpu` attach/detach/inline
chokepoint lives in `scripts/tasks/_common.py` (`_resolve_run_mode` /
`_attach_and_wait`) and `train()` routes through it (attach default). Covered by
`tests/test_daemon.py`. The `scripts/daemon/` → `anima_daemon/` relocation
(below) shipped as its own follow-on change (2026-07-03): package moved with git
history, `config.ROOT`/`_SRC_DIR`/`-m` launch retargeted, `gpu.no_window_kwargs`
inlined so the package is now **zero** `library` imports (enforced by
`test_anima_daemon_is_stdlib_only`), `scripts/daemon/` kept as a `sys.modules`
alias shim (+ `mcp.py` exec-forward, `__main__` forward) for old MCP
registrations / vendored trees, and vendor-sync / GUI / `update.py` / import
sites updated in lockstep.

### 0a. Code fingerprint + eager restart

- At boot, compute a fingerprint of the daemon's own source — content hash of
  `scripts/daemon/*.py` (not git HEAD: a dirty tree mid-edit is the common
  case) — and record it in the pidfile (`output/daemon/daemon.json`) and
  `/health`.
- `ensure_daemon()` (`scripts/daemon/client.py`) compares the running
  daemon's fingerprint against the on-disk source. On mismatch:
  `POST /shutdown {"kill_jobs": false}` → respawn → the new daemon's
  `_reconcile()` adopts the still-running job and the queued jobs persist on
  disk. Total cost ~1–2s, paid at most once per daemon-code change.
- Every submit path already goes through `ensure_daemon()`, so a stale daemon
  survives at most until the next submission. `daemon-status` should also
  report `stale_code: true` so a passive observer can see it.

### 0b. Submit-time env capture

- The submit chokepoints (`_queue_submit` / `queue_command` in
  `scripts/tasks/_common.py`, the client's `submit*`) snapshot a
  **whitelist** of the caller's env into the job record:
  `ANIMA_*`, `CUDA_*`, `HF_*`, `PYTORCH_*`, `TORCH_*`, `NCCL_*`. Explicitly
  *not* `PATH`/`VIRTUAL_ENV` — the daemon resolves the venv interpreter
  itself (`manager.py::_build_cmd`) and must keep doing so.
- Spawn layering becomes daemon-env ← captured-env ← explicit `extra_env`.
  A GUI submit captures the GUI's env; a shell submit captures the shell's.
  Kills vector 2 without letting a caller corrupt interpreter resolution.

### 0c. Attach-by-default (`run_gpu` chokepoint)

One helper in `scripts/tasks/_common.py`:

```python
run_gpu(spec, mode="attach")   # spec = train job | command job argv
```

- **Submits + attaches**: streams the job's stdout to the terminal (the SSE
  `/jobs/{id}/logs` path `daemon-attach` already uses; raw pass-through so
  tqdm `\r` redraws render live), then exits with the job's exit code.
- **Ctrl-C detaches** — the job survives (we are the parent of nothing, same
  contract as `daemon-attach` today). Print the `make daemon-attach JOB=…` /
  `make daemon-stop` one-liners on detach. Stopping is always an explicit
  second action, never a signal side effect.
- **Exit-code fidelity**: add `returncode` to the `Job` record
  (`jobs.py` — today only `state`/`error`/`detail` exist) and mirror it from
  the monitor loop. `run_gpu` exits with it, so bench harnesses, CI, and `&&`
  chains compose exactly as they do inline.
- **Modes**: `attach` (default), `detach` (today's `--queue`, kept as an
  alias), `inline` (escape hatch: run the child directly, no daemon —
  `ANIMA_RUN_MODE=inline` or `--inline`). Inline stays fully supported; it's
  the debugging path (pdb, py-spy, faulthandler all want a direct child).
- Attach must tolerate a daemon restart mid-stream (0a can cause one): on
  connection drop, re-resolve the pidfile and re-attach to the same job id;
  the on-disk `stdout.log` guarantees no lines are lost.

After Phase 0, `--queue` stops being a mode. Queueing is just what happens
when something else holds the card; when the queue is idle, submit+attach
feels like inline with a ~1s first-time daemon boot.

## Phase 1 — coverage (bench and tests become citizens)

### 1a. Result envelope lift (the hook, not a schema) — ✅ SHIPPED 2026-07-03

Implemented as described below. Landed surface: `_build_cmd` exports
`ANIMA_DAEMON_JOB_ID` + `ANIMA_DAEMON_JOB_DIR` into every job's env (train +
command kinds); `bench/_common.write_result` drops
`<job_dir>/result_path.json` → `{"path": <abs result.json>}` when
`ANIMA_DAEMON_JOB_DIR` is set (no-op inline); `Job` grew `result_path` +
`result_summary`; the monitor's `_finalize` calls `_lift_result` on every
terminal transition to follow the pointer and record the abs path + a
`{label, metrics}` digest. Best-effort + schema-blind (envelope stays
bench-owned). Covered by `tests/test_daemon.py` (Phase 1a section). 1b/1c
(the `make bench` dispatcher + `test-*` routing) remain proposals — migrate
opportunistically.

Mechanism — deliberately env-var-shaped so bench scripts stay standalone:

- The daemon exports `ANIMA_DAEMON_JOB_ID` + `ANIMA_DAEMON_JOB_DIR` into
  every job's env at spawn.
- `bench/_common.py::write_result` checks for `ANIMA_DAEMON_JOB_DIR`; when
  present it *additionally* writes
  `$ANIMA_DAEMON_JOB_DIR/result_path.json` → `{"path": "<abs run_dir>/result.json"}`.
  When absent (inline run) it's a no-op — zero coupling, bench scripts remain
  runnable as plain `python bench/<m>/run_bench.py`.
- On terminal state the monitor lifts the pointer plus the envelope's
  `label` / `metrics` summary into `job.json` (`result_path`,
  `result_summary` fields). `bench/<m>/results/` stays the canonical store —
  the daemon holds a pointer + digest, never the artifacts.

This also structurally fixes the bespoke-loop mirroring problem (turbo/spd/
mod loops re-implementing train.py's daemon niceties): outcome provenance
lives in the job record, written by the daemon, not by each loop.

### 1b. Bench migration

The migration is at the **launch layer, not inside bench scripts**. Scripts
keep their argparse + `sys.path` bootstrap + envelope contract unchanged.

- New dispatcher: `make bench ARGS="<method>/<script>.py --flags"` (task body
  `scripts/tasks/…` → `run_gpu(command_job(label=f"bench:{method}", argv=…))`).
  Attach by default, so a bench run feels identical to today — plus it queues
  behind (instead of OOM-colliding with) a live training run, survives the
  terminal closing, and lands in the ledger.
- Direct `python bench/<m>/run_bench.py` keeps working inline, forever. The
  `make bench` path is the *recommended* front door, not a gate.
- **Routing policy: only GPU-touching work goes through the daemon.**
  Pure-analysis scripts (CSV crunching, plot regeneration, probes over cached
  `.npz`) stay inline — queueing CPU work behind a 6-hour train run is
  strictly worse. The dispatcher takes `--cpu` (or the script self-declares
  via a `GPU = False` module flag `_common` can read) to bypass submission.
- Sweeps compose for free: a sweep driver submits N command jobs detached and
  the serial queue drains them — this is the pattern `_queue_submit`'s
  docstring already sells for artist sweeps, extended to bench.

### 1c. `test-*` inference targets

Route the `test-*` bodies (`scripts/tasks/inference.py`) through `run_gpu` as
command jobs, same attach semantics. The win is arbitration: today a `make
test` during training either OOMs or silently degrades the train run; queued,
it runs the moment the card frees. For deliberate concurrent use (tiny test
beside a training run that fits), `--inline` bypasses — the GPU guard
(`ANIMA_DAEMON_GPU_BUSY_FRAC`) is the daemon's own launch gate, not a global
lock on the card.

### 1c partial — `make gen` (batch generation) — ✅ SHIPPED 2026-07-03

The generic command-job chokepoint the `test-*` migration needs shipped first,
via the highest-demand surface (batch generation): `scripts/tasks/_common.py::
run_command(label, argv, mode=…)` — `run_gpu(command_job(…))` for non-train GPU
work, attach-by-default with the same `_resolve_run_mode` three modes as
`train`. `make gen` routes `inference.py` through it (shares `_base_test_args`,
so NOLORA/SPECTRUM/MOD/DAVE/FSG compose identically to `make test`); the
generation side of the result-lift is `inference.py`'s `write_gen_manifest`
(dropped a `gen_manifest.json` + pointer under `ANIMA_DAEMON_JOB_DIR`, no-op
inline). Covered by `tests/test_gen_manifest.py` + an end-to-end lift test in
`tests/test_daemon.py`. The remaining `test-*` bodies can now migrate by swapping
their `run([...])` for `run_command("test:<x>", argv)` opportunistically.

## Phase 2 — daemon owns the GPU: pause/resume, pipelines, resident server

The daemon-lifecycle bucket. Note the two halves have **different gates**: 2a
(pause/resume) is independent of Phase 1 adoption and similarly small — ship it
whenever; 2b/2c (pipelines + resident server) wait until Phase 1 demonstrates
people actually leave runs queued.

### 2a. Pause/resume a running job (tree-freeze) — ✅ SHIPPED 2026-07-03

Implemented as described below. Landed surface: `proc.suspend_tree` /
`proc.resume_tree` (psutil SIGSTOP/SIGCONT, parent-first freeze / children-first
thaw, beside `kill_tree`); `jobs.STATE_PAUSED` + `ACTIVE_STATES` frozenset +
`Job.paused_at` / `Job.accelerate_launched`; `manager.pause_job` / `resume_job`
(refuse non-running + accelerate-launch runs); the monitor loop skips the stall
watchdog while `paused`; `_current_running_locked` / `_queue_is_idle_locked` /
`_reconcile` treat paused as an active, queue-blocking, re-adoptable state;
`stop`/`_kill_job_tree` thaw-then-kill a frozen tree; `POST /jobs/{id}/pause` +
`/resume` endpoints + `pause_job`/`resume_job` TOOLS entries (auto-exposed over
MCP) + `DaemonClient.pause_job`/`resume_job`; `make daemon-pause` /
`daemon-resume` CLI verbs (`JOB=<id>` or the active job). Covered by
`tests/test_daemon.py` (Phase 2a section). 2b/2c remain proposals.

`POST /jobs/{id}/pause` / `POST /jobs/{id}/resume`, new `paused` job state
(`running ↔ paused` only). Mechanism: **suspend the process tree** —
`psutil.Process.suspend()/.resume()` (SIGSTOP/SIGCONT on Linux,
NtSuspendProcess on Windows; psutil is already the daemon's process layer,
`proc.py:21`). Implementation mirrors `kill_tree` (`proc.py:101`): suspend
parent first (so it can't spawn new children mid-freeze), then children —
dataloader workers included; resume in reverse order.

Why this level and not a cooperative flag: the freeze is **method-agnostic
and zero-cooperation** — it works identically on `train.py`, the bespoke
turbo/spd/mod loops, bench scripts, and inference, with no per-loop wiring
(the same mirroring tax this proposal keeps dodging). The CUDA context
survives, VRAM stays allocated, SM utilization drops to zero; resume is
instant — no reload, no recompile, mid-step optimizer state intact.

Semantics that must hold:

- **The queue does NOT advance past a paused job.** It still owns its VRAM,
  and the GPU guard launches anyway after `ANIMA_DAEMON_GPU_RETRIES` — the
  next job would OOM into the frozen allocation. The worker stays blocked on
  the paused job; `pause` is "hold my run", not "yield my slot".
- **Opportunistic side-runs are allowed *around* the daemon, not through
  it**: a paused train run holds only its allocated VRAM, so a small inline
  (`--inline`) inference fits in the remainder — poor-man's preemption
  without eviction. The daemon doesn't schedule into the gap; the human does.
- **`stale_for` is pause-aware** — the monitor freezes the staleness clock
  (and `daemon-status`/GUI show `paused`), otherwise every observer flags a
  wedged run. Wall-clock throughput/ETA metrics inside the run will blip
  across a pause; accepted, not compensated.
- **Single-process runs only.** The default direct-invoke path freezes
  cleanly; a multi-GPU `accelerate launch` run would trip NCCL heartbeat
  timeouts while frozen — refuse pause when the job was launched with
  `ANIMA_ACCELERATE_LAUNCH=1`.
- Pausing a `queued` job is just `queue/pause`-lite (skip it when picking the
  next job); the interesting case is `running`.

Deferred escalation — **cooperative pause (frees VRAM)**: signal `train.py`
to `save_state` + exit, resume later as a fresh job via `--resume`
(suspend-to-disk: durable across reboots, frees the card fully). That's
per-loop cooperation — exactly the mirroring tax — and a separate feature;
spec it only if the freeze level proves insufficient in practice.

### 2b. Pipelines — generalize `chain_train` → `chain`

A list of follow-on job specs run on success (linear only; no DAG until a real
need shows up). Unlocks `preprocess → train → test-grid → bench` living in the
daemon and surviving the submitter. `chain_train` becomes sugar for a
one-element chain.

### 2c. Fold the resident inference server under the daemon

Bring `scripts/inference_server.py` under the daemon as a managed resident:
daemon starts/stops it, and the current best-effort eviction
(`manager.py::_evict_resident_inference`) becomes an owned lifecycle transition
instead of a cross-pidfile courtesy. One process owns GPU policy.

## Repo location — `scripts/daemon/` → `anima_daemon/` — ✅ SHIPPED 2026-07-03

Promoting the package out of `scripts/` was worth doing, and never as `daemon/`:

- **Why**: the layering contract files `scripts/` as the entry-point tier;
  a service subsystem that everything routes through belongs at top level
  (same precedent as `bench/_anima.py` → `library/runtime/harness.py`). The
  move also makes the stdlib-only invariant a *testable boundary* — add a
  test asserting nothing under the package imports `library`/`networks`/
  torch — instead of a docstring promise.
- **What it does NOT buy**: importability (`pyproject.toml` already installs
  `scripts*`, so `scripts.daemon.client` imports from any cwd) or
  discoverability (HTTP + pidfile are location-independent).
- **Name**: `anima_daemon/`, never top-level `daemon/` — an installed
  `daemon` package collides with PyPI `python-daemon`'s import name.
- **Compat shim**: `scripts/daemon/` stays for a deprecation window as a
  thin re-export package, and `scripts/daemon/mcp.py` as an exec-forward —
  users' MCP registrations point at that path absolutely, and the trainer
  node resolves live-tree-then-vendor. Update `sync_vendor.py`'s mapping
  (`_vendor/scripts/daemon/` → `_vendor/anima_daemon/`), `scripts/update.py`,
  and the GUI import sites in the same change.

## Risks / non-goals

- **Not a scheduler.** The queue stays serial, one card, localhost. Multi-GPU
  lanes, priorities, remote submission — out of scope until the single-lane
  model actually pinches.
- **Windows**: attach/pass-through must be tested there (the daemon already
  handles pythonw/venv re-exec quirks in `_build_cmd`); nothing in Phase 0
  is POSIX-specific, but tqdm `\r` rendering through the SSE tail needs a
  check.
- **Interactive debugging** is explicitly served by `--inline`, not by
  making the daemon path debugger-friendly. Don't try.
- **Envelope schema stays bench-owned.** The daemon lifts `label`/`metrics`
  opaquely; it never validates or versions the envelope — that contract
  belongs to `bench/_common.py`.

## Sequencing

Phase 0 ✅ **done** — it was the actual unlock: fingerprint (~30 lines), env
capture (~15), `run_gpu` + `returncode` (mostly recombining the old
`_queue_submit` + `daemon-attach` + a poll loop). Phase 1 is adoption, not
architecture — do 1a
first (it's ~20 lines across `manager.py` + `bench/_common.py`), then migrate
bench dirs opportunistically as they're next touched, rather than in one
sweep. Phase 2a (pause) is independent of Phase 1 and similarly small —
`suspend_tree`/`resume_tree` beside `kill_tree`, one state, two endpoints — so
it can land any time. Phase 2b/2c (pipelines + resident server) wait until
Phase 1 demonstrates people actually leave runs queued.
