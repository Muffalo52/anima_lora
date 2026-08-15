"""Shared paths + constants for the local training daemon.

Single localhost process; one job at a time. State lives under
``output/daemon/`` so it sits beside the checkpoints the jobs produce and is
already covered by the repo's ``output/`` gitignore.

    output/daemon/
      daemon.json           pidfile: {"pid", "create_time", "port", "root",
                            "fingerprint"}
      daemon.log            the detached daemon's stdout/stderr
      jobs/<job_id>/
        job.json            persisted Job record (survives daemon restart)
        stdout.log          the training subprocess's captured stdout+stderr
        progress.jsonl      the Phase-0 structured progress stream (we point
                            train.py's --progress_jsonl here)
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

# anima_daemon/config.py -> parents[1] == anima_lora/
ROOT = Path(__file__).resolve().parents[1]

STATE_DIR = ROOT / "output" / "daemon"
JOBS_DIR = STATE_DIR / "jobs"
PIDFILE = STATE_DIR / "daemon.json"
DAEMON_LOG = STATE_DIR / "daemon.log"

# Fixed localhost port — the daemon binds 127.0.0.1 only (non-goal: remote /
# auth). Overridable for tests / odd setups via env.
DEFAULT_PORT = int(os.environ.get("ANIMA_DAEMON_PORT", "8765"))
HOST = "127.0.0.1"

# Pre-launch GPU guard (see manager._gpu_guard). Loose by default: on a serial
# single-GPU queue the guard only must catch VRAM leaked by our own dead jobs
# (reaped by pid, threshold-independent); busy_frac is just a heuristic for "some
# *other* process owns the card", kept high so a loaded ComfyUI/browser/desktop
# doesn't stall every launch. busy_frac = used/total above which the card is busy;
# retries/delay = how long to wait for it to free before launching anyway.
GPU_GUARD_BUSY_FRAC = float(os.environ.get("ANIMA_DAEMON_GPU_BUSY_FRAC", "0.85"))
GPU_GUARD_RETRIES = int(os.environ.get("ANIMA_DAEMON_GPU_RETRIES", "1"))
GPU_GUARD_DELAY = float(os.environ.get("ANIMA_DAEMON_GPU_DELAY", "2.0"))

# Stall watchdog: kill a *running* job whose output (stdout.log + progress.jsonl)
# hasn't advanced for this many seconds, finalizing as an error naming its last
# line — turns a wedged download / symlink-cycle walk / deadlock into a loud,
# queue-freeing failure instead of a hang. 0 disables (see manager._stall_reason).
# Only command jobs (preprocess/mask) are watched by default (post-load tqdm
# every ~10s → 120s silence = wedged). Training is unwatched (budget 0): its
# first step is legitimately silent for the whole torch.compile trace (can reach
# minutes), and a false kill mid-compile costs more than the rare hung run. Opt
# training in via ANIMA_DAEMON_JOB_STALL_TIMEOUT.
#
# Both are only *defaults*: a submitter that knows its loop goes quiet passes
# `stall_timeout` on POST /jobs (0 → unwatched) and that wins per job. The
# watchdog also cross-checks process-tree CPU time before firing, so a
# quiet-but-computing job isn't killed for being silent (manager._stall_reason).
CMD_STALL_TIMEOUT = float(os.environ.get("ANIMA_DAEMON_CMD_STALL_TIMEOUT", "120"))
JOB_STALL_TIMEOUT = float(os.environ.get("ANIMA_DAEMON_JOB_STALL_TIMEOUT", "0"))

# Job-dir retention (see jobs.prune_jobs). Nothing ever removed a finished job's
# directory, so `jobs/` grew without bound — a few hundred dirs and tens of MB of
# stdout.log within a couple of months. The daemon prunes *terminal* jobs older
# than RETENTION_DAYS at boot, always keeping the newest RETENTION_KEEP whatever
# their age (so a quiet week never empties the history you'd actually consult).
# Queued / running / paused dirs are never candidates. 0 days disables pruning.
JOB_RETENTION_DAYS = float(os.environ.get("ANIMA_DAEMON_JOB_RETENTION_DAYS", "30"))
JOB_RETENTION_KEEP = int(os.environ.get("ANIMA_DAEMON_JOB_RETENTION_KEEP", "200"))


# --- Phase 0a: daemon-source fingerprint ------------------------------------
# The daemon is disposable: a resident process serving last week's
# anima_daemon/*.py after an edit or pull is untrustworthy, so we detect the
# mismatch and eagerly restart (reconcile/adopt makes that lossless). The
# fingerprint keys on file *contents*, not git HEAD, because the common case is
# a dirty working tree mid-edit. A daemon records the fingerprint it BOOTED with
# in its pidfile + /health; a client re-computes the current on-disk value and
# compares (see client.daemon_is_stale).
_SRC_DIR = Path(__file__).resolve().parent


def source_fingerprint() -> str:
    """Short content hash of the daemon's own source (``anima_daemon/*.py``).

    Stable across runs (sorted by name), sensitive to any byte change in any
    module. Unreadable files are skipped rather than raising — a partial hash is
    still a useful staleness signal and must never crash a boot or a submit.
    """
    h = hashlib.sha256()
    for path in sorted(_SRC_DIR.glob("*.py")):
        try:
            data = path.read_bytes()
        except OSError:
            continue
        h.update(path.name.encode("utf-8"))
        h.update(b"\0")
        h.update(data)
        h.update(b"\0")
    return h.hexdigest()[:16]


# --- Phase 0b: submit-time env capture --------------------------------------
# A queued job must run with the SUBMITTER's environment, not the daemon's
# week-old boot env (a shell with a different CUDA_VISIBLE_DEVICES / ANIMA_DIT /
# HF_TOKEN would otherwise be silently ignored). The submit chokepoints snapshot
# this whitelist of prefixes into the job record; the daemon layers it under the
# job's explicit extra_env at spawn. PATH / VIRTUAL_ENV are deliberately NOT
# captured — the daemon resolves the venv interpreter itself and must keep doing
# so (a captured PATH from an odd shell would break that).
CAPTURED_ENV_PREFIXES = ("ANIMA_", "CUDA_", "HF_", "PYTORCH_", "TORCH_", "NCCL_")
# ANIMA_DAEMON_* configures the daemon process, not a job; capturing it into a
# job's env is noise at best and, for a job that re-submits, confusing. Drop it.
_CAPTURED_ENV_SKIP_PREFIXES = ("ANIMA_DAEMON_",)


def capture_env(source: dict | None = None) -> dict:
    """Snapshot the whitelisted env vars (see ``CAPTURED_ENV_PREFIXES``).

    Runs in the *submitting* process (CLI / GUI / node), so it captures that
    caller's shell — not the daemon's. ``source`` defaults to ``os.environ``.
    """
    env = os.environ if source is None else source
    return {
        k: v
        for k, v in env.items()
        if k.startswith(CAPTURED_ENV_PREFIXES)
        and not k.startswith(_CAPTURED_ENV_SKIP_PREFIXES)
    }


def ensure_state_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def global_pidfile() -> Path:
    """Stable, repo-independent pidfile location the daemon also mirrors to.

    The in-repo ``PIDFILE`` lives under *this checkout's* ``output/daemon/``; a
    ComfyUI node installed in a different directory tree (the published,
    standalone shape — this whole module is vendored into the node) can't
    compute that path. So the daemon mirrors its pidfile here too, at a fixed
    per-user location both sides derive without knowing each other's paths —
    letting the node discover a running daemon, and its possibly-ephemeral
    fallback port, on its own.

    Override with ``$ANIMA_DAEMON_PIDFILE`` (also honored by
    ``discover_pidfile``).
    """
    override = os.environ.get("ANIMA_DAEMON_PIDFILE")
    if override:
        return Path(override)
    return Path.home() / ".anima" / "daemon.json"


def discover_pidfile() -> Path:
    """Locate a running daemon's pidfile across install shapes.

    First existing candidate wins:

      1. the in-repo ``PIDFILE`` — authoritative when this module runs inside
         the anima_lora checkout (``parents[1]`` is the repo root);
      2. ``global_pidfile()`` — the per-user mirror (covers the
         ``$ANIMA_DAEMON_PIDFILE`` override and the standalone-node case where
         ``parents[1]`` points at the vendor dir and ``PIDFILE`` never exists);
      3. ``$ANIMA_LORA_ROOT/output/daemon/daemon.json`` — an explicit repo root;
      4. an upward search from this file for a repo root (``train.py`` +
         ``configs/``), then its ``output/daemon/daemon.json``.

    Falls back to ``PIDFILE`` (nonexistent → the client uses ``DEFAULT_PORT`` /
    ``$ANIMA_DAEMON_PORT``) when nothing is found.
    """
    if PIDFILE.exists():
        return PIDFILE

    mirror = global_pidfile()
    if mirror.exists():
        return mirror

    env_root = os.environ.get("ANIMA_LORA_ROOT")
    if env_root:
        cand = Path(env_root) / "output" / "daemon" / "daemon.json"
        if cand.exists():
            return cand

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "train.py").is_file() and (parent / "configs").is_dir():
            cand = parent / "output" / "daemon" / "daemon.json"
            if cand.exists():
                return cand
            break

    return PIDFILE
