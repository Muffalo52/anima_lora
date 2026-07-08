"""CLI surface for the local training daemon (``make daemon*``).

Four verbs, mapped to the lifecycle guarantees in ``plan.md`` Phase 1:

    daemon            start (idempotent — no-op if already up), wait /health
    daemon-attach     non-owning viewer; ctrl-C detaches only, training lives on
    daemon-kill       abort the running (or JOB=<id>) job, free GPU; daemon stays up
    daemon-terminate  shut the whole daemon down (active job dies too)

``daemon`` starts the daemon **console-detached** (see ``proc.spawn_detached``),
so the terminal's SIGINT reaches only the foreground group, never the daemon.
``daemon-attach`` is the parent of nothing, so its ctrl-C can't touch training.
Both teardown verbs verify the pidfile's ``(pid, create_time)`` before acting so
they never touch a PID-reused stranger.
"""

from __future__ import annotations

import json
import os
import sys

from anima_daemon import client as _client
from anima_daemon import config as _cfg
from anima_daemon import proc as _proc


def _job_arg(extra) -> str | None:
    """Resolve a job id from ``JOB=<id>`` env or the first positional arg."""
    job = os.environ.get("JOB")
    if not job and extra and not extra[0].startswith("-"):
        job = extra[0]
    return job or None


def cmd_daemon(extra):
    """Start the training daemon (idempotent). Detached + waits for /health."""
    existing = _proc.daemon_alive(_cfg.PIDFILE)
    if existing is not None:
        print(
            f"daemon already running (pid {existing.get('pid')}, "
            f"port {existing.get('port')})."
        )
        return
    try:
        cl = _client.ensure_daemon()
    except RuntimeError as e:
        print(f"failed to start daemon: {e}", file=sys.stderr)
        sys.exit(1)
    health = cl.health() or {}
    print(
        f"daemon up on {cl.base} (pid {health.get('pid')}). "
        f"Logs: {_cfg.DAEMON_LOG}\n"
        "  make daemon-attach        # follow events\n"
        "  make daemon-kill          # abort the running job\n"
        "  make daemon-terminate     # stop the daemon"
    )


# Compact per-job view daemon-status prints by default. Full records run ~1KB each
# and the history grows unboundedly — a polling agent needs "what's queued/running/
# failed", not the whole record (GET /jobs/{id} has that).
_STATUS_JOB_FIELDS = (
    "id",
    "method",
    "kind",
    "preset",
    "state",
    "submitted_at",
    "started_at",
    "ended_at",
    "error",
    "ckpt_path",
    "chained_job_id",
)

# Default job cap for the compact view — the full history grows unboundedly, so
# a bare `daemon-status` shows only the most-recent slice (newest first). `--all`
# lifts the cap; `--limit N` sets it. `jobs_total`/`jobs_shown` in the output
# report the truncation so a capped view never reads as "that's everything".
_STATUS_DEFAULT_LIMIT = 15

# Shorthand state groups for `--running` / `--failed` / `--done`.
_STATUS_ACTIVE_STATES = frozenset({"running", "paused"})
_STATUS_FAILED_STATES = frozenset({"error", "stopped"})


def _parse_status_flags(extra):
    """Parse the ``daemon-status`` filter flags out of ``extra``.

    ``--full`` (raw records) · ``--all`` (no cap) · ``--limit N`` ·
    ``--state s[,s]`` · ``--running``/``--active`` · ``--failed`` · ``--done``.
    Unknown tokens are ignored (forward-compatible with the make ARGS shim).
    """
    extra = list(extra or [])
    opts = {"full": False, "all": False, "states": None, "limit": _STATUS_DEFAULT_LIMIT}
    i = 0
    while i < len(extra):
        a = extra[i]
        if a == "--full":
            opts["full"] = True
        elif a == "--all":
            opts["all"] = True
        elif a in ("--running", "--active"):
            opts["states"] = set(_STATUS_ACTIVE_STATES)
        elif a == "--failed":
            opts["states"] = set(_STATUS_FAILED_STATES)
        elif a == "--done":
            opts["states"] = {"done"}
        elif a == "--state" and i + 1 < len(extra):
            i += 1
            opts["states"] = {s.strip() for s in extra[i].split(",") if s.strip()}
        elif a == "--limit" and i + 1 < len(extra):
            i += 1
            try:
                opts["limit"] = int(extra[i])
            except ValueError:
                pass
        i += 1
    return opts


def _job_target(job: dict) -> str | None:
    """Best-effort label for *what* a job operates on — the missing piece when
    skimming the queue. Command jobs (soup/preprocess/distill) carry it in
    ``argv`` (``--name`` / ``--path_pattern``, else the ``-m`` module); train jobs
    carry it as the ``output_name`` override, else fall back to ``method``."""
    argv = job.get("argv") or []
    if job.get("kind") == "command":
        for flag in ("--name", "--path_pattern", "--output_name"):
            if flag in argv:
                i = argv.index(flag)
                if i + 1 < len(argv):
                    return argv[i + 1]
        if "-m" in argv:
            i = argv.index("-m")
            if i + 1 < len(argv):
                return argv[i + 1]
        return argv[0] if argv else None
    overrides = job.get("overrides") or {}
    return overrides.get("output_name") or job.get("method")


def cmd_daemon_status(extra):
    """Daemon status as one JSON object on stdout — the agent/script surface.

    ``{"up", "base_url", "pid", "port", "root", "stale_code", "paused",
    "active_job", "jobs_total", "jobs_shown", "jobs"}``. Passive: never starts a
    daemon (safe to poll); ``up: false`` + exit 1 when nothing answers
    ``/health``. ``base_url`` is resolved from the pidfile each call, so it
    follows a fallback-to-ephemeral port — read it from here rather than assuming
    8765. ``stale_code: true`` means the resident daemon is serving source older
    than the current on-disk ``anima_daemon/*`` — the next submit will eagerly
    restart it (Phase 0a).

    Jobs are compact summaries (id/state/error/ckpt_path/… plus a derived
    ``target`` = what the job operates on), **newest first** and capped to the
    most-recent ``_STATUS_DEFAULT_LIMIT``; ``jobs_total`` vs ``jobs_shown`` report
    any truncation. Filter flags: ``--full`` (raw records) · ``--all`` (no cap) ·
    ``--limit N`` · ``--state s[,s]`` · ``--running`` · ``--failed`` · ``--done``.
    """
    opts = _parse_status_flags(extra)
    cl = _client.DaemonClient()
    health = cl.health()
    if health is None:
        print(json.dumps({"up": False, "base_url": None, "jobs": []}))
        sys.exit(1)
    jobs = cl.list_jobs()
    jobs.sort(key=lambda j: j.get("submitted_at") or 0, reverse=True)  # newest first
    jobs_total = len(jobs)
    if opts["states"] is not None:
        jobs = [j for j in jobs if j.get("state") in opts["states"]]
    if not opts["all"] and opts["limit"] is not None and opts["limit"] >= 0:
        jobs = jobs[: opts["limit"]]
    if opts["full"]:
        out_jobs = [{**j, "target": _job_target(j)} for j in jobs]
    else:
        out_jobs = [
            {**{k: j.get(k) for k in _STATUS_JOB_FIELDS}, "target": _job_target(j)}
            for j in jobs
        ]
    print(
        json.dumps(
            {
                "up": True,
                "base_url": cl.base,
                "pid": health.get("pid"),
                "port": health.get("port"),
                "root": health.get("root"),
                "stale_code": _client.daemon_is_stale(health),
                "paused": health.get("paused"),
                "active_job": health.get("active_job"),
                "jobs_total": jobs_total,
                "jobs_shown": len(out_jobs),
                "jobs": out_jobs,
            },
            indent=2,
        )
    )


def cmd_daemon_attach(extra):
    """Read-only viewer. ``JOB=<id>`` follows that job's stdout; otherwise the
    daemon event stream. Ctrl-C detaches this terminal only — never the daemon
    or the training subprocess (we are the parent of nothing)."""
    if not _client.is_running():
        print("no daemon; `make daemon` to start.", file=sys.stderr)
        sys.exit(1)
    cl = _client.DaemonClient()
    job = _job_arg(extra)
    stream = cl.stream_logs(job) if job else cl.stream_events()
    what = f"job {job}" if job else "daemon events"
    print(f"attached to {what} ({cl.base}) — ctrl-C to detach\n")
    try:
        for line in stream:
            print(line, flush=True)
    except KeyboardInterrupt:
        print("\ndetached (training continues).")
    except Exception as e:  # noqa: BLE001 — socket reset on daemon shutdown, etc.
        print(f"\nstream ended: {e}")


def cmd_daemon_kill(extra):
    """Abort a job; the daemon stays up and advances to the next queued job.
    ``JOB=<id>`` targets a specific job; otherwise the running one."""
    if not _client.is_running():
        print("no daemon running.", file=sys.stderr)
        sys.exit(1)
    cl = _client.DaemonClient()
    job = _job_arg(extra)
    result = cl.stop(job)
    if result.get("error"):
        print(result["error"], file=sys.stderr)
        sys.exit(1)
    print(f"job {result.get('job_id')} → {result.get('state')} (daemon still up).")


def cmd_daemon_pause(extra):
    """Freeze the running job's process tree (SIGSTOP) in place — VRAM stays put,
    resume is instant. ``JOB=<id>`` targets a specific job; otherwise the active
    one. The queue does not advance past a paused job (it still owns the card)."""
    if not _client.is_running():
        print("no daemon running.", file=sys.stderr)
        sys.exit(1)
    cl = _client.DaemonClient()
    result = cl.pause_job(_job_arg(extra))
    if result.get("error"):
        print(result["error"], file=sys.stderr)
        sys.exit(1)
    print(
        f"job {result.get('job_id')} → {result.get('state')} (frozen; VRAM held). "
        f"`make daemon-resume` to thaw."
    )


def cmd_daemon_resume(extra):
    """Thaw a paused job (SIGCONT) → back to running. ``JOB=<id>`` targets a
    specific job; otherwise the active (paused) one."""
    if not _client.is_running():
        print("no daemon running.", file=sys.stderr)
        sys.exit(1)
    cl = _client.DaemonClient()
    result = cl.resume_job(_job_arg(extra))
    if result.get("error"):
        print(result["error"], file=sys.stderr)
        sys.exit(1)
    print(f"job {result.get('job_id')} → {result.get('state')} (thawed).")


def cmd_daemon_terminate(extra):
    """Stop the whole daemon. The active job tree is killed and the GPU freed."""
    if not _client.is_running():
        print("no daemon running.", file=sys.stderr)
        return
    cl = _client.DaemonClient()
    cl.shutdown(kill_jobs=True)
    print("daemon terminated (active job killed, GPU freed, queue discarded).")
