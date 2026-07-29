"""Client-side verbs for ``python -m anima_daemon`` — submit / wait / status.

The daemon's HTTP surface was fully capable of "run this argv on the GPU queue"
long before anything on the command line could ask for it: submitting an
arbitrary command job meant writing a Python snippet against
``DaemonClient.submit_command``. These three verbs are that missing front door,
kept in the daemon package (rather than ``scripts/tasks/``) so they work from a
bare checkout, a vendored node tree, or an agent shell — no ``tasks.py`` import,
no ``library.*``, stdlib only.

    python -m anima_daemon submit [--label L] [--stall-timeout S] [--wait]
                                  [--hold] -- <argv…>
    python -m anima_daemon wait <job_id> [--timeout S]
    python -m anima_daemon status [job_id]

``make daemon-run`` (``scripts/tasks/daemon.py``) is the repo-flavored wrapper
over ``submit`` with attach-by-default streaming; this module is the plumbing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from . import client as _client
from . import config

VERBS = ("submit", "wait", "status")


def _label_for(argv: list[str]) -> str:
    """Derive a display label from the child argv: the script/module basename.

    ``["project/x/bench/run_pair_census.py", "--limit", "5"]`` → ``run_pair_census``;
    ``["-m", "scripts.distill_turbo.distill"]`` → ``distill``. An inline
    ``python -c <src>`` has no name to take, so it stays ``command`` rather than
    becoming a slice of source code.
    """
    for i, tok in enumerate(argv):
        if tok == "-m" and i + 1 < len(argv):
            return argv[i + 1].rsplit(".", 1)[-1]
        if tok == "-c":
            return "command"
        if tok.startswith("-"):
            continue
        return Path(tok).stem or tok
    return "command"


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2), flush=True)


def _result_envelope(record: dict) -> Optional[dict]:
    """The bench ``result.json`` a finished job lifted, if any (§ result-lift)."""
    path = record.get("result_path")
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _exit_code_for(record: dict) -> int:
    """The job's exit code: its OS ``returncode`` when known, else state-derived.
    A signal death (negative rc) maps to the conventional ``128+N`` so ``&&``
    chains behave as they would for an inline run."""
    rc = record.get("returncode")
    if isinstance(rc, int):
        return 128 + (-rc) if rc < 0 else rc
    return 0 if record.get("state") == "done" else 1


def cmd_submit(args: argparse.Namespace) -> int:
    argv = list(args.argv or [])
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print(
            "nothing to run: pass the child argv after `--`, e.g.\n"
            "  python -m anima_daemon submit -- project/x/bench/run_probe.py --limit 5",
            file=sys.stderr,
        )
        return 2
    label = args.label or _label_for(argv)
    cl = _client.ensure_daemon(expected_root=config.ROOT)
    resp = cl.submit_command(
        label=label,
        argv=argv,
        stall_timeout=args.stall_timeout,
        # `--hold` stages the job behind a paused gate; the default leaves the
        # gate alone so it runs when it reaches the front of the queue. Note this
        # is NOT `make …--queue`, which means "don't attach" — submitting here
        # never attaches, so returning immediately is already the default.
        start=False if args.hold else None,
    )
    job_id = resp.get("job_id")
    if not args.wait:
        _print_json({"job_id": job_id, "state": resp.get("state"), "base_url": cl.base})
        return 0
    return _wait_and_report(cl, job_id, timeout=args.timeout)


def cmd_wait(args: argparse.Namespace) -> int:
    cl = _client.DaemonClient()
    return _wait_and_report(cl, args.job_id, timeout=args.timeout)


def _wait_and_report(cl, job_id: str, *, timeout: Optional[float]) -> int:
    """Block on the job, print its final record (+ lifted result envelope), and
    return its exit code — ``124`` on wait timeout (matching ``timeout(1)``)."""
    try:
        record = cl.wait(job_id, timeout=timeout)
    except LookupError as e:
        print(str(e), file=sys.stderr)
        return 2
    except TimeoutError as e:
        print(str(e), file=sys.stderr)
        return 124
    except KeyboardInterrupt:
        print(f"\ndetached (job {job_id} continues).", file=sys.stderr)
        return 130
    out = {
        "job_id": job_id,
        "state": record.get("state"),
        "returncode": record.get("returncode"),
        "error": record.get("error"),
        "ckpt_path": record.get("ckpt_path"),
        "result_path": record.get("result_path"),
        "result_summary": record.get("result_summary"),
        "stdout_path": record.get("stdout_path"),
    }
    envelope = _result_envelope(record)
    if envelope is not None:
        out["result"] = envelope
    _print_json(out)
    return _exit_code_for(record)


def cmd_status(args: argparse.Namespace) -> int:
    cl = _client.DaemonClient()
    if args.job_id:
        record = cl.job_record(args.job_id)
        if record is None:
            print(json.dumps({"error": "no such job", "job_id": args.job_id}))
            return 2
        envelope = _result_envelope(record)
        if envelope is not None:
            record = {**record, "result": envelope}
        _print_json(record)
        return 0
    health = cl.health()
    if health is None:
        _print_json({"up": False, "base_url": None})
        return 1
    _print_json({"up": True, "base_url": cl.base, **health})
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m anima_daemon",
        description="Submit / wait on / inspect jobs on the local anima daemon.",
    )
    sub = p.add_subparsers(dest="verb", required=True)

    s = sub.add_parser("submit", help="enqueue a command job (`-- <argv…>`)")
    s.add_argument("--label", help="display label (default: script/module basename)")
    s.add_argument(
        "--stall-timeout",
        type=float,
        default=None,
        dest="stall_timeout",
        help="per-job stall-watchdog budget in seconds; 0 disables it "
        "(default: the daemon's 120s command-job budget)",
    )
    s.add_argument(
        "--hold",
        action="store_true",
        help="stage the job behind a paused queue gate (Start Queue releases it) "
        "instead of letting it run when it reaches the front",
    )
    s.add_argument(
        "--wait", action="store_true", help="block until the job is terminal"
    )
    s.add_argument("--timeout", type=float, default=None, help="--wait timeout (s)")
    s.add_argument("argv", nargs=argparse.REMAINDER, help="`-- <child argv…>`")
    s.set_defaults(func=cmd_submit)

    w = sub.add_parser("wait", help="block until a job is terminal; print its record")
    w.add_argument("job_id")
    w.add_argument(
        "--timeout", type=float, default=None, help="give up after S seconds"
    )
    w.set_defaults(func=cmd_wait)

    st = sub.add_parser("status", help="daemon health, or one job record")
    st.add_argument("job_id", nargs="?", help="omit for daemon-level health")
    st.set_defaults(func=cmd_status)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)
