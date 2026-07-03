"""DEPRECATED path — exec-forward to ``anima_daemon/mcp.py``.

The daemon moved to ``anima_daemon/`` (2026-07-03), but existing MCP
registrations point at this file path absolutely::

    claude mcp add anima-daemon -- <repo>/.venv/bin/python <repo>/scripts/daemon/mcp.py

so it must keep working. This shim adds the repo root to ``sys.path`` and hands
off to the real stdio bridge. New registrations should target
``<repo>/anima_daemon/mcp.py`` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/daemon/mcp.py -> parents[2] == repo root (anima_lora/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anima_daemon.mcp import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
