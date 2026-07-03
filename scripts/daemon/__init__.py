"""DEPRECATED location — the daemon moved to ``anima_daemon/`` (2026-07-03).

Thin compatibility shim kept for a deprecation window: external MCP
registrations, `import scripts.daemon.client` callers, and vendored trees that
predate the move keep working. It aliases every ``scripts.daemon.<mod>`` name to
the **same module object** as ``anima_daemon.<mod>`` (via ``sys.modules``), so
identity-sensitive uses — ``isinstance`` checks, test monkeypatching of module
globals — behave identically whether imported through the old or new path.

New code should import ``anima_daemon`` directly. ``scripts/daemon/mcp.py``
survives as an exec-forward (users' MCP configs point at that file path); this
package also forwards ``python -m scripts.daemon`` via ``__main__``.
"""

from __future__ import annotations

import importlib
import sys

# Submodules to alias. mcp is intentionally NOT aliased here: the physical
# scripts/daemon/mcp.py must stay runnable as a script path (`python
# scripts/daemon/mcp.py`), and it forwards to anima_daemon.mcp itself.
_SUBMODULES = (
    "config",
    "proc",
    "gpu",
    "tail",
    "jobs",
    "client",
    "server",
    "manager",
)

for _name in _SUBMODULES:
    _mod = importlib.import_module(f"anima_daemon.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _mod
    setattr(sys.modules[__name__], _name, _mod)

del _name, _mod
