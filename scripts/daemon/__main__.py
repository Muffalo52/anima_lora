"""DEPRECATED entry point — forwards ``python -m scripts.daemon`` to
``anima_daemon.__main__``. Kept so muscle memory / older spawn strings still
boot the current daemon. New callers should use ``python -m anima_daemon``.
"""

from __future__ import annotations

from anima_daemon.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
