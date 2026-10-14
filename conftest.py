"""Pytest bootstrap.

Makes the ``src`` layout importable without requiring an editable install,
so the test suite works from a plain checkout as well (``pytest`` only).
"""

from __future__ import annotations

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)
