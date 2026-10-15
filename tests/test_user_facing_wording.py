# SPDX-License-Identifier: Apache-2.0
"""User-facing strings state limits plainly, without internal planning words.

Regression (API-149): error messages and docstrings said "reserved seam", "not
built yet", "phase 1", or "MVP", which are internal project vocabulary. This reads
every string constant in the package (messages and docstrings, not comments) and
refuses those words.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import ergodic

_BANNED = re.compile(
    r"reserved seam|not built|\bMVP\b|post-MVP|\bphase[- ]1\b|later phase|engine spike"
    r"|causal layer lands|until it lands|\bE1\b|\bv1\b|lands in D2",
    re.IGNORECASE,
)


def _string_constants(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append((node.lineno, node.value))
    return out


def test_no_internal_planning_words_in_user_facing_strings() -> None:
    root = Path(ergodic.__file__).parent
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        for lineno, text in _string_constants(path):
            match = _BANNED.search(text)
            if match:
                offenders.append(f"{path.relative_to(root)}:{lineno}: {match.group(0)!r}")
    assert not offenders, "\n".join(offenders)
