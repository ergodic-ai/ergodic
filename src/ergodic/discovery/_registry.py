# SPDX-License-Identifier: Apache-2.0
"""Resolve a method name into a discovery algorithm class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .algorithms._astar import AStar
from .algorithms._pc import PC

if TYPE_CHECKING:
    from typing import Any

_BY_METHOD: dict[str, type[Any]] = {
    "pc": PC,
    "pc_stable": PC,
    "astar": AStar,
    "a_star": AStar,
    "exact": AStar,
}


def algorithm_for(method: str) -> type[Any]:
    """The algorithm class for a method name, with a readable error."""
    cls = _BY_METHOD.get(method.lower())
    if cls is None:
        allowed = ", ".join(sorted(_BY_METHOD))
        raise ValueError(f"unknown discovery method {method!r}; use one of {allowed}")
    return cls
