# SPDX-License-Identifier: Apache-2.0
"""Edge endpoint marks for mixed causal graphs."""

from __future__ import annotations

from enum import Enum


class Mark(Enum):
    """A mark at one end of an edge.

    A mixed graph edge has a mark at each end. ``TAIL`` (drawn ``-``) and
    ``ARROW`` (drawn ``>``) give directed (``->``), bidirected (``<->``), and
    undirected (``--``) edges. ``CIRCLE`` (drawn ``o``) means the orientation is
    unknown, the marks an FCI search leaves on a PAG.
    """

    TAIL = "tail"
    ARROW = "arrow"
    CIRCLE = "circle"

    def char(self, *, left: bool) -> str:
        """Return the glyph for this mark on the ``left`` or right side of an edge."""
        if self is Mark.TAIL:
            return "-"
        if self is Mark.CIRCLE:
            return "o"
        return "<" if left else ">"
