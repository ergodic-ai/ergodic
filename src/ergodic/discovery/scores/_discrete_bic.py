# SPDX-License-Identifier: Apache-2.0
"""The multinomial BIC: one contingency table per query, read off the codes.

A discrete node given discrete parents is a multinomial per parent
configuration, and its maximized log-likelihood depends on the data only
through the cell counts: ``sum_jk N_jk log(N_jk / N_j)`` over parent
configurations ``j`` and child levels ``k``. A query flattens the parent code
columns into one configuration index (``ravel_multi_index``), tallies every
cell with a single ``bincount``, and reads the likelihood off the table, so
no Python loop ever touches a row; empty cells contribute zero. The
complexity term charges ``(r - 1) q`` free parameters for ``r`` child levels
under ``q`` parent configurations (``q = 1`` with no parents), giving
``local = loglik - penalty * ((r - 1) q / 2) log n``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .._score import Score

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...data import TabularData


class DiscreteBIC(Score):
    """The multinomial BIC, decomposable and score-equivalent.

    Every column must be discrete (categorical, ordinal, or binary): the
    score runs on integer codes, so binding to data with a continuous column
    raises ``TypeError``. Parameter counts agree across a Markov equivalence
    class, so Markov-equivalent DAGs get equal totals, exactly as with the
    Gaussian ``BIC``.

    Args:
        data: Bind at construction, or later through ``bind``.
        penalty: A multiplier on the complexity term ``((r - 1) q / 2) log n``.
            ``1.0`` is the textbook BIC; smaller values prune less.
    """

    name: ClassVar[str] = "bic_discrete"
    needs: ClassVar[str] = "codes"
    decomposable: ClassVar[bool] = True
    score_equivalent: ClassVar[bool] = True

    def __init__(self, data: TabularData | None = None, *, penalty: float = 1.0) -> None:
        if penalty <= 0.0:
            raise ValueError(f"penalty must be positive, got {penalty}")
        self._penalty = float(penalty)
        super().__init__(data)

    @property
    def penalty(self) -> float:
        """The multiplier on the complexity term."""
        return self._penalty

    def local_from_codes(
        self,
        y: npt.NDArray[np.int64],
        y_card: int,
        zs: tuple[npt.NDArray[np.int64], ...],
        z_cards: tuple[int, ...],
    ) -> float:
        """The multinomial local from integer codes; see the module docstring."""
        n = int(y.shape[0])
        if n == 0:
            raise ValueError("DiscreteBIC needs at least one row")
        r = int(y_card)
        q = 1
        for card in z_cards:
            q *= int(card)
        flat = np.ravel_multi_index((*zs, y), (*z_cards, r))
        counts = np.bincount(flat, minlength=q * r).reshape(q, r)
        totals = np.broadcast_to(counts.sum(axis=1, keepdims=True), counts.shape)
        observed = counts > 0
        cells = counts[observed].astype(np.float64)
        rows = totals[observed].astype(np.float64)
        loglik = float(np.sum(cells * (np.log(cells) - np.log(rows))))
        complexity = self._penalty * ((r - 1) * q / 2.0) * math.log(n)
        return loglik - complexity
