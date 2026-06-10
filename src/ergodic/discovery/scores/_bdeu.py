# SPDX-License-Identifier: Apache-2.0
"""The BDeu score: a Dirichlet marginal likelihood, uniform so it stays equivalent.

BDeu is the log marginal likelihood of a discrete node under a Dirichlet
prior that spreads one total mass, the equivalent sample size ``ess``,
uniformly over the node's table: ``a_j = ess / q`` per parent configuration
``j`` and ``a_jk = ess / (q r)`` per cell. The closed form is a ratio of
gamma functions per configuration, accumulated here with ``math.lgamma``; a
configuration with no observations cancels exactly, so only observed ones are
visited.

The uniform spread is the "e" in BDeu: for any fixed ``ess`` the score is
likelihood-equivalent, so Markov-equivalent DAGs get exactly the same total.
The price is sensitivity to ``ess`` itself. It acts like an implicit
structure prior, larger values favor denser graphs, and the ranking of
non-equivalent graphs can flip as it moves, so treat it as a real modeling
choice rather than a constant to leave alone.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .._score import Score

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...data import TabularData


class BDeu(Score):
    """The Bayesian Dirichlet equivalent uniform score, decomposable.

    Every column must be discrete (categorical, ordinal, or binary): the
    score runs on integer codes, so binding to data with a continuous column
    raises ``TypeError``. Score-equivalent for any fixed ``ess``, but the
    value of ``ess`` shapes the result; see the module docstring.

    Args:
        data: Bind at construction, or later through ``bind``.
        ess: The equivalent sample size, the total prior mass spread over
            each node's table. Must be positive. Larger values smooth the
            cell estimates and favor denser graphs; ``10.0`` is a common
            default, and a sweep is worth it when the result matters.
    """

    name: ClassVar[str] = "bdeu"
    needs: ClassVar[str] = "codes"
    decomposable: ClassVar[bool] = True
    score_equivalent: ClassVar[bool] = True

    def __init__(self, data: TabularData | None = None, *, ess: float = 10.0) -> None:
        if ess <= 0.0:
            raise ValueError(f"ess must be positive, got {ess}")
        self._ess = float(ess)
        super().__init__(data)

    @property
    def ess(self) -> float:
        """The equivalent sample size of the Dirichlet prior."""
        return self._ess

    def local_from_codes(
        self,
        y: npt.NDArray[np.int64],
        y_card: int,
        zs: tuple[npt.NDArray[np.int64], ...],
        z_cards: tuple[int, ...],
    ) -> float:
        """The Dirichlet log marginal likelihood from integer codes."""
        n = int(y.shape[0])
        if n == 0:
            raise ValueError("BDeu needs at least one row")
        r = int(y_card)
        q = 1
        for card in z_cards:
            q *= int(card)
        flat = np.ravel_multi_index((*zs, y), (*z_cards, r))
        counts = np.bincount(flat, minlength=q * r).reshape(q, r)
        totals = counts.sum(axis=1)
        a_j = self._ess / q
        a_jk = self._ess / (q * r)
        lgamma_a_j = math.lgamma(a_j)
        lgamma_a_jk = math.lgamma(a_jk)
        local = 0.0
        for j in np.flatnonzero(totals):
            local += lgamma_a_j - math.lgamma(a_j + float(totals[j]))
            for count in counts[j]:
                if count:
                    local += math.lgamma(a_jk + float(count)) - lgamma_a_jk
        return local
