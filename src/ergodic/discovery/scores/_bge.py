# SPDX-License-Identifier: Apache-2.0
"""The BGe score: the marginal likelihood of a linear-Gaussian DAG.

BGe (Bayesian Gaussian equivalent, Geiger and Heckerman 2002) integrates a
linear-Gaussian network's parameters out under a normal-Wishart prior on the
joint mean and precision. Its local score is the log of
``p(d[Y]) / p(d[Pa])``, ``Y = Pa + {node}``, where for a set ``Y`` of
``|Y|`` of the ``n`` variables

``p(d[Y]) = (a_mu / (N + a_mu))^(|Y|/2) * G_|Y|((N + a_w - n + |Y|)/2)
/ (pi^(N|Y|/2) * G_|Y|((a_w - n + |Y|)/2)) * |T_YY|^((a_w - n + |Y|)/2)
/ |R_YY|^((N + a_w - n + |Y|)/2)``

with ``G_k`` the multivariate gamma function, ``N`` rows, ``T`` the prior scale
and ``R = T + S`` (``S`` the scatter matrix about the column means). The exponents
``a_w - n + |Y|`` are the correction of Kuipers, Moffa and Heckerman (2014):
the marginal of a normal-Wishart on ``Y`` has ``a_w - n + |Y|`` degrees of
freedom. The multivariate gammas reduce to one gamma ratio per local, so

``local = 1/2 log(a_mu/(N + a_mu)) + lgamma((N + a_w - n + l + 1)/2)
- lgamma((a_w - n + l + 1)/2) - (N/2) log pi
+ ((a_w - n + l + 1)/2)(l + 1) log t - ((a_w - n + l)/2) l log t
+ ((N + a_w - n + l)/2) log|R_PP| - ((N + a_w - n + l + 1)/2) log|R_YY|``

for ``l`` parents and ``T = t I``. ``a_mu = 1``, ``a_w = n + a_mu + 1`` and
``t = a_mu (a_w - n - 1) / (a_mu + 1)`` are the ``BiDAG`` defaults. The prior mean
is the sample mean, an empirical-Bayes choice: the value is the exact marginal
likelihood under a normal-Wishart prior centred on the data's own mean, so the
prior depends on the data. Score equivalence still holds, and the score does not
move when a column is shifted. ``BiDAG`` centres the prior at zero; the two agree
on centred data and differ otherwise. ``T`` is in the data's units: standardize columns whose scales
differ by orders of magnitude.

BGe is decomposable and score-equivalent: Markov-equivalent DAGs get the same
total, up to rounding.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .._score import Score

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...data import TabularData


class BGe(Score):
    """The BGe score: the linear-Gaussian marginal likelihood, decomposable and equivalent.

    Every node must be a single numeric column (continuous, datetime or binary);
    binding to data with a categorical column of more than two levels raises
    ``TypeError``. ``marginal_likelihood`` is ``True``: summed over a DAG the locals
    are the log marginal likelihood of the data under that DAG, so
    :func:`~ergodic.discovery.graph_posterior` reads it as an exact posterior
    under the prior, not an approximation. See the module docstring for the
    formula.

    Args:
        data: Bind at construction, or later through ``bind``.
        alpha_mu: The prior's equivalent sample size for the mean. Positive.
        alpha_w: The Wishart degrees of freedom; ``None`` takes
            ``n + alpha_mu + 1`` for ``n`` variables. Must exceed ``n + 1``.
    """

    name: ClassVar[str] = "bge"
    needs: ClassVar[str] = "numeric"
    decomposable: ClassVar[bool] = True
    score_equivalent: ClassVar[bool] = True
    marginal_likelihood: ClassVar[bool] = True

    def __init__(
        self,
        data: TabularData | None = None,
        *,
        alpha_mu: float = 1.0,
        alpha_w: float | None = None,
    ) -> None:
        if isinstance(alpha_mu, bool) or not math.isfinite(alpha_mu) or alpha_mu <= 0.0:
            raise ValueError(f"alpha_mu must be positive and finite, got {alpha_mu!r}")
        if alpha_w is not None and (isinstance(alpha_w, bool) or not math.isfinite(alpha_w)):
            raise ValueError(f"alpha_w must be a finite number or None, got {alpha_w!r}")
        self._alpha_mu = float(alpha_mu)
        self._alpha_w_option = None if alpha_w is None else float(alpha_w)
        self._alpha_w = 0.0
        self._t = 0.0
        self._index: dict[str, int] = {}
        self._r: npt.NDArray[np.float64] = np.empty((0, 0), dtype=np.float64)
        super().__init__(data)

    @property
    def alpha_mu(self) -> float:
        """The prior's equivalent sample size for the mean."""
        return self._alpha_mu

    @property
    def alpha_w(self) -> float:
        """The Wishart degrees of freedom (resolved at binding; 0 before)."""
        return self._alpha_w

    @property
    def t(self) -> float:
        """The prior scale's diagonal, ``t = alpha_mu (alpha_w - n - 1) / (alpha_mu + 1)``."""
        return self._t

    # -- one-time sufficient statistics ---------------------------------------

    def _prepare(self) -> None:
        if self._n == 0:
            raise ValueError("BGe needs at least one complete row; the data has none")
        names = self.variables
        wide = [name for name in names if self._blocks[name].shape[1] != 1]
        if wide:
            raise TypeError(
                f"BGe scores single-column numeric nodes; {wide} encode to several columns. "
                "Use BDeu for discrete data or MixedBIC for mixed data."
            )
        size = len(names)
        alpha_w = (
            self._alpha_w_option
            if self._alpha_w_option is not None
            else size + self._alpha_mu + 1.0
        )
        if alpha_w <= size + 1:
            raise ValueError(
                f"alpha_w must exceed n + 1 = {size + 1} for {size} variables, got {alpha_w}"
            )
        design = np.hstack([self._blocks[name] for name in names])
        centered = design - design.mean(axis=0, keepdims=True)
        self._alpha_w = alpha_w
        self._t = self._alpha_mu * (alpha_w - size - 1.0) / (self._alpha_mu + 1.0)
        self._index = {name: i for i, name in enumerate(names)}
        self._r = self._t * np.eye(size) + np.asarray(centered.T @ centered, dtype=np.float64)

    # -- the local score ---------------------------------------------------------

    def _local_value(self, node: str, parents: frozenset[str]) -> float:
        """The local by log-determinants of the posterior scale matrix."""
        columns = [self._index[parent] for parent in sorted(parents)]
        return self._bge_local(self._r, self._index[node], columns, self._n, len(self._index))

    def local_from_arrays(self, y: npt.NDArray[np.float64], X: npt.NDArray[np.float64]) -> float:
        """The same local from ``(N, 1)`` ``y`` and ``(N, l)`` parent columns ``X``.

        Uses the bound score's variable count ``n`` and prior, and the given rows'
        own scatter matrix, so it agrees with :meth:`local` on the bound data.
        """
        self._require_bound()
        if y.ndim != 2 or y.shape[1] != 1:
            raise TypeError("BGe scores single-column nodes; y must have shape (N, 1)")
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError("X must be an (N, l) matrix with as many rows as y")
        block = np.hstack([X, y])
        centered = block - block.mean(axis=0, keepdims=True)
        width = block.shape[1]
        scale = self._t * np.eye(width) + centered.T @ centered
        return self._bge_local(
            scale, width - 1, list(range(width - 1)), int(y.shape[0]), len(self._index)
        )

    def _bge_local(
        self, scale: npt.NDArray[np.float64], node: int, parents: list[int], rows: int, size: int
    ) -> float:
        alpha_mu, alpha_w, t = self._alpha_mu, self._alpha_w, self._t
        count = len(parents)
        base = alpha_w - size + count  # a_w - n + l
        value = 0.5 * (math.log(alpha_mu) - math.log(rows + alpha_mu))
        value += math.lgamma((rows + base + 1.0) / 2.0) - math.lgamma((base + 1.0) / 2.0)
        value -= 0.5 * rows * math.log(math.pi)
        value += ((base + 1.0) / 2.0) * (count + 1) * math.log(t) - (base / 2.0) * count * math.log(
            t
        )
        family = [*parents, node]
        value -= ((rows + base + 1.0) / 2.0) * _logdet(scale[np.ix_(family, family)])
        if parents:
            value += ((rows + base) / 2.0) * _logdet(scale[np.ix_(parents, parents)])
        return value


def _logdet(matrix: npt.NDArray[np.float64]) -> float:
    sign, value = np.linalg.slogdet(matrix)
    if sign <= 0:  # pragma: no cover - T + S is positive definite
        raise ValueError("BGe's posterior scale matrix is not positive definite")
    return float(value)
