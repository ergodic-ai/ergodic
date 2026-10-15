# SPDX-License-Identifier: Apache-2.0
"""The decomposable score: one queryable object behind every score-based search.

A score is configuration plus a local function. Constructed with data (or
bound later, same two states as a test), it answers ``local(node, parents)``
with caching, serves greedy searches through ``local_diff``, and serves the
exact solvers through ``enumerate_locals`` (all candidate parent sets under an
in-degree cap, up front). The graph score is the sum of locals, which is what
lets one user-defined ``local_from_arrays`` power hill climbing, GES, and A*
alike.

Two declared properties matter to the searches: ``decomposable`` (everything
here) and ``score_equivalent`` (equal scores across a Markov equivalence
class; GES and the order-based searches require it).

Like a test, a score declares the input form it needs: ``"numeric"`` hands
``local_from_arrays`` 2-D float blocks (one-hot for discrete columns),
``"codes"`` hands ``local_from_codes`` integer codes with cardinalities.
"""

from __future__ import annotations

import copy
from itertools import combinations
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ._encode import _codes_columns, _numeric_blocks

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Self

    import numpy.typing as npt

    from ..data import TabularData


class Score:
    """Base for decomposable scores; see the module docstring.

    Subclasses set ``name``, ``needs``, and the property flags, may take
    options in ``__init__`` (set them before calling ``super().__init__(data)``),
    and implement the method matching ``needs``: ``local_from_arrays`` for
    ``"numeric"``, ``local_from_codes`` for ``"codes"``. ``_prepare`` runs once
    after binding for sufficient statistics; a native score may also override
    ``_local_value`` to skip array assembly entirely (the covariance path in
    BIC).
    """

    name: ClassVar[str] = "score"
    needs: ClassVar[str] = "numeric"
    decomposable: ClassVar[bool] = True
    score_equivalent: ClassVar[bool] = True

    def __init__(self, data: TabularData | None = None) -> None:
        self._data: TabularData | None = None
        self._n: int = 0
        self._names: frozenset[str] = frozenset()
        self._blocks: dict[str, npt.NDArray[np.float64]] = {}
        self._codes: dict[str, npt.NDArray[np.int64]] = {}
        self._cards: dict[str, int] = {}
        self._cache: dict[tuple[str, frozenset[str]], float] = {}
        self._n_evals: int = 0
        self._cache_hits: int = 0
        if data is not None:
            self._attach(data)

    # -- binding ---------------------------------------------------------

    @property
    def is_bound(self) -> bool:
        """Whether the score is bound to data and can be evaluated."""
        return self._data is not None

    @property
    def data(self) -> TabularData | None:
        """The dataset this score is bound to, after dropping incomplete rows."""
        return self._data

    @property
    def n(self) -> int:
        """Rows the score is computed from (after dropping incomplete rows); 0 before binding."""
        return self._n

    @property
    def variables(self) -> tuple[str, ...]:
        """The variables this oracle answers about, sorted."""
        return tuple(sorted(self._names))

    def bind(self, data: TabularData) -> Self:
        """A new instance of this score bound to ``data``; never mutates ``self``."""
        fresh = copy.copy(self)
        fresh._attach(data)
        return fresh

    def _attach(self, data: TabularData) -> None:
        complete = data.dropna()
        self._data = complete
        self._n = complete.n_rows
        self._names = frozenset(complete.variables)
        self._blocks = {}
        self._codes = {}
        self._cards = {}
        if self.needs == "numeric":
            self._blocks = _numeric_blocks(complete)
        elif self.needs == "codes":
            self._codes, self._cards = _codes_columns(complete)
        else:
            raise ValueError(f"{type(self).__name__}.needs must be 'numeric' or 'codes'")
        self._cache = {}
        self._n_evals = 0
        self._cache_hits = 0
        self._prepare()

    def _prepare(self) -> None:
        """Hook for one-time sufficient statistics after binding."""

    # -- the queries -------------------------------------------------------

    def local(self, node: str, parents: Iterable[str] = ()) -> float:
        """The cached local score of ``node`` given ``parents``."""
        self._require_bound()
        parent_set = frozenset(parents)
        self._validate(node, parent_set)
        key = (node, parent_set)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache_hits += 1
            return hit
        value = self._local_value(node, parent_set)
        self._cache[key] = value
        self._n_evals += 1
        return value

    def local_diff(self, node: str, parents: Iterable[str], add: str) -> float:
        """The gain from adding ``add`` to the parents of ``node``.

        The default is two locals; an implementation may override it with a
        cheaper incremental form.
        """
        base = frozenset(parents)
        return self.local(node, base | {add}) - self.local(node, base)

    def enumerate_locals(
        self, node: str, candidates: Iterable[str], max_parents: int | None = None
    ) -> dict[frozenset[str], float]:
        """Every candidate parent set up to ``max_parents``, scored.

        The access pattern of the exact solvers (A*, dynamic programming,
        ILP): all the local scores up front under an in-degree cap.
        """
        pool = sorted(frozenset(candidates))
        cap = len(pool) if max_parents is None else min(max_parents, len(pool))
        out: dict[frozenset[str], float] = {}
        for size in range(cap + 1):
            for combo in combinations(pool, size):
                parent_set = frozenset(combo)
                out[parent_set] = self.local(node, parent_set)
        return out

    def graph_score(self, parents_of: dict[str, frozenset[str]]) -> float:
        """The decomposable total: the sum of locals over a parent map."""
        return float(sum(self.local(node, parents) for node, parents in parents_of.items()))

    def diagnostics(self) -> dict[str, float]:
        """Run counters: evaluations and cache hits."""
        return {"n_score_evals": float(self._n_evals), "cache_hits": float(self._cache_hits)}

    # -- subclass surface ---------------------------------------------------

    def local_from_arrays(self, y: npt.NDArray[np.float64], X: npt.NDArray[np.float64]) -> float:
        """The local score from arrays: ``y`` is ``(n, dy)``, ``X`` is ``(n, k)``."""
        raise NotImplementedError(f"{type(self).__name__} does not implement local_from_arrays")

    def local_from_codes(
        self,
        y: npt.NDArray[np.int64],
        y_card: int,
        zs: tuple[npt.NDArray[np.int64], ...],
        z_cards: tuple[int, ...],
    ) -> float:
        """The local score from codes: ``y`` is ``(n,)`` integer codes, ``zs`` the parents'."""
        raise NotImplementedError(f"{type(self).__name__} does not implement local_from_codes")

    # -- internals -----------------------------------------------------------

    def _local_value(self, node: str, parents: frozenset[str]) -> float:
        """Assemble the declared input form and score it; native scores may override."""
        ordered = sorted(parents)
        if self.needs == "numeric":
            y = self._blocks[node]
            X = (
                np.hstack([self._blocks[p] for p in ordered])
                if ordered
                else np.empty((self._n, 0), dtype=np.float64)
            )
            return float(self.local_from_arrays(y, X))
        zs = tuple(self._codes[p] for p in ordered)
        z_cards = tuple(self._cards[p] for p in ordered)
        return float(self.local_from_codes(self._codes[node], self._cards[node], zs, z_cards))

    def _require_bound(self) -> None:
        if not self.is_bound:
            raise RuntimeError(
                f"{type(self).__name__} is not bound to data; construct it with data "
                "or call bind(data)"
            )

    def _validate(self, node: str, parents: frozenset[str]) -> None:
        unknown = sorted(({node} | parents) - self._names)
        if unknown:
            raise ValueError(f"unknown variable(s) {unknown}; the data has {sorted(self._names)}")
        if node in parents:
            raise ValueError(f"{node!r} cannot be its own parent")
