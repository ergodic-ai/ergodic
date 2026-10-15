# SPDX-License-Identifier: Apache-2.0
"""The conditional-independence test: one class, two states.

A test is configuration plus a statistic. Constructed with data, it is the
stateful oracle from the design: encoding done once, sufficient statistics
precomputed, queries cached. Constructed bare, it is a reusable config that an
algorithm binds at fit. ``bind`` always returns a new instance; a bound test
never mutates back.

The query is ``test(x, y, given)`` returning a :class:`CIResult` with the
statistic and p-value. No decision lives in the record: alpha belongs to the
algorithm, so the cache stays alpha-independent and one warmed oracle serves
PC, then FCI, then a different alpha, for free.

Subclasses implement ``statistic`` against the input form they declare in
``needs``: ``"numeric"`` hands 2-D blocks (one-hot for discrete columns),
``"codes"`` hands integer codes with cardinalities. The base owns binding,
encoding, validation, caching, and counting.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ._encode import _codes_columns, _numeric_blocks

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Self

    import numpy.typing as npt

    from ..data import TabularData
    from ..graph import MixedGraph


@dataclass(frozen=True)
class CIResult:
    """One conditional-independence query: the statistic, its p-value, and dof."""

    statistic: float
    p_value: float
    dof: float | None = None


class CITest:
    """Base for conditional-independence tests; see the module docstring.

    Subclasses set ``name`` and ``needs``, may take options in ``__init__``
    (set them before calling ``super().__init__(data)``, since binding may
    read them), and implement ``statistic``. The optional ``_prepare`` hook
    runs once after binding, for sufficient statistics.
    """

    name: ClassVar[str] = "ci_test"
    needs: ClassVar[str] = "numeric"  # "numeric" or "codes"

    def __init__(self, data: TabularData | None = None) -> None:
        self._data: TabularData | None = None
        self._n: int = 0
        self._names: frozenset[str] = frozenset()
        self._blocks: dict[str, npt.NDArray[np.float64]] = {}
        self._codes: dict[str, npt.NDArray[np.int64]] = {}
        self._cards: dict[str, int] = {}
        self._cache: dict[tuple[str, str, frozenset[str]], CIResult] = {}
        self._n_tests: dict[int, int] = {}
        self._cache_hits: int = 0
        if data is not None:
            self._attach(data)

    # -- binding ---------------------------------------------------------

    @property
    def is_bound(self) -> bool:
        """Whether the test is bound to data and can answer queries."""
        return self._data is not None

    @property
    def data(self) -> TabularData | None:
        """The dataset this test is bound to, after dropping incomplete rows."""
        return self._data

    @property
    def n(self) -> int:
        """Rows the oracle answers from (after dropping incomplete rows)."""
        return self._n

    @property
    def variables(self) -> tuple[str, ...]:
        """The variables this oracle answers about, sorted."""
        return tuple(sorted(self._names))

    def bind(self, data: TabularData) -> Self:
        """A new instance of this test bound to ``data``; never mutates ``self``."""
        fresh = copy.copy(self)
        fresh._attach(data)
        return fresh

    def _attach(self, data: TabularData) -> None:
        complete = data.dropna()
        self._data = complete
        self._n = complete.n_rows
        self._names = frozenset(complete.variables)
        self._cache = {}
        self._n_tests = {}
        self._cache_hits = 0
        if self.needs == "numeric":
            self._blocks = _numeric_blocks(complete)
        elif self.needs == "codes":
            self._codes, self._cards = _codes_columns(complete)
        else:
            raise ValueError(f"{type(self).__name__}.needs must be 'numeric' or 'codes'")
        self._prepare()

    def _prepare(self) -> None:
        """Hook for one-time sufficient statistics after binding."""

    # -- the query ---------------------------------------------------------

    def test(self, x: str, y: str, given: Iterable[str] = ()) -> CIResult:
        """The cached conditional-independence query, symmetric in ``x`` and ``y``."""
        self._require_bound()
        conditioning = frozenset(given)
        self._validate(x, y, conditioning)
        a, b = sorted((x, y))
        key = (a, b, conditioning)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache_hits += 1
            return hit
        result = self._dispatch(a, b, conditioning)
        self._cache[key] = result
        size = len(conditioning)
        self._n_tests[size] = self._n_tests.get(size, 0) + 1
        return result

    def independent(self, x: str, y: str, given: Iterable[str] = (), *, alpha: float) -> bool:
        """The decision the algorithms read: fail to reject at ``alpha``."""
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        return self.test(x, y, given).p_value > alpha

    def diagnostics(self) -> dict[str, float]:
        """Run counters: total queries, largest conditioning set, cache hits."""
        total = float(sum(self._n_tests.values()))
        largest = float(max(self._n_tests, default=0))
        return {
            "n_ci_tests": total,
            "max_cond_size": largest,
            "cache_hits": float(self._cache_hits),
        }

    # -- subclass surface ---------------------------------------------------

    def statistic(self, *args: Any) -> tuple[float, float, float | None]:
        """Return ``(statistic, p_value, dof)``; signature depends on ``needs``.

        ``numeric``: ``statistic(x, y, z)`` with ``x`` ``(n, dx)``, ``y``
        ``(n, dy)``, ``z`` ``(n, dz)`` float blocks. ``codes``:
        ``statistic(x, y, z, cards)`` with ``x``, ``y`` ``(n,)`` integer
        codes, ``z`` a tuple of ``(n,)`` code columns, and ``cards`` the
        matching ``(card_x, card_y, cards_z)``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement statistic")

    # -- internals -----------------------------------------------------------

    def _require_bound(self) -> None:
        if not self.is_bound:
            raise RuntimeError(
                f"{type(self).__name__} is not bound to data; construct it with data "
                "or call bind(data)"
            )

    def _validate(self, x: str, y: str, given: frozenset[str]) -> None:
        unknown = sorted(({x, y} | given) - self._names)
        if unknown:
            raise ValueError(f"unknown variable(s) {unknown}; the data has {sorted(self._names)}")
        if x == y:
            raise ValueError(f"x and y must differ; both are {x!r}")
        if x in given or y in given:
            raise ValueError("the conditioning set must not contain x or y")

    def _dispatch(self, x: str, y: str, given: frozenset[str]) -> CIResult:
        ordered = sorted(given)
        if self.needs == "numeric":
            xb = self._blocks[x]
            yb = self._blocks[y]
            zb = (
                np.hstack([self._blocks[w] for w in ordered])
                if ordered
                else np.empty((self._n, 0), dtype=np.float64)
            )
            stat, p, dof = self.statistic(xb, yb, zb)
        else:
            zc = tuple(self._codes[w] for w in ordered)
            cards = (self._cards[x], self._cards[y], tuple(self._cards[w] for w in ordered))
            stat, p, dof = self.statistic(self._codes[x], self._codes[y], zc, cards)
        return CIResult(statistic=float(stat), p_value=float(p), dof=dof)


class DSeparationOracle(CITest):
    """The ground-truth oracle: answers queries by m-separation in a known graph.

    Quacks like any bound test (p-value 1.0 when separated, 0.0 when
    connected), so every constraint algorithm can be checked for exactness and
    run in teaching mode. Needs no data; ``bind`` returns ``self``.

    Latent and selection nodes are not query-able variables: the oracle
    exposes the observed margin, which is what a constraint algorithm
    searching over data would see. Separation is still computed on the full
    graph, latents included, and every query conditions on the selection
    nodes, because data sampled under selection only ever shows the
    distribution given them.
    """

    name: ClassVar[str] = "d_separation"

    def __init__(self, graph: MixedGraph) -> None:
        super().__init__(None)
        self._graph = graph
        self._names = frozenset(graph.nodes) - graph.latent - graph.selection

    @property
    def is_bound(self) -> bool:
        """Always ``True``: the oracle answers from its graph and needs no data."""
        return True

    @property
    def graph(self) -> MixedGraph:
        """The graph whose d-separations (m-separations) the oracle reports."""
        return self._graph

    def bind(self, data: TabularData) -> Self:
        return self

    def _require_bound(self) -> None:
        return

    def _dispatch(self, x: str, y: str, given: frozenset[str]) -> CIResult:
        separated = self._graph.m_separated(x, y, given | self._graph.selection)
        return CIResult(statistic=0.0, p_value=1.0 if separated else 0.0, dof=None)
