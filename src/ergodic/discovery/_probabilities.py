# SPDX-License-Identifier: Apache-2.0
"""Edge probabilities: a model's soft opinion about each directed edge.

Some discovery models do not return one graph. They return a number per ordered
pair of variables, read as "how likely is an edge from this source to this
target". :class:`EdgeProbabilities` holds that matrix over named variables, says
which of three readings it carries, and keeps where it came from. It is not a
graph posterior: the numbers are per-edge marginals (or bootstrap frequencies),
they ignore acyclicity, and they are uncalibrated unless the source says so.
Turning them into a DAG is an explicit modelling step, :func:`ergodic.decode`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..graph import Mark, mixed

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing as npt
    import pandas as pd

    from ..graph import MixedGraph
    from ._bootstrap import BootstrapResult

ProbabilityKind = Literal["independent", "pair_state", "frequency"]
_KINDS: tuple[str, ...] = ("independent", "pair_state", "frequency")
PAIR_SIMPLEX_TOLERANCE = 1e-12
"""Slack allowed on ``P[i, j] + P[j, i] <= 1`` for the pair-state readings."""


def _frozen(value: Any) -> Any:
    """A read-only deep copy of a provenance value (mappings and sequences)."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _frozen(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_frozen(v) for v in value)
    return value


def _check_variables(variables: Iterable[str]) -> tuple[str, ...]:
    names = tuple(variables)
    if not names:
        raise ValueError("edge probabilities need at least one variable")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("variables must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("variables must be unique")
    return names


def _check_values(values: Any, size: int) -> npt.NDArray[np.float64]:
    array = np.asarray(values)
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number):
        raise TypeError("values must be a real-valued numeric matrix")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError("values must be a real-valued numeric matrix")
    if array.shape != (size, size):
        raise ValueError(
            f"values must be a {size} x {size} matrix (source rows, target columns), "
            f"got shape {array.shape}"
        )
    result = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("values must be finite")
    if np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("values must lie in [0, 1]")
    if np.any(np.diag(result) != 0.0):
        raise ValueError("the diagonal must be zero: a variable has no edge to itself")
    return result


@dataclass(frozen=True, eq=False)
class EdgeProbabilities:
    """A matrix of directed-edge probabilities over named variables.

    ``values[i, j]`` is the number for an edge from ``variables[i]`` (the source
    row) to ``variables[j]`` (the target column). The diagonal is zero, every
    entry is finite and in ``[0, 1]``, and the stored matrix is a read-only copy,
    so the caller's array is never shared or changed.

    ``kind`` says how to read the numbers:

    - ``"independent"``: one marginal per directed edge, each on its own. Both
      directions of a pair may be high at once (AVICI and CDFM return these).
    - ``"pair_state"``: each unordered pair has three exclusive states (no edge,
      ``i -> j``, ``j -> i``) and the two directed cells are two of them, so
      ``P[i, j] + P[j, i] <= 1``; the rest is the no-edge mass.
    - ``"frequency"``: how often each directed edge appeared over resampled runs
      (see :meth:`from_bootstrap`). One edge per pair per run, so the pair
      simplex holds here too.

    ``calibrated`` is ``False`` unless the source states the numbers match
    observed edge frequencies; ergodic never sets it on its own. ``provenance``
    is free-form, read-only metadata (model, version, job id, threshold, ...).

    None of this is a posterior over graphs. The marginals do not account for
    acyclicity or for dependence between edges; a decoded graph built from them
    is a modelling choice, not the model's claim.

    Attributes:
        variables: Variable names, the row and column order of ``values``.
        values: The ``(d, d)`` float matrix, read-only.
        kind: ``"independent"``, ``"pair_state"`` or ``"frequency"``.
        calibrated: Whether the source states its numbers are calibrated.
        provenance: Read-only metadata about where the numbers came from.
    """

    variables: tuple[str, ...]
    values: npt.NDArray[np.float64] = field(repr=False)
    kind: ProbabilityKind = "independent"
    calibrated: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        names = _check_variables(self.variables)
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {list(_KINDS)}, got {self.kind!r}")
        if not isinstance(self.calibrated, bool):
            raise TypeError("calibrated must be a bool")
        values = _check_values(self.values, len(names))
        if self.kind != "independent":
            sums = values + values.T
            upper = np.triu_indices(len(names), k=1)
            bad = np.flatnonzero(sums[upper] > 1.0 + PAIR_SIMPLEX_TOLERANCE)
            if bad.size:
                i, j = int(upper[0][bad[0]]), int(upper[1][bad[0]])
                raise ValueError(
                    f"{self.kind} values need P[i, j] + P[j, i] <= 1 for every pair; "
                    f"{names[i]!r} and {names[j]!r} sum to {sums[i, j]:.6g}. "
                    "Independent marginals belong in kind='independent'"
                )
        values.setflags(write=False)
        object.__setattr__(self, "variables", names)
        object.__setattr__(self, "values", values)
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        object.__setattr__(self, "provenance", _frozen(dict(self.provenance)))

    # -- readings ------------------------------------------------------------

    def to_pair_states(self) -> EdgeProbabilities:
        """The pair-state reading: three exclusive states per unordered pair.

        For ``"independent"`` marginals ``p = P[i, j]`` and ``q = P[j, i]``, the
        two edges are treated as independent coin flips and conditioned on "not
        both" (a DAG cannot hold a two-cycle): no edge ``(1 - p)(1 - q)``,
        ``i -> j`` ``p(1 - q)``, ``j -> i`` ``q(1 - p)``, renormalized. When both
        are exactly 1 the limit is undecided, so the pair becomes ``(0, 0.5,
        0.5)``. This keeps each model's direction ranking and never looks at a
        true graph. It is the conversion the research decoders were qualified
        with (EXP-0626).

        ``"pair_state"`` values are returned unchanged (the same object).
        ``"frequency"`` values already satisfy the pair simplex (one edge per
        pair per run), so they are read as pair states as they are: a new object
        of kind ``"pair_state"`` with the same numbers.

        Returns:
            Edge probabilities of kind ``"pair_state"``, with the conversion
            recorded in ``provenance["pair_state_conversion"]``.
        """
        if self.kind == "pair_state":
            return self
        if self.kind == "frequency":
            conversion = "frequency_as_pair_state_v1"
            values = self.values.copy()
        else:
            conversion = "no_reciprocal_conditioning_v1"
            values = _condition_independent(self.values)
        provenance = {
            **_thaw(self.provenance),
            "pair_state_conversion": conversion,
            "converted_from": self.kind,
        }
        return EdgeProbabilities(self.variables, values, "pair_state", self.calibrated, provenance)

    def existence(self) -> npt.NDArray[np.float64]:
        """Skeleton probabilities: the chance each pair is adjacent, either way.

        Read from the pair-state view (:meth:`to_pair_states`), so it is
        ``P[i, j] + P[j, i]`` there: for ``"independent"`` marginals that is
        ``(p(1 - q) + q(1 - p)) / (1 - pq)``. The matrix is symmetric with a zero
        diagonal and read-only.
        """
        pair = self.to_pair_states().values
        result = pair + pair.T
        np.fill_diagonal(result, 0.0)
        result.setflags(write=False)
        return result

    def threshold(self, t: float = 0.5) -> MixedGraph:
        """The directed edges whose value is above ``t``, as a diagnostic graph.

        Keeps every edge with ``values[i, j] > t`` (strictly, as AVICI and CDFM
        threshold natively). The result is a ``mixed``-kind graph over every
        variable and may hold directed cycles. A pair whose two directions both
        pass cannot be drawn as two edges (one edge per pair), so it shows as an
        undirected ``X --- Y`` edge: adjacent, direction unresolved. This is not
        a DAG and not a decoded graph; use :func:`ergodic.decode` for one.

        Args:
            t: The threshold, in ``[0, 1)``.

        Returns:
            A ``mixed`` :class:`~ergodic.graph.MixedGraph`.
        """
        if isinstance(t, bool) or not math.isfinite(t) or not 0.0 <= t < 1.0:
            raise ValueError(f"t must be in [0, 1), got {t!r}")
        above = self.values > t
        edges: list[tuple[str, Mark, str, Mark]] = []
        names = self.variables
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if above[i, j] and above[j, i]:
                    edges.append((names[i], Mark.TAIL, names[j], Mark.TAIL))
                elif above[i, j]:
                    edges.append((names[i], Mark.TAIL, names[j], Mark.ARROW))
                elif above[j, i]:
                    edges.append((names[j], Mark.TAIL, names[i], Mark.ARROW))
        return mixed(edges, nodes=names)

    # -- dataframe round trip ---------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """The matrix as a dataframe: index ``source``, columns ``target``."""
        import pandas as pd

        frame = pd.DataFrame(
            self.values.copy(), index=list(self.variables), columns=list(self.variables)
        )
        frame.index.name = "source"
        frame.columns.name = "target"
        return frame

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        kind: ProbabilityKind = "independent",
        calibrated: bool = False,
        provenance: Mapping[str, Any] | None = None,
    ) -> EdgeProbabilities:
        """Build from a square dataframe: rows the source, columns the target.

        The index gives the variable order; the columns must hold the same names
        (in any order) and are reordered to match. Labels are read as strings.

        Args:
            frame: The square matrix.
            kind: How to read the numbers, as on the class.
            calibrated: Whether the source states its numbers are calibrated.
            provenance: Metadata to keep.
        """
        rows = [str(name) for name in frame.index]
        columns = [str(name) for name in frame.columns]
        if len(set(rows)) != len(rows) or set(rows) != set(columns) or len(rows) != len(columns):
            raise ValueError(
                "the frame's index and columns must hold the same unique variable names"
            )
        ordered = frame.copy()
        ordered.index = rows
        ordered.columns = columns
        values = ordered.loc[rows, rows].to_numpy()
        return cls(tuple(rows), values, kind, calibrated, dict(provenance or {}))

    @classmethod
    def from_bootstrap(
        cls, result: BootstrapResult, *, undirected: Literal["split", "drop"] = "split"
    ) -> EdgeProbabilities:
        """Directed-edge frequencies from a :func:`~ergodic.discovery.bootstrap` run.

        Each canonical glyph frequency lands on its directed cell: ``"X --> Y"``
        on ``[X, Y]`` and ``"X <-- Y"`` on ``[Y, X]``. An undirected ``X --- Y``
        (a CPDAG edge, "one of the two directions") is split half and half under
        ``undirected="split"`` and dropped under ``"drop"``. Bidirected and
        circle-marked glyphs say something other than "this direction" and are
        always dropped; the dropped mass is recorded in
        ``provenance["dropped_frequency"]``. Frequencies measure resampling
        stability, not probability.

        Args:
            result: An iid bootstrap result (window-graph results are refused).
            undirected: ``"split"`` or ``"drop"``, as above.

        Returns:
            Edge probabilities of kind ``"frequency"``.
        """
        if undirected not in ("split", "drop"):
            raise ValueError(f"undirected must be 'split' or 'drop', got {undirected!r}")
        names = tuple(result.variables)
        index = {name: i for i, name in enumerate(names)}
        values = np.zeros((len(names), len(names)), dtype=np.float64)
        dropped = 0.0
        for glyph, frequency in result.marks.items():
            parts = glyph.split()
            if len(parts) != 3 or parts[0] not in index or parts[2] not in index:
                raise ValueError(
                    f"glyph {glyph!r} does not name two of the data's variables; "
                    "from_bootstrap reads iid bootstrap results, not window graphs"
                )
            u, marks, v = index[parts[0]], parts[1], index[parts[2]]
            left, right = marks[0], marks[-1]
            if left == "-" and right == ">":
                values[u, v] += frequency
            elif left == "<" and right == "-":
                values[v, u] += frequency
            elif left == "-" and right == "-" and undirected == "split":
                values[u, v] += frequency / 2.0
                values[v, u] += frequency / 2.0
            else:
                dropped += frequency
        provenance = {
            "source": "bootstrap",
            "method": result.method,
            "n_runs": result.n_runs,
            "n_failed": result.n_failed,
            "undirected": undirected,
            "dropped_frequency": dropped,
        }
        return cls(names, np.clip(values, 0.0, 1.0), "frequency", False, provenance)

    # -- dunder ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EdgeProbabilities):
            return NotImplemented
        return (
            self.variables == other.variables
            and self.kind == other.kind
            and self.calibrated == other.calibrated
            and np.array_equal(self.values, other.values)
            and _thaw(self.provenance) == _thaw(other.provenance)
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        above = int((self.values > 0.5).sum())
        calibration = "calibrated" if self.calibrated else "uncalibrated"
        return (
            f"EdgeProbabilities({self.kind}, {len(self.variables)} variables, "
            f"{above} directed cells above 0.5, {calibration})"
        )


def _thaw(value: Any) -> Any:
    """A plain, mutable copy of a frozen provenance value."""
    if isinstance(value, MappingProxyType):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def _condition_independent(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Independent marginals conditioned on no two-cycle (EXP-0626 ``pair_probabilities``)."""
    result = np.zeros_like(values)
    size = values.shape[0]
    for low in range(size):
        for high in range(low + 1, size):
            forward = float(values[low, high])
            reverse = float(values[high, low])
            masses = np.array(
                [(1 - forward) * (1 - reverse), forward * (1 - reverse), reverse * (1 - forward)],
                dtype=np.float64,
            )
            total = float(masses.sum())
            if total <= 0:
                masses[:] = (0.0, 0.5, 0.5)
            else:
                masses /= total
            result[low, high] = masses[1]
            result[high, low] = masses[2]
    return result
