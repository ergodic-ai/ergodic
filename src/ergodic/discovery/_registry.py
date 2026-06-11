# SPDX-License-Identifier: Apache-2.0
"""Aliases and resolution: method names to algorithms, ingredient specs to bound oracles.

Modules resolve lazily so importing the registry never pulls an algorithm in;
``discover`` and the algorithm classes share one source of truth for names.
``resolve_test`` and ``resolve_score`` enforce the same rule: the data comes
from exactly one place.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

from ..data import TabularData
from ._ci import CITest
from ._score import Score

if TYPE_CHECKING:
    from typing import Any

_BY_METHOD: dict[str, tuple[str, str]] = {
    "pc": ("ergodic.discovery.algorithms._pc", "PC"),
    "pc_stable": ("ergodic.discovery.algorithms._pc", "PC"),
    "fci": ("ergodic.discovery.algorithms._fci", "FCI"),
    "mmpc": ("ergodic.discovery.algorithms._mmpc", "MMPC"),
    "astar": ("ergodic.discovery.algorithms._astar", "AStar"),
    "a_star": ("ergodic.discovery.algorithms._astar", "AStar"),
    "exact": ("ergodic.discovery.algorithms._astar", "AStar"),
    "ges": ("ergodic.discovery.algorithms._ges", "GES"),
    "hc": ("ergodic.discovery.algorithms._hc", "HillClimb"),
    "hill_climb": ("ergodic.discovery.algorithms._hc", "HillClimb"),
    "boss": ("ergodic.discovery.algorithms._boss", "BOSS"),
    "grasp": ("ergodic.discovery.algorithms._grasp", "GRaSP"),
    "lingam": ("ergodic.discovery.algorithms._lingam", "DirectLiNGAM"),
    "direct_lingam": ("ergodic.discovery.algorithms._lingam", "DirectLiNGAM"),
    "glasso": ("ergodic.discovery.algorithms._glasso", "Glasso"),
    "pcmci": ("ergodic.discovery.algorithms._pcmci", "PCMCI"),
}

# Methods that read temporal data (TimeSeriesData or PanelData) and own their
# dependence structure. Everything else assumes exchangeable rows.
TEMPORAL_METHODS: frozenset[str] = frozenset({"pcmci"})


def _require_exchangeable(data: Any) -> None:
    """Reject temporal datasets where an iid method would misread the rows."""
    if data is not None and not isinstance(data, TabularData):
        raise TypeError(
            f"{type(data).__name__} rows are not exchangeable, so an iid method would "
            "misread them; use method='pcmci' for time series and panels, or rebuild "
            "with tabular(...) if the rows really are independent"
        )


def algorithm_for(method: str) -> type[Any]:
    """The algorithm class for a method name, with a readable error."""
    entry = _BY_METHOD.get(method.lower())
    if entry is None:
        allowed = ", ".join(sorted(_BY_METHOD))
        raise ValueError(f"unknown discovery method {method!r}; use one of {allowed}")
    module_name, attr = entry
    return cast("type[Any]", getattr(import_module(module_name), attr))


_TEST_ALIASES: dict[str, tuple[str, str]] = {
    "linear": ("ergodic.discovery.ci._linear", "Linear"),
    "fisher_z": ("ergodic.discovery.ci._linear", "Linear"),
    "discrete": ("ergodic.discovery.ci._discrete", "Discrete"),
    "g_sq": ("ergodic.discovery.ci._discrete", "Discrete"),
    "gsq": ("ergodic.discovery.ci._discrete", "Discrete"),
    "mixed": ("ergodic.discovery.ci._mixed", "Mixed"),
    "dg": ("ergodic.discovery.ci._mixed", "Mixed"),
    "gcm": ("ergodic.discovery.ci._gcm", "GCM"),
}


def _auto_test_key(data: TabularData) -> str:
    """Pick a test family from the column schema."""
    kinds = [data.schema[name] for name in data.variables]
    if all(c.is_continuous for c in kinds):
        return "linear"
    if all(c.is_discrete for c in kinds):
        return "discrete"
    return "mixed"


def _test_class(key: str) -> type[CITest]:
    """Resolve a test name to its class, importing the module only when asked."""
    try:
        module_name, attr = _TEST_ALIASES[key]
    except KeyError:
        names = ", ".join(sorted(_TEST_ALIASES))
        raise ValueError(
            f"unknown test {key!r}; use 'auto', one of: {names}, or pass a CITest instance"
        ) from None
    return cast("type[CITest]", getattr(import_module(module_name), attr))


def resolve_test(spec: CITest | str, data: TabularData | None) -> CITest:
    """Resolve the test spec and the data into one bound oracle.

    Data comes from exactly one place: an unbound spec needs ``data`` at fit,
    a bound test carries its own and must not be handed a different dataset.
    """
    _require_exchangeable(data)
    if isinstance(spec, CITest):
        if spec.is_bound:
            if data is not None and data is not spec.data:
                raise ValueError(
                    f"{type(spec).__name__} is already bound to its own data; "
                    "omit data at fit, or pass an unbound test to bind here"
                )
            return spec
        if data is None:
            raise ValueError(
                f"{type(spec).__name__} is unbound and fit got no data; "
                "construct the test with data or pass data to fit"
            )
        return spec.bind(data)
    if isinstance(spec, str):
        if data is None:
            raise ValueError(f"test {spec!r} needs data to bind at fit; none was given")
        key = spec.lower()
        if key == "auto":
            key = _auto_test_key(data)
        return _test_class(key)(data)
    raise TypeError(f"test must be a CITest or a string, got {type(spec).__name__}")


_SCORE_ALIASES: dict[str, tuple[str, str]] = {
    "bic": ("ergodic.discovery.scores._bic", "BIC"),
    "bic_discrete": ("ergodic.discovery.scores._discrete_bic", "DiscreteBIC"),
    "discrete_bic": ("ergodic.discovery.scores._discrete_bic", "DiscreteBIC"),
    "bdeu": ("ergodic.discovery.scores._bdeu", "BDeu"),
    "bic_mixed": ("ergodic.discovery.scores._mixed_bic", "MixedBIC"),
    "mixed_bic": ("ergodic.discovery.scores._mixed_bic", "MixedBIC"),
}


def _auto_score_key(data: TabularData) -> str:
    """Pick a score family from the column schema."""
    kinds = [data.schema[name] for name in data.variables]
    if all(c.is_continuous for c in kinds):
        return "bic"
    if all(c.is_discrete for c in kinds):
        return "bic_discrete"
    return "bic_mixed"


def _score_class(key: str) -> type[Score]:
    """Resolve a score name to its class, importing the module only when asked."""
    try:
        module_name, attr = _SCORE_ALIASES[key]
    except KeyError:
        names = ", ".join(sorted(_SCORE_ALIASES))
        raise ValueError(
            f"unknown score {key!r}; use 'auto', one of: {names}, or pass a Score instance"
        ) from None
    return cast("type[Score]", getattr(import_module(module_name), attr))


def resolve_score(spec: Score | str, data: TabularData | None) -> Score:
    """Resolve the score spec and the data into one bound score.

    Data comes from exactly one place: an unbound spec needs ``data`` at fit,
    a bound score carries its own and must agree with any dataset it is handed,
    and a string needs data to bind (``"auto"`` reads its schema).
    """
    _require_exchangeable(data)
    if isinstance(spec, Score):
        if spec.is_bound:
            if data is not None:
                complete = data.dropna()
                same = (
                    complete.n_rows == spec.n
                    and tuple(sorted(complete.variables)) == spec.variables
                )
                if not same:
                    raise ValueError(
                        "the score is bound to a different dataset (rows or variables "
                        "differ after dropna); bind the score to this data or pass an "
                        "unbound score"
                    )
            return spec
        if data is None:
            raise ValueError(
                "no data to fit: pass data to fit(), or construct the score bound to data"
            )
        return spec.bind(data)
    if isinstance(spec, str):
        if data is None:
            raise ValueError(f"score {spec!r} needs data to bind at fit; none was given")
        key = spec.lower()
        if key == "auto":
            key = _auto_score_key(data)
        return _score_class(key)(data)
    raise TypeError(f"score must be a Score or a string, got {type(spec).__name__}")
