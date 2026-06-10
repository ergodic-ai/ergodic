# SPDX-License-Identifier: Apache-2.0
"""Aliases and resolution: method names to algorithms, score specs to bound scores.

Modules resolve lazily so importing the registry never pulls an algorithm in;
``discover`` and the algorithm classes share one source of truth for names.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

from ._score import Score

if TYPE_CHECKING:
    from typing import Any

    from ..data import TabularData

_BY_METHOD: dict[str, tuple[str, str]] = {
    "pc": ("ergodic.discovery.algorithms._pc", "PC"),
    "pc_stable": ("ergodic.discovery.algorithms._pc", "PC"),
    "astar": ("ergodic.discovery.algorithms._astar", "AStar"),
    "a_star": ("ergodic.discovery.algorithms._astar", "AStar"),
    "exact": ("ergodic.discovery.algorithms._astar", "AStar"),
    "ges": ("ergodic.discovery.algorithms._ges", "GES"),
    "hc": ("ergodic.discovery.algorithms._hc", "HillClimb"),
    "hill_climb": ("ergodic.discovery.algorithms._hc", "HillClimb"),
}


def algorithm_for(method: str) -> type[Any]:
    """The algorithm class for a method name, with a readable error."""
    entry = _BY_METHOD.get(method.lower())
    if entry is None:
        allowed = ", ".join(sorted(_BY_METHOD))
        raise ValueError(f"unknown discovery method {method!r}; use one of {allowed}")
    module_name, attr = entry
    return cast("type[Any]", getattr(import_module(module_name), attr))


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
