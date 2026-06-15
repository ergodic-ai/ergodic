# SPDX-License-Identifier: Apache-2.0
"""A small attribute-readable dict that carries a dataset and its causal question."""

from __future__ import annotations

from typing import Any


class Bunch(dict[str, Any]):
    """A dict whose keys are also attributes.

    Every loader returns one of these. ``data`` holds the typed dataset; the
    rest is the causal question. Read it either way::

        ds = load_birthweight()
        ds.data         # the TabularData
        ds.treatment    # "mbsmoke"
        ds["outcome"]   # "bweight", the same as ds.outcome
        ds.target       # {"value": ..., "tol": ..., "source": ...}

    The keys vary by task. An adjustment dataset carries ``treatment``,
    ``outcome``, ``confounders``, and ``target``; a synthetic-control dataset
    carries ``treated`` and ``at``; a discovery dataset carries the ground-truth
    ``target`` with its edges. Each loader's docstring lists its keys.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __dir__(self) -> list[str]:
        return [*super().__dir__(), *self.keys()]
