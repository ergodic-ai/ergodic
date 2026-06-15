# SPDX-License-Identifier: Apache-2.0
"""Small canonical Bayesian networks, written from their source papers, with a sampler.

Rather than redistribute someone else's data file, we write the network down (its
structure and its conditional tables) and draw samples from it. That keeps the
data ours and lets a reader pick the sample size, so they can watch a discovery
algorithm sharpen as the data grows. The structures are the textbook ones: the
Asia chest-clinic network (Lauritzen and Spiegelhalter, 1988) and the burglar-alarm
network (Pearl, 1988; the discrete "earthquake" net).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..data import tabular
from ._bunch import Bunch

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# A network is an ordered (topological) tuple of nodes. Each node is its name, its
# parents, its levels, and a table from a parent-value tuple to a probability over
# the levels. Probabilities are (P(level0), P(level1), ...) in the order of levels.
_Node = tuple[str, tuple[str, ...], tuple[str, ...], "Mapping[tuple[str, ...], tuple[float, ...]]"]

_ASIA: tuple[_Node, ...] = (
    ("asia", (), ("no", "yes"), {(): (0.99, 0.01)}),
    ("tub", ("asia",), ("no", "yes"), {("no",): (0.99, 0.01), ("yes",): (0.95, 0.05)}),
    ("smoke", (), ("no", "yes"), {(): (0.5, 0.5)}),
    ("lung", ("smoke",), ("no", "yes"), {("no",): (0.99, 0.01), ("yes",): (0.90, 0.10)}),
    ("bronc", ("smoke",), ("no", "yes"), {("no",): (0.70, 0.30), ("yes",): (0.40, 0.60)}),
    (
        "either",
        ("tub", "lung"),
        ("no", "yes"),
        {
            ("no", "no"): (1.0, 0.0),
            ("no", "yes"): (0.0, 1.0),
            ("yes", "no"): (0.0, 1.0),
            ("yes", "yes"): (0.0, 1.0),
        },
    ),
    ("xray", ("either",), ("no", "yes"), {("no",): (0.95, 0.05), ("yes",): (0.02, 0.98)}),
    (
        "dysp",
        ("either", "bronc"),
        ("no", "yes"),
        {
            ("no", "no"): (0.90, 0.10),
            ("no", "yes"): (0.20, 0.80),
            ("yes", "no"): (0.30, 0.70),
            ("yes", "yes"): (0.10, 0.90),
        },
    ),
)

_ASIA_EDGES = (
    "asia -> tub",
    "smoke -> lung",
    "smoke -> bronc",
    "tub -> either",
    "lung -> either",
    "either -> xray",
    "either -> dysp",
    "bronc -> dysp",
)

_EARTHQUAKE: tuple[_Node, ...] = (
    ("Burglary", (), ("False", "True"), {(): (0.99, 0.01)}),
    ("Earthquake", (), ("False", "True"), {(): (0.98, 0.02)}),
    (
        "Alarm",
        ("Burglary", "Earthquake"),
        ("False", "True"),
        {
            ("True", "True"): (0.05, 0.95),
            ("True", "False"): (0.06, 0.94),
            ("False", "True"): (0.71, 0.29),
            ("False", "False"): (0.999, 0.001),
        },
    ),
    (
        "JohnCalls",
        ("Alarm",),
        ("False", "True"),
        {("True",): (0.10, 0.90), ("False",): (0.95, 0.05)},
    ),
    (
        "MaryCalls",
        ("Alarm",),
        ("False", "True"),
        {("True",): (0.30, 0.70), ("False",): (0.99, 0.01)},
    ),
)

_EARTHQUAKE_EDGES = (
    "Burglary -> Alarm",
    "Earthquake -> Alarm",
    "Alarm -> JohnCalls",
    "Alarm -> MaryCalls",
)


def _sample_bn(nodes: Sequence[_Node], n: int, seed: int) -> pd.DataFrame:
    """Draw ``n`` rows by ancestral sampling in topological order."""
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    rng = np.random.default_rng(seed)
    cols: dict[str, np.ndarray] = {}
    for name, parents, levels, cpt in nodes:
        out = np.empty(n, dtype=object)
        if not parents:
            out[:] = rng.choice(levels, size=n, p=list(cpt[()]))
        else:
            parent_cols = [cols[p] for p in parents]
            configs = [tuple(str(v) for v in row) for row in zip(*parent_cols, strict=True)]
            groups: dict[tuple[str, ...], list[int]] = {}
            for i, cfg in enumerate(configs):
                groups.setdefault(cfg, []).append(i)
            for cfg, idxs in groups.items():
                out[np.asarray(idxs)] = rng.choice(levels, size=len(idxs), p=list(cpt[cfg]))
        cols[name] = out
    return pd.DataFrame(cols)


def _bayesnet(
    nodes: Sequence[_Node],
    edges: tuple[str, ...],
    n: int,
    seed: int,
    descr: str,
    source: str,
) -> Bunch:
    df = _sample_bn(nodes, n, seed)
    names = tuple(df.columns)
    data = tabular(df, types=dict.fromkeys(names, "categorical"))
    return Bunch(
        data=data,
        task="discovery",
        target={
            "estimand": "structure recovery",
            "nodes": names,
            "edges": edges,
            "n_edges": len(edges),
            "skeleton_recall_floor": 0.5,
            "source": source,
        },
        DESCR=descr,
    )


_ASIA_DESCR = """\
Asia: the chest-clinic teaching network (Lauritzen and Spiegelhalter, 1988).

Eight binary variables: a visit to Asia, tuberculosis, smoking, lung cancer,
bronchitis, a logical-or "either tuberculosis or lung cancer", a chest x-ray, and
dyspnoea. The data are sampled from the network's conditional tables, so the true
graph is known exactly.

Task: recover the structure. The teaching points are the logical-or collider at
"either" and the way smoking explains away across lung cancer and bronchitis. Run
discover(ds.data, method="pc") and score the result against the true edges in
ds.target["edges"], comparing skeletons through cpdag_of_dag so equivalence-class
edges are not penalized.
"""

_EARTHQUAKE_DESCR = """\
Earthquake: the burglar-alarm network (Pearl, 1988).

Five binary variables: a burglary, an earthquake, an alarm, and two neighbors
(John and Mary) who may call. A burglary or an earthquake can set off the alarm,
and the alarm is what prompts a call. The data are sampled from the network, so
the true graph is known exactly.

Task: recover the structure. The single clean lesson is the v-structure
Burglary -> Alarm <- Earthquake (a collider, and explaining away). Run
discover(ds.data, method="pc") and check that the recovered graph orients that
collider; the edges live in ds.target["edges"].
"""


def load_asia(n: int = 3000, seed: int = 0) -> Bunch:
    """The Asia chest-clinic network as sampled data with the true graph.

    Args:
        n: How many rows to draw.
        seed: The random seed; the draw is deterministic given ``n`` and ``seed``.

    Returns a :class:`Bunch` with ``data`` (a :class:`TabularData` of eight
    categorical columns) and ``target`` carrying the node names, the true edges,
    and a skeleton-recall floor. See the module description for the lesson.
    """
    return _bayesnet(_ASIA, _ASIA_EDGES, n, seed, _ASIA_DESCR, "Lauritzen and Spiegelhalter (1988)")


def load_earthquake(n: int = 3000, seed: int = 0) -> Bunch:
    """The burglar-alarm network as sampled data with the true graph.

    Args:
        n: How many rows to draw.
        seed: The random seed; the draw is deterministic given ``n`` and ``seed``.

    Returns a :class:`Bunch` with ``data`` (five categorical columns) and
    ``target`` carrying the true v-structure. See the module description.
    """
    return _bayesnet(_EARTHQUAKE, _EARTHQUAKE_EDGES, n, seed, _EARTHQUAKE_DESCR, "Pearl (1988)")
