# SPDX-License-Identifier: Apache-2.0
"""Error messages at two known sharp edges, pinned so they keep naming the fix.

The behavior is unchanged: a ``mixed``-face graph is still refused by
identification, and a bound test still refuses a second dataset. What these
tests hold is that the refusal says what to do instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ergodic import (
    DomainKnowledge,
    discover,
    estimate_effect,
    identify_effect,
    mixed,
    tabular,
)
from ergodic.discovery import GCM, PC, DSeparationOracle
from ergodic.graph import consistent_extension, dag
from ergodic.inference.learners import OLS

PREFIX = "identification needs a verified graph face (dag, admg, mag, cpdag, or pag), not 'mixed'; "
DAG_HINT = (
    "every edge is directed and there is no directed cycle (PC returns this when "
    "domain knowledge orients every edge), so if you read it as one causal world, "
    "re-face it with graph.as_kind('dag')"
)


def _partial_hint(n: int, example: str) -> str:
    count = "1 edge is" if n == 1 else f"{n} edges are"
    return (
        f"it is partially directed ({count} undirected, such as {example!r}) and is "
        "neither a dag nor a cpdag (PC returns this when domain knowledge orients "
        "edges the data leaves open). Orient the rest with more knowledge "
        "(DomainKnowledge.require) and re-face with graph.as_kind('dag'), take one "
        "member with ergodic.graph.consistent_extension(graph) (a dag; the answer "
        "then holds for that member only), or rediscover without the orienting "
        "knowledge to keep the cpdag face"
    )


def _world() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 2000
    z = rng.normal(size=n)
    x = (0.8 * z + rng.normal(size=n) > 0).astype(float)
    y = 1.5 * x + 0.7 * z + rng.normal(size=n)
    w = 0.5 * y + rng.normal(size=n)
    return pd.DataFrame({"z": z, "x": x, "y": y, "w": w})


# -- the mixed face: identification refuses it and names the re-face ----------


def test_pc_with_orienting_knowledge_returns_mixed_and_estimation_names_as_kind_dag() -> None:
    data = tabular(_world())
    knowledge = DomainKnowledge().with_tiers([["z"], ["x"], ["y"], ["w"]])
    result = discover(data, method="pc", knowledge=knowledge)
    assert result.graph.kind.value == "mixed"
    assert result.diagnostics["cpdag_valid"] == 0.0
    with pytest.raises(ValueError) as info:
        estimate_effect(result.graph, data, "x", "y")
    assert str(info.value) == PREFIX + DAG_HINT
    # The named fix works.
    estimate = estimate_effect(result.graph.as_kind("dag"), data, "x", "y")
    assert abs(estimate.value - 1.5) < 0.2


def test_pc_with_partial_knowledge_names_the_partially_directed_options() -> None:
    data = tabular(_world())
    knowledge = DomainKnowledge().with_tiers([["z", "x"], ["y", "w"]])
    result = discover(data, method="pc", knowledge=knowledge)
    assert result.graph.kind.value == "mixed"
    with pytest.raises(ValueError) as info:
        estimate_effect(result.graph, data, "x", "y")
    assert str(info.value) == PREFIX + _partial_hint(1, "x --- z")
    member = consistent_extension(result.graph)
    assert member is not None
    estimate = estimate_effect(member, data, "x", "y")
    assert abs(estimate.value - 1.5) < 0.2
    # Without the orienting knowledge PC keeps the cpdag face.
    assert discover(data, method="pc").graph.kind.value == "cpdag"


@pytest.mark.parametrize(
    ("edges", "hint"),
    [
        (["Z -> X", "X -> Y"], DAG_HINT),
        (["Z -- X", "X -> Y", "Z -> Y"], _partial_hint(1, "X --- Z")),
        (
            ["X -- Y", "Y -- Z"],
            "it satisfies the cpdag rules, so if it stands for a Markov equivalence "
            "class, re-face it with graph.as_kind('cpdag')",
        ),
        (
            ["X -> Y", "Y -> Z", "Z -> X"],
            "no DAG has its directed edges and v-structures (a directed cycle, or "
            "orientations that clash), so the orientations and the data disagree; "
            "check the domain knowledge or rerun discovery",
        ),
        (
            ["X -> Y", "Y <-> Z"],
            "its edges are directed or bidirected with no directed cycle, so if you "
            "read it as one causal world with hidden confounders, re-face it with "
            "graph.as_kind('admg')",
        ),
        (
            ["X o-> Y", "Z o-> Y"],
            "it satisfies the pag rules, so if it stands for an equivalence class, "
            "re-face it with graph.as_kind('pag')",
        ),
        (
            ["A -- B", "B -> C", "C <-> D"],
            "it satisfies the mag rules, so if it stands for an equivalence class, "
            "re-face it with graph.as_kind('mag')",
        ),
        (
            ["X o-> Y"],
            "its marks satisfy no face's rules (FCI and RFCI return this when the data "
            "and the orientation rules disagree, with diagnostics['pag_valid'] at 0.0), "
            "so identification has no certified reading of it; rerun discovery with "
            "more data or another alpha, or write the graph you believe with dag(), "
            "admg(), or pag()",
        ),
    ],
)
def test_mixed_refusal_names_the_face_it_fits(edges: list[str], hint: str) -> None:
    graph = mixed(edges)
    nodes = sorted(graph.nodes)
    with pytest.raises(ValueError) as info:
        identify_effect(graph, nodes[0], nodes[-1])
    assert str(info.value) == PREFIX + hint


# -- a bound test takes its data in one place ----------------------------------


def test_bound_gcm_with_data_at_fit_names_both_usages() -> None:
    rng = np.random.default_rng(0)
    data = tabular({"A": rng.normal(size=40), "B": rng.normal(size=40)})
    with pytest.raises(ValueError) as info:
        PC(test=GCM(data, learner=OLS())).fit(data)
    assert str(info.value) == (
        "GCM is already bound to its own data, so it takes no data here; give the "
        "data in one place only: keep the bound test and call fit() with no data, "
        "e.g. PC(test=GCM(data, ...)).fit(), or build the test unbound and pass the "
        "data to fit or discover, e.g. PC(test=GCM(...)).fit(data) or "
        "discover(data, method='pc', test=GCM(...))"
    )
    # Both named usages run.
    assert PC(test=GCM(data, learner=OLS())).fit().graph.nodes == {"A", "B"}
    assert PC(test=GCM(learner=OLS())).fit(data).graph.nodes == {"A", "B"}
    assert discover(data, method="pc", test=GCM(learner=OLS())).graph.nodes == {"A", "B"}


def test_oracle_with_data_at_fit_says_it_needs_none() -> None:
    rng = np.random.default_rng(0)
    data = tabular({"A": rng.normal(size=8), "B": rng.normal(size=8)})
    with pytest.raises(ValueError) as info:
        PC(DSeparationOracle(dag(["A -> B"]))).fit(data=data)
    assert str(info.value) == (
        "DSeparationOracle is already bound and answers without data; call fit() with no data"
    )
