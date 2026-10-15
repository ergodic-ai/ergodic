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
from ergodic.discovery import BIC, FCI, GCM, GES, PC, DSeparationOracle
from ergodic.graph import consistent_extension, dag
from ergodic.inference.learners import OLS

PREFIX = "identification needs a verified graph face (dag, admg, mag, cpdag, or pag), not 'mixed'; "
MAG_READING = (
    "; if it came from FCI or RFCI (or otherwise means ancestry, where a directed "
    "edge can hide confounding), re-face it with graph.as_kind('mag') instead, which "
    "identification reads as an equivalence class"
)
DAG_HINT = (
    "every edge is directed and there is no directed cycle. If it came from a search "
    "that assumes no hidden confounding (PC or GES with domain knowledge orienting "
    "every edge, say) and you read it as one causal world, re-face it with "
    "graph.as_kind('dag')" + MAG_READING
)
NO_FACE_HINT = (
    "its marks satisfy no face's rules. If the search ran with domain knowledge, the "
    "marks the knowledge seeded are the likely cause (FCI and RFCI keep them even "
    "where the class disagrees, and report diagnostics['pag_valid'] at 0.0): rerun "
    "without that knowledge, which usually returns the class face. Otherwise the data "
    "and the orientation rules disagree: rerun with more data or another alpha, or "
    "write the graph you believe with dag(), admg(), or pag()"
)


def _partial_hint(n: int, example: str) -> str:
    count = "1 edge is" if n == 1 else f"{n} edges are"
    return (
        f"it is partially directed ({count} undirected, such as {example!r}) and is "
        "neither a dag nor a cpdag, which is what a search such as PC can return when "
        "domain knowledge orients edges the data leaves open. Orient the rest with "
        "more knowledge (DomainKnowledge.require) and re-face with "
        "graph.as_kind('dag'), take one member with "
        "ergodic.graph.consistent_extension(graph) (a dag; the answer then holds for "
        "that member only), or rediscover without the orienting knowledge, which "
        "usually keeps the cpdag face"
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


def _confounded_world() -> pd.DataFrame:
    # A latent L confounds x and y; the true effect of x on y is 1.0.
    rng = np.random.default_rng(0)
    n = 4000
    latent = rng.normal(size=n)
    z = rng.normal(size=n)
    w = rng.normal(size=n)
    x = z + latent + rng.normal(size=n)
    y = 1.0 * x + latent + w + rng.normal(size=n)
    return pd.DataFrame({"z": z, "x": x, "y": y, "w": w})


@pytest.mark.parametrize("method", ["fci", "rfci"])
def test_fci_with_required_edges_points_at_the_mag_reading(method: str) -> None:
    # Review finding F1: knowledge seeds directed edges into an FCI/RFCI result,
    # which stays mixed. Its edges mean ancestry, so the DAG reading is unsound
    # (it adjusts a confounded effect to about 1.5) and the MAG reading is right.
    data = tabular(_confounded_world())
    knowledge = DomainKnowledge().require("z", "x").require("x", "y").require("w", "y")
    result = discover(data, method=method, knowledge=knowledge)
    assert result.graph.kind.value == "mixed"
    assert result.diagnostics["pag_valid"] == 0.0
    with pytest.raises(ValueError) as info:
        identify_effect(result.graph, "x", "y")
    assert str(info.value) == PREFIX + DAG_HINT
    assert identify_effect(result.graph.as_kind("mag"), "x", "y").strategy == "none"
    unsound = estimate_effect(result.graph.as_kind("dag"), data, "x", "y", strategy="dml")
    assert unsound.value > 1.3  # the bias the MAG reading guards against


def test_fci_with_tiers_that_fit_no_face_names_the_knowledge() -> None:
    # Review finding F4: strong data, tier knowledge, and FCI keeps a seeded mark.
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n)
    m = 0.8 * x + rng.normal(size=n)
    y = 0.7 * m + rng.normal(size=n)
    data = tabular(pd.DataFrame({"X": x, "M": m, "Y": y}))
    knowledge = DomainKnowledge().with_tiers([["X"], ["M"], ["Y"]])
    result = discover(data, method="fci", knowledge=knowledge)
    assert result.graph.kind.value == "mixed"
    with pytest.raises(ValueError) as info:
        identify_effect(result.graph, "X", "Y")
    assert str(info.value) == PREFIX + NO_FACE_HINT
    assert discover(data, method="fci").graph.kind.value == "pag"


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
            "its edges are directed or bidirected with no directed cycle. If you read "
            "it as one causal world whose only hidden confounding is the drawn <-> "
            "edges, re-face it with graph.as_kind('admg')" + MAG_READING,
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
            NO_FACE_HINT,
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


def _bound_message(algorithm: str) -> str:
    return (
        "GCM is already bound to its own data, but this run was handed data too (by "
        "fit, or by discover, which always passes it); give the data in one place "
        f"only: keep the bound test and fit with no data, e.g. {algorithm}(test=GCM("
        f"data, ...)).fit(), or build the test unbound and pass the data, e.g. "
        f"{algorithm}(test=GCM(...)).fit(data) or discover(data, method=..., "
        "test=GCM(...))"
    )


def test_bound_gcm_with_data_at_fit_names_both_usages() -> None:
    rng = np.random.default_rng(0)
    data = tabular({"A": rng.normal(size=40), "B": rng.normal(size=40)})
    with pytest.raises(ValueError) as info:
        PC(test=GCM(data, learner=OLS())).fit(data)
    assert str(info.value) == _bound_message("PC")
    with pytest.raises(ValueError) as info:
        FCI(test=GCM(data, learner=OLS())).fit(data)
    assert str(info.value) == _bound_message("FCI")
    with pytest.raises(ValueError) as info:
        discover(data, method="fci", test=GCM(data, learner=OLS()))
    assert str(info.value) == _bound_message("FCI")
    # Both named usages run.
    assert PC(test=GCM(data, learner=OLS())).fit().graph.nodes == {"A", "B"}
    assert PC(test=GCM(learner=OLS())).fit(data).graph.nodes == {"A", "B"}
    assert discover(data, method="pc", test=GCM(learner=OLS())).graph.nodes == {"A", "B"}


def test_oracle_with_data_says_it_needs_none() -> None:
    rng = np.random.default_rng(0)
    data = tabular({"A": rng.normal(size=8), "B": rng.normal(size=8)})
    oracle = DSeparationOracle(dag(["A -> B"]))
    expected = (
        "DSeparationOracle answers from its graph and takes no data, but this run was "
        "handed data (by fit, or by discover, which always passes it); run the "
        "algorithm with no data, e.g. {}(DSeparationOracle(graph)).fit()"
    )
    with pytest.raises(ValueError) as info:
        PC(oracle).fit(data=data)
    assert str(info.value) == expected.format("PC")
    with pytest.raises(ValueError) as info:
        discover(data, method="fci", test=oracle)
    assert str(info.value) == expected.format("FCI")
    assert PC(oracle).fit().graph.nodes == {"A", "B"}


def test_bound_score_with_other_data_names_both_usages() -> None:
    rng = np.random.default_rng(0)
    data = tabular({"A": rng.normal(size=40), "B": rng.normal(size=40)})
    other = tabular({"A": rng.normal(size=30), "B": rng.normal(size=30)})
    with pytest.raises(ValueError) as info:
        GES(score=BIC(data)).fit(other)
    assert str(info.value) == (
        "BIC is bound to a different dataset than the one this run was handed (rows "
        "or variables differ after dropna); give the data in one place only: keep the "
        "bound score and fit with no data, e.g. GES(score=BIC(data)).fit(), or build "
        "the score unbound and pass the data, e.g. GES(score=BIC()).fit(data) or "
        "discover(data, method=..., score=BIC())"
    )
    assert GES(score=BIC(data)).fit().graph.nodes == {"A", "B"}
    assert GES(score=BIC()).fit(data).graph.nodes == {"A", "B"}
