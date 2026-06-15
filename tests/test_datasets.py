# SPDX-License-Identifier: Apache-2.0
"""The bundled datasets load offline and recover their known answers.

Every loader returns a Bunch carrying the data and the causal question. These
tests run the documented analysis for each one and check it against the
literature figure in ds.target, so the datasets stay honest as the estimators
change.
"""

from __future__ import annotations

import pandas as pd

from ergodic import Mark, dag, discover, estimate_effect, synthetic_control
from ergodic import datasets as D
from ergodic.datasets import Bunch
from ergodic.graph import MixedGraph

ALL_LOADERS = [
    D.load_asia,
    D.load_earthquake,
    D.load_berkeley,
    D.load_birthweight,
    D.load_pension401k,
    D.load_smoking,
    D.load_germany,
]


def _skeleton(graph: MixedGraph) -> set[frozenset[str]]:
    return {frozenset((x, y)) for x in graph.nodes for y in graph.neighbors(x)}


def _adjustment_graph(treatment: str, outcome: str, confounders: tuple[str, ...]) -> MixedGraph:
    edges = [f"{c} -> {treatment}" for c in confounders]
    edges += [f"{c} -> {outcome}" for c in confounders]
    edges.append(f"{treatment} -> {outcome}")
    return dag(edges)


# -- the loaders are well-formed -------------------------------------------------


def test_top_level_module_is_wired() -> None:
    import ergodic

    assert ergodic.datasets.load_birthweight is D.load_birthweight


def test_every_loader_returns_a_bunch_with_the_core_keys() -> None:
    for load in ALL_LOADERS:
        ds = load()
        assert isinstance(ds, Bunch)
        assert ds.data.n_rows > 0
        assert ds.task in {"adjustment", "confounding", "synthetic_control", "discovery"}
        assert isinstance(ds.target, dict)
        assert "source" in ds.target
        assert isinstance(ds.DESCR, str) and ds.DESCR.strip()
        # attribute access and item access are the same view
        assert ds.data is ds["data"]


def test_bunch_raises_attributeerror_for_missing_keys() -> None:
    ds = D.load_berkeley()
    try:
        _ = ds.does_not_exist
    except AttributeError:
        pass
    else:  # pragma: no cover - the assertion above should fire
        raise AssertionError("expected AttributeError")


# -- adjustment and confounding --------------------------------------------------


def test_birthweight_adjustment_moves_toward_the_literature() -> None:
    ds = D.load_birthweight()
    graph = _adjustment_graph(ds.treatment, ds.outcome, ds.confounders)
    adjusted = estimate_effect(
        graph, ds.data, ds.treatment, ds.outcome, treated=ds.treated, strategy="aipw"
    )
    naive = estimate_effect(
        dag([f"{ds.treatment} -> {ds.outcome}"]),
        ds.data,
        ds.treatment,
        ds.outcome,
        treated=ds.treated,
    )
    assert abs(adjusted.value - ds.target["value"]) < ds.target["tol"]
    assert abs(naive.value - ds.target["naive"]) < 20.0
    # adjustment pulls the estimate up from the naive number
    assert adjusted.value > naive.value


def test_pension_double_ml_recovers_the_eligibility_effect() -> None:
    ds = D.load_pension401k()
    graph = _adjustment_graph(ds.treatment, ds.outcome, ds.confounders)
    dml = estimate_effect(
        graph, ds.data, ds.treatment, ds.outcome, strategy="dml", learner="knn", random_state=0
    )
    naive = estimate_effect(
        dag([f"{ds.treatment} -> {ds.outcome}"]), ds.data, ds.treatment, ds.outcome
    )
    assert abs(dml.value - ds.target["value"]) < ds.target["tol"]
    assert abs(naive.value - ds.target["naive"]) < 50.0
    # confounding by income inflates the naive gap well above the adjusted one
    assert naive.value > dml.value + 5000.0


def test_berkeley_simpsons_reversal() -> None:
    ds = D.load_berkeley()
    naive = estimate_effect(
        dag([f"{ds.treatment} -> {ds.outcome}"]),
        ds.data,
        ds.treatment,
        ds.outcome,
        treated=ds.treated,
    )
    adjusted = estimate_effect(
        dag(["dept -> gender", "dept -> admit", "gender -> admit"]),
        ds.data,
        ds.treatment,
        ds.outcome,
        treated=ds.treated,
    )
    assert abs(naive.value - ds.target["naive"]) < 0.01
    assert abs(adjusted.value - ds.target["value"]) < ds.target["tol"]
    # the pooled gap is against women; conditioning on department removes it
    assert naive.value < -0.1 < adjusted.value


# -- synthetic control -----------------------------------------------------------


def test_smoking_synthetic_control_recovers_proposition_99() -> None:
    ds = D.load_smoking()
    sc = synthetic_control(ds.data, ds.outcome, treated=ds.treated, at=ds.at)
    assert abs(sc.value - ds.target["value"]) < ds.target["tol"]
    assert sc.value < 0.0  # Proposition 99 cut sales
    assert sc.p_value is not None and sc.p_value <= 0.1  # California is extreme among placebos
    # the donor weights live on the simplex
    assert abs(sum(sc.weights.values()) - 1.0) < 1e-6


def test_germany_synthetic_control_is_negative() -> None:
    ds = D.load_germany()
    sc = synthetic_control(ds.data, ds.outcome, treated=ds.treated, at=ds.at)
    assert abs(sc.value - ds.target["value"]) < ds.target["tol"]
    assert sc.value < 0.0


# -- discovery -------------------------------------------------------------------


def test_asia_discovery_recovers_the_strong_edges() -> None:
    ds = D.load_asia()
    truth = _skeleton(dag(list(ds.target["edges"])))
    found = _skeleton(discover(ds.data, method="pc").graph)
    recall = len(truth & found) / len(truth)
    precision = len(truth & found) / max(1, len(found))
    assert recall >= ds.target["skeleton_recall_floor"]
    assert precision >= 0.9  # PC adds no spurious edges here


def test_earthquake_discovery_orients_the_collider() -> None:
    ds = D.load_earthquake()
    result = discover(ds.data, method="pc")
    truth = _skeleton(dag(list(ds.target["edges"])))
    found = _skeleton(result.graph)
    assert len(truth & found) / len(truth) >= ds.target["skeleton_recall_floor"]
    # Burglary -> Alarm <- Earthquake: both arrowheads at Alarm
    assert result.graph.mark_at("Burglary", "Alarm", "Alarm") is Mark.ARROW
    assert result.graph.mark_at("Earthquake", "Alarm", "Alarm") is Mark.ARROW


# -- the generated data is deterministic -----------------------------------------


def test_bayesnet_sampler_is_deterministic() -> None:
    a = D.load_asia(n=500, seed=7).data.to_pandas()
    b = D.load_asia(n=500, seed=7).data.to_pandas()
    pd.testing.assert_frame_equal(a, b)
    # a different seed gives different data
    c = D.load_asia(n=500, seed=8).data.to_pandas()
    assert not a.equals(c)
