"""API-248: trustworthy semantics, not a promise of universal model quality."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from ergodic import dag
from ergodic.decision import decide
from ergodic.decision._ranking import ProbabilityMean, assess_targeting, top_weights


def world(seed: int = 42, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    a = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, 0.4 + a * 0.2 * np.sign(x))
    return pd.DataFrame({"X": x, "A": a, "Y": y})


def test_fractional_ties_and_constant_scores() -> None:
    w = top_weights(np.array([3, 3, 3, 0, 0, 0, 0]))
    assert w.sum() == pytest.approx(1.4)
    assert np.allclose(w[:3], 1.4 / 3)
    assert np.allclose(top_weights(np.ones(11)), 0.2)


def test_zero_policy_difference_has_exact_zero_bound() -> None:
    scores = np.arange(100.0)
    r = assess_targeting(
        {"s": scores, "dr": scores},
        np.arange(100) % 2,
        np.ones(100),
        probability=0.5,
        family_size=3,
    )
    assert r["paired_dr_minus_s"]["interval"] == [0, 0]
    assert r["selected"] == "s"
    assert r["paired_dr_minus_s"]["status"] == "inconclusive"


def test_null_outcomes_never_promote_positive_targeting() -> None:
    rng = np.random.default_rng(1)
    for _ in range(10):
        scores = rng.normal(size=200)
        r = assess_targeting(
            {"s": scores, "dr": -scores},
            rng.binomial(1, 0.5, 200),
            np.zeros(200),
            probability=0.5,
            family_size=3,
        )
        assert r["selected"] == "s"
        assert all(v["status"] == "inconclusive" for v in r["candidates"].values())


def test_v2_defaults_and_complement_preserve_legacy() -> None:
    d = world()
    q = pd.DataFrame({"X": [-0.5, 0.5]})
    kwargs: dict[str, Any] = {
        "target": "Y",
        "target_kind": "binary",
        "interventions": {"A": {"kind": "binary"}},
        "rows": q,
        "n_folds": 3,
        "types": {"X": "continuous"},
    }
    graph = dag(["X -> Y", "A -> Y"])
    legacy = decide(graph, d, **kwargs)
    v2 = decide(graph, d, protocol="decision-contrasts_v2", **kwargs)
    assert legacy["individual"][0]["interventions"][0]["method"] == "dr"
    assert v2["population"][0]["method"] == "dr"
    for row in v2["individual"]:
        item = row["interventions"][0]
        assert item["method"] == "s"
        assert sum(c["effect"] for c in item["class_effects"]) == 0
        assert item["diagnostics"]["ranking_validation"]["status"] == "not_assessed"
    assert v2["population"][0]["diagnostics"]["binary_complement_derived"]


def test_validation_outcomes_cannot_change_fits_or_explicit_s_predictions() -> None:
    graph = dag(["X -> Y", "A -> Y"])
    d = world()
    v = world(3, 200)
    # Keep validation within frozen context support, without fitting on it.
    v["X"] = v.X.clip(d.X.min(), d.X.max())
    d["unit"] = np.arange(len(d))
    v["unit"] = np.arange(len(d), len(d) + len(v))
    kwargs: dict[str, Any] = {
        "target": "Y",
        "target_kind": "binary",
        "interventions": {"A": {"kind": "binary", "method": "s", "validation_probability": 0.5}},
        "rows": pd.DataFrame({"X": [-0.5, 0.5]}),
        "n_folds": 3,
        "types": {"X": "continuous"},
        "protocol": "decision-contrasts_v2",
        "validation_design": "independent_bernoulli",
        "unit_id": "unit",
    }
    one = decide(graph, d, validation=v, **kwargs)
    v["Y"] = 1 - v.Y
    two = decide(graph, d, validation=v, **kwargs)
    assert [r["interventions"][0]["effect"] for r in one["individual"]] == [
        r["interventions"][0]["effect"] for r in two["individual"]
    ]
    assert one["population"][0]["effect"] == two["population"][0]["effect"]


def test_probability_mean_respects_reversed_class_order() -> None:
    class Classifier:
        classes_ = np.array([1, 0])

        def fit(self, x: Any, y: Any) -> Any:
            return self

        def predict_proba(self, x: Any) -> Any:
            return np.tile([0.8, 0.2], (len(x), 1))

        def get_params(self, deep: bool = True) -> dict[str, Any]:
            return {}

    m = ProbabilityMean(Classifier()).fit(np.ones((2, 1)), np.array([0, 1]))
    assert np.allclose(m.predict(np.ones((3, 1))), 0.8)


def test_gate_support_against_and_minimize() -> None:
    # Fixed finite cohort, very strong binary potential-outcome heterogeneity.
    rng = np.random.default_rng(3)
    n = 20000
    x = np.linspace(-1, 1, n)
    a = rng.binomial(1, 0.5, n)
    y = a * (x > 0.6)
    good = assess_targeting({"s": -x, "dr": x}, a, y, probability=0.5, family_size=3)
    assert good["selected"] == "dr"
    assert good["candidates"]["dr"]["status"] == "supported"
    assert good["candidates"]["s"]["status"] == "evidence_against"
    reverse = assess_targeting(
        {"s": -x, "dr": x}, a, y, probability=0.5, family_size=3, direction=-1
    )
    assert reverse["candidates"]["s"]["status"] == "evidence_against"


def test_training_validation_overlap_is_refused() -> None:
    d = world()
    d["unit"] = np.arange(len(d))
    with pytest.raises(ValueError, match="disjoint unit"):
        decide(
            dag(["X -> Y", "A -> Y"]),
            d,
            target="Y",
            target_kind="binary",
            interventions={"A": {"kind": "binary"}},
            protocol="decision-contrasts_v2",
            validation=d.copy(),
            validation_design="independent_bernoulli",
            unit_id="unit",
        )


def test_null_graph_resolves_auto_without_false_validation() -> None:
    d = world()
    v = world(2, 100)
    d["unit"] = np.arange(len(d))
    v["unit"] = np.arange(1000, 1100)
    v["X"] = v.X.clip(d.X.min(), d.X.max())
    # A has no directed path to Y; graph identifies zero.
    graph = dag(["X -> A", "X -> Y"])
    result = decide(
        graph,
        d,
        target="Y",
        target_kind="binary",
        interventions={"A": {"kind": "binary", "method": "auto", "validation_probability": 0.5}},
        protocol="decision-contrasts_v2",
        validation=v,
        unit_id="unit",
        validation_design="independent_bernoulli",
        rows=pd.DataFrame({"X": [0.0]}),
    )
    item = result["individual"][0]["interventions"][0]
    assert item["method"] == "s"
    assert item["effect"] == 0
    assert item["diagnostics"]["ranking_validation"]["status"] == "not_assessed"


def test_reversed_contrast_probability_and_event() -> None:
    a = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), 1000)
    y = a.copy()
    s = np.arange(len(a), dtype=float)
    one = assess_targeting({"s": s}, a, y, probability=0.25, family_size=3)
    # Reversing assignment and utility restores the same targeting estimand.
    reverse = assess_targeting({"s": -s}, 1 - a, y, probability=0.75, family_size=3, direction=-1)
    assert one["candidates"]["s"]["gain"] == pytest.approx(reverse["candidates"]["s"]["gain"])
    assert one["candidates"]["s"]["interval"] == pytest.approx(
        reverse["candidates"]["s"]["interval"]
    )
