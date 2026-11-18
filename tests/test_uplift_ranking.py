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
        default="s",
    )
    assert r["paired_minus_default"]["dr"]["interval"] == [0, 0]
    assert r["selected"] == "s"
    assert r["paired_minus_default"]["dr"]["status"] == "inconclusive"


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
            default="s",
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
        assert item["method"] == "plugin"
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
    good = assess_targeting({"s": -x, "dr": x}, a, y, probability=0.5, family_size=3, default="s")
    assert good["selected"] == "dr"
    assert good["candidates"]["dr"]["status"] == "supported"
    assert good["candidates"]["s"]["status"] == "evidence_against"
    reverse = assess_targeting(
        {"s": -x, "dr": x}, a, y, probability=0.5, family_size=3, direction=-1, default="s"
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
    assert item["method"] == "plugin"
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


@pytest.mark.parametrize("target_kind", ["continuous", "categorical"])
def test_v2_preserves_other_target_defaults(target_kind: str) -> None:
    d = world()
    d["Y"] = 2 * d.A + d.X if target_kind == "continuous" else np.digitize(d.X, [-0.3, 0.3])
    result = decide(
        dag(["X -> Y", "A -> Y"]),
        d,
        target="Y",
        target_kind=target_kind,
        target_class=2 if target_kind == "categorical" else None,
        interventions={"A": {"kind": "binary"}},
        rows=pd.DataFrame({"X": [-0.5, 0.5]}),
        protocol="decision-contrasts_v2",
        n_folds=3,
    )
    assert not result["skipped"]
    assert result["individual"][0]["interventions"][0]["method"] == "dr"
    if target_kind == "categorical":
        assert result["population"][0]["diagnostics"]["class_contrast_zero_sum_projection"] == 1
        assert "binary_complement_derived" not in result["population"][0]["diagnostics"]


def test_plugin_is_the_fold_averaged_arm_model_difference() -> None:
    from ergodic.decision._categorical import ArmDR
    from ergodic.inference.learners import OLS

    d = world(5, 400)
    x, a, y = d[["X"]].to_numpy(), d.A.to_numpy(float), d.Y.to_numpy(float)
    q = np.array([[-0.5], [0.0], [0.5]])
    est = ArmDR(outcome=OLS(), n_folds=4).fit(x, a, y)
    point, se, plugin = est.contrast(0, 1, q, method="plugin")
    expected = np.mean([m[1].predict(q) - m[0].predict(q) for m in est._outcomes], axis=0)
    assert plugin is not None
    assert np.allclose(plugin, expected)
    assert len(est._outcomes) == 4
    # Only the query effects change with the method; the population estimate is the DR mean.
    assert est.contrast(0, 1, q, method="dr")[:2] == (point, se)
    assert est.contrast(0, 1, q, method="s")[:2] == (point, se)
    with pytest.raises(ValueError, match="plugin, s or dr"):
        est.contrast(0, 1, q, method="toggle")


def test_plugin_ranks_heterogeneous_effects() -> None:
    from scipy.stats import spearmanr

    rng = np.random.default_rng(7)
    n = 3000
    x = rng.uniform(-1, 1, n)
    a = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, np.clip(0.4 + a * 0.3 * x, 0, 1))
    grid = pd.DataFrame({"X": np.linspace(-0.9, 0.9, 19)})
    result = decide(
        dag(["X -> Y", "A -> Y"]),
        pd.DataFrame({"X": x, "A": a, "Y": y}),
        target="Y",
        target_kind="binary",
        target_class=1,
        interventions={"A": {"kind": "binary"}},
        rows=grid,
        n_folds=3,
        types={"X": "continuous"},
        protocol="decision-contrasts_v2",
    )
    effects = [r["interventions"][0]["effect"] for r in result["individual"]]
    assert all(r["interventions"][0]["method"] == "plugin" for r in result["individual"])
    assert spearmanr(effects, grid.X).statistic > 0.9


def test_auto_keeps_default_unless_another_method_is_proven_better() -> None:
    rng = np.random.default_rng(3)
    n = 20000
    x = np.linspace(-1, 1, n)
    a = rng.binomial(1, 0.5, n)
    y = a * (x > 0.6)
    noise = rng.normal(size=n)
    kept = assess_targeting(
        {"plugin": x, "s": -x, "dr": noise}, a, y, probability=0.5, family_size=5
    )
    assert kept["default"] == "plugin" and kept["selected"] == "plugin"
    assert set(kept["paired_minus_default"]) == {"s", "dr"}
    switched = assess_targeting(
        {"plugin": -x, "s": noise, "dr": x}, a, y, probability=0.5, family_size=5
    )
    assert switched["selected"] == "dr"
    assert switched["paired_minus_default"]["dr"]["interval"][0] > 0
    # Two qualifying alternatives: the larger paired lower bound wins.
    both = assess_targeting(
        {"plugin": -x, "s": x + 0.3 * noise, "dr": x}, a, y, probability=0.5, family_size=5
    )
    assert both["selected"] == "dr"
    with pytest.raises(ValueError, match="default ranking method"):
        assess_targeting({"s": x, "dr": x}, a, y, probability=0.5, family_size=5)


def test_v2_auto_reports_all_three_candidates_and_family_size() -> None:
    graph = dag(["X -> Y", "A -> Y"])
    d = world()
    v = world(3, 300)
    v["X"] = v.X.clip(d.X.min(), d.X.max())
    d["unit"] = np.arange(len(d))
    v["unit"] = np.arange(len(d), len(d) + len(v))
    result = decide(
        graph,
        d,
        target="Y",
        target_kind="binary",
        interventions={"A": {"kind": "binary", "method": "auto", "validation_probability": 0.5}},
        rows=pd.DataFrame({"X": [-0.5, 0.5]}),
        n_folds=3,
        types={"X": "continuous"},
        protocol="decision-contrasts_v2",
        validation=v,
        validation_design="independent_bernoulli",
        unit_id="unit",
    )
    quality = result["population"][0]["diagnostics"]["ranking_validation"]
    assert set(quality["candidates"]) == {"plugin", "s", "dr"}
    assert quality["default"] == "plugin"
    assert quality["family_size"] == 5
    assert result["individual"][0]["interventions"][0]["method"] == quality["selected"]


def test_plugin_requires_v2() -> None:
    with pytest.raises(ValueError, match="ranking method for this protocol"):
        decide(
            dag(["X -> Y", "A -> Y"]),
            world(),
            target="Y",
            target_kind="binary",
            interventions={"A": {"kind": "binary", "method": "plugin"}},
        )


def test_protocols_and_ranking_methods_are_exported() -> None:
    from ergodic import decision

    assert decision.PROTOCOLS == ("decision-contrasts_v1", "decision-contrasts_v2")
    assert decision.RANKING_METHODS == ("plugin", "s", "dr")


def test_plugin_multiarm_reversed_refit_and_support() -> None:
    from ergodic.decision._categorical import ArmDR
    from ergodic.inference.learners import OLS

    rng = np.random.default_rng(11)
    n = 1500
    x = rng.uniform(-1, 1, (n, 1))
    a = rng.integers(0, 3, n).astype(float)
    y = (a == 1) * 0.5 + (a == 2) * 2 * x[:, 0] + rng.normal(0, 0.1, n)
    q = np.array([[-0.5], [0.5]])
    est = ArmDR(outcome=OLS(), n_folds=3).fit(x, a, y)
    assert len(est._outcomes) == 3 and all(len(m) == 3 for m in est._outcomes)
    forward = est.contrast(0, 2, q, method="plugin")[2]
    backward = est.contrast(2, 0, q, method="plugin")[2]
    assert forward is not None and backward is not None
    assert np.allclose(forward, 2 * q[:, 0], atol=0.1)
    assert np.allclose(backward, -forward)
    middle = est.contrast(2, 1, q, method="plugin")[2]
    assert middle is not None and np.allclose(middle, 0.5 - 2 * q[:, 0], atol=0.1)
    with pytest.raises(ValueError, match="outside observed context support"):
        est.contrast(0, 2, np.array([[3.0]]), method="plugin")
    est.n_folds = 4
    est.fit(x, a, y)
    assert len(est._outcomes) == 4


def test_plugin_with_probability_means_gives_probability_differences() -> None:
    from ergodic.decision._categorical import ArmDR

    d = world(8, 800)
    x, a, y = d[["X"]].to_numpy(), d.A.to_numpy(float), d.Y.to_numpy(float)
    est = ArmDR(outcome=ProbabilityMean(), n_folds=3).fit(x, a, y)
    effects = est.contrast(0, 1, np.array([[-0.8], [0.8]]), method="plugin")[2]
    assert effects is not None
    assert np.all(np.abs(effects) <= 1)
    assert effects[1] > effects[0]  # the world's effect is +0.2 for x > 0 and -0.2 below


def test_family_size_must_count_every_reported_bound() -> None:
    x = np.linspace(-1, 1, 100)
    a, y = np.arange(100) % 2, np.ones(100)
    with pytest.raises(ValueError, match="every reported bound"):
        assess_targeting({"plugin": x, "s": x, "dr": x}, a, y, probability=0.5, family_size=4)
    one = assess_targeting({"plugin": x}, a, y, probability=0.5, family_size=1)
    assert one["selected"] == "plugin" and one["paired_minus_default"] == {}


@pytest.mark.parametrize(
    "spec,family",
    [
        ({"kind": "binary", "method": "auto", "validation_probability": 0.5}, 5),
        ({"kind": "binary", "validation_probability": 0.5}, 1),
        ({"kind": "binary"}, 0),
    ],
)
def test_family_counts_only_bounds_that_may_be_reported(spec: dict[str, Any], family: int) -> None:
    d = world()
    v = world(3, 300)
    v["X"] = v.X.clip(d.X.min(), d.X.max())
    d["unit"] = np.arange(len(d))
    v["unit"] = np.arange(len(d), len(d) + len(v))
    result = decide(
        dag(["X -> Y", "A -> Y"]),
        d,
        target="Y",
        target_kind="binary",
        interventions={"A": spec},
        rows=pd.DataFrame({"X": [-0.5, 0.5]}),
        n_folds=3,
        types={"X": "continuous"},
        protocol="decision-contrasts_v2",
        validation=v,
        validation_design="independent_bernoulli",
        unit_id="unit",
    )
    assert result["configuration"]["validation_family_size"] == family
    quality = result["population"][0]["diagnostics"]["ranking_validation"]
    if family == 1:
        assert set(quality["candidates"]) == {"plugin"}
        assert quality["paired_minus_default"] == {}
    if family == 0:
        assert quality["status"] == "not_assessed"
