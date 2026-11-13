# SPDX-License-Identifier: Apache-2.0
"""API-233 frozen synthetic matrix: five seeds, nine treatment x target types.

6000 independent context rows, 100 query rows. Assignment confounded by Z.
Continuous treatment: A=.3Z+U[-1,1]; categorical: softmax(.3Z * [0,1,2]).
Regression mean Z + (1+.2Z)A for discrete treatments, Z+A for continuous.
Binary probability .4+.10A+.03Z; multiclass probabilities
[.3+.08A+.02Z, .3-.04A-.01Z, remainder]. Probability effects are homogeneous.
Thresholds fixed before qualification: mean bias <.12 regression/.035 probability,
query RMSE <.25 regression/.07 probability; class contrast sums <1e-8 for OLS.
This is synthetic regression evidence, not broad model or interval calibration.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from ergodic import dag
from ergodic.decision import ArmDR, Intervention, decide

KINDS = ("continuous", "binary", "categorical")
GRAPH = dag(["Z -> A", "Z -> Y", "A -> Y"])


def world(
    seed: int, treatment: str, outcome: str, n: int = 6000
) -> tuple[pd.DataFrame, Any, Any, Any, npt.NDArray[np.float64]]:
    baseline: Any
    value: Any
    event: Any
    y: npt.NDArray[Any]
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1, 1, n)
    if treatment == "continuous":
        a = 0.3 * z + rng.uniform(-1, 1, n)
        labels = a
        baseline, value = -0.25, 0.25
    else:
        k = 2 if treatment == "binary" else 3
        p = np.exp(0.3 * z[:, None] * np.arange(k)[None, :])
        p /= p.sum(1)[:, None]
        a = (rng.uniform(size=n)[:, None] > p.cumsum(1)).sum(1)
        labels = a.astype(bool) if k == 2 else np.asarray(["control", "low", "high"])[a]
        baseline, value = (False, True) if k == 2 else ("control", "low")
    if outcome == "continuous":
        tau = np.ones(n) if treatment == "continuous" else 1 + 0.2 * z
        y = z + tau * a + rng.normal(0, 0.5, n)
        event = None
        truth = tau * (value - baseline if treatment == "continuous" else 1)
    elif outcome == "binary":
        y = rng.uniform(size=n) < 0.4 + 0.10 * a + 0.03 * z
        event = True
        truth = np.full(n, 0.10 * (value - baseline if treatment == "continuous" else 1))
    else:
        p = np.column_stack([0.3 + 0.08 * a + 0.02 * z, 0.3 - 0.04 * a - 0.01 * z])
        p = np.column_stack([p, 1 - p.sum(1)])
        y = np.asarray(["good", "bad", "other"])[
            (rng.uniform(size=n)[:, None] > p.cumsum(1)).sum(1)
        ]
        event = "good"
        truth = np.full(n, 0.08 * (value - baseline if treatment == "continuous" else 1))
    return pd.DataFrame({"Z": z, "A": labels, "Y": y}), baseline, value, event, truth


@pytest.mark.parametrize("treatment", KINDS)
@pytest.mark.parametrize("outcome", KINDS)
def test_nine_type_matrix(treatment: str, outcome: str) -> None:
    biases, errors = [], []
    for seed in range(5):
        frame, baseline, value, event, truth = world(seed, treatment, outcome)
        query, *_ = world(seed + 100, treatment, outcome, n=100)
        answer = decide(
            GRAPH,
            frame,
            target="Y",
            target_kind=outcome,
            target_class=event,
            interventions={"A": Intervention(kind=treatment, baseline=baseline, values=(value,))},
            rows=query[["Z"]],
            types={"Z": "continuous"},
            seed=seed,
        )
        assert not answer["skipped"], answer["skipped"]
        item = answer["population"][0]
        biases.append(item["effect"] - np.mean(truth))
        expected = (
            1 + 0.2 * query["Z"].to_numpy()
            if treatment != "continuous" and outcome == "continuous"
            else np.full(100, np.mean(truth))
        )
        predictions = np.array([r["interventions"][0]["effect"] for r in answer["individual"]])
        errors.extend(predictions - expected)
        assert item["improvement"] > 0
        if outcome != "continuous":
            assert sum(c["effect"] for c in item["class_effects"]) == pytest.approx(0, abs=1e-8)
    assert abs(np.mean(biases)) < (0.12 if outcome == "continuous" else 0.035)
    assert np.sqrt(np.mean(np.square(errors))) < (0.25 if outcome == "continuous" else 0.07)


def test_automatic_defaults_and_query_invariance() -> None:
    frame, *_ = world(11, "binary", "continuous")
    first = decide(GRAPH, frame, target="Y", interventions={"A": {}}, rows=frame.iloc[:2])
    altered = frame.iloc[:2].copy()
    altered["Y"] = 1e30
    altered["A"] = ~altered["A"]
    second = decide(GRAPH, frame, target="Y", interventions={"A": {}}, rows=altered)
    assert first == second
    assert first["population"][0]["baseline"] is False
    assert first["population"][0]["value"] is True
    numeric, *_ = world(12, "continuous", "continuous")
    out = decide(GRAPH, numeric, target="Y", interventions={"A": {}})
    assert len(out["population"]) == 2
    assert out["population"][0]["baseline"] == np.median(numeric["A"])


def test_multicategory_requires_explicit_ranking_event() -> None:
    frame, *_ = world(3, "categorical", "categorical")
    with pytest.raises(ValueError, match="target_class"):
        decide(GRAPH, frame, target="Y", interventions={"A": {}}, target_kind="categorical")


def test_null_effect_and_unknown_query_category() -> None:
    frame, *_ = world(3, "binary", "continuous")
    out = decide(dag(["Z -> A", "Z -> Y"]), frame, target="Y", interventions={"A": {}})
    assert out["population"][0]["effect"] == 0
    frame["Z"] = np.where(frame["Z"] > 0, "left", "right")
    with pytest.raises(ValueError, match="unknown query category"):
        decide(GRAPH, frame, target="Y", interventions={"A": {}}, rows=pd.DataFrame({"Z": ["new"]}))


def test_multicategory_full_population_not_pairwise_subset() -> None:
    frame, *_ = world(4, "categorical", "continuous")
    out = decide(
        GRAPH, frame, target="Y", interventions={"A": {"baseline": "control", "values": ("low",)}}
    )
    assert out["n_context"] == len(frame)
    assert out["population"][0]["effect"] == pytest.approx(1, abs=0.12)


def test_bad_overlap_and_failed_refit_clear_state() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 1))
    a = (x[:, 0] > 0).astype(float)
    model = ArmDR()
    with pytest.raises(ValueError, match="overlap"):
        model.fit(x, a, x[:, 0])
    with pytest.raises(RuntimeError, match="not fitted"):
        model.contrast(0, 1)


def test_declared_integer_category_and_atomic_refusal() -> None:
    frame, *_ = world(9, "categorical", "continuous")
    frame["A"] = frame["A"].map({"control": 0, "low": 1, "high": 2})
    result = decide(GRAPH, frame, target="Y", types={"A": "categorical"}, interventions={"A": {}})
    assert all(item["method"] == "dr" for item in result["population"])
    frame, *_ = world(10, "continuous", "continuous")
    result = decide(
        GRAPH,
        frame,
        target="Y",
        interventions={"A": {"method": "continuous_dr", "baseline": 0.0, "values": (0.2, 100.0)}},
    )
    assert not result["population"]
    assert result["skipped"][0]["variable"] == "A"


def test_query_extrapolation_refused() -> None:
    frame, *_ = world(3, "binary", "continuous")
    with pytest.raises(ValueError, match="outside observed context support"):
        decide(GRAPH, frame, target="Y", interventions={"A": {}}, rows=pd.DataFrame({"Z": [100.0]}))


@pytest.mark.parametrize("treatment", ["categorical", "continuous"])
def test_class_coherence_with_nonlinear_learners(treatment: str) -> None:
    from sklearn.tree import DecisionTreeRegressor

    frame, baseline, value, event, _ = world(8, treatment, "categorical")
    result = decide(
        GRAPH,
        frame,
        target="Y",
        target_kind="categorical",
        target_class=event,
        interventions={
            "A": {
                "kind": treatment,
                "baseline": baseline,
                "values": (value,),
                "method": "continuous_dr" if treatment == "continuous" else "dr",
            }
        },
        learners={
            "outcome": DecisionTreeRegressor(max_depth=3, random_state=0),
            "final": DecisionTreeRegressor(max_depth=3, random_state=0),
        },
        rows=pd.DataFrame({"Z": [-0.5, 0.5]}),
    )
    assert not result["skipped"]
    assert sum(c["effect"] for c in result["population"][0]["class_effects"]) == pytest.approx(
        0, abs=1e-10
    )
    assert result["population"][0]["standard_error"] > 0
    for row in result["individual"]:
        assert sum(c["effect"] for c in row["interventions"][0]["class_effects"]) == pytest.approx(
            0, abs=1e-10
        )


def test_target_class_and_label_permutation_invariance() -> None:
    from sklearn.tree import DecisionTreeRegressor

    frame, baseline, value, _, _ = world(21, "categorical", "categorical")
    kwargs: dict[str, Any] = {
        "target": "Y",
        "target_kind": "categorical",
        "interventions": {"A": {"baseline": baseline, "values": (value,)}},
        "learners": {"outcome": DecisionTreeRegressor(max_depth=3, random_state=1)},
    }
    first = decide(GRAPH, frame, target_class="good", **kwargs)["population"][0]
    second = decide(GRAPH, frame, target_class="bad", **kwargs)["population"][0]
    assert first["class_effects"] == second["class_effects"]
    renamed = frame.copy()
    renamed["Y"] = renamed["Y"].map({"good": "z", "bad": "a", "other": "m"})
    third = decide(GRAPH, renamed, target_class="z", **kwargs)["population"][0]
    assert third["effect"] == pytest.approx(first["effect"], abs=1e-12)
    assert third["standard_error"] == pytest.approx(first["standard_error"], abs=1e-12)


def test_query_rank_reversal_between_two_actions() -> None:
    rng = np.random.default_rng(25)
    n = 6000
    z = rng.uniform(-1, 1, n)
    a = rng.binomial(1, 0.5, n)
    b = rng.binomial(1, 0.5, n)
    y = (1 + z) * a + (1 - z) * b + z + rng.normal(0, 0.1, n)
    graph = dag(["Z -> Y", "A -> Y", "B -> Y"])
    frame = pd.DataFrame({"Z": z, "A": a, "B": b, "Y": y})
    result = decide(
        graph,
        frame,
        target="Y",
        interventions={"A": {}, "B": {}},
        rows=pd.DataFrame({"Z": [-0.8, 0.8], "A": [0, 0], "B": [0, 0]}),
    )
    assert result["individual"][0]["interventions"][0]["variable"] == "B"
    assert result["individual"][1]["interventions"][0]["variable"] == "A"


def test_posttreatment_mediator_not_conditioned() -> None:
    rng = np.random.default_rng(26)
    n = 3000
    z = rng.normal(size=n)
    a = 0.3 * z + rng.normal(size=n)
    m = a + rng.normal(size=n)
    y = m + z + rng.normal(size=n)
    frame = pd.DataFrame({"Z": z, "A": a, "M": m, "Y": y})
    graph = dag(["Z -> A", "Z -> Y", "A -> M", "M -> Y"])
    result = decide(graph, frame, target="Y", interventions={"A": {}}, rows=frame.iloc[:2])
    assert all("M" not in effect["adjustment_set"] for effect in result["population"])
