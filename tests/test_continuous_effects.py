# SPDX-License-Identifier: Apache-2.0
"""API-219: known-truth, score-algebra, leakage and refusal contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from ergodic import ContinuousDR, dag, estimate_contrast, identify_effect, tabular
from ergodic.inference import DML, OLS, GaussianTreatmentDensity

GRAPH = dag(["Z -> A", "Z -> Y", "A -> Y"])
IDENTIFIED = identify_effect(GRAPH, "A", "Y")


def world(n: int = 4000, seed: int = 17, nonlinear: bool = False, binary: bool = False) -> Any:
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1, 1, n)
    a = 0.3 * z + rng.normal(size=n)
    mean = 1.5 * a + z
    if nonlinear:
        mean += a**2 + 0.5 * a * z
    if binary:
        mean = 1 / (1 + np.exp(-mean))
        y = (rng.random(n) < mean).astype(float)
    else:
        y = mean + rng.normal(0, 0.15, n)
    return tabular({"A": a, "Z": z, "Y": y})


class PolynomialOutcome:
    def get_params(self, deep: bool = False) -> dict[str, Any]:
        return {}

    def design(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        a, z = X[:, 0], X[:, 1]
        return np.column_stack([a, a * a, z, a * z])

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> Any:
        self.model = OLS().fit(self.design(X), y)
        return self

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.model.predict(self.design(X))


class ConstantOutcome:
    def get_params(self, deep: bool = False) -> dict[str, Any]:
        return {}

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> Any:
        return self

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.zeros(len(X))


class KnownOutcome(ConstantOutcome):
    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(0.7 * X[:, 0] + 0.1 * X[:, 1], dtype=np.float64)


class OracleDensity:
    def __init__(self, wrong: bool = False) -> None:
        self.wrong = wrong

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        return {"wrong": self.wrong}

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> Any:
        return self

    def pdf(
        self, X: npt.NDArray[np.float64], treatment: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        mean = np.zeros(len(X)) if self.wrong else 0.3 * X[:, 0]
        scale = 2 if self.wrong else 1
        return np.asarray(
            np.exp(-0.5 * ((treatment - mean) / scale) ** 2) / (scale * np.sqrt(2 * np.pi)),
            dtype=np.float64,
        )


def test_partially_linear_contrast_preserves_slope_and_shared_uncertainty() -> None:
    data = world()
    estimator = DML(random_state=4).fit(IDENTIFIED, data)
    slope = estimator.effect()
    assert slope.std_error is not None
    assert slope.value == pytest.approx(1.5, abs=0.025)
    forward = estimator.contrast(baseline=-1, value=1)
    reverse = estimator.contrast(baseline=1, value=-1)
    assert forward.value == pytest.approx(2 * slope.value)
    assert reverse.value == -forward.value
    assert reverse.std_error == forward.std_error == 2 * slope.std_error
    assert forward.treatment == "A" and forward.outcome == "Y"
    assert estimator.effect().value == slope.value
    rows = estimator.effects(tabular({"unused": np.arange(100.0)}), baseline=-1, value=1)
    assert np.all(rows.values == forward.value)
    assert rows.ate().std_error == forward.std_error  # not divided by sqrt(query n)
    zero = estimator.contrast(baseline=0, value=0)
    assert zero.value == zero.std_error == 0


@pytest.mark.parametrize("method", ["partially_linear", "continuous_dr"])
def test_facade_known_linear_effect_and_reversal(method: Any) -> None:
    data = world()
    a = estimate_contrast(
        GRAPH, data, "A", "Y", baseline=-0.5, value=1, method=method, random_state=2
    )
    b = estimate_contrast(
        GRAPH, data, "A", "Y", baseline=1, value=-0.5, method=method, random_state=2
    )
    assert a.value == pytest.approx(2.25, abs=0.1)
    assert b.value == -a.value
    assert b.std_error == a.std_error
    assert a.interval()[0] < a.value < a.interval()[1]


def test_continuous_dr_recovers_nonlinearity_and_conditional_effects() -> None:
    data = world(n=12000, nonlinear=True)
    estimator = ContinuousDR({"outcome": PolynomialOutcome()}, random_state=2).fit(IDENTIFIED, data)
    contrast = estimator.contrast(baseline=-0.5, value=1)
    assert contrast.value == pytest.approx(3, abs=0.12)
    assert estimator.contrast(baseline=0, value=1).value == pytest.approx(2.5, abs=0.12)
    query = tabular({"Z": np.array([0.8, -0.8, 0.0])})
    effects = estimator.effects(query, baseline=-0.5, value=1)
    assert effects.values == pytest.approx([3.6, 2.4, 3.0], abs=0.18)
    assert effects.std_error is None
    with pytest.raises(ValueError, match="no uncertainty"):
        effects.interval()
    assert effects.ate().kind == "point"
    assert effects.ate().value == pytest.approx(np.mean(effects.values))
    assert contrast.diagnostics["bandwidth"] == estimator.bandwidth_
    zero = estimator.contrast(baseline=0, value=0)
    assert zero.value == zero.std_error == 0


@pytest.mark.parametrize("wrong_outcome", [True, False])
def test_both_double_robustness_directions(wrong_outcome: Any) -> None:
    # Twelve independent seeded worlds; assess bias and RMSE, not one lucky draw.
    # h=0.15 leaves O(h^2) bias; use a 0.05 smoothing allowance plus 3 MC SEs.
    estimates = []
    for seed in range(12):
        data = world(n=6000, nonlinear=True, seed=seed)
        estimator = ContinuousDR(
            {"outcome": ConstantOutcome() if wrong_outcome else PolynomialOutcome()},
            density=OracleDensity(wrong=not wrong_outcome),
            bandwidth=0.15,
            random_state=0,
        ).fit(IDENTIFIED, data)
        estimates.append(estimator.contrast(baseline=-0.5, value=1).value)
    errors = np.asarray(estimates) - 3
    assert abs(errors.mean()) < 0.05 + 3 * errors.std(ddof=1) / np.sqrt(len(errors))
    assert np.sqrt(np.mean(errors**2)) < 0.35


def test_scores_match_reference_equation_and_paired_variance() -> None:
    data = world(n=1000)
    est = ContinuousDR(
        {"outcome": KnownOutcome()}, density=OracleDensity(), bandwidth=0.4, random_state=0
    ).fit(IDENTIFIED, data)
    a, z, y = (data.column(k) for k in ("A", "Z", "Y"))

    def score(dose: float) -> npt.NDArray[np.float64]:
        k = 0.75 * np.maximum(1 - ((a - dose) / 0.4) ** 2, 0) / 0.4
        f = OracleDensity().pdf(z[:, None], np.full(len(z), dose))
        mu = 0.7 * dose + 0.1 * z
        return np.asarray(mu + k / f * (y - mu), dtype=np.float64)

    paired = score(0.2) - score(-0.2)
    result = est.contrast(baseline=-0.2, value=0.2)
    assert result.value == pytest.approx(paired.mean())
    assert result.std_error == pytest.approx(paired.std(ddof=1) / np.sqrt(len(paired)))
    independent_se = np.sqrt(np.var(score(0.2), ddof=1) + np.var(score(-0.2), ddof=1)) / np.sqrt(
        len(paired)
    )
    assert result.std_error is not None
    assert abs(result.std_error - independent_se) > 0.001


def test_binary_outcome_is_probability_difference() -> None:
    data = world(n=18000, binary=True)
    est = ContinuousDR(random_state=1).fit(IDENTIFIED, data)
    z = data.column("Z")
    truth = np.mean(1 / (1 + np.exp(-(1.5 + z))) - 1 / (1 + np.exp(-(-0.75 + z))))
    result = est.contrast(baseline=-0.5, value=1)
    assert result.value == pytest.approx(truth, abs=0.045)


class LeakCheckedOutcome(PolynomialOutcome):
    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> Any:
        self.seen = set(X[:, 1])
        return super().fit(X, y)

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        assert not self.seen.intersection(X[:, 1]), "held-out rows leaked into outcome fit"
        return super().predict(X)


class LeakCheckedDensity(OracleDensity):
    def get_params(self, deep: bool = False) -> dict[str, Any]:
        return {}

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> Any:
        self.seen = set(X[:, 0])
        return self

    def pdf(
        self, X: npt.NDArray[np.float64], treatment: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        assert not self.seen.intersection(X[:, 0]), "held-out rows leaked into density fit"
        return super().pdf(X, treatment)


def test_nuisance_folds_are_honest_and_seeded() -> None:
    data = world(nonlinear=True)
    est = ContinuousDR(
        {"outcome": LeakCheckedOutcome()}, density=LeakCheckedDensity(), random_state=6
    ).fit(IDENTIFIED, data)
    a = est.contrast(baseline=-0.5, value=1)
    est.fit(IDENTIFIED, data)
    b = est.contrast(baseline=-0.5, value=1)
    assert a.value == b.value and a.std_error == b.std_error


@pytest.mark.parametrize("method", [DML, ContinuousDR])
@pytest.mark.parametrize(
    "baseline,value", [(np.nan, 1), (0, np.inf), (True, 1), (0, "1"), (0, 100)]
)
def test_invalid_doses(method: Any, baseline: Any, value: Any) -> None:
    est = method(random_state=1).fit(IDENTIFIED, world(n=500))
    with pytest.raises(ValueError):
        est.contrast(baseline=baseline, value=value)


@pytest.mark.parametrize(
    "options",
    [
        {"bandwidth": 0},
        {"bandwidth": np.nan},
        {"n_folds": 1},
        {"n_folds": 2.5},
        {"min_density": 0},
        {"min_effective_n": 1},
    ],
)
def test_invalid_continuous_configuration(options: Any) -> None:
    with pytest.raises(ValueError):
        ContinuousDR(**options)


def test_density_contract_and_support_refusals() -> None:
    with pytest.raises(TypeError, match="density"):
        ContinuousDR(density=OLS())  # type: ignore[arg-type]
    data = world(n=500)
    est = ContinuousDR(min_density=100, random_state=0).fit(IDENTIFIED, data)
    with pytest.raises(ValueError, match="density"):
        est.contrast(baseline=0, value=1)
    est = ContinuousDR(bandwidth=1e-9, random_state=0).fit(IDENTIFIED, data)
    with pytest.raises(ValueError, match="effective local"):
        est.contrast(baseline=0, value=1)
    est = ContinuousDR(random_state=0).fit(IDENTIFIED, data)
    with pytest.raises(ValueError, match="boundary"):
        est.contrast(baseline=float(data.column("A").min()), value=0)
    with pytest.raises(ValueError, match="explicit doses"):
        est.effect()


@pytest.mark.parametrize("method", [DML, ContinuousDR])
def test_failed_refit_cannot_return_stale_estimates(method: Any) -> None:
    est = method(random_state=0).fit(IDENTIFIED, world(n=500))
    broken = tabular(
        {"A": [1.0, 2.0, 3.0, 4.0], "Z": [0.0, 0.0, 0.0, 0.0], "Y": [np.inf, 1.0, 2.0, 3.0]}
    )
    with pytest.raises(ValueError):
        est.fit(IDENTIFIED, broken)
    with pytest.raises(RuntimeError, match="not fitted"):
        est.contrast(baseline=0, value=1)


def test_identification_and_unsupported_types() -> None:
    data = world(n=500)
    unidentified = identify_effect(dag(["U -> A", "U -> Y", "A -> Y"], latent=["U"]), "A", "Y")
    with pytest.raises(ValueError, match="not identifiable"):
        ContinuousDR().fit(unidentified, data)
    binary = tabular({"A": np.tile([0.0, 1.0], 250), "Z": data.column("Z"), "Y": data.column("Y")})
    with pytest.raises(ValueError, match="distinct treatment"):
        ContinuousDR().fit(IDENTIFIED, binary)
    multi = tabular({"A": data.column("A"), "Z": data.column("Z"), "Y": np.tile(["a", "b"], 250)})
    with pytest.raises(ValueError, match="desired binary outcome"):
        ContinuousDR().fit(IDENTIFIED, multi)
    with pytest.raises(ValueError, match="unknown contrast method"):
        estimate_contrast(GRAPH, data, "A", "Y", baseline=0, value=1, method="x")


def test_density_fit_and_no_residual_variation() -> None:
    data = world(n=5000)
    density = GaussianTreatmentDensity().fit(data.column("Z")[:, None], data.column("A"))
    query = np.array([[-0.5], [0.5]])
    doses = np.array([0.0, 1.0])
    assert density.pdf(query, doses) == pytest.approx(OracleDensity().pdf(query, doses), abs=0.02)
    with pytest.raises(ValueError, match="residual variation"):
        density.fit(np.arange(10.0)[:, None], np.arange(10.0))


def test_query_categories_use_context_encoding() -> None:
    rng = np.random.default_rng(42)
    z = rng.choice(["a", "b", "c"], size=5000)
    a = rng.normal(size=len(z))
    y = 2 * a + (z == "c") + rng.normal(0, 0.1, len(z))
    data = tabular({"A": a, "Y": y, "Z": z})
    est = ContinuousDR(random_state=1).fit(IDENTIFIED, data)
    result = est.effects(tabular({"Z": ["c", "a"]}), baseline=0, value=1)
    assert result.values == pytest.approx([2.0, 2.0], abs=0.15)
    with pytest.raises(ValueError, match="unknown covariate category"):
        est.effects(tabular({"Z": ["unknown"]}), baseline=0, value=1)


@pytest.mark.parametrize("constant", [0.0, 1.0, 3.5])
def test_dml_rejects_constant_treatment(constant: float) -> None:
    data = world(n=100)
    data = tabular({"A": np.full(100, constant), "Z": data.column("Z"), "Y": data.column("Y")})
    with pytest.raises(ValueError, match="treatment variation"):
        DML().fit(IDENTIFIED, data)


def test_query_numeric_encoding_is_fixed_at_fit() -> None:
    est = ContinuousDR({"outcome": PolynomialOutcome()}, random_state=2).fit(
        IDENTIFIED, world(nonlinear=True)
    )
    ints = est.effects(tabular({"Z": [1, 0, -1]}), baseline=-0.5, value=1)
    floats = est.effects(tabular({"Z": [1.0, 0.0, -1.0]}), baseline=-0.5, value=1)
    assert ints.values == pytest.approx(floats.values, abs=1e-12)
    with pytest.raises(ValueError):
        est.effects(tabular({"Z": [np.inf]}), baseline=-0.5, value=1)


@pytest.mark.parametrize("method", [DML, ContinuousDR])
def test_binary_outcome_event_is_invariant_to_declared_categories(method: Any) -> None:
    data = world(n=4000, binary=True)

    def fitted(order: list[int]) -> Any:
        frame = pd.DataFrame(
            {
                "A": data.column("A"),
                "Z": data.column("Z"),
                "Y": pd.Categorical(data.column("Y"), categories=order),
            }
        )
        return method(random_state=1).fit(IDENTIFIED, tabular(frame))

    forward = fitted([0, 1])
    reversed_categories = fitted([1, 0])
    a = forward.contrast(baseline=-0.5, value=1)
    b = reversed_categories.contrast(baseline=-0.5, value=1)
    assert a.value > 0
    assert a.value == pytest.approx(b.value, abs=1e-10)
    assert a.std_error == pytest.approx(b.std_error, abs=1e-10)
    if method is ContinuousDR:
        assert a.learners["outcome"] == "LogisticRegression"
    else:
        # Old slope coding remains unchanged; new explicit contrasts use event=1.
        assert forward.effect().value == pytest.approx(-reversed_categories.effect().value)


def test_dml_contrast_uses_raw_treatment_values() -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(size=2000)
    a = rng.integers(0, 2, len(z))
    y = 2 * a + z + rng.normal(size=len(z))
    results = []
    for order in [[0, 1], [1, 0]]:
        data = tabular(pd.DataFrame({"A": pd.Categorical(a, categories=order), "Z": z, "Y": y}))
        for treated in [None, 0, 1]:
            est = DML(treated=treated, random_state=0).fit(IDENTIFIED, data)
            result = est.contrast(baseline=0, value=1).value
            positive_label = order[1] if treated is None else treated
            expected = est.effect().value * (1 if positive_label == 1 else -1)
            assert result == pytest.approx(expected)
            results.append(result)
    # Reversing coding changes seeded stratified folds, so finite-sample point
    # estimates can differ slightly; each must still recover the same raw effect.
    assert all(abs(result - 2) < 0.15 for result in results)
    assert results[0] == pytest.approx(2, abs=0.15)


class CountingFinal(OLS):
    fits = 0

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64] | None = None,
    ) -> Any:
        type(self).fits += 1
        return super().fit(X, y, sample_weight)


def test_final_fit_is_reused_across_query_batches_and_reset_on_refit() -> None:
    CountingFinal.fits = 0
    data = world()
    est = ContinuousDR({"final": CountingFinal()}, random_state=0).fit(IDENTIFIED, data)
    whole = est.effects(tabular({"Z": [-0.5, 0.5]}), baseline=0, value=1)
    first = est.effects(tabular({"Z": [-0.5]}), baseline=0, value=1)
    last = est.effects(tabular({"Z": [0.5]}), baseline=0, value=1)
    assert CountingFinal.fits == 1
    assert whole.values == pytest.approx(np.r_[first.values, last.values])
    est.fit(IDENTIFIED, data)
    est.effects(tabular({"Z": [0.5]}), baseline=0, value=1)
    assert CountingFinal.fits == 2
