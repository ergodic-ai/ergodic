# SPDX-License-Identifier: Apache-2.0
"""Adversarial stress tests for the CATE meta-learners and Tier-B explainability.

Seeded and deterministic. These do not duplicate ``test_cate.py``; they push the
harder cases: causal correctness against closed forms, honesty under the null, the
out-of-fold ``tau_hat`` guard, positivity blow-up, small-sample degeneracies, the
result surface contract, identification gating, reproducibility, and numerical
edge cases (singular designs, ties, non-finite inputs).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from ergodic import (
    CATEResult,
    ITEResult,
    PerUnitEffect,
    cate,
    dag,
    estimate_effect,
    identify_effect,
    tabular,
)
from ergodic.data import TabularData
from ergodic.graph import MixedGraph
from ergodic.inference import DRLearner, dr_scores
from ergodic.inference._scores import aipw_scores
from ergodic.inference.explain._blp import best_linear_projection
from ergodic.inference.explain._calibration import calibration_test
from ergodic.inference.explain._gates import gates
from ergodic.inference.explain._rate import rate as rate_fn


def _world(
    n: int = 4000, seed: int = 0, *, b: float = 1.0, slope: float = 6.0
) -> tuple[TabularData, MixedGraph, np.ndarray]:
    """tau(z) = 1 + b*z, confounded by z; ``slope`` sets the propensity steepness."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-slope * z / 6.0))).astype(float)
    y = (1.0 + b * z) * a + 1.5 * z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    return data, dag(["Z -> A", "Z -> Y", "A -> Y"]), z


# == 1. causal correctness vs closed forms ===================================


def test_aipw_score_matches_textbook_eif() -> None:
    """The AIPW score equals the textbook EIF ``mu1-mu0 + (W-e)/(e(1-e))(Y-m)``."""
    rng = np.random.default_rng(0)
    n = 9
    a = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    y = rng.normal(size=n)
    mu0, mu1 = rng.normal(size=n), rng.normal(size=n)
    e = rng.uniform(0.2, 0.8, size=n)
    got = aipw_scores(a, y, mu0, mu1, e)
    m = a * mu1 + (1.0 - a) * mu0
    textbook = (mu1 - mu0) + (a - e) / (e * (1.0 - e)) * (y - m)
    assert np.allclose(got, textbook)


@pytest.mark.parametrize("method", ["dr_learner", "r_learner", "x_learner", "t_learner"])
def test_cate_recovers_linear_truth_slope(method: str) -> None:
    """The heterogeneity-capable learners' CATE tracks tau(z)=1+z (slope ~1).

    The S-learner is excluded on purpose: with a linear base learner it has no
    interaction term, so it can only return a constant effect (see the separate
    S-learner test below); that is its documented regularization-toward-homogeneity
    limitation, not a recovery failure.
    """
    data, graph, z = _world(seed=0)
    res = cate(graph, data, "A", "Y", method=method, random_state=0)
    truth = 1.0 + z
    assert np.corrcoef(res.values, truth)[0, 1] > 0.6
    # the best-fit slope of the per-unit estimate on z is near the planted 1.0
    slope = float(np.polyfit(z, res.values, 1)[0])
    assert 0.5 < slope < 1.5


def test_linear_s_learner_is_constant_but_ate_correct() -> None:
    """A linear S-learner has no interaction term: a constant CATE at the right ATE.

    The single linear model ``y ~ a + z`` gives ``mu(1,z)-mu(0,z) = coef(a)``, a
    constant. So the S-learner recovers the ATE (1.0) but assigns zero heterogeneity,
    the canonical S-learner limitation. Heterogeneity needs a richer base learner or
    a T/X/DR-learner.
    """
    data, graph, _ = _world(seed=0)
    res = cate(graph, data, "A", "Y", method="s_learner", random_state=0)
    assert float(np.std(res.values)) == pytest.approx(0.0, abs=1e-9)
    assert abs(res.ate().value - 1.0) < 0.15


def test_dr_scores_mean_matches_aipw_and_truth() -> None:
    """The DR score mean is the AIPW ATE and recovers the planted ATE of 1.0."""
    data, graph, _ = _world(seed=0)
    scores = dr_scores(identify_effect(graph, "A", "Y"), data, random_state=0)
    aipw = estimate_effect(graph, data, "A", "Y", strategy="aipw", random_state=0)
    assert abs(float(np.mean(scores.scores)) - aipw.value) < 0.1
    assert abs(float(np.mean(scores.scores)) - 1.0) < 0.15


def test_blp_slope_coverage_over_seeds() -> None:
    """The BLP slope CI covers the true modifier (b=1) near the nominal rate."""
    covered = 0
    seeds = 40
    for seed in range(seeds):
        data, graph, _ = _world(n=3000, seed=seed, b=1.0)
        z = cate(graph, data, "A", "Y", random_state=seed).explain().blp().frame()["Z"]
        if z["ci_low"] < 1.0 < z["ci_high"]:
            covered += 1
    # nominal 90% CI: allow Monte-Carlo slack but demand it is not badly broken
    assert covered >= int(0.7 * seeds)


# == 2. honesty / nulls ======================================================


def test_homogeneous_null_does_not_false_positive() -> None:
    """A truly homogeneous effect: GATES covers 0, calibration/RATE do not reject."""
    data, graph, _ = _world(n=4000, seed=3, b=0.0)
    h = cate(graph, data, "A", "Y", random_state=3).explain()
    # the omnibus heterogeneity tests (the load-bearing honesty guards) do not reject
    assert not h.calibration_test().rejects()
    # RATE's one-sided HTE test (its stated use) does not reject under the null. NOTE:
    # the two-sided RATE CI is anti-conservative (the half-sample-bootstrap SE with the
    # /sqrt(2) rescaling under-estimates the AUTOC sampling SD, ~70% coverage in MC), so
    # we assert the one-sided test, not the CI covering 0. See the SE-calibration test.
    assert not h.rate("autoc", random_state=3).rejects()
    z = h.blp().frame()["Z"]
    assert z["ci_low"] < 0.0 < z["ci_high"]  # the covariate slope covers zero
    # the GATES contrast is not significantly positive (one-sided no-HTE direction)
    glo, ghi = h.gates(n_groups=5).interval()
    assert glo < 0.0 < ghi or ghi < 0.0  # covers zero or (rarely) dips negative


def test_rate_half_sample_se_is_anti_conservative() -> None:
    """Document flagged deviation #1: the RATE half-sample SE under-states dispersion.

    The half-sample-bootstrap SE rescales the subsample spread by 1/sqrt(2), assuming
    a size-n/2 estimator has exactly 2x the full-sample variance. For the AUTOC
    functional (a rank-weighted average over a sorted sequence) the without-replacement
    subsample SD under-estimates the true sampling SD, so the SE is biased low and the
    normal CI under-covers. We pin the DIRECTION (SE < true SD) so a future
    influence-function SE that fixes it trips this test and gets recalibrated.
    """
    rng = np.random.default_rng(0)

    def gen(m: int) -> tuple[np.ndarray, np.ndarray]:
        s = rng.normal(size=m)
        return s, s + rng.normal(size=m) * 0.5

    estimates, ses = [], []
    for _ in range(120):
        s, tau = gen(500)
        r = rate_fn(s, tau, metric="autoc", n_boot=120, random_state=int(rng.integers(1000000000)))
        estimates.append(r.estimate)
        ses.append(r.std_error)
    true_sd = float(np.std(estimates, ddof=1))
    mean_se = float(np.mean(ses))
    assert mean_se < true_sd  # the SE under-states the true dispersion
    assert mean_se > 0.4 * true_sd  # but not absurdly so (sanity bound)


def test_strong_heterogeneity_is_detected() -> None:
    """Strong HTE: calibration rejects, RATE excludes 0, GATES is monotone up."""
    data, graph, _ = _world(n=4000, seed=0, b=2.0)
    h = cate(graph, data, "A", "Y", random_state=0).explain()
    assert h.calibration_test().rejects()
    r = h.rate("autoc", random_state=0)
    assert r.rejects() and r.interval()[0] > 0.0
    assert h.gates(n_groups=5).is_monotone()


def test_out_of_fold_tau_hat_guard_is_load_bearing() -> None:
    """The OOF tau_hat tames the in-sample self-correlation a flexible final inflates.

    Under a homogeneous null with a flexible (KNN) final, the in-sample tau_hat
    chases its own pseudo-outcome noise: the calibration test massively false-rejects
    and RATE is hugely positive. The estimator's out-of-fold tau_hat (the honesty
    recipe) is computed on held-out folds, so it does not share rows with the scores
    and the false positive collapses. Flipping back to in-sample confirms the guard
    is doing the work.
    """
    rng = np.random.default_rng(11)
    n = 2000
    z = rng.normal(size=(n, 4))
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.5 * z[:, 0]))).astype(float)
    y = 0.0 * a + 1.0 * z[:, 0] + rng.normal(size=n)  # homogeneous, b=0
    df = pd.DataFrame({"A": a, "Y": y, **{f"Z{i}": z[:, i] for i in range(4)}})
    data = tabular(df)
    edges = ["A -> Y"] + [f"Z{i} -> A" for i in range(4)] + [f"Z{i} -> Y" for i in range(4)]
    ident = identify_effect(dag(edges), "A", "Y")
    res = DRLearner({"final": "knn"}, random_state=11).fit(ident, data).cate()
    assert res._scores is not None
    oof, insample, scores = res._scores.tau_hat, res.values, res._scores.scores
    assert oof is not None and not np.allclose(oof, insample)
    # the guard (out-of-fold) does NOT false-reject the calibration omnibus
    assert not calibration_test(scores, oof).rejects()
    # the in-sample ranking WOULD false-reject (so the guard is load-bearing)
    assert calibration_test(scores, insample).rejects()
    # and the in-sample RATE is far more optimistic than the out-of-fold one
    in_rate = rate_fn(scores, insample, random_state=11).estimate
    oof_rate = rate_fn(scores, oof, random_state=11).estimate
    assert in_rate > 3.0 * abs(oof_rate)


def test_constant_tau_hat_calibration_is_honest_not_a_coin_flip() -> None:
    """A degenerate (constant) tau_hat yields a clean non-rejection, not float-noise.

    DML's tau_hat is the constant theta. The differential regressor is then all-zero
    and the HC3 fit on the rank-deficient design produces a float-noise coefficient
    whose sign (hence the one-sided p) is arbitrary. The honest result is "no usable
    heterogeneity in this ranking": a zero differential and p == 1.
    """
    const = np.full(2000, 0.7)
    scores = np.random.default_rng(0).normal(size=2000)
    cal = calibration_test(scores, const)
    assert cal.differential_coefficient == 0.0
    assert cal.heterogeneity_pvalue == 1.0
    assert not cal.rejects()
    assert np.isnan(cal.differential_se)


# == 3. positivity / overlap =================================================


def test_positivity_blowup_stays_finite_via_clip() -> None:
    """Near-deterministic propensity: the clip keeps the AIPW signal finite (not NaN).

    Strong confounding pushes e(x) to the {0,1} edges. The 1/e and 1/(1-e) weights
    inflate the score variance enormously, but the propensity clip keeps every score
    finite and the point roughly right; nothing is NaN/inf, and the propensity is
    exposed for overlap diagnosis.
    """
    data, graph, _ = _world(n=3000, seed=0, slope=12.0)
    scores = dr_scores(identify_effect(graph, "A", "Y"), data, random_state=0)
    assert scores.propensity.min() >= 1e-3 - 1e-12
    assert scores.propensity.max() <= 1.0 - 1e-3 + 1e-12
    assert bool(np.all(np.isfinite(scores.scores)))
    assert float(np.abs(scores.scores).max()) > 50.0  # the variance really did inflate
    res = cate(graph, data, "A", "Y", random_state=0)
    assert bool(np.all(np.isfinite(res.values)))


# == 4. small-sample / degenerate ============================================


def test_empty_arm_raises() -> None:
    """A treatment with no control (or no treated) units raises a clear error."""
    rng = np.random.default_rng(0)
    n = 200
    z = rng.normal(size=n)
    a = np.ones(n)  # everyone treated: empty control arm
    y = a + 1.5 * z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    graph = dag(["Z -> A", "Z -> Y", "A -> Y"])
    with pytest.raises(ValueError, match="empty arm"):
        cate(graph, data, "A", "Y", random_state=0)


def test_tiny_sample_is_finite_or_raises() -> None:
    """Tiny n (folds capped at n): the result is finite, never silent NaN garbage."""
    for n in (10, 16, 20):
        data, graph, _ = _world(n=n, seed=0)
        res = cate(graph, data, "A", "Y", random_state=0)
        assert bool(np.all(np.isfinite(res.values)))


def test_zero_variance_treatment_after_strata_raises() -> None:
    """A treatment with a single distinct value is not two-level: a clear error."""
    rng = np.random.default_rng(0)
    n = 200
    z = rng.normal(size=n)
    y = 1.5 * z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": np.zeros(n), "Z": z, "Y": y}))
    graph = dag(["Z -> A", "Z -> Y", "A -> Y"])
    with pytest.raises((ValueError, NotImplementedError)):
        cate(graph, data, "A", "Y", random_state=0)


# == 5. types ================================================================


def test_continuous_treatment_dml() -> None:
    """A continuous treatment via DML recovers the linear coefficient (the APE)."""
    rng = np.random.default_rng(0)
    n = 2500
    z = rng.normal(size=n)
    a = 0.5 * z + rng.normal(size=n)
    y = 1.0 * a + 1.5 * z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    graph = dag(["Z -> A", "Z -> Y", "A -> Y"])
    res = cate(graph, data, "A", "Y", method="dml", random_state=0)
    assert abs(res.ate().value - 1.0) < 0.15


def test_continuous_treatment_rejected_by_propensity_learners() -> None:
    """A propensity-classifier meta-learner (R/X/DR) rejects a continuous treatment."""
    rng = np.random.default_rng(0)
    n = 1500
    z = rng.normal(size=n)
    a = 0.5 * z + rng.normal(size=n)
    y = a + 1.5 * z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    graph = dag(["Z -> A", "Z -> Y", "A -> Y"])
    with pytest.raises(NotImplementedError, match="binary"):
        cate(graph, data, "A", "Y", method="r_learner", random_state=0)


def test_treated_level_coding_for_string_treatment() -> None:
    """``treated=`` codes a non-numeric two-level treatment and recovers the ATE."""
    rng = np.random.default_rng(0)
    n = 2000
    z = rng.normal(size=n)
    a = np.where(rng.uniform(size=n) < 0.5, "ctrl", "trt")
    y = 1.0 * (a == "trt") + 1.5 * z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}), types={"A": "categorical"})
    graph = dag(["Z -> A", "Z -> Y", "A -> Y"])
    res = cate(graph, data, "A", "Y", method="dr_learner", treated="trt", random_state=0)
    assert abs(res.ate().value - 1.0) < 0.2


def test_multi_valued_treatment_raises_clearly() -> None:
    """A three-level treatment has no binary contrast: a clear error, not nonsense."""
    rng = np.random.default_rng(0)
    n = 1500
    z = rng.normal(size=n)
    a = rng.integers(0, 3, size=n).astype(int)
    y = 1.0 * (a == 1) + 1.5 * z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}), types={"A": "categorical"})
    graph = dag(["Z -> A", "Z -> Y", "A -> Y"])
    with pytest.raises((ValueError, NotImplementedError), match=r"binary|two-level|level"):
        cate(graph, data, "A", "Y", random_state=0)


def test_binary_outcome_logistic_recovers_gradient() -> None:
    """A binary outcome with a planted interaction: the CATE rises with the modifier."""
    rng = np.random.default_rng(1)
    n = 6000
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.5 * z))).astype(int)
    logit = -0.2 + 1.5 * a + 0.5 * z + 1.2 * a * z
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    data = tabular(
        pd.DataFrame({"A": a, "Z": z, "Y": y}),
        types={"A": "categorical", "Y": "categorical"},
    )
    graph = dag(["Z -> A", "Z -> Y", "A -> Y"])
    res = cate(graph, data, "A", "Y", method="dr_learner", treated=1, random_state=1)
    assert np.corrcoef(res.values, z)[0, 1] > 0.3
    assert res.explain().calibration_test().rejects()


# == 6. result surface invariants ============================================


def _cate(n: int = 12) -> CATEResult:
    draws = np.random.default_rng(0).normal(loc=1.0, scale=0.3, size=(80, n))
    return CATEResult(
        treatment="A",
        outcome="Y",
        covariates=("Z",),
        n=n,
        kind="bootstrap",
        source="dr",
        _values=draws.mean(axis=0),
        _draws=draws,
    )


def _ite(n: int = 12) -> ITEResult:
    rng = np.random.default_rng(1)
    return ITEResult(
        treatment="T",
        outcome="Y",
        covariates=(),
        n=n,
        kind="point",
        source="scm",
        framing="levels=(0.0, 1.0)",
        _values=rng.normal(size=n),
        _factual=rng.normal(size=n),
    )


@pytest.mark.parametrize("result", [_cate(), _ite()])
def test_array_drop_in_contract(result: PerUnitEffect) -> None:
    """``__array__`` makes the wrapped result a drop-in for the bare array."""
    n = len(result)
    assert np.asarray(result).shape == (n,)
    assert float(np.mean(result)) == pytest.approx(float(np.mean(result.values)))
    assert result.shape == (n,)
    # element-wise comparisons return arrays (not a scalar dataclass __eq__)
    eq = result == result.values
    assert isinstance(eq, np.ndarray) and eq.shape == (n,) and bool(np.all(eq))
    assert isinstance(result < 1e9, np.ndarray)
    assert isinstance(result != -1e9, np.ndarray)


@pytest.mark.parametrize("result", [_cate(), _ite()])
def test_results_are_unhashable_and_frozen(result: PerUnitEffect) -> None:
    """The base contract: per-unit results are unhashable and immutable."""
    with pytest.raises(TypeError):
        hash(result)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result._values = np.zeros(len(result))  # type: ignore[misc]


def test_cate_interval_ordered_and_draws_shape() -> None:
    res = _cate(20)
    lo, hi = res.interval(0.9)
    assert lo.shape == hi.shape == (20,)
    assert bool(np.all(lo <= hi))
    draws = res.draws()
    assert draws is not None and draws.shape == (80, 20)


def test_ate_bridge_consistency() -> None:
    """``.ate()`` is the mean of the per-unit values and carries a sane SE."""
    res = _cate(20)
    eff = res.ate()
    assert eff.value == pytest.approx(float(np.mean(res.values)))
    assert eff.std_error is not None and eff.std_error > 0.0


def test_analytic_ate_without_scores_refuses_a_standard_error() -> None:
    """Per-unit analytic SEs are correlated, so they never combine as independent.

    Regression (API-149): ``.ate()`` used to report ``sqrt(sum se^2)/n``, which
    treats the units' errors as independent although they share one fitted model
    (about ``sqrt(n)`` too narrow). With no scores to read, the mean comes back as a
    point and the interval refuses.
    """
    se = np.abs(np.random.default_rng(2).normal(size=12)) + 0.5
    values = np.random.default_rng(3).normal(size=12)
    res = CATEResult(
        treatment="A",
        outcome="Y",
        covariates=("Z",),
        n=12,
        kind="analytic",
        source="dr",
        _values=values,
        _std_error=se,
    )
    eff = res.ate()
    assert eff.value == pytest.approx(float(np.mean(values)))
    assert eff.std_error is None
    assert eff.kind == "point"
    with pytest.raises(ValueError, match="no uncertainty"):
        eff.interval()


def test_analytic_ate_of_a_homogeneous_result_keeps_its_standard_error() -> None:
    """One effect and one SE repeated over the units (``DML.cate()``) is one estimate."""
    res = CATEResult(
        treatment="A",
        outcome="Y",
        covariates=("Z",),
        n=50,
        kind="analytic",
        source="dml",
        _values=np.full(50, 0.7),
        _std_error=np.full(50, 0.05),
    )
    eff = res.ate()
    assert eff.value == pytest.approx(0.7)
    assert eff.std_error == pytest.approx(0.05)


def test_analytic_ate_reads_the_dr_scores_and_matches_sampling_spread() -> None:
    """The DR-learner's ``.ate()`` SE is the AIPW score SE, not the independent sum.

    Regression (API-149): on this world the old SE was about 0.0009 while the
    estimator's spread across replications is about 0.04; the AIPW score SE is
    0.039 and agrees with ``estimate_effect(strategy="aipw")``.
    """
    data, graph, _ = _world(seed=0, slope=4.2)
    res = cate(graph, data, "A", "Y", method="dr_learner", random_state=0)
    assert res.kind == "analytic"
    eff = res.ate()
    # the linear final stage: the AIPW mean is the mean per-unit effect
    assert eff.value == pytest.approx(float(np.mean(res.values)), abs=1e-8)
    assert eff.std_error is not None
    assert 0.025 < eff.std_error < 0.06
    aipw = estimate_effect(graph, data, "A", "Y", strategy="aipw", random_state=0)
    assert aipw.std_error is not None
    assert eff.std_error == pytest.approx(aipw.std_error, rel=0.2)


def test_index_alignment_and_to_frame_roundtrip() -> None:
    res = CATEResult(
        treatment="A",
        outcome="Y",
        covariates=(),
        n=3,
        kind="point",
        _values=np.array([1.0, 2.0, 3.0]),
        _index=np.array(["x", "y", "z"]),
    )
    assert list(res.index) == ["x", "y", "z"]
    frame = res.to_frame()
    assert list(frame.index) == ["x", "y", "z"]
    assert np.allclose(frame["effect"].to_numpy(), res.values)


def test_interval_without_uncertainty_raises() -> None:
    res = CATEResult(
        treatment="A", outcome="Y", covariates=(), n=3, kind="point", _values=np.ones(3)
    )
    with pytest.raises(ValueError, match="no uncertainty"):
        res.interval()


# == 7. ITEResult ============================================================


def test_ite_explain_raises_with_pointer() -> None:
    res = _ite()
    with pytest.raises(NotImplementedError, match="CATE"):
        res.explain()


def test_ite_carries_factual_and_framing() -> None:
    res = _ite()
    assert res.estimand == "ite"
    assert res.framing.startswith("levels=")
    assert res.factual is not None and res.factual.shape == (len(res),)


def test_ite_ate_bridge_is_mean() -> None:
    res = _ite()
    assert res.ate().value == pytest.approx(float(np.mean(res.values)))


# == 8. facade identification gating =========================================


def test_cate_not_identifiable_raises() -> None:
    """A latent-confounded effect with no adjustment set raises (estimate honesty)."""
    rng = np.random.default_rng(0)
    n = 500
    u = rng.normal(size=n)
    a = (rng.uniform(size=n) < 0.5).astype(float)
    y = a + u + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Y": y}))
    graph = dag(["A -> Y", "U -> A", "U -> Y"], latent=["U"])
    with pytest.raises(ValueError, match="not identifiable"):
        cate(graph, data, "A", "Y")


def test_cate_frontdoor_family_rejected_with_pointer() -> None:
    """A front-door-identified effect is refused by the adjustment-only CATE layer."""
    rng = np.random.default_rng(0)
    n = 800
    u = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-u))).astype(float)
    m = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-a))).astype(float)
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(m + u)))).astype(float)
    data = tabular(
        pd.DataFrame({"A": a.astype(int), "M": m.astype(int), "Y": y.astype(int)}),
        types={"A": "categorical", "M": "categorical", "Y": "categorical"},
    )
    graph = dag(["A -> M", "M -> Y", "U -> A", "U -> Y"], latent=["U"])
    with pytest.raises(ValueError, match="adjustment"):
        cate(graph, data, "A", "Y")


def test_cate_null_effect_graph_points_at_family() -> None:
    """A null-effect graph (no path treatment->outcome) is refused with a family pointer.

    The effect IS identified (as the constant zero), but the adjustment-family CATE
    meta-learners do not consume the ``null`` strategy, so the facade raises a clear
    family-mismatch error rather than fabricating a per-unit zero. (Reported gap vs
    ``estimate_effect``, which returns the zero effect via NullEffect; routing a null
    to a constant-zero CATE is a design call for the maintainer.)
    """
    rng = np.random.default_rng(0)
    n = 800
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 0.5).astype(float)
    y = 1.5 * z + rng.normal(size=n)  # A does not affect Y
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    graph = dag(["Z -> Y", "Z -> A"])  # A exists but has no directed path to Y
    assert identify_effect(graph, "A", "Y").strategy == "null"
    with pytest.raises(ValueError, match=r"adjustment|null"):
        cate(graph, data, "A", "Y", random_state=0)


def test_cate_unknown_method_and_seams() -> None:
    data, graph, _ = _world(n=400, seed=0)
    with pytest.raises(ValueError, match="unknown CATE method"):
        cate(graph, data, "A", "Y", method="not_a_method")
    with pytest.raises(NotImplementedError, match="neural"):
        cate(graph, data, "A", "Y", method="dragonnet")
    with pytest.raises(NotImplementedError, match="foundation"):
        cate(graph, data, "A", "Y", method="causal_pfn")


def test_dr_scores_slot_validation() -> None:
    data, graph, _ = _world(n=600, seed=0)
    ident = identify_effect(graph, "A", "Y")
    with pytest.raises(ValueError, match="slots"):
        dr_scores(ident, data, learners={"bogus": "ols"})
    with pytest.raises(TypeError, match="classifier"):
        dr_scores(ident, data, learners={"propensity": "ols"})


# == 9. reproducibility ======================================================


def test_reproducible_values_and_explain_cis() -> None:
    """Same seed -> identical CATE values AND identical explain() intervals."""
    data, graph, _ = _world(n=2500, seed=0)
    r1 = cate(graph, data, "A", "Y", random_state=0)
    r2 = cate(graph, data, "A", "Y", random_state=0)
    assert np.array_equal(r1.values, r2.values)
    h1, h2 = r1.explain(), r2.explain()
    lo1, hi1 = h1.blp().interval()
    lo2, hi2 = h2.blp().interval()
    assert np.allclose(lo1, lo2) and np.allclose(hi1, hi2)
    a1 = h1.rate("autoc", random_state=0)
    a2 = h2.rate("autoc", random_state=0)
    assert a1.estimate == a2.estimate and a1.std_error == a2.std_error


# == 10. numerical ===========================================================


def test_nan_input_raises_clearly() -> None:
    """A NaN in a used column is caught loudly (call dropna), not silently propagated."""
    rng = np.random.default_rng(0)
    n = 500
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 0.5).astype(float)
    y = a + 1.5 * z + rng.normal(size=n)
    y[3] = np.nan
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    graph = dag(["Z -> A", "Z -> Y", "A -> Y"])
    with pytest.raises(ValueError, match="missing"):
        cate(graph, data, "A", "Y", random_state=0)


def test_singular_blp_design_does_not_crash() -> None:
    """A constant (collinear) covariate makes the BLP design singular: pinv, not a crash."""
    rng = np.random.default_rng(0)
    n = 2000
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 0.5).astype(float)
    y = a + 1.5 * z + rng.normal(size=n)
    # two perfectly collinear covariates (Z and 2*Z) -> rank-deficient design
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Z2": 2.0 * z, "Y": y}))
    graph = dag(["Z -> A", "Z -> Y", "Z2 -> A", "Z2 -> Y", "A -> Y"])
    scores = dr_scores(identify_effect(graph, "A", "Y"), data, random_state=0)
    blp = best_linear_projection(scores.scores, scores.covariates, scores.covariate_names)
    assert bool(np.all(np.isfinite(blp.coef)))
    assert bool(np.all(np.isfinite(blp.std_error)))


def test_gates_and_rate_handle_heavy_ties() -> None:
    """A tau_hat with heavy ties still groups evenly and integrates without a crash."""
    data, graph, _ = _world(n=2000, seed=0)
    scores = dr_scores(identify_effect(graph, "A", "Y"), data, random_state=0)
    tied = np.zeros(data.n_rows)
    tied[: data.n_rows // 2] = 1.0  # only two distinct values
    g = gates(scores.scores, tied, n_groups=5)
    assert bool(np.all(np.isfinite(g.group_effect)))
    assert int(g.counts.sum()) == data.n_rows
    r = rate_fn(scores.scores, tied, metric="autoc", random_state=0)
    assert np.isfinite(r.estimate) and np.isfinite(r.std_error)


def test_rate_oracle_beats_random_and_metrics_run() -> None:
    """An oracle ranking's AUTOC dominates a random ranking's; both metrics run."""
    data, graph, z = _world(n=3000, seed=0, b=1.0)
    scores = dr_scores(identify_effect(graph, "A", "Y"), data, random_state=0)
    oracle = rate_fn(scores.scores, 1.0 + z, metric="autoc", random_state=0)
    rng = np.random.default_rng(0)
    rand = rate_fn(scores.scores, rng.normal(size=len(z)), metric="autoc", random_state=0)
    assert oracle.estimate >= rand.estimate
    qini = rate_fn(scores.scores, 1.0 + z, metric="qini", random_state=0)
    assert qini.metric == "qini" and np.isfinite(qini.estimate)
    with pytest.raises(ValueError, match="metric"):
        rate_fn(scores.scores, 1.0 + z, metric="bogus", random_state=0)
