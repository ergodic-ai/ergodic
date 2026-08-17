# SPDX-License-Identifier: Apache-2.0
"""Tests for the CATE meta-learner coherence layer and Tier-B explainability.

Seeded, deterministic, against planted truths. The heterogeneous worlds plant a
known modifier so the per-unit effects, the BLP slope, the GATES gradient, the
calibration test, and the RATE can be checked against the truth and against a
homogeneous null.
"""

from __future__ import annotations

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
from ergodic.inference import DRLearner, Heterogeneity, RLearner, XLearner, dr_scores


def _hetero(
    n: int = 4000, seed: int = 0, *, b: float = 1.0
) -> tuple[TabularData, MixedGraph, np.ndarray]:
    """Continuous outcome with tau(z) = 1 + b*z, so the ATE is 1.0 and the CATE tracks 1 + b*z."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.7 * z))).astype(float)
    y = (1.0 + b * z) * a + 1.5 * z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    return data, dag(["Z -> A", "Z -> Y", "A -> Y"]), z


def _binary_hetero(n: int = 6000, seed: int = 1) -> tuple[TabularData, MixedGraph, np.ndarray]:
    """Binary outcome with a real interaction: the treatment effect rises with z."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.5 * z))).astype(float)
    logit = -0.2 + 1.5 * a + 0.5 * z + 1.2 * a * z
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
    data = tabular(
        pd.DataFrame({"A": a.astype(int), "Z": z, "Y": y.astype(int)}),
        types={"A": "categorical", "Y": "categorical"},
    )
    return data, dag(["Z -> A", "Z -> Y", "A -> Y"]), z


# -- the uniform per-unit surface --------------------------------------------


def _synthetic_ite(n: int = 50) -> ITEResult:
    rng = np.random.default_rng(7)
    values = rng.normal(size=n)
    factual = rng.normal(size=n)
    return ITEResult(
        treatment="T",
        outcome="Y",
        covariates=(),
        n=n,
        kind="point",
        source="scm",
        framing="levels=(0.0, 1.0)",
        _values=values,
        _factual=factual,
    )


def _synthetic_cate(n: int = 50) -> CATEResult:
    rng = np.random.default_rng(8)
    draws = rng.normal(loc=1.0, scale=0.3, size=(200, n))
    return CATEResult(
        treatment="A",
        outcome="Y",
        covariates=("Z",),
        n=n,
        kind="bootstrap",
        source="dr_learner",
        _values=draws.mean(axis=0),
        _draws=draws,
    )


@pytest.mark.parametrize("result", [_synthetic_cate(), _synthetic_ite()])
def test_base_surface_uniform(result: PerUnitEffect) -> None:
    """A CATEResult and an ITEResult pass the same base-surface contract."""
    n = len(result)
    assert result.shape == (n,)
    assert result.values.shape == (n,)
    assert np.asarray(result).shape == (n,)
    assert result.index.shape == (n,)
    # arithmetic and numpy reductions go through __array__
    assert float(np.mean(result)) == pytest.approx(float(np.mean(result.values)))
    assert result[0] == pytest.approx(result.values[0])
    # ate() bridges to a scalar EffectEstimate
    eff = result.ate()
    assert eff.value == pytest.approx(float(np.mean(result.values)), abs=1e-6)
    frame = result.to_frame()
    assert len(frame) == n
    assert "effect" in frame.columns


def test_cate_result_interval_and_draws() -> None:
    res = _synthetic_cate()
    lo, hi = res.interval(0.9)
    assert lo.shape == hi.shape == (len(res),)
    assert bool(np.all(lo <= hi))
    draws = res.draws()
    assert draws is not None and draws.shape == (200, len(res))
    se = res.std_error
    assert se is not None and se.shape == (len(res),)


def test_ite_result_carries_factual_and_defers_explain() -> None:
    res = _synthetic_ite()
    assert res.estimand == "ite"
    assert res.factual is not None and res.factual.shape == (len(res),)
    with pytest.raises(NotImplementedError, match="CATE"):
        res.explain()


# -- recovering the planted CATE ---------------------------------------------


@pytest.mark.parametrize("method", ["dr_learner", "r_learner", "x_learner", "t_learner"])
def test_cate_recovers_continuous_tau(method: str) -> None:
    data, graph, z = _hetero(seed=0)
    res = cate(graph, data, "A", "Y", method=method, random_state=0)
    assert isinstance(res, CATEResult)
    assert res.estimand == "cate"
    assert np.corrcoef(res.values, 1.0 + z)[0, 1] > 0.6


def test_cate_ate_matches_estimate_effect_and_truth() -> None:
    data, graph, _ = _hetero(seed=0)
    res = cate(graph, data, "A", "Y", method="dr_learner", random_state=0)
    eff = res.ate()
    assert abs(eff.value - 1.0) < 0.15  # planted ATE is 1.0
    direct = estimate_effect(graph, data, "A", "Y", strategy="dr_learner", random_state=0)
    assert abs(eff.value - direct.value) < 0.1


def test_cate_binary_outcome_recovers_heterogeneity() -> None:
    data, graph, z = _binary_hetero(seed=1)
    res = cate(graph, data, "A", "Y", method="dr_learner", random_state=1, treated=1)
    # the planted effect rises with z, so the CATE correlates positively with z
    assert np.corrcoef(res.values, z)[0, 1] > 0.3
    h = res.explain()
    assert h.calibration_test().rejects()


def test_cate_predicts_out_of_sample() -> None:
    train, graph, _ = _hetero(seed=0)
    fresh, _, z_fresh = _hetero(seed=5)
    res = cate(graph, train, "A", "Y", method="dr_learner", random_state=0)
    # the fitted estimator scores a fresh design through the same facade-built model
    from ergodic.inference import DRLearner as _DR

    model = _DR(random_state=0).fit(identify_effect(graph, "A", "Y"), train)
    fresh_cate = model.cate(fresh)
    assert fresh_cate.shape == (fresh.n_rows,)
    assert np.corrcoef(fresh_cate.values, 1.0 + z_fresh)[0, 1] > 0.6
    assert res.shape == (train.n_rows,)


# -- per-unit uncertainty ----------------------------------------------------


def test_dr_analytic_per_unit_interval() -> None:
    data, graph, _ = _hetero(seed=0)
    res = cate(graph, data, "A", "Y", method="dr_learner", random_state=0)
    assert res.kind == "analytic"
    se = res.std_error
    assert se is not None and se.shape == (data.n_rows,)
    lo, hi = res.interval(0.9)
    assert bool(np.all(lo <= hi))


def test_bootstrap_per_unit_interval_brackets_truth() -> None:
    """The bootstrap per-unit interval (X-learner) brackets the linear truth often."""
    data, graph, z = _hetero(n=2000, seed=2)
    res = cate(graph, data, "A", "Y", method="x_learner", n_boot=200, random_state=2)
    assert res.kind == "bootstrap"
    lo, hi = res.interval(0.9)
    coverage = float(np.mean((lo <= 1.0 + z) & (1.0 + z <= hi)))
    assert coverage > 0.5  # the fitted-CATE bootstrap brackets the linear truth


# -- BLP ----------------------------------------------------------------------


def test_blp_slope_recovers_modifier() -> None:
    data, graph, _ = _hetero(n=4000, seed=1, b=1.0)
    h = cate(graph, data, "A", "Y", random_state=1).explain()
    frame = h.blp().frame()
    z = frame["Z"]
    assert abs(z["coef"] - 1.0) < 0.15  # the planted modifier slope is 1.0
    assert z["ci_low"] < 1.0 < z["ci_high"]  # CI covers the truth
    assert z["ci_low"] > 0.0  # and excludes zero
    assert abs(frame["(intercept)"]["coef"] - 1.0) < 0.15  # the intercept is the ATE


def test_blp_slope_covers_zero_under_null() -> None:
    data, graph, _ = _hetero(n=4000, seed=3, b=0.0)
    h = cate(graph, data, "A", "Y", random_state=3).explain()
    z = h.blp().frame()["Z"]
    assert z["ci_low"] < 0.0 < z["ci_high"]


# -- GATES --------------------------------------------------------------------


def test_gates_monotone_under_gradient() -> None:
    data, graph, _ = _hetero(n=4000, seed=0, b=1.0)
    h = cate(graph, data, "A", "Y", random_state=0).explain()
    g = h.gates(n_groups=5)
    assert g.is_monotone()
    lo, _hi = g.interval()
    assert lo > 0.0  # top-minus-bottom contrast excludes zero


def test_gates_flat_under_null() -> None:
    data, graph, _ = _hetero(n=4000, seed=3, b=0.0)
    h = cate(graph, data, "A", "Y", random_state=3).explain()
    g = h.gates(n_groups=5)
    lo, hi = g.interval()
    assert lo < 0.0 < hi  # the contrast CI covers zero


# -- CLAN ---------------------------------------------------------------------


def test_clan_profiles_the_high_group() -> None:
    data, graph, _ = _hetero(n=4000, seed=0, b=1.0)
    h = cate(graph, data, "A", "Y", random_state=0).explain()
    z = h.clan().frame()["Z"]
    # the high-effect group has larger z (the effect rises with z)
    assert z["difference"] > 0.0
    assert z["ci_low"] > 0.0


# -- calibration test ---------------------------------------------------------


def test_calibration_rejects_under_hte() -> None:
    data, graph, _ = _hetero(n=4000, seed=0, b=1.0)
    h = cate(graph, data, "A", "Y", random_state=0).explain()
    cal = h.calibration_test()
    assert cal.rejects()
    assert cal.differential_coefficient > 0.5


def test_calibration_not_rejected_under_null() -> None:
    data, graph, _ = _hetero(n=4000, seed=3, b=0.0)
    h = cate(graph, data, "A", "Y", random_state=3).explain()
    assert not h.calibration_test().rejects()


# -- RATE ---------------------------------------------------------------------


def test_rate_positive_under_hte() -> None:
    data, graph, _ = _hetero(n=4000, seed=0, b=1.0)
    h = cate(graph, data, "A", "Y", random_state=0).explain()
    r = h.rate("autoc", random_state=0)
    assert r.estimate > 0.0
    lo, _hi = r.interval()
    assert lo > 0.0  # excludes zero
    assert r.rejects()


def test_rate_zero_under_null() -> None:
    data, graph, _ = _hetero(n=4000, seed=3, b=0.0)
    h = cate(graph, data, "A", "Y", random_state=3).explain()
    r = h.rate("autoc", random_state=0)
    lo, hi = r.interval()
    assert lo < 0.0 < hi  # covers zero
    assert not r.rejects()


def test_rate_oracle_beats_random() -> None:
    data, graph, z = _hetero(n=3000, seed=0, b=1.0)
    scores = dr_scores(identify_effect(graph, "A", "Y"), data, random_state=0)
    from ergodic.inference.explain._rate import rate as rate_fn

    oracle = rate_fn(scores.scores, 1.0 + z, metric="autoc", random_state=0)
    rng = np.random.default_rng(0)
    random_rule = rate_fn(scores.scores, rng.normal(size=len(z)), metric="autoc", random_state=0)
    assert oracle.estimate >= random_rule.estimate


def test_qini_metric_runs() -> None:
    data, graph, _ = _hetero(n=2000, seed=0, b=1.0)
    h = cate(graph, data, "A", "Y", random_state=0).explain()
    q = h.rate("qini", random_state=0)
    assert q.metric == "qini"
    assert q.estimate > 0.0


# -- the shared score helper --------------------------------------------------


def test_dr_scores_mean_is_aipw_ate() -> None:
    data, graph, _ = _hetero(n=4000, seed=0)
    scores = dr_scores(identify_effect(graph, "A", "Y"), data, random_state=0)
    assert scores.scores.shape == (data.n_rows,)
    assert abs(float(np.mean(scores.scores)) - 1.0) < 0.15  # the AIPW ATE point
    aipw = estimate_effect(graph, data, "A", "Y", strategy="aipw", random_state=0)
    assert abs(float(np.mean(scores.scores)) - aipw.value) < 0.1


def test_explain_returns_heterogeneity() -> None:
    data, graph, _ = _hetero(seed=0)
    h = cate(graph, data, "A", "Y", random_state=0).explain()
    assert isinstance(h, Heterogeneity)
    assert h.treatment == "A"
    assert h.outcome == "Y"


# -- DML.cate -----------------------------------------------------------------


def test_dml_cate_is_constant_and_explains() -> None:
    data, graph, _ = _hetero(seed=0)
    res = cate(graph, data, "A", "Y", method="dml", random_state=0)
    assert isinstance(res, CATEResult)
    assert float(np.std(res.values)) == pytest.approx(0.0, abs=1e-9)  # homogeneous
    assert abs(res.ate().value - 1.0) < 0.15
    # DML's tau_hat is constant, so the tau_hat-ranked calibration test honestly
    # cannot detect heterogeneity (no usable ranking). The BLP regresses the DR
    # scores on the covariates directly, so it still detects what DML assumes away.
    h = res.explain()
    assert not h.calibration_test().rejects()  # degenerate ranking -> no detection
    blp_z = h.blp().frame()["Z"]
    assert blp_z["ci_low"] > 0.0  # the covariate projection still finds the gradient


# -- the meta-learners return a CATEResult directly ---------------------------


@pytest.mark.parametrize("cls", [DRLearner, RLearner, XLearner])
def test_estimator_cate_returns_result(cls: type) -> None:
    data, graph, z = _hetero(seed=0)
    ident = identify_effect(graph, "A", "Y")
    res = cls(random_state=0).fit(ident, data).cate()
    assert isinstance(res, CATEResult)
    assert np.corrcoef(res.values, 1.0 + z)[0, 1] > 0.6


def test_cate_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        DRLearner().cate()


# -- identification gating ----------------------------------------------------


def test_cate_gated_on_identifiability() -> None:
    rng = np.random.default_rng(0)
    n = 500
    u = rng.normal(size=n)
    a = (rng.uniform(size=n) < 0.5).astype(float)
    y = a + u + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Y": y}))
    # A <-> Y confounded by a latent, no adjustment set: not identifiable by back-door
    graph = dag(["A -> Y", "U -> A", "U -> Y"], latent=["U"])
    with pytest.raises(ValueError, match="not identifiable"):
        cate(graph, data, "A", "Y")


def test_cate_rejects_frontdoor_family() -> None:
    rng = np.random.default_rng(0)
    n = 800
    u = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-u))).astype(float)
    m = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(a)))).astype(float)
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(m + u)))).astype(float)
    data = tabular(
        pd.DataFrame({"A": a.astype(int), "M": m.astype(int), "Y": y.astype(int)}),
        types={"A": "categorical", "M": "categorical", "Y": "categorical"},
    )
    graph = dag(["A -> M", "M -> Y", "U -> A", "U -> Y"], latent=["U"])
    with pytest.raises(ValueError, match="adjustment"):
        cate(graph, data, "A", "Y")


def test_cate_unknown_method_raises() -> None:
    data, graph, _ = _hetero(seed=0)
    with pytest.raises(ValueError, match="unknown CATE method"):
        cate(graph, data, "A", "Y", method="not_a_method")


def test_cate_neural_seam_raises_not_implemented() -> None:
    data, graph, _ = _hetero(seed=0)
    with pytest.raises(NotImplementedError, match="neural"):
        cate(graph, data, "A", "Y", method="dragonnet")


def test_cate_foundation_seam_raises_not_implemented() -> None:
    data, graph, _ = _hetero(seed=0)
    with pytest.raises(NotImplementedError, match="foundation"):
        cate(graph, data, "A", "Y", method="causal_pfn")


# -- smoke --------------------------------------------------------------------


def test_cate_facade_smoke() -> None:
    data, graph, _ = _hetero(n=500, seed=0)
    res = cate(graph, data, "A", "Y", random_state=0)
    assert isinstance(res, CATEResult)
    assert len(res) == 500
    assert repr(res).startswith("CATEResult")
