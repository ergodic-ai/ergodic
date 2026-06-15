# SPDX-License-Identifier: Apache-2.0
"""SCM pillar: the pymc-gated golden-number live tests.

These run real PyMC against the closed-form ground truth, reproducing the
anchors the four-track pressure-test established. Seeded NUTS is bitwise
reproducible, so the numbers are golden; MCMC is kept cheap (draws/tune a few
hundred, two chains, a fixed seed) and tolerances are scaled to the sample size.

Skipped wholesale when pymc is absent (the ``test_bayes.py`` precedent), so the
bare install never needs it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pymc")

from ergodic import TabularData, dag, estimate_effect, tabular
from ergodic.scm import (
    SCM,
    CategoricalGumbelMax,
    CategoricalResult,
    SCMResult,
    Sensitivity,
)

DRAWS = 400
TUNE = 400
CHAINS = 2


# -- shared worlds (fit once, query many) ------------------------------------


@pytest.fixture(scope="module")
def fork() -> tuple[SCM, dict[str, np.ndarray]]:
    """The linear-Gaussian fork Z->X, Z->Y, X->Y.

    a=0.8, b=1.5, c=2.0, sx=0.7, sy=0.5 (strong confounding). The true direct
    effect is b=1.5; the confounded observational slope is ~2.92.
    """
    a, b, c, sx, sy = 0.8, 1.5, 2.0, 0.7, 0.5
    rng = np.random.default_rng(7)
    n = 4000
    z = rng.normal(0, 1, n)
    x = a * z + rng.normal(0, sx, n)
    y = b * x + c * z + rng.normal(0, sy, n)
    data = tabular({"Z": z, "X": x, "Y": y})
    graph = dag(["Z -> X", "Z -> Y", "X -> Y"])
    scm = SCM(graph, data, draws=DRAWS, tune=TUNE, chains=CHAINS, random_state=7).fit()
    return scm, {"Z": z, "X": x, "Y": y}


# -- ANCHOR: interventions sever the back-door -------------------------------


def test_intervention_severs_backdoor(fork: tuple[SCM, dict[str, np.ndarray]]) -> None:
    """``pm.do`` gives the causal slope ~1.5, not the confounded observational ~2.92."""
    scm, cols = fork
    r1 = scm.interventional("Y", {"X": 1.0})
    r0 = scm.interventional("Y", {"X": 0.0})
    interventional_slope = r1.mean() - r0.mean()
    observational_slope = float(np.cov(cols["X"], cols["Y"])[0, 1] / np.var(cols["X"]))
    assert interventional_slope == pytest.approx(1.5, abs=0.05)  # anchor 1.5092
    assert observational_slope == pytest.approx(2.92, abs=0.05)  # confounded
    # the gap is the confounding the do() removed
    assert observational_slope - interventional_slope > 1.0


def test_interventional_variance_matches_closed_form(
    fork: tuple[SCM, dict[str, np.ndarray]],
) -> None:
    """Y | do(X=x) ~ Normal(b x, c^2 + sy^2); the spread is the structural variance."""
    scm, _ = fork
    draws = scm.interventional("Y", {"X": 1.0}, n=4000).draws()
    # var ~ c^2 + sy^2 = 4 + 0.25 = 4.25 (plus a little parameter uncertainty)
    assert float(np.var(draws)) == pytest.approx(4.25, abs=0.5)


# -- ANCHOR: counterfactual exactness (invertible) ---------------------------


def test_counterfactual_is_machine_precision(
    fork: tuple[SCM, dict[str, np.ndarray]],
) -> None:
    """Y_cf = y + b(x' - x) to machine precision in the deterministic-residual form."""
    scm, cols = fork
    unit = scm.data.take([17])
    x_obs = float(cols["X"][17])
    y_obs = float(cols["Y"][17])
    x_prime = x_obs + 1.5
    cf = scm.counterfactual(unit, {"X": x_prime}, "Y", use_draws=False)
    # closed form at the posterior-mean b: Y_cf = y + b (x' - x)
    parents = scm.mechanisms["Y"].parents
    b_x = float(scm._fits["Y"].param_means["Y__coef"][parents.index("X")])
    y_cf_closed = y_obs + b_x * (x_prime - x_obs)
    assert isinstance(cf, SCMResult)
    assert cf.identified is True
    assert float(cf.mean()) == pytest.approx(y_cf_closed, abs=1e-10)  # machine precision


def test_batched_counterfactual_reproduces_analytic(
    fork: tuple[SCM, dict[str, np.ndarray]],
) -> None:
    """The units-axis abduction reproduces the analytic counterfactual (the 0.594 anchor style)."""
    scm, cols = fork
    units = scm.data.take(list(range(500)))
    cf = scm.counterfactual(units, {"X": 0.0}, "Y", use_draws=False)
    assert isinstance(cf, SCMResult)
    parents = scm.mechanisms["Y"].parents
    b_x = float(scm._fits["Y"].param_means["Y__coef"][parents.index("X")])
    analytic = float(np.mean(cols["Y"][:500] + b_x * (0.0 - cols["X"][:500])))
    assert float(cf.mean()) == pytest.approx(analytic, abs=1e-9)


def test_counterfactual_with_draws_carries_parameter_uncertainty(
    fork: tuple[SCM, dict[str, np.ndarray]],
) -> None:
    """With draws the counterfactual is a distribution; its sd ~ |x'-x| sd(b) > 0."""
    scm, cols = fork
    unit = scm.data.take([17])
    x_prime = float(cols["X"][17]) + 1.5
    cf = scm.counterfactual(unit, {"X": x_prime}, "Y", use_draws=True)
    assert isinstance(cf, SCMResult)
    assert cf.std_error is not None and cf.std_error > 0.0
    # mean still tracks the exact point
    parents = scm.mechanisms["Y"].parents
    b_x = float(scm._fits["Y"].param_means["Y__coef"][parents.index("X")])
    assert float(cf.mean()) == pytest.approx(float(cols["Y"][17]) + b_x * 1.5, abs=0.02)


# -- ANCHOR: chain propagation -----------------------------------------------


def test_chain_intervention_propagates() -> None:
    """do(B=2) on A->B->C->D leaves A unchanged, pins B, gives C~2.600, D~1.819."""
    rng = np.random.default_rng(11)
    n = 5000
    a = rng.normal(1, 1, n)
    b = 0.9 * a + rng.normal(0, 0.5, n)
    c = 1.3 * b + rng.normal(0, 0.4, n)
    d = 0.7 * c + rng.normal(0, 0.6, n)
    data = tabular({"A": a, "B": b, "C": c, "D": d})
    scm = SCM(
        dag(["A -> B", "B -> C", "C -> D"]),
        data,
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        random_state=11,
    ).fit()
    assert scm.interventional("A", {"B": 2.0}).mean() == pytest.approx(1.0, abs=0.06)
    assert scm.interventional("C", {"B": 2.0}).mean() == pytest.approx(2.600, abs=0.05)
    assert scm.interventional("D", {"B": 2.0}).mean() == pytest.approx(1.819, abs=0.05)


# -- ANCHOR: noise honesty ---------------------------------------------------


def test_noise_recovers_and_does_not_collapse(
    fork: tuple[SCM, dict[str, np.ndarray]],
) -> None:
    """Sigma recovers near the true scale under the Bayesian-linear mean."""
    scm, _ = fork
    sigma_y = float(scm._fits["Y"].param_means["Y__sigma"])
    sigma_x = float(scm._fits["X"].param_means["X__sigma"])
    assert sigma_y == pytest.approx(0.5, abs=0.05)  # true sy=0.5, does not collapse
    assert sigma_x == pytest.approx(0.7, abs=0.05)  # true sx=0.7
    assert sigma_y > 0.4  # the honesty floor: not driven to zero


def test_interpolator_control_residual_is_zero() -> None:
    """The control to avoid: a true interpolator's in-sample residual std is 0 exactly.

    A flexible mean stays honest (sigma above), a 1-NN interpolator does not. This
    pins the failure mode the noise-honesty claim guards against.
    """
    from ergodic.inference.learners import KNNRegressor

    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(-3, 3, n)
    y = np.sin(3 * x) + rng.normal(0, 0.5, n)
    knn = KNNRegressor(k=1).fit(x.reshape(-1, 1), y)
    resid = y - np.asarray(knn.predict(x.reshape(-1, 1)))
    assert float(np.std(resid)) == pytest.approx(0.0, abs=1e-12)


# -- ANCHOR: Gumbel-max categorical counterfactual ---------------------------


def _categorical_world() -> tuple[SCM, TabularData]:
    rng = np.random.default_rng(0)
    n, k = 2000, 4
    z = rng.normal(size=n)
    w = np.array([0.0, 0.8, -0.5, 0.3])
    bias = np.array([0.2, -0.1, 0.0, 0.4])
    logits = z[:, None] * w[None, :] + bias[None, :]
    g_true = rng.gumbel(0, 1, size=(n, k))
    obs = np.argmax(logits + g_true, axis=1)
    data = tabular({"Z": z, "C": obs}, types={"C": "categorical"})
    scm = SCM(dag(["Z -> C"]), data, draws=DRAWS, tune=TUNE, chains=CHAINS, random_state=0).fit()
    return scm, data


def test_gumbel_max_abduction_reconstructs_observed() -> None:
    """The abduction reconstructs the observed category (~1.000)."""
    scm, data = _categorical_world()
    mech = scm.mechanisms["C"]
    assert isinstance(mech, CategoricalGumbelMax)
    pv = data.matrix(["Z"])
    codes = np.asarray(data.codes("C"), dtype=np.int64)
    gumbel = mech.abduct(scm._fits["C"], codes, pv, seed=5)
    recon = mech.decide(mech.logits(scm._fits["C"], pv), gumbel)
    assert float(np.mean(recon == codes)) == pytest.approx(1.0)


def test_gumbel_max_counterfactual_vs_interventional_tv_small() -> None:
    """CF and interventional marginals agree in distribution (TV ~0.005, pinned)."""
    scm, data = _categorical_world()
    cf = scm.counterfactual(data, {"Z": 2.0}, "C")
    assert isinstance(cf, CategoricalResult)
    assert cf.identified is True
    assert cf.per_unit() is not None
    # interventional marginal under do(Z=2): a fresh Gumbel draw, not the abducted one
    mech = scm.mechanisms["C"]
    assert isinstance(mech, CategoricalGumbelMax)
    n = data.n_rows
    logits_cf = mech.logits(scm._fits["C"], np.full((n, 1), 2.0))
    fresh = np.random.default_rng(99).gumbel(0, 1, size=logits_cf.shape)
    interv = np.argmax(logits_cf + fresh, axis=1)
    iv_marg = np.bincount(interv, minlength=mech.n_categories) / n
    tv = 0.5 * float(np.abs(cf.probabilities - iv_marg).sum())
    # pinned to the seeded anchor 0.005 (the old abs=0.02 only asserted tv < 0.028).
    assert tv == pytest.approx(0.005, abs=0.002)


# -- ANCHOR: the honesty thesis (bow returns bounds) -------------------------


@pytest.fixture(scope="module")
def bow() -> tuple[SCM, dict[str, float]]:
    """The bow U->X, U->Y, X->Y with U latent. The pressure-test parameters.

    a_x=1.3, a_y=1.1, b=0.7, s_x=0.6, s_y=0.5, seed 20240615, N=400. The true
    causal effect is 0.7; the data cannot pin it (the effect is not identified).
    """
    a_x, a_y, b, s_x, s_y = 1.3, 1.1, 0.7, 0.6, 0.5
    rng = np.random.default_rng(20240615)
    n = 400
    u = rng.normal(0, 1, n)
    x = a_x * u + rng.normal(0, s_x, n)
    y = b * x + a_y * u + rng.normal(0, s_y, n)
    data = tabular({"X": x, "Y": y})
    graph = dag(["U -> X", "U -> Y", "X -> Y"], latent=["U"])
    scm = SCM(graph, data, draws=300, tune=300, chains=CHAINS, random_state=20240615).fit()
    return scm, {"a_x": a_x, "a_y": a_y, "b": b, "true_c": a_x * a_y}


def test_bow_returns_sensitivity_not_a_point(bow: tuple[SCM, dict[str, float]]) -> None:
    """A non-identified effect returns Sensitivity; a naked point read is impossible."""
    scm, _ = bow
    res = scm.ate("X", "Y")
    assert isinstance(res, Sensitivity)
    assert res.identified is False
    assert not hasattr(res, "value")
    assert not hasattr(res, "mean")


def test_bow_sensitivity_curve_brackets_the_truth(
    bow: tuple[SCM, dict[str, float]],
) -> None:
    """b(0) is the confounded slope (~1.406); the truth lies on the curve, near true_c.

    Two honest claims, neither slack: the no-confounding end is the confounded
    observational slope; and the curve is the exact line whose value at the *true*
    structural channel strength is the finite-sample 0.637 (not the population 0.7),
    while the truth 0.7 itself is hit at the empirical balancing c, which sits right
    beside true_c. That is what "the data cannot pin it, but the curve carries it"
    means here.
    """
    scm, p = bow
    res = scm.ate("X", "Y")
    assert isinstance(res, Sensitivity)
    # b(0): assume no confounding -> the observational slope, the confounded ~1.406.
    assert res.identified_at_zero == pytest.approx(1.4056, abs=0.005)
    # at the *true* structural confounding channel, the finite-sample curve gives
    # 0.637 -- pinned tight, the honest value (the population truth is 0.7).
    assert res.at(p["true_c"]) == pytest.approx(0.637, abs=0.005)
    # the curve still passes through the population truth 0.7, at the empirical
    # balancing channel c* = cov - 0.7 var, which lies right next to true_c=1.43.
    x = np.asarray(scm.data.matrix(["X"]).reshape(-1))
    y = np.asarray(scm.data.matrix(["Y"]).reshape(-1))
    cov = float(np.cov(x, y)[0, 1])
    var = float(np.var(x, ddof=1))
    c_star = cov - 0.7 * var
    assert res.at(c_star) == pytest.approx(0.7, abs=1e-6)  # exact line: hits the truth
    assert abs(c_star - p["true_c"]) < 0.2  # and c* sits beside the true channel
    # the truth lies inside the swept range; the confounded slope is the other end.
    lo, hi = res.range()
    assert lo <= 0.7 <= hi
    assert hi == pytest.approx(1.4056, abs=0.005)


def test_bow_ate_refuses_dml_and_returns_bounds(bow: tuple[SCM, dict[str, float]]) -> None:
    """Even with engine='dml', a non-identified effect short-circuits to bounds."""
    scm, _ = bow
    # the identification gate runs before the engine switch, so a non-identified
    # effect returns bounds rather than reaching the dml stub.
    res = scm.ate("X", "Y", engine="dml")
    assert isinstance(res, Sensitivity)


# -- ANCHOR: integration with estimate_effect --------------------------------


def test_ate_matches_estimate_effect_on_identified_world(
    fork: tuple[SCM, dict[str, np.ndarray]],
) -> None:
    """scm.ate() on an identified world is close to estimate_effect (one truth)."""
    scm, cols = fork
    ate = scm.ate("X", "Y")
    assert isinstance(ate, SCMResult)
    assert ate.identified is True
    # continuous treatment: compare against the DML estimator (binary-or-continuous).
    data = tabular(dict(cols))
    ee = estimate_effect(scm.graph, data, "X", "Y", strategy="dml", random_state=7)
    assert ate.mean() == pytest.approx(ee.value, abs=0.05)
    assert ate.mean() == pytest.approx(1.5, abs=0.05)  # true direct effect


def test_observational_sample_runs(fork: tuple[SCM, dict[str, np.ndarray]]) -> None:
    """The observational posterior predictive samples and is centered near the data."""
    scm, cols = fork
    res = scm.marginal("Y", n=2000)
    assert isinstance(res, SCMResult)
    # E[Y] ~ 0 under the generative law (all roots mean-zero); the data mean is ~0.
    assert res.mean() == pytest.approx(float(np.mean(cols["Y"])), abs=0.3)
