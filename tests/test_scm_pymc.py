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
    assert isinstance(r1, SCMResult) and isinstance(r0, SCMResult)
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
    res = scm.interventional("Y", {"X": 1.0}, n=4000)
    assert isinstance(res, SCMResult)
    draws = res.draws()
    # var ~ c^2 + sy^2 = 4 + 0.25 = 4.25 (plus a little parameter uncertainty)
    assert float(np.var(draws)) == pytest.approx(4.25, abs=0.5)


def test_interventional_and_sample_are_labelled_predictive(
    fork: tuple[SCM, dict[str, np.ndarray]],
) -> None:
    """Regression (API-149): the predictive spread of Y was printed as a 95% CrI.

    The interventional draws are one unit's outcome each, so their sd is the
    structural spread (~2.06 here), far wider than the uncertainty in the mean.
    """
    scm, _ = fork
    res = scm.interventional("Y", {"X": 1.0}, n=2000)
    assert isinstance(res, SCMResult)
    assert res.kind == "predictive"
    assert "predictive interval" in repr(res) and "CrI" not in repr(res)
    assert res.std_error is not None and res.std_error > 1.5
    sample = scm.marginal("Y", n=500)
    assert isinstance(sample, SCMResult) and sample.kind == "predictive"
    assert scm.cores == 1


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

    def mean_of(target: str) -> float:
        res = scm.interventional(target, {"B": 2.0})
        assert isinstance(res, SCMResult)
        return res.mean()

    assert mean_of("A") == pytest.approx(1.0, abs=0.06)
    assert mean_of("C") == pytest.approx(2.600, abs=0.05)
    assert mean_of("D") == pytest.approx(1.819, abs=0.05)


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


# -- ANCHOR: binary-treatment back-door ATE (the mixed-mechanism headline) ----


@pytest.fixture(scope="module")
def binary_fork() -> tuple[SCM, dict[str, np.ndarray]]:
    """Z (continuous) -> T (BINARY) and -> Y; T -> Y with a planted effect tau=2.0.

    T is logistic in Z, Y = tau*T + g*Z + noise (g=1.5). The back-door {Z} identifies
    the effect; the naive E[Y|T=1]-E[Y|T=0] is confounded upward (~3.29). The SCM
    must sample the binary T in the generative model (Gumbel-max) and pin it as an
    integer code under do(), so the g-formula recovers tau.
    """
    tau, g = 2.0, 1.5
    rng = np.random.default_rng(7)
    n = 4000
    z = rng.normal(0, 1, n)
    p_t = 1.0 / (1.0 + np.exp(-(1.1 * z)))
    t = rng.binomial(1, p_t).astype(int)
    y = tau * t + g * z + rng.normal(0, 0.6, n)
    data = tabular({"Z": z, "T": t, "Y": y}, types={"T": "categorical"})
    graph = dag(["Z -> T", "Z -> Y", "T -> Y"])
    scm = SCM(graph, data, draws=DRAWS, tune=TUNE, chains=CHAINS, random_state=7).fit()
    return scm, {"Z": z, "T": t, "Y": y, "tau": np.array([tau])}


def test_binary_treatment_ate_matches_estimate_effect_and_truth(
    binary_fork: tuple[SCM, dict[str, np.ndarray]],
) -> None:
    """THE anchor: scm.ate(binary T, Y) == estimate_effect == the planted tau.

    The seeded golden numbers: scm.ate ~ 1.994, estimate_effect ~ 1.989, both at the
    planted tau=2.0, while the confounded naive diff is ~3.288. This is the headline
    proof that a binary treatment is first-class in the generative model.
    """
    scm, cols = binary_fork
    ate = scm.ate("T", "Y")
    assert isinstance(ate, SCMResult)
    assert ate.identified is True

    data = tabular(
        {"Z": cols["Z"], "T": cols["T"].astype(int), "Y": cols["Y"]}, types={"T": "categorical"}
    )
    ee = estimate_effect(scm.graph, data, "T", "Y")
    naive = float(cols["Y"][cols["T"] == 1].mean() - cols["Y"][cols["T"] == 0].mean())

    # all three agree at the truth; the naive diff is confounded well above it.
    assert ate.mean() == pytest.approx(2.0, abs=0.06)  # golden ~1.994
    assert ate.mean() == pytest.approx(ee.value, abs=0.06)
    assert naive == pytest.approx(3.288, abs=0.05)  # confounded
    assert naive - ate.mean() > 1.0  # the confounding the back-door removed


def test_binary_treatment_ate_default_levels_are_zero_one(
    binary_fork: tuple[SCM, dict[str, np.ndarray]],
) -> None:
    """No explicit levels needed for a binary treatment: do(T=1) - do(T=0)."""
    scm, _ = binary_fork
    # explicit (0, 1) and the default agree exactly (same seed, same contrast).
    a_default = scm.ate("T", "Y")
    a_explicit = scm.ate("T", "Y", levels=(0.0, 1.0))
    assert isinstance(a_default, SCMResult) and isinstance(a_explicit, SCMResult)
    assert a_default.mean() == pytest.approx(a_explicit.mean(), abs=1e-9)


# -- ANCHOR: a BINARY confounder is sampled and marginalized over -------------


def test_binary_confounder_ate_marginalizes_correctly() -> None:
    """Zc BINARY confounder -> Tc (binary) and -> Yc; planted Tc effect 1.0.

    The confounder is binary, so the generative model samples it via Gumbel-max and
    the back-door g-formula marginalizes over it. Golden: scm.ate ~ 0.970,
    estimate_effect ~ 0.971, planted 1.0; naive diff ~ 1.642 (confounded).
    """
    rng = np.random.default_rng(11)
    n = 4000
    zc = rng.binomial(1, 0.5, n).astype(int)
    p_t = 1.0 / (1.0 + np.exp(-(1.3 * zc - 0.6)))
    tc = rng.binomial(1, p_t).astype(int)
    yc = 1.0 * tc + 2.0 * zc + rng.normal(0, 0.5, n)
    data = tabular({"Zc": zc, "Tc": tc, "Yc": yc}, types={"Zc": "categorical", "Tc": "categorical"})
    graph = dag(["Zc -> Tc", "Zc -> Yc", "Tc -> Yc"])
    scm = SCM(graph, data, draws=DRAWS, tune=TUNE, chains=CHAINS, random_state=11).fit()
    ate = scm.ate("Tc", "Yc")
    assert isinstance(ate, SCMResult)
    ee = estimate_effect(graph, data, "Tc", "Yc")
    naive = float(yc[tc == 1].mean() - yc[tc == 0].mean())
    assert ate.mean() == pytest.approx(1.0, abs=0.06)  # golden ~0.970
    assert ate.mean() == pytest.approx(ee.value, abs=0.05)
    assert naive == pytest.approx(1.642, abs=0.05)  # confounded


# -- ANCHOR: mixed-parent encoding recovers a known coefficient --------------


def test_gaussian_child_recovers_binary_parent_coefficient() -> None:
    """A Gaussian child of a BINARY parent recovers the planted slope (2.5).

    The binary parent is one drop-first 0/1 column in the child's mean, so the
    fitted coefficient is the true effect, and the interventional contrast
    do(B=1)-do(B=0) equals it. Golden: coef ~ 2.499, contrast ~ 2.499.
    """
    rng = np.random.default_rng(3)
    n = 3000
    b = rng.binomial(1, 0.5, n).astype(int)
    yk = 1.0 + 2.5 * b + rng.normal(0, 0.4, n)
    data = tabular({"B": b, "Yk": yk}, types={"B": "categorical"})
    scm = SCM(dag(["B -> Yk"]), data, draws=DRAWS, tune=TUNE, chains=CHAINS, random_state=3).fit()
    coef_b = float(scm._fits["Yk"].param_means["Yk__coef"][0])
    assert coef_b == pytest.approx(2.5, abs=0.06)  # golden ~2.499
    r1 = scm.interventional("Yk", {"B": 1})
    r0 = scm.interventional("Yk", {"B": 0})
    assert isinstance(r1, SCMResult) and isinstance(r0, SCMResult)
    assert (r1.mean() - r0.mean()) == pytest.approx(2.5, abs=0.06)  # contrast == coef


def test_categorical_child_logits_depend_on_parent_of_either_type() -> None:
    """A categorical child's class marginal shifts with a continuous parent's do().

    do(Z2) on Z2 -> C changes the per-category interventional distribution (the
    logits are linear in the parent), confirming the categorical generative node
    reads its parent correctly.
    """
    rng = np.random.default_rng(0)
    n, k = 2000, 4
    z = rng.normal(size=n)
    w = np.array([0.0, 0.9, -0.6, 0.3])
    logits = z[:, None] * w[None, :]
    obs = np.argmax(logits + rng.gumbel(0, 1, size=(n, k)), axis=1)
    data = tabular({"Z2": z, "C": obs}, types={"C": "categorical"})
    scm = SCM(dag(["Z2 -> C"]), data, draws=DRAWS, tune=TUNE, chains=CHAINS, random_state=0).fit()
    hi = scm.interventional("C", {"Z2": 2.0})
    lo = scm.interventional("C", {"Z2": -2.0})
    assert isinstance(hi, CategoricalResult) and isinstance(lo, CategoricalResult)
    # both are proper distributions
    assert float(hi.probabilities.sum()) == pytest.approx(1.0)
    assert float(lo.probabilities.sum()) == pytest.approx(1.0)
    # class 1 (slope +0.9) gains mass as Z2 rises; class 2 (slope -0.6) loses it.
    assert hi.probabilities[1] > lo.probabilities[1]
    assert hi.probabilities[2] < lo.probabilities[2]


# -- categorical TARGET readout (interventional + observational) -------------


def test_interventional_categorical_target_returns_distribution() -> None:
    """A categorical TARGET's interventional answer is a CategoricalResult marginal."""
    rng = np.random.default_rng(0)
    n, k = 2000, 4
    z = rng.normal(size=n)
    w = np.array([0.0, 0.8, -0.5, 0.3])
    bias = np.array([0.2, -0.1, 0.0, 0.4])
    logits = z[:, None] * w[None, :] + bias[None, :]
    obs = np.argmax(logits + rng.gumbel(0, 1, size=(n, k)), axis=1)
    data = tabular({"Z2": z, "C": obs}, types={"C": "categorical"})
    scm = SCM(dag(["Z2 -> C"]), data, draws=DRAWS, tune=TUNE, chains=CHAINS, random_state=0).fit()
    res = scm.interventional("C", {"Z2": 2.0})
    assert isinstance(res, CategoricalResult)
    assert res.identified is True
    assert len(res.categories) == 4
    assert float(res.probabilities.sum()) == pytest.approx(1.0)
    # sample() on a categorical sink also returns a CategoricalResult.
    s = scm.sample(n=500)
    assert isinstance(s, CategoricalResult)
    assert float(s.probabilities.sum()) == pytest.approx(1.0)
    # marginal() on the categorical node too.
    m = scm.marginal("C", n=500)
    assert isinstance(m, CategoricalResult)


# -- RV-naming discipline: the fit view and the generative view agree --------


def test_generative_rv_names_match_the_merged_posterior_golden() -> None:
    """Golden guard against silent name drift between the fit and generative views.

    ``sample_posterior_predictive`` matches by name and silently samples an unmatched
    name from the prior, so a rename gives garbage with no warning. This pins that
    every parameter the generative model declares is present in the merged posterior
    with the exact same name. If a mechanism's fit names and generative names ever
    diverge, this fails immediately.
    """
    from ergodic.scm._compose import _build_generative, _merged_idata, _resolved_mechanisms

    rng = np.random.default_rng(5)
    n = 800
    z = rng.normal(size=n)
    c = (z > 0).astype(int)  # binary node
    y = 1.0 + 2.0 * c + 0.5 * z + rng.normal(0, 0.4, n)
    data = tabular({"Z": z, "C": c, "Y": y}, types={"C": "categorical"})
    scm = SCM(
        dag(["Z -> C", "Z -> Y", "C -> Y"]), data, draws=200, tune=200, chains=2, random_state=5
    ).fit()

    mechs = _resolved_mechanisms(scm.mechanisms, scm._fits)
    model = _build_generative(scm._order, mechs, n)
    idata = _merged_idata(scm._fits)
    posterior_names = set(idata.posterior.data_vars)
    free_rv_names = {rv.name for rv in model.free_RVs}
    # every free PARAMETER RV (not a node RV) must be in the merged posterior. The
    # node RVs themselves (Z, C, Y) are the likelihood vars, not parameters.
    node_names = set(scm._order)
    param_rvs = free_rv_names - node_names
    assert param_rvs, "expected parameter RVs in the generative model"
    missing = param_rvs - posterior_names
    assert not missing, f"generative param RVs absent from the merged posterior: {missing}"
    # and the names are exactly the mechanism's declared names (single-sourced).
    expected = {
        "Z__intercept",
        "Z__sigma",
        "C__logit_b",
        "C__logit_w",
        "Y__intercept",
        "Y__sigma",
        "Y__coef",
    }
    assert param_rvs == expected


# -- mixed twin-network counterfactual: a continuous target over discrete ancestors


def test_continuous_counterfactual_with_binary_parent_is_exact() -> None:
    """Flipping a BINARY parent moves the continuous child by its coefficient, exactly.

    The Gaussian hop stays machine-precision in the deterministic-residual form even
    when a parent is the binary node's drop-first 0/1 column: Y_cf - Y_obs equals the
    fitted coefficient on the binary parent (the abducted residual cancels).
    """
    rng = np.random.default_rng(2)
    n = 2500
    z = rng.normal(size=n)
    b = rng.binomial(1, 1.0 / (1.0 + np.exp(-z))).astype(int)
    y = 0.5 + 3.0 * b + 0.8 * z + rng.normal(0, 0.4, n)
    data = tabular({"Z": z, "B": b, "Y": y}, types={"B": "categorical"})
    scm = SCM(
        dag(["Z -> B", "Z -> Y", "B -> Y"]),
        data,
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        random_state=2,
    ).fit()
    idx = int(np.where(b == 0)[0][0])
    unit = data.take([idx])
    cf = scm.counterfactual(unit, {"B": 1}, "Y", use_draws=False)
    assert isinstance(cf, SCMResult)
    coef_b = float(
        scm._fits["Y"].param_means["Y__coef"][list(scm.mechanisms["Y"].parents).index("B")]
    )
    # the counterfactual delta is exactly the binary-parent coefficient.
    assert (cf.mean() - float(y[idx])) == pytest.approx(coef_b, abs=1e-9)


def test_categorical_ancestor_counterfactual_interval_spreads_per_draw() -> None:
    """M1 golden: a CF through a categorical ANCESTOR carries its parameter width.

    Near-boundary topology Z -> C(binary) -> Y, Z -> Y at a SMALL n, so C's
    logit-slope posterior sd is large and C's decision flips across draws under an
    upstream do(Z). C dominates Y (coef ~6), so the flip is the lion's share of the
    counterfactual spread. The fix re-decides the ancestor PER posterior draw (the
    abducted Gumbel held fixed, the logit parameters integrated over), so that width
    enters the interval; the old code collapsed the ancestor to a single
    posterior-mean decision and dropped it.

    The discriminating signal: the per-draw counterfactual is BIMODAL (some draws on
    the C=0 branch near the Z direct effect, some on the C=1 branch near +6), so its
    95% width is coefficient-sized (>3). Under the posterior-mean collapse every draw
    sits in one narrow band (width ~0.3), so this test would fail if the code
    regressed. The point estimate (use_draws=False) and the far/large-n width are
    unchanged.
    """

    def _flip_world(n: int, seed: int) -> tuple[SCM, dict[str, np.ndarray]]:
        rng = np.random.default_rng(seed)
        z = rng.normal(0, 1, n)
        c = rng.binomial(1, 1.0 / (1.0 + np.exp(-(1.0 * z)))).astype(int)
        y = 6.0 * c + 0.1 * z + rng.normal(0, 0.05, n)
        data = tabular({"Z": z, "C": c, "Y": y}, types={"C": "categorical"})
        scm = SCM(
            dag(["Z -> C", "Z -> Y", "C -> Y"]),
            data,
            draws=300,
            tune=300,
            chains=CHAINS,
            random_state=seed,
        ).fit()
        return scm, {"Z": z, "C": c, "Y": y}

    def _pick_flip(scm: SCM, cols: dict[str, np.ndarray]) -> tuple[int, float]:
        # the (unit, do(Z)) whose C decision is closest to 50/50 across the logit
        # posterior, using the SAME abduction seed the counterfactual path uses.
        mech = scm.mechanisms["C"]
        assert isinstance(mech, CategoricalGumbelMax)
        fit = scm._fits["C"]
        c_seed = scm._seed(6) + list(scm._order).index("C")
        best = (0, 0.0, 1.0)
        for i in range(scm.data.n_rows):
            zf = float(cols["Z"][i])
            g = mech.abduct(fit, np.array([cols["C"][i]]), np.array([[zf]]), seed=c_seed)
            for zp in np.linspace(-6, 6, 61):
                ld = mech.logits_per_draw(fit, np.full((1, 1, 1), zp))
                frac1 = float(np.argmax(ld + g[None], axis=-1).mean())
                if abs(frac1 - 0.5) < best[2]:
                    best = (i, float(zp), abs(frac1 - 0.5))
        return best[0], best[1]

    # NEAR-BOUNDARY: small n -> wide logit posterior, the ancestor flips across draws.
    scm, cols = _flip_world(n=15, seed=3)
    idx, z_prime = _pick_flip(scm, cols)
    unit = scm.data.take([idx])

    near = scm.counterfactual(unit, {"Z": z_prime}, "Y", use_draws=True)
    assert isinstance(near, SCMResult)
    draws = near.draws()
    # the interval is coefficient-sized (the flip is priced in), NOT the ~0.3-wide
    # continuous-hop-only band the posterior-mean collapse would give.
    lo, hi = near.interval(0.95)
    assert (hi - lo) > 3.0, f"95% width {hi - lo:.3f} too narrow -- collapsed?"
    assert near.std_error is not None and near.std_error > 1.0
    # genuinely bimodal: both branches appear (C=0 branch near 0, C=1 branch near +6).
    assert float(draws.min()) < 1.0
    assert float(draws.max()) > 3.0
    assert 0.05 < float((draws > 3.0).mean()) < 0.95  # a real per-draw flip, not all-one

    # the POINT is unchanged: the deterministic use_draws=False path is exact and
    # lands at one branch (the posterior-mean decision), unmoved by the fix.
    point = scm.counterfactual(unit, {"Z": z_prime}, "Y", use_draws=False)
    assert isinstance(point, SCMResult)
    assert point.std_error is None  # a single deterministic value

    # FAR / LARGE-n: a tight logit posterior -> no per-draw flip -> width ~1x (tiny),
    # exactly as before (the fix touches only the flipping case).
    scm_far, cols_far = _flip_world(n=4000, seed=3)
    far_unit = scm_far.data.take([0])
    z_far = float(cols_far["Z"][0]) + 4.0  # push Z hard: C is firmly category 1
    far = scm_far.counterfactual(far_unit, {"Z": z_far}, "Y", use_draws=True)
    assert isinstance(far, SCMResult)
    flo, fhi = far.interval(0.95)
    assert (fhi - flo) < 0.2  # unchanged ~1x: no flip to spread


def test_interventional_propagates_through_a_binary_mediator() -> None:
    """do() on a root propagates through a BINARY mediator to a continuous outcome.

    Z -> B (binary) -> Y with Z -> Y too; raising Z raises P(B=1) and Y, so the
    interventional mean rises. Confirms the generative model samples the binary
    mediator forward under an upstream intervention.
    """
    rng = np.random.default_rng(2)
    n = 2500
    z = rng.normal(size=n)
    b = rng.binomial(1, 1.0 / (1.0 + np.exp(-z))).astype(int)
    y = 0.5 + 3.0 * b + 0.8 * z + rng.normal(0, 0.4, n)
    data = tabular({"Z": z, "B": b, "Y": y}, types={"B": "categorical"})
    scm = SCM(
        dag(["Z -> B", "Z -> Y", "B -> Y"]),
        data,
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        random_state=2,
    ).fit()
    hi = scm.interventional("Y", {"Z": 2.0})
    lo = scm.interventional("Y", {"Z": -2.0})
    assert isinstance(hi, SCMResult) and isinstance(lo, SCMResult)
    assert hi.mean() - lo.mean() > 1.0  # Z's total effect through B and directly


# -- honesty NOT regressed: a bow with a binary treatment still returns bounds


def test_bow_with_binary_treatment_returns_sensitivity_not_a_point() -> None:
    """The honesty thesis survives a binary treatment: a latent-confounded effect is
    bounds, never a naked point. U -> X(binary), U -> Y, X -> Y with U latent; the
    effect is not identified, so ate() returns a Sensitivity with no scalar accessor.
    """
    rng = np.random.default_rng(20240615)
    n = 600
    u = rng.normal(0, 1, n)
    x = rng.binomial(1, 1.0 / (1.0 + np.exp(-(1.3 * u)))).astype(int)
    y = 0.7 * x + 1.1 * u + rng.normal(0, 0.5, n)
    data = tabular({"X": x, "Y": y}, types={"X": "categorical"})
    graph = dag(["U -> X", "U -> Y", "X -> Y"], latent=["U"])
    scm = SCM(graph, data, draws=300, tune=300, chains=CHAINS, random_state=20240615).fit()
    res = scm.ate("X", "Y")
    assert isinstance(res, Sensitivity)
    assert res.identified is False
    assert not hasattr(res, "value")
    assert not hasattr(res, "mean")


# == THE LATENT-CONFOUNDED ADMG PHASE: golden numbers =========================
#
# The headline regression-flip and the Track-D anchors, run live against the
# planted truths. A continuous-treatment front-door cannot be cross-checked by
# estimate_effect (E1 is binary-only), so the front-door anchor pins the PLANTED
# truth; the binary-treatment confounded world cross-checks estimate_effect where
# it works (already covered by test_binary_treatment_ate_matches_estimate_effect).


@pytest.fixture(scope="module")
def frontdoor() -> tuple[SCM, dict[str, float]]:
    """The C1 front-door world X->M->Y with U->X, U->Y (U latent).

    b_xm=1.5, b_my=0.8 so the front-door effect is b_xm*b_my = 1.2. U confounds X and
    Y (loadings a_x=1.3, a_y=1.1), so the naive Y~M slope is biased; only the latent
    {X, Y} district fit makes the mediated effect computable. The MVP raised here.
    """
    b_xm, b_my, a_x, a_y = 1.5, 0.8, 1.3, 1.1
    rng = np.random.default_rng(20240616)
    n = 4000
    u = rng.normal(0, 1, n)
    x = a_x * u + rng.normal(0, 0.6, n)
    m = b_xm * x + rng.normal(0, 0.5, n)
    y = b_my * m + a_y * u + rng.normal(0, 0.5, n)
    data = tabular({"X": x, "M": m, "Y": y})
    graph = dag(["X -> M", "M -> Y", "U -> X", "U -> Y"], latent=["U"])
    scm = SCM(graph, data, draws=DRAWS, tune=TUNE, chains=CHAINS, random_state=20240616).fit()
    return scm, {"b_xm": b_xm, "b_my": b_my, "fd_effect": b_xm * b_my}


def test_ate_computes_frontdoor_effect_near_truth(
    frontdoor: tuple[SCM, dict[str, float]],
) -> None:
    """THE headline flip: front-door now COMPUTES ~1.2 (planted), where the MVP raised.

    The {X, Y} district fits its explicit latent L__X__Y, so Y = b_my*M + loading*L +
    noise fits the true b_my (the latent absorbs the X-Y confounding the MVP plug-in
    baked into a wrong slope ~2.37). do(X) severs X from L, the L term is its marginal
    and cancels in the contrast, leaving the mediated b_xm*b_my = 1.2. Golden ~1.211.
    """
    scm, p = frontdoor
    ate = scm.ate("X", "Y")
    assert isinstance(ate, SCMResult)
    assert ate.identified is True
    # anchor on the PLANTED truth (estimate_effect cannot cross-check a continuous
    # front-door: E1 is binary-only). Golden seeded value ~1.2114.
    assert ate.mean() == pytest.approx(p["fd_effect"], abs=0.05)
    assert ate.mean() == pytest.approx(1.211, abs=0.02)
    # the naive U-confounded Y~X regression slope is the wrong number the MVP would
    # have returned -- well above the truth, which is what the latent fit corrects.
    x = np.asarray(scm.data.matrix(["X"]).reshape(-1))
    y = np.asarray(scm.data.matrix(["Y"]).reshape(-1))
    naive = float(np.cov(x, y)[0, 1] / np.var(x))
    assert naive > 1.5  # confounded well above the 1.2 truth
    assert abs(ate.mean() - p["fd_effect"]) < abs(naive - p["fd_effect"])


def test_frontdoor_districts_are_split(frontdoor: tuple[SCM, dict[str, float]]) -> None:
    """The structure the computation rests on: {X, Y} confounded, M a singleton."""
    scm, _ = frontdoor
    assert scm._latent.confounded_districts == (("X", "Y"),)
    assert scm._latent.singletons == ("M",)
    assert scm._latent.latents == ("L__X__Y",)


# -- ANCHOR: pm.do on a fitted latent district is exact ----------------------


def test_pm_do_on_fitted_district_is_exact(bow: tuple[SCM, dict[str, float]]) -> None:
    """``pm.do`` on the fitted bow district matches the analytic linear interventional.

    The Track-D 1.4008-vs-1.4000 anchor, here against the *fitted* structural
    coefficient: E[Y|do(X=x)] from the generative pm.do must equal intercept + b_hat*x
    (the latent term is L's marginal, mean zero, so it cancels). This pins that the
    latent-aware generative model computes the interventional from the fitted
    parameters exactly, not from a re-drawn or confounded path. The do-slope equals
    the fitted b_hat to ~1e-3.
    """
    scm, _ = bow
    fit = scm._fits["Y"]
    parents = scm.mechanisms["Y"].parents
    b_hat = float(fit.param_means["Y__coef"][list(parents).index("X")])
    intercept = float(fit.param_means["Y__intercept"])
    r2 = scm.interventional("Y", {"X": 2.0}, n=20000)
    r0 = scm.interventional("Y", {"X": 0.0}, n=20000)
    assert isinstance(r2, SCMResult) and isinstance(r0, SCMResult)
    # E[Y|do(X=2)] == intercept + b_hat*2 (the latent contributes its zero-mean marginal).
    assert r2.mean() == pytest.approx(intercept + b_hat * 2.0, abs=0.02)
    # the interventional per-unit slope (a 2-unit step) is exactly the fitted coefficient.
    assert (r2.mean() - r0.mean()) / 2.0 == pytest.approx(b_hat, abs=0.01)


# -- ANCHOR: the three-node two-latent district splits identified from not ----


def test_three_node_two_latent_district_splits() -> None:
    """A 3-node district with two shared latents fits; the clean edge recovers, the
    confounded one is ridge-y -- the split is automatic (Track-D exp4).

    ADMG A->B->C with A<->C (latent L__A__C) and B<->C (latent L__B__C). A->B has no
    open bidirected back-door, so bAB recovers ~0.6 with tight sampling. B->C is
    confounded by B<->C, so bBC sits on the ridge (~1.5 against a true 0.5) with a
    wide loading posterior. No special-casing -- the explicit latents reproduce the
    identification structure. Golden: bAB ~0.590, bBC ~1.546.
    """
    seed = 5
    rng = np.random.default_rng(seed)
    n = 4000
    g1, g2, h1, h2 = 1.0, 0.9, 0.8, 1.1
    b_ab, b_bc = 0.6, 0.5
    u1 = rng.normal(0, 1, n)
    u2 = rng.normal(0, 1, n)
    a = g1 * u1 + rng.normal(0, 0.5, n)
    b = b_ab * a + g2 * u2 + rng.normal(0, 0.5, n)
    c = b_bc * b + h1 * u1 + h2 * u2 + rng.normal(0, 0.5, n)
    data = tabular({"A": a, "B": b, "C": c})
    graph = dag(
        ["A -> B", "B -> C", "U1 -> A", "U1 -> C", "U2 -> B", "U2 -> C"],
        latent=["U1", "U2"],
    )
    scm = SCM(graph, data, draws=500, tune=800, chains=CHAINS, random_state=seed).fit()

    # the structure: one confounded district {A, B, C}, two latents.
    assert scm._latent.confounded_districts == (("A", "B", "C"),)
    assert scm._latent.latents == ("L__A__C", "L__B__C")
    # B's parents are (A,) + latent L__B__C; C's are (B,) + both latents.
    assert scm.mechanisms["B"].latents == ("L__B__C",)
    assert set(scm.mechanisms["C"].latents) == {"L__A__C", "L__B__C"}

    fit_b = scm._fits["B"]
    coef_ab = float(fit_b.param_means["B__coef"][list(scm.mechanisms["B"].parents).index("A")])
    fit_c = scm._fits["C"]
    coef_bc = float(fit_c.param_means["C__coef"][list(scm.mechanisms["C"].parents).index("B")])

    # the CLEAN edge A->B recovers its coefficient with tight sampling.
    assert coef_ab == pytest.approx(b_ab, abs=0.05)  # golden ~0.590
    sd_ab = float(np.std(fit_b.params["B__coef"][:, list(scm.mechanisms["B"].parents).index("A")]))
    assert sd_ab < 0.05  # healthy: the clean edge samples tightly

    # the CONFOUNDED edge B->C is biased onto the ridge, NOT near the truth -- this is
    # the non-identification signal the honesty layer bounds, not a recovered point.
    assert abs(coef_bc - b_bc) > 0.3  # off the truth (golden ~1.546)
    sd_bc = float(np.std(fit_c.params["C__coef"][:, list(scm.mechanisms["C"].parents).index("B")]))
    assert sd_bc > sd_ab  # the confounded edge's coefficient is far less determined


# -- ANCHOR: a confounded-district counterfactual is wider, scaling with lever


def test_confounded_district_counterfactual_is_wider_and_scales() -> None:
    """A bow-district CF is wider than the Markovian one, the width growing with the
    lever, and reproducible (Track-D anchor 3).

    The unit's Y-residual a_y*U + e_y is pinned by the observation, so the CF point is
    y_obs + b*(x_cf - x_obs); the WIDTH comes from b's posterior, which is the
    non-identification ridge in the bow (sd(b) ~0.5) versus tight in the Markovian
    control (~0.02). So the bow CF is many times wider, and -- because width ~
    |lever|*sd(b) -- it grows with the lever. The point estimate is unchanged.
    """
    a_x, a_y, b, s_x, s_y = 1.3, 1.1, 0.7, 0.6, 0.5
    n = 400

    rng = np.random.default_rng(20240615)
    u = rng.normal(0, 1, n)
    x = a_x * u + rng.normal(0, s_x, n)
    y = b * x + a_y * u + rng.normal(0, s_y, n)
    data_bow = tabular({"X": x, "Y": y})
    scm_bow = SCM(
        dag(["U -> X", "U -> Y", "X -> Y"], latent=["U"]),
        data_bow,
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        random_state=20240615,
    ).fit()

    rng2 = np.random.default_rng(20240615)
    xm = rng2.normal(0, 1, n)
    ym = b * xm + rng2.normal(0, s_y, n)
    data_m = tabular({"X": xm, "Y": ym})
    scm_m = SCM(
        dag(["X -> Y"]), data_m, draws=DRAWS, tune=TUNE, chains=CHAINS, random_state=20240615
    ).fit()

    def width(scm: SCM, data: TabularData, idx: int, lever: float) -> float:
        x_obs = float(np.asarray(data.matrix(["X"]).reshape(-1))[idx])
        cf = scm.counterfactual(data.take([idx]), {"X": x_obs + lever}, "Y", use_draws=True)
        assert isinstance(cf, SCMResult)
        lo, hi = cf.interval(0.89)
        return hi - lo

    idx_bow = int(np.argmax(np.abs(x)))  # a high-|x| unit, where the lever bites
    idx_m = int(np.argmax(np.abs(xm)))

    w_bow_1 = width(scm_bow, data_bow, idx_bow, -1.0)
    w_bow_5 = width(scm_bow, data_bow, idx_bow, -5.0)
    w_m_5 = width(scm_m, data_m, idx_m, -5.0)

    # the bow CF is much wider than the Markovian one at a big lever.
    assert w_bow_5 > 4.0 * w_m_5
    # and the bow CF width grows with the lever (width ~ |lever| * sd(b)).
    assert w_bow_5 > w_bow_1
    # the underlying mechanism: sd(b) is the ridge in the bow, tight in the control.
    sd_b_bow = float(np.std(scm_bow._fits["Y"].params["Y__coef"][:, 0]))
    sd_b_m = float(np.std(scm_m._fits["Y"].params["Y__coef"][:, 0]))
    assert sd_b_bow > 5.0 * sd_b_m

    # reproducible: a seeded refit gives a bitwise-identical CF width.
    scm_bow2 = SCM(
        dag(["U -> X", "U -> Y", "X -> Y"], latent=["U"]),
        data_bow,
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        random_state=20240615,
    ).fit()
    assert width(scm_bow2, data_bow, idx_bow, -5.0) == pytest.approx(w_bow_5, abs=1e-9)

    # the POINT is unchanged from the deterministic route (the residual is pinned).
    x_obs = float(x[idx_bow])
    point = scm_bow.counterfactual(
        data_bow.take([idx_bow]), {"X": x_obs - 5.0}, "Y", use_draws=False
    )
    assert isinstance(point, SCMResult)
    assert point.std_error is None  # a single deterministic value


# -- ANCHOR: the latent gauge -- a seeded refit is reproducible, no reflection


def test_latent_gauge_is_reproducible_and_pins_the_sign(
    bow: tuple[SCM, dict[str, float]],
) -> None:
    """One loading pinned positive: a seeded refit is bitwise reproducible and free of
    the L -> -L reflection multimodality.

    X is the gauge anchor for L__X__Y (the alphabetically-first endpoint), so its
    loading is a HalfNormal -- pinned positive. A second fit at the same seed
    reproduces the structural coefficient exactly (no mode-hopping), and the gauged
    loading is non-negative on every draw.
    """
    scm, _ = bow
    parents = scm.mechanisms["Y"].parents
    b_hat = float(scm._fits["Y"].param_means["Y__coef"][list(parents).index("X")])

    # the gauged loading on X is non-negative on every posterior draw (the HalfNormal).
    load_x = np.asarray(scm._fits["X"].params["X__loading__L__X__Y"])
    assert float(load_x.min()) >= 0.0
    assert scm.mechanisms["X"].gauged_latents == ("L__X__Y",)

    # a seeded refit of the same world reproduces the coefficient bitwise (no
    # reflection multimodality: the gauge fixes the sign, the seed fixes the draws).
    rng = np.random.default_rng(20240615)
    n = 400
    u = rng.normal(0, 1, n)
    x = 1.3 * u + rng.normal(0, 0.6, n)
    y = 0.7 * x + 1.1 * u + rng.normal(0, 0.5, n)
    data = tabular({"X": x, "Y": y})
    refit = SCM(
        dag(["U -> X", "U -> Y", "X -> Y"], latent=["U"]),
        data,
        draws=300,
        tune=300,
        chains=CHAINS,
        random_state=20240615,
    ).fit()
    b_refit = float(refit._fits["Y"].param_means["Y__coef"][list(parents).index("X")])
    assert b_refit == pytest.approx(b_hat, abs=1e-9)


# -- ANCHOR: IV and general-ID keep raising (verify-then-widen, decision 3) ----


def test_iv_identified_effect_makes_ate_raise_with_a_pointer() -> None:
    """IV stays raising: an instrument identifies the effect only parametrically.

    Z -> X -> Y with U -> X, U -> Y (U latent): the effect is identified through the
    instrument Z, which needs a linear, homogeneous effect. The SCM's do-contrast
    claims no such assumption, so ``ate()`` points at ``estimate_effect`` (2SLS)
    instead of returning the latent fit's value. The decision is made before any
    sampling. (Review F2, API-149: an earlier version also asserted that the latent
    fit misses the truth, but its posterior covers it; the 2SLS standard error here
    is about 0.16, so a single seed's distance from 0.7 tests noise.)
    """
    rng = np.random.default_rng(7)
    n = 4000
    u = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    x = 0.2 * z + 2.0 * u + rng.normal(0, 0.6, n)  # weak Z, heavy confounding
    y = 0.7 * x + 2.0 * u + rng.normal(0, 0.5, n)
    data = tabular({"Z": z, "X": x, "Y": y})
    graph = dag(["Z -> X", "X -> Y", "U -> X", "U -> Y"], latent=["U"])

    scm = SCM(graph, data, draws=DRAWS, tune=TUNE, chains=CHAINS, random_state=7)
    scm._fits = dict.fromkeys(scm.mechanisms)  # type: ignore[assignment]
    with pytest.raises(NotImplementedError) as exc:
        scm.ate("X", "Y")
    message = str(exc.value)
    assert "iv" in message and "estimate_effect" in message
    assert "engine='dml'" not in message  # no pointer at a path that does not exist
    # estimate_effect takes the instrument route on the same graph
    iv = estimate_effect(graph, data, "X", "Y")
    lo, hi = iv.interval(0.95)
    assert lo < 0.7 < hi


def test_general_id_strategy_keeps_raising_with_pointer() -> None:
    """The general-ID functional keeps raising (verify-then-widen): a Verma world that
    identifies only through do-calculus ID points the caller at estimate_effect.

    A->B->C->D with A<->C (latent) and B<->D (latent): the effect of A on D is
    identified by the general ID algorithm (strategy 'id'), not adjustment/front-door.
    Its latent-fit do-contrast was not validated to tolerance in this phase, so ate()
    raises with the pointer rather than emit a possibly-wrong number. Pre-MCMC.
    """
    from ergodic import identify_effect

    graph = dag(
        ["A -> B", "B -> C", "C -> D", "U1 -> A", "U1 -> C", "U2 -> B", "U2 -> D"],
        latent=["U1", "U2"],
    )
    assert identify_effect(graph, "A", "D").strategy == "id"
    rng = np.random.default_rng(0)
    n = 30
    data = tabular({k: rng.normal(size=n) for k in "ABCD"})
    scm = SCM(graph, data)
    scm._fits = dict.fromkeys(scm.mechanisms)  # type: ignore[assignment]
    with pytest.raises(NotImplementedError) as exc:
        scm.ate("A", "D")
    assert "id" in str(exc.value)
    assert "estimate_effect" in str(exc.value)


# == ITE: per-unit individual treatment effects ===============================


def test_ite_levels_recovers_constant_effect_on_linear_fork(
    fork: tuple[SCM, dict[str, np.ndarray]],
) -> None:
    """Linear-Gaussian: every unit's ITE equals the coefficient; the mean is the ATE.

    The continuous ITE is constant across units under a linear-additive mechanism (the
    abducted noise cancels), so the per-unit array is ~1.5 everywhere and its mean
    matches ate() to tolerance. This is the honest 'no heterogeneity here' baseline.
    """
    scm, _ = fork
    units = scm.data.take(list(range(300)))
    ite = scm.ite(units, "X", "Y")  # default levels (0, 1)
    assert ite.shape == (300,)
    assert float(np.mean(ite)) == pytest.approx(1.5, abs=0.05)
    assert float(np.std(ite)) < 0.01  # constant across units (linear mechanism)
    ate = scm.ate("X", "Y")
    assert isinstance(ate, SCMResult)
    assert float(np.mean(ite)) == pytest.approx(ate.mean(), abs=0.02)


def test_ite_shift_matches_single_unit_counterfactual(
    fork: tuple[SCM, dict[str, np.ndarray]],
) -> None:
    """THE per-unit-do correctness anchor: the cohort shift equals the scalar path.

    A shift= contrast hands the engine a per-unit do-vector x_i + delta. For any unit
    that must equal the single-unit scalar counterfactual do(X = x_i + delta) minus its
    factual outcome, to machine precision. And for a linear mechanism the shift is the
    constant coef*delta, so shift=2 gives ~3.0 everywhere.
    """
    scm, cols = fork
    delta = 2.0
    units = scm.data.take(list(range(300)))
    ite = scm.ite(units, "X", "Y", shift=delta)
    assert float(np.mean(ite)) == pytest.approx(2.0 * 1.5, abs=0.1)  # coef*delta
    assert float(np.std(ite)) < 0.02  # constant (linear)
    # the per-unit do-vector path == the single-unit scalar counterfactual, exactly.
    x0, y0 = float(cols["X"][0]), float(cols["Y"][0])
    single = scm.counterfactual(scm.data.take([0]), {"X": x0 + delta}, "Y", use_draws=False)
    assert isinstance(single, SCMResult)
    assert float(ite[0]) == pytest.approx(single.mean() - y0, abs=1e-9)


def test_ite_shift_equals_levels_step_for_linear(
    fork: tuple[SCM, dict[str, np.ndarray]],
) -> None:
    """Linear: a shift of delta and a level step of width delta are the same contrast."""
    scm, _ = fork
    units = scm.data.take(list(range(200)))
    shift = scm.ite(units, "X", "Y", shift=1.0)
    levels = scm.ite(units, "X", "Y", levels=(0.0, 1.0))
    assert float(np.mean(shift)) == pytest.approx(float(np.mean(levels)), abs=0.02)


def test_ite_binary_outcome_is_heterogeneous() -> None:
    """A binary outcome's ITE is the per-unit counterfactual flip in {-1, 0, 1}.

    The logistic response varies with each unit's baseline, so the per-unit effect is
    genuinely heterogeneous (some units flip, most do not), unlike the constant
    continuous-linear ITE. The mean is the ATE on the outcome scale, and the treatment
    raises P(Y=1) here, so the flips are positive.
    """
    rng = np.random.default_rng(5)
    n = 1500
    z = rng.normal(0, 1, n)
    t = rng.binomial(1, 1.0 / (1.0 + np.exp(-z))).astype(int)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(0.4 + 1.6 * t - 1.0 * z)))).astype(int)
    data = tabular({"Z": z, "T": t, "Y": y}, types={"T": "categorical", "Y": "categorical"})
    scm = SCM(
        dag(["Z -> T", "Z -> Y", "T -> Y"]), data, draws=300, tune=300, chains=2, random_state=5
    ).fit()
    ite = scm.ite(data, "T", "Y")
    assert ite.shape == (n,)
    assert set(np.unique(ite).tolist()).issubset({-1.0, 0.0, 1.0})  # per-unit flips
    assert 0.05 < float((ite != 0.0).mean()) < 0.95  # genuine heterogeneity
    assert float(np.mean(ite)) > 0.0  # the treatment raises P(Y=1) on average


def test_ite_rejects_shift_on_binary_treatment(
    binary_fork: tuple[SCM, dict[str, np.ndarray]],
) -> None:
    """shift= is for a continuous treatment; a binary T has no additive shift."""
    scm, _ = binary_fork
    with pytest.raises(ValueError, match="continuous treatment"):
        scm.ite(scm.data.take([0, 1, 2]), "T", "Y", shift=1.0)


def test_ite_rejects_levels_and_shift_together(
    fork: tuple[SCM, dict[str, np.ndarray]],
) -> None:
    """levels= and shift= are two framings of one contrast; passing both is an error."""
    scm, _ = fork
    with pytest.raises(ValueError, match="either"):
        scm.ite(scm.data.take([0]), "X", "Y", levels=(0.0, 1.0), shift=1.0)


def test_ite_rejects_multicategory_outcome() -> None:
    """A 3+ category outcome has no scalar individual effect; ite points at counterfactual."""
    scm, data = _categorical_world()  # Z -> C, C has 4 categories
    with pytest.raises(ValueError, match="categories"):
        scm.ite(data, "Z", "C")
