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
