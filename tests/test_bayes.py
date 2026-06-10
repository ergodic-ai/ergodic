# SPDX-License-Identifier: Apache-2.0
"""Tests for the Bayesian learners and the posterior path.

The propagation (the part the library owns) is exercised with two small numpy
Bayesian learners, so these run with no extra dependency. The real PyMC backend is
checked in one integration test, gated behind an import of pymc.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from ergodic import EffectTarget, dag, identify_effect, tabular
from ergodic.data import TabularData
from ergodic.graph import MixedGraph
from ergodic.inference import AIPW, IPW, OLS, OutcomeRegression, PyMCLearner
from ergodic.inference._learner import _is_bayesian


def _hetero(
    n: int = 4000, seed: int = 0
) -> tuple[TabularData, MixedGraph, npt.NDArray[np.float64]]:
    """Heterogeneous effect tau(z) = 1 + z, so the ATE is 1.0 and the CATE tracks 1 + z."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.7 * z))).astype(float)
    y = (1.0 + z) * a + 1.5 * z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    return data, dag(["Z -> A", "Z -> Y", "A -> Y"]), z


# -- numpy Bayesian learners, used as test doubles for the posterior path ----


def _with_intercept(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    x = np.asarray(X, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    ones = np.ones((x.shape[0], 1), dtype=np.float64)
    return np.asarray(np.hstack([ones, x]), dtype=np.float64)


class _BayesLinear:
    """Bayesian linear regression under a flat prior: the OLS sampling law as posterior.

    Re-seeded per call, so two predictions in one contrast share their coefficient
    draw and the contrast is coherent within a draw.
    """

    def __init__(self, *, n_draws: int = 300, random_state: int | None = 0) -> None:
        self.n_draws = n_draws
        self.random_state = random_state
        self._mean: npt.NDArray[np.float64] | None = None
        self._cov: npt.NDArray[np.float64] | None = None

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64] | None = None,
    ) -> _BayesLinear:
        d = _with_intercept(X)
        t = np.asarray(y, dtype=np.float64).reshape(-1)
        gram = d.T @ d + 1e-6 * np.eye(d.shape[1], dtype=np.float64)
        gram_inv = np.linalg.inv(gram)
        beta = gram_inv @ d.T @ t
        resid = t - d @ beta
        dof = max(d.shape[0] - d.shape[1], 1)
        sigma2 = float(resid @ resid) / dof
        self._mean = np.asarray(beta, dtype=np.float64)
        self._cov = np.asarray(sigma2 * gram_inv, dtype=np.float64)
        return self

    def _coef_draws(self) -> npt.NDArray[np.float64]:
        if self._mean is None or self._cov is None:
            raise RuntimeError("fit first")
        rng = np.random.default_rng(self.random_state)
        return np.asarray(
            rng.multivariate_normal(self._mean, self._cov, size=self.n_draws), dtype=np.float64
        )

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self._mean is None:
            raise RuntimeError("fit first")
        return np.asarray(_with_intercept(X) @ self._mean, dtype=np.float64)

    def predict_draws(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        betas = self._coef_draws()
        out = np.asarray(_with_intercept(X) @ betas.T, dtype=np.float64)
        return np.asarray(out.T, dtype=np.float64)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"n_draws": self.n_draws, "random_state": self.random_state}

    def set_params(self, **params: Any) -> _BayesLinear:
        for key, value in params.items():
            setattr(self, key, value)
        return self


class _BayesLogit:
    """Laplace Bayesian logistic regression: a Gaussian posterior at the penalized mode."""

    def __init__(
        self, *, n_draws: int = 300, l2: float = 1e-6, random_state: int | None = 0
    ) -> None:
        self.n_draws = n_draws
        self.l2 = l2
        self.random_state = random_state
        self._mean: npt.NDArray[np.float64] | None = None
        self._cov: npt.NDArray[np.float64] | None = None

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64] | None = None,
    ) -> _BayesLogit:
        d = _with_intercept(X)
        lab = np.asarray(y, dtype=np.float64).reshape(-1)
        k = d.shape[1]
        pen = self.l2 * np.eye(k, dtype=np.float64)
        beta = np.zeros(k, dtype=np.float64)
        hess = pen
        for _ in range(100):
            p = 1.0 / (1.0 + np.exp(-(d @ beta)))
            grad = d.T @ (lab - p) - pen @ beta
            hess = d.T @ (d * (p * (1.0 - p))[:, None]) + pen
            step = np.linalg.solve(hess, grad)
            beta = beta + step
            if float(np.max(np.abs(step))) < 1e-8:
                break
        self._mean = np.asarray(beta, dtype=np.float64)
        self._cov = np.asarray(np.linalg.inv(hess), dtype=np.float64)
        return self

    def _coef_draws(self) -> npt.NDArray[np.float64]:
        if self._mean is None or self._cov is None:
            raise RuntimeError("fit first")
        rng = np.random.default_rng(self.random_state)
        return np.asarray(
            rng.multivariate_normal(self._mean, self._cov, size=self.n_draws), dtype=np.float64
        )

    def predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self._mean is None:
            raise RuntimeError("fit first")
        p = 1.0 / (1.0 + np.exp(-(_with_intercept(X) @ self._mean)))
        return np.asarray(np.column_stack([1.0 - p, p]), dtype=np.float64)

    def predict_proba_draws(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        betas = self._coef_draws()
        eta = _with_intercept(X) @ betas.T
        return np.asarray((1.0 / (1.0 + np.exp(-eta))).T, dtype=np.float64)

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(self.predict_proba(X)[:, 1] >= 0.5, dtype=np.float64)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"n_draws": self.n_draws, "l2": self.l2, "random_state": self.random_state}

    def set_params(self, **params: Any) -> _BayesLogit:
        for key, value in params.items():
            setattr(self, key, value)
        return self


# -- detection ---------------------------------------------------------------


def test_is_bayesian_detection() -> None:
    assert _is_bayesian(_BayesLinear())
    assert _is_bayesian(_BayesLogit())
    assert not _is_bayesian(OLS())


# -- the g-formula posterior -------------------------------------------------


def test_g_formula_posterior_recovers_ate() -> None:
    data, graph, _ = _hetero()
    ident = identify_effect(graph, "A", "Y")
    est = OutcomeRegression(_BayesLinear()).fit(ident, data).effect()
    assert est.kind == "posterior"
    assert est.draws is not None
    assert est.draws.shape[0] >= 100
    assert abs(est.value - 1.0) < 0.2
    lo, hi = est.interval()
    assert lo < 1.0 < hi


def test_g_formula_att_posterior() -> None:
    data, graph, _ = _hetero()
    ident = identify_effect(graph, "A", "Y")
    est = OutcomeRegression(_BayesLinear()).fit(ident, data).effect(target="att")
    assert est.kind == "posterior"
    assert est.target is EffectTarget.ATT


# -- per-unit CATE credible intervals ----------------------------------------


def test_t_learner_cate_posterior_intervals() -> None:
    data, graph, z = _hetero()
    ident = identify_effect(graph, "A", "Y")
    model = OutcomeRegression(_BayesLinear(), per_arm=True).fit(ident, data)
    draws = model.cate_draws()
    assert draws.shape[1] == data.n_rows
    lo, hi = model.cate_interval()
    assert lo.shape[0] == data.n_rows
    assert bool(np.all(lo <= hi))
    assert float(np.corrcoef(draws.mean(axis=0), 1.0 + z)[0, 1]) > 0.5


def test_cate_draws_out_of_sample() -> None:
    train, graph, _ = _hetero(seed=0)
    fresh, _, z_fresh = _hetero(seed=5)
    ident = identify_effect(graph, "A", "Y")
    model = OutcomeRegression(_BayesLinear(), per_arm=True).fit(ident, train)
    draws = model.cate_draws(fresh)
    assert draws.shape[1] == fresh.n_rows
    assert float(np.corrcoef(draws.mean(axis=0), 1.0 + z_fresh)[0, 1]) > 0.5


# -- the doubly-robust posterior ---------------------------------------------


def test_dr_posterior_recovers_ate() -> None:
    data, graph, _ = _hetero()
    ident = identify_effect(graph, "A", "Y")
    est = AIPW({"outcome": _BayesLinear(), "propensity": _BayesLogit()}).fit(ident, data).effect()
    assert est.kind == "posterior"
    assert est.draws is not None
    assert abs(est.value - 1.0) < 0.25
    assert est.std_error is not None
    assert est.std_error > 0.0


def test_dr_posterior_without_bayesian_bootstrap() -> None:
    data, graph, _ = _hetero()
    ident = identify_effect(graph, "A", "Y")
    model = AIPW(
        {"outcome": _BayesLinear(), "propensity": _BayesLogit()}, bayesian_bootstrap=False
    ).fit(ident, data)
    est = model.effect()
    assert est.kind == "posterior"
    assert abs(est.value - 1.0) < 0.25


def test_dr_att_posterior() -> None:
    data, graph, _ = _hetero()
    ident = identify_effect(graph, "A", "Y")
    est = (
        AIPW({"outcome": _BayesLinear(), "propensity": _BayesLogit()})
        .fit(ident, data)
        .effect(target="att")
    )
    assert est.kind == "posterior"
    assert est.target is EffectTarget.ATT


# -- graceful degrade, never an MCMC bootstrap -------------------------------


def test_bayesian_learner_degrades_without_a_posterior_path() -> None:
    data, graph, _ = _hetero()
    ident = identify_effect(graph, "A", "Y")
    # IPW has no _posterior; a Bayesian propensity must fall back to a single-fit
    # point, not a bootstrap that would refit the nuisance many times.
    est = IPW({"propensity": _BayesLogit()}).fit(ident, data).effect()
    assert est.kind in {"point", "analytic"}
    assert abs(est.value - 1.0) < 0.4


def test_mixed_slots_do_not_take_the_posterior_path() -> None:
    data, graph, _ = _hetero()
    ident = identify_effect(graph, "A", "Y")
    # Bayesian outcome, native propensity: AIPW cannot form a joint posterior, so it
    # returns its analytic ATE (one fit), not a posterior.
    est = AIPW({"outcome": _BayesLinear(), "propensity": "logistic"}).fit(ident, data).effect()
    assert est.kind == "analytic"
    assert abs(est.value - 1.0) < 0.25


# -- the real PyMC backend (skipped without pymc or a C compiler) ------------


def test_pymc_learner_recovers_g_formula_ate() -> None:
    pytest.importorskip("pymc")
    import pymc as pm

    class _LinearOutcome(PyMCLearner):
        def build(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> Any:
            n_features = X.shape[1]
            with pm.Model() as model:
                x_data = pm.Data("X", X)
                intercept = pm.Normal("intercept", 0.0, 5.0)
                coef = pm.Normal("coef", 0.0, 5.0, shape=n_features)
                pm.Deterministic("mu", intercept + x_data @ coef)
                sigma = pm.HalfNormal("sigma", 5.0)
                pm.Normal("y", mu=model["mu"], sigma=sigma, observed=y)
            return model

    data, graph, _ = _hetero(n=800, seed=0)
    ident = identify_effect(graph, "A", "Y")
    learner = _LinearOutcome(draws=400, tune=400, chains=2, random_state=0)
    try:
        est = OutcomeRegression(learner, per_arm=True).fit(ident, data).effect()
    except Exception as exc:  # pytensor needs a C compiler and Python headers to sample
        message = str(exc).lower()
        if "compil" in message or "python.h" in message:
            pytest.skip(f"pytensor cannot compile here: {exc}")
        raise
    assert est.kind == "posterior"
    assert abs(est.value - 1.0) < 0.25
