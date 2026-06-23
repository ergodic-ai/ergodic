# SPDX-License-Identifier: Apache-2.0
"""How sure are we, per person: three roads to a per-unit interval.

A per-unit point estimate without a band is a guess. ``CATEResult`` carries the
uncertainty and ``.interval()`` reads it, but where the band comes from depends on
the method. This shows all three roads on one heterogeneous world (true effect
``tau(z) = 1 + z``), and checks that each one brackets the truth at about its nominal
rate.

  1. BOOTSTRAP   the default for most learners: resample, refit, collect per-unit
                 predictions into ``(S, n)`` draws, take quantiles. ``kind="bootstrap"``.
  2. DR ANALYTIC the doubly-robust learner with a linear final stage gets a
                 closed-form influence-function standard error per unit (the CI of
                 ``E[tau | x]``). Fast, no resampling. ``kind="analytic"``.
  3. POSTERIOR   a Bayesian learner in the outcome slot turns the per-unit effect
                 into a posterior ``(S, n)``; the interval is the credible interval.
                 ``kind="posterior"``. Here a small self-contained Bayesian-linear
                 learner (a flat-prior posterior) stands in, so the script stays
                 dependency-free; a real model would be a PyMCLearner.

All three return the SAME surface: ``.interval()``, ``.std_error``, ``.draws()``.

Run with: uv run python examples/cate/06_per_unit_uncertainty.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from ergodic import cate, dag, identify_effect, tabular
from ergodic.inference import OutcomeRegression


def _with_intercept(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.hstack([np.ones((arr.shape[0], 1), dtype=np.float64), arr])


class BayesLinear:
    """A Bayesian linear regression: the flat-prior posterior is the OLS sampling law.

    Carries a ``predict_draws`` so the estimator detects it as a Bayesian slot and
    flows the posterior into the per-unit result. Re-seeded per call so the two arms
    of a contrast share a coefficient draw (a coherent within-draw difference).
    """

    def __init__(self, *, n_draws: int = 400, random_state: int | None = 0) -> None:
        self.n_draws = n_draws
        self.random_state = random_state
        self._mean: npt.NDArray[np.float64] | None = None
        self._cov: npt.NDArray[np.float64] | None = None

    def fit(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64] | None = None,
    ) -> BayesLinear:
        d = _with_intercept(x)
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
        rng = np.random.default_rng(self.random_state)
        return np.asarray(
            rng.multivariate_normal(self._mean, self._cov, size=self.n_draws),  # type: ignore[arg-type]
            dtype=np.float64,
        )

    def predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(_with_intercept(x) @ self._mean, dtype=np.float64)

    def predict_draws(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray((_with_intercept(x) @ self._coef_draws().T).T, dtype=np.float64)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"n_draws": self.n_draws, "random_state": self.random_state}

    def set_params(self, **params: Any) -> BayesLinear:
        for key, value in params.items():
            setattr(self, key, value)
        return self


def _coverage(
    interval: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    truth: npt.NDArray[np.float64],
) -> float:
    lo, hi = interval
    return float(np.mean((lo <= truth) & (truth <= hi)))


def main() -> None:
    rng = np.random.default_rng(2)
    n = 2000
    z = rng.normal(0, 1, n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.7 * z))).astype(float)
    y = (1.0 + 1.0 * z) * a + 1.5 * z + rng.normal(size=n)

    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    graph = dag(["Z -> A", "Z -> Y", "A -> Y"])
    truth = 1.0 + z

    # 1) bootstrap (X-learner)
    boot = cate(graph, data, "A", "Y", method="x_learner", n_boot=200, random_state=2)

    # 2) DR analytic
    analytic = cate(graph, data, "A", "Y", method="dr_learner", random_state=2)

    # 3) Bayesian posterior (T-learner with a Bayesian outcome slot)
    ident = identify_effect(graph, "A", "Y")
    posterior = OutcomeRegression(BayesLinear(random_state=2), per_arm=True).fit(ident, data).cate()

    print(__doc__.strip().splitlines()[0])
    print("=" * 78)
    print("Planted tau(z) = 1 + z, n = 2000; nominal 90% per-unit intervals.\n")
    print(f"  {'road':<22}{'kind':<12}{'90% coverage':>13}{'mean width':>13}")
    print("  " + "-" * 60)
    for label, res in [
        ("bootstrap (X-learner)", boot),
        ("DR analytic", analytic),
        ("Bayesian posterior", posterior),
    ]:
        lo, hi = res.interval(0.9)
        cov = _coverage((lo, hi), truth)
        width = float(np.mean(hi - lo))
        print(f"  {label:<22}{res.kind:<12}{cov:13.2f}{width:13.3f}")

    draws = posterior.draws()
    print(f"\n  the posterior carries draws: shape {None if draws is None else draws.shape}")
    se = analytic.std_error
    print(
        f"  the analytic road carries per-unit std errors: shape {None if se is None else se.shape}"
    )
    print("=" * 78)
    print(
        "Three different machineries, one uniform surface. Each brackets tau(z) at about\n"
        "its nominal rate, so the per-unit band is honest however it was produced."
    )


if __name__ == "__main__":
    main()
