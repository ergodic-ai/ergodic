# SPDX-License-Identifier: Apache-2.0
"""Tests for the D1 conditional-independence tests and the CITest base machinery."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest

from ergodic import tabular
from ergodic.discovery._ci import CIResult, CITest
from ergodic.discovery._stats import _chi2_sf, _norm_cdf, _norm_two_sided
from ergodic.discovery.algorithms._pc import PC
from ergodic.discovery.ci._discrete import Discrete
from ergodic.discovery.ci._gcm import GCM
from ergodic.discovery.ci._linear import Linear
from ergodic.discovery.ci._mixed import Mixed
from ergodic.inference import OLS

if TYPE_CHECKING:
    import numpy.typing as npt

    from ergodic.data import TabularData

# -- simulators ---------------------------------------------------------------


def _gaussian_chain(n: int = 2000, seed: int = 0) -> TabularData:
    """X -> M -> Y, all Gaussian: X and Y dependent, independent given M."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    m = 0.8 * x + rng.normal(size=n)
    y = 0.8 * m + rng.normal(size=n)
    return tabular({"X": x, "M": m, "Y": y})


def _gaussian_collider(n: int = 2000, seed: int = 2) -> TabularData:
    """X -> C <- Y: X and Y independent, dependent given the collider C."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    c = x + y + 0.5 * rng.normal(size=n)
    return tabular({"X": x, "C": c, "Y": y})


def _discrete_chain(n: int = 2000, seed: int = 2) -> TabularData:
    """Three-level X -> M -> Y through noisy copy channels."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 3, size=n)
    m = np.where(rng.uniform(size=n) < 0.7, x, rng.integers(0, 3, size=n))
    y = np.where(rng.uniform(size=n) < 0.7, m, rng.integers(0, 3, size=n))
    return tabular(
        {"X": x, "M": m, "Y": y},
        types={"X": "categorical", "M": "categorical", "Y": "categorical"},
    )


def _discrete_collider(n: int = 2000, seed: int = 4) -> TabularData:
    """Three-level X -> C <- Y with C a noisy sum mod 3."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 3, size=n)
    y = rng.integers(0, 3, size=n)
    c = np.where(rng.uniform(size=n) < 0.8, (x + y) % 3, rng.integers(0, 3, size=n))
    return tabular(
        {"X": x, "C": c, "Y": y},
        types={"X": "categorical", "C": "categorical", "Y": "categorical"},
    )


def _mixed_chain(n: int = 2000, seed: int = 4) -> TabularData:
    """Continuous X -> three-level categorical M -> continuous Y."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    m = np.digitize(x + 0.5 * rng.normal(size=n), [-0.6, 0.6])
    y = 1.0 * m + rng.normal(size=n)
    return tabular({"X": x, "M": m, "Y": y}, types={"M": "categorical"})


# -- Linear: partial correlation with the Fisher z transform ------------------


def test_linear_chain_separates() -> None:
    test = Linear(_gaussian_chain())
    marginal = test.test("X", "Y")
    assert marginal.p_value < 1e-6
    assert marginal.dof is None
    assert test.test("X", "Y", ["M"]).p_value > 0.05


def test_linear_collider_connects() -> None:
    test = Linear(_gaussian_collider())
    assert test.test("X", "Y").p_value > 0.05
    assert test.test("X", "Y", ["C"]).p_value < 1e-6


def test_linear_null_p_values_are_calibrated() -> None:
    p_values: list[float] = []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        data = tabular({"X": rng.normal(size=200), "Y": rng.normal(size=200)})
        p_values.append(Linear(data).test("X", "Y").p_value)
    assert abs(float(np.mean(p_values)) - 0.5) < 0.1


def test_linear_rejects_categorical_endpoints() -> None:
    test = Linear(_mixed_chain())
    with pytest.raises(TypeError, match="Mixed"):
        test.test("M", "Y")


def test_linear_needs_enough_rows_for_the_conditioning_size() -> None:
    data = tabular(
        {"X": [0.1, 0.9, 0.4, 0.7], "M": [0.3, 0.2, 0.8, 0.5], "Y": [0.6, 0.1, 0.9, 0.2]}
    )
    test = Linear(data)
    with pytest.raises(ValueError, match="too few rows"):
        test.test("X", "Y", ["M"])  # n - dz - 3 = 4 - 1 - 3 = 0


# -- Discrete: the G squared test ----------------------------------------------


def test_discrete_chain_separates() -> None:
    test = Discrete(_discrete_chain())
    assert test.test("X", "Y").p_value < 1e-6
    assert test.test("X", "Y", ["M"]).p_value > 0.05


def test_discrete_collider_connects() -> None:
    test = Discrete(_discrete_collider())
    assert test.test("X", "Y").p_value > 0.05
    assert test.test("X", "Y", ["C"]).p_value < 1e-6


def test_discrete_dof_counts_nonempty_strata() -> None:
    x = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64)
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.int64)
    z = (np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),)
    # two nonempty z strata, binary x and y: dof = (2 - 1)(2 - 1) * 2 = 2
    _, _, dof = Discrete().statistic(x, y, z, (2, 2, (2,)))
    assert dof == 2.0
    # a declared but never-observed z level adds no degrees of freedom
    _, _, dof = Discrete().statistic(x, y, z, (2, 2, (3,)))
    assert dof == 2.0
    # unconditional: one stratum, dof = (3 - 1)(2 - 1) = 2 for a 3-level x
    _, _, dof = Discrete().statistic(np.array([0, 1, 2, 0, 1, 2]), y[:6], (), (3, 2, ()))
    assert dof == 2.0


def test_discrete_exact_independence_gives_zero_statistic() -> None:
    x = np.array([0, 0, 1, 1], dtype=np.int64)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    stat, p, dof = Discrete().statistic(x, y, (), (2, 2, ()))
    assert stat == pytest.approx(0.0, abs=1e-12)
    assert p == pytest.approx(1.0)
    assert dof == 1.0


def test_discrete_single_level_variable_is_vacuous() -> None:
    x = np.zeros(6, dtype=np.int64)
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    assert Discrete().statistic(x, y, (), (1, 2, ())) == (0.0, 1.0, 0.0)


# -- Mixed: the degenerate-Gaussian test ---------------------------------------


def test_mixed_matches_linear_on_continuous_data() -> None:
    data = _gaussian_chain()
    mixed_test = Mixed(data)
    linear_test = Linear(data)
    queries: list[tuple[str, str, tuple[str, ...]]] = [
        ("X", "Y", ()),
        ("X", "Y", ("M",)),
        ("X", "M", ()),
        ("M", "Y", ("X",)),
    ]
    for x, y, given in queries:
        p_mixed = mixed_test.test(x, y, given).p_value
        p_linear = linear_test.test(x, y, given).p_value
        assert abs(p_mixed - p_linear) < 0.05


def test_mixed_chain_separates_mixed_data() -> None:
    test = Mixed(_mixed_chain())
    assert test.test("X", "Y").p_value < 1e-6
    assert test.test("X", "Y", ["M"]).p_value > 0.05


def test_mixed_dof_is_the_block_product() -> None:
    test = Mixed(_mixed_chain())
    # M is 3-level, one-hot drop-first: dx = 2 against the scalar Y, dof = 2 * 1
    assert test.test("M", "Y").dof == 2.0
    assert test.test("X", "M", ["Y"]).dof == 2.0
    assert test.test("X", "Y", ["M"]).dof == 1.0


def test_mixed_single_level_block_is_vacuous() -> None:
    rng = np.random.default_rng(5)
    data = tabular({"K": ["a"] * 50, "Y": rng.normal(size=50)}, types={"K": "categorical"})
    assert Mixed(data).test("K", "Y") == CIResult(statistic=0.0, p_value=1.0, dof=0.0)


def test_mixed_needs_enough_rows_for_the_block_sizes() -> None:
    data = tabular(
        {"M": ["a", "b", "c", "a"], "Y": [0.1, 0.5, 0.9, 0.3]},
        types={"M": "categorical"},
    )
    test = Mixed(data)
    with pytest.raises(ValueError, match="too few rows"):
        test.test("M", "Y")  # n - dz - (dx + dy) - 1 = 4 - 0 - 3 - 1 = 0


# -- the base: binding, caching, validation ------------------------------------


def test_bind_returns_a_new_bound_instance() -> None:
    spec = Linear()
    assert not spec.is_bound
    bound = spec.bind(_gaussian_chain())
    assert bound is not spec
    assert bound.is_bound
    assert not spec.is_bound  # the original stays an unbound config


def test_unbound_query_raises() -> None:
    with pytest.raises(RuntimeError, match="not bound"):
        Linear().test("X", "Y")


def test_cache_is_symmetric_in_x_and_y() -> None:
    test = Linear(_gaussian_chain())
    first = test.test("X", "Y", ["M"])
    second = test.test("Y", "X", ["M"])
    assert second == first
    diagnostics = test.diagnostics()
    assert diagnostics["n_ci_tests"] == 1.0
    assert diagnostics["cache_hits"] == 1.0


def test_alpha_reads_one_cached_entry() -> None:
    test = Linear(_gaussian_chain())
    test.independent("X", "Y", ["M"], alpha=0.01)
    test.independent("X", "Y", ["M"], alpha=0.2)
    diagnostics = test.diagnostics()
    assert diagnostics["n_ci_tests"] == 1.0
    assert diagnostics["cache_hits"] == 1.0


def test_validation_rejects_bad_queries() -> None:
    test = Linear(_gaussian_chain())
    with pytest.raises(ValueError, match="unknown variable"):
        test.test("X", "W")
    with pytest.raises(ValueError, match="must differ"):
        test.test("X", "X")
    with pytest.raises(ValueError, match="conditioning set"):
        test.test("X", "Y", ["X"])


def test_discrete_refuses_continuous_columns_at_bind() -> None:
    with pytest.raises(TypeError, match="discrete columns"):
        Discrete(_gaussian_chain())
    with pytest.raises(TypeError, match="discrete columns"):
        Discrete().bind(_gaussian_chain())


class _Coin(CITest):
    """A tiny subclass: the base machinery serves any statistic."""

    name: ClassVar[str] = "coin"
    needs: ClassVar[str] = "numeric"

    def statistic(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> tuple[float, float, float | None]:
        return (float(x.shape[0]), 0.5, None)


def test_custom_subclass_runs_through_the_base() -> None:
    test = _Coin(_gaussian_chain(n=100))
    assert test.test("X", "Y", ["M"]) == CIResult(statistic=100.0, p_value=0.5, dof=None)
    assert test.independent("X", "Y", alpha=0.05)
    assert not test.independent("X", "Y", alpha=0.7)


# -- the scipy-free statistics helpers ------------------------------------------


def test_chi2_survival_matches_the_classic_quantiles() -> None:
    assert _chi2_sf(3.841, 1) == pytest.approx(0.05, abs=1e-3)
    assert _chi2_sf(5.991, 2) == pytest.approx(0.05, abs=1e-3)
    assert _chi2_sf(18.31, 10) == pytest.approx(0.05, abs=1e-3)


def test_the_two_sided_normal_tail_survives_far_out() -> None:
    assert _norm_two_sided(1.959964) == pytest.approx(0.05, abs=1e-6)
    assert _norm_two_sided(-1.959964) == _norm_two_sided(1.959964)
    # 1 - Phi(10) is zero in double precision; the tail itself is 1.52e-23.
    assert 2.0 * (1.0 - _norm_cdf(10.0)) == 0.0
    assert _norm_two_sided(10.0) == pytest.approx(1.523970604832e-23, rel=1e-9)


def test_norm_cdf_matches_the_classic_quantile() -> None:
    assert _norm_cdf(1.959964) == pytest.approx(0.975, abs=1e-6)


# -- GCM: the generalized covariance measure -------------------------------------


def _quadratic_confounder(n: int = 2000, seed: int = 0) -> TabularData:
    """X <- Z -> Y through Z**2 on a positive range: X indep Y given Z.

    A linear regression of either endpoint on Z leaves the quadratic part in
    the residual, those leftovers correlate, and the partial correlation
    false-positives; a flexible regression takes the curve out.
    """
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.5, 2.0, size=n)
    x = z**2 + 0.5 * rng.normal(size=n)
    y = z**2 + 0.5 * rng.normal(size=n)
    return tabular({"X": x, "Y": y, "Z": z})


def test_gcm_calibrates_under_the_null_with_a_linear_learner() -> None:
    p_values: list[float] = []
    for seed in range(100):
        rng = np.random.default_rng(seed)
        z = rng.normal(size=300)
        x = z + rng.normal(size=300)
        y = z + rng.normal(size=300)
        data = tabular({"X": x, "Y": y, "Z": z})
        p_values.append(GCM(data, learner=OLS()).test("X", "Y", ["Z"]).p_value)
    assert 0.35 < float(np.mean(p_values)) < 0.65
    assert float(np.mean(np.asarray(p_values) < 0.05)) < 0.15


def test_gcm_clears_the_quadratic_confounder_that_fools_the_linear_test() -> None:
    # Both endpoints read Z**2 over a positive Z, so the linear residuals
    # share the leftover curvature and the partial correlation rejects the
    # true null; the default KNN regression removes the curve. Seed 0 leaves
    # both margins wide (linear p ~ 1e-4, GCM p ~ 0.9).
    data = _quadratic_confounder()
    assert Linear(data).test("X", "Y", ["Z"]).p_value < 0.05
    assert GCM(data).test("X", "Y", ["Z"]).p_value > 0.05


def test_gcm_detects_dependence_a_nonlinear_regression_reveals() -> None:
    rng = np.random.default_rng(3)
    z = rng.normal(size=1500)
    x = z + rng.normal(size=1500)
    y = 0.5 * x + z**2 + rng.normal(size=1500)
    data = tabular({"X": x, "Y": y, "Z": z})
    assert GCM(data).test("X", "Y", ["Z"]).p_value < 0.01


def test_gcm_requires_scalar_endpoints() -> None:
    test = GCM(_mixed_chain())
    with pytest.raises(TypeError, match="scalar"):
        test.test("M", "Y")


def test_gcm_accepts_a_sklearn_regressor_in_the_slot() -> None:
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestRegressor

    data = _quadratic_confounder()
    test = GCM(data, learner=RandomForestRegressor(n_estimators=30, random_state=0))
    result = test.test("X", "Y", ["Z"])
    assert isinstance(result, CIResult)
    assert 0.0 <= result.p_value <= 1.0
    assert result.p_value > 0.05


def test_pc_over_gcm_drops_the_spurious_edge_the_linear_test_keeps() -> None:
    data = _quadratic_confounder()
    linear_pairs = {frozenset((e.u, e.v)) for e in PC(Linear(data)).fit().graph.edges}
    gcm_pairs = {frozenset((e.u, e.v)) for e in PC(GCM(data)).fit().graph.edges}
    assert frozenset(("X", "Y")) in linear_pairs
    assert gcm_pairs == {frozenset(("X", "Z")), frozenset(("Z", "Y"))}


def test_gcm_with_no_conditioning_is_a_covariance_test() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=500)
    dependent = tabular({"X": x, "Y": x + 0.5 * rng.normal(size=500)})
    assert GCM(dependent).test("X", "Y").p_value < 0.01
    rng = np.random.default_rng(8)
    independent = tabular({"X": rng.normal(size=500), "Y": rng.normal(size=500)})
    assert GCM(independent).test("X", "Y").p_value > 0.05
