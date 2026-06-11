# SPDX-License-Identifier: Apache-2.0
"""Tests for the graphical-lasso screen: solver exactness, EBIC selection, knowledge.

The load-bearer cross-checks the block coordinate descent against
``sklearn.covariance.graphical_lasso`` entry by entry at fixed penalties, on
the same correlation input and the same convention (diagonal unpenalized).
The EBIC tests pin worlds by seed and demand the measured recovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from ergodic import DomainKnowledge, tabular
from ergodic.data import timeseries
from ergodic.discovery.algorithms._glasso import Glasso, _penalized_precision, _support
from ergodic.graph import GraphKind

if TYPE_CHECKING:
    import numpy.typing as npt

    from ergodic.data import TabularData
    from ergodic.graph import MixedGraph


def _pairs(graph: MixedGraph) -> set[frozenset[str]]:
    return {frozenset((e.u, e.v)) for e in graph.edges}


def _names(p: int) -> list[str]:
    return [f"X{i}" for i in range(p)]


def _sparse_precision(
    rng: np.random.Generator, p: int = 8, density: float = 0.25
) -> npt.NDArray[np.float64]:
    """A random sparse SPD precision: off-diagonals in 0.2..0.5, eigenvalues >= 0.3."""
    theta = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            if rng.uniform() < density:
                theta[i, j] = theta[j, i] = rng.uniform(0.2, 0.5) * rng.choice([-1.0, 1.0])
    smallest = float(np.linalg.eigvalsh(theta).min())
    if smallest < 0.3:
        theta += (0.3 - smallest) * np.eye(p)
    return theta


def _gaussian_data(theta: npt.NDArray[np.float64], rng: np.random.Generator, n: int) -> TabularData:
    sigma = np.linalg.inv(theta)
    x = rng.multivariate_normal(np.zeros(theta.shape[0]), sigma, size=n)
    return tabular(pd.DataFrame(dict(zip(_names(theta.shape[0]), x.T, strict=True))))


def _chain_data(seed: int = 11, p: int = 8, n: int = 4000) -> TabularData:
    """A tridiagonal SPD precision world: neighbour partial correlations of 0.45."""
    theta = np.eye(p)
    for i in range(p - 1):
        theta[i, i + 1] = theta[i + 1, i] = -0.45
    return _gaussian_data(theta, np.random.default_rng(seed), n)


_CHAIN_PAIRS = {frozenset((f"X{i}", f"X{i + 1}")) for i in range(7)}


def _correlation(data: TabularData) -> npt.NDArray[np.float64]:
    return np.asarray(np.corrcoef(data.matrix(list(data.variables)), rowvar=False))


def _lam_max(s: npt.NDArray[np.float64]) -> float:
    return float(np.max(np.abs(s - np.diag(np.diag(s)))))


def _assert_spd_and_symmetric(theta: npt.NDArray[np.float64]) -> None:
    assert np.array_equal(theta, theta.T)
    assert float(np.linalg.eigvalsh(theta).min()) > 0.0


# -- the sklearn cross-check (the load-bearer) -------------------------------------


def test_precision_matches_sklearn_graphical_lasso_entry_by_entry() -> None:
    # Same correlation input, same convention (diagonal unpenalized, shrunk
    # start), fixed penalties across the path: the precisions agree to well
    # under the advertised rtol 1e-3 / atol 1e-4, and the edge sets at the
    # 1e-8 threshold match exactly. Observed worst deviation ~6e-10.
    pytest.importorskip("sklearn")
    from sklearn.covariance import graphical_lasso

    iu = np.triu_indices(8, 1)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        data = _gaussian_data(_sparse_precision(rng), rng, n=500)
        s = _correlation(data)
        top = _lam_max(s)
        for frac in (0.9, 0.5, 0.2, 0.05, 0.01):
            lam = frac * top
            ours, _, _ = _penalized_precision(s, lam, tol=1e-10, max_iter=500)
            _, theirs = graphical_lasso(s, alpha=lam, tol=1e-12, enet_tol=1e-14, max_iter=2000)
            assert np.allclose(ours, theirs, rtol=1e-6, atol=1e-8)
            assert np.array_equal(_support(ours), np.abs(theirs[iu]) > 1e-8)
            _assert_spd_and_symmetric(ours)
        # the public route reports the same support sklearn finds
        lam = 0.2 * top
        result = Glasso(lam=lam, tol=1e-10, max_iter=500).fit(data)
        _, theirs = graphical_lasso(s, alpha=lam, tol=1e-12, enet_tol=1e-14, max_iter=2000)
        skl_pairs = {
            frozenset((f"X{i}", f"X{j}"))
            for i, j in zip(*iu, strict=True)
            if abs(theirs[i, j]) > 1e-8
        }
        assert _pairs(result.graph) == skl_pairs


# -- EBIC support recovery -----------------------------------------------------------


def test_the_ebic_pick_recovers_the_chain_adjacency_exactly() -> None:
    data = _chain_data(seed=11)
    result = Glasso().fit(data)
    assert _pairs(result.graph) == _CHAIN_PAIRS
    assert result.graph.kind is GraphKind.MIXED
    assert all(e.is_undirected for e in result.graph.edges)
    assert result.method == "glasso"
    assert result.test is None
    assert result.score is None
    assert result.alpha is None
    assert result.sepsets is None
    assert result.n == 4000
    d = result.diagnostics
    assert d["n_edges"] == 7.0
    assert d["n_lambdas_tried"] == 100.0
    assert "ebic" in d
    s = _correlation(data)
    assert 0.01 * _lam_max(s) <= d["lam"] <= _lam_max(s)
    theta, _, _ = _penalized_precision(s, d["lam"])
    _assert_spd_and_symmetric(theta)


def test_ebic_support_on_a_denser_world_stays_close_to_truth() -> None:
    # Pinned world (seed 8, density 0.35, n=2000): measured jaccard 0.909
    # with no missed true edges; assert the stable floor.
    rng = np.random.default_rng(8)
    theta_true = _sparse_precision(rng, density=0.35)
    data = _gaussian_data(theta_true, rng, n=2000)
    truth = {
        frozenset((f"X{i}", f"X{j}"))
        for i in range(8)
        for j in range(i + 1, 8)
        if abs(theta_true[i, j]) > 0.0
    }
    result = Glasso().fit(data)
    got = _pairs(result.graph)
    assert truth <= got  # every true edge survives
    assert len(got & truth) / len(got | truth) >= 0.8
    theta, _, _ = _penalized_precision(_correlation(data), result.diagnostics["lam"])
    _assert_spd_and_symmetric(theta)


# -- the penalty parameter ------------------------------------------------------------


def test_lambda_max_gives_the_empty_graph() -> None:
    data = _chain_data(n=1000)
    result = Glasso(lam=_lam_max(_correlation(data))).fit(data)
    assert len(result.graph.edges) == 0
    assert result.graph.nodes == frozenset(_names(8))


def test_an_explicit_lam_runs_single_shot() -> None:
    data = _chain_data(n=1000)
    result = Glasso(lam=0.1).fit(data)
    assert result.diagnostics["lam"] == 0.1
    assert result.diagnostics["n_lambdas_tried"] == 1.0
    assert "ebic" not in result.diagnostics
    assert result.diagnostics["n_edges"] >= 7.0


def test_lam_zero_with_p_at_least_n_raises() -> None:
    rng = np.random.default_rng(0)
    data = tabular(pd.DataFrame(rng.normal(size=(3, 4)), columns=_names(4)))
    with pytest.raises(ValueError, match="MLE"):
        Glasso(lam=0.0).fit(data)
    # with n > p the unpenalized solve is the plain MLE and runs
    wide = tabular(pd.DataFrame(rng.normal(size=(50, 4)), columns=_names(4)))
    assert Glasso(lam=0.0).fit(wide).diagnostics["lam"] == 0.0


def test_unconverged_raises_at_the_max_iter_cap() -> None:
    data = _chain_data(n=1000)
    with pytest.raises(RuntimeError, match="did not converge"):
        Glasso(lam=0.05, tol=1e-12, max_iter=1).fit(data)


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="lam"):
        Glasso(lam=-0.1)
    with pytest.raises(ValueError, match="gamma"):
        Glasso(gamma=-1.0)
    with pytest.raises(ValueError, match="n_lambdas"):
        Glasso(n_lambdas=0)
    with pytest.raises(ValueError, match="tol"):
        Glasso(tol=0.0)
    with pytest.raises(ValueError, match="max_iter"):
        Glasso(max_iter=0)


# -- knowledge -------------------------------------------------------------------------


def test_a_banned_pair_is_absent_even_when_strongly_correlated() -> None:
    data = _chain_data()
    pair = frozenset(("X0", "X1"))
    assert pair in _pairs(Glasso().fit(data).graph)
    knowledge = DomainKnowledge().forbid_adjacency("X0", "X1")
    assert pair not in _pairs(Glasso().fit(data, knowledge=knowledge).graph)


def test_a_required_pair_survives_a_lambda_that_would_kill_it() -> None:
    data = _chain_data(n=1000)
    top = _lam_max(_correlation(data))
    assert len(Glasso(lam=top).fit(data).graph.edges) == 0
    knowledge = DomainKnowledge().require("X0", "X1")
    survived = Glasso(lam=top).fit(data, knowledge=knowledge)
    assert _pairs(survived.graph) == {frozenset(("X0", "X1"))}
    # on the EBIC path the unpenalized pair also disables the empty-endpoint
    # short-circuit, and still comes out adjacent
    on_path = Glasso().fit(data, knowledge=knowledge)
    assert frozenset(("X0", "X1")) in _pairs(on_path.graph)


def test_a_skeleton_restriction_caps_the_support() -> None:
    data = _chain_data()
    allowed = [("X0", "X1"), ("X3", "X4")]
    result = Glasso().fit(data, knowledge=DomainKnowledge().restrict_to(allowed))
    assert _pairs(result.graph) <= {frozenset(pair) for pair in allowed}


# -- data validation -------------------------------------------------------------------


def test_a_categorical_column_raises_naming_it() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "X0": rng.normal(size=100),
            "X1": rng.normal(size=100),
            "shift": rng.choice(["day", "night"], size=100),
        }
    )
    with pytest.raises(ValueError, match="shift"):
        Glasso().fit(tabular(frame))


def test_a_temporal_shape_raises_type_error() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"t": np.arange(50.0), "X0": rng.normal(size=50)})
    with pytest.raises(TypeError, match="TabularData"):
        Glasso().fit(timeseries(frame, time="t"))  # type: ignore[arg-type]


# -- determinism -----------------------------------------------------------------------


def test_two_fits_are_identical() -> None:
    data = _chain_data(n=1500)
    first = Glasso().fit(data)
    second = Glasso().fit(data)
    assert first.graph == second.graph
    assert dict(first.diagnostics) == dict(second.diagnostics)
