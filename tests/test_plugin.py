# SPDX-License-Identifier: Apache-2.0
"""Tests for the discrete plug-in estimator: ID end to end on categorical data.

The plug-in evaluates a general do-calculus functional on the empirical joint, so
the anchor is recovery of a planted truth. Two worlds carry an analytically known
ATE: a discrete front-door (latent X-Y confounding, the effect flows through M)
and a discrete Verma-style world that only the general ID algorithm identifies.
For each we assert the functional on the *true* joint is exact and on a large
sample is close. The honesty guards (continuous columns, a non-binary treatment,
an empty stratum) and reproducibility round out the file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from ergodic.graph import admg, dag
from ergodic.identification import Identification, identify_distribution, identify_effect
from ergodic.identification._evaluate import Factor, evaluate, marginalize
from ergodic.identification._functional import Functional, Marginal, Prob, Product
from ergodic.inference import estimate, estimate_effect
from ergodic.inference.estimators._plugin import plugin_estimate

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ergodic.data import TabularData
    from ergodic.graph import MixedGraph

# -- a tiny discrete-SCM toolkit (test machinery) ----------------------------

CPTs = dict[str, "tuple[list[str], np.ndarray]"]


def _random_cpts(graph: MixedGraph, card: dict[str, int], rng: np.random.Generator) -> CPTs:
    """A random CPT per node: a Dirichlet draw over its domain per parent cell."""
    cpts: CPTs = {}
    for v in sorted(graph.nodes):
        pa = sorted(graph.parents(v))
        rows = int(np.prod([card[p] for p in pa], dtype=int))
        table = rng.dirichlet(np.ones(card[v]), size=rows)
        cpts[v] = (pa, table.reshape((*(card[p] for p in pa), card[v])))
    return cpts


def _order(graph: MixedGraph) -> list[str]:
    """A topological order over the directed edges (latents included), name tie-broken."""
    nodes = sorted(graph.nodes)
    done: list[str] = []
    while len(done) < len(nodes):
        for v in nodes:
            if v in done:
                continue
            if set(graph.parents(v)) <= set(done):
                done.append(v)
                break
    return done


def _sample(
    graph: MixedGraph, card: dict[str, int], cpts: CPTs, n: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Ancestral sampling: draw every node (observed and latent) for ``n`` rows."""
    cols: dict[str, np.ndarray] = {}
    for v in _order(graph):
        pa, table = cpts[v]
        if pa:
            idx = tuple(cols[p] for p in pa)
            probs = table[idx]  # shape (n, card[v])
        else:
            probs = np.broadcast_to(table, (n, card[v]))
        u = rng.random(n)
        thresh = np.cumsum(probs, axis=1)
        cols[v] = (u[:, None] < thresh).argmax(axis=1).astype(np.int64)
    return cols


def _place(table: np.ndarray, axes: list[str], order: tuple[str, ...]) -> np.ndarray:
    perm = sorted(range(len(axes)), key=lambda i: order.index(axes[i]))
    moved = np.transpose(table, perm)
    shape = [1] * len(order)
    for i, ax in enumerate(axes[p] for p in perm):
        shape[order.index(ax)] = moved.shape[i]
    return moved.reshape(shape)


def _full_joint(
    graph: MixedGraph, card: dict[str, int], cpts: CPTs, *, do: dict[str, int] | None = None
) -> Factor:
    """The joint over every node via the truncated factorization, optionally intervened."""
    do = do or {}
    order = tuple(sorted(graph.nodes))
    values = np.ones(tuple(card[n] for n in order), dtype=float)
    for v in order:
        if v in do:
            mass = np.zeros(card[v])
            mass[do[v]] = 1.0
            values = values * _place(mass, [v], order)
        else:
            pa, table = cpts[v]
            values = values * _place(table, [*pa, v], order)
    return Factor(order, values)


def _observational(
    graph: MixedGraph, card: dict[str, int], cpts: CPTs, observed: set[str]
) -> Factor:
    joint = _full_joint(graph, card, cpts)
    return marginalize(joint, frozenset(joint.axes) - observed)


def _true_ate(
    graph: MixedGraph, card: dict[str, int], cpts: CPTs, x: str, y: str, values: np.ndarray
) -> float:
    """The planted ``E[Y|do(X=1)] - E[Y|do(X=0)]`` from the generating DAG."""

    def mean_at(level: int) -> float:
        joint = _full_joint(graph, card, cpts, do={x: level})
        py = marginalize(joint, frozenset(joint.axes) - {y})
        return float(np.dot(py.values, values))

    return mean_at(1) - mean_at(0)


def _discrete(cols: Mapping[str, np.ndarray]) -> TabularData:
    """Build TabularData with every column declared categorical (no inference warning)."""
    from ergodic import tabular

    frame = pd.DataFrame(dict(cols))
    types = dict.fromkeys(cols, "categorical")
    return tabular(frame, types=types)


def _id_record(graph: MixedGraph, x: str, y: str) -> Identification:
    """An ``Identification`` carrying the raw ID functional, bypassing recognition.

    Recognition routes a front-door functional to the named front-door estimator,
    so to exercise the plug-in on it we wrap the raw functional from
    ``identify_distribution`` as a ``strategy="id"`` record directly.
    """
    res = identify_distribution(graph, y, x)
    assert res.identifiable and res.functional is not None
    return Identification(
        treatment=frozenset({x}),
        outcome=frozenset({y}),
        identifiable=True,
        strategy="id",
        estimand=res.functional.render(),
        functional=res.functional,
    )


# -- the planted worlds ------------------------------------------------------


def _frontdoor_world(rng: np.random.Generator) -> tuple[MixedGraph, dict[str, int], CPTs]:
    """X -> M -> Y with latent U confounding X and Y; all binary."""
    gen = dag(["U -> X", "X -> M", "M -> Y", "U -> Y"], latent=["U"])
    card = dict.fromkeys(gen.nodes, 2)
    cpts = _random_cpts(gen, card, rng)
    return gen, card, cpts


def _verma_world(rng: np.random.Generator) -> tuple[MixedGraph, dict[str, int], CPTs]:
    """X -> Z1 -> Z2 -> Y, with bows X<->Z2 and Z1<->Y written as explicit latents.

    Only the general ID algorithm identifies P(Y | do(X)) here; it is neither a
    back-door, a front-door, nor an instrument problem.
    """
    gen = dag(
        ["X -> Z1", "Z1 -> Z2", "Z2 -> Y", "U1 -> X", "U1 -> Z2", "U2 -> Z1", "U2 -> Y"],
        latent=["U1", "U2"],
    )
    card = dict.fromkeys(gen.nodes, 2)
    cpts = _random_cpts(gen, card, rng)
    return gen, card, cpts


# -- exactness on the true joint ---------------------------------------------


def _plugin_ate_on_joint(functional: Functional, joint: Factor, card: dict[str, int]) -> float:
    """Evaluate a functional on a given joint and read the ATE for a binary 0/1 Y."""
    effect = evaluate(functional, joint, card)
    levels = np.array([0.0, 1.0])
    e1 = _read_arm(effect, "Y", "X", 1, levels)
    e0 = _read_arm(effect, "Y", "X", 0, levels)
    return e1 - e0


def _read_arm(effect: Factor, y: str, x: str, level: int, values: np.ndarray) -> float:
    pos = effect.axes.index(x)
    sliced = Factor(
        tuple(a for a in effect.axes if a != x), np.take(effect.values, level, axis=pos)
    )
    for axis in [a for a in sliced.axes if a != y]:
        p = sliced.axes.index(axis)
        sliced = Factor(
            tuple(a for a in sliced.axes if a != axis), np.take(sliced.values, 0, axis=p)
        )
    return float(np.dot(sliced.values, values))


def test_frontdoor_exact_on_true_joint() -> None:
    """The front-door functional on the TRUE observational joint is the exact ATE."""
    rng = np.random.default_rng(0)
    gen, card, cpts = _frontdoor_world(rng)
    observed = {"X", "M", "Y"}
    res = identify_distribution(admg(["X -> M", "M -> Y", "X <-> Y"]), "Y", "X")
    assert res.functional is not None
    obs = _observational(gen, card, cpts, observed)
    got = _plugin_ate_on_joint(res.functional, obs, card)
    want = _true_ate(gen, card, cpts, "X", "Y", np.array([0.0, 1.0]))
    assert got == pytest.approx(want, abs=1e-9)


def test_verma_exact_on_true_joint() -> None:
    """The Verma functional on the TRUE observational joint is the exact ATE."""
    rng = np.random.default_rng(1)
    gen, card, cpts = _verma_world(rng)
    g = admg(["X -> Z1", "Z1 -> Z2", "Z2 -> Y", "X <-> Z2", "Z1 <-> Y"])
    res = identify_distribution(g, "Y", "X")
    assert res.functional is not None
    obs = _observational(gen, card, cpts, {"X", "Z1", "Z2", "Y"})
    got = _plugin_ate_on_joint(res.functional, obs, card)
    want = _true_ate(gen, card, cpts, "X", "Y", np.array([0.0, 1.0]))
    assert got == pytest.approx(want, abs=1e-9)


# -- the plug-in on a sample -------------------------------------------------


def test_frontdoor_plugin_recovers_truth() -> None:
    """The plug-in on a large front-door sample recovers the planted ATE."""
    rng = np.random.default_rng(20260615)
    gen, card, cpts = _frontdoor_world(rng)
    cols = _sample(gen, card, cpts, 60000, rng)
    data = _discrete({k: cols[k] for k in ("X", "M", "Y")})
    truth = _true_ate(gen, card, cpts, "X", "Y", np.array([0.0, 1.0]))
    ident = _id_record(admg(["X -> M", "M -> Y", "X <-> Y"]), "X", "Y")
    est = plugin_estimate(ident, data, random_state=0, n_boot=200)
    assert est.estimator == "plug_in"
    assert est.kind == "bootstrap"
    assert est.value == pytest.approx(truth, abs=0.03)
    assert est.std_error is not None and est.std_error > 0.0


def test_verma_end_to_end_through_estimate_effect() -> None:
    """A discrete Verma world identifies as 'id' and the plug-in estimates the truth."""
    rng = np.random.default_rng(7)
    gen, card, cpts = _verma_world(rng)
    cols = _sample(gen, card, cpts, 80000, rng)
    data = _discrete({k: cols[k] for k in ("X", "Z1", "Z2", "Y")})
    truth = _true_ate(gen, card, cpts, "X", "Y", np.array([0.0, 1.0]))

    g = admg(["X -> Z1", "Z1 -> Z2", "Z2 -> Y", "X <-> Z2", "Z1 <-> Y"])
    ident = identify_effect(g, "X", "Y")
    assert ident.strategy == "id"
    assert ident.functional is not None
    est = estimate_effect(g, data, "X", "Y", random_state=0, n_boot=200)
    assert est.estimator == "plug_in"
    assert est.value == pytest.approx(truth, abs=0.04)


def test_plugin_is_consistent_on_growing_sample() -> None:
    """A very large sample's empirical ATE approaches the analytic truth."""
    rng = np.random.default_rng(3)
    gen, card, cpts = _frontdoor_world(rng)
    ident = _id_record(admg(["X -> M", "M -> Y", "X <-> Y"]), "X", "Y")
    truth = _true_ate(gen, card, cpts, "X", "Y", np.array([0.0, 1.0]))
    errors = []
    for n in (2000, 20000, 200000):
        cols = _sample(gen, card, cpts, n, rng)
        data = _discrete({k: cols[k] for k in ("X", "M", "Y")})
        est = plugin_estimate(ident, data, random_state=0, n_boot=50)
        errors.append(abs(est.value - truth))
    # the largest sample is the most accurate (consistency), and tight at the top
    assert errors[-1] < 0.01
    assert errors[-1] <= errors[0] + 1e-9


# -- a multi-level discrete outcome ------------------------------------------


def test_multilevel_outcome_weights_levels() -> None:
    """A three-level numeric outcome weights P(Y=k|do(x)) by the level value k."""
    rng = np.random.default_rng(11)
    gen = dag(["U -> X", "X -> M", "M -> Y", "U -> Y"], latent=["U"])
    card = {"U": 2, "X": 2, "M": 2, "Y": 3}  # Y takes 0, 1, 2
    cpts = _random_cpts(gen, card, rng)
    cols = _sample(gen, card, cpts, 80000, rng)
    data = _discrete({k: cols[k] for k in ("X", "M", "Y")})
    ident = _id_record(admg(["X -> M", "M -> Y", "X <-> Y"]), "X", "Y")
    truth = _true_ate(gen, card, cpts, "X", "Y", np.array([0.0, 1.0, 2.0]))
    est = plugin_estimate(ident, data, random_state=0, n_boot=100)
    assert est.value == pytest.approx(truth, abs=0.05)


# -- treated= ----------------------------------------------------------------


def test_treated_flips_the_contrast_sign() -> None:
    """Choosing the other level as treated negates the ATE."""
    rng = np.random.default_rng(5)
    gen, card, cpts = _frontdoor_world(rng)
    cols = _sample(gen, card, cpts, 40000, rng)
    # a string-coded treatment so treated= names a level
    x_str = np.where(cols["X"] == 1, "on", "off")
    frame = pd.DataFrame({"X": x_str, "M": cols["M"], "Y": cols["Y"]})
    from ergodic import tabular

    data = tabular(frame, types={"X": "categorical", "M": "categorical", "Y": "categorical"})
    ident = _id_record(admg(["X -> M", "M -> Y", "X <-> Y"]), "X", "Y")
    on = plugin_estimate(ident, data, treated="on", random_state=0, n_boot=50)
    off = plugin_estimate(ident, data, treated="off", random_state=0, n_boot=50)
    assert on.value == pytest.approx(-off.value, abs=1e-12)


# -- honesty guards ----------------------------------------------------------


def test_continuous_column_raises() -> None:
    """A functional whose variables include a continuous column raises clearly."""
    rng = np.random.default_rng(9)
    gen, card, cpts = _frontdoor_world(rng)
    cols = _sample(gen, card, cpts, 2000, rng)
    frame = pd.DataFrame(
        {"X": cols["X"], "M": cols["M"], "Y": cols["Y"].astype(float) + rng.normal(size=2000)}
    )
    from ergodic import tabular

    data = tabular(frame, types={"X": "categorical", "M": "categorical", "Y": "continuous"})
    ident = _id_record(admg(["X -> M", "M -> Y", "X <-> Y"]), "X", "Y")
    with pytest.raises(NotImplementedError, match=r"continuous functional|SCM"):
        plugin_estimate(ident, data, random_state=0)


def test_nonbinary_treatment_raises() -> None:
    """A treatment with three levels cannot form the two-level ATE contrast."""
    rng = np.random.default_rng(13)
    gen = dag(["U -> X", "X -> M", "M -> Y", "U -> Y"], latent=["U"])
    card = {"U": 2, "X": 3, "M": 2, "Y": 2}
    cpts = _random_cpts(gen, card, rng)
    cols = _sample(gen, card, cpts, 3000, rng)
    data = _discrete({k: cols[k] for k in ("X", "M", "Y")})
    ident = _id_record(admg(["X -> M", "M -> Y", "X <-> Y"]), "X", "Y")
    with pytest.raises(ValueError, match="two-level contrast"):
        plugin_estimate(ident, data, random_state=0)


def test_positivity_empty_stratum_raises() -> None:
    """An empty treatment-covariate stratum inside a populated covariate cell raises.

    The adjustment functional sum_z P(Y|X,Z) P(Z) reads P(Y | X=1, Z=z) for every
    z with P(Z=z) > 0. If no row has X=1 at some such z, that conditional is
    undefined; the plug-in raises rather than zero-filling.
    """
    # An adjustment world Z -> X, Z -> Y, X -> Y, but force X=1 to never occur at Z=1.
    rng = np.random.default_rng(0)
    n = 4000
    z = rng.integers(0, 2, n)
    x = np.where(z == 1, 0, rng.integers(0, 2, n))  # X is always 0 when Z=1: empty stratum
    y = ((x + z + (rng.random(n) < 0.3)) >= 1).astype(int)
    data = _discrete({"Z": z, "X": x, "Y": y})
    # the adjustment functional, written directly
    functional: Functional = Marginal(
        frozenset({"Z"}),
        Product(
            (
                Prob(frozenset({"Y"}), frozenset({"X", "Z"})),
                Prob(frozenset({"Z"})),
            )
        ),
    )
    ident = Identification(
        treatment=frozenset({"X"}),
        outcome=frozenset({"Y"}),
        identifiable=True,
        strategy="id",
        estimand=functional.render(),
        functional=functional,
    )
    with pytest.raises(ValueError, match="positivity violated"):
        plugin_estimate(ident, data, random_state=0)


def test_empty_arm_raises() -> None:
    """A treatment whose second level never occurs fails the overlap check.

    X is declared binary (categories 0 and 1) but every row is 0, so the
    do(X=1) arm is empty. The arm guard fires before any conditional is read.
    """
    rng = np.random.default_rng(0)
    n = 1000
    frame = pd.DataFrame(
        {
            "X": pd.Categorical(np.zeros(n, dtype=int), categories=[0, 1]),
            "M": rng.integers(0, 2, n),
            "Y": rng.integers(0, 2, n),
        }
    )
    from ergodic import tabular

    data = tabular(frame, types={"M": "categorical", "Y": "categorical"})
    assert data.schema["X"].cardinality == 2
    ident = _id_record(admg(["X -> M", "M -> Y", "X <-> Y"]), "X", "Y")
    with pytest.raises(ValueError, match=r"positivity violated.*level 1|no rows at level 1"):
        plugin_estimate(ident, data, random_state=0)


# -- reproducibility ---------------------------------------------------------


def test_same_random_state_identical_draws() -> None:
    """Two runs at the same random_state give identical point and bootstrap draws."""
    rng = np.random.default_rng(2)
    gen, card, cpts = _frontdoor_world(rng)
    cols = _sample(gen, card, cpts, 10000, rng)
    data = _discrete({k: cols[k] for k in ("X", "M", "Y")})
    ident = _id_record(admg(["X -> M", "M -> Y", "X <-> Y"]), "X", "Y")
    a = plugin_estimate(ident, data, random_state=42, n_boot=100)
    b = plugin_estimate(ident, data, random_state=42, n_boot=100)
    assert a.value == b.value
    assert a.draws is not None and b.draws is not None
    np.testing.assert_array_equal(a.draws, b.draws)


# -- the dispatch ------------------------------------------------------------


def test_estimate_dispatch_rejects_named_strategy_for_id() -> None:
    """Selecting a named estimator for a general functional is refused with a pointer."""
    rng = np.random.default_rng(4)
    gen, card, cpts = _verma_world(rng)
    cols = _sample(gen, card, cpts, 5000, rng)
    data = _discrete({k: cols[k] for k in ("X", "Z1", "Z2", "Y")})
    g = admg(["X -> Z1", "Z1 -> Z2", "Z2 -> Y", "X <-> Z2", "Z1 <-> Y"])
    ident = identify_effect(g, "X", "Y")
    assert ident.strategy == "id"
    with pytest.raises(ValueError, match="no named estimator"):
        estimate(ident, data, strategy="aipw")


def test_plugin_matches_hand_computed_adjustment() -> None:
    """On a tiny labelled world the plug-in's ATE equals the hand g-formula.

    A back-door functional sum_z P(Y|X,Z) P(Z) over a small designed table: the
    cell frequencies make E[Y|do(1)] and E[Y|do(0)] computable by hand, and the
    plug-in reads off exactly that on the empirical joint.
    """
    # 100 rows over (Z, X, Y) with chosen counts (read off the joint exactly).
    rows: list[tuple[int, int, int]] = []
    plan = {(0, 0, 1): 10, (0, 0, 0): 30, (0, 1, 1): 30, (0, 1, 0): 10}  # Z=0: 40 ctrl, 40 trt
    plan |= {(1, 0, 1): 8, (1, 0, 0): 2, (1, 1, 1): 6, (1, 1, 0): 4}  # Z=1: 10 ctrl, 10 trt
    for (z, x, y), count in plan.items():
        rows += [(z, x, y)] * count
    arr = np.array(rows)
    data = _discrete({"Z": arr[:, 0], "X": arr[:, 1], "Y": arr[:, 2]})
    functional: Functional = Marginal(
        frozenset({"Z"}),
        Product((Prob(frozenset({"Y"}), frozenset({"X", "Z"})), Prob(frozenset({"Z"})))),
    )
    ident = Identification(
        treatment=frozenset({"X"}),
        outcome=frozenset({"Y"}),
        identifiable=True,
        strategy="id",
        estimand=functional.render(),
        functional=functional,
    )
    est = plugin_estimate(ident, data, random_state=0, n_boot=50)
    # by hand: P(Z=0)=0.8, P(Z=1)=0.2.
    # E[Y|do(1)] = 0.8*(30/40) + 0.2*(6/10) = 0.6 + 0.12 = 0.72
    # E[Y|do(0)] = 0.8*(10/40) + 0.2*(8/10) = 0.2 + 0.16 = 0.36
    assert est.value == pytest.approx(0.72 - 0.36, abs=1e-12)
