# SPDX-License-Identifier: Apache-2.0
"""The general ID algorithm: soundness against the truth, the catalog, IDC, units.

The anchor is :func:`test_id_sound_against_truth`: random discrete latent DAGs
with random CPTs, the true ``P(y | do(x))`` computed by intervening in the
generating DAG, ID's functional evaluated on the observational ``P(V)``, asserted
equal. That single test proves the estimand algebra is right, not just the yes/no
verdict. The rest pin the catalog, IDC, the hedge, and the functional/evaluator
units.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from ergodic.graph import GraphKind, MixedGraph, dag, latent_projection
from ergodic.identification._evaluate import (
    Factor,
    evaluate,
    marginalize,
    product,
    ratio,
    read_off,
)
from ergodic.identification._functional import (
    Marginal,
    Prob,
    Product,
    Ratio,
    cond,
)
from ergodic.identification._id import Hedge, IDResult, _identify_admg

# -- discrete-SCM ground truth (test machinery) ------------------------------


def _random_cpts(
    graph: MixedGraph, card: dict[str, int], rng: np.random.Generator
) -> dict[str, tuple[list[str], np.ndarray]]:
    """A random CPT per node: a Dirichlet draw over the node's domain per parent cell."""
    cpts: dict[str, tuple[list[str], np.ndarray]] = {}
    for v in sorted(graph.nodes):
        pa = sorted(graph.parents(v))
        rows = int(np.prod([card[p] for p in pa], dtype=int))
        table = rng.dirichlet(np.ones(card[v]), size=rows)
        cpts[v] = (pa, table.reshape((*(card[p] for p in pa), card[v])))
    return cpts


def _place(table: np.ndarray, axes: list[str], order: tuple[str, ...]) -> np.ndarray:
    """Lay ``table`` (over ``axes``) into the broadcast shape of ``order``."""
    perm = sorted(range(len(axes)), key=lambda i: order.index(axes[i]))
    moved = np.transpose(table, perm)
    shape = [1] * len(order)
    for i, ax in enumerate(axes[p] for p in perm):
        shape[order.index(ax)] = moved.shape[i]
    return moved.reshape(shape)


def _full_joint(
    graph: MixedGraph,
    card: dict[str, int],
    cpts: dict[str, tuple[list[str], np.ndarray]],
    *,
    do: dict[str, int] | None = None,
) -> Factor:
    """The joint over every node (observed and latent) via the truncated factorization.

    With ``do`` an intervention assignment, the do-targets are fixed to their
    value (a point mass) and their incoming factor is dropped: the truncated
    factorization that defines ``P(V | do(x))``.
    """
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


def _marginal_over(joint: Factor, keep: set[str]) -> Factor:
    return marginalize(joint, frozenset(joint.axes) - keep)


def _true_interventional(
    graph: MixedGraph,
    card: dict[str, int],
    cpts: dict[str, tuple[list[str], np.ndarray]],
    y: set[str],
    x_assign: dict[str, int],
) -> Factor:
    """The ground-truth ``P(y | do(x = x_assign))`` over ``y``, from the generating DAG."""
    joint = _full_joint(graph, card, cpts, do=x_assign)
    return _marginal_over(joint, y)


def _observational(
    graph: MixedGraph,
    card: dict[str, int],
    cpts: dict[str, tuple[list[str], np.ndarray]],
    observed: set[str],
) -> Factor:
    """The observational ``P(observed)``: the full joint with the latents summed out."""
    return _marginal_over(_full_joint(graph, card, cpts), observed)


def _random_latent_dag(
    rng: np.random.Generator,
) -> tuple[MixedGraph, list[str], list[str]]:
    """A random small DAG with one or two latent confounders.

    Observed nodes ``X1..Xn`` (n in 3..5) in a random topological order with
    random directed edges among them. One or two latent nodes, each a parent of
    two distinct observed nodes (a confounder). Returns the full DAG, the
    observed names, and the latent names.
    """
    n = int(rng.integers(3, 6))
    observed = [f"V{i}" for i in range(n)]
    edges: list[str] = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                edges.append(f"{observed[i]} -> {observed[j]}")
    latents: list[str] = []
    for k in range(int(rng.integers(1, 3))):
        a, b = rng.choice(n, size=2, replace=False)
        latents.append(f"U{k}")
        edges.append(f"U{k} -> {observed[a]}")
        edges.append(f"U{k} -> {observed[b]}")
    full = dag(edges, nodes=observed + latents, latent=latents)
    return full, observed, latents


# -- the anchor: soundness against the truth ---------------------------------


def test_id_sound_against_truth() -> None:
    """ID's functional, evaluated on P(V), equals the true interventional.

    Many random latent worlds, every identifiable (x, y) split checked to
    numerical tolerance. The not-vacuous guard requires a meaningful count of
    identifiable checks actually pass, so the test cannot trivially succeed by
    everything being a hedge.
    """
    rng = np.random.default_rng(20260615)
    checked = 0
    hedges = 0
    for _ in range(120):
        full, observed, latents = _random_latent_dag(rng)
        card = {n: int(rng.integers(2, 4)) for n in observed}
        card |= {u: int(rng.integers(2, 4)) for u in latents}
        cpts = _random_cpts(full, card, rng)
        obs_joint = _observational(full, card, cpts, set(observed))

        for k in (1, 2):
            for x in combinations(sorted(observed), k):
                xs = set(x)
                rest = [v for v in observed if v not in xs]
                for y in combinations(rest, 1):
                    ys = set(y)
                    result = _identify_admg(full, frozenset(ys), frozenset(xs))
                    if not result.identifiable:
                        hedges += 1
                        continue
                    assert result.functional is not None
                    est = evaluate(result.functional, obs_joint, card)
                    _assert_matches_truth(est, full, card, cpts, xs, ys)
                    checked += 1
    assert checked > 200, f"only {checked} identifiable checks ran (hedges: {hedges})"


def _assert_matches_truth(
    est: Factor,
    full: MixedGraph,
    card: dict[str, int],
    cpts: dict[str, tuple[list[str], np.ndarray]],
    xs: set[str],
    ys: set[str],
) -> None:
    """Assert the evaluated estimand equals the true ``P(y | do(x))`` everywhere.

    The functional is exactly ``P(y | do(x))``, so it must be a function of the
    treatment values and the outcome only. It may, however, *carry* extra free
    variables (intervention targets the algorithm folded in, and their
    topological predecessors) that it is provably constant in. Iterating over
    every assignment of all its free axes and asserting equality to the truth
    (which depends only on the original ``x``) checks both correctness and that
    constancy at once.
    """
    extra = set(est.axes) - xs - ys
    assert extra <= set(full.nodes)
    for assignment in _assignments(set(est.axes), card):
        x_do = {v: assignment[v] for v in xs if v in assignment}
        y_at = {v: assignment[v] for v in ys if v in assignment}
        true = _true_interventional(full, card, cpts, ys, x_do)
        got = read_off(est, assignment, {})
        want = read_off(true, y_at, {})
        assert got == pytest.approx(want, abs=1e-9), (full.edge_list(), xs, ys)


def _assignments(nodes: set[str], card: dict[str, int]) -> list[dict[str, int]]:
    names = sorted(nodes)
    out: list[dict[str, int]] = []

    def rec(i: int, acc: dict[str, int]) -> None:
        if i == len(names):
            out.append(dict(acc))
            return
        for level in range(card[names[i]]):
            acc[names[i]] = level
            rec(i + 1, acc)

    rec(0, {})
    return out


# -- the identifiability catalog ---------------------------------------------


def _backdoor() -> MixedGraph:
    return dag(["Z -> X", "Z -> Y", "X -> Y"])


def _frontdoor() -> MixedGraph:
    return dag(["X -> M", "M -> Y", "U -> X", "U -> Y"], latent=["U"])


def _bow() -> MixedGraph:
    return latent_projection(dag(["U -> X", "U -> Y", "X -> Y"], latent=["U"]))


def _napkin() -> MixedGraph:
    return dag(
        ["W -> R", "R -> X", "X -> Y", "U1 -> W", "U1 -> X", "U2 -> W", "U2 -> Y"],
        latent=["U1", "U2"],
    )


def _verma() -> MixedGraph:
    return dag(["A -> B", "B -> C", "C -> D", "UBD -> B", "UBD -> D"], latent=["UBD"])


def _iv_bow() -> MixedGraph:
    return dag(["Z -> X", "U -> X", "U -> Y", "X -> Y"], latent=["U"])


def _check_against_truth(
    full: MixedGraph, observed: list[str], y: set[str], x: set[str], seed: int
) -> IDResult:
    """Identify and, if identifiable, assert the functional matches the truth."""
    rng = np.random.default_rng(seed)
    card = dict.fromkeys(full.nodes, 2)
    cpts = _random_cpts(full, card, rng)
    obs_joint = _observational(full, card, cpts, set(observed))
    result = _identify_admg(full, frozenset(y), frozenset(x))
    if result.identifiable:
        assert result.functional is not None
        est = evaluate(result.functional, obs_joint, card)
        _assert_matches_truth(est, full, card, cpts, x, y)
    return result


def test_catalog_backdoor_identifies() -> None:
    g = _backdoor()
    r = _check_against_truth(g, ["X", "Y", "Z"], {"Y"}, {"X"}, seed=1)
    assert r.identifiable


def test_catalog_frontdoor_identifies() -> None:
    g = _frontdoor()
    r = _check_against_truth(g, ["X", "M", "Y"], {"Y"}, {"X"}, seed=2)
    assert r.identifiable


def test_catalog_bow_is_a_hedge() -> None:
    g = _bow()
    r = _identify_admg(g, frozenset({"Y"}), frozenset({"X"}))
    assert not r.identifiable
    assert r.hedge is not None


def test_catalog_napkin_identifies() -> None:
    g = _napkin()
    r = _check_against_truth(g, ["W", "R", "X", "Y"], {"Y"}, {"X"}, seed=3)
    assert r.identifiable


def test_catalog_verma_identifies() -> None:
    g = _verma()
    r = _check_against_truth(g, ["A", "B", "C", "D"], {"D"}, {"C"}, seed=4)
    assert r.identifiable


def test_catalog_iv_bow_not_identifiable() -> None:
    g = _iv_bow()
    r = _identify_admg(g, frozenset({"Y"}), frozenset({"X"}))
    assert not r.identifiable
    assert r.hedge is not None


# -- IDC: conditional effects ------------------------------------------------


def test_idc_rule2_exchange_fires() -> None:
    """A pure chain Z -> X -> Y: in P(Y | do(X), Z), Z separates from Y once X is fixed.

    The rule-2 condition holds, so IDC moves Z into the intervention. The
    conditioning on Z then washes out: the estimand is provably constant in Z
    and equals P(Y | do(X)) = P(Y | X). The numerical check is that varying Z
    never moves the answer, which it equals the unconditional interventional.
    """
    full = dag(["Z -> X", "X -> Y"])
    observed = ["X", "Y", "Z"]
    rng = np.random.default_rng(31)
    card = dict.fromkeys(full.nodes, 2)
    cpts = _random_cpts(full, card, rng)
    obs_joint = _observational(full, card, cpts, set(observed))
    r = _identify_admg(full, frozenset({"Y"}), frozenset({"X"}), frozenset({"Z"}))
    assert r.identifiable
    assert r.functional is not None
    est = evaluate(r.functional, obs_joint, card)
    for xv in (0, 1):
        true = _true_interventional(full, card, cpts, {"Y"}, {"X": xv})  # no Z dependence
        for yv in (0, 1):
            want = read_off(true, {"Y": yv}, {})
            for zv in (0, 1):  # the answer must not vary with Z
                assert read_off(est, {"Y": yv, "X": xv, "Z": zv}, {}) == pytest.approx(
                    want, abs=1e-9
                )


def test_idc_ratio_form_when_rule2_does_not_fire() -> None:
    """When Z cannot move into the do-set, IDC returns the normalized joint.

    Z and Y share a latent confounder (Y <-> Z) and X -> Y. The rule-2 test
    fails (Z reaches Y off the treatment), so the estimand is the genuine ratio
    P(y, z | do(x)) / P(z | do(x)), which depends on Z. Checked against the
    truth, and the joint numerator must really be a function of Z.
    """
    full = dag(["U -> Z", "U -> Y", "X -> Y"], latent=["U"])
    observed = ["X", "Y", "Z"]
    rng = np.random.default_rng(32)
    card = dict.fromkeys(full.nodes, 2)
    cpts = _random_cpts(full, card, rng)
    obs_joint = _observational(full, card, cpts, set(observed))
    r = _identify_admg(full, frozenset({"Y"}), frozenset({"X"}), frozenset({"Z"}))
    assert r.identifiable
    assert r.functional is not None
    assert isinstance(r.functional, Ratio)  # the normalization, not a rule-2 collapse
    est = evaluate(r.functional, obs_joint, card)
    varies = False
    for xv in (0, 1):
        for zv in (0, 1):
            true = _true_conditional_interventional(full, card, cpts, {"Y"}, {"X": xv}, {"Z": zv})
            for yv in (0, 1):
                got = read_off(est, {"Y": yv}, {"X": xv, "Z": zv})
                assert got == pytest.approx(read_off(true, {"Y": yv}, {}), abs=1e-9)
                if abs(got - read_off(est, {"Y": yv}, {"X": xv, "Z": 1 - zv})) > 1e-6:
                    varies = True
    assert varies, "the conditional effect should genuinely depend on Z here"


def test_idc_conditional_matches_truth() -> None:
    """P(Y | do(X), Z) on a confounded graph equals the truth, ratio form or not."""
    full = dag(["Z -> X", "Z -> Y", "X -> Y", "X -> W", "W -> Y"])
    observed = ["Z", "X", "Y", "W"]
    rng = np.random.default_rng(11)
    card = dict.fromkeys(full.nodes, 2)
    cpts = _random_cpts(full, card, rng)
    obs_joint = _observational(full, card, cpts, set(observed))
    r = _identify_admg(full, frozenset({"Y"}), frozenset({"X"}), frozenset({"Z"}))
    assert r.identifiable
    assert r.functional is not None
    est = evaluate(r.functional, obs_joint, card)
    # truth: P(Y | do(X=x), Z=z) = sum_w P(Y | x, z, w) P(w | x) ... compute directly.
    for xv in (0, 1):
        for zv in (0, 1):
            true = _true_conditional_interventional(full, card, cpts, {"Y"}, {"X": xv}, {"Z": zv})
            for yv in (0, 1):
                got = read_off(est, {"Y": yv}, {"X": xv, "Z": zv})
                assert got == pytest.approx(read_off(true, {"Y": yv}, {}), abs=1e-9)


def _true_conditional_interventional(
    graph: MixedGraph,
    card: dict[str, int],
    cpts: dict[str, tuple[list[str], np.ndarray]],
    y: set[str],
    x_assign: dict[str, int],
    z_assign: dict[str, int],
) -> Factor:
    """Ground-truth ``P(y | do(x), z)`` = ``P(y, z | do(x)) / P(z | do(x))`` over ``y``."""
    joint = _full_joint(graph, card, cpts, do=x_assign)
    pyz = _marginal_over(joint, y | set(z_assign))
    pz = _marginal_over(joint, set(z_assign))
    cond_factor = ratio(pyz, pz)
    # slice at z = z_assign
    keep_y = tuple(a for a in cond_factor.axes if a in y)
    sliced = cond_factor.values
    for a in sorted(z_assign):
        ax = cond_factor.axes.index(a)
        sliced = np.take(sliced, z_assign[a], axis=ax)
        cond_factor = Factor(tuple(x for x in cond_factor.axes if x != a), sliced)
    return Factor(keep_y, cond_factor.values)


# -- the functional and the evaluator (units) --------------------------------


def test_render_matches_estimand_style() -> None:
    adj = Marginal(
        frozenset({"Z"}),
        Product((Prob(frozenset({"Y"}), frozenset({"X", "Z"})), Prob(frozenset({"Z"})))),
    )
    assert adj.render() == "sum_z P(Y | X, Z) P(Z)"
    assert Prob(frozenset({"Y"}), frozenset({"X", "Z"})).render() == "P(Y | X, Z)"
    assert Prob(frozenset({"Y"})).render() == "P(Y)"


def test_free_vars() -> None:
    adj = Marginal(
        frozenset({"Z"}),
        Product((Prob(frozenset({"Y"}), frozenset({"X", "Z"})), Prob(frozenset({"Z"})))),
    )
    assert adj.free_vars() == frozenset({"X", "Y"})


def test_simplify_is_conservative() -> None:
    # nested products flatten
    nested = Product((Product((Prob(frozenset({"A"})), Prob(frozenset({"B"})))),))
    flat = nested.simplify()
    assert isinstance(flat, Product)
    assert len(flat.factors) == 2
    # an empty-index marginal drops
    m = Marginal(frozenset(), Prob(frozenset({"A"})))
    assert m.simplify() == Prob(frozenset({"A"}))
    # an identically-one factor (P(s | s)) drops from a product
    one = Product((Prob(frozenset({"A"}), frozenset({"A"})), Prob(frozenset({"B"}))))
    assert one.simplify() == Prob(frozenset({"B"}))
    # a marginal over a variable the body never mentions is the identity
    unused = Marginal(frozenset({"Q"}), Prob(frozenset({"A"})))
    assert unused.simplify() == Prob(frozenset({"A"}))


def test_cond_top_level_is_a_plain_conditional() -> None:
    p = Prob(frozenset({"X", "Y"}))
    c = cond(p, frozenset({"Y"}), frozenset({"X"}))
    assert c == Prob(frozenset({"Y"}), frozenset({"X"}))


def test_cond_on_a_marginal_is_a_ratio() -> None:
    p = Marginal(frozenset({"W"}), Prob(frozenset({"X", "Y", "W"})))
    c = cond(p, frozenset({"Y"}), frozenset({"X"}))
    assert isinstance(c, Ratio)
    # numerator marginalizes to {X, Y}; denominator to {X}.
    assert c.num.free_vars() == frozenset({"X", "Y"})
    assert c.den.free_vars() == frozenset({"X"})


def test_evaluator_product_marginal_ratio() -> None:
    # P(X) and P(Y | X) on a hand joint.
    pj = np.array([[0.6 * 0.8, 0.6 * 0.2], [0.4 * 0.3, 0.4 * 0.7]])
    joint = Factor(("X", "Y"), pj)
    px = marginalize(joint, frozenset({"Y"}))
    assert px.values == pytest.approx([0.6, 0.4])
    py = marginalize(joint, frozenset({"X"}))
    assert py.values == pytest.approx([0.6, 0.4])
    # product P(X) P(Y) broadcasts to a 2x2
    outer = product([px, Factor(("Y",), py.values)], {"X": 2, "Y": 2})
    assert outer.values.shape == (2, 2)
    assert outer.values.sum() == pytest.approx(1.0)
    # ratio: P(X, Y) / P(X) = P(Y | X)
    cond_yx = ratio(joint, px)
    assert read_off(cond_yx, {"Y": 1}, {"X": 1}) == pytest.approx(0.7)
    # rows of the conditional sum to one
    assert cond_yx.values.sum(axis=1) == pytest.approx([1.0, 1.0])


def test_evaluator_prob_from_joint() -> None:
    pj = np.array([[0.6 * 0.8, 0.6 * 0.2], [0.4 * 0.3, 0.4 * 0.7]])
    joint = Factor(("X", "Y"), pj)
    py = evaluate(Prob(frozenset({"Y"})), joint, {"X": 2, "Y": 2})
    assert py.values == pytest.approx([0.6, 0.4])
    pygx = evaluate(Prob(frozenset({"Y"}), frozenset({"X"})), joint, {"X": 2, "Y": 2})
    assert read_off(pygx, {"Y": 1}, {"X": 0}) == pytest.approx(0.2)


def test_evaluator_ratio_guards_zero() -> None:
    # a zero denominator cell yields zero, not NaN.
    num = Factor(("A",), np.array([0.0, 0.5]))
    den = Factor(("A",), np.array([0.0, 1.0]))
    out = ratio(num, den)
    assert out.values[0] == 0.0
    assert out.values[1] == pytest.approx(0.5)


# -- the hedge ---------------------------------------------------------------


def test_hedge_witness_is_well_formed() -> None:
    g = _bow()
    r = _identify_admg(g, frozenset({"Y"}), frozenset({"X"}))
    assert not r.identifiable
    h = r.hedge
    assert isinstance(h, Hedge)
    # the witness names the failing subgraph and the district inside it
    assert h.nodes == frozenset({"X", "Y"})
    assert h.district <= h.nodes
    assert "not identifiable" in h.render()
    assert "hedge" in h.render()


def test_hedge_is_raised_and_carried() -> None:
    # the engine raises the Hedge internally; the runner captures it.
    g = _iv_bow()
    r = _identify_admg(g, frozenset({"Y"}), frozenset({"X"}))
    assert r.hedge is not None
    assert str(r.hedge) == r.hedge.render()
    assert bool(r) is False


def test_idresult_repr_reads_cleanly() -> None:
    g = _backdoor()
    r = _identify_admg(g, frozenset({"Y"}), frozenset({"X"}))
    text = repr(r)
    assert "IDResult" in text
    assert "do(['X'])" in text


# -- a guard on the primitive the runner leans on ----------------------------


def test_runner_projects_latents() -> None:
    # a DAG carrying latent= is projected to an ADMG before ID runs.
    g = dag(["X -> Y", "U -> X", "U -> Y"], latent=["U"])
    r = _identify_admg(g, frozenset({"Y"}), frozenset({"X"}))
    # X <-> Y after projection: the bow, not identifiable.
    assert not r.identifiable
    assert latent_projection(g).kind is GraphKind.ADMG
