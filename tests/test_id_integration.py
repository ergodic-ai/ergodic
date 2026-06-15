# SPDX-License-Identifier: Apache-2.0
"""Integration of the ID engine into the public surface: A5, A6, A7.

A6 wires ``identify_effect`` to fall through to the general ID algorithm on a
single graph (so graphs outside back-door/front-door/IV, the Verma graph and its
kin, identify with ``strategy="id"`` and the functional in hand), exposes the raw
engine through ``identify_distribution``, and keeps ``estimate_effect`` coherent
(a clear raise on ``strategy="id"``). A5 recognizes a ladder-missed front-door
and never relabels without the complete criterion's blessing. A7 is ``ida`` over
an equivalence class.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from ergodic import (
    IDA,
    Functional,
    Hedge,
    IDResult,
    cpdag,
    dag,
    estimate_effect,
    ida,
    identify_distribution,
    identify_effect,
    mixed,
    tabular,
)
from ergodic.graph import (
    MixedGraph,
    cpdag_of_dag,
    dags_of_cpdag,
    latent_projection,
    mag_of_dag,
    pag_of_mag,
)
from ergodic.identification import (
    adjustment_set,
    frontdoor_set,
    instruments,
    is_frontdoor_set,
    is_valid_adjustment_set,
)
from ergodic.identification._criteria import _descendants
from ergodic.identification._recognize import _recognize_frontdoor, recognize

# -- canonical graphs --------------------------------------------------------


def _double_verma() -> MixedGraph:
    """The Verma graph A -> B -> C -> D with bows A <-> C and B <-> D.

    The query A -> D is identifiable only by the general ID algorithm: no single
    adjustment set, no front-door, no instrument. The textbook example of an
    effect outside the three named criteria.
    """
    return dag(
        ["A -> B", "B -> C", "C -> D", "U1 -> A", "U1 -> C", "U2 -> B", "U2 -> D"],
        latent=["U1", "U2"],
    )


def _bow() -> MixedGraph:
    """The bow arc X <-> Y, X -> Y: not identifiable, a hedge."""
    return latent_projection(dag(["U -> X", "U -> Y", "X -> Y"], latent=["U"]))


# -- A6: identify_effect falls through to ID ---------------------------------


def test_identify_effect_falls_through_to_id_on_verma() -> None:
    """A graph the named ladder misses but general ID identifies: strategy='id'."""
    g = _double_verma()
    # the ladder genuinely misses A -> D
    assert adjustment_set(g, "A", "D") is None
    assert frontdoor_set(g, "A", "D") is None
    assert instruments(g, "A", "D") == []
    result = identify_effect(g, "A", "D")
    assert result.identifiable
    assert result.strategy == "id"
    assert result.functional is not None
    assert isinstance(result.functional, Functional)
    # the estimand string is the rendered functional
    assert result.estimand == result.functional.render()
    assert result.hedge is None


def test_identify_effect_id_estimand_is_the_rendered_functional() -> None:
    g = _double_verma()
    result = identify_effect(g, "A", "D")
    assert "P(" in result.estimand  # a real estimand, not a placeholder reason


def test_identify_effect_not_identifiable_carries_a_hedge() -> None:
    """The bow: not identifiable, and the result now carries the hedge witness."""
    g = _bow()
    result = identify_effect(g, "X", "Y")
    assert not result.identifiable
    assert result.strategy == "none"
    assert result.hedge is not None
    assert isinstance(result.hedge, Hedge)
    # the estimand reason is the hedge's, not the old generic ladder message
    assert "hedge" in result.estimand
    assert "adjustment, front-door, or an instrument" not in result.estimand


# -- A6: identify_distribution -----------------------------------------------


def test_identify_distribution_returns_idresult() -> None:
    g = _double_verma()
    res = identify_distribution(g, "D", "A")
    assert isinstance(res, IDResult)
    assert res.identifiable
    assert res.functional is not None
    assert res.query == (frozenset({"D"}), frozenset({"A"}), frozenset())


def test_identify_distribution_conditional_query() -> None:
    """A conditional given= query runs IDC and returns a sensible IDResult."""
    chain = dag(["Z -> X", "X -> Y"])
    res = identify_distribution(chain, "Y", "X", "Z")
    assert isinstance(res, IDResult)
    assert res.identifiable
    assert res.functional is not None
    assert res.query == (frozenset({"Y"}), frozenset({"X"}), frozenset({"Z"}))


def test_identify_distribution_bow_is_not_identifiable_with_hedge() -> None:
    g = _bow()
    res = identify_distribution(g, "Y", "X")
    assert not res.identifiable
    assert res.functional is None
    assert res.hedge is not None
    assert isinstance(res.hedge, Hedge)


def test_identify_distribution_raises_on_cpdag() -> None:
    c = cpdag_of_dag(dag(["X -> Y", "Y -> Z"]))
    with pytest.raises(NotImplementedError, match="ida"):
        identify_distribution(c, "Z", "X")


def test_identify_distribution_raises_on_pag() -> None:
    p = pag_of_mag(mag_of_dag(dag(["X -> Y", "Z -> Y"])))
    with pytest.raises(NotImplementedError, match="single graph"):
        identify_distribution(p, "Y", "X")


def test_identify_distribution_raises_on_mixed() -> None:
    m = mixed(["X o-> Y"])
    with pytest.raises(ValueError, match="verified graph face"):
        identify_distribution(m, "Y", "X")


def test_identify_distribution_validates_names_and_disjointness() -> None:
    g = dag(["X -> Y"])
    with pytest.raises(ValueError, match="unknown node"):
        identify_distribution(g, "Q", "X")
    with pytest.raises(ValueError, match="disjoint"):
        identify_distribution(g, "Y", "Y")
    chain = dag(["Z -> X", "X -> Y"])
    with pytest.raises(ValueError, match="do-set and the conditioning set"):
        identify_distribution(chain, "Y", "X", "X")
    with pytest.raises(ValueError, match="outcome and the conditioning set"):
        identify_distribution(chain, "Y", "X", "Y")


# -- A6: back-compat, the named strategies are unchanged ---------------------


def test_backcompat_adjustment_strategy() -> None:
    result = identify_effect(dag(["Z -> X", "Z -> Y", "X -> Y"]), "X", "Y")
    assert result.strategy == "adjustment"
    assert result.adjustment_set == frozenset({"Z"})
    assert result.functional is None  # named strategies set no functional


def test_backcompat_frontdoor_strategy() -> None:
    from ergodic import admg

    result = identify_effect(admg(["X -> M", "M -> Y", "X <-> Y"]), "X", "Y")
    assert result.strategy == "frontdoor"
    assert result.mediators == frozenset({"M"})
    assert result.functional is None


def test_backcompat_iv_strategy() -> None:
    g = dag(["Z -> X", "X -> Y", "U -> X", "U -> Y"], latent=["U"])
    result = identify_effect(g, "X", "Y")
    assert result.strategy == "iv"
    assert result.functional is None


def test_backcompat_null_strategy() -> None:
    g = dag(["X -> Z", "Y -> Z"])  # no directed path X -> Y
    result = identify_effect(g, "X", "Y")
    assert result.strategy == "null"
    assert result.functional is None


# -- A6: estimate_effect coherence -------------------------------------------


def test_estimate_effect_runs_the_plugin_on_strategy_id() -> None:
    """A strategy='id' graph on discrete data now estimates through the plug-in."""
    g = _double_verma()
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame({c: rng.integers(0, 2, n) for c in ["A", "B", "C", "D"]})
    data = tabular(df, types=dict.fromkeys(["A", "B", "C", "D"], "categorical"))
    est = estimate_effect(g, data, "A", "D", random_state=0, n_boot=50)
    assert est.estimator == "plug_in"
    assert est.kind == "bootstrap"
    # independent columns: no effect, the plug-in lands near zero.
    assert abs(est.value) < 0.1


def test_estimate_effect_strategy_id_continuous_raises_with_scm_pointer() -> None:
    """A strategy='id' functional over a continuous column points at the SCM path."""
    g = _double_verma()
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({c: rng.integers(0, 2, n) for c in ["A", "B", "D"]})
    df["C"] = rng.normal(size=n)  # the mediator is continuous
    data = tabular(
        df, types={"A": "categorical", "B": "categorical", "D": "categorical", "C": "continuous"}
    )
    with pytest.raises(NotImplementedError) as excinfo:
        estimate_effect(g, data, "A", "D")
    msg = str(excinfo.value)
    assert "SCM" in msg  # names the SCM path for continuous functionals
    assert "C" in msg  # names the offending column


# -- A5: recognition ---------------------------------------------------------


def _frontdoor_miss_graph() -> MixedGraph:
    """A graph where the front-door set {A, B} is valid but the ladder misses it.

    The non-exhaustive ``frontdoor_set`` search tries the full causal-node pool
    and singletons; here neither is a valid front-door, yet the proper subset
    {A, B} is. Found by an offline brute-force search over small latent DAGs.
    """
    return dag(
        [
            "A -> B",
            "A -> C",
            "A -> Y",
            "B -> Y",
            "C -> Y",
            "L0 -> C",
            "L0 -> Y",
            "L1 -> A",
            "L1 -> B",
            "X -> A",
            "X -> B",
        ],
        nodes=["X", "A", "B", "C", "Y", "L0", "L1"],
        latent=["L0", "L1"],
    )


def test_recognition_finds_a_ladder_missed_frontdoor() -> None:
    """recognition's front-door subset search catches what frontdoor_set misses."""
    g = _frontdoor_miss_graph()
    x, y = frozenset({"X"}), frozenset({"Y"})
    # the ladder misses it
    assert frontdoor_set(g, "X", "Y") is None
    # recognition's exhaustive subset search finds a valid set
    found = _recognize_frontdoor(g, x, y)
    assert found is not None
    assert len(found) >= 2  # a genuine multi-mediator set, not a singleton
    # and it really is a front-door set by the complete criterion
    assert is_frontdoor_set(g, "X", "Y", found)


def test_recognition_relabels_when_only_frontdoor_applies() -> None:
    """When adjustment fails but a recognizable front-door exists, recognize returns it.

    Built by removing the back-door-free shortcut: a confounder on X forces
    adjustment to fail, so recognition does not short-circuit to adjustment, and
    the front-door subset search is what fires.
    """
    g = dag(
        ["X -> M1", "M1 -> Y", "X -> M2", "M2 -> Y", "U -> X", "U -> Y"],
        latent=["U"],
    )
    x, y = frozenset({"X"}), frozenset({"Y"})
    assert adjustment_set(g, "X", "Y") is None  # the U confounder blocks adjustment
    functional = identify_distribution(g, "Y", "X").functional
    assert functional is not None
    named = recognize(g, x, y, functional)
    assert named is not None
    strategy, chosen = named
    assert strategy == "frontdoor"
    assert is_frontdoor_set(g, "X", "Y", chosen)


def test_recognition_returns_none_on_a_genuine_functional() -> None:
    """The double-Verma A -> D stays a general functional: recognition gives None."""
    g = _double_verma()
    res = identify_distribution(g, "D", "A")
    assert res.functional is not None
    assert recognize(g, frozenset({"A"}), frozenset({"D"}), res.functional) is None


def test_recognition_never_returns_an_invalid_set() -> None:
    """Property: any set recognition returns passes the corresponding complete check.

    Over many random small latent DAGs and every identifiable (x, y), recognition
    is asked for a named reduction; whatever it returns is verified against the
    complete criterion it claims. This is the safety contract: recognition can
    never disagree with the criteria.
    """
    rng = np.random.default_rng(20260615)
    asked = 0
    relabeled = 0
    for _ in range(300):
        names = ["V0", "V1", "V2", "V3"]
        edges = [
            f"{names[i]} -> {names[j]}" for i, j in combinations(range(4), 2) if rng.random() < 0.5
        ]
        latents: list[str] = []
        for k in range(int(rng.integers(0, 3))):
            a, b = rng.choice(4, size=2, replace=False)
            latents.append(f"L{k}")
            edges.append(f"L{k} -> {names[a]}")
            edges.append(f"L{k} -> {names[b]}")
        try:
            g = dag(edges, nodes=names + latents, latent=latents)
        except ValueError:
            continue
        for x_name in names:
            x = frozenset({x_name})
            for y_name in names:
                if x_name == y_name:
                    continue
                y = frozenset({y_name})
                if y_name not in _descendants(g, x):
                    continue  # no causal path: not an ID-fallthrough case
                res = identify_distribution(g, y_name, x_name)
                if not res.identifiable or res.functional is None:
                    continue
                asked += 1
                named = recognize(g, x, y, res.functional)
                if named is None:
                    continue
                relabeled += 1
                strategy, chosen = named
                if strategy == "adjustment":
                    assert is_valid_adjustment_set(g, x_name, y_name, chosen)
                elif strategy == "frontdoor":
                    assert is_frontdoor_set(g, x_name, y_name, chosen)
                else:  # no other named form is allowed
                    raise AssertionError(f"unexpected strategy {strategy!r}")
    assert asked > 100, f"too few recognition queries ran ({asked})"
    assert relabeled > 0, "recognition never relabeled; the property check is vacuous"


# -- A7: ida over an equivalence class ---------------------------------------


def test_ida_on_cpdag_matches_per_member_identify_effect() -> None:
    """ida's member multiset equals running identify_effect on each DAG member."""
    c = cpdag_of_dag(dag(["X -> Y", "Y -> Z"]))
    members = dags_of_cpdag(c)
    result = ida(c, "X", "Z")
    assert isinstance(result, IDA)
    expected = tuple(identify_effect(m, "X", "Z") for m in members)
    assert result.members == expected
    assert result.n_members == len(members)


def test_ida_summary_fields() -> None:
    c = cpdag_of_dag(dag(["X -> Y", "Y -> Z"]))
    result = ida(c, "X", "Z")
    # every member identifies here (null or adjustment), so all_identifiable holds
    assert result.all_identifiable
    assert result.n_identifiable == result.n_members
    assert result.strategies <= {"null", "adjustment"}
    # this class disagrees on the estimand across members, so agree is False
    assert not result.agree
    assert bool(result) is True
    assert "IDA" in repr(result)


def test_ida_agree_when_class_is_unanimous() -> None:
    """A class whose members all share one estimand reports agree=True."""
    # a fully-directed CPDAG (one member) trivially agrees
    c = cpdag_of_dag(dag(["X -> Z", "Y -> Z", "Z -> W"]))
    members = dags_of_cpdag(c)
    if len(members) == 1:  # if the class collapses to one member, agreement is trivial
        result = ida(c, "X", "W")
        assert result.agree


def test_ida_raises_on_single_graph() -> None:
    with pytest.raises(ValueError, match="single graph has one identify_effect"):
        ida(dag(["X -> Y"]), "X", "Y")
    # an ADMG (the projected bow) is also a single graph, not a class
    with pytest.raises(ValueError, match="single graph"):
        ida(_bow(), "X", "Y")


def test_ida_raises_on_mixed() -> None:
    with pytest.raises(ValueError, match="verified class face"):
        ida(mixed(["X o-> Y"]), "X", "Y")


def test_ida_on_pag_class_level_adjustment() -> None:
    """A PAG where class-level adjustment identifies: ida returns a one-element record."""
    p = pag_of_mag(mag_of_dag(dag(["V0 -> V2", "V1 -> V2", "V2 -> V3"])))
    class_level = identify_effect(p, "V2", "V3")
    assert class_level.strategy == "adjustment"  # the visible V2 --> V3 edge
    result = ida(p, "V2", "V3")
    assert result.n_members == 1
    assert result.members[0] == class_level
    assert result.all_identifiable
    assert result.agree  # one member-invariant answer


def test_ida_on_pag_defers_when_general_id_needed() -> None:
    """A PAG the class-level adjustment criterion cannot identify: an honest raise."""
    p = pag_of_mag(mag_of_dag(dag(["X -> Y", "Z -> Y"])))
    assert not identify_effect(p, "X", "Y").identifiable  # not amenable
    with pytest.raises(NotImplementedError, match="DAG realization"):
        ida(p, "X", "Y")


def test_ida_max_members_caps() -> None:
    """The capped-with-raise enumeration discipline is honored."""
    # an empty CPDAG over four nodes: many DAG members
    c = cpdag(["A -- B", "B -- C", "C -- D"])
    with pytest.raises(ValueError, match="max_members"):
        ida(c, "A", "D", max_members=1)


# -- a smoke that the soundness of the id estimand survives the public path --


def test_public_path_id_functional_is_the_engine_functional() -> None:
    """identify_effect's functional equals identify_distribution's (no drift)."""
    g = _double_verma()
    via_effect = identify_effect(g, "A", "D")
    via_dist = identify_distribution(g, "D", "A")
    assert via_effect.functional == via_dist.functional
