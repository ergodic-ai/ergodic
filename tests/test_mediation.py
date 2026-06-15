# SPDX-License-Identifier: Apache-2.0
"""Mediation: the identification conditions and the g-computation estimator.

Two anchors. ``test_linear_closed_form_recovers_planted`` plants a linear SCM
where the natural direct effect is the direct ``X -> Y`` coefficient and the total
effect is the direct part plus the indirect part through a binary mediator, and
asserts the g-computation recovers them. ``test_canonical_non_identification``
plants a mediator-outcome confounder that is itself affected by the treatment and
asserts the natural effects are not identifiable with the cross-world reason,
while the controlled direct effect still is. The rest pin the M1-M3 conditions,
CDE-through-ID against a direct adjustment computation, discrete agreement,
proportion-mediated stability, reproducibility, and the guards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from ergodic.graph import cpdag, dag
from ergodic.identification import (
    MediationID,
    identify_distribution,
    identify_mediation,
)
from ergodic.inference import MediationEstimate, estimate_mediation
from ergodic.inference.learners import OLS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ergodic.data import TabularData
    from ergodic.graph import MixedGraph

# -- planted-world machinery -------------------------------------------------


def _frame(cols: Mapping[str, np.ndarray], *, continuous: tuple[str, ...]) -> TabularData:
    """Build TabularData: the named columns continuous, the rest categorical."""
    from ergodic import tabular

    types = {name: ("continuous" if name in continuous else "categorical") for name in cols}
    return tabular(pd.DataFrame(dict(cols)), types=types)


def _confounded_world(
    rng: np.random.Generator,
    n: int,
    *,
    b_x: float,
    b_m: float,
    b_xm: float = 0.0,
    b_w: float = 1.0,
    alpha_x: float = 1.0,
) -> dict[str, np.ndarray]:
    """W -> {X, M, Y}, X -> {M, Y}, M -> Y; binary X and M, continuous Y.

    ``b_x`` is the direct X->Y coefficient, ``b_m`` the M->Y coefficient,
    ``b_xm`` an optional X*M interaction, ``b_w`` the W->Y coefficient. ``alpha_x``
    is X's pull on the mediator's logit.
    """
    w = rng.normal(size=n)
    x = (rng.random(n) < 1.0 / (1.0 + np.exp(-0.5 * w))).astype(int)
    pm = 1.0 / (1.0 + np.exp(-(-0.2 + alpha_x * x + 0.3 * w)))
    m = (rng.random(n) < pm).astype(int)
    y = b_x * x + b_m * m + b_xm * (x * m) + b_w * w + rng.normal(scale=0.4, size=n)
    return {"W": w, "X": x, "M": m, "Y": y}


def _confounded_graph() -> MixedGraph:
    return dag(["W -> X", "W -> Y", "W -> M", "X -> M", "M -> Y", "X -> Y"])


def _planted_natural_effects(
    cols: Mapping[str, np.ndarray], *, b_x: float, b_m: float, alpha_x: float = 1.0
) -> tuple[float, float]:
    """The true NDE/NIE for the additive ``_confounded_world``, from the parameters.

    With ``Y = b_x X + b_m M + b_w W + noise`` the natural direct effect is exactly
    ``b_x`` (the mediator probabilities sum to one), and the natural indirect effect
    is ``b_m`` times the treatment-induced change in ``P(M=1)`` averaged over ``W``,
    ``b_m * E_w[p(1|1,w) - p(1|0,w)]``. The mediator logit is ``-0.2 + alpha_x*X +
    0.3*W`` (matching ``_confounded_world``), so this is exact given the empirical W.
    """
    w = cols["W"]

    def p1(xl: int) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(-0.2 + alpha_x * xl + 0.3 * w)))

    nie = b_m * float(np.mean(p1(1) - p1(0)))
    return b_x, nie


def _hand_mediation_formula_no_covariate(
    cols: Mapping[str, np.ndarray],
) -> tuple[float, float]:
    """The discrete mediation formula computed by hand, no covariates.

    The plain Pearl formula on the empirical conditional means, the closed form the
    saturated g-computation must match. Returns (NDE, NIE).
    """
    x, m, y = cols["X"], cols["M"], cols["Y"]

    def ey(xl: int, ml: int) -> float:
        sel = (x == xl) & (m == ml)
        return float(y[sel].mean())

    def pm(ml: int, xl: int) -> float:
        return float((m[x == xl] == ml).mean())

    nde = sum((ey(1, ml) - ey(0, ml)) * pm(ml, 0) for ml in (0, 1))
    nie = sum(ey(0, ml) * (pm(ml, 1) - pm(ml, 0)) for ml in (0, 1))
    return nde, nie


# ============================================================================
# B1: identify_mediation
# ============================================================================


def test_confounded_world_identifies_with_w() -> None:
    """The classic W -> {X,M,Y} world identifies NDE/NIE adjusting for W."""
    g = _confounded_graph()
    res = identify_mediation(g, "X", "Y", "M")
    assert res.identifiable
    assert res.covariates == frozenset({"W"})
    assert res.cde_identifiable
    assert "M | X', W" in res.nde_estimand  # mediation-formula style
    assert "P(M | X, W) - P(M | X', W)" in res.nie_estimand
    # the cross-world caveat is always stated
    assert any("cross-world" in a.lower() or "untestable" in a.lower() for a in res.assumptions)


def test_no_confounder_needs_no_covariates() -> None:
    """X -> M -> Y, X -> Y with no confounding identifies with an empty W."""
    g = dag(["X -> M", "M -> Y", "X -> Y"])
    res = identify_mediation(g, "X", "Y", "M")
    assert res.identifiable
    assert res.covariates == frozenset()
    assert res.cde_identifiable


def test_canonical_non_identification() -> None:
    """A mediator-outcome confounder affected by the treatment defeats NDE/NIE.

    X -> L, L -> M, L -> Y: L confounds M and Y but is a descendant of X, so it
    cannot enter a non-descendant covariate set; M3 has no valid set and the
    natural effects are not identifiable. The CDE, a plain do(X),do(M) contrast,
    is still identifiable (L can be adjusted for once both are intervened).
    """
    g = dag(["X -> L", "L -> M", "L -> Y", "X -> M", "M -> Y", "X -> Y"])
    res = identify_mediation(g, "X", "Y", "M")
    assert not res.identifiable
    assert res.nde_estimand == ""
    assert res.nie_estimand == ""
    assert res.covariates is None
    reason = res.assumptions[0].lower()
    assert "cross-world" in reason or "recanting" in reason or "not identifiable" in reason
    # CDE: P(Y | do(X), do(M)) is identifiable here
    assert res.cde_identifiable
    joint = identify_distribution(g, frozenset({"Y"}), frozenset({"X", "M"}))
    assert joint.identifiable


def test_m1_violation_unblocked_xy_backdoor() -> None:
    """Latent X-Y confounding breaks M1, so NDE/NIE fail.

    An unobserved U confounds X and Y, leaving an open back-door X..Y that no
    observed W can block, so M1 fails for every candidate. The natural effects are
    not identifiable; the CDE (a do(X),do(M) contrast) is defeated by the same
    confounding.
    """
    g = dag(["U -> X", "U -> Y", "X -> M", "M -> Y", "X -> Y"], latent=["U"])
    res = identify_mediation(g, "X", "Y", "M")
    assert not res.identifiable
    assert not res.cde_identifiable


def test_m2_violation_unblocked_xm_backdoor() -> None:
    """Latent X-M confounding breaks M2 (no W blocks the X..M back-door).

    The natural effects fail, but the CDE survives: intervening on both X and M
    cuts the confounded X..M path, so the do(X),do(M) contrast is still identified.
    """
    g = dag(["U -> X", "U -> M", "X -> M", "M -> Y", "X -> Y"], latent=["U"])
    res = identify_mediation(g, "X", "Y", "M")
    assert not res.identifiable
    assert res.cde_identifiable


def test_m3_violation_unblocked_my_backdoor() -> None:
    """Latent M-Y confounding breaks M3 (no W u {X} blocks the M..Y back-door)."""
    g = dag(["U -> M", "U -> Y", "X -> M", "M -> Y", "X -> Y"], latent=["U"])
    res = identify_mediation(g, "X", "Y", "M")
    assert not res.identifiable


def test_validation_rejects_bad_inputs() -> None:
    """Names, disjointness, the mediator-on-a-causal-path check, and the class face."""
    g = _confounded_graph()
    with pytest.raises(ValueError, match="unknown node"):
        identify_mediation(g, "X", "Y", "Q")
    with pytest.raises(ValueError, match="disjoint"):
        identify_mediation(g, "X", "Y", "X")
    with pytest.raises(ValueError, match="disjoint"):
        identify_mediation(g, "X", "M", "M")
    # W is not on a causal path X -> Y
    with pytest.raises(ValueError, match="directed path"):
        identify_mediation(g, "X", "Y", "W")
    # a class face raises
    with pytest.raises(NotImplementedError, match=r"equivalence class|member"):
        identify_mediation(cpdag(["X -- M", "M -- Y"]), "X", "Y", "M")


def test_mediationid_is_frozen_and_repr() -> None:
    """MediationID is a frozen value object with a readable repr and truthiness."""
    res = identify_mediation(_confounded_graph(), "X", "Y", "M")
    assert isinstance(res, MediationID)
    assert bool(res) is True
    with pytest.raises(AttributeError):
        res.identifiable = False  # type: ignore[misc]
    assert "X" in repr(res) and "Y" in repr(res)


# ============================================================================
# B2: estimate_mediation, the linear closed form
# ============================================================================


def test_linear_closed_form_recovers_planted() -> None:
    """NDE is the direct coefficient; TE = NDE + NIE; the discrete formula matches.

    A linear additive outcome (no X*M interaction) makes the natural direct effect
    exactly the direct X->Y coefficient ``b_x``. The natural indirect effect is the
    treatment-induced shift in the mediator times its outcome coefficient; rather
    than re-derive it through the logistic link, we anchor it to the discrete
    mediation formula computed by hand on the same sample. TE = NDE + NIE by
    construction.
    """
    rng = np.random.default_rng(20260615)
    n = 60000
    b_x, b_m = 2.0, 3.0
    cols = _confounded_world(rng, n, b_x=b_x, b_m=b_m)
    data = _frame(cols, continuous=("W", "Y"))
    g = _confounded_graph()

    est = estimate_mediation(g, data, "X", "Y", "M", random_state=1, n_boot=100)
    assert isinstance(est, MediationEstimate)

    nde_true, nie_true = _planted_natural_effects(cols, b_x=b_x, b_m=b_m)
    # the direct effect is the planted coefficient (additive outcome)
    assert est.nde.value == pytest.approx(b_x, abs=0.06)
    assert est.nde.value == pytest.approx(nde_true, abs=0.06)
    # the indirect effect is b_m times the treatment-induced shift in P(M=1)
    assert est.nie.value == pytest.approx(nie_true, abs=0.06)
    # TE = NDE + NIE exactly (same computation)
    assert est.total.value == pytest.approx(est.nde.value + est.nie.value, abs=1e-12)
    # every quantity carries a bootstrap standard error and interval
    assert est.nde.std_error is not None and est.nde.std_error > 0.0
    lo, hi = est.total.interval()
    assert lo < est.total.value < hi


def test_cde_per_level_under_interaction() -> None:
    """With an X*M interaction the CDE differs by mediator level and matches direct g-formula.

    The interaction makes CDE(0) = b_x and CDE(1) = b_x + b_xm. The estimator's
    per-level CDE must recover both and equal a direct outcome-regression g-formula
    on the same data (the CDE-through-ID anchor: the identified do(X),do(M) contrast
    estimated two ways agrees).
    """
    rng = np.random.default_rng(7)
    n = 60000
    b_x, b_m, b_xm = 2.0, 3.0, 1.5
    cols = _confounded_world(rng, n, b_x=b_x, b_m=b_m, b_xm=b_xm)
    data = _frame(cols, continuous=("W", "Y"))
    g = _confounded_graph()

    # the joint interventional is identifiable (the CDE engine)
    assert identify_distribution(g, frozenset({"Y"}), frozenset({"X", "M"})).identifiable

    est = estimate_mediation(g, data, "X", "Y", "M", random_state=0, n_boot=60)
    assert est.cde[0].value == pytest.approx(b_x, abs=0.07)
    assert est.cde[1].value == pytest.approx(b_x + b_xm, abs=0.07)

    # direct adjustment g-formula for CDE(m): fit E[Y | X, M, X*M, W], contrast at m
    x, m, w, y = cols["X"], cols["M"], cols["W"], cols["Y"]
    design = np.column_stack([x, m, x * m, w]).astype(float)
    model = OLS().fit(design, y)

    def cde_direct(m_level: int) -> float:
        d1 = np.column_stack([np.ones(n), np.full(n, m_level), np.full(n, m_level), w]).astype(
            float
        )
        d0 = np.column_stack([np.zeros(n), np.full(n, m_level), np.zeros(n), w]).astype(float)
        return float((model.predict(d1) - model.predict(d0)).mean())

    assert est.cde[0].value == pytest.approx(cde_direct(0), abs=0.03)
    assert est.cde[1].value == pytest.approx(cde_direct(1), abs=0.03)


def test_discrete_agreement_matches_hand_formula() -> None:
    """A no-confounder discrete world: the g-computation equals the hand formula.

    Without covariates and with a saturated (x, m) outcome model the g-computation
    reduces exactly to the discrete mediation formula on the empirical conditional
    means; assert the two agree.
    """
    rng = np.random.default_rng(3)
    n = 60000
    x = (rng.random(n) < 0.5).astype(int)
    pm = np.where(x == 1, 0.7, 0.3)
    m = (rng.random(n) < pm).astype(int)
    y = 1.0 * x + 2.0 * m + rng.normal(scale=0.3, size=n)
    cols = {"X": x, "M": m, "Y": y}
    data = _frame(cols, continuous=("Y",))
    g = dag(["X -> M", "M -> Y", "X -> Y"])

    est = estimate_mediation(g, data, "X", "Y", "M", random_state=0, n_boot=50)
    nde_hand, nie_hand = _hand_mediation_formula_no_covariate(cols)
    assert est.nde.value == pytest.approx(nde_hand, abs=0.02)
    assert est.nie.value == pytest.approx(nie_hand, abs=0.02)
    assert est.covariates == frozenset()


def test_binary_outcome_runs() -> None:
    """A binary outcome uses a classifier outcome model and still decomposes."""
    rng = np.random.default_rng(11)
    n = 40000
    cols = _confounded_world(rng, n, b_x=2.0, b_m=3.0)
    # binarize Y
    y_cont = cols["Y"]
    cols = dict(cols)
    cols["Y"] = (y_cont > np.median(y_cont)).astype(int)
    data = _frame(cols, continuous=("W",))
    g = _confounded_graph()
    est = estimate_mediation(g, data, "X", "Y", "M", random_state=0, n_boot=40)
    # the effects are on the probability scale and sum to the total
    assert est.total.value == pytest.approx(est.nde.value + est.nie.value, abs=1e-12)
    assert np.isfinite(est.nde.value) and np.isfinite(est.nie.value)


# -- proportion mediated -----------------------------------------------------


def test_proportion_mediated_stable_away_from_zero() -> None:
    """A real total effect gives a stable, unflagged proportion mediated."""
    rng = np.random.default_rng(5)
    n = 50000
    cols = _confounded_world(rng, n, b_x=2.0, b_m=3.0)
    data = _frame(cols, continuous=("W", "Y"))
    est = estimate_mediation(_confounded_graph(), data, "X", "Y", "M", random_state=0, n_boot=60)
    pm = est.proportion_mediated
    assert 0.0 < pm < 1.0
    assert not est.proportion_mediated_flagged
    assert pm == pytest.approx(est.nie.value / est.total.value, abs=1e-12)


def test_proportion_mediated_flagged_near_zero() -> None:
    """A canceling design (direct and indirect of opposite sign, TE~0) is flagged.

    The direct coefficient is negative and the indirect path positive, tuned so the
    total effect is near zero; the proportion mediated is then unstable and the
    estimate flags it.
    """
    rng = np.random.default_rng(2)
    n = 50000
    # b_x negative, indirect (alpha_x>0, b_m>0) positive, sized to roughly cancel
    cols = _confounded_world(rng, n, b_x=-0.7, b_m=3.0, alpha_x=1.0, b_w=1.0)
    data = _frame(cols, continuous=("W", "Y"))
    est = estimate_mediation(_confounded_graph(), data, "X", "Y", "M", random_state=0, n_boot=60)
    assert abs(est.total.value) < 0.2
    assert est.proportion_mediated_flagged


# -- reproducibility and guards ----------------------------------------------


def test_same_random_state_identical() -> None:
    """Two runs at the same random_state give identical points and draws."""
    rng = np.random.default_rng(8)
    n = 20000
    cols = _confounded_world(rng, n, b_x=2.0, b_m=3.0)
    data = _frame(cols, continuous=("W", "Y"))
    g = _confounded_graph()
    a = estimate_mediation(g, data, "X", "Y", "M", random_state=42, n_boot=50)
    b = estimate_mediation(g, data, "X", "Y", "M", random_state=42, n_boot=50)
    assert a.nde.value == b.nde.value
    assert a.nie.value == b.nie.value
    assert a.nde.draws is not None and b.nde.draws is not None
    np.testing.assert_array_equal(a.nde.draws, b.nde.draws)


def test_continuous_mediator_raises() -> None:
    """A continuous mediator points at the future density / multiply-robust path."""
    rng = np.random.default_rng(0)
    n = 5000
    w = rng.normal(size=n)
    x = (rng.random(n) < 0.5).astype(int)
    m = 1.0 * x + 0.5 * w + rng.normal(size=n)  # continuous mediator
    y = 2.0 * x + 3.0 * m + w + rng.normal(size=n)
    data = _frame({"W": w, "X": x, "M": m, "Y": y}, continuous=("W", "M", "Y"))
    g = _confounded_graph()
    with pytest.raises(NotImplementedError, match=r"discrete|continuous"):
        estimate_mediation(g, data, "X", "Y", "M", random_state=0)


def test_non_binary_treatment_raises() -> None:
    """A three-level treatment cannot form the 0/1 mediation contrast."""
    rng = np.random.default_rng(0)
    n = 6000
    x = rng.integers(0, 3, n)
    m = (rng.random(n) < 0.5).astype(int)
    y = x + 2.0 * m + rng.normal(size=n)
    data = _frame({"X": x, "M": m, "Y": y}, continuous=("Y",))
    g = dag(["X -> M", "M -> Y", "X -> Y"])
    with pytest.raises(NotImplementedError, match="binary treatment"):
        estimate_mediation(g, data, "X", "Y", "M", random_state=0)


def test_treated_picks_the_level() -> None:
    """treated= names the level coded to 1 for a string-coded treatment."""
    rng = np.random.default_rng(4)
    n = 30000
    cols = _confounded_world(rng, n, b_x=2.0, b_m=3.0)
    x_str = np.where(cols["X"] == 1, "on", "off")
    frame = pd.DataFrame({"W": cols["W"], "X": x_str, "M": cols["M"], "Y": cols["Y"]})
    from ergodic import tabular

    data = tabular(
        frame,
        types={"W": "continuous", "X": "categorical", "M": "categorical", "Y": "continuous"},
    )
    g = _confounded_graph()
    on = estimate_mediation(g, data, "X", "Y", "M", treated="on", random_state=0, n_boot=40)
    off = estimate_mediation(g, data, "X", "Y", "M", treated="off", random_state=0, n_boot=40)
    # flipping the treated level negates the total effect
    assert on.total.value == pytest.approx(-off.total.value, abs=0.05)


def test_not_identifiable_estimate_raises() -> None:
    """estimate_mediation refuses a non-identifiable decomposition with a pointer."""
    rng = np.random.default_rng(0)
    n = 5000
    cols = _confounded_world(rng, n, b_x=2.0, b_m=3.0)
    # add an L that is a child of X and parent of M and Y -> non-identifiable
    l_node = (rng.random(n) < 1.0 / (1.0 + np.exp(-cols["X"].astype(float)))).astype(int)
    cols = dict(cols)
    cols["L"] = l_node
    data = _frame(cols, continuous=("W", "Y"))
    g = dag(["W -> X", "W -> Y", "X -> L", "L -> M", "L -> Y", "X -> M", "M -> Y", "X -> Y"])
    with pytest.raises(ValueError, match="not identifiable"):
        estimate_mediation(g, data, "X", "Y", "M", random_state=0)
