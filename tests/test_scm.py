# SPDX-License-Identifier: Apache-2.0
"""SCM pillar: the dependency-free logic.

These tests run with no pymc present: mechanism auto-assignment by schema (pure
dispatch), the ``Sensitivity`` result API (no naked-point accessor), the
identification routing that decides point-vs-bounds (the routing logic, not the
sampler), input validation, and the guarded-import boundary (importing the
package and ``ergodic.scm`` must not import pymc). The pymc-gated golden-number
tests live in ``test_scm_pymc.py``.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from ergodic import SCM, Sensitivity, TabularData, dag, identify_effect, tabular
from ergodic.scm import (
    BinaryGumbelMax,
    Capability,
    CategoricalGumbelMax,
    GaussianAdditive,
    SCMResult,
)
from ergodic.scm._bounds import linear_sensitivity_curve
from ergodic.scm._mechanism import auto_mechanism, observed_parents, resolve_mechanism
from ergodic.scm._noise import gaussian_residual, gumbel_max_abduct, gumbel_max_decide

# -- the guarded-import boundary ---------------------------------------------


def test_importing_ergodic_does_not_import_pymc() -> None:
    """A fresh ``import ergodic`` (and ``ergodic.scm``) must not import pymc.

    Run in a subprocess so a pymc already imported by the rest of the suite does
    not mask a leak. The bare install must keep working.
    """
    code = (
        "import sys, ergodic, ergodic.scm\n"
        "from ergodic import SCM, Sensitivity\n"
        "assert 'pymc' not in sys.modules, 'pymc leaked into import ergodic'\n"
        "assert 'pytensor' not in sys.modules, 'pytensor leaked into import ergodic'\n"
        "print('clean')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_scm_top_level_exports() -> None:
    import ergodic

    # SCM core + the mechanism builders (n3): from ergodic import GaussianAdditive.
    for name in (
        "SCM",
        "SCMResult",
        "Sensitivity",
        "GaussianAdditive",
        "CategoricalGumbelMax",
        "BinaryGumbelMax",
    ):
        assert name in ergodic.__all__
        assert hasattr(ergodic, name)


def test_mechanism_builders_importable_from_top_level() -> None:
    """The plan's "top-level exports: SCM and the mechanism builders" (n3)."""
    from ergodic import BinaryGumbelMax as B
    from ergodic import CategoricalGumbelMax as C
    from ergodic import GaussianAdditive as G

    assert (GaussianAdditive, CategoricalGumbelMax, BinaryGumbelMax) == (G, C, B)


# -- mechanism auto-assignment by schema (pure dispatch) ---------------------


def _toy_data() -> TabularData:
    rng = np.random.default_rng(0)
    n = 50
    return tabular(
        {
            "cont": rng.normal(size=n),
            "binc": rng.integers(0, 2, size=n),
            "cat": rng.integers(0, 4, size=n),
        },
        types={"binc": "categorical", "cat": "categorical"},
    )


def test_auto_mechanism_continuous_is_gaussian_additive() -> None:
    data = _toy_data()
    g = dag(["cont -> cat"])
    mech = auto_mechanism(g, data, "cont")
    assert isinstance(mech, GaussianAdditive)
    assert mech.capability is Capability.INVERTIBLE


def test_auto_mechanism_binary_is_binary_gumbel_max() -> None:
    data = _toy_data()
    g = dag(["cont -> binc"])
    mech = auto_mechanism(g, data, "binc")
    assert isinstance(mech, BinaryGumbelMax)
    assert mech.n_categories == 2


def test_auto_mechanism_categorical_is_categorical_gumbel_max() -> None:
    data = _toy_data()
    g = dag(["cont -> cat"])
    mech = auto_mechanism(g, data, "cat")
    assert isinstance(mech, CategoricalGumbelMax)
    assert not isinstance(mech, BinaryGumbelMax)
    assert mech.n_categories == 4


def test_auto_mechanism_reads_parents_from_graph() -> None:
    data = _toy_data()
    g = dag(["cont -> cat", "binc -> cat"])
    mech = auto_mechanism(g, data, "cat")
    assert set(mech.parents) == {"cont", "binc"}


def test_resolve_mechanism_builtin_names() -> None:
    data = _toy_data()
    g = dag(["cont -> cat"])
    assert isinstance(resolve_mechanism("linear", g, data, "cont"), GaussianAdditive)
    assert isinstance(resolve_mechanism("categorical", g, data, "cat"), CategoricalGumbelMax)
    assert isinstance(resolve_mechanism("binary", g, data, "binc"), BinaryGumbelMax)


def test_resolve_mechanism_unknown_name_raises() -> None:
    data = _toy_data()
    g = dag(["cont -> cat"])
    with pytest.raises(ValueError, match="unknown mechanism"):
        resolve_mechanism("wizardry", g, data, "cont")


def test_resolve_mechanism_instance_fills_parents_immutably() -> None:
    """A blank mechanism is filled into a NEW instance; the caller's spec is untouched.

    Mechanisms are frozen value objects (M2): ``resolve_mechanism`` returns a copy
    with name/parents filled, never mutating the passed spec, so one mechanism can
    be reused across SCMs without cross-corruption.
    """
    data = _toy_data()
    g = dag(["cont -> cat"])
    spec = GaussianAdditive(name="", parents=())
    out = resolve_mechanism(spec, g, data, "cont")
    assert out is not spec  # a copy, not the same object
    assert out.name == "cont"
    assert spec.name == ""  # the original is unchanged
    # an already-named, already-parented mechanism passes straight through.
    ready = GaussianAdditive(name="cont", parents=("a",))
    assert resolve_mechanism(ready, g, data, "cont") is ready


def test_mechanism_is_frozen_and_resolves_independently_across_nodes() -> None:
    """One mechanism reused under two node names yields two independent results (M2).

    The frozen dataclass refuses in-place mutation, and ``resolve_mechanism``
    returns a fresh copy per node, so two SCMs sharing one mechanism instance never
    cross-corrupt.
    """
    import dataclasses

    data = _toy_data()
    g = dag(["cont -> cat", "binc -> cat"])
    shared = GaussianAdditive(name="", parents=())

    # frozen: a direct in-place mutation is refused.
    with pytest.raises(dataclasses.FrozenInstanceError):
        shared.name = "cont"  # type: ignore[misc]

    first = resolve_mechanism(shared, g, data, "cont")
    second = resolve_mechanism(shared, g, data, "cat")
    # the original is pristine; the two resolutions are independent objects.
    assert shared.name == "" and shared.parents == ()
    assert first.name == "cont" and first.parents == ()
    assert second.name == "cat" and set(second.parents) == {"cont", "binc"}
    assert first is not second is not shared


def test_observed_parents_drops_latents() -> None:
    g = dag(["U -> X", "U -> Y", "X -> Y"], latent=["U"])
    assert observed_parents(g, "Y") == ("X",)
    assert observed_parents(g, "X") == ()


def test_auto_mechanism_datetime_raises() -> None:
    import pandas as pd

    df = pd.DataFrame({"t": pd.date_range("2020-01-01", periods=10), "y": np.arange(10.0)})
    data = tabular(df)
    g = dag(["t -> y"])
    with pytest.raises(ValueError, match="datetime"):
        auto_mechanism(g, data, "t")


# -- the Sensitivity result API (no naked point) -----------------------------


def test_sensitivity_has_no_naked_point_accessor() -> None:
    """The honesty thesis: a non-identified effect cannot be read as a number."""
    sens = linear_sensitivity_curve(
        treatment="X", outcome="Y", cov_xy=1.2, var_x=1.0, obstruction="bow"
    )
    assert isinstance(sens, Sensitivity)
    assert not hasattr(sens, "value")
    assert not hasattr(sens, "mean")
    assert sens.identified is False


def test_sensitivity_curve_is_exact_line() -> None:
    """``b(c) = (cov - c) / var`` is exact and linear; ``at`` interpolates it."""
    sens = linear_sensitivity_curve(
        treatment="X", outcome="Y", cov_xy=2.0, var_x=2.0, obstruction="bow"
    )
    # b(0) is the observational slope cov/var = 1.0
    assert sens.identified_at_zero == pytest.approx(1.0)
    assert sens.at(0.0) == pytest.approx(1.0)
    # b(cov) = 0 (all covariance is confounding)
    assert sens.at(2.0) == pytest.approx(0.0, abs=1e-9)
    # b(1.0) = (2 - 1)/2 = 0.5
    assert sens.at(1.0) == pytest.approx(0.5)


def test_sensitivity_range_spans_the_band() -> None:
    sens = linear_sensitivity_curve(
        treatment="X", outcome="Y", cov_xy=1.4, var_x=1.0, obstruction="bow"
    )
    lo, hi = sens.range()
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(1.4)


def test_sensitivity_default_band_orients_with_sign() -> None:
    """A negatively confounded effect sweeps ``[cov, 0]``, still a valid band."""
    sens = linear_sensitivity_curve(
        treatment="X", outcome="Y", cov_xy=-1.0, var_x=1.0, obstruction="bow"
    )
    lo, hi = sens.range()
    assert lo == pytest.approx(-1.0)
    assert hi == pytest.approx(0.0, abs=1e-9)


def test_linear_sensitivity_zero_variance_raises() -> None:
    with pytest.raises(ValueError, match="var_x must be positive"):
        linear_sensitivity_curve(
            treatment="X", outcome="Y", cov_xy=1.0, var_x=0.0, obstruction="bow"
        )


# -- the SCMResult API -------------------------------------------------------


def test_scmresult_accessors() -> None:
    draws = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    res = SCMResult(query="ate", target="Y", kind="posterior", identified=True, _draws=draws)
    assert res.mean() == pytest.approx(3.0)
    assert res.value == pytest.approx(3.0)
    assert res.std_error == pytest.approx(float(np.std(draws, ddof=1)))
    lo, hi = res.interval(0.90)
    assert lo < 3.0 < hi
    assert res.draws().tolist() == draws.tolist()


def test_scmresult_interval_bad_level_raises() -> None:
    res = SCMResult(
        query="ate", target="Y", kind="posterior", identified=True, _draws=np.arange(10.0)
    )
    with pytest.raises(ValueError, match="level must be in"):
        res.interval(1.5)


def test_scmresult_carries_identified_flag() -> None:
    res = SCMResult(
        query="counterfactual",
        target="Y",
        kind="posterior",
        identified=False,
        _draws=np.arange(5.0),
    )
    assert res.identified is False
    assert "not identified" in repr(res)


# -- the noise / abduction primitives (pure numpy) ---------------------------


def test_gaussian_residual_is_exact() -> None:
    value = np.array([1.0, 2.0, 3.0])
    mean = np.array([0.5, 2.5, 2.0])
    assert gaussian_residual(value, mean).tolist() == [0.5, -0.5, 1.0]


def test_gumbel_max_abduction_reconstructs_observed_category() -> None:
    """The abducted Gumbel vector reproduces the observed category exactly."""
    rng = np.random.default_rng(0)
    n, k = 500, 4
    logits = rng.normal(size=(n, k))
    observed = rng.integers(0, k, size=n)
    g = gumbel_max_abduct(logits, observed, np.random.default_rng(1))
    recon = gumbel_max_decide(logits, g)
    assert np.all(recon == observed)


def test_gumbel_max_abduct_validates_categories() -> None:
    logits = np.zeros((3, 2))
    with pytest.raises(ValueError, match="observed categories"):
        gumbel_max_abduct(logits, np.array([0, 1, 5]), np.random.default_rng(0))


# -- SCM construction + input validation (no fit, no pymc) -------------------


def test_scm_requires_dag() -> None:
    from ergodic import cpdag

    data = _toy_data()
    g = cpdag(["cont -- cat"])
    with pytest.raises(ValueError, match="SCM fits a DAG"):
        SCM(g, data)


def test_scm_requires_a_column_per_observed_node() -> None:
    data = _toy_data()
    g = dag(["cont -> missingnode"])
    with pytest.raises(ValueError, match="needs a data column"):
        SCM(g, data)


def test_scm_latent_node_needs_no_column() -> None:
    """A latent confounder has no data column; the SCM fits the observed margin."""
    rng = np.random.default_rng(0)
    data = tabular({"X": rng.normal(size=30), "Y": rng.normal(size=30)})
    g = dag(["U -> X", "U -> Y", "X -> Y"], latent=["U"])
    scm = SCM(g, data)
    assert scm.observed == ("X", "Y")
    assert set(scm.mechanisms) == {"X", "Y"}
    # the latent parent is dropped from Y's mechanism
    assert scm.mechanisms["Y"].parents == ("X",)


def test_scm_query_before_fit_raises() -> None:
    data = _toy_data()
    g = dag(["cont -> cat"])
    scm = SCM(g, data)
    assert not scm.fitted
    with pytest.raises(RuntimeError, match="not fitted"):
        scm.interventional("cat", {"cont": 1.0})


def test_scm_unknown_node_raises() -> None:
    data = _toy_data()
    g = dag(["cont -> cat"])
    scm = SCM(g, data)
    with pytest.raises(ValueError, match="unknown node"):
        scm.do({"nope": 1.0})


def test_intervention_view_is_chainable_and_immutable() -> None:
    data = _toy_data()
    g = dag(["cont -> cat"])
    scm = SCM(g, data)
    view = scm.do({"cont": 1.0})
    view2 = view.do({"cont": 2.0})
    assert view.intervention == {"cont": 1.0}  # unchanged
    assert view2.intervention == {"cont": 2.0}


def test_ate_dml_engine_is_stubbed_with_pointer() -> None:
    """The default posterior path is complete; engine='dml' is a clear stub.

    The routing reaches the engine branch only for an identified effect, so we
    drive it directly to confirm the NotImplementedError carries the pointer
    without needing a sampler.
    """

    # an identified Identification, so ate() falls through to the engine switch.
    rng = np.random.default_rng(0)
    data = tabular({"X": rng.normal(size=20), "Y": rng.normal(size=20)})
    scm = SCM(dag(["X -> Y"]), data)
    # short-circuit the fit gate and the identification by monkeypatching state:
    scm._fits = {"X": None, "Y": None}  # type: ignore[dict-item]
    assert scm.fitted
    with pytest.raises(NotImplementedError, match="engine='dml'"):
        scm.ate("X", "Y", engine="dml")


def test_ate_raises_on_frontdoor_identified_effect_not_a_wrong_point() -> None:
    """C1: an effect identified by a NON-back-door criterion raises, never a point.

    The front-door world U->X, U->Y, X->M, M->Y (U latent) has a KNOWN-identified
    effect (true 1.2), but the SCM posterior plug-in computes back-door
    g-computation only -- on the U-confounded Y~M fit it would return a confident
    wrong number (~2.37) stamped identified=True. So the engine must refuse: it is
    gated on the identification *strategy* and raises with a pointer to
    estimate_effect. The raise is decidable before any MCMC.
    """
    graph = dag(["U -> X", "U -> Y", "X -> M", "M -> Y"], latent=["U"])

    # the effect IS identified -- by front-door, not adjustment. This documents
    # that the MVP defers a known-identified effect, not a non-identified one.
    ident = identify_effect(graph, "X", "Y")
    assert ident.identifiable is True
    assert ident.strategy == "frontdoor"

    rng = np.random.default_rng(0)
    n = 30
    data = tabular({"X": rng.normal(size=n), "M": rng.normal(size=n), "Y": rng.normal(size=n)})
    scm = SCM(graph, data)
    # short-circuit only the fit gate; the strategy gate fires before any sampling.
    scm._fits = dict.fromkeys(scm.mechanisms)  # type: ignore[assignment]
    assert scm.fitted
    with pytest.raises(NotImplementedError) as exc:
        scm.ate("X", "Y")
    msg = str(exc.value)
    assert "frontdoor" in msg
    assert "estimate_effect" in msg


def test_latent_node_rejected_as_intervention_target() -> None:
    """n1: do/interventional/counterfactual on a LATENT node raise clearly, not KeyError.

    A latent node has no column and no fitted mechanism, so it cannot be a do-target
    or a query target; the guard fires up front instead of an opaque KeyError from
    deep in PyMC. ``do()`` needs no fit, so this is dep-free.
    """
    rng = np.random.default_rng(0)
    data = tabular({"X": rng.normal(size=20), "Y": rng.normal(size=20)})
    g = dag(["U -> X", "U -> Y", "X -> Y"], latent=["U"])
    scm = SCM(g, data)
    # U is a graph node (so _check_node passes) but latent (so _check_observable fails).
    assert "U" in g.nodes
    with pytest.raises(ValueError, match="latent"):
        scm.do({"U": 1.0})
    # the fitted-gate guards interventional/counterfactual; the latent guard sits
    # behind it, so drive it past the gate with a monkeypatched fit.
    scm._fits = dict.fromkeys(scm.mechanisms)  # type: ignore[assignment]
    with pytest.raises(ValueError, match="latent"):
        scm.interventional("U")
    with pytest.raises(ValueError, match="latent"):
        scm.counterfactual({"X": [0.0], "Y": [0.0]}, {"X": 1.0}, "U")
