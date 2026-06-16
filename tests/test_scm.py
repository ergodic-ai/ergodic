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
from ergodic.scm._mechanism import (
    FitResult,
    auto_mechanism,
    encode_parent_design,
    encoded_width,
    observed_parents,
    parent_cardinalities,
    resolve_mechanism,
)
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


# -- mixed-parent encoding (the inference _covariate_matrix convention) -------


def test_parent_cardinalities_reads_schema() -> None:
    """A continuous parent is None; a discrete parent carries its category count."""
    data = _toy_data()
    cards = parent_cardinalities(data, ["cont", "binc", "cat"])
    assert cards == (None, 2, 4)


def test_encoded_width_is_drop_first_one_hot() -> None:
    """Continuous -> 1 column, k-category -> k-1 columns (drop first), binary -> 1."""
    # cont(None)=1, binc(2)=1, cat(4)=3  => 5
    assert encoded_width((None, 2, 4)) == 5
    assert encoded_width(()) == 0
    assert encoded_width((None, None)) == 2


def test_encode_parent_design_matches_covariate_matrix_convention() -> None:
    """The SCM parent encoder reproduces the inference pillar's ``_covariate_matrix``.

    Same drop-first one-hot for a discrete parent, continuous passes through. This
    is the single encoding convention the fit view and the generative view share.
    """
    from ergodic.inference._estimator import _covariate_matrix

    rng = np.random.default_rng(0)
    n = 40
    data = tabular(
        {
            "cont": rng.normal(size=n),
            "cat": rng.integers(0, 4, size=n),
        },
        types={"cat": "categorical"},
    )
    # the SCM raw parent matrix (codes for the discrete column), then encode.
    raw = data.matrix(["cat", "cont"])
    cards = parent_cardinalities(data, ["cat", "cont"])
    scm_design = encode_parent_design(raw, cards)
    # the inference design over the same columns: cat one-hot drop-first (3 cols) +
    # cont (1 col), in sorted-name order ("cat" then "cont").
    inf_design = _covariate_matrix(data, ["cat", "cont"])
    assert scm_design.shape == inf_design.shape == (n, 4)
    assert np.allclose(scm_design, inf_design)


def test_encode_parent_design_binary_parent_is_single_column() -> None:
    """A binary parent is one 0/1 column (the code==1 indicator), not two."""
    raw = np.array([[0.0], [1.0], [1.0], [0.0]])
    enc = encode_parent_design(raw, (2,))
    assert enc.shape == (4, 1)
    assert enc.reshape(-1).tolist() == [0.0, 1.0, 1.0, 0.0]


def test_mechanism_design_width_expands_for_categorical_parents() -> None:
    """A node with a 4-category parent sizes its coefficient at the encoded width."""
    data = _toy_data()
    g = dag(["cont -> y", "binc -> y", "cat -> y"])
    # add a continuous child y
    rng = np.random.default_rng(1)
    data2 = tabular(
        {
            "cont": data.column("cont"),
            "binc": data.codes("binc"),
            "cat": data.codes("cat"),
            "y": rng.normal(size=data.n_rows),
        },
        types={"binc": "categorical", "cat": "categorical"},
    )
    mech = auto_mechanism(g, data2, "y")
    assert isinstance(mech, GaussianAdditive)
    assert mech.parent_cards == (2, 4, None)  # binc, cat, cont (sorted)
    assert mech._design_width == 5  # 1 + 3 + 1


# -- ATE level resolution + numeric-outcome gate (pre-MCMC, dep-free) ---------


def _binary_treatment_scm() -> SCM:
    """A built (unfitted) SCM with a binary treatment, fits monkeypatched."""
    rng = np.random.default_rng(0)
    n = 30
    data = tabular(
        {"Z": rng.normal(size=n), "T": rng.integers(0, 2, size=n), "Y": rng.normal(size=n)},
        types={"T": "categorical"},
    )
    scm = SCM(dag(["Z -> T", "Z -> Y", "T -> Y"]), data)
    return scm


def test_ate_categorical_outcome_raises_before_any_mcmc() -> None:
    """ate() requires a numeric outcome; a categorical outcome raises with a pointer.

    The gate fires before identification/MCMC, so it is decidable dep-free.
    """
    rng = np.random.default_rng(0)
    n = 30
    data = tabular(
        {"X": rng.normal(size=n), "Y": rng.integers(0, 3, size=n)},
        types={"Y": "categorical"},
    )
    scm = SCM(dag(["X -> Y"]), data)
    scm._fits = dict.fromkeys(scm.mechanisms)  # type: ignore[assignment]
    with pytest.raises(ValueError, match="numeric outcome"):
        scm.ate("X", "Y")


def test_ate_levels_default_binary_treatment_is_zero_one() -> None:
    """A binary treatment defaults to the (0, 1) contrast (no explicit levels needed)."""
    scm = _binary_treatment_scm()
    scm._fits = {n: _fake_fit(2 if n == "T" else 0) for n in scm.mechanisms}
    assert scm._ate_levels("T", None) == (0.0, 1.0)


def test_ate_levels_multicategory_treatment_requires_explicit_levels() -> None:
    """A k>2 treatment has no canonical pair; levels=None raises (the treated= pattern)."""
    rng = np.random.default_rng(0)
    n = 30
    data = tabular(
        {"T": rng.integers(0, 4, size=n), "Y": rng.normal(size=n)},
        types={"T": "categorical"},
    )
    scm = SCM(dag(["T -> Y"]), data)
    scm._fits = {n: _fake_fit(4 if n == "T" else 0) for n in scm.mechanisms}
    with pytest.raises(ValueError, match="no default contrast"):
        scm._ate_levels("T", None)
    # an explicit valid pair resolves
    assert scm._ate_levels("T", (1, 3)) == (1.0, 3.0)
    # an out-of-range code raises
    with pytest.raises(ValueError, match="not a category code"):
        scm._ate_levels("T", (0, 9))


def test_ate_levels_continuous_treatment_defaults_to_unit_step() -> None:
    rng = np.random.default_rng(0)
    n = 30
    data = tabular({"T": rng.normal(size=n), "Y": rng.normal(size=n)})
    scm = SCM(dag(["T -> Y"]), data)
    scm._fits = {n: _fake_fit(0) for n in scm.mechanisms}
    assert scm._ate_levels("T", None) == (0.0, 1.0)
    assert scm._ate_levels("T", (-2.0, 2.0)) == (-2.0, 2.0)


def _fake_fit(n_categories: int) -> FitResult:
    """A parameter-free FitResult carrying only a class count, for the level-resolution
    tests that need ``_fits[node].n_categories`` without running MCMC."""
    return FitResult(params={}, param_means={}, n_categories=n_categories)


def test_intervention_vector_dtypes_by_node_kind() -> None:
    """A categorical do-target pins as an int code; a continuous one as a float.

    The Categorical RV is integer-valued, so a float pin would corrupt it; the
    compose helper dtypes the do-vector to the node's kind. Dep-free (pure numpy).
    """
    from ergodic.scm._compose import _intervention_vector
    from ergodic.scm._mechanism import BinaryGumbelMax, GaussianAdditive

    cat = BinaryGumbelMax(name="T", parents=())
    cont = GaussianAdditive(name="Y", parents=())
    vt = _intervention_vector(cat, 1.0, 5)
    vy = _intervention_vector(cont, 2.5, 5)
    assert vt.dtype == np.int64 and vt.tolist() == [1, 1, 1, 1, 1]
    assert vy.dtype == np.float64 and vy.tolist() == [2.5] * 5


def test_marginal_from_codes_is_normalized_bincount() -> None:
    from ergodic.scm._compose import marginal_from_codes

    codes = np.array([0, 0, 1, 2, 2, 2])
    marg = marginal_from_codes(codes, 3)
    assert marg.tolist() == [2 / 6, 1 / 6, 3 / 6]
    # empty draws -> all-zero marginal of the right length
    assert marginal_from_codes(np.array([], dtype=np.int64), 4).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_c1_frontdoor_gate_holds_for_a_binary_treatment() -> None:
    """C1 is type-agnostic: a binary-treatment front-door effect still raises a pointer.

    The strategy gate fires before any MCMC, so a known-identified-by-front-door
    effect raises NotImplementedError (never a confident wrong point) even when the
    treatment is binary. Dep-free: the raise is pre-sampler.
    """
    graph = dag(["U -> X", "U -> Y", "X -> M", "M -> Y"], latent=["U"])
    assert identify_effect(graph, "X", "Y").strategy == "frontdoor"
    rng = np.random.default_rng(0)
    n = 30
    data = tabular(
        {"X": rng.integers(0, 2, size=n), "M": rng.normal(size=n), "Y": rng.normal(size=n)},
        types={"X": "categorical"},
    )
    scm = SCM(graph, data)
    scm._fits = dict.fromkeys(scm.mechanisms)  # type: ignore[assignment]
    with pytest.raises(NotImplementedError) as exc:
        scm.ate("X", "Y")
    assert "frontdoor" in str(exc.value)
    assert "estimate_effect" in str(exc.value)
