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

from ergodic import SCM, Sensitivity, TabularData, dag, tabular
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

    for name in ("SCM", "SCMResult", "Sensitivity"):
        assert name in ergodic.__all__
        assert hasattr(ergodic, name)


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


def test_resolve_mechanism_instance_passthrough_fills_parents() -> None:
    data = _toy_data()
    g = dag(["cont -> cat"])
    spec = GaussianAdditive(name="", parents=())
    out = resolve_mechanism(spec, g, data, "cont")
    assert out is spec
    assert out.name == "cont"


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
