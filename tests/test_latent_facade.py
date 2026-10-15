# SPDX-License-Identifier: Apache-2.0
"""The latent facade: refusals, eager spec validation, and the result's arithmetic.

Nothing in this file samples. The refusals all fire before the engine is
reached, the ``latent_model`` validation is eager by design, and the
``LatentResult`` accessors (verdicts, gates, tick placement) are pure
arithmetic that a hand-built result exercises exactly.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

import ergodic.latent
from ergodic import panel, tabular, timeseries
from ergodic.latent import (
    LatentResult,
    MarkovChain,
    RegimeLikelihood,
    calibrate,
    certify,
    consensus,
    latent_model,
    recover,
    semi_markov,
)
from ergodic.latent._facade import wilson
from ergodic.latent._result import CalibrationReport, CalibrationWorld, CertifyResult, GateOutcome
from ergodic.latent.likelihoods import SpectralWhittle, SwitchingAR, spectral_whittle, switching_ar

if TYPE_CHECKING:
    import numpy.typing as npt

    from ergodic.data import TimeSeriesData

    Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class _Stub(RegimeLikelihood):
    """A likelihood that exists only to be an instance; never fitted here."""


def _series(n: int = 300, seed: int = 0) -> TimeSeriesData:
    rng = np.random.default_rng(seed)
    return timeseries({"t": np.arange(n), "signal": rng.normal(size=n)}, time="t")


# -- recover() refusals -------------------------------------------------------


def test_recover_refuses_tabular() -> None:
    data = tabular({"signal": np.arange(10.0)})
    with pytest.raises(TypeError, match="TimeSeriesData"):
        recover(data, "signal", model=_Stub())  # type: ignore[arg-type]


def test_recover_refuses_panel() -> None:
    data = panel(
        {"entity": [0, 0, 1, 1], "t": [0, 1, 0, 1], "signal": [1.0, 2.0, 3.0, 4.0]},
        entity="entity",
        time="t",
    )
    with pytest.raises(TypeError, match="one per entity"):
        recover(data, "signal", model=_Stub())  # type: ignore[arg-type]


def test_recover_refuses_non_likelihood_model() -> None:
    with pytest.raises(TypeError, match="RegimeLikelihood"):
        recover(_series(), "signal", model="spectral")  # type: ignore[arg-type]


def test_recover_refuses_graph() -> None:
    with pytest.raises(NotImplementedError, match="graph= is not supported"):
        recover(_series(), "signal", model=_Stub(), graph=object())


def test_recover_refuses_bad_chain() -> None:
    with pytest.raises(TypeError, match="MarkovChain"):
        recover(_series(), "signal", model=_Stub(), chain=(60.0, 1.0))  # type: ignore[arg-type]


def test_recover_refuses_unknown_target() -> None:
    with pytest.raises(KeyError, match="missing"):
        recover(_series(), "missing", model=_Stub())


def test_recover_refuses_categorical_target() -> None:
    data = timeseries({"t": np.arange(40), "label": ["a", "b"] * 20}, time="t")
    with pytest.raises(TypeError, match="continuous"):
        recover(data, "label", model=_Stub())


def test_recover_refuses_bad_budgets() -> None:
    with pytest.raises(ValueError, match="draws"):
        recover(_series(), "signal", model=_Stub(), draws=0)
    with pytest.raises(ValueError, match="warmup"):
        recover(_series(), "signal", model=_Stub(), warmup=0)
    with pytest.raises(ValueError, match="chains"):
        recover(_series(), "signal", model=_Stub(), chains=0)
    with pytest.raises(ValueError, match="decode_draws"):
        recover(_series(), "signal", model=_Stub(), decode_draws=0)


def test_recover_refuses_non_finite_series() -> None:
    values = np.random.default_rng(0).normal(size=50)
    values[7] = np.nan
    data = timeseries({"t": np.arange(50), "signal": values}, time="t")
    with pytest.raises(ValueError, match="non-finite"):
        recover(data, "signal", model=_Stub())


# -- certify() and calibrate() refusals (before anything samples) -------------


def test_certify_refuses_tabular() -> None:
    data = tabular({"signal": np.arange(10.0)})
    with pytest.raises(TypeError, match="TimeSeriesData"):
        certify(data, "signal", model=_Stub())  # type: ignore[arg-type]


def test_certify_refuses_non_likelihood_model() -> None:
    with pytest.raises(TypeError, match="RegimeLikelihood"):
        certify(_series(), "signal", model="spectral")  # type: ignore[arg-type]


def test_certify_refuses_unknown_target() -> None:
    with pytest.raises(KeyError, match="missing"):
        certify(_series(), "missing", model=_Stub())


def test_certify_refuses_bad_budgets() -> None:
    with pytest.raises(ValueError, match="n_null"):
        certify(_series(), "signal", model=_Stub(), n_null=0)
    with pytest.raises(ValueError, match="refit_draws"):
        certify(_series(), "signal", model=_Stub(), refit_draws=0)
    with pytest.raises(ValueError, match="refit_warmup"):
        certify(_series(), "signal", model=_Stub(), refit_warmup=0)


def test_calibrate_refuses_non_spec() -> None:
    with pytest.raises(TypeError, match="LatentModel"):
        calibrate(spectral_whittle(), _series(), "signal")  # type: ignore[arg-type]


def test_calibrate_refuses_bad_n_controls() -> None:
    spec = latent_model(model=_Stub())
    with pytest.raises(ValueError, match="n_controls"):
        calibrate(spec, _series(), "signal", n_controls=0)


def test_calibrate_refuses_unknown_target() -> None:
    spec = latent_model(model=_Stub())
    with pytest.raises(KeyError, match="missing"):
        calibrate(spec, _series(), "missing")


# -- latent_model eager validation --------------------------------------------


def test_latent_model_refuses_unknown_option() -> None:
    with pytest.raises(TypeError, match="horizon"):
        latent_model(model=_Stub(), horizon=12)


def test_latent_model_refuses_non_likelihood() -> None:
    with pytest.raises(TypeError, match="RegimeLikelihood"):
        latent_model(model="spectral")  # type: ignore[arg-type]


def test_latent_model_refuses_bad_chain_eagerly() -> None:
    with pytest.raises(TypeError, match="MarkovChain"):
        latent_model(model=_Stub(), chain=0.996)


def test_latent_model_holds_options() -> None:
    spec = latent_model(model=_Stub(), draws=250, chains=1, chain=MarkovChain(4.0, 2.0))
    assert spec.options["draws"] == 250
    assert isinstance(spec.model, _Stub)
    assert "Stub" in repr(spec)


# -- chain and contract objects -----------------------------------------------


def test_markov_chain_validates() -> None:
    with pytest.raises(ValueError, match="positive"):
        MarkovChain(0.0, 1.0)
    assert MarkovChain() == MarkovChain(60.0, 1.0)


def test_semi_markov_is_reserved() -> None:
    with pytest.raises(NotImplementedError, match="not available"):
        semi_markov(min_dwell=100)


def test_k_other_than_two_is_reserved() -> None:
    with pytest.raises(NotImplementedError, match="k=3"):
        _Stub(k=3)


def test_base_contract_refuses_by_name() -> None:
    stub = _Stub()
    prepared = spectral_whittle().prepare(np.random.default_rng(0).normal(size=800))
    with pytest.raises(NotImplementedError, match="prepare"):
        stub.prepare(np.zeros(10))
    with pytest.raises(NotImplementedError, match="emissions"):
        stub.emissions("signal", prepared)
    with pytest.raises(NotImplementedError, match="emissions_np"):
        stub.emissions_np(prepared, {})
    with pytest.raises(NotImplementedError, match="null_loglik"):
        stub.null_loglik(prepared)
    with pytest.raises(NotImplementedError, match="simulate_null"):
        stub.simulate_null(prepared, np.random.default_rng(0))


# -- likelihood preparation (numpy only, no sampling) -------------------------


def test_spectral_prepare_shapes() -> None:
    prep = spectral_whittle().prepare(np.random.default_rng(0).normal(size=1440))
    assert prep.T_eff == 15
    assert prep.tick_multiplier == 96
    assert prep.offset == 0
    assert prep.obs["means"].shape == (15,)
    assert prep.obs["pgram"].shape == (15, 40)
    assert prep.obs["freqs"].shape == (40,)


def test_spectral_prepare_refuses_short_series() -> None:
    with pytest.raises(ValueError, match="frame"):
        spectral_whittle().prepare(np.random.default_rng(0).normal(size=200))


def test_spectral_refuses_bad_bins() -> None:
    with pytest.raises(ValueError, match="bin_lo"):
        spectral_whittle(bin_lo=0)
    with pytest.raises(ValueError, match="window"):
        spectral_whittle(window=8)


def test_spectral_prepare_refuses_constant_series() -> None:
    with pytest.raises(ValueError, match="constant"):
        spectral_whittle().prepare(np.ones(1000))


def test_switching_ar_prepare_shapes() -> None:
    prep = switching_ar().prepare(np.random.default_rng(0).normal(size=480))
    assert prep.T_eff == 477
    assert prep.tick_multiplier == 1
    assert prep.offset == 3
    assert prep.obs["y"].shape == (477,)
    assert prep.obs["lags"].shape == (477, 3)
    # Standardized by the target's own moments.
    assert prep.obs["y"].mean() == pytest.approx(0.0, abs=1e-12)
    assert prep.obs["y"].std() == pytest.approx(1.0, abs=1e-12)


def test_switching_ar_prepare_refuses_short_series() -> None:
    with pytest.raises(ValueError, match="lags"):
        switching_ar().prepare(np.arange(10.0))


def test_switching_ar_refuses_bad_order() -> None:
    with pytest.raises(ValueError, match="lag order"):
        switching_ar(p=0)


def test_likelihood_factories_are_frozen_values() -> None:
    assert spectral_whittle() == SpectralWhittle()
    assert switching_ar(p=2) == SwitchingAR(p=2)
    with pytest.raises(AttributeError):
        spectral_whittle().window = 48  # type: ignore[misc]


# -- LatentResult arithmetic --------------------------------------------------


def _result(
    delta_draws: dict[str, Array] | None = None,
    gates: tuple[GateOutcome, ...] = (),
) -> LatentResult:
    rng = np.random.default_rng(0)
    n_draws = 200
    gammas = np.clip(
        np.tile(np.array([0.05] * 4 + [0.95] * 4 + [0.05] * 2), (n_draws, 1))
        + rng.normal(0.0, 0.02, (n_draws, 10)),
        0.0,
        1.0,
    )
    if delta_draws is None:
        delta_draws = {
            "level": rng.normal(1.0, 0.05, n_draws),  # far from zero: fires
            "spread": rng.normal(0.0, 0.05, n_draws),  # straddles zero: silent
        }
    return LatentResult(
        target="signal",
        gammas=gammas,
        flips=np.zeros(n_draws, dtype=bool),
        delta_draws=delta_draws,
        gates=gates,
        tick_multiplier=96,
        offset=0,
        chain=MarkovChain(4.0, 2.0),
        likelihood=_Stub(),
        n_draws=800,
    )


def test_verdicts_fire_on_zero_exclusion_only() -> None:
    verdicts = _result().verdicts()
    assert verdicts == {"level": True, "spread": False}


def test_failed_gate_vetoes_its_blocks() -> None:
    rng = np.random.default_rng(1)
    draws = {
        "level": rng.normal(1.0, 0.05, 200),
        "amp": rng.normal(1.0, 0.05, 200),
    }
    gate = GateOutcome("wave_present", ("amp",), value=-0.3, passed=False)
    verdicts = _result(draws, gates=(gate,)).verdicts()
    assert verdicts == {"level": True, "amp": False}
    passed = GateOutcome("wave_present", ("amp",), value=0.4, passed=True)
    assert _result(draws, gates=(passed,)).verdicts() == {"level": True, "amp": True}


def test_deltas_are_equal_tailed_intervals() -> None:
    lo, mean, hi = _result().deltas(level=0.9)["level"]
    assert lo < mean < hi
    assert mean == pytest.approx(1.0, abs=0.02)
    with pytest.raises(ValueError, match="level"):
        _result().deltas(level=1.2)


def test_gamma_path_and_tick_placement() -> None:
    result = _result()
    gamma = result.gamma()
    assert gamma.shape == (10, 2)
    np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-12)
    assert result.path().tolist() == [0] * 4 + [1] * 4 + [0] * 2
    ticks = result.tick_path()
    assert ticks.shape == (960,)
    assert ticks[4 * 96] == 1 and ticks[0] == 0
    frame = result.to_frame()
    assert list(frame.columns) == ["step", "tick", "gamma", "state"]
    assert frame["tick"].tolist() == [96 * i for i in range(10)]


def test_result_owns_and_freezes_its_arrays() -> None:
    result = _result()
    with pytest.raises(ValueError, match="read-only"):
        result.gammas[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        result.delta_draws["level"][0] = 5.0


def test_result_repr_names_fired_blocks() -> None:
    assert "level" in repr(_result())


# -- bare-install import safety -----------------------------------------------

_SAMPLER_ONLY = {"pymc", "pytensor", "arviz", "jax", "numpyro"}


def test_no_module_level_sampler_imports() -> None:
    """The bare install imports ergodic.latent: no top-level pymc anywhere.

    Walks every module of the pillar and checks the module-level import
    statements only; imports inside functions are the sanctioned pattern.
    """
    root = Path(ergodic.latent.__file__).parent
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for stmt in tree.body:
            if isinstance(stmt, ast.Import):
                names = {alias.name.split(".")[0] for alias in stmt.names}
            elif isinstance(stmt, ast.ImportFrom):
                names = {(stmt.module or "").split(".")[0]}
            else:
                continue
            hit = names & _SAMPLER_ONLY
            assert not hit, f"{path.name} imports {sorted(hit)} at module level"


def test_import_survives_without_pymc() -> None:
    """The stronger check: import the pillar in a process where pymc cannot load."""
    code = (
        "import sys\n"
        "for name in ('pymc', 'pytensor', 'arviz', 'jax', 'numpyro'):\n"
        "    sys.modules[name] = None\n"
        "import ergodic\n"
        "import ergodic.latent\n"
        "import ergodic.latent.likelihoods\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


# -- wilson() -----------------------------------------------------------------


def test_wilson_is_honest_at_the_boundaries() -> None:
    lo, hi = wilson(0, 20)
    assert lo == 0.0
    assert 0.1 < hi < 0.2  # the Wald interval would collapse to [0, 0]
    lo, hi = wilson(20, 20)
    assert hi == pytest.approx(1.0)
    assert 0.8 < lo < 0.9
    assert wilson(0, 0) == (0.0, 0.0)


def test_wilson_contains_the_point_rate() -> None:
    lo, hi = wilson(7, 20)
    assert lo < 7 / 20 < hi
    assert 0.0 <= lo <= hi <= 1.0


# -- the certify and calibration records (pure arithmetic) --------------------


def test_certify_result_record() -> None:
    result = CertifyResult(observed=5.0, null=np.array([1.0, 6.0, 2.0]), p_value=0.5)
    assert result.n_null == 3
    frame = result.to_frame()
    assert list(frame.columns) == ["statistic", "source"]
    assert frame["source"].tolist() == ["observed", "null", "null", "null"]
    assert frame["statistic"].tolist() == [5.0, 1.0, 6.0, 2.0]
    with pytest.raises(ValueError, match="read-only"):
        result.null[0] = 9.0
    assert "n_null=3" in repr(result)
    with pytest.raises(ValueError, match="1-D"):
        CertifyResult(observed=1.0, null=np.zeros((2, 2)), p_value=0.5)


def test_calibration_report_record() -> None:
    worlds = (
        CalibrationWorld(world=0, seed=0, fired=False, fired_blocks=()),
        CalibrationWorld(world=1, seed=1, fired=True, fired_blocks=("level",)),
    )
    report = CalibrationReport(n=2, fires=1, rate=0.5, ci_lo=0.1, ci_hi=0.9, worlds=worlds)
    assert "1 of 2" in report.summary()
    frame = report.to_frame()
    assert frame["fired"].tolist() == [False, True]
    assert frame["fired_blocks"].tolist() == ["", "level"]
    assert "CalibrationReport" in repr(report)


# -- consensus() --------------------------------------------------------------


def _consensus_result(
    blocks: dict[str, float], likelihood: RegimeLikelihood, target: str = "signal"
) -> LatentResult:
    """A result whose blocks sit where the test wants: near a value, sd 0.05."""
    rng = np.random.default_rng(7)
    n_draws = 200
    gammas = np.clip(
        np.tile(np.linspace(0.0, 1.0, 12), (n_draws, 1)) + rng.normal(0.0, 0.01, (n_draws, 12)),
        0.0,
        1.0,
    )
    return LatentResult(
        target=target,
        gammas=gammas,
        flips=np.zeros(n_draws, dtype=bool),
        delta_draws={name: rng.normal(value, 0.05, n_draws) for name, value in blocks.items()},
        gates=(),
        tick_multiplier=1,
        offset=0,
        chain=MarkovChain(),
        likelihood=likelihood,
        n_draws=800,
    )


def test_consensus_separates_agreement_absence_and_contest() -> None:
    a = _consensus_result({"level": 1.0, "spread": 0.0, "amp": 1.0}, spectral_whittle())
    b = _consensus_result({"level": 1.0, "spread": 1.0, "rhythm": 0.0}, switching_ar())
    cons = consensus([a, b])
    assert cons.models == ("SpectralWhittle", "SwitchingAR")
    assert cons.blocks == ("level", "spread", "amp", "rhythm")
    # Absence (not in the vocabulary) is None, never False.
    assert cons.verdicts["amp"] == (True, None)
    assert cons.verdicts["rhythm"] == (None, False)
    assert cons.agreed == {"level": True, "amp": True, "rhythm": False}
    assert cons.contested == ("spread",)
    frame = cons.to_frame()
    assert list(frame.columns) == ["block", "SpectralWhittle", "SwitchingAR", "consensus"]
    by_block = frame.set_index("block")["consensus"]
    assert by_block["level"] == "fires"
    assert by_block["spread"] == "contested"
    assert by_block["rhythm"] == "quiet"
    assert "spread" in repr(cons)


def test_consensus_refusals() -> None:
    a = _consensus_result({"level": 1.0}, spectral_whittle())
    with pytest.raises(ValueError, match="at least two"):
        consensus([a])
    with pytest.raises(TypeError, match="LatentResult"):
        consensus([a, "not a result"])  # type: ignore[list-item]
    b = _consensus_result({"level": 1.0}, switching_ar(), target="other")
    with pytest.raises(ValueError, match="different targets"):
        consensus([a, b])


def test_consensus_disambiguates_duplicate_model_names() -> None:
    a = _consensus_result({"level": 1.0}, spectral_whittle())
    b = _consensus_result({"level": 0.0}, spectral_whittle(window=48, bin_hi=20))
    cons = consensus([a, b])
    assert cons.models == ("SpectralWhittle", "SpectralWhittle#2")
    assert cons.contested == ("level",)


def test_consensus_amp_spread_rule_is_reserved() -> None:
    a = _consensus_result({"level": 1.0}, spectral_whittle())
    b = _consensus_result({"level": 1.0}, switching_ar())
    with pytest.raises(NotImplementedError, match="vol_garch"):
        consensus([a, b]).amp_vs_spread()
