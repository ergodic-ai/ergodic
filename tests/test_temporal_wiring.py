# SPDX-License-Identifier: Apache-2.0
"""Facade wiring for the temporal methods: exports, dispatch, and shape guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ergodic
from ergodic.data import panel, tabular, timeseries
from ergodic.discovery import PCMCI, TimeSeriesDiscoveryResult, bootstrap, discover, granger
from ergodic.inference import Design, DiDEffect


def _ts_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.normal()
        y[t] = 0.8 * x[t - 1] + 0.5 * y[t - 1] + rng.normal()
    return pd.DataFrame({"t": np.arange(n), "X": x, "Y": y})


def test_top_level_exports() -> None:
    for name in ("did", "event_study", "synthetic_control", "interrupted_time_series", "granger"):
        assert callable(getattr(ergodic, name))


def test_discover_dispatches_pcmci() -> None:
    data = timeseries(_ts_frame(), time="t")
    result = discover(data, method="pcmci", max_lag=1, alpha=0.01)
    assert isinstance(result, TimeSeriesDiscoveryResult)
    assert result.method == "pcmci"
    assert ("X", "Y", 1) in result.links
    assert ("X", "Y") in result.summary_edges()


def test_iid_methods_reject_temporal_data() -> None:
    data = timeseries(_ts_frame(), time="t")
    with pytest.raises(TypeError, match="not exchangeable"):
        discover(data, method="pc")
    with pytest.raises(TypeError, match="not exchangeable"):
        discover(data, method="ges")


def test_pcmci_rejects_tabular_data() -> None:
    frame = _ts_frame().drop(columns="t")
    with pytest.raises(TypeError):
        discover(tabular(frame), method="pcmci", max_lag=1)


def test_pcmci_direct_class_matches_facade() -> None:
    data = timeseries(_ts_frame(), time="t")
    direct = PCMCI(alpha=0.01, max_lag=1).fit(data)
    via_facade = discover(data, method="pcmci", max_lag=1, alpha=0.01)
    assert isinstance(via_facade, TimeSeriesDiscoveryResult)
    assert direct.links == via_facade.links
    assert direct.graph == via_facade.graph


def test_bootstrap_refuses_temporal_methods() -> None:
    # The guard fires on the method name, before any rows are touched.
    data = tabular(_ts_frame().drop(columns="t"))
    with pytest.raises(ValueError, match="moving-block"):
        bootstrap(data, method="pcmci", n_bootstrap=5)


def test_granger_via_top_level() -> None:
    data = timeseries(_ts_frame(), time="t")
    result = granger(data, max_lag=1, alpha=0.01)
    assert ("X", "Y") in result.significant
    assert ("Y", "X") not in result.significant


def test_did_via_top_level_with_design() -> None:
    rng = np.random.default_rng(0)
    units = np.repeat(np.arange(60), 8)
    periods = np.tile(np.arange(8), 60)
    treated_unit = units < 20
    d = (treated_unit & (periods >= 4)).astype(float)
    alpha_unit = rng.normal(size=60)[units] + 1.5 * treated_unit
    y = alpha_unit + 0.3 * periods + 2.0 * d + rng.normal(scale=0.5, size=units.size)
    frame = pd.DataFrame({"unit": units, "week": periods, "promo": d, "sales": y})
    data = panel(frame, entity="unit", time="week")
    est = ergodic.did(data, "sales", "promo", n_boot=100, random_state=0)
    assert isinstance(est, DiDEffect)
    assert isinstance(est.design, Design)
    assert est.design.strategy == "did"
    assert abs(est.value - 2.0) < 0.3
    low, high = est.interval()
    assert low < 2.0 < high
