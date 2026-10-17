# SPDX-License-Identifier: Apache-2.0
"""Run Welt's hosted foundation models from ergodic.

`Welt <https://welt.ergodic.dev>`_ hosts foundation models for tabular data and
time series. This optional module connects them to ergodic's discovery,
estimation and forecasting:

- :func:`discover` runs a Welt causal discovery model (``arrow``, ``cdfm``,
  ``avici``) and returns a :class:`~ergodic.discovery.DiscoveryResult`. The same
  run is ``ergodic.discover(data, method="welt:arrow")``.
- :class:`WeltLearner` fills an estimator's nuisance slots with a Welt tabular
  model, so ``estimate_effect(..., learner=WeltLearner("tabicl-v2"))`` works.
- :class:`WeltForecaster` (also importable as ``Chronos``) forecasts a
  :class:`~ergodic.data.TimeSeriesData` with Welt's Chronos-2, covariates
  included, as an ergodic :class:`~ergodic.forecast.ForecastModel`, so
  ``backtest(WeltForecaster(), data, test_size=12)`` and
  :func:`~ergodic.forecast.reconcile` work on it.
- :func:`upload_dataset` sends a :class:`~ergodic.data.TabularData` to your Welt
  workspace, and :func:`load_research_dataset` reads a Welt research dataset as
  one.

Install it with ``pip install "ergodic[welt]"``, which adds ``welt-client`` on
Python 3.11 to 3.13 (``welt-client`` does not support Python 3.14 yet). Every
call here is a request to Welt and most are paid; nothing runs at import. The
rest of ergodic never imports this module, ``welt`` or ``httpx``, and makes no
network calls.

Each function takes an optional ``client``: a ``welt.Client`` you opened (it is
used as is and never closed here). Without one, a short-lived ``welt.Client()``
is opened per call, reading the key from ``WELT_API_KEY`` or ``welt login``.
"""

from __future__ import annotations

try:
    import welt as _welt  # noqa: F401
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "ergodic.welt needs welt-client: pip install 'ergodic[welt]' (Python 3.11 to 3.13)"
    ) from error

from ._data import load_research_dataset, upload_dataset
from ._discovery import WeltDecodeError, WeltGraphNotDAGError, discover
from ._forecast import NATIVE_QUANTILE_LEVELS, WeltForecastDiagnostics, WeltForecaster
from ._forecast import Chronos as Chronos  # the same class; see WeltForecaster
from ._learner import WeltLearner

__all__ = [
    "NATIVE_QUANTILE_LEVELS",
    "WeltDecodeError",
    "WeltForecastDiagnostics",
    "WeltForecaster",
    "WeltGraphNotDAGError",
    "WeltLearner",
    "discover",
    "load_research_dataset",
    "upload_dataset",
]
