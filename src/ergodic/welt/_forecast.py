# SPDX-License-Identifier: Apache-2.0
"""Welt's hosted Chronos-2 as an ergodic forecasting model.

:class:`WeltForecaster` satisfies :class:`~ergodic.forecast.ForecastModel` (and
:class:`~ergodic.forecast.CovariateForecastModel`): ``fit`` keeps the training
window locally, ``forecast`` sends one paid ``POST /v1/forecasts`` job (DEC-085)
and turns the quantiles Welt returns into draws with
:func:`~ergodic.forecast.quantile_draws`. So the result goes through
:func:`~ergodic.forecast.backtest`, :func:`~ergodic.forecast.reconcile` and every
metric like any other forecast.

The wire mapping (DEC-087): a :class:`~ergodic.data.TimeSeriesData` is wide, one
column per series; Welt reads long tables. By default every target column becomes
one series of a panel (an id column holding the column name and one value
column), forecast independently (``cross_learning=False``). ``multivariate=True``
sends them as the targets of one series instead, forecast jointly. Covariates ride
along on every row.
"""

from __future__ import annotations

import copy
import math
import time
import uuid
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

from ..data import HierarchicalTSData, TimeSeriesData
from ..forecast import RECONCILE_METHODS, ForecastResult, quantile_draws, reconcile
from ..forecast._facade import _fill_aggregates, _hierarchy_order, _horizon_index
from ._client import _cell, using

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt
    import pandas as pd

__all__ = ["NATIVE_QUANTILE_LEVELS", "WeltForecastDiagnostics", "WeltForecaster"]

#: The 21 quantile levels Chronos-2 predicts natively (DEC-085 §1), the default
#: request. Any strictly increasing subset of them is accepted.
NATIVE_QUANTILE_LEVELS: tuple[float, ...] = (
    0.01,
    *(round(k / 20, 2) for k in range(1, 20)),
    0.99,
)

#: How the draws are made from Welt's quantiles, recorded on every result.
CONVERSION = "inverse-cdf: linear between levels, exponential tails, steps independent"

_ID = "__ergodic_series__"
_VALUE = "__ergodic_value__"
# Integer and float time indexes are sent as consecutive days from this origin:
# Welt reads ISO timestamps, and Chronos-2 reads the order of the rows, not the
# calendar.
_STEP_ORIGIN = np.datetime64("2000-01-01T00:00:00", "s")
_SECONDS_PER_DAY = 86_400
_RECONCILE_HERE = tuple(m for m in RECONCILE_METHODS if m not in ("middle_out", "none"))


@dataclass(frozen=True)
class WeltForecastDiagnostics:
    """Where one series' draws came from: the Welt job and its native quantiles.

    One record per forecast node, in :attr:`ForecastResult.diagnostics
    <ergodic.forecast.ForecastResult.diagnostics>`. ``quantiles`` holds exactly what
    Welt returned for the node, shape ``(H, K)`` against ``quantile_levels``, so
    the native numbers stay readable next to the draws made from them.

    Attributes:
        node: The forecast series.
        model: The Welt model id, for example ``"chronos-2"``.
        model_version: The exact model version Welt ran.
        configuration: The resolved configuration version, for example
            ``"chronos-2-forecast-v1"``.
        job_id: The Welt job id.
        quantile_levels: The levels requested and returned.
        quantiles: Welt's quantiles for this node, ``(H, K)``, read-only.
        context_length: The history rows Welt read for this series.
        rearranged_steps: Steps of the whole job whose native quantiles crossed
            and that Welt sorted (DEC-085 §1); zero when nothing was repaired.
        cross_learning: Whether the series shared information.
        multivariate: Whether the targets were sent as one multivariate series.
        conversion: How the draws were made, :data:`CONVERSION`.
        time_encoding: ``"timestamps"``, or ``"steps-as-days"`` for an integer
            or float time index sent as consecutive days.
    """

    node: str
    model: str
    model_version: str
    configuration: str
    job_id: str
    quantile_levels: tuple[float, ...]
    quantiles: npt.NDArray[np.float64] = field(repr=False)
    context_length: int
    rearranged_steps: int
    cross_learning: bool
    multivariate: bool
    conversion: str = CONVERSION
    time_encoding: str = "timestamps"
    point_semantics: str = "median"
    calibrated_intervals: bool = False


class _Session:
    """What a forecaster and every copy of it share: the client, budget and versions."""

    def __init__(self, client: Any, max_forecasts: int | None) -> None:
        self.client = client
        self.max_forecasts = max_forecasts
        self.used = 0
        self.metadata: dict[str, Any] | None = None
        # (model_version, configuration) of the first job, then expected of the rest.
        self.versions: tuple[str, str] | None = None

    def charge(self, model: str) -> None:
        if self.max_forecasts is not None and self.used >= self.max_forecasts:
            raise RuntimeError(
                f"WeltForecaster reached max_forecasts={self.max_forecasts} paid Welt "
                f"forecasts of {model!r} and stopped before sending another. A backtest "
                "sends one per fold; use fewer folds, or raise max_forecasts if you mean "
                "to pay for more"
            )
        self.used += 1


def _names(values: Sequence[str] | str, label: str) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    out = tuple(str(v) for v in values)
    if len(set(out)) != len(out):
        raise ValueError(f"{label} repeats a column: {list(out)}")
    return out


class WeltForecaster:
    """Welt's hosted Chronos-2 as an ergodic forecasting model.

    ``WeltForecaster().fit(data).forecast(12)`` forecasts every numeric column of
    ``data`` 12 steps ahead with one paid Welt job and returns a
    :class:`~ergodic.forecast.ForecastResult` of ``draws`` sample paths per series.
    ``backtest(WeltForecaster(), data, test_size=12)`` scores it, one paid job
    per fold. ``Chronos`` is the same class.

    Welt answers each step with quantiles (all 21 native levels by default); the
    draws come from them by inverse-CDF sampling
    (:func:`~ergodic.forecast.quantile_draws`), seeded by ``forecast``'s ``seed``.
    They match Welt's marginal at each step, but each step is sampled on its own:
    a path has no dependence across steps, and series have none between them.
    Per-step readings (median, intervals, coverage, CRPS) are unaffected; a sum
    over steps or a bottom-up aggregate reads the errors as independent. The
    native quantiles stay in each node's :class:`WeltForecastDiagnostics`.

    ``fit`` is local and free: it checks the columns and keeps the window. Every
    ``forecast`` is a paid job. The first one also reads the model's catalogue
    entry (free) and checks the request against its measured
    ``forecast_envelopes`` before paying, and the model and configuration versions
    of the first job are required of every later one (``X-Welt-Model-Version``),
    so the folds of one backtest all run the same model. The forecaster and every
    copy of it (``backtest`` copies it per fold) share one budget of
    ``max_forecasts`` jobs and one client.

    Args:
        model: The Welt forecasting model id.
        targets: The columns to forecast. By default every variable that is not a
            covariate, or every node of the hierarchy for a
            :class:`~ergodic.data.HierarchicalTSData` (an aggregate without a
            column is forecast from the sum of its children's history). On a
            hierarchy, an aggregate left out of ``targets`` is filled with the sum
            of its children's draws.
        past_covariates: Columns observed in the history only.
        future_covariates: Columns whose future values are known. ``forecast``
            then needs ``future=``; ``backtest`` supplies the held-out values.
            A categorical or ordinal column (by the dataset's schema) is sent as a
            categorical covariate, any other must be numeric.
        multivariate: ``False`` sends each target as its own series of a panel;
            ``True`` sends them as the targets of one series, forecast jointly.
            Welt prices a forecast by series length times variates, so with
            covariates the multivariate request is the cheaper one.
        cross_learning: Let the series of a panel share information. ``False``
            forecasts each as if it were sent alone.
        quantile_levels: The levels to request, a strictly increasing subset of
            :data:`NATIVE_QUANTILE_LEVELS` with at least two levels.
        draws: Draws per series and step.
        max_context: Send at most this many of the most recent rows. ``None``
            sends the whole window, and a window longer than the model's measured
            envelope is refused before paying.
        reconcile: On a hierarchy, reconcile the forecast before returning it:
            ``"bottom_up"``, ``"top_down"`` (historical shares of the training
            window) or ``"mint"`` (posterior weights). ``None`` returns the levels
            as forecast; :func:`~ergodic.forecast.reconcile` does the rest.
        client: A ``welt.Client`` to use as is, shared with every copy. ``None``
            opens a short-lived ``welt.Client()`` per forecast.
        max_forecasts: The most paid forecasts this object and its copies may
            send; ``None`` removes the cap.
        configuration: The Welt model configuration.
        timeout: Seconds to wait for each forecast job. A timeout leaves the job
            running in Welt.
        name: What a backtest report calls the model; ``"welt:<model>"`` by
            default.

    Attributes:
        forecasts_used: Paid forecasts sent so far by this object and its copies.
    """

    def __init__(
        self,
        model: str = "chronos-2",
        *,
        targets: Sequence[str] | str | None = None,
        past_covariates: Sequence[str] | str = (),
        future_covariates: Sequence[str] | str = (),
        multivariate: bool = False,
        cross_learning: bool = False,
        quantile_levels: Sequence[float] = NATIVE_QUANTILE_LEVELS,
        draws: int = 1000,
        max_context: int | None = None,
        reconcile: str | None = None,
        client: Any = None,
        max_forecasts: int | None = 10,
        configuration: str = "default",
        timeout: float = 600.0,
        name: str | None = None,
    ) -> None:
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a Welt model id such as 'chronos-2'")
        self.model = model
        self.targets = None if targets is None else _names(targets, "targets")
        self.past_covariates = _names(past_covariates, "past_covariates")
        self._future_covariates = _names(future_covariates, "future_covariates")
        overlap = sorted(set(self.past_covariates) & set(self._future_covariates))
        if overlap:
            raise ValueError(f"{overlap} are both past and future covariates; pick one")
        covariates = set(self.past_covariates) | set(self._future_covariates)
        clash = sorted(set(self.targets or ()) & covariates)
        if clash:
            raise ValueError(f"{clash} are both targets and covariates")
        reserved = sorted({_ID, _VALUE} & (covariates | set(self.targets or ())))
        if reserved:
            raise ValueError(f"{reserved} are reserved column names")
        levels = tuple(float(v) for v in quantile_levels)
        native = set(NATIVE_QUANTILE_LEVELS)
        if (
            len(levels) < 2
            or any(v not in native for v in levels)
            or any(b <= a for a, b in pairwise(levels))
        ):
            raise ValueError(
                "quantile_levels must be a strictly increasing subset of at least two of "
                f"the native levels {list(NATIVE_QUANTILE_LEVELS)}"
            )
        self.quantile_levels = levels
        if not isinstance(draws, int) or isinstance(draws, bool) or draws < 1:
            raise ValueError("draws must be a positive integer")
        if max_context is not None and (not isinstance(max_context, int) or max_context < 1):
            raise ValueError("max_context must be a positive integer or None")
        if reconcile is not None and reconcile not in _RECONCILE_HERE:
            raise ValueError(
                f"reconcile must be one of {list(_RECONCILE_HERE)} or None, got {reconcile!r}; "
                "for middle_out or options, call ergodic.forecast.reconcile on the result"
            )
        if max_forecasts is not None and (not isinstance(max_forecasts, int) or max_forecasts < 1):
            raise ValueError("max_forecasts must be a positive integer or None")
        if not (isinstance(timeout, (int, float)) and math.isfinite(timeout) and timeout > 0):
            raise ValueError("timeout must be a positive number of seconds")
        self.multivariate = bool(multivariate)
        self.cross_learning = bool(cross_learning)
        self.draws = draws
        self.max_context = max_context
        self.reconcile = reconcile
        self.max_forecasts = max_forecasts
        self.configuration = configuration
        self.timeout = float(timeout)
        self.name = name or f"welt:{model}"
        self._session = _Session(client, max_forecasts)
        self._data: TimeSeriesData | None = None

    # -- the protocol --------------------------------------------------------

    @property
    def future_covariates(self) -> tuple[str, ...]:
        """The covariates whose future values ``forecast`` reads from ``future=``."""
        return self._future_covariates

    @property
    def client(self) -> Any:
        """The ``welt.Client`` this forecaster and its copies use, or ``None``."""
        return self._session.client

    @property
    def forecasts_used(self) -> int:
        """Paid forecasts sent so far by this forecaster and every copy of it."""
        return self._session.used

    @property
    def fitted(self) -> bool:
        """True once :meth:`fit` has run."""
        return self._data is not None

    def __deepcopy__(self, memo: dict[int, Any]) -> WeltForecaster:
        """A copy that shares the client, the budget and the pinned versions.

        ``backtest`` deep-copies the model per fold. Sharing the session is what
        makes ``max_forecasts`` cap the whole backtest and keeps every fold on
        one model version; the data is immutable, so it is shared too.
        """
        twin = copy.copy(self)
        memo[id(self)] = twin
        return twin

    def fit(self, data: TimeSeriesData, *, seed: int = 0) -> WeltForecaster:
        """Check the columns and keep the training window; nothing is sent.

        Args:
            data: The history, a :class:`~ergodic.data.TimeSeriesData` or
                :class:`~ergodic.data.HierarchicalTSData` with a datetime or an
                evenly spaced numeric time index.
            seed: Accepted for the protocol. Welt's forecast is deterministic, so
                only :meth:`forecast`'s seed (for the draws) matters.

        Returns:
            A fitted copy; this object is unchanged.

        Raises:
            TypeError: ``data`` is not a time series.
            ValueError: A named column is missing, a target is not numeric, a
                future covariate has missing values, or the time index is not
                usable.
        """
        if not isinstance(data, TimeSeriesData):
            raise TypeError(
                f"WeltForecaster fits a TimeSeriesData (or a HierarchicalTSData), got "
                f"{type(data).__name__}"
            )
        if self.reconcile is not None and not isinstance(data, HierarchicalTSData):
            raise ValueError("reconcile= needs a HierarchicalTSData")
        if data.n_rows < 2:
            raise ValueError("WeltForecaster needs at least two rows of history")
        variables = set(data.variables)
        covariates = (*self.past_covariates, *self._future_covariates)
        missing = sorted(set(covariates) - variables)
        if missing:
            raise ValueError(f"covariates {missing} are not variables of the data")
        targets = self._targets(data, set(covariates))
        for name in targets:
            if name in variables and not data.schema[name].is_continuous:
                raise ValueError(
                    f"target {name!r} is {data.schema[name].kind}; Welt forecasts numbers"
                )
        for name in self._future_covariates:
            if _has_missing(data.column(name)):
                raise ValueError(
                    f"future covariate {name!r} has missing values; Welt needs them complete"
                )
        _encode_times(data.times, data)  # raises when the index cannot be sent
        fitted = copy.copy(self)
        fitted._data = data
        return fitted

    def forecast(
        self,
        horizon: int,
        *,
        seed: int = 0,
        future: TimeSeriesData | pd.DataFrame | None = None,
    ) -> ForecastResult:
        """Forecast ``horizon`` steps past the training window: one paid Welt job.

        Args:
            horizon: Steps to forecast.
            seed: Seeds the draws made from Welt's quantiles.
            future: The future covariates over the horizon, ``horizon`` rows with
                the time column and every name in ``future_covariates``; only those
                columns are read. Required exactly when there are future
                covariates. A :class:`~ergodic.data.TimeSeriesData` or a dataframe.

        Returns:
            A :class:`~ergodic.forecast.ForecastResult` of ``(draws, horizon)``
            draws per target, ``kind="draws"``, with one
            :class:`WeltForecastDiagnostics` per node in ``diagnostics``.

        Raises:
            ValueError: Not fitted, a bad ``horizon`` or ``future``, or a request
                outside the model's measured envelopes (nothing is sent).
            RuntimeError: ``max_forecasts`` is spent, or the Welt job failed.
            TimeoutError: The job did not finish within ``timeout``; it keeps
                running in Welt.
        """
        data = self._data
        if data is None:
            raise ValueError("this WeltForecaster is not fitted; call fit(data) first")
        if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
        horizon = int(horizon)
        if self.max_context is not None and data.n_rows > self.max_context:
            data = data.take(np.arange(data.n_rows - self.max_context, data.n_rows, dtype=np.int64))
        future_rows = self._future_rows(data, horizon, future)
        targets = self._targets(data, {*self.past_covariates, *self._future_covariates})
        history = {name: _history(data, name) for name in targets}
        payload, time_encoding = self._payload(data, horizon, targets, history, future_rows)

        session = self._session
        with using(session.client) as client:
            meta = session.metadata
            if meta is None:
                meta = dict(client.model(self.model))
                session.metadata = meta
            n_series = 1 if self.multivariate else len(targets)
            variates = (len(targets) if self.multivariate else 1) + len(self.past_covariates)
            variates += len(self._future_covariates)
            _admit(
                meta,
                self.model,
                context=data.n_rows,
                variates=variates,
                series=n_series,
                horizon=horizon,
            )
            session.charge(self.model)
            job_id, result = self._run(client, payload)

        series = self._read(result, targets, horizon, job_id, data.n_rows, time_encoding)
        draws: dict[str, npt.NDArray[np.float64]] = {}
        rng = np.random.default_rng(seed)
        for node in targets:
            draws[node] = quantile_draws(
                self.quantile_levels, series[node].quantiles, draws=self.draws, seed=rng
            )
        return self._result(data, horizon, draws, series)

    # -- request -------------------------------------------------------------

    def _targets(self, data: TimeSeriesData, covariates: set[str]) -> tuple[str, ...]:
        if isinstance(data, HierarchicalTSData):
            tree = data.hierarchy
            if self.targets is None:
                return _hierarchy_order(tree, ())
            unknown = sorted(
                n for n in self.targets if n not in tree.nodes and n not in data.variables
            )
        else:
            if self.targets is None:
                chosen = tuple(n for n in data.variables if n not in covariates)
                if not chosen:
                    raise ValueError("no target left to forecast: every variable is a covariate")
                return chosen
            unknown = sorted(n for n in self.targets if n not in data.variables)
        if unknown:
            raise ValueError(f"targets {unknown} are not variables of the data")
        return self.targets

    def _future_rows(
        self, data: TimeSeriesData, horizon: int, future: TimeSeriesData | pd.DataFrame | None
    ) -> dict[str, list[Any]] | None:
        """The future table's columns (time plus future covariates), checked."""
        names = self._future_covariates
        if not names:
            if future is not None:
                raise ValueError("future= was given but this forecaster has no future_covariates")
            return None
        if future is None:
            raise ValueError(
                f"forecast needs future= with the values of {list(names)} over the horizon"
            )
        if isinstance(future, TimeSeriesData):
            columns = {
                name: np.asarray(future.column(name)) for name in names if name in future.schema
            }
            times = np.asarray(future.times) if future.time == data.time else None
        else:
            frame = future.sort_values(data.time) if data.time in future.columns else future
            columns = {name: frame[name].to_numpy() for name in names if name in frame.columns}
            times = frame[data.time].to_numpy() if data.time in frame.columns else None
        absent = sorted(set(names) - set(columns))
        if absent:
            raise ValueError(f"future= lacks the future covariates {absent}")
        if times is None:
            raise ValueError(f"future= needs the time column {data.time!r}")
        if len(times) != horizon:
            raise ValueError(f"future= has {len(times)} rows; the horizon is {horizon}")
        for name, values in columns.items():
            if _has_missing(values):
                raise ValueError(f"future covariate {name!r} has missing values in future=")
        encoded = _encode_times(np.asarray(times), data, future=True)
        return {"__time__": encoded, **{name: list(values) for name, values in columns.items()}}

    def _payload(
        self,
        data: TimeSeriesData,
        horizon: int,
        targets: tuple[str, ...],
        history: Mapping[str, npt.NDArray[np.float64]],
        future_rows: Mapping[str, list[Any]] | None,
    ) -> tuple[dict[str, Any], str]:
        time_name = data.time
        stamps = _encode_times(data.times, data)
        encoding = "timestamps" if np.asarray(data.times).dtype.kind == "M" else "steps-as-days"
        past = {name: data.column(name) for name in self.past_covariates}
        known = {name: data.column(name) for name in self._future_covariates}
        covariate_names = [*self.past_covariates, *self._future_covariates]
        covariates = {**past, **known}

        if self.multivariate:
            columns = [time_name, *targets, *covariate_names]
            rows = [
                [
                    stamps[t],
                    *(_cell(history[name][t]) for name in targets),
                    *(_cell(covariates[name][t]) for name in covariate_names),
                ]
                for t in range(data.n_rows)
            ]
        else:
            columns = [time_name, _ID, _VALUE, *covariate_names]
            rows = [
                [
                    stamps[t],
                    node,
                    _cell(history[node][t]),
                    *(_cell(covariates[name][t]) for name in covariate_names),
                ]
                for node in targets
                for t in range(data.n_rows)
            ]
        categorical = [name for name in covariate_names if data.schema[name].is_discrete]
        payload: dict[str, Any] = {
            "model_id": self.model,
            "configuration": self.configuration,
            "history": {"table": {"columns": columns, "rows": rows}},
            "time_column": time_name,
            "targets": list(targets) if self.multivariate else [_VALUE],
            "past_covariates": list(self.past_covariates),
            "future_covariates": list(self._future_covariates),
            "categorical_covariates": categorical,
            "horizon": horizon,
            "quantile_levels": list(self.quantile_levels),
            "cross_learning": self.cross_learning,
        }
        if not self.multivariate:
            payload["id_column"] = _ID
        frequency = _frequency(data) if encoding == "timestamps" else {"unit": "day", "every": 1}
        if frequency is not None:
            payload["frequency"] = frequency
        if future_rows is not None:
            names = list(self._future_covariates)
            ids = [None] if self.multivariate else list(targets)
            future_columns = [time_name, *names] if self.multivariate else [time_name, _ID, *names]
            future_table = [
                [
                    future_rows["__time__"][step],
                    *([] if ident is None else [ident]),
                    *(_cell(future_rows[name][step]) for name in names),
                ]
                for ident in ids
                for step in range(horizon)
            ]
            payload["future"] = {"table": {"columns": future_columns, "rows": future_table}}
        return payload, encoding

    def _run(self, client: Any, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Submit one forecast job and wait for its result."""
        headers = {"Idempotency-Key": uuid.uuid4().hex}
        pinned = self._session.versions
        if pinned is not None:
            headers["X-Welt-Model-Version"] = pinned[0]
            headers["X-Welt-Configuration-Version"] = pinned[1]
        accepted = client.request("POST", "/v1/forecasts", json=payload, headers=headers)
        job_id = str(accepted["id"])
        deadline = time.monotonic() + self.timeout
        interval = 0.25
        while True:
            job = client.job(job_id).inspect()
            status = job.get("status")
            if status == "succeeded":
                break
            if status in ("failed", "cancelled"):
                error = job.get("error") or {}
                raise RuntimeError(
                    f"Welt forecast job {job_id!r} {status}: "
                    f"{error.get('message', 'no message')} ({error.get('code', 'no code')})"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Welt forecast job {job_id!r} did not finish within {self.timeout:g} s; it "
                    f"keeps running in Welt. Read it later with welt.Client().job({job_id!r}) "
                    "rather than forecasting again"
                )
            time.sleep(min(interval, remaining))
            interval = min(interval * 1.4, 5.0)
        result: dict[str, Any] = client.job_result(job_id)
        if pinned is None:
            self._session.versions = (str(result["model_version"]), str(result["configuration"]))
        return job_id, result

    # -- result --------------------------------------------------------------

    def _read(
        self,
        result: Mapping[str, Any],
        targets: tuple[str, ...],
        horizon: int,
        job_id: str,
        context: int,
        time_encoding: str,
    ) -> dict[str, WeltForecastDiagnostics]:
        """Check Welt's result against the request and pick out each node's quantiles."""

        def bad(reason: str) -> ValueError:
            return ValueError(
                f"Welt forecast job {job_id!r} returned a result ergodic cannot read: {reason}. "
                f"The result stays in Welt (welt.Client().job_result({job_id!r}))"
            )

        if result.get("task") != "forecasting":
            raise bad(f"task {result.get('task')!r}, expected 'forecasting'")
        if int(result.get("horizon", -1)) != horizon:
            raise bad(f"horizon {result.get('horizon')!r}, expected {horizon}")
        levels = tuple(float(v) for v in result.get("quantile_levels") or ())
        if levels != self.quantile_levels:
            raise bad(f"quantile levels {list(levels)}, expected {list(self.quantile_levels)}")
        found: dict[str, Mapping[str, Any]] = {}
        for entry in result.get("series") or ():
            key = str(entry.get("target")) if self.multivariate else str(entry.get("id"))
            if key in found:
                raise bad(f"series {key!r} appears twice")
            found[key] = entry
        missing = [node for node in targets if node not in found]
        if missing or len(found) != len(targets):
            raise bad(f"series {sorted(found)}, expected {list(targets)}")
        out: dict[str, WeltForecastDiagnostics] = {}
        for node in targets:
            quantiles = np.asarray(found[node].get("quantiles"), dtype=np.float64)
            if quantiles.shape != (horizon, len(levels)) or not bool(
                np.all(np.isfinite(quantiles))
            ):
                raise bad(f"series {node!r} has quantiles of shape {quantiles.shape}")
            quantiles.setflags(write=False)
            out[node] = WeltForecastDiagnostics(
                node=node,
                model=str(result.get("model_id", self.model)),
                model_version=str(result.get("model_version")),
                configuration=str(result.get("configuration")),
                job_id=job_id,
                quantile_levels=levels,
                quantiles=quantiles,
                context_length=int(found[node].get("context_length", context)),
                rearranged_steps=int(result.get("rearranged_steps", 0)),
                cross_learning=bool(result.get("cross_learning", self.cross_learning)),
                multivariate=self.multivariate,
                time_encoding=time_encoding,
            )
        return out

    def _result(
        self,
        data: TimeSeriesData,
        horizon: int,
        draws: dict[str, npt.NDArray[np.float64]],
        diagnostics: dict[str, WeltForecastDiagnostics],
    ) -> ForecastResult:
        index = _horizon_index(data, horizon)
        start = index[0] if index else None
        if not isinstance(data, HierarchicalTSData):
            return ForecastResult(
                series=draws,
                horizon=horizon,
                kind="draws",
                start=start,
                index=index,
                diagnostics=diagnostics,
            )
        tree = data.hierarchy
        if all(leaf in draws for leaf in tree.leaves):
            _fill_aggregates(tree, draws)
        ordered = _hierarchy_order(tree, tuple(draws))
        raw = ForecastResult(
            series={node: draws[node] for node in ordered if node in draws},
            horizon=horizon,
            kind="draws",
            start=start,
            index=index,
            diagnostics=diagnostics,
        )
        if self.reconcile is None:
            return raw
        return reconcile(
            raw,
            tree,
            method=self.reconcile,
            data=data if self.reconcile == "top_down" else None,
        )

    def __repr__(self) -> str:
        parts = [repr(self.model)]
        if self.targets is not None:
            parts.append(f"targets={list(self.targets)}")
        if self.past_covariates:
            parts.append(f"past_covariates={list(self.past_covariates)}")
        if self._future_covariates:
            parts.append(f"future_covariates={list(self._future_covariates)}")
        if self.multivariate:
            parts.append("multivariate=True")
        if self.reconcile:
            parts.append(f"reconcile={self.reconcile!r}")
        state = "fitted" if self.fitted else "unfitted"
        return f"WeltForecaster({', '.join(parts)}; {state})"


# ``Chronos`` is WeltForecaster under the name of the model it runs by default
# (DEC-085 §9): ``Chronos()`` is ``WeltForecaster("chronos-2")``. Importable from
# ergodic.welt but left out of ``__all__``, so the reference documents the class once.
Chronos = WeltForecaster


# -- helpers -------------------------------------------------------------------


def _has_missing(values: Any) -> bool:
    array = np.asarray(values)
    if array.dtype.kind == "f":
        return bool(np.any(np.isnan(array)))
    if array.dtype.kind == "M":
        return bool(np.any(np.isnat(array)))
    if array.dtype.kind == "O":
        return any(v is None or (isinstance(v, float) and math.isnan(v)) for v in array.tolist())
    return False


def _history(data: TimeSeriesData, node: str) -> npt.NDArray[np.float64]:
    """One target's history; a hierarchy aggregate without a column is built from below."""
    if node in data.variables:
        return np.asarray(data.column(node), dtype=np.float64)
    if isinstance(data, HierarchicalTSData):
        return data.aggregate(node)
    raise ValueError(f"target {node!r} is not a variable of the data")  # pragma: no cover


def _encode_times(times: Any, data: TimeSeriesData, *, future: bool = False) -> list[str]:
    """ISO-8601 UTC timestamps for Welt, or consecutive days for a numeric index.

    A datetime index is sent as is, read as UTC. A numeric index must be evenly
    spaced; step ``k`` (counted from the start of the training window) is sent as
    day ``k`` from 2000-01-01, and a future window must continue it exactly.
    """
    history = np.asarray(data.times)
    values = np.asarray(times)
    if history.dtype.kind == "M":
        if values.dtype.kind != "M":
            raise ValueError("future= times must be datetimes, like the training index")
        if future and values.size and values[0] <= history[-1]:
            raise ValueError("future= must start after the end of the training window")
        if bool(np.any(np.isnat(values))):
            raise ValueError("the time index has missing values")
        seconds = values.astype("datetime64[s]")
        return [f"{stamp}Z" for stamp in np.datetime_as_string(seconds, unit="s")]
    if history.dtype.kind not in "iuf":
        raise ValueError(
            f"Welt reads a datetime or numeric time index; {data.time!r} is {history.dtype}"
        )
    steps = np.diff(history.astype(np.float64))
    if history.size < 2 or not bool(np.all(steps == steps[0])) or steps[0] <= 0:
        raise ValueError(
            f"the numeric time index {data.time!r} must be evenly spaced and increasing; "
            "use a datetime index for anything else"
        )
    step = float(steps[0])
    offsets = (values.astype(np.float64) - float(history[0])) / step
    if not bool(np.all(offsets == np.round(offsets))):
        raise ValueError("future= times must continue the training index's spacing")
    if future:
        expected = np.arange(history.size, history.size + values.size, dtype=np.float64)
        if not np.array_equal(offsets, expected):
            raise ValueError("future= times must be the steps right after the training window")
    days = _STEP_ORIGIN + offsets.astype(np.int64) * np.timedelta64(_SECONDS_PER_DAY, "s")
    return [f"{stamp}Z" for stamp in np.datetime_as_string(days, unit="s")]


_UNITS = {
    "Day": "day",
    "Hour": "hour",
    "Minute": "minute",
    "Second": "second",
    "Week": "week",
    "MonthEnd": "month",
    "MonthBegin": "month",
    "QuarterEnd": "quarter",
    "QuarterBegin": "quarter",
    "YearEnd": "year",
    "YearBegin": "year",
}


def _frequency(data: TimeSeriesData) -> dict[str, Any] | None:
    """Welt's ``frequency`` for the dataset's declared pandas ``freq``, or ``None``.

    ``None`` leaves Welt to infer it from the timestamps, which it does when they
    are regular.
    """
    if not data.freq:
        return None
    import pandas as pd

    try:
        offset = pd.tseries.frequencies.to_offset(data.freq)
    except (TypeError, ValueError):
        return None
    unit = _UNITS.get(type(offset).__name__)
    if unit is None or int(offset.n) < 1:
        return None
    return {"unit": unit, "every": int(offset.n)}


def _admit(
    meta: Mapping[str, Any], model: str, *, context: int, variates: int, series: int, horizon: int
) -> None:
    """Refuse, before paying, a model that does not forecast or a request it cannot take.

    The catalogue publishes measured corners (DEC-085 §3); a request is admitted
    when some corner dominates it. A catalogue entry without envelopes is left to
    Welt to judge.
    """
    operations = meta.get("operations")
    if isinstance(operations, list) and "forecast" not in operations:
        raise ValueError(
            f"Welt model {model!r} does not serve forecasts (its operations are {operations}); "
            "nothing was sent. List the forecasting models with welt.Client().models()"
        )
    envelopes = meta.get("forecast_envelopes")
    if isinstance(envelopes, dict):
        corners = list(envelopes.get("corners") or ())
        floor = envelopes.get("min_context", meta.get("min_context", 16))
    else:
        corners = list(envelopes or ())
        floor = meta.get("min_context", 16)
    if not corners:
        return
    asked = {"context": context, "variates": variates, "series": series, "horizon": horizon}
    for corner in corners:
        low = int(corner.get("min_context", floor))
        if (
            low <= context <= int(corner["max_context"])
            and variates <= int(corner["max_variates"])
            and series <= int(corner["max_series"])
            and horizon <= int(corner["max_horizon"])
        ):
            return
    measured = [
        {k: corner[k] for k in ("max_context", "max_variates", "max_series", "max_horizon")}
        for corner in corners
    ]
    raise ValueError(
        f"no measured envelope of {model!r} covers this forecast {asked}; nothing was sent. "
        f"Measured corners: {measured} (minimum context {floor}). Shorten the history with "
        "max_context=, forecast fewer series or covariates per job, or use multivariate=True "
        "for fewer series"
    )
