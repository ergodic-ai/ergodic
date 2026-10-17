# SPDX-License-Identifier: Apache-2.0
"""Welt's hosted Chronos-2 as an ergodic forecasting model.

:class:`WeltForecaster` satisfies :class:`~ergodic.forecast.ForecastModel` (and
:class:`~ergodic.forecast.CovariateForecastModel`): ``fit`` keeps the training
window locally, ``forecast`` sends one paid ``POST /v1/forecasts`` job (DEC-085)
through ``welt.Client.submit_forecast``, waits on it with the SDK's ``Job.result``,
and turns the quantiles Welt returns into draws with
:func:`~ergodic.forecast.quantile_draws`. So the result goes through
:func:`~ergodic.forecast.backtest`, :func:`~ergodic.forecast.reconcile` and every
metric like any other forecast.

The wire mapping (DEC-087): a :class:`~ergodic.data.TimeSeriesData` is wide, one
column per series; Welt reads long tables. By default every target column becomes
one series of a panel (an id column holding the column name and one value
column), forecast independently (``cross_learning=False``). ``multivariate=True``
sends them as the targets of one series instead, forecast jointly. Covariates ride
along on every row. Both tables always travel inline in the request, as
``welt.Forecast`` sends them; a request over the SDK's
``welt.forecast.MAX_REQUEST_BYTES`` (the API's 10 MiB body limit) is refused
before anything is sent.
"""

from __future__ import annotations

import copy
import json
import math
import uuid
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

from ..data import HierarchicalTSData, TimeSeriesData
from ..forecast import RECONCILE_METHODS, ForecastResult, quantile_draws, reconcile
from ..forecast._facade import _fill_aggregates, _hierarchy_order, _horizon_index
from ._client import using

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

# The API's request body limit, used when welt-client does not state its own.
_DEFAULT_MAX_REQUEST_BYTES = 10 * 1024 * 1024
_DAY = {"unit": "day", "every": 1}

_ID = "__ergodic_series__"
_VALUE = "__ergodic_value__"
# A numeric index, or a datetime index on a business or custom calendar, is sent as
# consecutive days from this origin: Welt reads ISO timestamps, and Chronos-2 reads
# the order of the rows, not the calendar.
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
        configuration: The resolved configuration version Welt ran, as the job
            result reports it.
        job_id: The Welt job id.
        quantile_levels: The levels requested and returned.
        quantiles: Welt's quantiles for this node, ``(H, K)``, read-only.
        context_length: The history rows Welt read for this series.
        rearranged_steps: Steps of the whole job whose native quantiles crossed
            and that Welt sorted (DEC-085 §1); zero when nothing was repaired.
        cross_learning: Whether the series shared information.
        multivariate: Whether the targets were sent as one multivariate series.
        conversion: How the draws were made: inverse-CDF sampling, linear between
            levels, exponential tails, each step sampled on its own.
        time_encoding: ``"timestamps"``, or ``"steps-as-days"`` when the rows were
            sent as consecutive days from 2000-01-01 (a numeric index, or a
            datetime index on a business or custom calendar).
        point_semantics: Always ``"median"``: Welt's point forecast is the 0.5
            quantile.
        calibrated_intervals: Always ``False``: Chronos-2's quantiles are not
            calibrated. Check coverage with a backtest.
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

    def check(self, model: str) -> None:
        """Refuse before sending when the budget of accepted jobs is spent."""
        if self.max_forecasts is not None and self.used >= self.max_forecasts:
            raise RuntimeError(
                f"WeltForecaster reached max_forecasts={self.max_forecasts} paid Welt "
                f"forecasts of {model!r} and stopped before sending another. A backtest "
                "sends one per fold; use fewer folds, or raise max_forecasts if you mean "
                "to pay for more"
            )


def _names(values: Sequence[str] | str, label: str) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    out = tuple(str(v) for v in values)
    if len(set(out)) != len(out):
        raise ValueError(f"{label} repeats a column: {list(out)}")
    return out


class WeltForecaster:
    """Welt's hosted Chronos-2 as an ergodic forecasting model.

    ``WeltForecaster().fit(data).forecast(12)`` forecasts every variable of
    ``data`` that is not a covariate (each must be continuous) 12 steps ahead with
    one paid Welt job and returns a
    :class:`~ergodic.forecast.ForecastResult` of ``draws`` sample paths per series.
    ``backtest(WeltForecaster(), data, test_size=12)`` scores it, one paid job
    per fold. ``Chronos`` is the same class.

    Welt answers each step with quantiles (all 21 native levels by default); the
    draws come from them by inverse-CDF sampling
    (:func:`~ergodic.forecast.quantile_draws`), seeded by ``forecast``'s ``seed``.
    They match Welt's marginal at each step, but each step is sampled on its own:
    a path has no dependence across steps, and series have none between them.
    Each unreconciled node's per-step readings (median, intervals, coverage,
    CRPS) follow Welt's quantiles. A sum over steps, a bottom-up aggregate, and
    the intervals, coverage and CRPS of a ``bottom_up`` or ``mint`` reconciliation
    all treat the errors as independent across steps or series. The native
    quantiles stay in each node's :class:`WeltForecastDiagnostics`.

    ``fit`` is local and free: it checks the columns and keeps the window. Every
    ``forecast`` is a paid job. The first one also reads the model's catalogue
    entry (free) and checks the request against its measured
    ``forecast_envelopes`` before paying, and the model and configuration versions
    of the first job are required of every later one (``X-Welt-Model-Version``),
    so the folds of one backtest all run the same model. The forecaster and every
    copy of it (``backtest`` copies it per fold) share one budget of
    ``max_forecasts`` jobs and one client. The budget counts jobs Welt accepted; a
    request it refuses (for example a 422) costs nothing and is not counted.

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
            categorical covariate, its cells as strings or integers (``True``
            becomes ``1``, ``2.0`` becomes ``2``); any other covariate must be
            continuous and is sent as numbers (a boolean as 0.0 or 1.0).
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
            as forecast; :func:`~ergodic.forecast.reconcile` does the rest. The
            draws carry no dependence between series, so ``bottom_up`` and
            ``mint`` intervals (and the coverage and CRPS a backtest reads from
            them) treat the series as independent; ``top_down`` splits the root's
            draws by fixed shares and does not depend on it.
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
        forecasts_used: Forecast jobs Welt accepted so far from this object and
            its copies.
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
        """Forecast jobs Welt accepted so far from this forecaster and every copy of it."""
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
                evenly spaced numeric time index (spacing compared with a relative
                tolerance of 1e-9).
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
        clash = sorted(set(targets) & set(covariates))
        if clash:
            raise ValueError(
                f"{clash} are nodes of the hierarchy, forecast by default, and covariates; "
                "pass targets= without them"
            )
        for name in covariates:
            kind = data.schema[name]
            if not (kind.is_continuous or kind.is_discrete):
                raise ValueError(
                    f"covariate {name!r} is {kind.kind}; Welt takes numbers or categories"
                )
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
        _time_plan(data)  # raises when the index cannot be sent
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
            RuntimeError: ``max_forecasts`` is spent, or welt-client is older
                than 0.14.
            welt.WeltError: Welt refused the job, or it failed
                (``welt.ExecutionError``) or was cancelled.
            TimeoutError: The job did not finish within ``timeout``
                (``welt.JobTimeoutError``); it keeps running in Welt.
        """
        data = self._data
        if data is None:
            raise ValueError("this WeltForecaster is not fitted; call fit(data) first")
        if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
        horizon = int(horizon)
        if self.max_context is not None and data.n_rows > self.max_context:
            data = data.take(np.arange(data.n_rows - self.max_context, data.n_rows, dtype=np.int64))
        encoding, frequency = _time_plan(data)
        future_rows = self._future_rows(data, horizon, future, encoding)
        targets = self._targets(data, {*self.past_covariates, *self._future_covariates})
        history = {name: _history(data, name) for name in targets}
        payload = self._payload(data, horizon, targets, history, future_rows, encoding, frequency)
        size, limit = _encoded_bytes(payload), _max_request_bytes()
        if size > limit:
            raise ValueError(
                f"the forecast request would be {size:,} bytes of JSON, over the Welt API's "
                f"{limit:,}-byte request limit; nothing was sent. Send fewer rows "
                "(max_context=), fewer series per job, or multivariate=True, which sends "
                "the covariates once"
            )

        session = self._session
        with using(session.client) as client:
            if not hasattr(client, "submit_forecast"):
                raise RuntimeError(
                    "WeltForecaster needs welt-client 0.14 or later (Client.submit_forecast): "
                    'pip install -U "ergodic[welt]"'
                )
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
            session.check(self.model)
            job_id, result = self._run(client, payload)

        series = self._read(result, targets, horizon, job_id, data.n_rows, encoding)
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
        self,
        data: TimeSeriesData,
        horizon: int,
        future: TimeSeriesData | pd.DataFrame | None,
        encoding: str,
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
        encoded = _encode_times(np.asarray(times), data, encoding, future=True)
        return {"__time__": encoded, **{name: list(values) for name, values in columns.items()}}

    def _payload(
        self,
        data: TimeSeriesData,
        horizon: int,
        targets: tuple[str, ...],
        history: Mapping[str, npt.NDArray[np.float64]],
        future_rows: Mapping[str, list[Any]] | None,
        encoding: str,
        frequency: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        time_name = data.time
        stamps = _encode_times(data.times, data, encoding)
        covariate_names = [*self.past_covariates, *self._future_covariates]
        categorical = [name for name in covariate_names if data.schema[name].is_discrete]
        # Cells by role, as the API validates them: categorical covariates as strings
        # or integers, everything else as finite numbers (a bool is neither for Welt).
        cast = {name: _category if name in categorical else _number for name in covariate_names}
        covariates = {
            name: [cast[name](v, name) for v in data.column(name).tolist()]
            for name in covariate_names
        }
        values = {node: [_number(v, node) for v in history[node].tolist()] for node in targets}

        if self.multivariate:
            columns = [time_name, *targets, *covariate_names]
            rows = [
                [
                    stamps[t],
                    *(values[name][t] for name in targets),
                    *(covariates[name][t] for name in covariate_names),
                ]
                for t in range(data.n_rows)
            ]
        else:
            columns = [time_name, _ID, _VALUE, *covariate_names]
            rows = [
                [
                    stamps[t],
                    node,
                    values[node][t],
                    *(covariates[name][t] for name in covariate_names),
                ]
                for node in targets
                for t in range(data.n_rows)
            ]
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
        if frequency is not None:
            payload["frequency"] = dict(frequency)
        if future_rows is not None:
            names = list(self._future_covariates)
            ids = [None] if self.multivariate else list(targets)
            future_columns = [time_name, *names] if self.multivariate else [time_name, _ID, *names]
            future_table = [
                [
                    future_rows["__time__"][step],
                    *([] if ident is None else [ident]),
                    *(cast[name](future_rows[name][step], name) for name in names),
                ]
                for ident in ids
                for step in range(horizon)
            ]
            payload["future"] = {"table": {"columns": future_columns, "rows": future_table}}
        return payload

    def _run(self, client: Any, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Submit one forecast job with ``client.submit_forecast`` and wait for its result.

        The first job's versions become the expectations of every later one, so a
        version change in Welt fails the next job instead of mixing versions; the
        result's versions are also compared with the pin here.
        """
        session = self._session
        pinned = session.versions
        job = client.submit_forecast(
            payload,
            idempotency_key=uuid.uuid4().hex,
            expected_model_version=None if pinned is None else pinned[0],
            expected_configuration_version=None if pinned is None else pinned[1],
        )
        session.used += 1  # Welt accepted the job (202): it counts against the budget
        job_id = str(job.id)
        result: dict[str, Any] = job.result(timeout=self.timeout)
        model_id = result.get("model_id")
        version = result.get("model_version")
        configuration = result.get("configuration")
        if model_id != self.model:
            raise _unreadable(job_id, f"model {model_id!r}, expected {self.model!r}")
        if not (isinstance(version, str) and version and isinstance(configuration, str)):
            raise _unreadable(job_id, "no model_version or configuration")
        if not configuration:
            raise _unreadable(job_id, "an empty configuration")
        if pinned is None:
            session.versions = (version, configuration)
        elif (version, configuration) != pinned:
            raise _unreadable(
                job_id,
                f"versions {(version, configuration)}, but every job of this forecaster runs "
                f"{pinned}",
            )
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
            return _unreadable(job_id, reason)

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
                model=str(result["model_id"]),
                model_version=str(result["model_version"]),
                configuration=str(result["configuration"]),
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


def _unreadable(job_id: str, reason: str) -> ValueError:
    return ValueError(
        f"Welt forecast job {job_id!r} returned a result ergodic cannot read: {reason}. "
        f"The result stays in Welt (welt.Client().job_result({job_id!r}))"
    )


def _scalar(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _number(value: Any, name: str) -> float | None:
    """A numeric cell as the API validates it: a finite float, or None for missing."""
    value = _scalar(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number) or abs(number) > _FP32_MAX:
            raise ValueError(f"{name!r} holds {value!r}; Welt needs finite FP32 numbers")
        return number
    raise ValueError(f"{name!r} holds a {type(value).__name__} cell; Welt needs numbers")


def _category(value: Any, name: str) -> str | int | None:
    """A categorical cell as the API validates it: a string or an integer."""
    value = _scalar(value)
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return str(value)


_FP32_MAX = float(np.finfo(np.float32).max)


def _encoded_bytes(body: Any) -> int:
    """Bytes of the compact JSON the SDK sends for ``body``."""
    text = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return len(text.encode())


def _max_request_bytes() -> int:
    """The largest request body welt-client sends (``welt.forecast.MAX_REQUEST_BYTES``)."""
    import welt.forecast

    return int(getattr(welt.forecast, "MAX_REQUEST_BYTES", _DEFAULT_MAX_REQUEST_BYTES))


def _offset_name(freq: str | None) -> str | None:
    if not freq:
        return None
    import pandas as pd

    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except (TypeError, ValueError):
        return None
    return None if offset is None else type(offset).__name__


def _time_plan(data: TimeSeriesData) -> tuple[str, dict[str, Any] | None]:
    """How the index goes to Welt: ``("timestamps", frequency)`` or steps as days.

    A datetime index is sent as timestamps when its frequency is known (the
    dataset's ``freq``, else pandas' inferred one, mapped to Welt's units) or its
    spacing is constant, so Welt can infer it. An index on a calendar pandas
    recognises but Welt has no unit for (business days ``B``, custom ``C``,
    business-month ends ``BME``, semi-month ``SME`` and the like), declared or
    inferred by pandas (which cannot infer every one, ``SME`` for instance), is sent
    as consecutive days, which Chronos-2 reads the same way. A declared calendar is
    trusted as given: its rows are not checked against it. Any
    other irregular datetime index, a daily series with missing days say, is
    refused: sending it as steps would close up the gaps. A numeric index must be
    evenly spaced (relative tolerance 1e-9) and is always sent as days.
    """
    times = np.asarray(data.times)
    if times.dtype.kind == "M":
        if bool(np.any(np.isnat(times))):
            raise ValueError("the time index has missing values")
        declared = _frequency(data.freq)
        if declared is not None:
            return "timestamps", declared
        if data.freq and _offset_name(data.freq) is not None:
            return "steps-as-days", dict(_DAY)  # a calendar Welt has no unit for
        inferred: str | None = None
        if times.size >= 3:
            import pandas as pd

            try:
                inferred = pd.infer_freq(pd.DatetimeIndex(times))
            except (TypeError, ValueError):
                inferred = None
            mapped = _frequency(inferred)
            if mapped is not None:
                return "timestamps", mapped
        if inferred and _offset_name(inferred) is not None:
            return "steps-as-days", dict(_DAY)  # a regular calendar Welt has no unit for
        seconds = np.diff(times.astype("datetime64[s]").astype(np.int64))
        if seconds.size and bool(np.all(seconds == seconds[0])) and seconds[0] > 0:
            return "timestamps", None
        raise ValueError(
            f"the time index {data.time!r} is irregular (missing or uneven steps), and Welt "
            "forecasts regular series only. Reindex it to its regular frequency and leave "
            "the missing targets as NaN, which Chronos-2 reads natively, for example "
            "frame.set_index(time).asfreq('D').reset_index(); or, if the rows follow a "
            "calendar, declare the calendar with freq=, e.g. 'B' or 'BME'"
        )
    if times.dtype.kind not in "iuf":
        raise ValueError(
            f"Welt reads a datetime or numeric time index; {data.time!r} is {times.dtype}"
        )
    values = times.astype(np.float64)
    steps = np.diff(values)
    if (
        values.size < 2
        or steps[0] <= 0
        or not bool(np.allclose(steps, steps[0], rtol=1e-9, atol=0.0))
    ):
        raise ValueError(
            f"the numeric time index {data.time!r} must be evenly spaced and increasing; "
            "use a datetime index for anything else"
        )
    return "steps-as-days", dict(_DAY)


def _encode_times(
    times: Any, data: TimeSeriesData, encoding: str, *, future: bool = False
) -> list[str]:
    """The time cells for Welt: ISO-8601 UTC timestamps, or row positions as days.

    With timestamps, a datetime index is sent as is, read as UTC. As days, row ``k``
    of the training window is day ``k`` from 2000-01-01, and a future window must
    be the rows right after it: for a numeric index the exact next steps, for a
    datetime index any increasing times after its end.
    """
    history = np.asarray(data.times)
    values = np.asarray(times)
    if history.dtype.kind == "M":
        if values.dtype.kind != "M":
            raise ValueError("future= times must be datetimes, like the training index")
        if bool(np.any(np.isnat(values))):
            raise ValueError("the time index has missing values")
        if future and values.size and values[0] <= history[-1]:
            raise ValueError("future= must start after the end of the training window")
        if encoding == "timestamps":
            seconds = values.astype("datetime64[s]")
            return [f"{stamp}Z" for stamp in np.datetime_as_string(seconds, unit="s")]
        if future and values.size > 1 and not bool(np.all(np.diff(values) > np.timedelta64(0))):
            raise ValueError("future= times must be increasing")
    elif future:
        origin = float(history[0])
        step = (float(history[-1]) - origin) / max(history.size - 1, 1)
        offsets = (values.astype(np.float64) - origin) / step
        expected = np.arange(history.size, history.size + values.size, dtype=np.float64)
        if not bool(np.allclose(offsets, expected, rtol=0.0, atol=1e-6)):
            raise ValueError("future= times must be the steps right after the training window")
    start = history.size if future else 0
    positions = np.arange(start, start + values.size, dtype=np.int64)
    days = _STEP_ORIGIN + positions * np.timedelta64(_SECONDS_PER_DAY, "s")
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


def _frequency(freq: str | None) -> dict[str, Any] | None:
    """Welt's ``frequency`` for a pandas frequency string, or ``None`` when it has none."""
    if not freq:
        return None
    import pandas as pd

    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except (TypeError, ValueError):
        return None
    if offset is None:
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
    stated = meta.get("limits")
    limits: Mapping[str, Any] = stated if isinstance(stated, dict) else {}
    floor = int(limits.get("min_context", meta.get("min_context", 16)))
    if isinstance(envelopes, dict):
        corners = list(envelopes.get("corners") or ())
        floor = int(envelopes.get("min_context", floor))
    else:
        corners = list(envelopes or ())
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
