# SPDX-License-Identifier: Apache-2.0
"""ergodic.welt.WeltForecaster against a fake Welt forecasting API (DEC-085, DEC-087).

The fake serves ``GET /v1/models/chronos-2`` (with ``forecast_envelopes``) and
``POST /v1/forecasts`` plus the job routes, through ``httpx.MockTransport``, so no
test touches the network. Its "Chronos" is deterministic arithmetic: the median is
the mean of the last three observations plus ``beta`` times each numeric future
covariate, and the quantiles are normal around it with a spread that grows with
the step. It checks every request against the DEC-085 §1 contract as it goes.
"""

from __future__ import annotations

import copy
import json
import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("welt")
httpx = pytest.importorskip("httpx")

import welt  # noqa: E402
from tests.test_welt import FakeWelt  # noqa: E402

from ergodic import hts, timeseries  # noqa: E402
from ergodic.data import TimeSeriesData  # noqa: E402
from ergodic.forecast import backtest, reconcile  # noqa: E402
from ergodic.welt import (  # noqa: E402
    NATIVE_QUANTILE_LEVELS,
    Chronos,
    WeltForecastDiagnostics,
    WeltForecaster,
)

_ONE_CORNER = [{"max_context": 512, "max_variates": 1, "max_series": 1, "max_horizon": 64}]
_WIDE = [
    {"max_context": 512, "max_variates": 1, "max_series": 1, "max_horizon": 64},
    {
        "max_context": 2000,
        "max_variates": 10,
        "max_series": 20,
        "max_horizon": 256,
    },
]


def _z(level: float) -> float:
    return NormalDist().inv_cdf(level)


class FakeChronos(FakeWelt):
    """FakeWelt plus the forecasting routes of DEC-085."""

    def __init__(
        self,
        *,
        envelopes: list[dict[str, int]] | None = None,
        beta: float = 10.0,
        rearranged_steps: int = 0,
        fail: bool = False,
        wrong_horizon: bool = False,
        ignore_expectations: bool = False,
        drop_version: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.envelopes = _WIDE if envelopes is None else envelopes
        self.beta = beta
        self.rearranged_steps = rearranged_steps
        self.fail = fail
        self.wrong_horizon = wrong_horizon
        self.ignore_expectations = ignore_expectations
        self.drop_version = drop_version
        self.version = "chronos-2-fixture-1"
        self.refused: list[str] = []
        self.forecasts: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []

    def handle(self, request: Any) -> Any:
        path = request.url.path
        if request.method == "GET" and path == "/v1/models/chronos-2":
            self.calls.append((request.method, path))
            return httpx.Response(200, json=self._metadata())
        if request.method == "POST" and path == "/v1/forecasts":
            self.calls.append((request.method, path))
            return self._forecast(request)
        return super().handle(request)

    def _metadata(self) -> dict[str, Any]:
        return {
            "id": "chronos-2",
            "family": "forecasting",
            "operations": ["forecast"],
            "tasks": ["forecasting"],
            "version": self.version,
            "quantile_levels": list(NATIVE_QUANTILE_LEVELS),
            "forecast_envelopes": self.envelopes,
            "limits": {"min_context": 16},
        }

    def _forecast(self, request: Any) -> Any:
        body = json.loads(request.content)
        headers = {k.lower(): v for k, v in request.headers.items()}
        assert headers.get("idempotency-key")
        expected = headers.get("x-welt-model-version")
        if expected is not None and not self.ignore_expectations:
            assert headers.get("x-welt-configuration-version") == "chronos-2-forecast-v1"
            if expected != self.version:
                error = {"code": "model_version_mismatch", "message": "Version changed."}
                return httpx.Response(409, json={"error": error})
        problem = self._cells(body)
        if problem is not None:
            self.refused.append(problem)
            return httpx.Response(
                422, json={"error": {"code": "invalid_input", "message": problem}}
            )
        self.forecasts.append(body)
        self.headers.append(headers)
        result = self._chronos(body)
        job = f"job_c{len(self.jobs) + 1}"
        status: dict[str, Any] = {
            "id": job,
            "status": "failed" if self.fail else "succeeded",
            "operation": "forecast",
            "task": "forecasting",
            "model_id": "chronos-2",
            "model_version": self.version,
        }
        if self.fail:
            status["error"] = {"code": "execution_failed", "message": "Chronos child exited."}
        self.jobs[job] = {"status": status, "result": result}
        return httpx.Response(202, json={"id": job, "status": "queued"})

    def _table(self, source: dict[str, Any]) -> dict[str, Any]:
        """An inline table, or the uploaded dataset a source names."""
        if "dataset_id" in source:
            dataset = self.datasets[source["dataset_id"]]
            return {"columns": dataset["columns"], "rows": dataset["rows"]}
        table: dict[str, Any] = source["table"]
        return table

    def _cells(self, body: dict[str, Any]) -> str | None:
        """The server's cell rules (welt_api.forecasts.numeric / category), or None if met.

        Numbers are int or float but never bool, finite FP32; categorical cells are
        str or exact int; targets and past covariates may be missing, future
        covariates (in both tables), time and id may not.
        """
        categorical = set(body["categorical_covariates"])
        future = set(body["future_covariates"])
        nullable = set(body["targets"]) | set(body["past_covariates"])

        def bad(value: Any, name: str) -> str | None:
            if value is None:
                return None if name in nullable else f"{name} cells cannot be missing."
            if name in categorical:
                ok = type(value) in (str, int)
                return None if ok else f"{name}: categorical cells must be strings or integers."
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return f"{name} cells must be numbers."
            if not math.isfinite(float(value)) or abs(float(value)) > 3.4028234663852886e38:
                return f"{name} cells must be finite FP32 values."
            return None

        checked = set(body["targets"]) | set(body["past_covariates"]) | future
        for role in ("history", "future"):
            if role not in body:
                continue
            table = self._table(body[role])
            for row in table["rows"]:
                for name, value in zip(table["columns"], row, strict=True):
                    if name in (body["time_column"], body.get("id_column")):
                        if type(value) is not str:
                            return f"{name} cells must be strings."
                    elif name in checked and (problem := bad(value, name)):
                        return problem
        return None

    def _chronos(self, body: dict[str, Any]) -> dict[str, Any]:
        """Check the request against DEC-085 §1 and forecast it deterministically."""
        assert body["model_id"] == "chronos-2"
        levels = body["quantile_levels"]
        assert levels == sorted(set(levels)) and set(levels) <= set(NATIVE_QUANTILE_LEVELS)
        horizon = int(body["horizon"])
        table = self._table(body["history"])
        columns = table["columns"]
        time_name, id_name = body["time_column"], body.get("id_column")
        targets = body["targets"]
        future_names = body["future_covariates"]
        assert set(body["categorical_covariates"]) <= set(body["past_covariates"] + future_names)
        assert ("future" in body) == bool(future_names)
        rows = table["rows"]
        ids: list[Any] = []
        for row in rows:
            ident = row[columns.index(id_name)] if id_name else None
            if ident not in ids:
                ids.append(ident)
        futures: dict[Any, list[list[Any]]] = {ident: [] for ident in ids}
        if future_names:
            ftable = self._table(body["future"])
            fcols = ftable["columns"]
            assert fcols == [time_name, *([id_name] if id_name else []), *future_names]
            assert not set(targets) & set(fcols)
            for row in ftable["rows"]:
                futures[row[fcols.index(id_name)] if id_name else None].append(row)
            assert all(len(futures[ident]) == horizon for ident in ids)
        series: list[dict[str, Any]] = []
        for ident in ids:
            mine = [row for row in rows if not id_name or row[columns.index(id_name)] == ident]
            for target in targets:
                values = np.array(
                    [
                        np.nan if r[columns.index(target)] is None else r[columns.index(target)]
                        for r in mine
                    ],
                    dtype=float,
                )
                finite = values[np.isfinite(values)]
                level = float(np.mean(finite[-3:]))
                sd = max(float(np.std(np.diff(finite))), 0.1)
                effect = np.zeros(horizon)
                for name in future_names:
                    if name in body["categorical_covariates"]:
                        continue
                    k = [time_name, *([id_name] if id_name else []), *future_names].index(name)
                    effect += self.beta * np.array([r[k] for r in futures[ident]], dtype=float)
                steps = np.arange(1, horizon + 1)
                median = level + effect
                quantiles = [
                    [median[h] + sd * np.sqrt(steps[h]) * _z(p) for p in levels]
                    for h in range(horizon)
                ]
                n = horizon + 1 if self.wrong_horizon else horizon
                series.append(
                    {
                        "id": None if ident is None else str(ident),
                        "target": target,
                        "context_length": len(mine),
                        "timestamps": [f"step-{h}" for h in range(n)],
                        "predictions": median.tolist(),
                        "quantiles": quantiles,
                    }
                )
        return {
            "task": "forecasting",
            "model_id": "chronos-2",
            "model_version": None if self.drop_version else self.version,
            "configuration": "chronos-2-forecast-v1",
            "horizon": horizon + 1 if self.wrong_horizon else horizon,
            "frequency": body.get("frequency", {"unit": "day", "every": 1}),
            "quantile_levels": levels,
            "point_semantics": "median",
            "calibrated_intervals": False,
            "cross_learning": body["cross_learning"],
            "rearranged_steps": self.rearranged_steps,
            "columns": {
                "time": time_name,
                "id": id_name,
                "targets": targets,
                "past_covariates": body["past_covariates"],
                "future_covariates": future_names,
                "categorical_covariates": body["categorical_covariates"],
            },
            "series": series,
        }


@pytest.fixture(autouse=True)
def _no_ambient_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WELT_WORKSPACE", raising=False)
    monkeypatch.delenv("WELT_BASE_URL", raising=False)


def _days(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=n, freq="D")


def sales(n: int = 60, seed: int = 0) -> TimeSeriesData:
    rng = np.random.default_rng(seed)
    promo = (np.arange(n) % 7 == 3).astype(float)
    frame = pd.DataFrame(
        {
            "day": _days(n),
            "sales": 100 + 10 * promo + rng.normal(0, 1, n),
            "temperature": 20 + rng.normal(0, 2, n),
            "promo": promo,
            "holiday": np.where(np.arange(n) % 30 == 0, "yes", "no"),
        }
    )
    return timeseries(frame, time="day", freq="D", types={"promo": "continuous"})


def regions(n: int = 40, *, with_total: bool = True) -> Any:
    rng = np.random.default_rng(1)
    north = 60 + rng.normal(0, 2, n)
    south = 40 + rng.normal(0, 2, n)
    frame = pd.DataFrame({"day": _days(n), "north": north, "south": south})
    if with_total:
        frame["total"] = north + south
    return hts(frame, time="day", freq="D", hierarchy={"total": ["north", "south"]})


# -- mapping ------------------------------------------------------------------


def test_one_series_maps_to_a_one_id_panel_and_comes_back_as_draws() -> None:
    fake = FakeChronos()
    data = sales().select(["sales"])
    with fake.client() as client:
        model = WeltForecaster(client=client)
        result = model.fit(data).forecast(7, seed=0)
    assert result.nodes == ("sales",)
    assert result.kind == "draws"
    assert result.draws().shape == (1000, 7)
    assert result.start == np.datetime64("2026-03-02")
    (body,) = fake.forecasts
    table = body["history"]["table"]
    assert table["columns"] == ["day", "__ergodic_series__", "__ergodic_value__"]
    assert len(table["rows"]) == 60
    assert table["rows"][0][:2] == ["2026-01-01T00:00:00Z", "sales"]
    assert body["id_column"] == "__ergodic_series__"
    assert body["targets"] == ["__ergodic_value__"]
    assert body["quantile_levels"] == list(NATIVE_QUANTILE_LEVELS)
    assert len(NATIVE_QUANTILE_LEVELS) == 21
    assert body["frequency"] == {"unit": "day", "every": 1}
    assert body["cross_learning"] is False and "future" not in body
    # Provenance: the job, the versions, and Welt's own quantiles.
    record = result.diagnostics["sales"]  # type: ignore[index]
    assert isinstance(record, WeltForecastDiagnostics)
    assert record.job_id == "job_c1"
    assert record.model_version == "chronos-2-fixture-1"
    assert record.configuration == "chronos-2-forecast-v1"
    assert record.quantiles.shape == (7, 21) and not record.quantiles.flags.writeable
    assert record.context_length == 60 and record.rearranged_steps == 0
    assert record.conversion.startswith("inverse-cdf")
    # The draws' median is Welt's median, up to Monte Carlo error.
    q = record.quantiles
    native_median = q[:, NATIVE_QUANTILE_LEVELS.index(0.5)]
    spread = q[:, NATIVE_QUANTILE_LEVELS.index(0.75)] - q[:, NATIVE_QUANTILE_LEVELS.index(0.25)]
    assert np.all(np.abs(result.median() - native_median) < 0.15 * spread)
    assert model.forecasts_used == 1
    assert fake.count("GET", "/v1/models/chronos-2") == 1


def test_the_seed_reproduces_the_draws_and_the_catalogue_is_read_once() -> None:
    fake = FakeChronos()
    with fake.client() as client:
        fitted = Chronos(client=client, draws=200).fit(sales().select(["sales"]))
        a, b, c = (fitted.forecast(5, seed=s) for s in (0, 0, 1))
    np.testing.assert_array_equal(a.draws(), b.draws())
    assert not np.array_equal(a.draws(), c.draws())
    assert fake.count("POST", "/v1/forecasts") == 3
    assert fake.count("GET", "/v1/models/chronos-2") == 1
    assert Chronos is WeltForecaster


def test_several_columns_are_a_panel_and_multivariate_is_one_series() -> None:
    data = regions(with_total=False)
    plain = TimeSeriesData(
        schema=data.schema, n_rows=data.n_rows, _store=data._store, time="day", freq="D"
    )
    fake = FakeChronos()
    with fake.client() as client:
        panel = WeltForecaster(client=client).fit(plain).forecast(4)
        joint = WeltForecaster(client=client, multivariate=True).fit(plain).forecast(4)
    assert panel.nodes == joint.nodes == ("north", "south")
    first, second = fake.forecasts
    assert len(first["history"]["table"]["rows"]) == 80
    assert {row[1] for row in first["history"]["table"]["rows"]} == {"north", "south"}
    assert second["history"]["table"]["columns"] == ["day", "north", "south"]
    assert second["targets"] == ["north", "south"] and "id_column" not in second
    assert len(second["history"]["table"]["rows"]) == 40
    np.testing.assert_allclose(panel.median("north"), joint.median("north"), atol=0.5)


# -- covariates -----------------------------------------------------------------


def test_past_future_and_categorical_covariates_map_to_their_roles() -> None:
    fake = FakeChronos()
    data = sales()
    later = pd.DataFrame(
        {
            "day": pd.date_range("2026-03-02", periods=7, freq="D"),
            "promo": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "holiday": ["no"] * 6 + ["yes"],
            "sales": [999.0] * 7,  # present in the frame, never read
        }
    )
    with fake.client() as client:
        model = WeltForecaster(
            client=client,
            past_covariates=["temperature"],
            future_covariates=["promo", "holiday"],
        ).fit(data)
        with pytest.raises(ValueError, match="needs future="):
            model.forecast(7)
        assert fake.count("POST", "/v1/forecasts") == 0
        result = model.forecast(7, future=later)
    (body,) = fake.forecasts
    assert body["past_covariates"] == ["temperature"]
    assert body["future_covariates"] == ["promo", "holiday"]
    assert body["categorical_covariates"] == ["holiday"]
    assert body["history"]["table"]["columns"] == [
        "day",
        "__ergodic_series__",
        "__ergodic_value__",
        "temperature",
        "promo",
        "holiday",
    ]
    future = body["future"]["table"]
    assert future["columns"] == ["day", "__ergodic_series__", "promo", "holiday"]
    assert [row[2] for row in future["rows"]] == later["promo"].tolist()
    assert future["rows"][0][0] == "2026-03-02T00:00:00Z"
    assert 999.0 not in [cell for row in future["rows"] for cell in row]
    # The fake adds ten per promo day, so the known promo shows in the forecast.
    median = result.median("sales")
    assert median[1] - median[0] == pytest.approx(10.0, abs=0.6)


def test_future_rows_are_checked_before_anything_is_sent() -> None:
    fake = FakeChronos()
    with fake.client() as client:
        data = sales().select(["sales", "promo"])
        model = WeltForecaster(client=client, future_covariates="promo").fit(data)
        short = pd.DataFrame({"day": _days(3), "promo": [0.0, 1.0, 0.0]})
        with pytest.raises(ValueError, match="has 3 rows"):
            model.forecast(7, future=short)
        stale = pd.DataFrame({"day": _days(7), "promo": [0.0] * 7})
        with pytest.raises(ValueError, match="start after"):
            model.forecast(7, future=stale)
        with pytest.raises(ValueError, match="lacks"):
            model.forecast(7, future=pd.DataFrame({"day": _days(7)}))
    assert fake.count("POST", "/v1/forecasts") == 0


def test_backtest_sends_the_true_future_covariates_and_no_target_from_the_window() -> None:
    fake = FakeChronos()
    data = sales()
    model_sales = data.column("sales")
    times = data.times
    with fake.client() as client:
        model = WeltForecaster(client=client, future_covariates=["promo"], draws=300)
        report = backtest(model, data.select(["sales", "promo"]), test_size=7, n_splits=3)
    assert report.model == "welt:chronos-2"
    assert report.future_covariates == ("promo",)
    assert report.nodes == ("sales",)
    assert model.forecasts_used == 3 and not model.fitted
    assert len(fake.forecasts) == 3
    for fold, body in zip(report.folds, fake.forecasts, strict=True):
        cut = fold.cutoff
        rows = body["history"]["table"]["rows"]
        # History: exactly the training prefix, ending the step before the cutoff.
        assert len(rows) == cut
        last = np.datetime_as_string(times[cut - 1].astype("datetime64[s]"), unit="s")
        assert rows[-1][0] == f"{last}Z"
        # Future: the true promo over the held-out window, and no target column.
        future = body["future"]["table"]
        assert future["columns"] == ["day", "__ergodic_series__", "promo"]
        assert [row[2] for row in future["rows"]] == data.column("promo")[cut : cut + 7].tolist()
        sent = {cell for row in rows + future["rows"] for cell in row if isinstance(cell, float)}
        assert not sent & set(model_sales[cut:].tolist())
    # The first job's versions are required of every later fold.
    assert "x-welt-model-version" not in fake.headers[0]
    assert all(h["x-welt-model-version"] == "chronos-2-fixture-1" for h in fake.headers[1:])
    assert max(report.values("sales", "mae")) < 3.0


def test_the_budget_is_shared_by_every_copy_and_stops_before_paying() -> None:
    fake = FakeChronos()
    with fake.client() as client:
        model = WeltForecaster(client=client, max_forecasts=2, draws=50)
        twin = copy.deepcopy(model)
        assert twin.client is client
        with pytest.raises(RuntimeError, match="max_forecasts=2"):
            backtest(model, sales().select(["sales"]), test_size=7, n_splits=3)
    assert fake.count("POST", "/v1/forecasts") == 2
    assert twin.forecasts_used == 2


def test_a_changed_model_version_fails_the_next_fold_rather_than_mixing_versions() -> None:
    fake = FakeChronos()
    with fake.client() as client:
        fitted = WeltForecaster(client=client, draws=50).fit(sales().select(["sales"]))
        fitted.forecast(3)
        fake.version = "chronos-2-fixture-2"
        with pytest.raises(welt.WeltError):
            fitted.forecast(3)


# -- hierarchies and reconciliation ---------------------------------------------------


def test_a_hierarchy_is_forecast_at_every_level_and_reconciles_on_the_draws() -> None:
    fake = FakeChronos()
    data = regions()
    tree = data.hierarchy
    with fake.client() as client:
        raw = WeltForecaster(client=client, draws=400).fit(data).forecast(5, seed=2)
        reconciled = (
            WeltForecaster(client=client, draws=400, reconcile="mint").fit(data).forecast(5)
        )
    assert raw.nodes == ("total", "north", "south")
    assert raw.coherent is None and raw.coherence_error(tree) > 0.0
    for method in ("bottom_up", "mint"):
        fixed = reconcile(raw, tree, method=method)
        assert fixed.coherent is True
        assert fixed.coherence_error(tree) < 1e-12
    top = reconcile(raw, tree, method="top_down", data=data)
    assert top.coherent is True
    assert reconciled.coherent is True and reconciled.reconciliation is not None
    assert {row[1] for row in fake.forecasts[0]["history"]["table"]["rows"]} == {
        "total",
        "north",
        "south",
    }


def test_leaves_only_fill_the_total_with_their_sum_and_an_aggregate_column_is_built() -> None:
    fake = FakeChronos()
    with fake.client() as client:
        leaves = WeltForecaster(client=client, targets=["north", "south"], draws=100)
        result = leaves.fit(regions()).forecast(3)
        no_total = WeltForecaster(client=client, draws=100).fit(regions(with_total=False))
        built = no_total.forecast(3)
    assert result.nodes == ("total", "north", "south")
    assert result.coherence_error(regions().hierarchy) == 0.0
    assert {row[1] for row in fake.forecasts[0]["history"]["table"]["rows"]} == {"north", "south"}
    total_rows = [row for row in fake.forecasts[1]["history"]["table"]["rows"] if row[1] == "total"]
    data = regions(with_total=False)
    np.testing.assert_allclose(
        [row[2] for row in total_rows], data.column("north") + data.column("south")
    )
    assert built.nodes == ("total", "north", "south")


# -- limits, time and failures -----------------------------------------------------


def test_a_request_outside_every_measured_envelope_is_refused_before_paying() -> None:
    fake = FakeChronos(envelopes=_ONE_CORNER)
    data = regions(with_total=False)
    plain = TimeSeriesData(
        schema=data.schema, n_rows=data.n_rows, _store=data._store, time="day", freq="D"
    )
    with fake.client() as client:
        with pytest.raises(ValueError, match="no measured envelope"):
            WeltForecaster(client=client).fit(plain).forecast(4)
        with pytest.raises(ValueError, match="no measured envelope"):
            WeltForecaster(client=client).fit(sales(600).select(["sales"])).forecast(4)
        with pytest.raises(ValueError, match="no measured envelope"):
            WeltForecaster(client=client).fit(sales(10).select(["sales"])).forecast(4)
        assert fake.count("POST", "/v1/forecasts") == 0
        trimmed = WeltForecaster(client=client, max_context=512, draws=50)
        trimmed.fit(sales(600).select(["sales"])).forecast(4)
    assert len(fake.forecasts[0]["history"]["table"]["rows"]) == 512


def test_a_model_that_does_not_serve_forecasts_is_refused_before_paying() -> None:
    fake = FakeChronos()
    hidden = {**fake._metadata(), "operations": ["model_run"]}
    fake._metadata = lambda: hidden  # type: ignore[method-assign]
    with fake.client() as client, pytest.raises(ValueError, match="does not serve forecasts"):
        WeltForecaster(client=client).fit(sales().select(["sales"])).forecast(3)
    assert fake.count("POST", "/v1/forecasts") == 0


def test_an_integer_time_index_is_sent_as_consecutive_days() -> None:
    fake = FakeChronos()
    frame = pd.DataFrame({"t": np.arange(30), "y": np.linspace(0, 3, 30), "x": np.ones(30)})
    data = timeseries(frame, time="t", types={"x": "continuous"})
    with fake.client() as client:
        model = WeltForecaster(client=client, future_covariates=["x"], draws=50).fit(data)
        with pytest.raises(ValueError, match="right after"):
            model.forecast(2, future=pd.DataFrame({"t": [40, 41], "x": [1.0, 1.0]}))
        result = model.forecast(2, future=pd.DataFrame({"t": [30, 31], "x": [1.0, 1.0]}))
    (body,) = fake.forecasts
    assert body["history"]["table"]["rows"][0][0] == "2000-01-01T00:00:00Z"
    assert body["future"]["table"]["rows"][0][0] == "2000-01-31T00:00:00Z"
    assert body["frequency"] == {"unit": "day", "every": 1}
    assert result.diagnostics["y"].time_encoding == "steps-as-days"  # type: ignore[index]
    assert result.index == (30.0, 31.0)


def test_rearranged_steps_are_reported_and_failures_carry_the_job() -> None:
    fake = FakeChronos(rearranged_steps=2)
    with fake.client() as client:
        result = WeltForecaster(client=client, draws=50).fit(sales().select(["sales"])).forecast(3)
    assert result.diagnostics["sales"].rearranged_steps == 2  # type: ignore[index]

    failing = FakeChronos(fail=True)
    with failing.client() as client, pytest.raises(welt.ExecutionError) as error:
        WeltForecaster(client=client).fit(sales().select(["sales"])).forecast(3)
    assert error.value.job_id == "job_c1" and "Chronos child exited" in str(error.value)

    odd = FakeChronos(wrong_horizon=True)
    with odd.client() as client, pytest.raises(ValueError, match=r"cannot read.*horizon"):
        WeltForecaster(client=client).fit(sales().select(["sales"])).forecast(3)


def test_an_sdk_without_submit_forecast_is_refused_before_paying() -> None:
    class OldClient:
        def model(self, model_id: str) -> dict[str, Any]:
            return FakeChronos()._metadata()

    model = WeltForecaster(client=OldClient(), draws=10).fit(sales().select(["sales"]))
    with pytest.raises(RuntimeError, match=r"welt-client 0\.14"):
        model.forecast(3)


def test_bad_configurations_are_refused_locally() -> None:
    with pytest.raises(ValueError, match="native levels"):
        WeltForecaster(quantile_levels=(0.1, 0.33, 0.9))
    with pytest.raises(ValueError, match="both targets and covariates"):
        WeltForecaster(targets=["sales"], future_covariates=["sales"])
    with pytest.raises(ValueError, match="both past and future"):
        WeltForecaster(past_covariates=["x"], future_covariates=["x"])
    with pytest.raises(ValueError, match="reconcile must be"):
        WeltForecaster(reconcile="middle_out")
    with pytest.raises(ValueError, match="needs a HierarchicalTSData"):
        WeltForecaster(reconcile="mint").fit(sales().select(["sales"]))
    with pytest.raises(ValueError, match="Welt forecasts numbers"):
        WeltForecaster(targets=["holiday"]).fit(sales())
    with pytest.raises(ValueError, match="not variables"):
        WeltForecaster(future_covariates=["price"]).fit(sales())
    with pytest.raises(ValueError, match="not fitted"):
        WeltForecaster().forecast(3)
    frame = sales().select(["sales", "promo"]).to_pandas()
    frame.loc[5, "promo"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        WeltForecaster(future_covariates=["promo"]).fit(
            timeseries(frame, time="day", types={"promo": "continuous"})
        )
    assert repr(WeltForecaster(future_covariates=["promo"])) == (
        "WeltForecaster('chronos-2', future_covariates=['promo']; unfitted)"
    )


# -- review findings (API-163 independent review) --------------------------------------


def test_boolean_and_float_coded_categoricals_are_sent_as_integers() -> None:
    """F1: a bool flag or float codes in a categorical column are not valid Welt cells."""
    n = 40
    frame = pd.DataFrame(
        {
            "day": _days(n),
            "sales": np.linspace(10, 20, n),
            "flag": np.arange(n) % 7 == 3,  # bool, inferred categorical
            "code": (np.arange(n) % 3).astype(float),  # float codes
            "lift": np.arange(n) % 5 == 0,  # bool, declared continuous
        }
    )
    data = timeseries(
        frame, time="day", freq="D", types={"code": "categorical", "lift": "continuous"}
    )
    assert data.schema["flag"].is_discrete and data.schema["code"].is_discrete
    later = pd.DataFrame(
        {
            "day": pd.date_range("2026-02-10", periods=3, freq="D"),
            "flag": [True, False, False],
            "code": [2.0, 0.0, 1.0],
            "lift": [False, True, False],
        }
    )
    fake = FakeChronos()
    with fake.client() as client:
        model = WeltForecaster(client=client, future_covariates=["flag", "code", "lift"], draws=20)
        model.fit(data).forecast(3, future=later)
    assert fake.refused == []
    (body,) = fake.forecasts
    assert body["categorical_covariates"] == ["flag", "code"]
    row = body["history"]["table"]["rows"][3]
    assert row[3:] == [1, 0, 0.0] and type(row[3]) is int and type(row[4]) is int
    future = body["future"]["table"]["rows"]
    assert [r[2:] for r in future] == [[1, 2, 0.0], [0, 0, 1.0], [0, 1, 0.0]]
    assert type(future[0][4]) is float


def test_the_fake_applies_the_server_cell_rules() -> None:
    """F1: FakeChronos refuses what welt_api.forecasts.numeric/category refuse."""
    fake = FakeChronos()
    table: dict[str, Any] = {
        "columns": ["t", "y", "c"],
        "rows": [["2026-01-01T00:00:00Z", 1.0, True]],
    }
    body: dict[str, Any] = {
        "model_id": "chronos-2",
        "history": {"table": table},
        "time_column": "t",
        "targets": ["y"],
        "past_covariates": ["c"],
        "future_covariates": [],
        "categorical_covariates": ["c"],
        "horizon": 1,
        "quantile_levels": [0.1, 0.5, 0.9],
        "cross_learning": False,
    }
    assert fake._cells(body) is not None
    table["rows"] = [["2026-01-01T00:00:00Z", True, 1]]
    assert fake._cells(body) == "y cells must be numbers."
    table["rows"] = [["2026-01-01T00:00:00Z", None, "a"]]
    assert fake._cells(body) is None


def test_a_refused_submission_does_not_count_against_the_budget() -> None:
    """F6: only jobs Welt accepted count; a 422 and an old SDK cost nothing."""
    fake = FakeChronos()
    with fake.client() as client:
        model = WeltForecaster(client=client, max_forecasts=1, draws=20)
        fitted = model.fit(sales().select(["sales"]))
        original = fake._cells
        fake._cells = lambda body: "refused for the test"  # type: ignore[method-assign]
        with pytest.raises(welt.WeltError):
            fitted.forecast(3)
        assert model.forecasts_used == 0
        fake._cells = original  # type: ignore[method-assign]
        fitted.forecast(3)
        assert model.forecasts_used == 1
        with pytest.raises(RuntimeError, match="max_forecasts=1"):
            fitted.forecast(3)

    class OldClient:
        def model(self, model_id: str) -> dict[str, Any]:
            return FakeChronos()._metadata()

    old = WeltForecaster(client=OldClient(), draws=10)
    with pytest.raises(RuntimeError, match=r"welt-client 0\.14"):
        old.fit(sales().select(["sales"])).forecast(3)
    assert old.forecasts_used == 0


def test_a_hierarchy_node_named_as_a_covariate_is_refused_at_fit() -> None:
    """F5: the default targets of a hierarchy are its nodes; a node cannot be both."""
    with pytest.raises(ValueError, match="nodes of the hierarchy"):
        WeltForecaster(future_covariates=["north"]).fit(regions())


def test_a_float_index_with_a_fractional_step_is_evenly_spaced() -> None:
    """F7: spacing is compared with a relative tolerance, not exact float equality."""
    t = np.arange(30) * 0.1
    data = timeseries(pd.DataFrame({"t": t, "y": np.sin(t)}), time="t")
    fake = FakeChronos()
    with fake.client() as client:
        result = WeltForecaster(client=client, draws=20).fit(data).forecast(2)
    assert result.diagnostics["y"].time_encoding == "steps-as-days"  # type: ignore[index]
    assert fake.forecasts[0]["history"]["table"]["rows"][-1][0] == "2000-01-30T00:00:00Z"


def test_a_business_day_index_is_sent_as_consecutive_days() -> None:
    """F7: Welt refuses business days as irregular; Chronos-2 reads row order only."""
    days = pd.bdate_range("2026-01-05", periods=30)
    frame = pd.DataFrame({"day": days, "y": np.linspace(1, 2, 30), "promo": np.arange(30) % 5 == 0})
    data = timeseries(frame, time="day", freq="B", types={"promo": "continuous"})
    later = pd.DataFrame({"day": pd.bdate_range(days[-1] + pd.offsets.BDay(), periods=3)})
    later["promo"] = [1.0, 0.0, 0.0]
    fake = FakeChronos()
    with fake.client() as client:
        model = WeltForecaster(client=client, future_covariates=["promo"], draws=20).fit(data)
        result = model.forecast(3, future=later)
    (body,) = fake.forecasts
    assert body["frequency"] == {"unit": "day", "every": 1}
    assert body["history"]["table"]["rows"][29][0] == "2000-01-30T00:00:00Z"
    assert body["future"]["table"]["rows"][0][0] == "2000-01-31T00:00:00Z"
    assert result.diagnostics["y"].time_encoding == "steps-as-days"  # type: ignore[index]


def test_month_ends_without_a_declared_freq_keep_their_timestamps() -> None:
    """F7: pandas infers the month-end frequency, so Welt gets real dates and the unit."""
    months = pd.date_range("2024-01-31", periods=24, freq="ME")
    data = timeseries(pd.DataFrame({"m": months, "y": np.arange(24.0)}), time="m")
    fake = FakeChronos()
    with fake.client() as client:
        WeltForecaster(client=client, draws=20).fit(data).forecast(2)
    (body,) = fake.forecasts
    assert body["frequency"] == {"unit": "month", "every": 1}
    assert body["history"]["table"]["rows"][0][0] == "2024-01-31T00:00:00Z"


def test_a_table_over_the_inline_threshold_is_uploaded_as_a_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F8: like welt.Forecast, a large table goes up as a workspace dataset first."""
    from ergodic.welt import _forecast

    monkeypatch.setattr(_forecast, "_inline_limit", lambda: 2_000)
    fake = FakeChronos()
    with fake.client() as client:
        model = WeltForecaster(client=client, future_covariates=["promo"], draws=20)
        later = pd.DataFrame(
            {"day": pd.date_range("2026-03-02", periods=3), "promo": [0.0, 1.0, 0.0]}
        )
        result = model.fit(sales().select(["sales", "promo"])).forecast(3, future=later)
    (body,) = fake.forecasts
    assert body["history"] == {"dataset_id": "ds_1"}
    assert "table" in body["future"]  # small enough to stay inline
    assert fake.datasets["ds_1"]["columns"] == [
        "day",
        "__ergodic_series__",
        "__ergodic_value__",
        "promo",
    ]
    assert result.diagnostics["sales"].datasets == {"history": "ds_1"}  # type: ignore[index]


def test_a_table_over_the_request_limit_is_refused_before_anything_is_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F8: past the API's body limit neither inline nor an upload can carry it."""
    from ergodic.welt import _forecast

    monkeypatch.setattr(_forecast, "_MAX_TABLE_BYTES", 1_000)
    fake = FakeChronos()
    with fake.client() as client, pytest.raises(ValueError, match="request limit"):
        WeltForecaster(client=client).fit(sales().select(["sales"])).forecast(3)
    assert fake.calls == []


def test_a_result_without_versions_is_refused_and_pins_nothing() -> None:
    """F10: no 'None' pin from a result that lacks its model version."""
    fake = FakeChronos(drop_version=True)
    with fake.client() as client:
        model = WeltForecaster(client=client, draws=20)
        with pytest.raises(ValueError, match="no model_version"):
            model.fit(sales().select(["sales"])).forecast(3)
    assert model._session.versions is None


def test_the_result_versions_are_compared_with_the_pin_locally() -> None:
    """F10: even if a server ignored the expectation headers, versions cannot mix."""
    fake = FakeChronos(ignore_expectations=True)
    with fake.client() as client:
        fitted = WeltForecaster(client=client, draws=20).fit(sales().select(["sales"]))
        fitted.forecast(3)
        fake.version = "chronos-2-fixture-2"
        with pytest.raises(ValueError, match="every job of this forecaster runs"):
            fitted.forecast(3)
