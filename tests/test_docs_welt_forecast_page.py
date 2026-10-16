# SPDX-License-Identifier: Apache-2.0
"""The Chronos-2 forecasting page's programs, run against the fake Welt forecasting API.

The docs checker never runs a program marked ``requires="…"`` (it needs a Welt key
and spends credits), so this test runs the exact published code against
``FakeChronos`` from ``test_welt_forecast.py``: every request is answered locally,
and what the page says each step prints and pays is held to what the program does.
"""

from __future__ import annotations

import contextlib
import io
import runpy
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("welt")
httpx = pytest.importorskip("httpx")

import welt  # noqa: E402
from scripts.docsite.config import load_site  # noqa: E402
from scripts.docsite.examples import programs_of  # noqa: E402
from tests.test_welt_forecast import MEASURED_CORNERS, FakeChronos  # noqa: E402


def _run(
    name: str, fake: FakeChronos, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[str]:
    site = load_site()
    page = site.pages_dir / "forecast-with-welt.md"
    (program,) = [p for p in programs_of(page, site) if p.name == name]
    assert program.requires == "a Welt API key"

    monkeypatch.delenv("WELT_WORKSPACE", raising=False)
    monkeypatch.delenv("WELT_BASE_URL", raising=False)
    real = welt.Client

    def client(*args: Any, **kwargs: Any) -> Any:
        transport = httpx.MockTransport(fake.handle)
        return real(base_url="https://welt.test", api_key="wk_fixture_only", transport=transport)

    monkeypatch.setattr(welt, "Client", client)
    script = tmp_path / program.name
    script.write_text(program.code)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        runpy.run_path(str(script), run_name="__main__")
    return printed.getvalue().splitlines()


def test_the_forecast_and_backtest_program_runs_against_the_fake(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeChronos()
    assert fake.envelopes == MEASURED_CORNERS  # the published API-160 envelope
    lines = _run("welt_forecast.py", fake, monkeypatch, tmp_path)

    # Step 2: one series, 14 steps, 1,000 draws; the versions and the 21 levels.
    assert lines[0] == "ForecastResult(draws: 1 series [sales], horizon=14, S=1000)"
    assert "chronos-2-fixture-1 chronos-2-forecast-v1 21" in lines
    # Step 3: one row per model, then the paid forecasts.
    assert any(line.startswith("welt:chronos-2") for line in lines)
    assert any(line.startswith("Naive") for line in lines)
    assert any(line.startswith("SeasonalNaive") for line in lines)
    assert lines[-1] == "paid forecasts: 3"

    # "Step 2 ... one paid job" plus "three paid jobs" for the backtest.
    assert fake.count("POST", "/v1/forecasts") == 4
    first = fake.forecasts[0]
    assert first["future_covariates"] == ["promo"]
    assert first["future"]["table"]["columns"] == ["day", "__ergodic_series__", "promo"]
    # The backtest's future tables carry the promotion calendar, never the target.
    for body in fake.forecasts:
        assert "sales" not in body["future"]["table"]["columns"]


def test_the_reconcile_program_runs_against_the_fake(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeChronos()
    assert fake.envelopes == MEASURED_CORNERS
    lines = _run("welt_reconcile.py", fake, monkeypatch, tmp_path)

    assert lines[0] == "ForecastResult(draws: 3 series [total, north, south], horizon=14, S=1000)"
    assert lines[1].startswith("total minus regions:")
    assert lines[2].split() == ["bottom_up", "coherent=True"]
    assert lines[3].split() == ["mint(weights=posterior)", "coherent=True"]
    # "Forecast all three levels in one paid job"; reconciling sends nothing.
    assert fake.count("POST", "/v1/forecasts") == 1
    assert {row[1] for row in fake.forecasts[0]["history"]["table"]["rows"]} == {
        "total",
        "north",
        "south",
    }


def test_both_programs_fit_a_measured_corner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """F3: every request the page sends lies inside a published corner, by shape."""
    for name, series, variates in (("welt_forecast.py", 1, 2), ("welt_reconcile.py", 3, 1)):
        fake = FakeChronos()
        _run(name, fake, monkeypatch, tmp_path)
        for body in fake.forecasts:
            rows = body["history"]["table"]["rows"]
            ids = {row[1] for row in rows}
            assert len(ids) == series
            assert 1 + len(body["past_covariates"]) + len(body["future_covariates"]) == variates
            context = len(rows) // series
            assert any(
                16 <= context <= c["max_context"]
                and variates <= c["max_variates"]
                and series <= c["max_series"]
                and body["horizon"] <= c["max_horizon"]
                for c in MEASURED_CORNERS
            )
