"""The Welt documentation page's program, run against the fake Welt API.

The docs checker never runs a program marked ``requires="…"`` (it needs a Welt key and
spends credits), so this test runs the exact published code against the
``httpx.MockTransport`` fake from ``test_welt.py``: every request it makes is answered
locally, and the call counts the page states are held to what the program sends.
"""

from __future__ import annotations

import contextlib
import io
import runpy
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("welt")
httpx = pytest.importorskip("httpx")

import welt  # noqa: E402
from scripts.docsite.config import load_site  # noqa: E402
from scripts.docsite.examples import programs_of  # noqa: E402
from tests.test_welt import FakeWelt  # noqa: E402


def test_the_welt_page_program_runs_against_the_fake(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site = load_site()
    page = site.pages_dir / "welt.md"
    (program,) = [p for p in programs_of(page, site) if p.name == "welt_effect.py"]
    assert program.requires == "a Welt API key"

    monkeypatch.delenv("WELT_WORKSPACE", raising=False)
    monkeypatch.delenv("WELT_BASE_URL", raising=False)
    fake = FakeWelt()
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
    lines = printed.getvalue().splitlines()

    # The fake's discovery returns a chain over the uploaded column order (the last
    # column left unconnected), so the page's story holds: a path from nprenatal to
    # bweight, identified by adjustment, estimated by DML with four paid fits.
    assert lines[0] == "(400, 5)"
    assert "medu --> nprenatal" in lines and "bweight <-- nprenatal" in lines
    assert "arrow-fixture-1" in lines
    assert any(line.startswith("Identification(adjustment:") for line in lines)
    assert any(line.startswith("EffectEstimate(dml, analytic:") for line in lines)
    assert lines[-1] == "paid fits: 4"

    # "13 paid requests: one discovery job, four fits and eight predictions."
    sent = Counter((method, path.split("/")[2]) for method, path in fake.calls)
    assert sent[("POST", "discoveries")] == 1
    assert sent[("POST", "fits")] == 4
    assert sent[("POST", "predictors")] == 8
    assert sent[("POST", "datasets")] == 5


# CDFM in the fake: an adaptive threshold of 0.3 and four native edges above it, three
# of which (medu -> nprenatal -> bweight -> medu) form a cycle.
_COLUMNS = ["mage", "medu", "nprenatal", "bweight", "fage"]
_CDFM_CELLS = {
    ("medu", "nprenatal"): 0.6,
    ("nprenatal", "bweight"): 0.5,
    ("mage", "medu"): 0.45,
    ("bweight", "medu"): 0.35,
    ("fage", "mage"): 0.2,
}
_NATIVE_EDGES = [pair for pair, value in _CDFM_CELLS.items() if value > 0.3]
_CDFM = {
    "values": [
        [0.0 if a == b else _CDFM_CELLS.get((a, b), 0.05) for b in _COLUMNS] for a in _COLUMNS
    ],
    "edges": _NATIVE_EDGES,
    "decoder": {
        "name": "native suggest_threshold exact calibration over logits",
        "parameters": {"threshold": 0.3, "threshold_mode": "native_adaptive"},
    },
}


def test_the_welt_page_decoder_program_runs_against_the_fake(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site = load_site()
    page = site.pages_dir / "welt.md"
    (program,) = [p for p in programs_of(page, site) if p.name == "welt_decode.py"]
    assert program.requires == "a Welt API key"

    monkeypatch.delenv("WELT_WORKSPACE", raising=False)
    monkeypatch.delenv("WELT_BASE_URL", raising=False)
    fake = FakeWelt(graph_type="directed", native={"cdfm": _CDFM})
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
    lines = printed.getvalue().splitlines()

    # Step 2: greedy at CDFM's native 0.3 keeps three edges and drops bweight -> medu,
    # the weakest edge of the cycle.
    assert lines[:3] == ["bweight <-- nprenatal", "mage --> medu", "medu --> nprenatal"]
    assert lines[3] == "True welt_native"
    assert lines[4] == "0.3 (1,)"
    # Step 3: the native edges, the cycle included.
    assert lines[5] == str(sorted(_NATIVE_EDGES))
    # Step 4: the edge probabilities, decoded again locally.
    assert lines[6].startswith("EdgeProbabilities(independent, 5 variables")
    assert lines[7].startswith("MixedGraph(kind=dag")

    # One upload and one paid discovery job; the local decode sends nothing.
    sent = Counter((method, path.split("/")[2]) for method, path in fake.calls)
    assert sent[("POST", "discoveries")] == 1
    assert sent[("POST", "datasets")] == 1
