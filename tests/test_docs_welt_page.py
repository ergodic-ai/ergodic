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
