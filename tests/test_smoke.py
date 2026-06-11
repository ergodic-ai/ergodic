# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the package skeleton."""

from __future__ import annotations

import importlib

import ergodic


def test_version_is_nonempty_string() -> None:
    assert isinstance(ergodic.__version__, str)
    assert ergodic.__version__


def test_subpackages_import() -> None:
    for name in ("discovery", "inference", "process"):
        module = importlib.import_module(f"ergodic.{name}")
        assert module is not None


def test_process_exports() -> None:
    import ergodic.process

    assert len(ergodic.process.__all__) >= 15
    for name in (
        "case_table",
        "process_map",
        "kpi_panel",
        "discover_model",
        "conformance",
        "decision_points",
        "decision_table",
    ):
        assert callable(getattr(ergodic, name))


def test_model_layer_end_to_end() -> None:
    import pandas as pd

    from ergodic import conformance, decision_points, discover_model, event_log
    from ergodic.process import is_sound, seq, xor

    rows = []
    for i, trace in enumerate(["a x c", "a y c"] * 6):
        for j, act in enumerate(trace.split()):
            rows.append((f"c{i:02d}", act, float(j)))
    log = event_log(
        pd.DataFrame(rows, columns=["case", "act", "ts"]),
        case="case",
        activity="act",
        timestamp="ts",
    )

    model = discover_model(log)
    assert model.tree == seq("a", xor("x", "y"), "c")
    assert is_sound(model.net)
    result = conformance(log, model)
    assert result.fitness == 1.0
    assert result.alignment_fitness == 1.0
    (point,) = decision_points(log, model)
    assert point.branches == ("x", "y")
    assert point.counts == (6, 6)
