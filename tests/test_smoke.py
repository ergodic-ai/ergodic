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


def test_cate_facade_end_to_end() -> None:
    import numpy as np
    import pandas as pd

    from ergodic import CATEResult, ITEResult, PerUnitEffect, cate, dag, tabular
    from ergodic.inference import Heterogeneity

    rng = np.random.default_rng(0)
    n = 1500
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.5 * z))).astype(float)
    y = (1.0 + z) * a + z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    graph = dag(["Z -> A", "Z -> Y", "A -> Y"])

    result = cate(graph, data, "A", "Y", random_state=0)
    assert isinstance(result, CATEResult)
    assert isinstance(result, PerUnitEffect)
    assert len(result) == n
    assert abs(result.ate().value - 1.0) < 0.2  # planted ATE
    assert np.corrcoef(result.values, 1.0 + z)[0, 1] > 0.5

    explanation = result.explain()
    assert isinstance(explanation, Heterogeneity)
    assert explanation.calibration_test().rejects()  # real heterogeneity is present

    # the structural ITE shares the per-unit surface
    ite = ITEResult(
        treatment="A",
        outcome="Y",
        covariates=(),
        n=3,
        kind="point",
        source="scm",
        _values=np.array([1.0, 2.0, 3.0]),
    )
    assert ite.shape == (3,)
    assert float(np.mean(ite)) == 2.0
