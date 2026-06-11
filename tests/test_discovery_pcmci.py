# SPDX-License-Identifier: Apache-2.0
"""Tests for the lagged design matrix and PCMCI+ temporal discovery.

Recovery tests run on simulated VAR worlds with planted structure and fixed
seeds, so every assertion is deterministic. The lagged-frame unit tests check
shapes, alignment, panel stacking, and the spacing warning directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ergodic.data import panel, tabular, timeseries
from ergodic.discovery._lagged import lagged_node, lagged_tabular, parse_lagged
from ergodic.discovery._tsresult import TimeSeriesDiscoveryResult
from ergodic.discovery.algorithms._pcmci import PCMCI
from ergodic.graph import GraphKind
from ergodic.knowledge import DomainKnowledge

if TYPE_CHECKING:
    import numpy.typing as npt

    from ergodic.data import PanelData, TimeSeriesData

# -- simulators ---------------------------------------------------------------


def _var_arrays(seed: int, n: int = 2000) -> dict[str, npt.NDArray[np.float64]]:
    """X[t] = 0.7 X[t-1] + e; Y[t] = 0.8 X[t-1] + 0.5 Y[t-1] + e; Z is AR(1)."""
    rng = np.random.default_rng(seed)
    ex, ey, ez = rng.normal(size=(3, n))
    x, y, z = np.zeros(n), np.zeros(n), np.zeros(n)
    x[0], y[0], z[0] = ex[0], ey[0], ez[0]
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + ex[t]
        y[t] = 0.8 * x[t - 1] + 0.5 * y[t - 1] + ey[t]
        z[t] = 0.6 * z[t - 1] + ez[t]
    return {"X": x, "Y": y, "Z": z}


def _var_world(seed: int, n: int = 2000) -> TimeSeriesData:
    arrays = _var_arrays(seed, n)
    return timeseries({"t": np.arange(n), **arrays}, time="t")


def _var_panel(seed: int, n_entities: int = 40, periods: int = 60) -> PanelData:
    blocks = [_var_arrays(seed + i, periods) for i in range(n_entities)]
    columns = {
        "e": np.repeat(np.arange(n_entities), periods),
        "t": np.tile(np.arange(periods), n_entities),
        "X": np.concatenate([b["X"] for b in blocks]),
        "Y": np.concatenate([b["Y"] for b in blocks]),
        "Z": np.concatenate([b["Z"] for b in blocks]),
    }
    return panel(columns, entity="e", time="t")


def _contemporaneous_world(seed: int, n: int = 2000) -> TimeSeriesData:
    """W[t] = e; V[t] = 0.9 W[t] + 0.5 V[t-1] + e: one contemporaneous link."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    ev = rng.normal(size=n)
    v = np.zeros(n)
    v[0] = 0.9 * w[0] + ev[0]
    for t in range(1, n):
        v[t] = 0.9 * w[t] + 0.5 * v[t - 1] + ev[t]
    return timeseries({"t": np.arange(n), "W": w, "V": v}, time="t")


def _collider_world(seed: int, n: int = 3000) -> TimeSeriesData:
    """A and B are mild AR(1); C[t] = 0.8 A[t] + 0.8 B[t] + e is their collider."""
    rng = np.random.default_rng(seed)
    ea, eb, ec = rng.normal(size=(3, n))
    a, b = np.zeros(n), np.zeros(n)
    a[0], b[0] = ea[0], eb[0]
    for t in range(1, n):
        a[t] = 0.4 * a[t - 1] + ea[t]
        b[t] = 0.4 * b[t - 1] + eb[t]
    c = 0.8 * a + 0.8 * b + ec
    return timeseries({"t": np.arange(n), "A": a, "B": b, "C": c}, time="t")


def _common_driver_world(seed: int, n: int = 2000) -> TimeSeriesData:
    """Z[t-1] drives both X[t] and Y[t]; there is no X-Y link at any lag."""
    rng = np.random.default_rng(seed)
    ex, ey, ez = rng.normal(size=(3, n))
    z = np.zeros(n)
    z[0] = ez[0]
    for t in range(1, n):
        z[t] = 0.6 * z[t - 1] + ez[t]
    x = ex.copy()
    y = ey.copy()
    x[1:] += 0.8 * z[:-1]
    y[1:] += 0.8 * z[:-1]
    return timeseries({"t": np.arange(n), "X": x, "Y": y, "Z": z}, time="t")


def _lag2_world(seed: int, n: int = 3000) -> TimeSeriesData:
    """X is AR(1); Y[t] = 0.8 X[t-2] + 0.5 Y[t-1] + e: the X link sits at lag 2."""
    rng = np.random.default_rng(seed)
    ex, ey = rng.normal(size=(2, n))
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = ex[0], ey[0]
    x[1] = 0.6 * x[0] + ex[1]
    y[1] = 0.5 * y[0] + ey[1]
    for t in range(2, n):
        x[t] = 0.6 * x[t - 1] + ex[t]
        y[t] = 0.8 * x[t - 2] + 0.5 * y[t - 1] + ey[t]
    return timeseries({"t": np.arange(n), "X": x, "Y": y}, time="t")


def _lagged_links(result: TimeSeriesDiscoveryResult) -> set[tuple[str, str, int]]:
    return {key for key in result.links if key[2] >= 1}


# -- the headline recovery: a VAR world, exactly -------------------------------


def test_var_world_recovers_the_lagged_links_exactly() -> None:
    result = PCMCI(max_lag=1, alpha=0.01).fit(_var_world(seed=0))
    assert _lagged_links(result) == {("X", "X", 1), ("X", "Y", 1), ("Y", "Y", 1), ("Z", "Z", 1)}
    assert all(lag >= 1 for _, _, lag in result.links)  # no contemporaneous edges
    assert result.graph.kind is GraphKind.DAG
    assert result.summary_edges() == frozenset({("X", "Y"), ("X", "X"), ("Y", "Y"), ("Z", "Z")})
    assert result.method == "pcmci"
    assert result.test == "linear"
    assert result.n == 1998  # the deep frame at lag 2
    assert result.max_lag == 1
    assert result.diagnostics["mci_truncated"] == 0.0
    assert result.diagnostics["n_lagged_links"] == 4.0
    assert result.diagnostics["n_contemporaneous"] == 0.0


def test_panel_pooling_recovers_the_same_links() -> None:
    result = PCMCI(max_lag=1, alpha=0.01).fit(_var_panel(seed=10))
    assert _lagged_links(result) == {("X", "X", 1), ("X", "Y", 1), ("Y", "Y", 1), ("Z", "Z", 1)}
    assert all(lag >= 1 for _, _, lag in result.links)
    assert result.n == 40 * 58  # 40 entities, 60 periods, deep frame at lag 2


# -- contemporaneous structure ---------------------------------------------------


def test_contemporaneous_adjacency_found_and_lagged_link_killed() -> None:
    result = PCMCI(max_lag=1, alpha=0.01).fit(_contemporaneous_world(seed=1))
    assert _lagged_links(result) == {("V", "V", 1)}  # no spurious lagged W -> V
    assert ("V", "W", 0) in result.links  # undirected pair, keyed once, sorted
    assert result.graph.get_edge("V[t]", "W[t]") is not None


def test_collider_orients_into_the_common_child() -> None:
    result = PCMCI(max_lag=1, alpha=0.01).fit(_collider_world(seed=2))
    edge_ac = result.graph.get_edge("A[t]", "C[t]")
    edge_bc = result.graph.get_edge("B[t]", "C[t]")
    assert edge_ac is not None and edge_ac.directed_pair() == ("A[t]", "C[t]")
    assert edge_bc is not None and edge_bc.directed_pair() == ("B[t]", "C[t]")
    assert result.graph.get_edge("A[t]", "B[t]") is None
    assert ("A", "C", 0) in result.links and ("B", "C", 0) in result.links
    assert result.diagnostics["n_oriented"] == 2.0
    assert result.diagnostics["orientation_conflicts"] == 0.0


def test_common_lagged_driver_leaves_no_xy_link() -> None:
    result = PCMCI(max_lag=1, alpha=0.01).fit(_common_driver_world(seed=3))
    assert _lagged_links(result) == {("Z", "X", 1), ("Z", "Y", 1), ("Z", "Z", 1)}
    assert result.graph.get_edge("X[t]", "Y[t]") is None
    assert ("X", "Y", 0) not in result.links and ("Y", "X", 0) not in result.links


def test_max_lag_is_honest_about_what_it_can_see() -> None:
    data = _lag2_world(seed=4)
    shallow = PCMCI(max_lag=1, alpha=0.01).fit(data)
    assert not any(key[:2] == ("X", "Y") for key in shallow.links)  # the lag-2 link is missed
    deep = PCMCI(max_lag=2, alpha=0.01).fit(data)
    assert ("X", "Y", 2) in deep.links
    assert ("X", "Y", 1) not in deep.links


# -- validation and determinism ---------------------------------------------------


def test_pcmci_rejects_tabular_data() -> None:
    data = tabular({"X": np.arange(20.0), "Y": np.arange(20.0)})
    with pytest.raises(TypeError, match="TabularData"):
        PCMCI().fit(data)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="NoneType"):
        PCMCI().fit()


def test_pcmci_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError, match="alpha"):
        PCMCI(alpha=1.5)
    with pytest.raises(ValueError, match="alpha"):
        PCMCI(alpha=0.0)
    with pytest.raises(ValueError, match="max_lag"):
        PCMCI(max_lag=0)
    with pytest.raises(ValueError, match="max_cond_size"):
        PCMCI(max_cond_size=-1)


def test_pcmci_knowledge_is_not_supported_yet() -> None:
    with pytest.raises(NotImplementedError, match="later phase"):
        PCMCI().fit(_var_world(seed=0, n=50), knowledge=DomainKnowledge())


def test_pcmci_too_short_series_raises() -> None:
    short = timeseries({"t": np.arange(8), "X": np.linspace(0.0, 1.0, 8)}, time="t")
    with pytest.raises(ValueError, match="at least max_lag"):
        PCMCI(max_lag=1).fit(short)


def test_pcmci_truncates_the_deep_frame_when_rows_are_scarce() -> None:
    rng = np.random.default_rng(5)
    data = timeseries(
        {"t": np.arange(9), "X": rng.normal(size=9), "Y": rng.normal(size=9)}, time="t"
    )
    result = PCMCI(max_lag=1, alpha=0.01).fit(data)  # 9 rows: enough for lag 1, not lag 2
    assert result.diagnostics["mci_truncated"] == 1.0
    assert result.n == 8


def test_pcmci_is_deterministic() -> None:
    data = _var_world(seed=6)
    first = PCMCI(max_lag=1, alpha=0.01).fit(data)
    second = PCMCI(max_lag=1, alpha=0.01).fit(data)
    assert first.links == second.links
    assert first.graph == second.graph


# -- the lagged frame ---------------------------------------------------------------


def test_lagged_node_and_parse_lagged_round_trip() -> None:
    assert lagged_node("X", 0) == "X[t]"
    assert lagged_node("X", 3) == "X[t-3]"
    for name, lag in [("X", 0), ("price", 5), ("a_b", 1)]:
        assert parse_lagged(lagged_node(name, lag)) == (name, lag)
    with pytest.raises(ValueError, match="lag"):
        lagged_node("X", -1)
    with pytest.raises(ValueError, match="whitespace"):
        lagged_node("a b", 0)
    for bad in ["X", "X[t+1]", "[t]", "X[t-0]", "X[t-]", "X[t-01]"]:
        with pytest.raises(ValueError, match="lagged"):
            parse_lagged(bad)


def test_lagged_tabular_shapes_and_alignment() -> None:
    n = 30
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    data = timeseries({"t": np.arange(n), "X": x, "Y": y}, time="t")
    frame = lagged_tabular(data, max_lag=2)
    assert frame.n_rows == n - 2
    assert set(frame.variables) == {"X[t]", "X[t-1]", "X[t-2]", "Y[t]", "Y[t-1]", "Y[t-2]"}
    assert np.array_equal(frame.column("X[t]"), x[2:])
    assert np.array_equal(frame.column("X[t-1]"), x[1:-1])  # X shifted by one
    assert np.array_equal(frame.column("X[t-2]"), x[:-2])
    with pytest.raises(ValueError, match="max_lag"):
        lagged_tabular(data, 0)
    with pytest.raises(TypeError, match="TabularData"):
        lagged_tabular(tabular({"X": x}), 1)  # type: ignore[arg-type]


def test_lagged_tabular_preserves_column_types() -> None:
    n = 24
    rng = np.random.default_rng(1)
    data = timeseries(
        {"t": np.arange(n), "X": rng.normal(size=n), "F": np.arange(n) % 2},
        time="t",
        types={"F": "categorical"},
    )
    frame = lagged_tabular(data, max_lag=1)
    assert frame.schema["X[t]"].is_continuous
    assert frame.schema["F[t]"].is_categorical
    assert frame.schema["F[t-1]"].categories == data.schema["F"].categories


def test_lagged_tabular_panel_stacks_and_skips_short_entities() -> None:
    lengths = [12, 9, 3]
    rng = np.random.default_rng(2)
    x = rng.normal(size=sum(lengths))
    data = panel(
        {
            "e": np.concatenate([np.full(k, i) for i, k in enumerate(lengths)]),
            "t": np.concatenate([np.arange(k) for k in lengths]),
            "X": x,
        },
        entity="e",
        time="t",
    )
    with pytest.warns(UserWarning, match="skipped 1 of 3"):
        frame = lagged_tabular(data, max_lag=2)
    assert frame.n_rows == (12 - 2) + (9 - 2)  # the 3-row entity is skipped
    assert np.array_equal(frame.column("X[t]")[:10], x[2:12])  # first entity's windows
    assert np.array_equal(frame.column("X[t-2]")[:10], x[0:10])


def test_lagged_tabular_panel_raises_when_no_rows_survive() -> None:
    data = panel(
        {
            "e": np.repeat(np.arange(3), 2),
            "t": np.tile(np.arange(2), 3),
            "X": np.arange(6.0),
        },
        entity="e",
        time="t",
    )
    with pytest.raises(ValueError, match="no rows survive"):
        lagged_tabular(data, max_lag=1)


def test_lagged_tabular_warns_on_irregular_spacing() -> None:
    steps = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 11])
    data = timeseries({"t": steps, "X": np.arange(10.0)}, time="t")
    with pytest.warns(UserWarning, match="positional"):
        lagged_tabular(data, max_lag=1)
