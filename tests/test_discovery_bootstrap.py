# SPDX-License-Identifier: Apache-2.0
"""Bootstrap stability: edge frequencies under resampling, over any method.

Tabular data gets the classic row resample. The temporal designs are pinned
here too: moving blocks become panel entities for a time series under
``method="pcmci"``, and a panel resamples whole entities (the cluster
bootstrap), with the sample builder unit-tested directly.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd
import pytest

from ergodic import DomainKnowledge, tabular
from ergodic.data import PanelData, TabularData, TimeSeriesData, panel, timeseries
from ergodic.discovery._bootstrap import BootstrapResult, _temporal_sample, bootstrap
from ergodic.graph import GraphKind

TRUE_PAIRS = {frozenset({"X", "Y"}), frozenset({"Y", "Z"})}


def _sem(n: int = 400, seed: int = 0) -> TabularData:
    """X -> Y -> Z with unit coefficients, plus W independent of everything."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 1.0 * x + rng.normal(size=n)
    z = 1.0 * y + rng.normal(size=n)
    w = rng.normal(size=n)
    return tabular(pd.DataFrame({"X": x, "Y": y, "Z": z, "W": w}))


@cache
def _pc_run() -> BootstrapResult:
    """One shared PC bootstrap on the SEM; cached because several tests read it."""
    return bootstrap(_sem(), n_bootstrap=30, seed=1)


def _hand_made() -> BootstrapResult:
    """A result with known frequencies, to pin the summary logic exactly."""
    return BootstrapResult(
        method="pc",
        n_runs=10,
        n_failed=2,
        edges={frozenset({"A", "B"}): 0.9, frozenset({"B", "C"}): 0.4},
        marks={"A --> B": 0.5, "A <-- B": 0.4, "B --- C": 0.4},
        variables=("A", "B", "C", "D"),
    )


# -- determinism ----------------------------------------------------------------


def test_the_same_seed_reproduces_the_run_exactly() -> None:
    data = _sem()
    first = bootstrap(data, n_bootstrap=20, seed=3)
    second = bootstrap(data, n_bootstrap=20, seed=3)
    assert first.edges == second.edges
    assert first.marks == second.marks
    assert list(first.marks) == list(second.marks)  # insertion order is pinned too
    assert (first.n_runs, first.n_failed) == (second.n_runs, second.n_failed)
    bootstrap(data, n_bootstrap=20, seed=4)  # a different seed may differ; it just runs


# -- signal on an easy SEM --------------------------------------------------------


def test_strong_edges_are_stable_and_absent_pairs_are_not() -> None:
    result = _pc_run()
    assert result.method == "pc"
    assert result.n_runs + result.n_failed == 30
    assert result.frequency("X", "Y") >= 0.9
    assert result.frequency("Y", "Z") >= 0.9
    assert result.frequency("X", "W") <= 0.2
    noise = [freq for pair, freq in result.edges.items() if pair not in TRUE_PAIRS]
    assert all(freq <= 0.2 for freq in noise)
    assert all(0.0 < freq <= 1.0 for freq in result.edges.values())


def test_glyph_frequencies_sum_to_the_adjacency_frequency() -> None:
    result = _pc_run()
    spread: dict[frozenset[str], float] = {}
    for glyph, freq in result.marks.items():
        parts = glyph.split()
        pair = frozenset((parts[0], parts[-1]))
        spread[pair] = spread.get(pair, 0.0) + freq
    assert set(spread) == set(result.edges)
    for pair, total in spread.items():
        assert abs(total - result.edges[pair]) < 1e-9


# -- the stability graph ----------------------------------------------------------


def test_summary_recovers_the_true_skeleton_at_the_default_threshold() -> None:
    graph = _pc_run().summary()
    assert graph.kind is GraphKind.MIXED
    assert graph.nodes == frozenset({"X", "Y", "Z", "W"})  # W survives isolated
    assert {frozenset((e.u, e.v)) for e in graph.edges} == TRUE_PAIRS


def test_summary_thresholds_on_known_frequencies() -> None:
    result = _hand_made()
    at_default = result.summary()
    assert {frozenset((e.u, e.v)) for e in at_default.edges} == {frozenset({"A", "B"})}
    assert at_default.nodes == frozenset({"A", "B", "C", "D"})  # C and D isolated
    edge = at_default.get_edge("A", "B")
    assert edge is not None
    assert edge.directed_pair() == ("A", "B")  # the modal glyph, 0.5 over 0.4
    kept = result.summary(threshold=0.4)  # the threshold is inclusive
    assert {frozenset((e.u, e.v)) for e in kept.edges} == {
        frozenset({"A", "B"}),
        frozenset({"B", "C"}),
    }
    dropped = result.summary(threshold=0.41)  # just above the weak pair's frequency
    assert {frozenset((e.u, e.v)) for e in dropped.edges} == {frozenset({"A", "B"})}


def test_summary_breaks_glyph_ties_by_the_sorted_glyph_string() -> None:
    result = BootstrapResult(
        method="pc",
        n_runs=10,
        n_failed=0,
        edges={frozenset({"A", "B"}): 0.9},
        marks={"A --> B": 0.45, "A <-- B": 0.45},
        variables=("A", "B"),
    )
    edge = result.summary().get_edge("A", "B")
    assert edge is not None
    assert edge.glyph() == "A --> B"  # "A --> B" sorts before "A <-- B"


def test_summary_validates_the_threshold() -> None:
    result = _hand_made()
    with pytest.raises(ValueError, match="threshold"):
        result.summary(threshold=0.0)
    with pytest.raises(ValueError, match="threshold"):
        result.summary(threshold=1.5)
    assert result.summary(threshold=1.0).nodes == frozenset({"A", "B", "C", "D"})


# -- other methods and kwargs ------------------------------------------------------


def test_a_score_method_runs_through_the_same_wrapper() -> None:
    result = bootstrap(_sem(), method="ges", n_bootstrap=20, seed=2)
    assert result.method == "ges"
    assert result.n_runs + result.n_failed == 20
    assert result.frequency("X", "Y") >= 0.8
    assert result.frequency("Y", "Z") >= 0.8
    stable = {frozenset((e.u, e.v)) for e in result.summary().edges}
    assert stable >= TRUE_PAIRS


def test_kwargs_flow_to_the_algorithm_and_stay_deterministic() -> None:
    data = _sem()
    strict = bootstrap(data, n_bootstrap=10, seed=5, alpha=0.01, max_cond_size=1)
    again = bootstrap(data, n_bootstrap=10, seed=5, alpha=0.01, max_cond_size=1)
    assert strict.edges == again.edges
    assert strict.n_runs + strict.n_failed == 10


# -- failure paths ------------------------------------------------------------------


def test_every_replicate_failing_raises_with_the_chained_cause() -> None:
    # PC never outputs a bidirected edge, so confound knowledge fails its
    # final validation on every replicate, whatever the resample.
    impossible = DomainKnowledge().confound("X", "W")
    with pytest.raises(RuntimeError, match="every bootstrap replicate failed") as caught:
        bootstrap(_sem(n=120), n_bootstrap=3, seed=0, knowledge=impossible)
    assert isinstance(caught.value.__cause__, ValueError)


def test_frequency_returns_zero_for_a_never_seen_pair() -> None:
    result = _pc_run()
    assert result.frequency("X", "W") == 0.0  # real variables, never adjacent
    assert result.frequency("X", "nope") == 0.0  # not even a variable


def test_n_bootstrap_must_be_at_least_one() -> None:
    data = _sem(n=50)
    with pytest.raises(ValueError, match="n_bootstrap"):
        bootstrap(data, n_bootstrap=0)


def test_repr_is_short_and_counts_the_stable_pairs() -> None:
    assert repr(_hand_made()) == "BootstrapResult(pc: 10 runs, 2 failed, 1/2 edges stable at 0.5)"


# -- the temporal designs: moving blocks and whole entities ---------------------------


def _var_ts(n: int = 400, seed: int = 0) -> TimeSeriesData:
    """The pcmci test world: X[t] = 0.7 X[t-1] + e; Y[t] = 0.8 X[t-1] + 0.5 Y[t-1] + e."""
    rng = np.random.default_rng(seed)
    ex, ey = rng.normal(size=(2, n))
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = ex[0], ey[0]
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + ex[t]
        y[t] = 0.8 * x[t - 1] + 0.5 * y[t - 1] + ey[t]
    return timeseries({"t": np.arange(n), "X": x, "Y": y}, time="t")


def _var_panel(seed: int = 0, n_entities: int = 12, periods: int = 40) -> PanelData:
    """The same VAR world per entity, stacked into a panel."""
    blocks = [_var_ts(periods, seed + i) for i in range(n_entities)]
    columns = {
        "e": np.repeat(np.arange(n_entities), periods),
        "t": np.tile(np.arange(periods), n_entities),
        "X": np.concatenate([b.column("X") for b in blocks]),
        "Y": np.concatenate([b.column("Y") for b in blocks]),
    }
    return panel(columns, entity="e", time="t")


def _toy_panel() -> PanelData:
    """Three entities with unequal lengths and recognizable values."""
    lengths = [10, 12, 14]
    return panel(
        {
            "e": np.concatenate([np.full(k, f"u{i}") for i, k in enumerate(lengths)]),
            "t": np.concatenate([np.arange(k) for k in lengths]),
            "X": np.concatenate([100.0 * i + np.arange(k) for i, k in enumerate(lengths)]),
        },
        entity="e",
        time="t",
    )


@cache
def _pcmci_run() -> BootstrapResult:
    """One shared moving-block PCMCI bootstrap; several tests read it."""
    return bootstrap(_var_ts(), method="pcmci", n_bootstrap=10, max_lag=1, alpha=0.01, seed=7)


def test_moving_block_bootstrap_keeps_the_planted_lagged_link() -> None:
    result = _pcmci_run()
    assert result.method == "pcmci"
    assert result.n_runs == 10 and result.n_failed == 0
    # Measured at seed 7: the planted link and both self links hold at 1.0,
    # and the only noise is a contemporaneous X[t] - Y[t] pair at 0.2.
    assert result.marks["X[t-1] --> Y[t]"] >= 0.9
    assert result.frequency("X[t-1]", "Y[t]") >= 0.9
    assert result.frequency("X[t-1]", "X[t]") >= 0.9
    assert result.frequency("Y[t-1]", "Y[t]") >= 0.9
    assert result.frequency("X[t]", "Y[t]") <= 0.4  # no planted contemporaneous edge
    edge = result.summary(threshold=0.9).get_edge("X[t-1]", "Y[t]")
    assert edge is not None and edge.directed_pair() == ("X[t-1]", "Y[t]")


def test_the_same_seed_reproduces_the_temporal_run_exactly() -> None:
    again = bootstrap(_var_ts(), method="pcmci", n_bootstrap=10, max_lag=1, alpha=0.01, seed=7)
    result = _pcmci_run()
    assert result.edges == again.edges
    assert result.marks == again.marks
    assert list(result.marks) == list(again.marks)


def test_panel_cluster_bootstrap_keeps_the_planted_lagged_link() -> None:
    result = bootstrap(_var_panel(), method="pcmci", n_bootstrap=5, max_lag=1, alpha=0.01, seed=3)
    assert result.n_runs == 5 and result.n_failed == 0
    assert result.marks["X[t-1] --> Y[t]"] >= 0.8  # measured 1.0 at seed 3


def test_temporal_sample_blocks_are_contiguous_slices_of_the_series() -> None:
    n = 50
    x = np.arange(float(n))  # a ramp, so contiguity is visible in the values
    y = np.random.default_rng(3).normal(size=n)
    data = timeseries({"t": np.arange(n), "X": x, "Y": y}, time="t")
    sample = _temporal_sample(data, np.random.default_rng(0), block_length=7, max_lag=1)
    assert isinstance(sample, PanelData)
    assert sample.n_entities == 8  # ceil(50 / 7) blocks, one entity each
    assert sample.n_rows == 8 * 7
    entity_column = sample.column(sample.entity)
    for label in sample.entities:
        rows = np.flatnonzero(entity_column == label)
        assert rows.size == 7
        xs = sample.column("X")[rows]
        start = int(xs[0])
        assert np.array_equal(xs, x[start : start + 7])  # one contiguous slice
        assert np.array_equal(sample.column("Y")[rows], y[start : start + 7])
        assert np.array_equal(sample.column(sample.time)[rows], np.arange(7))  # fresh stamps


def test_temporal_sample_default_block_length_formula() -> None:
    data = _var_ts(n=100)
    sample = _temporal_sample(data, np.random.default_rng(1), max_lag=3)
    assert sample.n_entities == 10  # L = max(ceil(sqrt(100)), 2 * (3 + 1)) = 10
    assert sample.n_rows == 100
    deep = _temporal_sample(data, np.random.default_rng(1), max_lag=6)
    assert deep.n_entities == 8  # L = max(10, 2 * 7) = 14, ceil(100 / 14) = 8
    assert deep.n_rows == 8 * 14


def test_panel_cluster_resample_keeps_entities_whole() -> None:
    data = _toy_panel()
    sample = _temporal_sample(data, np.random.default_rng(5))
    assert isinstance(sample, PanelData)
    assert sample.n_entities == 3
    assert sample.entities == ("entity_0", "entity_1", "entity_2")  # relabeled in draw order
    sample_entity = sample.column(sample.entity)
    data_entity = data.column(data.entity)
    matched: list[str] = []
    for label in sample.entities:
        rows = np.flatnonzero(sample_entity == label)
        xs = sample.column("X")[rows]
        hits = [
            source
            for source in data.entities
            if np.array_equal(xs, data.column("X")[np.flatnonzero(data_entity == source)])
        ]
        assert len(hits) == 1  # the rows are exactly one input entity, whole and in order
        source_rows = np.flatnonzero(data_entity == hits[0])
        times = data.column(data.time)[source_rows]
        assert np.array_equal(sample.column(sample.time)[rows], times)
        matched.append(str(hits[0]))
    assert len(set(matched)) == 2  # seed 5 draws one entity twice; labels keep them distinct


def test_block_length_on_tabular_data_raises() -> None:
    with pytest.raises(ValueError, match="block_length is for temporal data"):
        bootstrap(_sem(n=50), n_bootstrap=2, block_length=5)


def test_block_length_on_a_panel_raises() -> None:
    with pytest.raises(ValueError, match="entities whole"):
        bootstrap(_toy_panel(), method="pcmci", n_bootstrap=2, block_length=5)
    with pytest.raises(ValueError, match="entities whole"):
        _temporal_sample(_toy_panel(), np.random.default_rng(0), block_length=5)


def test_a_temporal_method_on_tabular_rows_raises() -> None:
    # The old refusal, rewritten: pcmci now bootstraps, what it refuses is
    # tabular rows, which carry no order for the moving blocks to respect.
    with pytest.raises(ValueError, match="moving-block"):
        bootstrap(_sem(n=50), method="pcmci", n_bootstrap=2)


def test_an_iid_method_on_temporal_data_still_refuses() -> None:
    # The exchangeability guard lives in the registry and fires through bootstrap.
    with pytest.raises(TypeError, match="not exchangeable"):
        bootstrap(_var_ts(n=60), method="pc", n_bootstrap=2)


def test_degenerate_block_lengths_raise() -> None:
    data = _var_ts(n=40)
    with pytest.raises(ValueError, match="out-span"):
        _temporal_sample(data, np.random.default_rng(0), block_length=1, max_lag=1)
    with pytest.raises(ValueError, match="too short"):
        _temporal_sample(data, np.random.default_rng(0), block_length=25, max_lag=1)
    with pytest.raises(ValueError, match="too short"):
        bootstrap(data, method="pcmci", n_bootstrap=2, block_length=25, max_lag=1)
