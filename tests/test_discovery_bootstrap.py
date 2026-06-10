# SPDX-License-Identifier: Apache-2.0
"""Bootstrap stability: edge frequencies under row resampling, over any method."""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd
import pytest

from ergodic import DomainKnowledge, tabular
from ergodic.data import TabularData
from ergodic.discovery._bootstrap import BootstrapResult, bootstrap
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
