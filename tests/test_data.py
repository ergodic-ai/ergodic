# SPDX-License-Identifier: Apache-2.0
"""Tests for the typed dataset family: schema inference, accessors, and edits."""

from __future__ import annotations

import dataclasses
import warnings
from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ergodic.data import (
    HierarchicalTSData,
    Hierarchy,
    Ordinal,
    SchemaInferenceWarning,
    TimeSeriesData,
    event_log,
    hts,
    panel,
    tabular,
    timeseries,
)
from ergodic.graph import dag

# -- inference ---------------------------------------------------------------


def test_infers_basic_kinds_from_pandas() -> None:
    df = pd.DataFrame(
        {
            "age": [20.0, 35.0, 50.0, 28.0],
            "city": ["NY", "LA", "NY", "SF"],
            "smoker": [True, False, True, False],
            "joined": pd.to_datetime(["2020-01-01", "2021-02-01", "2019-05-01", "2022-07-01"]),
        }
    )
    ds = tabular(df)
    assert ds.schema["age"].is_continuous
    assert ds.schema["city"].is_categorical
    assert ds.schema["smoker"].is_binary
    assert ds.schema["joined"].is_datetime
    assert ds.shape == (4, 4)
    assert set(ds.variables) == {"age", "city", "smoker", "joined"}


def test_integer_low_cardinality_warns_categorical() -> None:
    df = pd.DataFrame({"dose": [1, 2, 3, 1, 2, 3, 1, 2]})
    with pytest.warns(SchemaInferenceWarning):
        ds = tabular(df)
    assert ds.schema["dose"].is_categorical
    assert ds.schema["dose"].cardinality == 3


def test_integer_high_cardinality_is_continuous() -> None:
    df = pd.DataFrame({"income": list(range(100))})
    with warnings.catch_warnings():
        warnings.simplefilter("error", SchemaInferenceWarning)
        ds = tabular(df)
    assert ds.schema["income"].is_continuous


def test_id_like_string_column_warns() -> None:
    df = pd.DataFrame({"uid": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]})
    with pytest.warns(SchemaInferenceWarning):
        tabular(df)


def test_ordered_categorical_from_pandas_becomes_ordinal() -> None:
    risk = pd.Categorical(
        ["med", "low", "high", "low"], categories=["low", "med", "high"], ordered=True
    )
    ds = tabular(pd.DataFrame({"risk": risk}))
    assert ds.schema["risk"].is_ordinal
    assert ds.schema["risk"].categories == ("low", "med", "high")
    assert ds.codes("risk").tolist() == [1, 0, 2, 0]


def test_tz_aware_column_is_datetime_with_utc_stamps() -> None:
    eastern = timezone(timedelta(hours=-5))
    stamps = pd.to_datetime(["2020-01-01 00:00", "2020-01-02 00:00"]).tz_localize(eastern)
    ds = tabular(pd.DataFrame({"d": stamps, "x": [1.0, 2.0]}))
    assert ds.schema["d"].is_datetime
    expected = [
        float(pd.Timestamp("2020-01-01 05:00").value),
        float(pd.Timestamp("2020-01-02 05:00").value),
    ]
    assert ds.matrix(["d"])[:, 0].tolist() == expected


# -- overrides ---------------------------------------------------------------


def test_types_override_forces_continuous_and_silences() -> None:
    df = pd.DataFrame({"dose": [1, 2, 3, 1, 2, 3]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", SchemaInferenceWarning)
        ds = tabular(df, types={"dose": "continuous"})
    assert ds.schema["dose"].is_continuous
    assert not ds.schema["dose"].inferred


def test_ordinal_override_orders_codes() -> None:
    df = pd.DataFrame({"stage": ["II", "I", "III", "I"]})
    ds = tabular(df, types={"stage": Ordinal(["I", "II", "III"])})
    col = ds.schema["stage"]
    assert col.is_ordinal
    assert col.ordered
    assert col.categories == ("I", "II", "III")
    assert ds.codes("stage").tolist() == [1, 0, 2, 0]


def test_ordinal_override_rejects_missing_level() -> None:
    with pytest.raises(ValueError, match=r"'stage'.*'IV'"):
        tabular({"stage": ["I", "II", "I", "IV"]}, types={"stage": Ordinal(["I", "II", "III"])})


def test_exclude_keeps_column_out_of_variables() -> None:
    df = pd.DataFrame({"id": [1, 2, 3, 4, 5], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    ds = tabular(df, exclude=["id"])
    assert ds.schema["id"].is_excluded
    assert ds.variables == ("x",)
    assert "id" in ds.names


# -- accessors ---------------------------------------------------------------


def test_codes_and_categories_are_sorted() -> None:
    ds = tabular(pd.DataFrame({"color": ["r", "g", "b", "r"]}))
    assert ds.categories("color") == ("b", "g", "r")
    assert ds.codes("color").tolist() == [2, 1, 0, 2]


def test_codes_rejects_continuous() -> None:
    ds = tabular(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    with pytest.raises(ValueError, match="not categorical"):
        ds.codes("x")


def test_matrix_mixes_continuous_and_codes() -> None:
    ds = tabular(pd.DataFrame({"x": [1.0, 2.0, 3.0], "c": ["a", "b", "a"]}))
    m = ds.matrix()
    assert m.shape == (3, 2)
    assert m[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert m[:, 1].tolist() == [0.0, 1.0, 0.0]


def test_matrix_rejects_excluded() -> None:
    ds = tabular(pd.DataFrame({"id": ["p1", "p2", "p3"], "x": [1.0, 2.0, 3.0]}), exclude=["id"])
    assert ds.matrix().shape == (3, 1)
    with pytest.raises(ValueError, match="excluded"):
        ds.matrix(["id"])


# -- missing data ------------------------------------------------------------


def test_missing_and_dropna() -> None:
    ds = tabular(pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [1.0, 2.0, np.nan]}))
    assert ds.missing() == {"x": 1, "y": 1}
    assert ds.has_missing
    clean = ds.dropna()
    assert clean.n_rows == 1
    assert clean.column("x").tolist() == [1.0]
    assert not clean.has_missing
    assert ds.n_rows == 3  # original unchanged


def test_matrix_missing_category_is_nan() -> None:
    ds = tabular({"c": np.array(["a", None, "b", "a"], dtype=object)})
    assert ds.codes("c").tolist() == [0, -1, 1, 0]
    column = ds.matrix(["c"])[:, 0]
    assert np.isnan(column[1])
    assert column[[0, 2, 3]].tolist() == [0.0, 1.0, 0.0]
    assert ds.missing() == {"c": 1}
    assert ds.dropna().n_rows == 3


def test_matrix_nat_is_nan() -> None:
    ds = tabular(pd.DataFrame({"d": pd.to_datetime(["2020-01-01", "NaT", "2020-03-01"])}))
    column = ds.matrix(["d"])[:, 0]
    assert column[0] == float(pd.Timestamp("2020-01-01").value)
    assert np.isnan(column[1])
    assert column[2] == float(pd.Timestamp("2020-03-01").value)
    assert ds.missing() == {"d": 1}
    assert ds.dropna().n_rows == 2


# -- immutable edits ---------------------------------------------------------


def test_take_resamples_rows_with_repeats_and_order() -> None:
    ds = tabular(pd.DataFrame({"a": [1.0, 2.0, 3.0], "c": ["x", "y", "x"]}))
    boot = ds.take([2, 0, 2, -1])
    assert boot.n_rows == 4
    assert boot.column("a").tolist() == [3.0, 1.0, 3.0, 3.0]
    assert boot.codes("c").tolist() == [0, 0, 0, 0]
    assert boot.schema == ds.schema
    assert ds.n_rows == 3  # the original is untouched


def test_take_validates_positions() -> None:
    ds = tabular(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    with pytest.raises(IndexError, match="out of range"):
        ds.take([3])
    with pytest.raises(IndexError, match="out of range"):
        ds.take([-4])
    with pytest.raises(ValueError, match="one-dimensional"):
        ds.take([[0, 1]])  # type: ignore[list-item]
    assert ds.take([]).n_rows == 0


def test_take_keeps_the_shape_it_was_called_on() -> None:
    """A prefix of a time series is still a time series, hierarchy and all.

    The backtester slices its training folds with ``take``, so the subclass,
    the index columns, the frequency, and the hierarchy all have to survive.
    """
    times = np.arange(12)
    series = timeseries(
        {"t": times, "y": np.arange(12.0), "label": ["a", "b"] * 6}, time="t", freq="MS"
    )
    prefix = series.take(np.arange(8))
    assert type(prefix) is TimeSeriesData
    assert (prefix.time, prefix.freq) == ("t", "MS")
    assert prefix.schema == series.schema
    assert prefix.variables == series.variables
    assert prefix.n_rows == 8
    assert prefix.times.tolist() == list(range(8))
    assert prefix.codes("label").tolist() == [0, 1] * 4

    north, south = np.arange(12.0), np.arange(12.0) * 2.0
    tree = hts(
        {"t": times, "north": north, "south": south, "total": north + south},
        time="t",
        hierarchy=Hierarchy.tree({"total": ["north", "south"]}),
    )
    cut = tree.take(np.arange(8))
    assert type(cut) is HierarchicalTSData
    assert cut.hierarchy == tree.hierarchy
    assert cut.is_coherent()
    assert cut.aggregate("total").tolist() == (north + south)[:8].tolist()


def test_select_subset_and_leaves_original() -> None:
    ds = tabular(pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 4.0, 5.0], "c": ["x", "y", "x"]}))
    sub = ds.select(["c", "a"])
    assert sub.names == ("c", "a")
    assert sub.column("a").tolist() == [1.0, 2.0, 3.0]
    assert ds.names == ("a", "b", "c")
    with pytest.raises(KeyError):
        ds.select(["nope"])


def test_with_type_recomputes_categories_from_data() -> None:
    df = pd.DataFrame({"g": [0, 1, 0, 1, 2]})
    with pytest.warns(SchemaInferenceWarning):
        ds = tabular(df)
    numeric = ds.with_type("g", "continuous")
    assert numeric.schema["g"].is_continuous
    assert not numeric.schema["g"].inferred
    back = numeric.with_type("g", "categorical")
    assert back.schema["g"].is_categorical
    assert back.schema["g"].categories == (0, 1, 2)


def test_with_type_rejects_categories_that_miss_data() -> None:
    ds = tabular({"c": ["a", "b", "a", "z"]})
    with pytest.raises(ValueError, match=r"'c'.*'z'"):
        ds.with_type("c", "categorical", categories=["a", "b"])


def test_frozen_dataset_rejects_mutation() -> None:
    ds = tabular({"x": [1.0, 2.0]})
    with pytest.raises(dataclasses.FrozenInstanceError):
        ds.n_rows = 5  # type: ignore[misc]


# -- constructors ------------------------------------------------------------


def test_from_dict_and_numpy() -> None:
    ds = tabular({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    assert ds.shape == (3, 2)
    assert ds.schema["x"].is_continuous

    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    ds2 = tabular(arr, columns=["a", "b"])
    assert ds2.names == ("a", "b")
    assert ds2.matrix().shape == (2, 2)


def test_numpy_source_mutation_does_not_leak() -> None:
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    ds = tabular(arr, columns=["a", "b"])
    arr[0, 0] = 99.0
    assert ds.column("a").tolist() == [1.0, 3.0]
    assert ds.matrix()[:, 0].tolist() == [1.0, 3.0]


def test_dict_source_mutation_does_not_leak() -> None:
    source = {"x": np.array([1.0, 2.0, 3.0])}
    ds = tabular(source)
    source["x"][1] = 99.0
    assert ds.column("x").tolist() == [1.0, 2.0, 3.0]
    assert ds.matrix()[:, 0].tolist() == [1.0, 2.0, 3.0]


def test_pandas_source_mutation_does_not_leak() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    ds = tabular(df)
    df.iloc[0, 0] = 99.0
    df.iloc[2, 1] = -1.0
    assert ds.column("x").tolist() == [1.0, 2.0, 3.0]
    assert ds.matrix()[:, 1].tolist() == [4.0, 5.0, 6.0]


class _FakeFrame:
    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame({"x": [1.0, 2.0, 3.0]})


def test_accepts_objects_with_to_pandas() -> None:
    ds = tabular(_FakeFrame())
    assert ds.schema["x"].is_continuous
    assert ds.shape == (3, 1)


# -- graph alignment ---------------------------------------------------------


def test_aligns_with_graph() -> None:
    ds = tabular(pd.DataFrame({"X": [1.0, 2.0], "Y": [3.0, 4.0], "Z": [5.0, 6.0]}))
    assert ds.aligns_with(dag(["X -> Y", "Z -> Y"]))
    assert ds.alignment(dag(["X -> Y", "Z -> Y"])) == []

    issues = ds.alignment(dag(["X -> Y", "Y -> W"]))
    assert any("W" in message for message in issues)
    assert any("Z" in message for message in issues)


def test_repr_reports_shape_and_kinds() -> None:
    ds = tabular(pd.DataFrame({"x": [1.0, 2.0, 3.0], "c": ["a", "b", "a"]}))
    text = repr(ds)
    assert "TabularData" in text
    assert "rows=3" in text


# -- time-series -------------------------------------------------------------


def test_timeseries_sorts_and_excludes_time() -> None:
    df = pd.DataFrame(
        {
            "t": pd.to_datetime(["2020-03-01", "2020-01-01", "2020-02-01"]),
            "y": [3.0, 1.0, 2.0],
        }
    )
    ts = timeseries(df, time="t")
    assert ts.time == "t"
    assert ts.variables == ("y",)
    assert ts.schema["t"].is_datetime
    assert ts.column("y").tolist() == [1.0, 2.0, 3.0]  # rows sorted by time


def test_timeseries_integer_step_index_no_warning() -> None:
    df = pd.DataFrame({"step": [2, 0, 1], "y": [20.0, 0.0, 10.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", SchemaInferenceWarning)
        ts = timeseries(df, time="step")
    assert ts.schema["step"].is_continuous
    assert ts.variables == ("y",)
    assert ts.column("y").tolist() == [0.0, 10.0, 20.0]


def test_timeseries_tz_aware_time_column() -> None:
    df = pd.DataFrame(
        {
            "t": pd.to_datetime(["2020-03-01", "2020-01-01", "2020-02-01"]).tz_localize("UTC"),
            "y": [3.0, 1.0, 2.0],
        }
    )
    ts = timeseries(df, time="t")
    assert ts.schema["t"].is_datetime
    assert ts.times.dtype.kind == "M"
    assert ts.column("y").tolist() == [1.0, 2.0, 3.0]


def test_timeseries_missing_time_raises() -> None:
    with pytest.raises(KeyError):
        timeseries(pd.DataFrame({"y": [1.0, 2.0]}), time="t")


def test_timeseries_select_keeps_time() -> None:
    df = pd.DataFrame({"t": [0, 1, 2], "a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    ts = timeseries(df, time="t")
    sub = ts.select(["a"])
    assert "t" in sub.names
    assert sub.variables == ("a",)
    assert sub.time == "t"


# -- panel -------------------------------------------------------------------


def test_panel_keys_sorts_and_counts() -> None:
    df = pd.DataFrame(
        {
            "firm": ["B", "A", "B", "A"],
            "year": [2020, 2020, 2021, 2021],
            "rev": [2.0, 1.0, 4.0, 3.0],
        }
    )
    p = panel(df, entity="firm", time="year")
    assert p.entity == "firm"
    assert p.time == "year"
    assert p.variables == ("rev",)
    assert set(p.index_columns) == {"firm", "year"}
    assert p.column("firm").tolist() == ["A", "A", "B", "B"]  # grouped by entity
    assert p.column("rev").tolist() == [1.0, 3.0, 2.0, 4.0]  # then ordered in time
    assert p.n_entities == 2
    assert p.n_periods == 2
    assert p.is_balanced


def test_panel_rejects_duplicate_keys() -> None:
    df = pd.DataFrame({"id": ["A", "A"], "t": [1, 1], "x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="unique"):
        panel(df, entity="id", time="t")


def test_panel_unbalanced_and_time_invariant() -> None:
    df = pd.DataFrame(
        {
            "id": ["A", "A", "B"],
            "t": [1, 2, 1],
            "group": ["x", "x", "y"],
            "val": [10.0, 20.0, 30.0],
        }
    )
    p = panel(df, entity="id", time="t")
    assert not p.is_balanced
    assert "group" in p.time_invariant()
    assert "val" not in p.time_invariant()


# -- hierarchy ---------------------------------------------------------------


def test_hierarchy_tree_structure() -> None:
    h = Hierarchy.tree({"total": ["a", "b"], "a": ["a1", "a2"]})
    assert set(h.nodes) == {"total", "a", "b", "a1", "a2"}
    assert h.roots == ("total",)
    assert set(h.leaves) == {"b", "a1", "a2"}
    assert h.children_of("total") == ("a", "b")
    assert h.parent_of("a1") == "a"
    assert h.levels()["a1"] == 2
    assert set(h.leaf_descendants("total")) == {"b", "a1", "a2"}


def test_hierarchy_rejects_multiple_parents() -> None:
    with pytest.raises(ValueError, match="parent"):
        Hierarchy.tree({"p1": ["c"], "p2": ["c"]})


def test_hierarchy_rejects_bad_aggregation() -> None:
    with pytest.raises(ValueError, match="aggregation"):
        Hierarchy.tree({"total": ["a"]}, aggregation="median")


# -- hierarchical time-series ------------------------------------------------


def test_hts_detects_coherent_and_incoherent() -> None:
    t = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
    coherent = pd.DataFrame(
        {"t": t, "a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "total": [5.0, 7.0, 9.0]}
    )
    h = hts(coherent, time="t", hierarchy={"total": ["a", "b"]})
    assert h.time == "t"
    assert "t" not in h.variables
    assert h.is_coherent()
    assert h.coherence() == []

    broken = coherent.assign(total=[0.0, 0.0, 0.0])
    assert not hts(broken, time="t", hierarchy={"total": ["a", "b"]}).is_coherent()


def test_hts_aggregate_builds_absent_parent() -> None:
    t = pd.to_datetime(["2020-01-01", "2020-02-01"])
    df = pd.DataFrame({"t": t, "a": [1.0, 2.0], "b": [4.0, 5.0]})
    h = hts(df, time="t", hierarchy={"total": ["a", "b"]})
    assert h.aggregate("total").tolist() == [5.0, 7.0]


def test_hts_requires_leaf_columns() -> None:
    df = pd.DataFrame({"t": [0, 1], "a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="leaves"):
        hts(df, time="t", hierarchy={"total": ["a", "b"]})


def test_hts_nested_mean_coherence_matches_aggregate() -> None:
    tree = Hierarchy.tree({"total": ["a", "mid"], "mid": ["b", "c"]}, aggregation="mean")
    nested = pd.DataFrame(
        {
            "t": [0, 1],
            "a": [0.0, 0.0],
            "b": [2.0, 2.0],
            "c": [4.0, 4.0],
            "mid": [3.0, 3.0],
            "total": [1.5, 1.5],
        }
    )
    h = hts(nested, time="t", hierarchy=tree)
    assert h.is_coherent()
    assert h.aggregate("total").tolist() == [1.5, 1.5]

    # the flat mean over the leaves, which the old aggregate produced
    flat = hts(nested.assign(total=[2.0, 2.0]), time="t", hierarchy=tree)
    assert not flat.is_coherent()
    assert any("total" in issue for issue in flat.coherence())
    assert flat.aggregate("total").tolist() == [1.5, 1.5]


def test_hts_aggregate_recurses_through_absent_parent() -> None:
    tree = Hierarchy.tree({"total": ["a", "mid"], "mid": ["b", "c"]}, aggregation="mean")
    df = pd.DataFrame({"t": [0, 1], "a": [0.0, 0.0], "b": [2.0, 2.0], "c": [4.0, 4.0]})
    h = hts(df, time="t", hierarchy=tree)
    assert h.aggregate("mid").tolist() == [3.0, 3.0]
    assert h.aggregate("total").tolist() == [1.5, 1.5]


# -- event log ---------------------------------------------------------------


def test_event_log_traces_and_variants() -> None:
    df = pd.DataFrame(
        {
            "case": ["1", "1", "1", "2", "2"],
            "act": ["A", "B", "C", "A", "C"],
            "ts": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-01", "2020-01-02"]
            ),
        }
    )
    log = event_log(df, case="case", activity="act", timestamp="ts")
    assert log.n_cases == 2
    assert log.activities == ("A", "B", "C")
    assert log.traces() == {"1": ("A", "B", "C"), "2": ("A", "C")}
    variants = log.variants()
    assert variants[("A", "B", "C")] == 1
    assert variants[("A", "C")] == 1


def test_event_log_directly_follows() -> None:
    df = pd.DataFrame(
        {
            "case": ["1", "1", "1", "2", "2"],
            "act": ["A", "B", "C", "A", "B"],
            "ts": [1, 2, 3, 1, 2],
        }
    )
    dfg = event_log(df, case="case", activity="act", timestamp="ts").directly_follows()
    assert dfg[("A", "B")] == 2
    assert dfg[("B", "C")] == 1


def test_event_log_sorts_events_within_case() -> None:
    df = pd.DataFrame({"case": ["1", "1", "1"], "act": ["C", "A", "B"], "ts": [3, 1, 2]})
    log = event_log(df, case="case", activity="act", timestamp="ts")
    assert log.traces() == {"1": ("A", "B", "C")}


def test_event_log_missing_column_raises() -> None:
    df = pd.DataFrame({"case": ["1"], "act": ["A"]})
    with pytest.raises(KeyError):
        event_log(df, case="case", activity="act", timestamp="ts")
