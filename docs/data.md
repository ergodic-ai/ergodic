# Data

The data objects are the typed view the algorithms read. You hand in a dataframe;
you get back a dataset that knows the type of every column and the shape of the
rows. Discovery and inference then read that typed view and never guess again.

Two things describe a dataset, and they are kept separate:

- The shape of the rows: tabular, time-series, panel, hierarchical time-series,
  or an event log. This sets which algorithms apply.
- The type of each column: continuous, categorical, ordinal, datetime, or
  excluded. This sets which test or estimator fits a given variable.

Every shape carries the same column schema, so the typing below works the same
way everywhere.

## Column types

`tabular` and the other builders infer a `Schema` from the data. Floats read as
continuous, strings and pandas categoricals as categorical, an ordered pandas
categorical as ordinal, datetimes as datetime. A tz-aware datetime column
converts to UTC and drops the zone, so it types as datetime like a naive one.

```python
import pandas as pd
from ergodic import tabular, Ordinal

df = pd.DataFrame({
    "age": [20.0, 35, 50, 28],
    "city": ["NY", "LA", "NY", "SF"],
    "stage": ["II", "I", "III", "I"],
    "id": [1, 2, 3, 4],
})

ds = tabular(df, types={"stage": Ordinal(["I", "II", "III"])}, exclude=["id"])
ds.schema
# Schema(4 columns: 1 continuous, 1 categorical, 1 ordinal, 1 excluded)
ds.variables
# ('age', 'city', 'stage')
```

Two overrides do the work above. `types` forces a column's type, and `Ordinal`
gives an ordinal column its order. `exclude` keeps a column in the data but out of
the variables, which is what you want for identifiers and free text.

The one type the data can't settle is an integer column, which might be a count
or an encoded category. Low-cardinality integers infer categorical with a
warning; high-cardinality integers infer continuous. An all-distinct text column
warns that it looks like an identifier. A `types` override silences either.

```python
tabular(pd.DataFrame({"dose": [1, 2, 3, 1, 2]}))
# SchemaInferenceWarning: column 'dose': inferred categorical from 3 integer levels.
# Pass types={'dose': 'continuous'} to keep it numeric.
```

## Tabular

`tabular` is the base shape: rows are independent units. It reads a pandas or
polars dataframe, a dict of columns, or a numpy array.

The accessors are what an estimator needs. `codes` gives stable integer codes for
a discrete column, in category order, so an ordinal's order is respected. `matrix`
gives a float design matrix, with discrete columns turned into their codes.

```python
ds.codes("stage")    # array([1, 0, 2, 0])  (I=0, II=1, III=2)
ds.matrix().shape    # (4, 3)
```

Missing values stay visible: a missing category is -1 in `codes` and NaN in
`matrix`, and a missing datetime (NaT) is NaN there too, so a gap can't slip
into an estimator as a real number. Declared categories have to cover the data;
a value outside them raises a `ValueError` naming the column and the stray
values.

The dataset shares its variable names with graphs and knowledge, so it checks
against a graph directly.

```python
from ergodic import dag

g = dag(["age -> city", "stage -> city"])
ds.aligns_with(g)   # True
ds.alignment(g)     # [] when they match, else what differs
```

Datasets are immutable. Retyping a column, selecting columns, dropping rows, or
excluding a column each returns a new dataset.

```python
ds.with_type("age", "continuous")   # new dataset, age forced continuous
ds.select(["age", "stage"])          # new dataset, two columns
ds.dropna()                          # new dataset, rows with any missing value gone
ds.missing()                         # {'age': 0, 'city': 0, ...}
```

## Time-series

`timeseries` names a time column and sorts the rows by it, since the temporal
methods read them in order. The time column is an index, so it is typed (datetime,
or continuous for an integer step) and kept out of the variables.

```python
from ergodic import timeseries

tdf = pd.DataFrame({"t": pd.to_datetime(["2020-03", "2020-01", "2020-02"]), "y": [3.0, 1, 2]})
ts = timeseries(tdf, time="t")
ts.variables            # ('y',)
ts.column("y").tolist() # [1.0, 2.0, 3.0]  (rows now in time order)
```

## Panel

`panel` names an entity column and a time column. The pair keys the rows, so it
has to be unique, and rows come back grouped by entity and ordered in time. The
panel queries are the ones fixed-effects and difference-in-differences need.

```python
from ergodic import panel

pdf = pd.DataFrame({
    "firm": ["B", "A", "B", "A"],
    "year": [2020, 2020, 2021, 2021],
    "country": ["US", "US", "US", "US"],
    "rev": [2.0, 1, 4, 3],
})
p = panel(pdf, entity="firm", time="year")
p.n_entities, p.n_periods   # (2, 2)
p.is_balanced               # True
p.time_invariant()          # ('country',)  -- constant within every firm
```

## Hierarchical time-series

`hts` is a time-series whose columns form an aggregation tree: a parent series is
the sum of its children. That tree is a `Hierarchy`, which you give as a
parent-to-children mapping. The hierarchy is hard structure, so the dataset can
check coherence (does the parent match its children's sum) and build a parent that
has no column of its own.

```python
from ergodic import hts

hdf = pd.DataFrame({
    "t": pd.to_datetime(["2020-01", "2020-02"]),
    "a": [1.0, 2], "b": [4.0, 5], "total": [5.0, 7],
})
h = hts(hdf, time="t", hierarchy={"total": ["a", "b"]})
h.is_coherent()         # True   (total == a + b on every row)
h.coherence()           # []     (the mismatches, empty here)
h.aggregate("total")    # array([5., 7.])  (works even with no 'total' column)
```

The `Hierarchy` object is usable on its own. It validates the tree (one parent
per node, no cycles), and it answers the structure questions.

```python
from ergodic import Hierarchy

tree = Hierarchy.tree({"total": ["a", "b"], "a": ["a1", "a2"]})
tree.leaves          # ('a1', 'a2', 'b')
tree.children_of("total")   # ('a', 'b')
tree.levels()["a1"]  # 2
```

Mean aggregation works too: `Hierarchy.tree(..., aggregation="mean")`. With
mean, a parent is the mean of its direct children. `aggregate` and `coherence`
share that definition, so they agree on trees deeper than one level. Crossed
groupings (region by category) come later.

## Event log

`event_log` names the case, activity, and timestamp columns. Rows are events,
grouped by case and ordered in time. The log reads as the process-mining
primitives a discovery algorithm needs.

```python
from ergodic import event_log

edf = pd.DataFrame({
    "case": ["1", "1", "1", "2", "2"],
    "act": ["A", "B", "C", "A", "C"],
    "ts": [1, 2, 3, 1, 2],
})
log = event_log(edf, case="case", activity="act", timestamp="ts")
log                       # EventLog(cases=2, events=5, activities=3)
log.traces()              # {'1': ('A', 'B', 'C'), '2': ('A', 'C')}
dict(log.variants())      # {('A', 'B', 'C'): 1, ('A', 'C'): 1}
dict(log.directly_follows())  # {('A', 'B'): 1, ('B', 'C'): 1, ('A', 'C'): 1}
```

## The boundary

A dataframe builds a dataset, the way networkx builds a graph, and stays at the
edge. Inside, the data is a plain column store and the math runs in numpy. The
builders take pandas natively and any object with a `to_pandas` method (polars,
arrow), so you can hand in either. Every column is copied on the way in, so
mutating the source dataframe or array afterwards doesn't reach the dataset.

## What's next on the data side

The containers are here, and most of the consumers are too: estimation and
discovery read a `TabularData`, the quasi-experimental designs read a `PanelData`
(`did`, `event_study`, `synthetic_control`) and a `TimeSeriesData`
(`interrupted_time_series`), and temporal discovery reads both time-series shapes
(`discover(..., method="pcmci")`, `granger`). The process pillar now reads an `EventLog` too:
`case_table` projects it into a `TabularData` and `kpi_panel` into a `PanelData` or
`TimeSeriesData`. Still ahead: hierarchical reconciliation over `HierarchicalTSData`.
