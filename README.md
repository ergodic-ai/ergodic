# Ergodic

Causal AI: causal discovery, causal inference, and process intelligence, in one Python package.

> **Pre-alpha.** This is early scaffolding. The public API is not settled and will change.

By [ergodic.ai](https://ergodic.ai).

## What's inside

`ergodic` is organized around three pillars, all built on two shared objects.

| Subpackage | Focus |
| --- | --- |
| [`ergodic.discovery`](src/ergodic/discovery) | Causal discovery: learn causal structure (DAGs) from data. |
| [`ergodic.inference`](src/ergodic/inference) | Causal inference: estimate causal effects from data and a causal model. |
| [`ergodic.process`](src/ergodic/process) | Process mining and intelligence: discover and analyze processes from event logs. |

The shared objects are written and tested: `ergodic.graph` (a mixed-graph family covering DAG, ADMG, MAG, CPDAG, and PAG), `ergodic.knowledge` (`DomainKnowledge` for prior constraints), `ergodic.identification` (graph to estimand: adjustment, front-door, instruments; on a CPDAG, MAG, or PAG the generalized adjustment criterion answers for every graph in the class), and `ergodic.data` (typed datasets: tabular, time series, panel, hierarchical, event log).

Causal inference is built through effect estimation: `estimate_effect` turns a graph and a dataset into an effect with its uncertainty, over a library of estimators (doubly robust, double machine learning, the meta-learners, instruments) with sklearn-style learners in every nuisance slot, heterogeneous effects through `cate`, and an optional Bayesian posterior path through PyMC (`pip install ergodic[bayes]`). When treatment varies over time instead of across a graph, the quasi-experimental designs read the panel directly: `did` (canonical and group-time for staggered adoption), `event_study` (leads as the visible pre-trend check), `synthetic_control` (simplex weights with placebo inference), and `interrupted_time_series` (segmented regression with HAC errors), each carrying its identifying assumptions in a `Design` record.

Causal discovery is built on two pluggable ingredients: a conditional independence test is a class constructed with data (the stateful, caching oracle behind PC, FCI, RFCI, and the MMPC skeleton screen, with a GCM test that turns any regression learner into a test), and a decomposable score drives the score-based searches (GES, hill climbing with tabu, exact A*, and the order-based BOSS and GRaSP, held to the exact optimum in tests), with scores for continuous, categorical, and mixed data. `discover(data)` returns a CPDAG with its separating sets and diagnostics (PC ships the conservative and majority collider rules alongside the standard one), `discover(data, method="fci")` returns a PAG that tolerates latent confounders and selection bias both (Zhang's complete ten rules; RFCI is the fast relaxation), `discover(data, method="lingam")` fully orients a DAG when the noise is non-Gaussian, two screens (MMPC, glasso) confine any search through `restrict_to`, a bootstrap wrapper turns any method into edge stability frequencies, domain knowledge enters every search, and a d-separation oracle makes the recovery guarantees executable. Time series and panels get temporal discovery: `discover(data, method="pcmci", max_lag=...)` returns a window graph over lagged nodes like `X[t-1]` with domain knowledge read at every lag, the bootstrap resamples moving blocks (a series) or whole entities (a panel) so stability extends to PCMCI, and `granger` is the predictive baseline, a pair-level table on purpose. The pillars then meet in two calls: `estimate_effect` identifies on the discovered class itself and estimates whenever one estimand covers every member.

Process intelligence reads event logs. The descriptive layer recovers the daily workflow (`process_map` with provenance-tracked simplification, `performance` with a waiting/service split when lifecycle data exists, `variant_table`, and a composable filter vocabulary that says whether it drops events or cases), and the causal layer asks what the tools above can't: `case_table` encodes a log into one row per case with timestamp-gated features, workload confounders, and auto-built `DomainKnowledge` tiers, feeding `discover` and `estimate_effect` unchanged, and `kpi_panel` aggregates a log into the panel that `did`, `event_study`, and `synthetic_control` consume, so a process change becomes a natural experiment. The model layer goes past the map: `discover_model` is an inductive miner that cuts the log into a block-structured process tree and a workflow net that is sound by construction (`is_sound` verifies any net by reachability), `conformance` checks the log against a model by token replay and optimal alignments, and `decision_points` with `decision_table` read an exclusive split causally: the branch taken is a treatment, the adjusters are measured at the moment the case reveals its branch, and positivity ships as a first-class report.

New to causal inference? The docs open with a learn series: ten notebook guides that teach the ideas from zero on commercial examples (Simpson's paradox, bad controls, doubly robust estimation, uplift targeting, instruments and the front door, structure discovery, quasi-experiments with difference-in-differences and synthetic control, process mining over event logs with the maps drawn for real, causal process intelligence, and decision points in a mined process model), each one a simulation with the true answer written in the code and executed live when the docs build. See the [docs](https://ergodic.ai) for the series and the reference guides.

## Installation

```bash
pip install ergodic
# or, with uv:
uv add ergodic
```

Requires Python 3.11+.

## Quick start

```python
from ergodic import dag, DomainKnowledge

# build a causal graph from edge glyphs
g = dag(["Smoking -> Tar", "Tar -> Cancer", "Smoking -> Cancer"])
g.d_separated("Smoking", "Cancer", "Tar")   # False: the direct edge remains
g.do("Tar")                                  # the graph after intervening on Tar

# state prior knowledge and check a graph against it
dk = DomainKnowledge().with_tiers([["Smoking"], ["Tar"], ["Cancer"]])
dk.is_consistent(g)   # True
```

Estimate a causal effect in one call:

```python
import numpy as np
import pandas as pd
from ergodic import dag, tabular, estimate_effect

rng = np.random.default_rng(0)
n = 4000
z = rng.normal(size=n)
a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.8 * z))).astype(float)
y = 2.0 * a + 1.5 * z + rng.normal(size=n)   # the true effect of A on Y is 2.0

g = dag(["Z -> A", "Z -> Y", "A -> Y"])
data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
estimate_effect(g, data, "A", "Y")
# EffectEstimate(aipw, analytic: ate=1.997, se=0.0342, 95% CI [1.93, 2.06], n=4000)
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
# create the dev environment (with docs deps)
uv sync --group docs

# common tasks (run `make help` to list them)
make lint        # ruff lint
make format      # ruff format + autofix
make typecheck   # mypy (strict)
make test        # pytest
make check       # lint + typecheck + test
make docs-serve  # live docs preview at http://127.0.0.1:8000
```

Optional git hooks:

```bash
uv run pre-commit install
```

## License

[Apache-2.0](LICENSE).
