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

The pillars are placeholders today. The shared objects they build on are written and tested: `ergodic.graph` (a mixed-graph family covering DAG, ADMG, CPDAG, MAG, and PAG) and `ergodic.knowledge` (`DomainKnowledge` for prior constraints). See the [docs](https://ergodic.ai) for guides.

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
