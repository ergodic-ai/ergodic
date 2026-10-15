# Contributing to ergodic

Thanks for helping. This page covers the setup, the checks a change has to pass,
and how commits and pull requests are shaped here.

## Set up

Everything runs through [uv](https://docs.astral.sh/uv/). CI pins uv 0.12.8; a recent
uv works locally (very old releases can't read `uv.lock`).

```bash
git clone https://github.com/ergodic-ai/ergodic.git
cd ergodic
make sync            # uv sync --group docs: the dev tools plus the docs toolchain
uvx pre-commit install      # optional: ruff and whitespace hooks on every commit
```

`make sync` creates `.venv/` with the locked dependencies, including PyMC, JAX and
NumPyro for the Bayesian tests. The package supports Python 3.11 to 3.14;
`.python-version` picks 3.12 locally, and `UV_PYTHON=3.11 make check` runs another one.

## Check your change

```bash
make check    # ruff lint, ruff format check, mypy --strict, pytest
make format   # ruff format plus autofixes, if the format check fails
```

CI runs `make check` on Python 3.11, 3.12, 3.13 and 3.14 for every pull request and
every push to `main` and `dev`. Keep all four green before asking for review.

The suite includes PyMC tests. The first run compiles PyTensor's C code into
`~/.pytensor`, so it is slower than the runs after it.

## Build the docs

The documentation site at [docs.ergodic.dev](https://docs.ergodic.dev) is built from
`docsite/` by the builder in `scripts/docsite`.

```bash
make docs            # build the site into site/
make docs-check      # run every page program; check excerpts, recorded outputs and links
make docs-record     # rerun the page programs and rewrite their output panels
make docs-notebooks  # execute every learn and case-study notebook (slow; release CI runs it)
```

If you change behaviour a page program shows, run `make docs-record` and review the
diff of the recorded outputs. If you change a simulation the notebooks call, run
`make docs-notebooks` and check that every number quoted in the prose still matches
the seeded output.

## What a good change looks like

- **Tests with the behaviour.** A behaviour change comes with a test that fails
  without it. Seed every random draw so results are deterministic.
- **Types stay strict.** mypy runs in strict mode over `src/` and `tests/`.
- **The core stays light.** `import ergodic` needs only numpy, pandas and networkx.
  PyMC, torch and other heavy libraries are optional extras, imported lazily inside
  the code that needs them. networkx appears only at the interop boundary
  (`graph/_io.py`); the algorithms are native.
- **Value objects stay immutable.** Graphs, knowledge and datasets return new
  objects from every edit.
- **Public names are documented.** A new public function or class gets a docstring
  and a place in the docs.
- **Licence headers.** Python files in `src/`, `tests/` and `examples/` start with
  `# SPDX-License-Identifier: Apache-2.0`.
- **CHANGELOG.** Add a line under `## [Unreleased]` in `CHANGELOG.md` for anything a
  user would notice.
- **Docs prose** is plain and direct: short sentences, plain words, contractions are
  fine, and no em dashes.

## Third-party code and data

ergodic is Apache-2.0. Don't copy code from projects under licences that can't be
relicensed that way (GPL, AGPL, LGPL, or no licence at all). Process mining is a
common trap: pm4py is AGPL, so ergodic's miner and conformance checking are written
from the papers, not ported. If you implement a published method, cite the paper in
the docstring.

Bundled datasets must allow redistribution. Record the source and licence of any new
dataset in `NOTICE`. Never commit credentials, customer data, private datasets or
generated artifacts (model outputs, caches, build directories).

## Commits and pull requests

- Branch from `main` and open the pull request against `main`.
- Write commit subjects in the imperative, prefixed with the area they touch, for
  example `discovery: add the RFCI orientation rules` or `docs: fix the panel guide
  example`. Use the body to say why, not what.
- Keep a pull request to one topic. Describe what changed, why, and how you tested
  it; the template lists the checks.
- By contributing you agree that your contribution is licensed under the Apache
  License 2.0, as section 5 of the licence describes.

## Reporting bugs and security issues

Open a [GitHub issue](https://github.com/ergodic-ai/ergodic/issues) for bugs and
feature requests. Report security vulnerabilities privately, as
[SECURITY.md](SECURITY.md) describes, never in a public issue.
