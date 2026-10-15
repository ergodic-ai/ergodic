# Install and extras

Install ergodic from PyPI, add the extras your work needs, and check what your
environment has. The core install needs only numpy, pandas and networkx.

!!! need "You'll need"
    - Python 3.11 or later.
    - pip or uv.

## Install the core

```sh tab="pip"
pip install "ergodic"
```

```sh tab="uv"
uv add "ergodic"
```

The core covers graphs, domain knowledge, identification, discovery, effect
estimation, process intelligence and the bundled datasets. It never imports an
optional package and makes no network calls.

## Add extras

Each extra adds the packages one part of ergodic needs. Quote the name, so your
shell does not read the brackets as a pattern:

```sh tab="pip"
pip install "ergodic[forecast]"
```

```sh tab="uv"
uv add "ergodic[forecast]"
```

| Extra | Adds | For |
| --- | --- | --- |
| `bayes` | pymc 5 or later | Bayesian learners such as `PyMCLearner` |
| `scm` | pymc 5 or later | Structural causal models (`ergodic.scm`) |
| `forecast` | pymc 5 or later | Forecasting (`ergodic.forecast`) |
| `latent` | pymc 5 or later | Hidden regimes (`ergodic.latent`) |
| `welt` | welt-client 0.13 or later | Welt's hosted models (`ergodic.welt`), Python 3.11 to 3.13 |

`bayes`, `scm`, `forecast` and `latent` add the same package; any one of them is
enough for all four. Combine extras with commas: `pip install "ergodic[scm,welt]"`.

## Check your environment

Run this program to print the installed version and the optional packages it finds.
The recording shows only the first line, because the rest depends on what you
installed.

```python file="install_check.py"
import importlib.util

import ergodic

# 1. Print the installed version and the graph format it writes.
print("ergodic", ergodic.__version__, "writes", ergodic.GRAPH_FORMAT)

# 2. Check which optional packages this environment has: found or missing.
optional = {
    "pymc": "bayes, scm, forecast and latent",
    "jax": "faster sampling, with numpyro",
    "numpyro": "faster sampling, with jax",
    "welt": "welt",
}
for module, use in optional.items():
    status = "found" if importlib.util.find_spec(module) else "missing"
    print(f"{module:8} {status:8} {use}")
```

```output from="install_check.py" lines="1"
ergodic 0.2.0 writes ergodic.graph/1
```

- **Step 1.** The version, and `ergodic.graph/1`: the format `MixedGraph.to_dict()`
  writes. Welt reads this format to convert its graphs.
- **Step 2.** One line per optional package, such as `welt     missing  welt`.
  `missing` means you have not installed the extra in the last column.

## Speed up sampling

Forecasting and hidden-regime models sample with PyMC. When jax and numpyro are
installed, they sample through PyMC's JAX backend, which ergodic's own measurement
found about four times faster on the forecasting model. Without them, sampling
falls back to PyMC's C backend and warns once. Neither extra installs them:

```sh
pip install jax numpyro
```

## Use your own learners

Effect estimation takes any model with scikit-learn's `fit` and `predict` methods
as a learner, and a classifier adds `predict_proba`. ergodic ships its own OLS and
logistic regression and never imports scikit-learn. Install scikit-learn or
xgboost yourself to use their models (the `welt` extra brings scikit-learn in).

!!! note "Python 3.14"
    ergodic installs on Python 3.14, but the `welt` extra adds nothing there:
    welt-client supports Python 3.11 to 3.13.

<details markdown="1"><summary>What are the neural and foundation extras?</summary>

`neural` and `foundation` install PyTorch 2.2 or later. ergodic 0.2.0 has no
learner that uses it: asking `cate` for a neural or foundation-model learner raises
`NotImplementedError`.

</details>

## Next steps

<div class="cards" markdown="1">

- [Your first causal effect](start.md): discover, identify and estimate on bundled data.
- [Discover in Welt, estimate in ergodic](welt.md): use the `welt` extra.
- [Compatibility with Welt](compatibility.md): which versions work together.

</div>
