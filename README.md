# ergodic

Causal AI in one Python package. ergodic learns causal graphs from data, tells you
whether a question can be answered from them, and estimates the effect with an
interval. It also covers quasi-experiments on panels, process mining on event logs,
structural causal models, and Bayesian forecasting with drivers. ergodic runs
locally; the optional `[welt]` extra calls the hosted Welt service. ergodic is
pre-1.0, so the API can change between minor versions.

**Documentation: [ergodic.dev](https://ergodic.dev)**

## Install

```bash
pip install "ergodic"
# or
uv add "ergodic"
```

Requires Python 3.11 or later. The extras `"ergodic[scm]"`, `"ergodic[forecast]"` and
`"ergodic[latent]"` add PyMC for structural causal models, forecasting and hidden
regimes. `"ergodic[welt]"` connects to
[Welt](https://welt.ergodic.dev), which runs hosted models on your data.

## Your first causal effect

```python
from ergodic import discover, estimate_effect, identify_effect
from ergodic.datasets import load_earthquake

alarm = load_earthquake(n=10_000, seed=0)  # the burglar-alarm network (Pearl, 1988)
graph = discover(alarm.data, method="pc").graph  # learn the graph from data alone
for edge in sorted(edge.glyph() for edge in graph.edges):
    print(edge)

print(identify_effect(graph, "Burglary", "JohnCalls"))  # can the data answer it?
print(estimate_effect(graph, alarm.data, "Burglary", "JohnCalls", treated="True"))
```

```text
Alarm --> JohnCalls
Alarm --> MaryCalls
Alarm <-- Burglary
Alarm <-- Earthquake
Identification(adjustment: P(JohnCalls | do(Burglary)) = sum_z P(JohnCalls | Burglary, z) P(z), with z = {Earthquake})
EffectEstimate(aipw, analytic: ate=0.7982, se=0.0333, 95% CI [0.733, 0.863], n=10000)
```

The true effect, from the network's own tables, is 0.793: inside the interval. The
[documentation](https://ergodic.dev) starts from this program and goes step by step.

## Links

- [Documentation](https://ergodic.dev)
- [Welt](https://welt.ergodic.dev): hosted models for prediction, causal discovery and
  forecasting
- [Changelog](CHANGELOG.md), [contributing](CONTRIBUTING.md), [security](SECURITY.md)

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Made by
[Ergodic](https://ergodic.ai).
