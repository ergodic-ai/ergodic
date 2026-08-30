# Examples

Short, runnable scripts that load a real dataset, run a causal analysis, and print
the estimate next to the answer the literature already measured. Each one uses a
loader from `ergodic.datasets`, so the data comes with its causal question
attached (the treatment, the outcome, the confounders, and the target).

Run any of them after an editable install (`uv sync`):

```
uv run python examples/adjustment_birthweight.py
```

| Script | Dataset | Question | What it shows |
| --- | --- | --- | --- |
| `adjustment_birthweight.py` | birthweight | Does smoking lower birthweight? | Backdoor adjustment, naive -275 g moves to about -210 g |
| `confounding_berkeley.py` | berkeley | Did admissions favor men? | Simpson's paradox: a pooled gap reverses under the confounder |
| `double_ml_pension.py` | pension401k | Effect of 401(k) eligibility on wealth | Double machine learning, $19.5k naive falls to about $9k |
| `synthetic_control_smoking.py` | smoking | Did Proposition 99 cut sales? | Synthetic control with placebo inference |
| `discovery_asia.py` | asia | Recover the network structure | Constraint-based discovery scored against a known graph |

The numbers are checked in `tests/test_datasets.py`, so they stay honest as the
estimators change.

One script is the odd one out. `latent_switch.py` loads no dataset: it generates a
world with a sealed hidden switch, recovers the switch blind with `ergodic.latent`,
certifies that a regime exists against a simulated null, and calibrates the detector
on control worlds. It needs the latent extra (`uv sync --extra latent`) and a few
minutes of sampling.
