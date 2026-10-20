# Assumptions behind each step

Every causal answer rests on claims the data cannot check. See how one graph
assumption moves an estimate, then look up what each step of ergodic assumes.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.

## Run the program

The bundled birthweight data (Almond, Chay and Lee, 2005; Cattaneo, 2010) records 4,642 births, the
mother's smoking and 17 background variables. The program asks the same question
under three different graphs.

```python file="assumptions_demo.py"
from ergodic import dag, estimate_effect, identify_effect
from ergodic.datasets import load_birthweight

# 1. Load the births and name the question: smoking's effect on birthweight.
births = load_birthweight()
smoking, weight = births.treatment, births.outcome


def graph(confounders, latent=()):
    edges = [f"{c} -> {smoking}" for c in confounders]
    edges += [f"{c} -> {weight}" for c in confounders]
    return dag([*edges, f"{smoking} -> {weight}"], latent=latent)


# 2. Assume every confounder is measured, and estimate.
full = estimate_effect(
    graph(births.confounders), births.data, smoking, weight,
    treated=births.treated, random_state=0,
)
print("17 confounders:", full)

# 3. Assume there are none: the same data, another answer.
none = estimate_effect(
    graph([]), births.data, smoking, weight, treated=births.treated, random_state=0
)
print("no confounders:", none)

# 4. Assume a hidden common cause: no adjustment can remove it.
hidden = graph([*births.confounders, "U"], latent=["U"])
print(identify_effect(hidden, smoking, weight))
```

```output from="assumptions_demo.py"
17 confounders: EffectEstimate(aipw, analytic: ate=-219.5, se=26.3, 95% CI [-271, -168], n=4642)
no confounders: EffectEstimate(aipw, analytic: ate=-275.3, se=21.2, 95% CI [-317, -234], n=4642)
Identification(not identifiable, ['mbsmoke'] -> ['bweight'])
```

- **Step 2.** With all 17 confounders in the graph, smoking lowers birthweight by
  about 220 g.
- **Step 3.** With none, the estimate is the raw difference, about 275 g. Nothing in
  the data says which graph is right; the 53 g gap is your assumption at work.
- **Step 4.** With an unmeasured cause `U` of both smoking and birthweight, and no
  mediator or instrument to route around it, the effect is not identifiable. ergodic
  says so instead of returning a number.

## Know what each step assumes

| Step | You assume | ergodic checks | ergodic cannot check |
| --- | --- | --- | --- |
| Discovery: PC, GES | Independent rows; every common cause measured; independences in the data come from the graph (faithfulness); the test or score fits the data | Refuses time-series and panel data for these methods | Hidden causes, faithfulness |
| Discovery: FCI, RFCI | As PC, but hidden common causes and selection are allowed | The same | Faithfulness |
| Discovery: LiNGAM | Linear effects, independent non-Gaussian noise, no hidden causes | Nothing about Gaussianity: on Gaussian data the order it returns is arbitrary | Every assumption |
| Testing a graph | Independent rows; the test fits the data | The independences each missing edge implies (`test_graph`) | Edges the graph has, and directions within its equivalence class |
| Identification | The graph is right | Whether the graph identifies the effect, and how | The graph itself |
| Instruments | The graph is right, and effects are linear | Instrument validity in the graph | Linearity |
| Estimation | Overlap: every unit could get either treatment; the models fit, or for doubly robust estimators, one of the two models | Refuses an empty treatment arm | Overlap beyond that, model fit |
| Mediation | No mediator-outcome confounder is affected by the treatment | That the graph certifies it | That the world agrees |

The checks are graph checks. A graph you write down, or one discovery returns, is
still an assumption about the world. [Test a graph against data](test-a-graph.md) can
reject one whose missing edges the data contradict, but a pass doesn't confirm it.

!!! caution "A refutation does not prove an estimate"
    `refute_effect` reruns an estimate under a perturbation, such as a placebo
    treatment. A failed check is a real warning. A passed check proves nothing.

## Read the stated assumptions

Some results carry their assumptions in words, so they travel with the number:

- `identify_mediation` returns a record whose `assumptions` name the sequential
  ignorability and cross-world claims the decomposition needs.
- Quasi-experimental estimates carry a `Design` record. Difference-in-differences
  states parallel trends and no anticipation. Synthetic control states that the
  donors span the treated unit's untreated path, that nothing spills over to the
  donors, and that the pre-period fit is good. An interrupted time series states
  that the pre-period trend would have continued and that no other shock hit at the
  break.
- An instrumental-variable identification says it holds "under a linear
  (parametric) model".

<details markdown="1"><summary>What about selection bias?</summary>

ergodic's graphs can mark selection, and FCI can return it as an undirected edge.
Identification refuses a graph with selection marks, because the adjustment
criterion it uses assumes no selection.

</details>

## Next steps

<div class="cards" markdown="1">

- [Graphs and edge marks](graphs-and-marks.md): what each kind of graph says.
- [Test a graph against data](test-a-graph.md): check a graph's independences on your data.
- [Refute an estimate](refute-an-estimate.md): stress-test a number you have.
- [Estimate an average effect](estimate-average-effect.md): the full estimation walkthrough.

</div>
