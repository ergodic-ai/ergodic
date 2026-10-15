# Glossary

Short definitions of the terms these pages use, grouped by topic. Each group links
to the page that puts the terms to work.

## Causal ideas

- **Treatment and outcome**: the variable you change and the variable you measure.
- **Confounder**: a common cause of the treatment and the outcome. Left
  unadjusted, it makes them move together without one causing the other.
- **Mediator**: a variable on a causal path from treatment to outcome.
- **Collider**: a node two arrows point into (`X -> Z <- Y`). It blocks a path until
  you condition on it.
- **Intervention, `do(X)`**: setting X yourself rather than observing it. On a graph,
  it removes every arrowhead into X.
- **Overlap (positivity)**: every kind of unit could receive either treatment, so
  each has someone to be compared with.
- **Selection bias**: distortion from keeping only the rows where some condition
  held. Conditioning on it can make its causes look related.

See [Assumptions behind each step](assumptions.md).

## Graphs

- **Mark**: the symbol at one end of an edge: tail, arrowhead or circle.
- **Glyph**: an edge written as a string, such as `A -> B`, `A <-> B` or `A o-> B`.
- **DAG**: directed acyclic graph. Every edge has a direction, and there are no
  cycles.
- **ADMG**: a DAG that adds `<->` edges for hidden common causes.
- **CPDAG**: the graphs that data cannot tell apart, drawn as one graph. Directed
  edges hold in every member; `--` edges point either way.
- **MAG**: a graph over the measured variables that keeps what hidden causes and
  selection imply about independence.
- **PAG**: the MAGs that data cannot tell apart, drawn as one graph. A circle marks
  an end that differs between them.
- **Kind**: which of these readings a graph uses; `mixed` means none is checked.
- **Latent node**: a cause in the graph that is not a column in the data.
- **Markov equivalence class**: the graphs that imply the same independences.
- **v-structure**: a collider whose two parents are not adjacent. Data can orient
  it.
- **d-separation, m-separation**: the graph test for whether two variables are
  independent given a set of others.
- **Graph format**: the version tag `to_dict` writes, `ergodic.graph/1` today.

See [Graphs and edge marks](graphs-and-marks.md).

## Discovery

- **Conditional-independence test**: a test of whether two variables are
  independent given others. PC and FCI are built on one.
- **Score**: a number that rates how well a graph fits the data, such as BIC.
  GES and exact search maximize one.
- **Faithfulness**: every independence in the data comes from the graph, not from
  effects cancelling out.
- **Domain knowledge**: constraints you know before looking at data: required or
  forbidden edges, tiers, roots and sinks.

See [Discover a graph](discover-a-graph.md) and
[Add domain knowledge](domain-knowledge.md).

## Identification

- **Identifiable**: the effect can be computed from the distribution of the
  measured variables, given the graph.
- **Estimand**: the formula identification returns, written in measurable
  quantities.
- **Adjustment set**: variables to condition on so that every non-causal path
  between treatment and outcome is blocked.
- **Front-door**: identifying a confounded effect through mediators that the hidden
  confounder does not reach.
- **Instrument**: a variable that moves the treatment and reaches the outcome only
  through it.
- **Null strategy**: no directed path runs from treatment to outcome, so the effect
  is zero by the graph.
- **ID algorithm**: the general procedure that decides identifiability for any
  single DAG or ADMG; a **hedge** is its proof that an effect is not identifiable.
- **Natural direct and indirect effects**: the part of an effect that does not go
  through a mediator, and the part that does.

See [Check whether an effect is identifiable](identify-an-effect.md).

## Estimation

- **Average treatment effect (ATE)**: the mean outcome if everyone were treated,
  minus the mean if no one were.
- **ATT**: the same contrast, averaged over the treated units only.
- **CATE**: the average effect for units that share given features.
- **Learner**: a model in a nuisance slot, with `fit` and `predict`.
- **Nuisance slot**: a model an estimator needs but does not report, such as the
  outcome or the propensity model.
- **Propensity**: the probability of being treated, given the confounders.
- **AIPW**: augmented inverse-probability weighting. It combines an outcome model
  and a propensity model and stays consistent if either one is right.
- **DML**: double machine learning. It removes what the confounders predict from
  the treatment and the outcome, then regresses one residual on the other.
- **Cross-fitting**: fitting nuisance models on some folds and predicting on the
  others, so no row is predicted by a model trained on it.
- **Refutation**: rerunning an estimate under a perturbation it should survive or
  fail.

See [Estimate an average effect](estimate-average-effect.md).

## Designs over time

- **Difference-in-differences**: compares the change in treated units with the
  change in comparison units; rests on parallel trends.
- **Synthetic control**: a weighted mix of untreated units that tracks the treated
  unit before treatment.
- **Interrupted time series**: compares a series after a break with the trend it
  followed before.

See [Difference-in-differences and event studies](difference-in-differences.md).

## Welt

- **Welt**: Ergodic's hosted service for tabular foundation models, causal
  discovery included.
- **Native graph type**: the kind of graph a Welt model declares it returns. ergodic
  converts only a declared DAG.
- **Fit budget**: `max_fits`, the most paid fits a `WeltLearner` and its copies may
  make.

See [Discover in Welt, estimate in ergodic](welt.md).

## Next steps

<div class="cards" markdown="1">

- [Your first causal effect](start.md): the terms in a single program.
- [Graphs and edge marks](graphs-and-marks.md): marks and kinds in depth.
- [Python reference](reference.md): every function and its parameters.

</div>
