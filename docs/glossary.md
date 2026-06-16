# Glossary

A plain-language reference for the terms used across these docs. On other pages, hover any
underlined term to see a short definition without leaving the page. This page is the full list,
grouped by topic, with a pointer to where each idea is taught in depth.

New to causal inference? Read the [learn series](learn/index.md) first. This page is a quick
reference, not a course.

## Core causal ideas

- **causal inference**{ #causal-inference } — working out what *causes* what, so you can predict what an action would do, not just what tends to move with what. *See [learn: why causality](learn/01-why-causality.py).*
- **correlation (association)** — two things tending to move together, with no claim that one causes the other.
- **confounder**{ #confounder } — a common cause sitting behind both the treatment and the outcome, which makes them move together and fakes an effect. *See [learn: why causality](learn/01-why-causality.py).*
- **back-door path**{ #back-door } — a non-causal route from treatment to outcome that runs through a common cause and lets association leak through. *See [identification](identification.md).*
- **back-door adjustment** — removing confounding by conditioning on (holding fixed) the variables that block every back-door path.
- **collider**{ #collider } — a node that two arrows point into (X → Z ← Y); a shared effect. A collider blocks a path until you condition on it, which then opens it. *See [learn: good and bad controls](learn/02-good-and-bad-controls.py).*
- **fork** — one variable causing two others; the shape behind a confounder. *See [learn: good and bad controls](learn/02-good-and-bad-controls.py).*
- **chain** — A causes B causes C; the effect passes through the middle variable (a mediator).
- **mediator**{ #mediator } — a variable on the causal path from treatment to outcome that carries part of the effect. *See [identification](identification.md).*
- **selection bias**{ #selection-bias } — distortion that appears when you only keep rows where some condition held; conditioning on that condition can correlate its causes even when they are unrelated. *See [graph](graph.md).*
- **intervention (do(X))**{ #do-operator } — *setting* X yourself rather than just observing it. On a graph this cuts the arrows into X. *See [graph](graph.md).*
- **observational data** — data you collected by watching the world, not by running an experiment.
- **counterfactual** — what would have happened to a unit under the treatment it did not actually get.
- **Simpson's paradox** — a pattern that holds in every subgroup yet flips when you pool the groups. *See [learn: why causality](learn/01-why-causality.py).*
- **Berkson's paradox** — a fake correlation created by looking only at a selected subgroup. *See [learn: good and bad controls](learn/02-good-and-bad-controls.py).*
- **explaining away** — once you know a shared effect happened, its independent causes start to look dependent.
- **interference (SUTVA)** — one unit's treatment affecting another's outcome (for example, through a shared queue). SUTVA is the assumption that this does not happen.
- **exclusion restriction** — the assumption that an instrument reaches the outcome only through the treatment. *See [learn: hidden confounding](learn/05-hidden-confounding.py).*
- **positivity (overlap)**{ #positivity } — every kind of unit appears both treated and untreated, so there is somebody to compare with. *See [learn: estimating effects](learn/03-estimating-effects.py).*

## Identification

- **identification**{ #identification } — deciding whether a causal effect can be computed from the data you have, and if so writing the formula for it, before touching the numbers. *See [identification](identification.md).*
- **estimand**{ #estimand } — the formula identification produces: the effect written in terms of quantities you can actually observe. *See [identification](identification.md).*
- **adjustment set**{ #adjustment-set } — the variables you condition on to close every back-door path. *See [learn: good and bad controls](learn/02-good-and-bad-controls.py).*
- **minimal adjustment set** — the smallest valid adjustment set, so you measure as little as possible.
- **front-door**{ #front-door } — identifying a confounded effect through a fully-mediating, unconfounded middle step. *See [learn: hidden confounding](learn/05-hidden-confounding.py).*
- **instrument (IV)**{ #instrument } — a variable that moves the treatment but reaches the outcome only through the treatment. *See [learn: hidden confounding](learn/05-hidden-confounding.py).*
- **null (zero-effect) strategy** — the verdict that no directed path runs from treatment to outcome, so the effect is zero by structure.
- **do-calculus** — the rule system that decides whether any single-graph effect is identifiable and writes the formula. *See [identification](identification.md).*
- **ID / IDC algorithm** — the complete procedure (Shpitser-Pearl) that implements do-calculus; IDC is its version for a conditional effect.
- **functional** — ergodic's typed estimand object: an algebra of conditional-probability terms (`Prob`, `Product`, `Marginal`, `Ratio`).
- **hedge** — the structure that proves an effect is *not* identifiable (the witness ID returns when no formula exists).
- **bow** — a treatment with both a direct effect and hidden confounding on the same pair; the classic effect you cannot identify.
- **generalized adjustment criterion** — the version of back-door adjustment that holds across every graph in an equivalence class, not just one.
- **amenability** — a class is amenable when every possibly-causal path out of the treatment starts with a visible edge, so one adjustment set works for all its members.
- **visible edge** — a directed edge certified to hide no confounder behind it.
- **possibly causal path / possible ancestor** — a path (or reachability) that could be causal in *some* member of an equivalence class.
- **IDA** — identify an effect across every member of a class and report the set of answers, rather than a single number.
- **mediation (NDE / NIE / CDE)** — splitting a total effect into its direct part, its part through a mediator (indirect), and the direct effect with the mediator held fixed. *See [identification](identification.md).*
- **cross-world quantity** — a counterfactual that compares two hypothetical worlds at once, so no single experiment can check it.
- **recanting witness** — a post-treatment variable affecting both mediator and outcome, which blocks the natural mediation effects.

## Graphs, marks, and equivalence classes

- **causal graph** — a diagram with one node per variable and an arrow for each direct cause. *See [concepts](concepts.md).*
- **DAG** — a directed acyclic graph: one-way arrows, no loops; a fully known causal model. *See [concepts](concepts.md).*
- **ADMG**{ #admg } — a graph that adds two-headed arrows for hidden common causes. *See [learn: hidden confounding](learn/05-hidden-confounding.py).*
- **CPDAG**{ #cpdag } — what constraint-based discovery returns: arrows where the data decides direction, dashes where several DAGs fit equally well. *See [concepts](concepts.md).*
- **MAG**{ #mag } — a graph type that handles hidden confounders and selection bias together. *See [concepts](concepts.md).*
- **PAG**{ #pag } — what FCI returns: a whole class of MAGs, with circle marks where the class cannot decide an endpoint. *See [concepts](concepts.md).*
- **mixed graph** — ergodic's single object whose edges carry a mark on each end, which lets it represent all the kinds above. *See [graph](graph.md).*
- **mark (tail / arrowhead / circle)** — the symbol on an edge end: no-arrow, arrow, or unknown. *See [concepts](concepts.md).*
- **glyph** — the string spelling of an edge, like `A -> B` or `A <-> B`.
- **bidirected edge (`<->`)** — an edge meaning two variables share a hidden common cause.
- **undirected edge (`--`)** — an edge where adjacency is known but direction is not.
- **skeleton**{ #skeleton } — a graph's adjacencies (who is connected to whom) with all arrow directions stripped off. *See [discovery](discovery.md).*
- **v-structure** — a collider whose two parents are not adjacent (X → Z ← Y with no X-Y edge); the fingerprint that discovery can orient from data.
- **unshielded triple** — three nodes in a line A-B-C where A and C are not adjacent.
- **Markov equivalence class** — the set of graphs that imply the same independences, so the data alone cannot tell them apart.
- **face (kind)**{ #face } — which graph type a result presents as: dag, cpdag, pag, or mixed. The "mixed face" means the result could not be certified as a valid member of a named class, often because knowledge (not data) oriented some edges. *See [discovery](discovery.md).*
- **`cpdag_valid` / `pag_valid`** — a 1.0 / 0.0 flag for whether a discovered graph is a valid CPDAG or PAG.
- **d-separation / m-separation** — the graph rule for whether two variables are independent given a conditioning set (m-separation extends it to graphs with bidirected edges). *See [graph](graph.md).*
- **conditioning set** — the variables you hold fixed when asking an independence question.
- **latent (hidden) node**{ #latent } — a real cause that was never measured, so it is not a column in the data. *See [learn: hidden confounding](learn/05-hidden-confounding.py).*
- **latent projection** — building the MAG or ADMG over just the observed variables, after projecting out the latents.
- **Meek rules** — the deterministic rules that orient further edges in a CPDAG after the colliders are found.
- **FCI rules (Zhang's ten rules)** — the orientation rules that complete a PAG, including the rules for selection bias.
- **almost-directed cycle** — a loop that would become a directed cycle if a bidirected edge were read as directed; forbidden in a MAG.
- **ancestral graph / maximality** — the conditions a MAG must satisfy; maximality means every non-adjacent pair can be separated by some set.

## Estimation

- **estimation** — plugging data into an estimand to get a number with its uncertainty. *See [inference](inference.md).*
- **`EffectEstimate`** — ergodic's result object: a point with a standard error, draws, or a posterior, behind one interface.
- **estimator (strategy)** — the object encoding *how* to turn an estimand into a number (for example, `AIPW` or `DML`).
- **learner** — the swappable predict model (a scikit-learn-style fit/predict object) plugged into an estimator's slots. *See [inference](inference.md).*
- **nuisance model**{ #nuisance } — a helper model the estimator needs (the outcome model, the treatment model) but does not itself report as the answer. *See [inference](inference.md).*
- **propensity (score)**{ #propensity } — the modeled probability of getting the treatment given the covariates, written P(treatment | covariates). *See [learn: estimating effects](learn/03-estimating-effects.py).*
- **g-formula (outcome regression)** — model the outcome, predict each unit with and without treatment, and average the gap.
- **IPW (inverse-probability weighting)** — reweight units by one over their treatment probability, to mimic a randomized comparison.
- **AIPW (doubly robust)**{ #aipw } — combine an outcome model and a treatment model; the estimate is right if *either* model is right, and it comes with an exact standard error. The default. *See [inference](inference.md).*
- **doubly robust** — the property that an estimator stays correct if at least one of its two models is correct.
- **DML (double machine learning)**{ #dml } — a cross-fit estimator built to use flexible machine-learning models for its nuisances while keeping valid error bars; handles binary or continuous treatment. *See [learn: estimating effects](learn/03-estimating-effects.py).*
- **meta-learner (S / T / X / R / DR-learner)** — recipes for estimating per-unit (CATE) effects from standard predictive models.
- **CATE**{ #cate } — the conditional average treatment effect: the effect as a function of who the unit is, instead of one average. *See [learn: heterogeneous effects](learn/04-heterogeneous-effects.py).*
- **influence function** — a formula for how much each data point pulls on the estimate; its spread gives the standard error without resampling.
- **Neyman-orthogonality** — an estimator built so that small errors in the nuisance models cancel to first order, which is what lets flexible models keep valid inference.
- **cross-fitting (out-of-fold)** — fit the nuisance models on one slice of the data and score them on another, so flexible models cannot bias the estimate. *See [learn: estimating effects](learn/03-estimating-effects.py).*
- **2SLS / Wald / PLIV** — instrumental-variable estimators: two-stage least squares, the simplest ratio form, and the flexible orthogonal version.
- **ATE / ATT / LATE** — the average effect over everyone / over the units actually treated / for the compliers an instrument moves.
- **standard error** — how much the estimate would wobble if you reran the study. *See [learn: estimating effects](learn/03-estimating-effects.py).*
- **confidence interval** — a range of effect values compatible with the data; the 95% claim is about the procedure across repeats, not this one interval.
- **unbiased** — an estimate centered on the true value on average, even if any single run is noisy.
- **posterior / credible interval** — a Bayesian result: a full distribution over the estimate, and the interval that holds it with a stated probability.
- **Bayesian bootstrap** — reweighting rows with random (Dirichlet) weights to carry sampling uncertainty into a posterior.
- **refutation (placebo test)**{ #refutation } — re-running the whole pipeline with the treatment deliberately broken (for example, scrambled); a trustworthy estimate should then collapse to about zero. *See [case studies](case-studies/index.md).*
- **balance table / standardized mean difference** — a side-by-side of confounder averages in the treated versus untreated groups; the gap divided by the spread is a unit-free measure of imbalance. *See [case study: birthweight](case-studies/01-birthweight.py).*

## Discovery

- **causal discovery (structure learning)** — learning the graph's edges from data instead of drawing them by hand. *See [discovery](discovery.md).*
- **constraint-based** — discovery that orients edges by testing which variables are conditionally independent (PC, FCI).
- **score-based** — discovery that searches for the graph that maximizes a fit score (GES, BOSS, GRaSP).
- **conditional independence (CI) test** — a statistical test of whether X and Y are independent once you hold a set Z fixed.
- **oracle** — an object you query for independence answers: either the data-bound test, or a known graph used as ground truth.
- **separating set (sepset)** — the conditioning set that makes two variables independent; the reason an edge between them is removed.
- **decomposable / score-equivalent** — a score that is a sum of one local term per node (decomposable) and that scores graphs in the same class equally (score-equivalent).
- **PC**{ #pc } — the classic constraint-based algorithm: build the skeleton, find colliders, propagate with Meek rules; returns a CPDAG. *See [discovery](discovery.md).*
- **FCI / RFCI** — discovery that tolerates hidden confounders and returns a PAG; RFCI is the faster relaxation.
- **MMPC / MMHC** — a max-min independence screen that returns just a skeleton; MMHC adds a score search inside it.
- **GES** — Greedy Equivalence Search: add edges then remove them, searching over equivalence classes; returns a CPDAG.
- **HillClimb / tabu** — steepest-ascent search over DAGs with a short memory of recent moves to avoid cycling.
- **A\*** — an exact search for the single best-scoring DAG.
- **BOSS / GRaSP** — order-based searches that rearrange a variable ordering; fast and, in tests, as good as the exact optimum.
- **DirectLiNGAM** — discovery for linear data with non-Gaussian noise, which can orient edges a CPDAG would leave undecided.
- **glasso (graphical lasso)** — a sparse estimate of the inverse covariance matrix whose nonzero entries form a skeleton.
- **PCMCI** — time-series discovery on a lagged frame; the result is a window graph with one node per variable per lag. *See [discovery](discovery.md).*
- **window graph** — a discovered time-series graph whose nodes are variables at lags, like `X[t-1]`.
- **contemporaneous edge** — a same-time-step edge, which has no time order to help orient it.
- **Granger causality** — a forecasting test (do X's past values improve predicting Y); it is confounding-blind, so ergodic returns it as a table, not a graph.
- **BIC / BDeu / EBIC** — penalized-fit scores used to compare graphs (Gaussian, Bayesian-discrete, and the criterion that picks the lasso penalty).
- **bootstrap (stability)** — rerun a method on resampled rows to see which edges keep showing up; a confidence signal, not a posterior.

## Panel and quasi-experimental designs

- **panel data** — the same units observed over the same time periods. *See [panel designs](panel.md).*
- **difference-in-differences (DiD)**{ #did } — subtract the comparison group's before-and-after change from the treated group's, to cancel both fixed differences and the shared calendar. *See [learn: quasi-experiments](learn/07-quasi-experiments.py).*
- **parallel trends**{ #parallel-trends } — the DiD assumption: without the treatment, the two groups' outcomes would have moved up and down together (they can sit at different heights). *See [panel designs](panel.md).*
- **staggered adoption (cohort)** — units that adopt the treatment at different dates, in groups.
- **group-time ATT (Callaway-Sant'Anna)** — for staggered adoption, one small DiD per cohort and period, averaged together with cohort-size weights.
- **TWFE (two-way fixed effects)** — a regression with unit and time dummies; it can be biased when adoption is staggered and the effect changes over time.
- **Goodman-Bacon decomposition** — the result showing TWFE is a weighted mix of small comparisons, with weights that can go negative.
- **event study** — one DiD estimate per distance from adoption (event time), tracing the effect over time and checking for pre-trends.
- **synthetic control**{ #synthetic-control } — build a weighted blend of untreated "donor" units that tracks the treated unit before treatment; the gap afterward is the effect. *See [learn: quasi-experiments](learn/07-quasi-experiments.py).*
- **donor pool** — the untreated units available to blend into a synthetic control.
- **placebo permutation** — refit the method pretending each donor was the treated one, then rank the real unit's gap among them for a p-value.
- **RMSPE** — root-mean-square prediction error; for synthetic control, how closely the blend tracked the treated unit before treatment.
- **interrupted time series (ITS)** — for one series with a dated change and no comparison group, fit the level and slope before and after the break.
- **Newey-West (HAC)** — standard errors that stay valid when a single series' errors are correlated over time.
- **`Design` record** — the plain-words statement of a design's untestable assumptions; the panel-side counterpart of an `Identification`.

## Process intelligence

- **event log**{ #event-log } — the native process record: one row per event, with a case id, an activity, and a timestamp. *See [process intelligence](process.md).*
- **case / activity / timestamp** — the thing flowing through the process, the step that happened, and when it happened.
- **process map (directly-follows graph, DFG)** — counts of how often one activity directly follows another, with the time gaps; the first picture any process tool draws. *See [process intelligence](process.md).*
- **variant** — one distinct activity sequence; cases are grouped by it.
- **lifecycle (start / complete)** — two stamps per step that let you split waiting time from working (service) time.
- **`case_table`** — flattens a log to one row per case (treatment, outcome, covariates measured before the treatment moment) for the causal tools.
- **landmark** — the fixed moment in a case's life where features are measured and the treatment is assessed.
- **post-treatment leakage** — letting information from after the treatment into the adjustment set, which biases the estimate.
- **`kpi_panel`** — aggregates a log into a panel (or time series) of per-period KPIs, ready for the panel designs.
- **process tree**{ #process-tree } — a structured model of a process built from `seq` / `xor` / `par` / `loop` operators, with each activity a leaf. *See [process intelligence](process.md).*
- **seq / xor / par / loop / TAU** — run children in order / pick exactly one / interleave / repeat, plus `TAU`, a silent step that does nothing visible.
- **inductive miner** — the algorithm that recovers a process tree by repeatedly cutting the directly-follows graph; its `noise` knob drops rare arcs.
- **workflow net (Petri net)** — the executable form of a process tree. Tokens sit in *places* and flow through *transitions* (the steps); a *marking* is the current arrangement of tokens, and a transition *fires* to move them. *See [process intelligence](process.md).*
- **soundness (`is_sound`)** — the check that every run can finish, no run leaves a token stranded, and every step is reachable on some path.
- **conformance**{ #conformance } — measuring how well a log's real traces fit a model. *See [process intelligence](process.md).*
- **token replay** — push each trace through the net and count the tokens it had to invent or leave behind; a fast, forgiving fitness score.
- **optimal alignment** — walk a trace and the model together and name each disagreement: a sync move (agree), a model move (a step the trace skipped), or a log move (an extra event).
- **decision point** — an `xor` split read causally: the branch a case took becomes the treatment, and features from before the split are the covariates.
- **`decision_table`** — one row per case for a single split, with the timing discipline built in, ready to estimate the effect of the branch.
- **`PositivityReport` / `overlap_share`** — a per-branch coverage check and its warning flag, so you know when a comparison rests on too little data.

## ergodic objects and entry points

- **`MixedGraph` (and `dag`, `admg`, `cpdag`, `pag`, `mixed`)** — the one graph object and the typed builders that construct and validate it. *See [graph](graph.md).*
- **`DomainKnowledge`** — the immutable set of hard prior constraints (tiers, required and forbidden edges, roots, sinks) that discovery and inference read. *See [domain knowledge](knowledge.md).*
- **tiers** — a layering of variables by order in time; a later tier cannot cause an earlier one.
- **`Dataset` (and `tabular`, `timeseries`, `panel`, `hts`, `event_log`)** — the typed data family; the shape picks which algorithms apply. *See [data](data.md).*
- **`Schema`** — the per-column type record (continuous, categorical, ordinal, datetime, excluded; binary is derived).
- **`identify_effect` / `Identification`** — the call that turns a graph into an estimand, and the record it returns (`identifiable`, `strategy`, `estimand`).
- **`estimate_effect`** — the call that takes a graph, data, a treatment, and an outcome, and returns an `EffectEstimate`. *See [inference](inference.md).*
- **`discover` / `bootstrap`** — the discovery entry point and the resampling-stability wrapper. *See [discovery](discovery.md).*
- **`refute_effect`** — the entry point for refutation checks (placebo treatment and friends).
