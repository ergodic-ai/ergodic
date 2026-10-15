# Identification

Before you can estimate an effect, one question has to be settled: can the data you have answer it at all? That is identification. Given a graph, a treatment, and an outcome, it decides whether the effect of *setting* the treatment (written `do(X)`, an intervention, rather than merely observing it) can be recovered from data you only watched, and if it can, writes the formula. The work is graph-only: no data and no estimator yet, just the structure.

## One call

`identify_effect` settles the trivial case first (no directed path from treatment to outcome, see below), then tries the strategies in order (adjustment, then front-door, then instruments) and returns a record with the strategy, the sets involved, and a readable estimand.

```python
from ergodic import dag, admg, identify_effect

identify_effect(dag(["Z -> X", "Z -> Y", "X -> Y"]), "X", "Y")
# Identification(adjustment: P(Y | do(X)) = sum_z P(Y | X, z) P(z), with z = {Z})
```

The record is truthy when the effect is identifiable, so it reads well in a branch:

```python
result = identify_effect(graph, "X", "Y")
if result:
    print(result.strategy, result.estimand)
```

## Adjustment

The most common case: find a set you can condition on to read the effect off the data. `adjustment_set` returns the canonical valid set, or `None` when no set works.

```python
from ergodic import adjustment_set
from ergodic.identification import is_valid_adjustment_set, minimal_adjustment_set

g = dag(["Z -> X", "Z -> Y", "X -> Y"])
adjustment_set(g, "X", "Y")              # frozenset({'Z'})
is_valid_adjustment_set(g, "X", "Y", set())  # False: the back-door stays open
```

This uses the complete adjustment criterion, which is more than the back-door criterion: it finds a valid set whenever one exists. Two things it gets right that catch people out:

```python
# a pure mediator needs no adjustment, and adjusting on it would block the effect
m = dag(["X -> M", "M -> Y"])
adjustment_set(m, "X", "Y")                   # frozenset(): nothing to adjust for
is_valid_adjustment_set(m, "X", "Y", {"M"})   # False: M is on the causal path
```

`minimal_adjustment_set` prunes the canonical set to a minimal one:

```python
g = dag(["W -> Z", "Z -> X", "Z -> Y", "X -> Y"])
adjustment_set(g, "X", "Y")          # frozenset({'Z', 'W'})
minimal_adjustment_set(g, "X", "Y")  # frozenset({'Z'})
```

## Front-door

When the confounder is unobserved, adjustment fails, but a mediator that carries the whole effect can still identify it.

```python
g = admg(["X -> M", "M -> Y", "X <-> Y"])   # <-> is unobserved confounding
adjustment_set(g, "X", "Y")                 # None
identify_effect(g, "X", "Y").strategy       # 'frontdoor'
```

`frontdoor_set` returns the mediator set, and `is_frontdoor_set` checks a candidate you supply.

## Instruments

An instrument affects the outcome only through the treatment, which is enough to identify the effect even with unobserved confounding between treatment and outcome. One caveat: this identification is parametric. Adjustment and the front-door write the effect nonparametrically; an instrument pins it down under a linear model, and the estimand says so.

```python
from ergodic.identification import instruments, is_instrument

g = dag(["I -> X", "X -> Y", "U -> X", "U -> Y"], latent=["U"])
is_instrument(g, "I", "X", "Y")   # True
instruments(g, "X", "Y")          # [('I', frozenset())]
```

The pair is the instrument and the conditioning set it needs. Add a direct path from the instrument to the outcome and it stops being valid:

```python
g = dag(["I -> X", "I -> Y", "X -> Y", "U -> X", "U -> Y"], latent=["U"])
is_instrument(g, "I", "X", "Y")   # False
```

A descendant of the treatment is never an instrument. A child Z of X passes the two textbook checks on their own: it correlates with X, and once X's outgoing edges are cut it separates from Y. Certifying it anyway would hand a Wald estimator a confidently wrong number, so `is_instrument` rejects any candidate, and any conditioning node, that sits downstream of the treatment.

## Latent nodes

A graph can mark nodes as latent (`dag(edges, latent=["U"])`). A latent node shapes the paths like any other node, but you can't condition on what you can't observe, so identification never returns one: not in an adjustment set, not as a mediator, not as an instrument or a conditioning node. When every valid set would need a latent node, the strategy reports failure instead.

```python
g = dag(["U -> X", "U -> Y", "X -> Y"], latent=["U"])
adjustment_set(g, "X", "Y")   # None: U is the only node that would close the back door
```

## No causal path

When no directed path runs from the treatment to the outcome, the intervention can't move the outcome. Then P(Y | do(X)) = P(Y), the effect is zero, and the case identifies trivially. `identify_effect` settles this before trying any strategy and reports it as strategy `"null"`:

```python
result = identify_effect(dag(["Y -> X"]), "X", "Y")
result.identifiable   # True: the arrow points the wrong way, so do(X) changes nothing
result.strategy       # 'null'
```

Two disconnected nodes land here too. Before this rule, such graphs read as "not identifiable" or, worse, turned up an instrument for an effect that does not exist.

## Equivalence classes: identify on what discovery returns

Discovery does not hand back a single DAG. `discover(data)` returns a CPDAG and `discover(data, method="fci")` returns a PAG: equivalence classes, where an undirected edge or a circle mark means the data could not decide. `identify_effect` accepts those faces directly (and MAGs), and the answer is stronger than it looks: a success holds in every graph the class represents, through the generalized adjustment criterion (Perkovic et al., 2018). You do not have to pick a member and hope.

```python
from ergodic.graph import cpdag_of_dag

cp = cpdag_of_dag(dag(["A -> X", "B -> X", "X -> Y"]))
identify_effect(cp, "X", "Y").adjustment_set   # frozenset({'A', 'B'}): valid in every member
```

Three things decide a class query:

- No possibly causal path: when no member can route a directed path from treatment to outcome, the effect is zero everywhere and the strategy is `"null"`, exactly as on a DAG.
- Amenability: every possibly causal path out of the treatment must start with a directed edge that is *visible* (on a CPDAG every directed edge is; on a MAG or PAG visibility is Zhang's certificate that no member hides a confounder behind the edge). When the first edge is undecided or invisible, members disagree about exactly what adjustment needs, no set works uniformly, and the report says so.
- The criterion itself: possible ancestry replaces ancestry, the forbidden set grows to possible descendants, and blocking reads definite-status paths, so the answer is the class invariant.

When the class is amenable, one estimand covers it; when it is not, enumerate the members and identify each (the IDA pattern):

```python
from ergodic.graph import dags_of_cpdag

chain = cpdag_of_dag(dag(["X -> M", "M -> Y"]))   # X - M - Y: nothing is compelled
identify_effect(chain, "X", "Y").identifiable     # False: not amenable
{identify_effect(m, "X", "Y").strategy for m in dags_of_cpdag(chain)}
# {'adjustment', 'null'}: the members genuinely disagree about the effect
```

`mags_of_pag` does the same for a PAG. The front-door and instrument criteria are not class-invariant, so on a class face they raise and point you at the enumerators; `identify_effect` reports not-identifiable after adjustment fails. A `mixed` graph raises too: it certifies neither a single world nor a class, so re-face it with `as_kind` first (an FCI fallback output with `pag_valid` 0.0 stays mixed on purpose).

Because `estimate_effect` runs identification first, the whole bridge composes in two calls: discover a class from data, then estimate straight off it. The discovery guide closes with that loop.

## Bad inputs raise

`identify_effect` and the helpers check their inputs up front. An empty treatment or outcome, a treatment that overlaps the outcome, or a name that is not a node in the graph raises `ValueError` naming the offender, instead of returning an estimand for a variable the graph has never seen. A graph that marks selection nodes (`selection=`) raises on every face: the criteria assume the data weren't selected on any node, and an effect generally isn't identifiable from a selected sample.

## The result record

`Identification` carries the outcome of `identify_effect`:

- `identifiable`: whether any strategy worked (the record is truthy on this).
- `strategy`: `"null"`, `"adjustment"`, `"frontdoor"`, `"iv"`, `"id"`, or `"none"`.
- `adjustment_set`, `mediators`, `instruments`: filled in for the matching strategy.
- `estimand`: a readable formula string.
- `functional`: the typed estimand, set when `strategy` is `"id"` (the general do-calculus formula, below).
- `hedge`: the non-identifiability witness, set when a graph is not identifiable by anything.

The first four fields cover the named strategies and are unchanged. The last two ride along for the general ID algorithm, which the next section describes.

---

The strategies so far (adjustment, front-door, instruments) cover the everyday cases, and most work stops here. The rest of the page is the general theory: the complete do-calculus algorithm behind any single graph, identification across an equivalence class, and mediation. Reach for it when the named strategies don't apply.

## The general ID algorithm

Adjustment, the front-door, and an instrument are the three patterns people reach for, and they cover most graphs. They do not cover all of them. Some effects are identifiable only through a formula that is none of the three, and some are not identifiable at all. The do-calculus settles every case: given a single graph, it decides whether P(Y | do(X)) is a function of the observational distribution, and when it is, it writes the formula. `identify_distribution` runs that algorithm.

```python
from ergodic import admg, identify_distribution

# the Verma graph: identifiable, but outside adjustment, front-door, and IV
g = admg(["A -> B", "B -> C", "C -> D", "A <-> C", "B <-> D"])
res = identify_distribution(g, "D", "A")
res.identifiable          # True
res.functional.render()   # the estimand, a sum-product of conditionals
```

This is the Shpitser-Pearl ID algorithm, complete for a single DAG or ADMG: if a formula exists, it finds one; if none exists, it proves it. The result is an `IDResult` carrying `identifiable`, the `functional` (the estimand) when it holds, and the `hedge` (the witness) when it does not. The `query` field records what you asked.

An empty `do` set is allowed, and it means the observational query. `P(y | do())` is just the marginal `P(y)`, read off the joint:

```python
from ergodic import dag

g = dag(["Z -> X", "Z -> Y", "X -> Y"])
identify_distribution(g, "Y", []).render()   # 'sum_x sum_z P(X, Y, Z)', i.e. P(Y)
```

When you pass `given`, the algorithm runs IDC, the conditional version, which handles `P(y | do(x), z)`. An empty `do` with a nonempty `given` is the observational conditional `P(y | z)`.

### The functional is a typed object

The estimand is no longer a bare string. `identify_distribution` returns a `Functional`: a small algebra of `Prob` (a conditional term), `Product`, `Marginal` (a sum), and `Ratio`. Each one renders to the readable string you have seen all along (`functional.render()`), and the algebra exposes `free_vars()` (the variables it reads) and `simplify()` (conservative rewrites only, so two runs render the same formula). The plug-in estimator below reads the tree directly, never the string.

The surface is public, so you can inspect or transform an estimand. It is also provisional: the subclasses and their fields may shift while the ID layer settles, so read through `render`, `free_vars`, and `simplify` where you can.

### `identify_effect` falls through to ID

You do not have to call `identify_distribution` by hand. `identify_effect` keeps its fast ladder (null, adjustment, front-door, instruments), and when all four miss on a single graph, it runs general ID as the last step. A graph that is do-calculus identifiable but outside the three named strategies now identifies through the same one call, with `strategy="id"` and the functional in hand:

```python
from ergodic import admg, identify_effect

g = admg(["A -> B", "B -> C", "C -> D", "A <-> C", "B <-> D"])
result = identify_effect(g, "A", "D")
result.strategy             # 'id'
result.functional is None   # False: the typed estimand is set
result.estimand             # the same formula as a string
```

When the fallthrough recognizes a named closed form (an adjustment or front-door formula in disguise), it returns that strategy instead, so the existing estimators still consume it. You get `strategy="id"` only for the formulas that genuinely need the general path.

### The hedge: the honest "not identifiable"

When no formula exists, the algorithm does not shrug. It exhibits a hedge, the structural reason the effect cannot be written from observational data. The classic case is the bow: a treatment with a direct effect on the outcome and unobserved confounding on the same pair. Write it with an explicit latent confounder.

```python
from ergodic import dag, identify_distribution

bow = dag(["U -> X", "U -> Y", "X -> Y"], latent=["U"])
res = identify_distribution(bow, "Y", "X")
res.identifiable     # False
res.hedge.render()   # 'not identifiable: a hedge formed by [...] and the c-component [...]'
```

Through `identify_effect`, the same graph returns `strategy="none"` with the hedge attached, so the failure reason is the do-calculus witness, not a generic "adjustment, front-door, and IV all missed":

```python
result = identify_effect(bow, "X", "Y")
bool(result)         # False
result.hedge.render()
```

The hedge is the object the SCM layer will read to decide where to return bounds instead of a point. That handoff is not built yet; the witness is.

### Identified is not estimable

A subtle point worth stating plainly. ID returning a functional means the effect is a function of the observational distribution, not that a ready estimator exists for it. The named forms (adjustment, front-door) route to today's estimators. A general `strategy="id"` functional is estimated on discrete data by the plug-in (covered in the [inference guide](inference.md)); a continuous one waits for the SCM, which does not exist yet. So do not read `strategy="id"` as "ready to estimate" on continuous data. `estimate_effect` raises a clear pointer there rather than guessing.

## `ida()`: identify across a class

`identify_distribution` runs on a single graph. Discovery hands back a class. `ida` bridges them: it enumerates the members of a CPDAG and runs `identify_effect` on each, returning the multiset of answers.

```python
from ergodic import dag, ida
from ergodic.graph import cpdag_of_dag

cp = cpdag_of_dag(dag(["X -> M", "M -> Y"]))   # the chain X - M - Y, nothing compelled
res = ida(cp, "X", "Y")
res.n_identifiable    # 3: every member identifies
res.agree             # False: but they disagree about the effect
res.strategies        # frozenset({'adjustment', 'null'})
res.estimands         # the distinct formulas seen across members
```

This is the IDA pattern (Maathuis-Kalisch-Buhlmann) made first class. The `IDA` record carries `members` (the per-member `Identification`s), `n_identifiable`, `all_identifiable`, `strategies`, `estimands`, and `agree` (true when every member identifies and shares one estimand). When the class agrees, you have a single member-invariant answer; when it does not, the spread is the honest range.

Scope is stated up front. A CPDAG enumerates DAG members, so full ID runs per member. A PAG or MAG member is a MAG, not a DAG, and general per-member ID over one needs a DAG realization that is not built. So on a PAG or MAG, `ida` returns the class-level adjustment answer where it applies and raises a clear "not built" for the rest, rather than pretending coverage.

## Mediation

A total effect splits into the part that flows through a mediator and the part that does not. `identify_mediation` decides whether that split is identifiable, for the natural direct effect (NDE), the natural indirect effect (NIE), and the controlled direct effect (CDE).

```python
from ergodic import dag, identify_mediation

# X affects Y directly and through M; W confounds X with Y
g = dag(["X -> M", "M -> Y", "X -> Y", "W -> X", "W -> Y"])
med = identify_mediation(g, "X", "Y", "M")
med.identifiable     # True: the natural effects identify
med.covariates       # frozenset({'W'}): the set that makes the formula valid
med.nde_estimand     # the NDE formula
med.nie_estimand     # the NIE formula
```

The natural effects are identified by the mediation formula (Pearl, 2001) under a covariate set `W` of non-descendants of the treatment that meets three adjustment conditions: `W` blocks the back-doors from `X` to `Y` and from `X` to `M`, and `W` together with `X` blocks the back-doors from `M` to `Y`. `identify_mediation` searches for such a `W`, prefers the canonical one, and minimizes it.

The fourth condition is the catch, and no data can check it. The natural effects are cross-world quantities: they compare a world where the mediator takes the value it would have had under one treatment against an outcome under another. That comparison rests on there being no mediator-outcome confounder that is itself affected by the treatment. `identify_mediation` states this in the result's `assumptions` every time, in words, because it is graphically certified here but untestable from data. When no valid `W` exists, the natural effects are not identifiable, the canonical case being exactly that post-treatment confounder (a recanting witness), and the result says so:

```python
# L is a mediator-outcome confounder that X itself affects
g = dag(["X -> M", "M -> Y", "X -> L", "L -> Y", "L -> M"])
med = identify_mediation(g, "X", "Y", "M")
med.identifiable        # False: the cross-world obstruction
med.cde_identifiable    # True: the controlled direct effect survives it
```

The controlled direct effect is a plainer quantity: the effect of the treatment with the mediator held fixed by intervention, `E[Y | do(x), do(m)]` contrasted across treatment levels. It is a plain interventional contrast on `{X, M}`, so it routes through the general ID engine and needs no cross-world assumption. CDE can identify when NDE and NIE do not, and the other way around, so `MediationID` tracks `cde_identifiable` and `cde_estimand` separately.

To estimate a decomposition, see `estimate_mediation` in the [inference guide](inference.md).

## Still ahead

Path-specific effects, the general mediation case along a chosen subset of paths, identified through the recanting-witness criterion. The conditional front-door, where the criterion holds only given extra covariates, where the plain three-condition criterion misses (general ID already identifies many of these). Class-level front-door and instrument reasoning beyond the per-member `ida` route. The SCM that reads a hedge and returns bounds instead of a point.
