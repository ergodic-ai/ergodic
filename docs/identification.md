# Identification

Identification is the step between a graph and an estimate. Given a graph, a treatment, and an outcome, it answers whether the effect P(Y | do(X)) can be written from observational data, and how. The work is graph-only: no data and no estimator yet, just the structure.

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

Discovery does not hand back a single DAG. `discover(data)` returns a CPDAG and `discover(data, method="fci")` returns a PAG: equivalence classes, where an undirected edge or a circle mark means the data could not decide. `identify_effect` accepts those faces directly (and MAGs), and the answer is stronger than it looks: a success holds in **every** graph the class represents, through the generalized adjustment criterion (Perkovic et al., 2018). You do not have to pick a member and hope.

```python
from ergodic.graph import cpdag_of_dag

cp = cpdag_of_dag(dag(["A -> X", "B -> X", "X -> Y"]))
identify_effect(cp, "X", "Y").adjustment_set   # frozenset({'A', 'B'}): valid in every member
```

Three things decide a class query:

- **No possibly causal path.** When no member can route a directed path from treatment to outcome, the effect is zero everywhere and the strategy is `"null"`, exactly as on a DAG.
- **Amenability.** Every possibly causal path out of the treatment must start with a directed edge that is *visible* (on a CPDAG every directed edge is; on a MAG or PAG visibility is Zhang's certificate that no member hides a confounder behind the edge). When the first edge is undecided or invisible, members disagree about the very thing adjustment needs, no set works uniformly, and the report says so.
- **The criterion itself.** Possible ancestry replaces ancestry, the forbidden set grows to possible descendants, and blocking reads definite-status paths, so the answer is the class invariant.

When the class is amenable, one estimand covers it; when it is not, enumerate the members and identify each (the IDA pattern):

```python
from ergodic.graph import dags_of_cpdag

chain = cpdag_of_dag(dag(["X -> M", "M -> Y"]))   # X - M - Y: nothing is compelled
identify_effect(chain, "X", "Y").identifiable     # False: not amenable
{identify_effect(m, "X", "Y").strategy for m in dags_of_cpdag(chain)}
# {'adjustment', 'null'}: the members genuinely disagree about the effect
```

`mags_of_pag` does the same for a PAG. The front-door and instrument criteria are not class-invariant, so on a class face they raise and point you at the enumerators; `identify_effect` simply reports not-identifiable after adjustment fails. A `mixed` graph raises too: it certifies neither a single world nor a class, so re-face it with `as_kind` first (an FCI fallback output with `pag_valid` 0.0 stays mixed on purpose).

Because `estimate_effect` runs identification first, the whole bridge composes in two calls: discover a class from data, then estimate straight off it. The discovery guide closes with that loop.

## Bad inputs raise

`identify_effect` and the helpers check their inputs up front. An empty treatment or outcome, a treatment that overlaps the outcome, or a name that is not a node in the graph raises `ValueError` naming the offender, instead of returning an estimand for a variable the graph has never seen.

## The result record

`Identification` carries the outcome of `identify_effect`:

- `identifiable`: whether any strategy worked (the record is truthy on this).
- `strategy`: `"null"`, `"adjustment"`, `"frontdoor"`, `"iv"`, or `"none"`.
- `adjustment_set`, `mediators`, `instruments`: filled in for the matching strategy.
- `estimand`: a readable formula string.

## Not built yet

The full do-calculus ID algorithm decides every identifiable case on a single graph, including ones that adjustment and the front-door miss, and it returns a general formula or a proof that no formula exists. That is the next identification piece. The front-door here is the plain three-condition criterion; the conditional front-door, where the criterion holds only given extra covariates, is a known gap. On classes, only adjustment generalizes for now: class-level front-door and instrument reasoning, and an `ida()` convenience that returns the multiset of member effects, would build on the enumerators. Estimation, turning a named strategy and its sets into a number over real data, is the step after.
