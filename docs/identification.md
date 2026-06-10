# Identification

Identification is the step between a graph and an estimate. Given a graph, a treatment, and an outcome, it answers whether the effect P(Y | do(X)) can be written from observational data, and how. The work is graph-only: no data and no estimator yet, just the structure.

## One call

`identify_effect` tries the strategies in order (adjustment, then front-door, then instruments) and returns a record with the strategy, the sets involved, and a readable estimand.

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

An instrument affects the outcome only through the treatment, which is enough to identify the effect even with unobserved confounding between treatment and outcome.

```python
from ergodic.identification import instruments, is_instrument

g = admg(["I -> X", "X -> Y", "X <-> Y"])
is_instrument(g, "I", "X", "Y")   # True
instruments(g, "X", "Y")          # [('I', frozenset())]
```

The pair is the instrument and the conditioning set it needs. Add a direct path from the instrument to the outcome and it stops being valid:

```python
g = admg(["I -> X", "I -> Y", "X -> Y", "X <-> Y"])
is_instrument(g, "I", "X", "Y")   # False
```

## The result record

`Identification` carries the outcome of `identify_effect`:

- `identifiable`: whether any strategy worked (the record is truthy on this).
- `strategy`: `"adjustment"`, `"frontdoor"`, `"iv"`, or `"none"`.
- `adjustment_set`, `mediators`, `instruments`: filled in for the matching strategy.
- `estimand`: a readable formula string.

## Not built yet

The full do-calculus ID algorithm decides every identifiable case, including ones that adjustment and the front-door miss, and it returns a general formula or a proof that no formula exists. That is the next identification piece. Identification on PAGs (circle marks) is rejected for now and needs its own reasoning. Estimation, turning a named strategy and its sets into a number over real data, is the step after.
