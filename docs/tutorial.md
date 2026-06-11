# Getting started

This walks through both core objects on one example: the wet-grass model. A season decides whether the sprinkler runs and whether it rains, both wet the grass, and wet grass is slippery. By the end you'll have built a causal graph, asked it independence questions, intervened on it, and checked it against what you know.

Install first:

```bash
uv add ergodic   # or: pip install ergodic
```

## Build a graph

```python
from ergodic import dag

g = dag([
    "season -> sprinkler",
    "season -> rain",
    "sprinkler -> wet",
    "rain -> wet",
    "wet -> slippery",
])

g.node_list()            # ['rain', 'season', 'slippery', 'sprinkler', 'wet']
g.parents("wet")         # frozenset({'rain', 'sprinkler'})
g.ancestors("slippery")  # frozenset({'rain', 'season', 'sprinkler', 'wet'})
```

`dag(...)` reads each edge from a glyph string and checks the result is fully directed and acyclic. Family lookups return frozensets, since a graph is an immutable value.

## Ask independence questions

d-separation tells you which independences the graph implies. The sprinkler and the rain share a cause (the season) and share an effect (wet grass), so the answer turns on what you condition on.

```python
g.d_separated("sprinkler", "rain")                     # False
g.d_separated("sprinkler", "rain", "season")           # True
g.d_separated("sprinkler", "rain", {"season", "wet"})  # False
```

With nothing fixed, the season connects them. Hold the season and that path closes, so the two are independent. Add wet grass and you open the collider, where the sprinkler and the rain meet head to head, so they depend on each other again. The conditioning argument takes a single name or any iterable of names.

The Markov blanket is the set that screens a variable off from the rest:

```python
g.markov_blanket("wet")  # frozenset({'rain', 'slippery', 'sprinkler'})
```

## Intervene

`do` is graph surgery for an intervention. Setting the sprinkler from outside cuts its incoming edge from the season, while its effect on the grass stays.

```python
mutilated = g.do("sprinkler")
mutilated.parents("sprinkler")  # frozenset()
mutilated.parents("wet")        # frozenset({'rain', 'sprinkler'})
g.parents("sprinkler")          # frozenset({'season'})
```

The intervened graph drops the season's influence on the sprinkler and keeps the sprinkler's effect on the grass. The original `g` is untouched, because every edit returns a new graph. The observational graph and the intervened graph both exist, which is what you want when comparing a do against the baseline.

## Add what you know

Most prior knowledge is an order. The season comes before the sprinkler and the rain, those come before wet grass, and wet grass comes before slippery. State it as tiers.

```python
from ergodic import DomainKnowledge

dk = DomainKnowledge().with_tiers([
    ["season"],
    ["sprinkler", "rain"],
    ["wet"],
    ["slippery"],
])

dk.permits_edge("season", "wet")  # True
dk.permits_edge("wet", "season")  # False
dk.is_consistent(g)               # True
```

A later tier cannot cause an earlier one, so `wet -> season` is not permitted. The graph you built respects the order, so it is consistent.

## Catch a bad graph

`is_consistent` checks a candidate, and `violations` says why one failed.

```python
bad = dag(["wet -> season"])
dk.is_consistent(bad)  # False
dk.violations(bad)     # ['order violated: wet is an ancestor of season']
```

## Orient a skeleton from what you know

Knowledge alone can orient some edges. Give `apply_to` an undirected skeleton and the tiers point each edge whose direction is forced.

```python
from ergodic import mixed

skeleton = mixed([
    "season -- sprinkler", "season -- rain",
    "sprinkler -- wet", "rain -- wet", "wet -- slippery",
])

oriented = dk.apply_to(skeleton)
oriented.is_dag()                                         # True
oriented.get_edge("season", "sprinkler").directed_pair()  # ('season', 'sprinkler')
```

Here the tiers pin every edge, so the skeleton turns into the original DAG. With looser knowledge some edges stay undirected, which is the honest answer.

## Next

- [Learn causal inference](learn/index.md) if the ideas themselves are new: nine notebook guides, no background assumed.
- [Concepts](concepts.md) for the model behind these objects: marks, kinds, and why graphs are immutable.
- [Causal graphs](graph.md) and [Domain knowledge](knowledge.md) for the method-by-method guides.
- [API reference](reference.md) for the full surface.
