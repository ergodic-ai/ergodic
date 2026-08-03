# Learn causal inference

This series teaches causal inference from zero, the ideas before the library. Each guide
takes one question a working analyst faces, shows the obvious answer failing, and
walks through why the causal answer is different and how to compute it. If you've never seen
the words "confounder" or "estimand", start at guide one and read in order. If you know the
field and want the API, the [getting started](../tutorial.md) page and the reference guides
are the faster route.

!!! tip "Unfamiliar term?"
    Every piece of jargon in this series is defined in plain words in the [glossary](../glossary.md).

Every guide runs on simulated data, and that's a deliberate choice, not a shortcut. On real
data nobody can tell you whether your estimate of an effect is right, because the true effect
is exactly the thing nobody measured. In a simulation we write the answer into the code, in
plain sight, and then watch the methods try to recover it. The naive method misses by a lot.
The causal one lands on it. You can rerun every page yourself and change the world to see
what breaks; each page is a notebook, executed live when these docs are built, so the outputs
you read are real.

The guides, in order:

1. [Why causality](01-why-causality.py): a loyalty program that looks like it loses money and
   earns it instead. Confounding, Simpson's paradox, and the graph as the cure.
2. [Good and bad controls](02-good-and-bad-controls.py): controlling for everything is not
   safe. Forks, chains, and colliders, and how the graph says which is which.
3. [Estimating effects](03-estimating-effects.py): from a graph to a number with an
   uncertainty interval, and what a doubly robust estimator buys you.
4. [Heterogeneous effects](04-heterogeneous-effects.py): the average customer doesn't exist.
   Per-customer effects and what they're worth in targeting.
5. [Hidden confounding](05-hidden-confounding.py): when the confounder isn't in the data at
   all. Instruments and the front-door, the two escape routes.
6. [Discovering structure](06-discovering-structure.py): when you can't draw the graph,
   learn it from the data, and estimate straight off what discovery returns.
7. [Quasi-experiments](07-quasi-experiments.py): a rollout nobody randomized. Stores
   compared through time with difference-in-differences and event studies, and a market
   with no twin gets a synthetic one.
8. [Process mining](08-process-mining.py): when the data is an event log, see the process
   first. The map, the waiting-versus-service split, the variant ranking, and the drill-down,
   drawn for real in three steps, the daily workflow of the commercial process tools.
9. [Process intelligence](09-process-intelligence.py): the map blames a review step for
   double its true cost; case tables and KPI panels carry the log into the causal
   machinery and answer what changing the process would do.
10. [Decision points](10-decision-points.py): mine the process model itself, check the log
    against it, and read a routing split as a decision: what sending a case down the other
    lane would actually cost, and when the data is too one-sided to say.

The examples are commercial because money concentrates the mind, but nothing is specific to
commerce. Swap "discount" for "drug" and "spend" for "blood pressure" and every argument
goes through unchanged; the kidney-stone study in guide one is the famous scientific case of
the same arithmetic.
