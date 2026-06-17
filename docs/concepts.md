# Concepts

Everything in ergodic builds on a small set of shared objects: the causal graph, the domain-knowledge set, the typed dataset, and the identification step that turns a graph into an estimand. This page covers the model behind the first two, the ones with the most ideas in them, so the method-by-method guides read easily.

## One graph, many kinds

Causal work uses more than one kind of graph. A known model is a DAG. Constraint-based discovery (the PC algorithm) returns a CPDAG, a whole class of DAGs that fit the data equally well. Allow hidden confounders and FCI returns a PAG. Inference with latent variables uses an ADMG. ergodic represents all of them with one object by putting a mark on each end of every edge.

A mark is a tail, an arrowhead, or a circle. The pair of marks on an edge says what the edge means:

| Glyph | Marks | Meaning |
| --- | --- | --- |
| `A -> B` | tail, arrowhead | A is a direct cause of B |
| `A <-> B` | arrowhead, arrowhead | A and B share a hidden common cause |
| `A -- B` | tail, tail | adjacent, orientation unspecified |
| `A o-> B` | circle, arrowhead | arrow at B, unknown at A |

Each named kind is the same core under a rule about which marks it allows:

- `DAG`: directed, acyclic.
- `ADMG`: directed and bidirected, acyclic. The object for latent confounding.
- `CPDAG`: directed and undirected. What constraint-based discovery returns.
- `MAG`: an ADMG with no almost-directed cycle.
- `PAG`: any marks, circles included. What FCI returns.

`validate` enforces the rule, so a graph always knows its kind and refuses an illegal mark or a cycle. Building with `dag`, `admg`, `cpdag`, or `pag` validates for you and raises `InvalidGraphError` on a bad input.

## Graphs are immutable

A graph is a value, like a number or a string. Two graphs with the same nodes, edges, and kind are equal and hash the same, so they work as dict keys and set members. Every edit (`add_edge`, `orient`, `do`, `remove_edge`) returns a new graph and leaves the original alone.

This matters because causal work is mostly graph surgery. The do-operator builds a mutilated graph, discovery orients one edge at a time, and adjustment criteria reason over modified graphs. With values, the version before a step and the version after it both exist, results cache cleanly, and there is no aliasing to track.

One detail to know: edges store the smaller node name first, so a printed edge can read reversed. `season -> rain` shows as `rain <-- season` when `rain` sorts first. It is the same edge.

## Domain knowledge narrows the graphs

`DomainKnowledge` is a set of constraints on structure: edges that must or must not be there, the order variables come in, roots and sinks, confounded pairs. Read it as a filter on the space of possible graphs, where each constraint rules some out. Two knowledge objects combine with `&`, and the result keeps the graphs that satisfy both, so more knowledge means fewer graphs.

Today the constraints are hard. A graph either satisfies them or it does not. Soft knowledge, a belief with a weight or a prior over graphs, attaches at `energy` and `log_prior`, which raise for now. That arrives with the Bayesian discovery work, and it will carry provenance too: where a belief came from, and how much to trust it.

## How the pieces fit

Discovery takes data and domain knowledge and returns a graph, often a CPDAG or a PAG. Inference takes a graph and data and returns an effect. Domain knowledge feeds both: it prunes and orients during discovery, and it states the conditions an effect needs to be identified. The graph and the knowledge are the two inputs every pillar reads.

## What is built

The representation is done, and the first two pillars run on it end to end. Separation reads every face, circle marks included, and on a CPDAG or PAG it answers for the whole equivalence class. Identification covers adjustment, the front-door, and instruments on a single graph, and the generalized adjustment criterion on a class, so a discovered graph identifies directly. Estimation turns an estimand into a number with its uncertainty, over a library of estimators and an optional Bayesian posterior path. When the question is a rollout no graph can carry, the quasi-experimental designs answer it instead: difference-in-differences, synthetic control, and interrupted time series. Discovery learns a CPDAG (PC with its conservative and majority collider rules, GES, the order searches BOSS and GRaSP, and friends), a full DAG when the noise is non-Gaussian (DirectLiNGAM), or a PAG that tolerates hidden confounders and selection bias both (FCI and the faster RFCI), with domain knowledge entering every search, skeleton screens (MMPC, glasso) confining any of them, and time series read through PCMCI window graphs. The process pillar reads event logs: a case table with timestamp-gated features carries a log into discovery and estimation, a KPI panel turns a process change into a natural experiment, and the model layer mines a sound process model, checks the log against it, and reads each exclusive split as a decision whose branches have estimable effects. Still ahead: off-policy evaluation on logs and soft knowledge, tracked in the implementation journal under `plans/`.
