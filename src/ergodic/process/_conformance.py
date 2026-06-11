# SPDX-License-Identifier: Apache-2.0
"""Conformance checking: how well an event log fits a process model.

:func:`conformance` plays every distinct trace of an
:class:`~ergodic.data.EventLog` against a model (a process tree or a workflow
net) and answers the fit question twice. Token replay, after Rozinat and
van der Aalst (2008), pushes the trace through the net and counts tokens: the
ones it had to add (missing) and the ones left behind (remaining) measure how
far the trace strays, and the standard half-and-half ratio turns the counts
into a fitness number. Replay is fast and diagnostic, but it can part-credit
a trace that never forms a real run. Optimal alignments, after Adriansyah
(2014), are the gold standard: a shortest-path search walks the trace and the
model together and names every deviation, a log move for an event the model
cannot mirror and a model move for a required visible step the trace skipped.
Both run once per distinct variant; the aggregates weight by case count.

Replay accounting, spelled out because every convention moves the number: the
initial token on the source counts as produced and the environment's final
take from the sink counts as consumed (Rozinat's convention); silent
transitions fired to enable a step count their arcs like any other firing; a
token added to force a firing counts as missing and is later consumed but
never counts as produced; and an event whose label no transition carries
counts one missing and one consumed token (the unmatchable consume), so an
off-model label always costs fitness. In alignments, silent model moves are
free and are omitted from the reported move list.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._mine import ProcessModel
from ._net import WorkflowNet, marking
from ._tree import ProcessTree

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ..data import EventLog
    from ._net import Marking, NetIndex

    _Move = tuple[str, str]
    _State = tuple[Marking, int]

__all__ = ["ConformanceResult", "VariantConformance", "conformance"]


@dataclass(frozen=True)
class VariantConformance:
    """Replay and alignment results for one distinct trace.

    ``sequence`` is the trace and ``n_cases`` how many cases follow it. The
    token-replay counters come next: ``missing`` tokens had to be added,
    ``remaining`` were left behind, and ``consumed`` and ``produced`` count
    every token the replay moved, the initial token and the final take from
    the sink included. ``replay_fitness`` is Rozinat's half-and-half of the
    two ratios. ``cost`` is the optimal alignment cost, ``worst_cost`` the
    reference cost of deleting the whole trace and running the model's
    cheapest visible path instead, and ``alignment_fitness`` is
    ``1 - cost / worst_cost``. ``alignment`` lists the moves: ``("sync", a)``
    where trace and model agree, ``("log", a)`` for an event the model cannot
    mirror, ``("model", a)`` for a required visible model step the trace
    skipped. Silent model steps cost nothing and are omitted from the list.
    """

    sequence: tuple[str, ...]
    n_cases: int
    replay_fitness: float
    missing: int
    remaining: int
    consumed: int
    produced: int
    cost: int
    worst_cost: int
    alignment_fitness: float
    alignment: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ConformanceResult:
    """How well a log fits a model, by token replay and by alignments.

    ``fitness`` is the token-replay log fitness in Rozinat's form: the
    missing/consumed and remaining/produced sums aggregate over all cases
    before the two halves average, so frequent variants weigh more.
    ``alignment_fitness`` is the case-weighted mean of the per-variant
    alignment fitness. A case is fitting when its variant aligns at cost
    zero; ``n_fitting`` and ``share_fitting`` count those. ``variants``
    holds one :class:`VariantConformance` per distinct trace, most frequent
    first, ties by sequence.
    """

    fitness: float
    alignment_fitness: float
    n_cases: int
    n_variants: int
    n_fitting: int
    share_fitting: float
    variants: tuple[VariantConformance, ...]

    def __repr__(self) -> str:
        return (
            f"ConformanceResult(fitness={self.fitness:.3f}, "
            f"alignment_fitness={self.alignment_fitness:.3f}, "
            f"fitting={self.n_fitting}/{self.n_cases} cases)"
        )


# -- shared search pieces --------------------------------------------------------


def _is_enabled(index: NetIndex, state: Marking, tid: str) -> bool:
    """Whether ``tid`` can fire at ``state``."""
    tokens = dict(state)
    return all(tokens.get(place, 0) >= 1 for place in index.pre[tid])


def _enables(index: NetIndex, tid: str) -> Callable[[Marking], bool]:
    """A goal predicate: the marking enables ``tid``."""

    def check(state: Marking) -> bool:
        return _is_enabled(index, state, tid)

    return check


def _has_token(place: str) -> Callable[[Marking], bool]:
    """A goal predicate: the marking holds a token on ``place``."""

    def check(state: Marking) -> bool:
        return dict(state).get(place, 0) >= 1

    return check


def _by_label(index: NetIndex) -> dict[str, tuple[str, ...]]:
    """Visible transition ids per activity label, in id order."""
    out: dict[str, list[str]] = {}
    for tid, label in index.labels.items():
        if label is not None:
            out.setdefault(label, []).append(tid)
    return {label: tuple(tids) for label, tids in out.items()}


def _silent_closure(
    index: NetIndex,
    start: Marking,
    goal: Callable[[Marking], bool],
    silents: tuple[str, ...],
    max_states: int,
) -> tuple[str, ...] | None:
    """The shortest silent-only firing sequence from ``start`` to a goal marking.

    Breadth-first over silent firings, so the first goal marking found is at
    minimal depth; returns ``None`` when silent firings cannot reach one.
    Explored markings count against ``max_states`` and exceeding it raises.
    """
    if goal(start):
        return ()
    seen: set[Marking] = {start}
    queue: deque[tuple[Marking, tuple[str, ...]]] = deque([(start, ())])
    while queue:
        state, path = queue.popleft()
        for tid in silents:
            if not _is_enabled(index, state, tid):
                continue
            nxt = index.fire(state, tid)
            if nxt in seen:
                continue
            seen.add(nxt)
            if len(seen) > max_states:
                raise ValueError(
                    f"silent-transition search exceeded max_states={max_states}; "
                    "raise the cap if the model is genuinely this large"
                )
            if goal(nxt):
                return (*path, tid)
            queue.append((nxt, (*path, tid)))
    return None


# -- token replay ----------------------------------------------------------------


def _replay_variant(
    net: WorkflowNet,
    index: NetIndex,
    by_label: Mapping[str, tuple[str, ...]],
    silents: tuple[str, ...],
    trace: tuple[str, ...],
    max_states: int,
) -> tuple[int, int, int, int]:
    """Token replay of one trace: ``(missing, remaining, consumed, produced)``.

    Follows the conventions in the module docstring: every firing, silent
    ones included, consumes ``len(pre)`` and produces ``len(post)`` tokens;
    the initial token is produced and the final take from the sink is
    consumed by the environment. When several transitions carry the event's
    label, the one needing the fewest missing tokens after silent closure
    fires, ties broken by transition id.
    """
    state = net.initial_marking
    missing = 0
    consumed = 0
    produced = 1

    def fire(at: Marking, tid: str) -> Marking:
        nonlocal consumed, produced
        consumed += len(index.pre[tid])
        produced += len(index.post[tid])
        return index.fire(at, tid)

    for act in trace:
        candidates = by_label.get(act, ())
        if not candidates:
            missing += 1
            consumed += 1
            continue
        options: list[tuple[int, str, tuple[str, ...] | None]] = []
        for tid in candidates:
            closure = _silent_closure(index, state, _enables(index, tid), silents, max_states)
            if closure is not None:
                options.append((0, tid, closure))
            else:
                tokens = dict(state)
                need = sum(1 for place in index.pre[tid] if tokens.get(place, 0) < 1)
                options.append((need, tid, None))
        need, tid, closure = min(options, key=lambda option: (option[0], option[1]))
        if closure is None:
            missing += need
            tokens = dict(state)
            for place in index.pre[tid]:
                if tokens.get(place, 0) < 1:
                    tokens[place] = 1
            state = marking(tokens)
        else:
            for silent in closure:
                state = fire(state, silent)
        state = fire(state, tid)

    closing = _silent_closure(index, state, _has_token(net.sink), silents, max_states)
    if closing is not None:
        for silent in closing:
            state = fire(state, silent)
    consumed += 1
    tokens = dict(state)
    if tokens.get(net.sink, 0) >= 1:
        tokens[net.sink] -= 1
    else:
        missing += 1
    remaining = sum(tokens.values())
    return missing, remaining, consumed, produced


# -- optimal alignments ----------------------------------------------------------


def _align_variant(
    net: WorkflowNet,
    index: NetIndex,
    by_label: Mapping[str, tuple[str, ...]],
    trace: tuple[str, ...],
    max_states: int,
) -> tuple[int, tuple[_Move, ...]]:
    """The optimal alignment of one trace: ``(cost, visible moves)``.

    Dijkstra over ``(marking, position)`` states: a synchronous move fires a
    transition labeled with the next event and costs 0, a model move fires
    any enabled transition without touching the trace (cost 0 when silent,
    1 when visible), and a log move skips the next event for cost 1. Ties
    break by push order, so the result is deterministic. Settled states
    count against ``max_states`` and exceeding it raises.
    """
    start: _State = (net.initial_marking, 0)
    goal: _State = (net.final_marking, len(trace))
    dist: dict[_State, int] = {start: 0}
    parent: dict[_State, tuple[_State, _Move | None]] = {}
    heap: list[tuple[int, int, _State]] = [(0, 0, start)]
    pushes = 1
    settled: set[_State] = set()
    while heap:
        cost, _, state = heapq.heappop(heap)
        if state in settled:
            continue
        settled.add(state)
        if len(settled) > max_states:
            raise ValueError(
                f"alignment search exceeded max_states={max_states}; "
                "raise the cap if the model is genuinely this large"
            )
        if state == goal:
            moves: list[_Move] = []
            cursor = state
            while cursor != start:
                cursor, move = parent[cursor]
                if move is not None:
                    moves.append(move)
            moves.reverse()
            return cost, tuple(moves)
        state_marking, pos = state
        steps: list[tuple[int, _State, _Move | None]] = []
        if pos < len(trace):
            act = trace[pos]
            for tid in by_label.get(act, ()):
                if _is_enabled(index, state_marking, tid):
                    steps.append((0, (index.fire(state_marking, tid), pos + 1), ("sync", act)))
        for tid in index.enabled(state_marking):
            label = index.labels[tid]
            if label is None:
                steps.append((0, (index.fire(state_marking, tid), pos), None))
            else:
                steps.append((1, (index.fire(state_marking, tid), pos), ("model", label)))
        if pos < len(trace):
            steps.append((1, (state_marking, pos + 1), ("log", trace[pos])))
        for step_cost, nxt, move in steps:
            total = cost + step_cost
            if nxt not in dist or total < dist[nxt]:
                dist[nxt] = total
                parent[nxt] = (state, move)
                heapq.heappush(heap, (total, pushes, nxt))
                pushes += 1
    raise ValueError(
        "alignment search exhausted the state space without reaching the final marking"
    )


def _model_path_cost(net: WorkflowNet, index: NetIndex, max_states: int) -> int:
    """The fewest visible firings on any run from the initial to the final marking.

    One Dijkstra over markings, silent firings free, computed once per net;
    ``len(trace)`` plus this is the worst alignment cost a trace can incur.
    Settled markings count against ``max_states`` and exceeding it raises.
    """
    start = net.initial_marking
    goal = net.final_marking
    dist: dict[Marking, int] = {start: 0}
    heap: list[tuple[int, int, Marking]] = [(0, 0, start)]
    pushes = 1
    settled: set[Marking] = set()
    while heap:
        cost, _, state = heapq.heappop(heap)
        if state in settled:
            continue
        settled.add(state)
        if len(settled) > max_states:
            raise ValueError(
                f"model-run search exceeded max_states={max_states}; "
                "raise the cap if the model is genuinely this large"
            )
        if state == goal:
            return cost
        for tid in index.enabled(state):
            total = cost + (0 if index.labels[tid] is None else 1)
            nxt = index.fire(state, tid)
            if nxt not in dist or total < dist[nxt]:
                dist[nxt] = total
                heapq.heappush(heap, (total, pushes, nxt))
                pushes += 1
    raise ValueError(
        "the model cannot reach its final marking from its initial marking, "
        "so alignments are undefined; check the model with is_sound"
    )


def _alignment_fitness(cost: int, worst_cost: int) -> float:
    """``1 - cost / worst_cost``, and 1.0 when ``worst_cost`` is 0.

    A zero worst cost needs an empty trace on a model that can run silently
    end to end; log cases always hold at least one event, so this is an
    internal edge case, guarded all the same.
    """
    if worst_cost == 0:
        return 1.0
    return 1.0 - cost / worst_cost


# -- the public entry point ------------------------------------------------------


def conformance(
    log: EventLog,
    model: ProcessModel | ProcessTree | WorkflowNet,
    *,
    max_states: int = 100_000,
) -> ConformanceResult:
    """Check ``log`` against ``model`` by token replay and by optimal alignments.

    A mined :class:`~ergodic.process.ProcessModel` contributes its net, and a
    process tree converts to its workflow net up front. Each distinct
    variant replays and aligns once, then weights by its case count:
    ``fitness`` aggregates the replay token sums over all cases (Rozinat's
    form), ``alignment_fitness`` is the case-weighted mean of the per-variant
    alignment fitness, and a case is fitting when its variant aligns at cost
    zero. ``max_states`` caps every state search (the silent closures inside
    replay, the alignment search, and the model's cheapest-run search); past
    the cap a ``ValueError`` raises rather than returning a partial answer.
    A model that cannot reach its final marking at all raises too, since no
    alignment exists for it.
    """
    if isinstance(model, ProcessModel):
        net = model.net
    elif isinstance(model, ProcessTree):
        net = model.to_net()
    elif isinstance(model, WorkflowNet):
        net = model
    else:
        raise TypeError(
            f"model must be a ProcessModel, a ProcessTree, or a WorkflowNet, "
            f"got {type(model).__name__}"
        )
    counts = log.variants()
    if not counts:
        raise ValueError("conformance needs a log with at least one case")
    index = net.index()
    silents = tuple(tid for tid, label in index.labels.items() if label is None)
    by_label = _by_label(index)
    min_visible = _model_path_cost(net, index, max_states)

    records: list[VariantConformance] = []
    for sequence, n_cases in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        missing, remaining, consumed, produced = _replay_variant(
            net, index, by_label, silents, sequence, max_states
        )
        cost, moves = _align_variant(net, index, by_label, sequence, max_states)
        worst_cost = len(sequence) + min_visible
        records.append(
            VariantConformance(
                sequence=sequence,
                n_cases=n_cases,
                replay_fitness=0.5 * (1.0 - missing / consumed)
                + 0.5 * (1.0 - remaining / produced),
                missing=missing,
                remaining=remaining,
                consumed=consumed,
                produced=produced,
                cost=cost,
                worst_cost=worst_cost,
                alignment_fitness=_alignment_fitness(cost, worst_cost),
                alignment=moves,
            )
        )

    total_cases = sum(record.n_cases for record in records)
    total_missing = sum(record.n_cases * record.missing for record in records)
    total_consumed = sum(record.n_cases * record.consumed for record in records)
    total_remaining = sum(record.n_cases * record.remaining for record in records)
    total_produced = sum(record.n_cases * record.produced for record in records)
    n_fitting = sum(record.n_cases for record in records if record.cost == 0)
    return ConformanceResult(
        fitness=0.5 * (1.0 - total_missing / total_consumed)
        + 0.5 * (1.0 - total_remaining / total_produced),
        alignment_fitness=sum(r.n_cases * r.alignment_fitness for r in records) / total_cases,
        n_cases=total_cases,
        n_variants=len(records),
        n_fitting=n_fitting,
        share_fitting=n_fitting / total_cases,
        variants=tuple(records),
    )
