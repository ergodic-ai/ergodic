# SPDX-License-Identifier: Apache-2.0
"""Soundness of workflow nets, decided by reachability.

Structural validation (the :class:`~ergodic.process.WorkflowNet` constructor)
says a net has the right shape; it says nothing about behavior. A net can
pass every shape check and still deadlock halfway, finish with tokens left
behind, or carry a transition that never fires. Van der Aalst's classical
soundness is the behavioral property that rules all three out, and
:func:`is_sound` decides it exactly by exploring the reachable markings with
the net's one shared firing rule (:meth:`~ergodic.process.WorkflowNet.index`).

A sound net is bounded, so its reachable state space is finite and the
exploration terminates. An unbounded net is not, which is why the search
carries a ``max_states`` cap that raises instead of guessing.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._net import Marking, WorkflowNet

__all__ = ["is_sound"]


def is_sound(net: WorkflowNet, *, max_states: int = 100_000) -> bool:
    """Whether ``net`` is sound in van der Aalst's classical sense.

    Three conditions, all decided on the real state space. Option to
    complete: from every marking reachable from the initial marking, the
    final marking is still reachable. Proper completion: every reachable
    marking that puts a token on the sink is exactly the final marking, so
    no run finishes with tokens left behind. No dead transitions: every
    transition fires on some reachable path.

    The check explores markings breadth-first from
    :attr:`~ergodic.process.WorkflowNet.initial_marking`, recording which
    marking fires which transition into which marking. Proper completion is
    then read off the explored markings, dead transitions off the union of
    fired transition ids, and option to complete by walking the recorded
    state graph backward from the final marking: every explored marking must
    reach it, which also settles whether the final marking is reachable at
    all.

    A sound net is bounded, so the exploration terminates. When the number
    of distinct markings passes ``max_states``, the net is either unbounded
    (and so unsound) or genuinely this big, and the function raises a
    ``ValueError`` rather than guess; raise the cap if the net really is
    this big.
    """
    index = net.index()
    final = net.final_marking
    sink = net.sink

    # Forward exploration: the state graph as marking -> [(tid, next marking)].
    edges: dict[Marking, list[tuple[str, Marking]]] = {net.initial_marking: []}
    queue: deque[Marking] = deque([net.initial_marking])
    fired: set[str] = set()
    while queue:
        state = queue.popleft()
        for tid in index.enabled(state):
            following = index.fire(state, tid)
            fired.add(tid)
            edges[state].append((tid, following))
            if following not in edges:
                edges[following] = []
                queue.append(following)
                if len(edges) > max_states:
                    raise ValueError(
                        f"the reachable state space exceeded max_states={max_states}; "
                        "a sound workflow net is bounded, so either this net is "
                        "unbounded or it really is this big: raise max_states to "
                        "keep exploring"
                    )

    # Proper completion: a token on the sink means the run is over, exactly.
    for state in edges:
        if state != final and any(place == sink for place, _ in state):
            return False

    # No dead transitions: every transition fired somewhere above.
    if fired != set(index.labels):
        return False

    # Option to complete: every explored marking reaches the final marking,
    # read backward over the recorded state graph.
    if final not in edges:
        return False
    predecessors: dict[Marking, set[Marking]] = {state: set() for state in edges}
    for state, outgoing in edges.items():
        for _tid, following in outgoing:
            predecessors[following].add(state)
    completes = {final}
    frontier = [final]
    while frontier:
        state = frontier.pop()
        for prev in predecessors[state]:
            if prev not in completes:
                completes.add(prev)
                frontier.append(prev)
    return len(completes) == len(edges)
