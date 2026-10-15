# SPDX-License-Identifier: Apache-2.0
"""How many processes the SCM's NUTS runs may use.

PyMC's ``pm.sample`` runs ``chains`` chains in up to ``cores`` worker processes
and, left to itself, picks ``min(chains, cpu_count, 4)``. The SCM fixes the
number instead: :class:`~ergodic.scm.SCM` takes ``cores=`` (default 1, chains run
one after another) and every built-in sampler call reads it from here, so a fit
never spawns more processes than asked for. A context variable carries it, so the
:class:`~ergodic.scm.Mechanism` protocol (which custom mechanisms implement) keeps
its signature; the built-in mechanisms read :func:`sampler_cores` when they
sample, and a direct ``mechanism.fit`` outside an SCM runs on one core.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_CORES: ContextVar[int] = ContextVar("ergodic_scm_cores", default=1)


def sampler_cores(chains: int) -> int:
    """The ``cores`` to hand ``pm.sample``: the SCM's setting, at most ``chains``."""
    return max(1, min(_CORES.get(), chains))


@contextmanager
def using_cores(cores: int) -> Iterator[None]:
    """Run the enclosed sampler calls with ``cores`` worker processes at most."""
    token = _CORES.set(cores)
    try:
        yield
    finally:
        _CORES.reset(token)
