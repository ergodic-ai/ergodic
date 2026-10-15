# SPDX-License-Identifier: Apache-2.0
"""The regime likelihood: the one object a new emission model has to be.

A :class:`RegimeLikelihood` owns everything that depends on how a hidden
regime shows up in the data, and nothing else. It prepares the raw series into
whatever tensors its emission wants (frames and periodograms for the spectral
model, a lag matrix for the autoregression), registers its priors inside the
active PyMC model, and hands back one log-density tensor per time step and
state. The engine owns the rest: the chain priors, the forward pass that
marginalizes the discrete path, the sampler, the decode, and the label
alignment. That split is what lets a new likelihood be written without touching
any inference code, and it is the same division of labor the forecast pillar
uses between a mechanism and its fit.

Deltas are declared, not discovered. A likelihood names its mechanism blocks
up front (:class:`DeltaBlock`), each pointing at a shrinkage-prior site whose
posterior measures how state 1 differs from state 0 in that one respect. The
result's verdicts are then credible-interval zero-exclusions over those named
sites, familywise-corrected, with the likelihood's own gates
(:class:`Gate`) applied on top. Existence of a regime is never read off a
delta; that question belongs to ``certify()``.

Site naming follows the forecast contract: every registered site is
``f"{target}_{param}"``, so a posterior can be walked by prefix and a draw can
be handed back to :meth:`RegimeLikelihood.emissions_np` under bare parameter
names.

The chain spec rides along here because a transition prior lives at the
likelihood's own step resolution. :class:`MarkovChain` holds the Beta prior on
the stay probabilities; the default concentrates near one because a regime
worth naming dwells for hundreds of ticks, and a likelihood whose emission
step covers many ticks (the spectral model's 96-tick frames) overrides
:meth:`RegimeLikelihood.default_chain` with the same dwell arithmetic done at
its own resolution.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import numpy.typing as npt

    Array = npt.NDArray[np.float64]

__all__ = [
    "DeltaBlock",
    "EmissionSpec",
    "Gate",
    "MarkovChain",
    "PreparedData",
    "RegimeLikelihood",
    "require_pymc",
    "resolve_nuts_sampler",
    "semi_markov",
]


# -- the engine binding -------------------------------------------------------
#
# require_pymc and resolve_nuts_sampler are duplicated from the forecast
# pillar (forecast/_mechanism.py and forecast/_fit.py) rather than imported:
# those are private modules of another pillar, and a cross-pillar import of a
# private module couples the two release-by-release. The measured facts they
# encode carry over unchanged: PyMC picks nutpie when nuts_sampler is omitted,
# and the forecast engine spike measured nutpie at 3.8x the wall clock of the
# JAX path on scan-shaped models like the forward pass here.

_fallback_warned = False


def require_pymc() -> Any:
    """Import pymc, or say how to get it.

    The engine is imported inside the functions that sample, never at module
    import, so the bare install of the package never loads it and never pays
    for it. This is the one place that turns its absence into a sentence a
    user can act on.
    """
    try:
        import pymc
    except ImportError as exc:  # pragma: no cover - exercised by import interception
        raise ImportError(
            "the latent pillar samples with pymc, which the bare install does not "
            "carry: pip install 'ergodic[latent]' (and jax plus numpyro for the fast "
            "sampler path)"
        ) from exc
    return pymc


def resolve_nuts_sampler() -> str:
    """The NUTS backend to name at the call site: the JAX path when it is installed.

    Never left to PyMC's own default. PyMC picks nutpie when it is present and
    ``nuts_sampler`` is omitted, and a benchmark on the forecast model measured nutpie
    at 3.8 times the wall clock of the JAX path on this shape of model, so the
    choice is made here and stated.

    Falling back to the C backend is correct but slow, so it says so once.
    """
    global _fallback_warned
    try:
        import jax
        import numpyro  # noqa: F401
    except ImportError:
        if not _fallback_warned:
            _fallback_warned = True
            warnings.warn(
                "sampling through PyMC's C backend because jax and numpyro are not "
                "installed; the JAX path measured about four times faster on this "
                "model. `pip install jax numpyro` to use it.",
                UserWarning,
                stacklevel=2,
            )
        return "pymc"
    # PyTensor works in float64 and JAX defaults to float32, so the backend has
    # to be told, or the sampler quietly loses half the precision of the states.
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    return "numpyro"


# -- the chain spec -----------------------------------------------------------


@dataclass(frozen=True)
class MarkovChain:
    """The prior on the hidden chain: ``stay ~ Beta(stay_alpha, stay_beta)`` per state.

    The stay probability is the whole personality of a two-state chain, and
    the prior on it is where dwell knowledge enters. The default,
    ``Beta(60, 1)``, concentrates near one: a regime with a mean dwell around
    240 ticks stays put with probability about 0.996 per tick, and a prior
    that allows rapid flip-flopping invites the sampler to explain noise with
    a busy path. A likelihood whose emission step spans many ticks supplies
    its own default at its own resolution through
    :meth:`RegimeLikelihood.default_chain`.

    The initial distribution is always ``Dirichlet(1, 1)``; there is nothing
    worth configuring in where a long chain happens to start.
    """

    stay_alpha: float = 60.0
    stay_beta: float = 1.0

    def __post_init__(self) -> None:
        if not (self.stay_alpha > 0.0 and self.stay_beta > 0.0):
            raise ValueError(
                f"MarkovChain needs positive Beta parameters, got "
                f"stay_alpha={self.stay_alpha}, stay_beta={self.stay_beta}"
            )


def semi_markov(*, min_dwell: int) -> MarkovChain:
    """Reserved: a chain with an explicit dwell-time distribution.

    A geometric dwell is what a Markov chain can express, and the worlds this
    pillar was measured on have negative-binomial dwells, which the Beta stay
    prior only approximates. An explicit semi-Markov chain (dwell floor, or a
    full dwell distribution) needs its own forward pass and is not available, so
    this raises.
    """
    raise NotImplementedError(
        f"semi_markov(min_dwell={min_dwell}) is not available: chains are "
        "Markov with a Beta stay prior. Use MarkovChain(stay_alpha, stay_beta) "
        "to encode dwell knowledge."
    )


# -- the emission contract ----------------------------------------------------


@dataclass(frozen=True)
class PreparedData:
    """What :meth:`RegimeLikelihood.prepare` hands the engine: numpy only.

    ``obs`` holds the named arrays the emission reads (frame means and
    periodograms, or a lag matrix), already standardized. ``T_eff`` is the
    number of emission steps, which is also the length of the decoded path.
    ``tick_multiplier`` says how many ticks of the original series one
    emission step covers (96 for a spectral frame, 1 for an AR step), and
    ``offset`` how many ticks the preparation consumed at the start (the lag
    order for an autoregression). Together they place the decoded path back
    on the original clock.
    """

    obs: Mapping[str, Array]
    T_eff: int
    tick_multiplier: int = 1
    offset: int = 0


@dataclass(frozen=True)
class DeltaBlock:
    """One named mechanism block: a verdict-bearing site of the emission model.

    ``name`` is the vocabulary word (``"level"``, ``"spread"``), ``site`` the
    full site name whose posterior draws measure the state-1-minus-state-0
    difference. ``transform`` reduces a vector-valued site to the scalar the
    verdict reads; it receives the sign-aligned draws of the whole decode
    subsample at once (shape ``(S, ...)``), because a sensible reduction (the
    dominant component of a lag-coefficient delta, say) is a property of the
    posterior, and returns shape ``(S,)``.
    """

    name: str
    site: str
    transform: Callable[[Array], Array] | None = None


@dataclass(frozen=True)
class Gate:
    """A likelihood-declared sanity condition that can veto named verdicts.

    A delta interval can exclude zero for a parameter the data never pinned
    down, so a likelihood may require its own evidence before a verdict
    stands: the spectral model's wave gate demands that the fitted line
    actually clears the noise continuum before a frequency or amplitude
    verdict is allowed to fire. ``check`` receives the decode subsample's
    posterior draws under bare parameter names and returns the gate statistic
    and whether it passed. A gate should be label-invariant (built from
    max or symmetric functions over the two states), since it runs on raw,
    unaligned draws.
    """

    name: str
    blocks: tuple[str, ...]
    check: Callable[[Mapping[str, Array]], tuple[float, bool]]


@dataclass(frozen=True)
class EmissionSpec:
    """What :meth:`RegimeLikelihood.emissions` returns to the engine.

    ``loglik`` is the emission log-density tensor: ``(T, K)``, or ``(T, K, K)``
    for an emission that conditions on the previous state as well. ``deltas``
    are the declared mechanism blocks, ``gates`` the vetoes.
    """

    loglik: Any
    deltas: tuple[DeltaBlock, ...]
    gates: tuple[Gate, ...] = ()


@dataclass(frozen=True)
class RegimeLikelihood:
    """The unit of extension: how a hidden regime shows up in one series.

    Subclass as a frozen dataclass and implement four methods. ``prepare``
    turns the raw series into the numpy tensors the emission reads.
    ``emissions`` runs inside the active ``pm.Model()``, registers every prior
    site as ``f"{name}_{param}"``, and returns the emission log-density tensor
    with the declared delta blocks and gates. ``emissions_np`` is the numpy
    mirror of the same density for one posterior draw, which is what the
    decode's forward-backward runs on; it has to be the same math, because a
    decode under a different density than the fit answers a different
    question. ``default_chain`` states the transition prior at this
    likelihood's own step resolution.

    ``null_loglik`` and ``simulate_null`` are the one-state null model that
    ``certify()`` and ``calibrate()`` read: the log-likelihood of the best
    regime-free fit on the prepared series, and a fresh control world drawn
    from that same fit. The two must describe the same fitted null, or the
    p-values built on them answer a question nobody asked.

    ``k`` is fixed at two states: one switch, on or off. More states
    change the delta vocabulary (which pair differs?) and are a design of
    their own, so anything else is refused here rather than half-supported.
    """

    k: int = 2

    def __post_init__(self) -> None:
        if self.k != 2:
            raise NotImplementedError(
                f"k={self.k} states is not supported: the likelihoods recover one "
                "binary switch (k=2)."
            )

    def default_chain(self) -> MarkovChain:
        """The transition prior at this likelihood's step resolution."""
        return MarkovChain()

    def prepare(self, y: Array) -> PreparedData:
        """Numpy-only: standardize, frame, or lag-slice ``y`` into emission inputs."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement prepare(); a regime "
            "likelihood has to turn the raw series into its own emission inputs"
        )

    def emissions(self, name: str, data: PreparedData) -> EmissionSpec:
        """Register priors inside the active ``pm.Model()`` and return the emission spec."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement emissions(); a regime "
            "likelihood has to register its sites and return its log-density tensor"
        )

    def emissions_np(self, data: PreparedData, draw: Mapping[str, Array]) -> Array:
        """The numpy mirror of :meth:`emissions` for one draw: ``(T, K)`` log-densities."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement emissions_np(); decoding "
            "needs the same emission math in numpy, one draw at a time"
        )

    def null_loglik(self, data: PreparedData) -> float:
        """The fitted one-state null's log-likelihood on the prepared series.

        This is the regime-free half of ``certify()``'s improvement statistic,
        so it must be conditioned exactly as the two-state marginal
        log-likelihood is (the autoregression conditions both on the first
        ``p`` ticks), or the two are not comparable.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement null_loglik(); certify() "
            "needs the likelihood's own one-state null to compare against"
        )

    def simulate_null(self, data: PreparedData, rng: np.random.Generator) -> Array:
        """A fresh control world drawn from the same fitted one-state null.

        Returned in the prepared series' own (standardized) space, at the
        original series length, so it can go straight back through
        :meth:`prepare`. ``certify()`` builds its null distribution from these
        worlds and ``calibrate()`` measures the detector's fire rate on them.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement simulate_null(); certify() "
            "and calibrate() need control worlds drawn from the fitted one-state null"
        )
