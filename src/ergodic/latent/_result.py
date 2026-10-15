# SPDX-License-Identifier: Apache-2.0
"""The recovery result: when the regime was on, and which block moved.

:class:`LatentResult` is the latent pillar's sibling of
:class:`~ergodic.forecast.ForecastResult`: a frozen record of posterior draws
with eager numpy accessors and warn-never-raise diagnostics. It is not a reuse
of that class, because the two answer different questions. A forecast is a
cloud of future trajectories; a recovery is a posterior over a *past* hidden
path plus a handful of named mechanism deltas, and its accessors (the state
marginals, the hard path, the interval verdicts) have no forecast analog.

The draws behind it are the decode subsample: a fixed, evenly spaced subset of
the posterior (150 draws by default), because decoding runs a forward-backward
pass per draw and the path marginals stabilize long before the full posterior
is spent. Deltas are stored sign-aligned: a draw the label alignment flipped
has its delta draws negated, so the intervals here already speak about "the
regime the aligned path calls state 1".

Verdicts are two-tier by design. The first tier is arithmetic on the delta
draws: an equal-tailed credible interval per block at a familywise-corrected
level (Bonferroni across the declared blocks), firing only when it excludes
zero. The second tier is the likelihood's own gates: a declared sanity
condition (the spectral wave gate) can veto the blocks it names, because an
interval can exclude zero for a parameter the data never actually pinned.
Existence of a regime is deliberately not a verdict; that is ``certify()``'s
question, and reading it off a delta conflates "which block" with "whether at
all".

Diagnostics (divergences, split R-hat) are carried, and the engine warns on
breach, but nothing here raises on a rough fit: a diagnostic is information
about a posterior, not a verdict on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt
    import pandas as pd

    from ._likelihood import MarkovChain

    Array = npt.NDArray[np.float64]
    BoolArray = npt.NDArray[np.bool_]
    IntArray = npt.NDArray[np.int64]

__all__ = [
    "CalibrationReport",
    "CalibrationWorld",
    "CertifyResult",
    "Consensus",
    "GateOutcome",
    "LatentResult",
]


@dataclass(frozen=True)
class GateOutcome:
    """One gate's measured statistic and whether it passed.

    ``blocks`` names the delta blocks this gate vetoes when it fails; the
    statistic and its meaning are the declaring likelihood's own (the spectral
    wave gate's is the 3rd percentile of the line's log-power excess over the
    continuum under its peak).
    """

    name: str
    blocks: tuple[str, ...]
    value: float
    passed: bool


@dataclass(frozen=True)
class LatentResult:
    """Posterior state path and mechanism deltas for one recovered switch.

    ``gammas`` is ``(S, T_eff)``: the decode subsample's aligned posterior
    probability of state 1 per emission step. ``delta_draws`` maps each
    declared block to its ``(S,)`` sign-aligned draws. ``tick_multiplier`` and
    ``offset`` place the emission steps back on the original series' clock:
    step ``t`` covers ticks ``offset + t * tick_multiplier`` onward.

    Arrays are copied in and frozen, so the result owns its draws.
    """

    target: str
    gammas: Array
    flips: BoolArray
    delta_draws: Mapping[str, Array]
    gates: tuple[GateOutcome, ...]
    tick_multiplier: int
    offset: int
    chain: MarkovChain
    likelihood: Any
    #: Total posterior draws the fit produced; the decode subsample is
    #: ``n_decoded`` of them, evenly spaced.
    n_draws: int
    divergences: int | None = None
    #: Worst split R-hat over the delta blocks after label alignment (the one to
    #: judge the verdicts by), over the decode subsample grouped by chain.
    rhat_max: float | None = None
    #: Worst split R-hat over the raw sampler draws, before alignment. Chains in
    #: mirror modes push it high even when the aligned fit is fine.
    rhat_max_unaligned: float | None = None

    def __post_init__(self) -> None:
        gammas = np.array(self.gammas, dtype=np.float64)
        gammas.setflags(write=False)
        object.__setattr__(self, "gammas", gammas)
        flips = np.array(self.flips, dtype=bool)
        flips.setflags(write=False)
        object.__setattr__(self, "flips", flips)
        if gammas.ndim != 2:
            raise ValueError(f"gammas must be (S, T), got shape {gammas.shape}")
        if flips.shape != (gammas.shape[0],):
            raise ValueError(f"flips has shape {flips.shape}, expected ({gammas.shape[0]},)")
        frozen: dict[str, Array] = {}
        for name, draws in dict(self.delta_draws).items():
            arr = np.array(draws, dtype=np.float64)
            if arr.shape != (gammas.shape[0],):
                raise ValueError(
                    f"delta block {name!r} has shape {arr.shape}, expected "
                    f"({gammas.shape[0]},): one sign-aligned scalar per decoded draw"
                )
            arr.setflags(write=False)
            frozen[str(name)] = arr
        object.__setattr__(self, "delta_draws", frozen)

    # -- shape ----------------------------------------------------------------

    @property
    def n_decoded(self) -> int:
        """How many posterior draws were decoded and aligned."""
        return int(self.gammas.shape[0])

    @property
    def n_steps(self) -> int:
        """Emission steps in the decoded path."""
        return int(self.gammas.shape[1])

    @property
    def blocks(self) -> tuple[str, ...]:
        """The declared delta blocks, in declaration order."""
        return tuple(self.delta_draws)

    # -- the path -------------------------------------------------------------

    def gamma(self) -> Array:
        """Posterior state marginals, ``(T_eff, K)``: the mean over decoded draws."""
        on = self.gammas.mean(axis=0)
        return np.stack([1.0 - on, on], axis=1)

    def path(self) -> IntArray:
        """The hard path at emission resolution: state 1 where its marginal clears one half."""
        on = np.asarray(self.gammas.mean(axis=0), dtype=np.float64)
        return np.asarray(on > 0.5, dtype=np.int64)

    def tick_path(self) -> IntArray:
        """The hard path on the original clock, one entry per tick.

        Each emission step's state is repeated over the ticks it covers. The
        array starts at tick ``offset`` of the original series and ends at
        ``offset + n_steps * tick_multiplier``; ticks the preparation consumed
        or truncated carry no state and are simply absent.
        """
        return np.repeat(self.path(), self.tick_multiplier)

    # -- the deltas -----------------------------------------------------------

    def deltas(self, level: float = 0.94) -> dict[str, tuple[float, float, float]]:
        """Equal-tailed ``(lo, mean, hi)`` per block at ``level``, no correction.

        This is the descriptive read of the posterior; :meth:`verdicts` is the
        corrected, gated decision.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        tail = (1.0 - level) / 2.0 * 100.0
        out: dict[str, tuple[float, float, float]] = {}
        for name, draws in self.delta_draws.items():
            lo, hi = np.percentile(draws, [tail, 100.0 - tail])
            out[name] = (float(lo), float(draws.mean()), float(hi))
        return out

    def verdicts(self, level: float = 0.99) -> dict[str, bool]:
        """Which blocks moved: zero-exclusion at familywise ``level``, gates applied.

        The per-block interval runs at the Bonferroni share ``1 - (1 - level) /
        n_blocks``, so with the default level and four blocks each interval is
        equal-tailed at 99.75%, and the chance any inert block fires stays at
        or under one percent. A block a failed gate names is vetoed whatever
        its interval says.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        n_blocks = len(self.delta_draws)
        if n_blocks == 0:
            return {}
        tail = (1.0 - level) / n_blocks / 2.0 * 100.0
        out: dict[str, bool] = {}
        for name, draws in self.delta_draws.items():
            lo, hi = np.percentile(draws, [tail, 100.0 - tail])
            out[name] = bool(lo > 0.0 or hi < 0.0)
        for gate in self.gates:
            if not gate.passed:
                for block in gate.blocks:
                    if block in out:
                        out[block] = False
        return out

    # -- export ---------------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """The decoded path as a frame: one row per emission step.

        Columns: ``step``, ``tick`` (the step's first tick on the original
        clock), ``gamma`` (posterior probability of state 1), ``state`` (the
        hard path).
        """
        import pandas as pd

        steps = np.arange(self.n_steps, dtype=np.int64)
        return pd.DataFrame(
            {
                "step": steps,
                "tick": self.offset + steps * self.tick_multiplier,
                "gamma": self.gammas.mean(axis=0),
                "state": self.path(),
            }
        )

    def __repr__(self) -> str:
        fired = [name for name, hit in self.verdicts().items() if hit]
        verdict = ", ".join(fired) if fired else "none"
        return (
            f"LatentResult({self.target!r}, steps={self.n_steps}, "
            f"decoded={self.n_decoded}/{self.n_draws}, verdicts=[{verdict}])"
        )


@dataclass(frozen=True)
class CertifyResult:
    """Whether a regime exists at all: the observed improvement in its own null.

    ``observed`` is twice the log-likelihood gain of the two-state model over
    the likelihood's fitted one-state null on the real series; ``null`` holds
    the same statistic recomputed on control worlds simulated from that null,
    one entry per world. ``p_value`` is the standard simulation p,
    ``(1 + #{null >= observed}) / (1 + B)``: with ``B`` control worlds the
    finest attainable evidence is ``1 / (1 + B)``, and reporting zero would
    claim more resolution than the simulation bought.

    This is deliberately a different object from :meth:`LatentResult.verdicts`.
    A delta interval says which block moved *given* a regime; this record says
    whether the regime story beats a world that never had one, with the null
    distribution measured rather than assumed (the BIC alternative prices an
    extra state at a fixed ``ln n`` per parameter, and the reference lab
    measured that price wrong in both directions on these worlds).
    """

    observed: float
    null: Array
    p_value: float

    def __post_init__(self) -> None:
        null = np.array(self.null, dtype=np.float64)
        if null.ndim != 1 or null.size < 1:
            raise ValueError(f"null must be a non-empty 1-D array, got shape {null.shape}")
        null.setflags(write=False)
        object.__setattr__(self, "null", null)

    @property
    def n_null(self) -> int:
        """How many control worlds the null distribution holds."""
        return int(self.null.size)

    def to_frame(self) -> pd.DataFrame:
        """The whole test as a frame: one row per statistic, observed first.

        Columns ``statistic`` and ``source`` (``"observed"`` or ``"null"``),
        so the null histogram and the observed line plot straight off it.
        """
        import pandas as pd

        return pd.DataFrame(
            {
                "statistic": np.concatenate([[self.observed], self.null]),
                "source": ["observed"] + ["null"] * self.n_null,
            }
        )

    def __repr__(self) -> str:
        q95 = float(np.quantile(self.null, 0.95))
        return (
            f"CertifyResult(observed={self.observed:.2f}, null_q95={q95:.2f}, "
            f"p_value={self.p_value:.4f}, n_null={self.n_null})"
        )


@dataclass(frozen=True)
class CalibrationWorld:
    """One control world's outcome: did the detector fire, and on which blocks."""

    world: int
    seed: int
    fired: bool
    fired_blocks: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationReport:
    """The measured false-fire rate of one configured detector on control worlds.

    ``n`` control worlds were simulated from the likelihood's fitted one-state
    null on the observed series and each ran through the full ``recover``
    pipeline; a world counts as fired when any block's verdict came back true.
    ``rate`` is ``fires / n`` and the interval is Wilson's, which stays honest
    at 0 of n and n of n where the textbook interval collapses to a point.

    A detector that fires on control worlds is not slightly wrong, it is
    uninformative: every fire on real data then carries the same doubt. The
    spectral likelihood's own history says why this report exists: an
    intermediate version fired on 45% of control worlds and looked excellent
    on effect worlds while doing it.
    """

    n: int
    fires: int
    rate: float
    ci_lo: float
    ci_hi: float
    worlds: tuple[CalibrationWorld, ...]

    def summary(self) -> str:
        """One sentence, the number a decision reads."""
        return (
            f"fired on {self.fires} of {self.n} control worlds "
            f"({self.rate:.0%}, 95% Wilson [{self.ci_lo:.0%}, {self.ci_hi:.0%}])"
        )

    def to_frame(self) -> pd.DataFrame:
        """Per-world rows: ``world``, ``seed``, ``fired``, ``fired_blocks``."""
        import pandas as pd

        return pd.DataFrame(
            {
                "world": [w.world for w in self.worlds],
                "seed": [w.seed for w in self.worlds],
                "fired": [w.fired for w in self.worlds],
                "fired_blocks": [", ".join(w.fired_blocks) for w in self.worlds],
            }
        )

    def __repr__(self) -> str:
        return f"CalibrationReport({self.summary()})"


@dataclass(frozen=True)
class Consensus:
    """Several likelihoods' verdicts on one series, laid side by side.

    Each model speaks its own vocabulary, so a block can be missing from a
    model rather than quiet in it: ``verdicts`` maps each block to one entry
    per model, ``True``/``False`` where the model declares the block and
    ``None`` where it does not (the switching AR has no amplitude block, and
    that absence is structural, not a non-fire). ``agreed`` collects the
    blocks every declaring model settles the same way; ``contested`` the
    blocks where declaring models disagree, which is where a cross-model rule
    would adjudicate.

    One rule is not available: the amp/spread rule (a spectral amplitude fire
    against a volatility model's spread silence separates a louder wave from a
    noisier floor) needs a ``vol_garch`` likelihood that ergodic does not have,
    so :meth:`amp_vs_spread` raises by name.
    """

    target: str
    models: tuple[str, ...]
    verdicts: Mapping[str, tuple[bool | None, ...]]
    level: float

    def __post_init__(self) -> None:
        frozen = {str(name): tuple(entry) for name, entry in dict(self.verdicts).items()}
        for name, entry in frozen.items():
            if len(entry) != len(self.models):
                raise ValueError(
                    f"block {name!r} carries {len(entry)} verdicts for {len(self.models)} models"
                )
        object.__setattr__(self, "verdicts", frozen)

    @property
    def blocks(self) -> tuple[str, ...]:
        """Every block any model declares, in first-seen order."""
        return tuple(self.verdicts)

    @property
    def agreed(self) -> dict[str, bool]:
        """Blocks where every model that declares them says the same thing."""
        out: dict[str, bool] = {}
        for name, entry in self.verdicts.items():
            spoken = {v for v in entry if v is not None}
            if len(spoken) == 1:
                out[name] = spoken.pop()
        return out

    @property
    def contested(self) -> tuple[str, ...]:
        """Blocks where declaring models disagree; a cross-model rule's domain."""
        return tuple(
            name
            for name, entry in self.verdicts.items()
            if len({v for v in entry if v is not None}) > 1
        )

    def amp_vs_spread(self) -> Any:
        """Reserved: the amplitude-versus-spread cross-model rule.

        A spectral amplitude fire and a volatility model's spread silence
        together separate "the wave got louder" from "the noise floor rose",
        which no single built-in likelihood can do alone. The rule needs a
        ``vol_garch`` likelihood, which is not available, so this raises.
        """
        raise NotImplementedError(
            "the amp/spread cross-model rule needs a vol_garch likelihood, which "
            "is not available; consensus() reports the agreement table"
        )

    def to_frame(self) -> pd.DataFrame:
        """One row per block: a verdict column per model, then the consensus.

        Model verdicts are ``True``/``False``/``None`` (None: not in that
        model's vocabulary); ``consensus`` is ``"fires"``/``"quiet"`` when
        the declaring models agree and ``"contested"`` when they do not.
        """
        import pandas as pd

        agreed = self.agreed
        rows: list[dict[str, object]] = []
        for name, entry in self.verdicts.items():
            row: dict[str, object] = {"block": name}
            row.update(dict(zip(self.models, entry, strict=True)))
            if name in agreed:
                row["consensus"] = "fires" if agreed[name] else "quiet"
            else:
                row["consensus"] = "contested"
            rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        fired = [name for name, hit in self.agreed.items() if hit]
        body = ", ".join(fired) if fired else "none"
        contested = ", ".join(self.contested) if self.contested else "none"
        return (
            f"Consensus({self.target!r}, models=[{', '.join(self.models)}], "
            f"agreed_fires=[{body}], contested=[{contested}])"
        )
