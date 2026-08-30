# SPDX-License-Identifier: Apache-2.0
"""The switching autoregression: regimes in the conditional mean and variance.

State 1 is state 0 plus one delta per mechanism block: level moves the
intercept, rhythm moves the lag coefficients, spread moves the log innovation
standard deviation, each with Laplace shrinkage so an untouched block stays
at zero. The emission is an AR(p) conditional density, so a step's
log-likelihood depends on the lagged signal values, which are observed data;
the emission tensor is a plain ``(T, 2)`` and the hidden path still only
enters through the forward pass.

This likelihood exists for two reasons. It proves the contract generalizes: a
model with a completely different preparation (a lag matrix instead of frame
periodograms), a vector-valued delta block, and no gates rides the same
engine unchanged. And it sees rhythm the classical way: a frequency change
lives entirely in the lag coefficients, where a Gaussian-emission HMM
(state-indexed mean and variance only) measured 0.51 path agreement on the
rhythm arm, a coin flip.

The measured blindness runs the other way too, and it is structural, not a
tuning problem. On the amplitude arm this model fired 0 of 2 while the
spectral model fired; on level, spread, and rhythm it fired 6 of 6 at the 94%
level. The reason sits in the algebra of linear recurrences: the AR
coefficients that reproduce a sinusoid are set by its frequency alone (the
characteristic roots sit on the unit circle at the wave's angle, at any
amplitude), and the amplitude lives in the state the recurrence carries, so
doubling the wave changes neither the fitted intercept, nor the lag
coefficients, nor the innovation variance. Both regimes then present
identical AR(p) parameters and the likelihood has nothing to split on.
"amplitude" is simply absent from this model's delta vocabulary; use the
spectral likelihood when that mechanism is on the table.

The rhythm delta is a vector (one entry per lag), and a verdict needs a
scalar, so the block's transform picks the component with the largest
posterior-mean magnitude after sign alignment, exactly as the reference
implementation did. That is a reduction for reporting, chosen by the
posterior itself; the fit shrinks all p components jointly.

At full size this model is the slow one (median around 80 seconds at 1440
ticks against the spectral model's seconds), because the forward pass scans
every tick rather than fifteen frames. Tests run it on small worlds.

The one-state null behind ``certify()`` and ``calibrate()`` is the same
recipe with the regime deleted: a single AR(p) fitted by OLS, whose
conditional Gaussian log-likelihood conditions on the first ``p`` ticks
exactly as the marginalized two-state likelihood does, and whose control
worlds are rolled forward from the real series' own opening ticks with fresh
innovations. This is a faithful port of the reference lab's certify stage
(its ``_one_state_fit`` and ``_simulate_no_switch``), reverse-slice pitfall
and all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .._likelihood import DeltaBlock, EmissionSpec, MarkovChain, PreparedData, RegimeLikelihood

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

    Array = npt.NDArray[np.float64]

__all__ = ["SwitchingAR", "switching_ar"]

#: Fewer effective observations than this cannot pin two states of an AR(p).
_MIN_OBS = 32


def _dominant_component(draws: Array) -> Array:
    """The lag component with the largest posterior-mean magnitude, per draw.

    Receives the sign-aligned ``(S, p)`` draws of the rhythm delta and returns
    the ``(S,)`` column the posterior itself points at. Picking by the mean
    across draws (rather than per draw) keeps every draw reporting the same
    component, so the interval is an interval over one parameter.
    """
    k = int(np.argmax(np.abs(draws.mean(axis=0))))
    return np.asarray(draws[:, k], dtype=np.float64)


@dataclass(frozen=True)
class SwitchingAR(RegimeLikelihood):
    """Regimes as AR(p) parameter shifts: level, rhythm, and spread deltas.

    ``p`` is the lag order. Three lags cover the worlds this was measured on
    (an EMA signal with short memory); more lags buy nothing there and dilute
    the rhythm delta across more components.
    """

    p: int = 3

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.p < 1:
            raise ValueError(f"the lag order must be >= 1, got p={self.p}")

    def default_chain(self) -> MarkovChain:
        """``Beta(60, 1)``: a 240-tick mean dwell is a per-tick stay near 0.996."""
        return MarkovChain(60.0, 1.0)

    def prepare(self, y: Array) -> PreparedData:
        """Slice into target and lag matrix, standardized by the target's own moments."""
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1:
            raise ValueError(f"the series must be 1-D, got shape {y.shape}")
        n = len(y)
        if n - self.p < _MIN_OBS:
            raise ValueError(
                f"{n} ticks leave {max(n - self.p, 0)} observations after {self.p} "
                f"lags, and the switching AR needs at least {_MIN_OBS}; supply a "
                "longer series or a smaller p"
            )
        target = y[self.p :]
        lags = np.column_stack([y[self.p - j : n - j] for j in range(1, self.p + 1)])
        mu, sd = target.mean(), target.std()
        if sd == 0.0:
            raise ValueError("the series is constant, so there is no regime to recover")
        return PreparedData(
            obs={"y": (target - mu) / sd, "lags": (lags - mu) / sd},
            T_eff=n - self.p,
            tick_multiplier=1,
            offset=self.p,
        )

    def emissions(self, name: str, data: PreparedData) -> EmissionSpec:
        """Register the AR sites and return the ``(T, 2)`` log emission tensor."""
        import pymc as pm
        import pytensor.tensor as pt

        c0 = pm.Normal(f"{name}_c0", 0.0, 1.0)
        a0 = pm.Normal(f"{name}_a0", 0.0, 0.5, shape=self.p)
        log_s0 = pm.Normal(f"{name}_log_s0", -1.0, 1.0)

        d_level = pm.Laplace(f"{name}_d_level", 0.0, 0.3)
        d_rhythm = pm.Laplace(f"{name}_d_rhythm", 0.0, 0.3, shape=self.p)
        d_spread = pm.Laplace(f"{name}_d_spread", 0.0, 0.3)

        c = pt.stack([c0, c0 + d_level])
        a = pt.stack([a0, a0 + d_rhythm])  # (2, p)
        sigma = pt.exp(pt.stack([log_s0, log_s0 + d_spread]))

        y = pt.as_tensor_variable(data.obs["y"])
        lags = pt.as_tensor_variable(data.obs["lags"])
        # pm.math.dot is the typed-Any re-export of pt.dot (see the SCM's note
        # on pytensor ops tripping mypy's no-untyped-call).
        mu = c[None, :] + pm.math.dot(lags, a.T)  # (T, 2)
        resid = (y[:, None] - mu) / sigma[None, :]
        loglik = -0.5 * resid**2 - pt.log(sigma)[None, :] - 0.5 * np.log(2 * np.pi)

        return EmissionSpec(
            loglik=loglik,
            deltas=(
                DeltaBlock("level", f"{name}_d_level"),
                DeltaBlock("rhythm", f"{name}_d_rhythm", transform=_dominant_component),
                DeltaBlock("spread", f"{name}_d_spread"),
            ),
        )

    def emissions_np(self, data: PreparedData, draw: Mapping[str, Array]) -> Array:
        """The same emission math in numpy for one draw: ``(T, 2)`` log densities."""
        y = data.obs["y"]
        lags = data.obs["lags"]
        c = np.array([draw["c0"], draw["c0"] + draw["d_level"]])
        a = np.stack([draw["a0"], draw["a0"] + draw["d_rhythm"]])  # (2, p)
        sigma = np.exp(np.array([draw["log_s0"], draw["log_s0"] + draw["d_spread"]]))
        mu = c[None, :] + lags @ a.T
        resid = (y[:, None] - mu) / sigma[None, :]
        ll: Array = -0.5 * resid**2 - np.log(sigma)[None, :] - 0.5 * np.log(2.0 * np.pi)
        return ll

    # -- the one-state null (certify / calibrate) -----------------------------

    def _null_fit(self, data: PreparedData) -> tuple[Array, float]:
        """The no-switch model: one AR(p) by OLS on the standardized series.

        Returns the coefficient vector (intercept first) and the innovation
        standard deviation, which is everything a control world is rolled
        from. The variance is floored so a pathologically perfect fit cannot
        hand :meth:`simulate_null` a zero noise scale.
        """
        y = data.obs["y"]
        base = np.column_stack([np.ones(len(y)), data.obs["lags"]])
        beta, *_ = np.linalg.lstsq(base, y, rcond=None)
        resid = y - base @ beta
        var = max(float(np.mean(resid**2)), 1e-12)
        return np.asarray(beta, dtype=np.float64), float(np.sqrt(var))

    def null_loglik(self, data: PreparedData) -> float:
        """Conditional Gaussian log-likelihood of the OLS one-state AR(p).

        Conditioned on the first ``p`` observations, exactly as the two-state
        marginal log-likelihood is (the forward pass runs over the ``n - p``
        emission steps, each conditioning on observed lags), so the two are
        directly comparable and their difference is a real improvement. At
        the OLS optimum the Gaussian log-likelihood collapses to
        ``-T/2 (log(2 pi var) + 1)``.
        """
        y = data.obs["y"]
        _, sd = self._null_fit(data)
        return float(-0.5 * len(y) * (np.log(2.0 * np.pi * sd * sd) + 1.0))

    def simulate_null(self, data: PreparedData, rng: np.random.Generator) -> Array:
        """Roll the fitted no-switch AR(p) forward from the real opening ticks.

        The control world starts from the observed series' own first ``p``
        ticks (in the standardized space the fit lives in) and continues with
        the fitted coefficients plus fresh innovations, so it matches the real
        series in length, scale, and memory, and differs only in having no
        regime. Feeding it back through :meth:`prepare` re-standardizes by its
        own moments, exactly as a real control series would be.
        """
        beta, sd = self._null_fit(data)
        p = self.p
        # The first lag row holds ticks p-1 .. 0 of the standardized series,
        # so reversing it recovers the opening ticks in time order.
        opening = data.obs["lags"][0][::-1]
        n = data.offset + data.T_eff
        out = np.empty(n)
        out[:p] = opening
        noise = rng.normal(0.0, sd, size=n)
        for t in range(p, n):
            # Reverse an explicit forward slice: a strided `t-1 : t-p-1 : -1`
            # silently empties at t == p, where the stop index reaches -1 and
            # is read as end-of-array rather than as "one before the start".
            out[t] = beta[0] + beta[1:] @ out[t - p : t][::-1] + noise[t]
        return out


def switching_ar(p: int = 3) -> SwitchingAR:
    """The reference second likelihood: AR(p) emissions with level, rhythm, and spread.

    Deliberately without an amplitude block; that blindness is structural (see
    the module docstring) and pretending otherwise would be a verdict slot
    that can never fire.
    """
    return SwitchingAR(p=p)
