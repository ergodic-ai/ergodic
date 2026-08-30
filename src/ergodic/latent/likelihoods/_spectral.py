# SPDX-License-Identifier: Apache-2.0
"""The spectral Whittle likelihood: mechanisms as local-spectrum features.

The idea is that every mechanism this vocabulary names lives in a short
window's spectrum. Chop the series into non-overlapping 96-tick Hann frames
and each frame gives a mean and a periodogram; per regime the model says the
spectrum is an AR(1) continuum plus one Gaussian line,

    S_k(f) = sigma_k^2 / (1 - 2 phi cos(2 pi f) + phi^2) + L_k exp(-(f - f_k)^2 / (2 delta^2)),

with the Whittle approximation ``I(b) ~ Exp(S_k(f_b))`` per bin and the frame
mean Gaussian around a regime level. The vocabulary then reads straight off
the parameters: level is the frame mean, spread the continuum's sigma, rhythm
the line's frequency, amplitude the line's power, one Laplace-shrunk delta
each. Phase never enters anywhere, which is what makes an amplitude change
visible here when an autoregression cannot see it (see the switching-AR
module for that measured blindness).

Frames do not overlap, deliberately. A rolling window hands consecutive steps
shared data, and that manufactured memory reads as regime persistence (the
overlap trap the reference work measured first). Non-overlap costs resolution
instead: the decoded path moves in 96-tick steps, and that is the honest
price. Fifteen frames also mean the forward pass is fifteen unrolled steps,
so a fit lands in seconds, and cheap fits are what make calibration over many
control worlds affordable.

Calibration is the reason for everything here that looks fussy. Intermediate
versions of this model fired on 45% of control worlds. Two habits caused it:
treating the periodogram's bins as independent overcounts the evidence,
because a Hann window smears each frequency over about 1.5 bins, and a frame
that still carries the signal's slow drift reads as low-frequency power that
a spread or amplitude delta happily absorbs. The shipped state deflates the
Whittle log-likelihood by 1/1.5 (the window's equivalent noise bandwidth),
removes a per-frame linear trend on top of the mean, and keeps all bins 1
through 40. Measured on twenty control worlds, that state fires zero times at
the 99% familywise level, while the effect arms still fire: the spread arm's
delta lands near +/-1.1 (log 3, exactly the planted tripling), and the
amplitude arm fires at seed 0 with path agreement 0.92.

Two honest caveats carry over from the reference. A deterministic line's
periodogram is noncentral chi-squared rather than exponential, so Whittle is
a quasi-likelihood at the peak bins. And a line the model can place anywhere
will happily "find" one in noise, which is why the frequency and amplitude
verdicts pass through the wave gate: the line's power must clear the
continuum directly under its own peak (3rd percentile of the log excess above
zero) before either verdict is allowed to stand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .._likelihood import (
    DeltaBlock,
    EmissionSpec,
    Gate,
    MarkovChain,
    PreparedData,
    RegimeLikelihood,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

    Array = npt.NDArray[np.float64]

__all__ = ["SpectralWhittle", "spectral_whittle"]

#: The Hann window's equivalent noise bandwidth in bins: adjacent periodogram
#: bins share this much information, so the independent-bin Whittle sum is
#: deflated by its reciprocal.
ENBW = 1.5

#: How many frames the preparation must produce for the chain to mean anything.
_MIN_FRAMES = 4


def _wave_gate(draws: Mapping[str, Array]) -> tuple[float, bool]:
    """Line power must clear the continuum under its own peak, in every likely draw.

    Label-invariant by construction: the excess is maximized over the two
    states, so a flipped draw scores the same.
    """
    log_line = np.stack([draws["log_L0"], draws["log_L0"] + 2.0 * draws["d_amp"]], axis=1)
    peak = np.exp(np.stack([draws["log_f0"], draws["log_f0"] + draws["d_freq"]], axis=1))
    s2 = np.exp(2.0 * np.stack([draws["log_s0"], draws["log_s0"] + draws["d_spread"]], axis=1))
    phi = draws["phi"][:, None]
    continuum = s2 / (1.0 - 2.0 * phi * np.cos(2.0 * np.pi * peak) + phi**2)
    excess = (log_line - np.log(continuum)).max(axis=1)
    value = float(np.percentile(excess, 3))
    return value, value > 0.0


@dataclass(frozen=True)
class SpectralWhittle(RegimeLikelihood):
    """Regimes as local spectra: level, spread, rhythm, and amplitude deltas.

    ``window`` is the frame length in ticks; the decoded path moves in steps
    of one frame. ``bin_lo`` and ``bin_hi`` bound the periodogram bins used
    (half-open, skipping DC and the top by default). The line width is tied
    to the window (1.5 bins, the Hann main lobe), so it needs no field.
    """

    window: int = 96
    bin_lo: int = 1
    bin_hi: int = 41

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.window < 16:
            raise ValueError(
                f"window={self.window} is too short for a periodogram worth "
                "reading; use at least 16 ticks per frame"
            )
        n_bins = self.window // 2 + 1
        if not 1 <= self.bin_lo < self.bin_hi <= n_bins:
            raise ValueError(
                f"bins [{self.bin_lo}, {self.bin_hi}) must satisfy "
                f"1 <= bin_lo < bin_hi <= {n_bins} for window={self.window} "
                "(bin 0 is DC and is always skipped)"
            )

    @property
    def _delta_f(self) -> float:
        """The spectral line's width: about the Hann main lobe."""
        return 1.5 / self.window

    def default_chain(self) -> MarkovChain:
        """``Beta(4, 2)``: dwell knowledge restated at frame resolution.

        The tick-resolution default (``Beta(60, 1)``, stay near 0.996)
        describes a 240-tick dwell per tick. One frame is 96 ticks, so the
        same dwell is about 2.5 frames and the per-frame stay probability sits
        near 0.6, which ``Beta(4, 2)`` centers on without forbidding longer
        runs.
        """
        return MarkovChain(4.0, 2.0)

    def prepare(self, y: Array) -> PreparedData:
        """Standardize, frame, detrend per frame, window, and take periodograms."""
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1:
            raise ValueError(f"the series must be 1-D, got shape {y.shape}")
        n_frames = len(y) // self.window
        if n_frames < _MIN_FRAMES:
            raise ValueError(
                f"{len(y)} ticks give {n_frames} frame(s) of {self.window}, and "
                f"the spectral model needs at least {_MIN_FRAMES}; supply a longer "
                "series or a shorter window"
            )
        sd = y.std()
        if sd == 0.0:
            raise ValueError("the series is constant, so there is no regime to recover")
        y = (y - y.mean()) / sd

        frames = y[: n_frames * self.window].reshape(n_frames, self.window)
        means = frames.mean(axis=1)
        # Per-frame linear detrend: the EMA's slow drift is low-frequency power
        # inside a frame, and leaving it in is one of the two habits that had
        # intermediate versions firing on control worlds.
        t = np.arange(self.window) - (self.window - 1) / 2.0
        slope = (frames * t[None, :]).sum(axis=1) / (t**2).sum()
        centred = frames - means[:, None] - slope[:, None] * t[None, :]
        win = np.hanning(self.window)
        spec = np.abs(np.fft.rfft(centred * win[None, :], axis=1)) ** 2 / np.sum(win**2)
        pgram = spec[:, self.bin_lo : self.bin_hi]
        freqs = np.asarray(
            np.fft.rfftfreq(self.window)[self.bin_lo : self.bin_hi], dtype=np.float64
        )
        return PreparedData(
            obs={"means": means, "pgram": pgram, "freqs": freqs},
            T_eff=n_frames,
            tick_multiplier=self.window,
            offset=0,
        )

    def emissions(self, name: str, data: PreparedData) -> EmissionSpec:
        """Register the spectral sites and return the ``(T, 2)`` log emission tensor."""
        import pymc as pm
        import pytensor.tensor as pt

        mu0 = pm.Normal(f"{name}_mu0", 0.0, 1.0)
        d_level = pm.Laplace(f"{name}_d_level", 0.0, 0.5)
        log_s0 = pm.Normal(f"{name}_log_s0", -1.5, 1.0)
        d_spread = pm.Laplace(f"{name}_d_spread", 0.0, 0.3)
        phi = pm.Beta(f"{name}_phi", 5.0, 2.0)
        tau = pm.HalfNormal(f"{name}_tau", 0.5)
        log_line0 = pm.Normal(f"{name}_log_L0", -2.0, 2.0)
        d_amp = pm.Laplace(f"{name}_d_amp", 0.0, 0.5)  # log amplitude ratio
        log_f0 = pm.Normal(f"{name}_log_f0", np.log(0.045), 0.6)
        d_freq = pm.Laplace(f"{name}_d_freq", 0.0, 0.3)

        mu = pt.stack([mu0, mu0 + d_level])
        s2 = pt.exp(2.0 * pt.stack([log_s0, log_s0 + d_spread]))
        line = pt.exp(pt.stack([log_line0, log_line0 + 2.0 * d_amp]))
        peak = pt.exp(pt.stack([log_f0, log_f0 + d_freq]))

        freqs = data.obs["freqs"]
        f = pt.as_tensor_variable(freqs)  # (B,)
        cont = s2[:, None] / (1.0 - 2.0 * phi * pt.cos(2 * np.pi * f)[None, :] + phi**2)  # (2, B)
        gauss = line[:, None] * pt.exp(
            -((f[None, :] - peak[:, None]) ** 2) / (2.0 * self._delta_f**2)
        )
        spectrum = cont + gauss  # (2, B)

        pgram = pt.as_tensor_variable(data.obs["pgram"])  # (T, B)
        # 1/ENBW: Hann smears ~1.5 bins, so independent-bin Whittle overcounts.
        # pm.math is the typed-Any re-export of the pytensor ops (pt.sum trips
        # mypy's no-untyped-call; the SCM uses the same surface for the same reason).
        ll_whittle = -(1.0 / ENBW) * pm.math.sum(
            pt.log(spectrum)[None, :, :] + pgram[:, None, :] / spectrum[None, :, :],
            axis=2,
        )  # (T, 2)
        means = pt.as_tensor_variable(data.obs["means"])
        ll_mean = (
            -0.5 * np.log(2 * np.pi)
            - pt.log(tau)
            - 0.5 * (means[:, None] - mu[None, :]) ** 2 / tau**2
        )
        return EmissionSpec(
            loglik=ll_whittle + ll_mean,
            deltas=(
                DeltaBlock("level", f"{name}_d_level"),
                DeltaBlock("spread", f"{name}_d_spread"),
                DeltaBlock("freq", f"{name}_d_freq"),
                DeltaBlock("amp", f"{name}_d_amp"),
            ),
            gates=(Gate("wave_present", ("freq", "amp"), _wave_gate),),
        )

    def emissions_np(self, data: PreparedData, draw: Mapping[str, Array]) -> Array:
        """The same emission math in numpy for one draw: ``(T, 2)`` log densities."""
        freqs = data.obs["freqs"]
        pgram = data.obs["pgram"]
        means = data.obs["means"]

        mu = np.array([draw["mu0"], draw["mu0"] + draw["d_level"]])
        s2 = np.exp(2.0 * np.array([draw["log_s0"], draw["log_s0"] + draw["d_spread"]]))
        line = np.exp(np.array([draw["log_L0"], draw["log_L0"] + 2.0 * draw["d_amp"]]))
        peak = np.exp(np.array([draw["log_f0"], draw["log_f0"] + draw["d_freq"]]))
        phi = float(draw["phi"])
        tau = float(draw["tau"])

        cont = s2[:, None] / (1.0 - 2.0 * phi * np.cos(2.0 * np.pi * freqs)[None, :] + phi**2)
        gauss = line[:, None] * np.exp(
            -((freqs[None, :] - peak[:, None]) ** 2) / (2.0 * self._delta_f**2)
        )
        spectrum = cont + gauss
        ll: Array = (
            -(1.0 / ENBW)
            * np.sum(
                np.log(spectrum)[None, :, :] + pgram[:, None, :] / spectrum[None, :, :], axis=2
            )
            - 0.5 * np.log(2.0 * np.pi)
            - np.log(tau)
            - 0.5 * (means[:, None] - mu[None, :]) ** 2 / tau**2
        )
        return ll

    # -- the one-state null (certify / calibrate) -----------------------------

    def _null_objective(
        self, theta: Array, means: Array, pgram: Array, freqs: Array
    ) -> tuple[float, float]:
        """The one-regime log-likelihood and log-prior at an unconstrained point.

        ``theta`` is ``(mu, log_tau, log_s, z_phi, log_L, log_f)`` with ``phi``
        through a logistic. The likelihood is exactly the two-state emission
        with one regime: the deflated Whittle sum plus the Gaussian frame-mean
        term. The prior is the state-0 prior of :meth:`emissions` (with the
        change-of-variable Jacobians for the log and logistic transforms), so
        the maximizer is the MAP of the one-regime model, not a bare MLE that
        could chase an unidentified line into a corner.
        """
        mu, log_tau, log_s, z_phi, log_line, log_f = (float(v) for v in theta)
        tau = float(np.exp(log_tau))
        phi = float(1.0 / (1.0 + np.exp(-z_phi)))
        s2 = float(np.exp(2.0 * log_s))
        line = float(np.exp(log_line))
        peak = float(np.exp(log_f))

        cont = s2 / (1.0 - 2.0 * phi * np.cos(2.0 * np.pi * freqs) + phi**2)
        gauss = line * np.exp(-((freqs - peak) ** 2) / (2.0 * self._delta_f**2))
        spectrum = cont + gauss
        loglik = float(
            -(1.0 / ENBW) * np.sum(np.log(spectrum)[None, :] + pgram / spectrum[None, :])
            + np.sum(-0.5 * np.log(2.0 * np.pi) - np.log(tau) - 0.5 * (means - mu) ** 2 / tau**2)
        )
        # The state-0 priors of emissions(), on the unconstrained scale.
        logprior = float(
            -0.5 * mu**2  # mu0 ~ N(0, 1)
            - 2.0 * tau**2
            + log_tau  # tau ~ HalfNormal(0.5), + log-Jacobian
            - 0.5 * (log_s + 1.5) ** 2  # log_s0 ~ N(-1.5, 1)
            + 4.0 * np.log(phi)
            + 1.0 * np.log1p(-phi)  # phi ~ Beta(5, 2)
            + np.log(phi * (1.0 - phi))  # logistic Jacobian
            - 0.5 * ((log_line + 2.0) / 2.0) ** 2  # log_L0 ~ N(-2, 2)
            - 0.5 * ((log_f - np.log(0.045)) / 0.6) ** 2  # log_f0 ~ N(log 0.045, 0.6)
        )
        return loglik, logprior

    def _null_fit(self, data: PreparedData) -> tuple[dict[str, float], float]:
        """A short MAP of the one-regime model; returns its parameters and log-likelihood.

        Fitted by maximizing the one-regime posterior (likelihood plus the
        state-0 priors) with Nelder-Mead over the six unconstrained
        parameters, from two starts: the line initialized at the mean
        periodogram's peak bin, and at the prior's center frequency. MAP
        rather than MLE because the line is weakly identified on a control
        series, and the priors keep it from wandering; MAP rather than a
        sampler because this fit runs once per control world and has to stay
        cheap. The returned scalar is the likelihood term alone at the
        optimum, so it compares directly against the two-state marginal
        log-likelihood, which also carries no prior.
        """
        try:
            from scipy.optimize import minimize
        except ImportError as exc:  # pragma: no cover - scipy rides in with pymc
            raise ImportError(
                "the spectral null fit optimizes with scipy, which rides in with "
                "pymc: pip install 'ergodic[latent]'"
            ) from exc

        means = data.obs["means"]
        pgram = data.obs["pgram"]
        freqs = data.obs["freqs"]

        def loss(theta: Array) -> float:
            loglik, logprior = self._null_objective(theta, means, pgram, freqs)
            return -(loglik + logprior)

        mu0 = float(means.mean())
        tau0 = max(float(means.std()), 1e-3)
        s0 = max(float(np.sqrt(np.mean(pgram))), 1e-4)
        peak_bin = int(np.argmax(pgram.mean(axis=0)))
        line0 = max(float(pgram.mean(axis=0)[peak_bin]), 1e-4)
        starts = [
            np.array(
                [mu0, np.log(tau0), np.log(s0), 0.8, np.log(line0), np.log(float(freqs[peak_bin]))]
            ),
            np.array([mu0, np.log(tau0), np.log(s0), 0.8, np.log(line0), np.log(0.045)]),
        ]
        best = min(
            (
                minimize(
                    loss,
                    start,
                    method="Nelder-Mead",
                    options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8},
                )
                for start in starts
            ),
            key=lambda res: float(res.fun),
        )
        theta = np.asarray(best.x, dtype=np.float64)
        loglik, _ = self._null_objective(theta, means, pgram, freqs)
        mu, log_tau, log_s, z_phi, log_line, log_f = (float(v) for v in theta)
        params = {
            "mu": mu,
            "tau": float(np.exp(log_tau)),
            "sigma": float(np.exp(log_s)),
            "phi": float(1.0 / (1.0 + np.exp(-z_phi))),
            "line": float(np.exp(log_line)),
            "f0": float(np.exp(log_f)),
        }
        return params, loglik

    def null_loglik(self, data: PreparedData) -> float:
        """The one-regime Whittle + frame-mean log-likelihood at its MAP.

        The same emission the two-state model uses, with one regime instead
        of two, so the difference against the two-state marginal
        log-likelihood is a real improvement on identical data. The value is
        the likelihood term at the MAP point; the priors regularize the fit
        (see :meth:`_null_fit`) but are not part of the reported number.
        """
        return self._null_fit(data)[1]

    def _unit_line_peak(self, f0: float) -> float:
        """Peak periodogram of a unit-amplitude sinusoid through the frame pipeline.

        Computed numerically through the exact preparation (per-frame linear
        detrend, Hann window, the same normalization), so leakage and off-bin
        placement are priced in rather than approximated by the textbook
        ``W/6``.
        """
        t = np.arange(self.window, dtype=np.float64)
        frame = np.sin(2.0 * np.pi * f0 * t)
        centred = frame - frame.mean()
        tc = t - (self.window - 1) / 2.0
        centred = centred - (centred * tc).sum() / (tc**2).sum() * tc
        win = np.hanning(self.window)
        spec = np.abs(np.fft.rfft(centred * win)) ** 2 / np.sum(win**2)
        return float(spec.max())

    def simulate_null(self, data: PreparedData, rng: np.random.Generator) -> Array:
        """Synthesize a control world from the fitted one-regime model, in the time domain.

        The fitted null is a spectrum, not a generator, so the control world
        is built from the three things the spectrum describes: an AR(1) with
        the fitted continuum's innovation scale and memory (``sigma``,
        ``phi``), one sinusoid at the fitted line frequency whose amplitude
        is chosen so its periodogram peak through this exact frame pipeline
        equals the fitted line power, and the fitted mean. The phase of the
        sinusoid is drawn uniformly, and that is the honest choice rather
        than a compromise: phase is not a parameter of the Whittle model (the
        periodogram never sees it), so a null world must not pin it. Two
        stated approximations remain: a deterministic line's periodogram is
        noncentral chi-squared where the model says exponential (the same
        quasi-likelihood caveat the fit itself carries), and the frame-mean
        scatter ``tau`` is not injected separately, because the continuum's
        own low-frequency power supplies the frame-mean wobble and a refit on
        the control world absorbs it into ``tau``. A line the pipeline cannot
        see (peak response below 1e-6, which only happens against the DC
        detrend) is dropped rather than amplified into absurdity.
        """
        params, _ = self._null_fit(data)
        n = data.T_eff * self.window
        sigma, phi = params["sigma"], params["phi"]
        eps = rng.normal(0.0, sigma, size=n)
        out = np.empty(n)
        out[0] = float(rng.normal(0.0, sigma / np.sqrt(max(1.0 - phi**2, 1e-12))))
        for t in range(1, n):
            out[t] = phi * out[t - 1] + eps[t]
        q = self._unit_line_peak(params["f0"])
        if q > 1e-6:
            amplitude = float(np.sqrt(params["line"] / q))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            out += amplitude * np.sin(2.0 * np.pi * params["f0"] * np.arange(n) + phase)
        return out + params["mu"]


def spectral_whittle(*, window: int = 96, bin_lo: int = 1, bin_hi: int = 41) -> SpectralWhittle:
    """The flagship likelihood: frame spectra with level, spread, freq, and amp deltas.

    The defaults are the measured configuration: 96-tick Hann frames, bins 1
    through 40. Change them only with a reason at hand, because the
    calibration numbers in this module's docstring were measured at exactly
    these settings.
    """
    return SpectralWhittle(window=window, bin_lo=bin_lo, bin_hi=bin_hi)
