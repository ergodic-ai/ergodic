# SPDX-License-Identifier: Apache-2.0
"""The structural workhorse: damped trend, season, and a regression on parents.

One series, three pieces, composed rather than fused:

- a **damped local linear trend**, the level that carries its own slope and the
  slope that decays toward zero at rate ``phi``. Damping is what keeps a
  twelve-step extrapolation from running away on a slope the last few points
  happened to suggest.
- a **season**, either the twelve monthly factors or a Fourier pair, whichever
  the period calls for.
- a **regression** on the node's parents, one coefficient each.

The mean path is ``mu = (trend + reg + season) * season_factor``. Exactly one
seasonal form is ever active: the monthly form moves ``season_factor`` and
leaves ``season`` at zero, the Fourier form does the reverse, and with no season
both sit at their neutral values. That is what makes the decomposition
subtractable later: switching a piece off means setting its raw parameters to
zero, and its neutral value falls out of the same arithmetic.

The states are written non-centered (a standard normal innovation times a scale)
because the centered form funnels: the sampler cannot explore a narrow slope
scale and its innovations at the same time. Every innovation is its own named
site, which is both the explicit-noise discipline and the mechanism that lets a
forecast replay the fitted history and simulate only the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._mechanism import TemporalMechanism, auto_scaler, parent_path, require_pymc
from .._scalers import Scaler

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

    Array = npt.NDArray[np.float64]

__all__ = ["StructuralMechanism"]

_SEASONALITIES = ("auto", "monthly", "fourier")
_TRENDS = ("damped_llt",)


@dataclass(frozen=True)
class StructuralMechanism(TemporalMechanism):
    """A damped local linear trend with seasonality and regression on parents.

    Options:

    - ``trend``: ``"damped_llt"``, the only form so far.
    - ``seasonality``: ``"auto"``, ``"monthly"``, ``"fourier"``, or ``None``.
      ``"auto"`` reads the period: twelve gives the monthly factors, any other
      period gives Fourier terms, and no period gives no season.
    - ``period``: the season length. Monthly means twelve by definition.
    - ``n_harmonics``: how many sine and cosine pairs the Fourier form uses.
    - ``robust``: swap the Normal likelihood for a Student-t, which lets a few
      wild points stay wild instead of dragging the level after them.
    - ``reg_scale``: the prior scale on each parent coefficient, on the scaled
      series.
    - ``scaler``: override the automatic choice (positive series to ratio,
      signed series to z-score). The mechanism fits whatever it is given.

    The priors are written for the ratio scale, where the series sits near one:
    ``phi ~ Beta(8, 2)`` (damping near but below one), ``sigma_level ~
    HalfNormal(0.5)``, ``sigma_slope ~ HalfNormal(0.05)`` (a slope moves slowly
    or it is not a trend), ``sigma_obs ~ HalfNormal(0.5)``, ``level0 ~
    Normal(1, 0.5)``, ``slope0 ~ Normal(0, 0.1)``, and ``season_raw ~
    Normal(0, 0.5)``.
    """

    trend: str = "damped_llt"
    seasonality: str | None = "auto"
    period: int | None = None
    n_harmonics: int = 2
    robust: bool = False
    reg_scale: float = 1.0
    scaler: Scaler | None = None

    def __post_init__(self) -> None:
        if self.trend not in _TRENDS:
            raise ValueError(f"trend must be one of {_TRENDS}, got {self.trend!r}")
        if self.seasonality is not None and self.seasonality not in _SEASONALITIES:
            raise ValueError(
                f"seasonality must be one of {(*_SEASONALITIES, None)}, got {self.seasonality!r}"
            )
        if self.period is not None and int(self.period) < 2:
            raise ValueError(f"period must be at least 2, got {self.period}")
        if self.seasonality == "monthly" and self.period not in (None, 12):
            raise ValueError(
                f"monthly seasonality is twelve factors by definition, so period={self.period} "
                "does not fit it; use seasonality='fourier' for another period"
            )
        if self.n_harmonics < 1:
            raise ValueError(f"n_harmonics must be at least 1, got {self.n_harmonics}")
        if self.scaler is not None and not isinstance(self.scaler, Scaler):
            raise TypeError(f"scaler must be a Scaler, got {type(self.scaler).__name__}")

    # -- resolution ----------------------------------------------------------

    def seasonal_form(self) -> tuple[str | None, int | None]:
        """The seasonal form actually used, as ``(kind, period)``.

        ``kind`` is ``"monthly"``, ``"fourier"``, or ``None``. This is what the
        result's provenance reports, so a caller can see which season the data's
        own index bought them.
        """
        if self.seasonality is None:
            return None, None
        if self.seasonality == "monthly":
            return "monthly", 12
        if self.seasonality == "fourier":
            if self.period is None:
                raise ValueError(
                    "seasonality='fourier' needs a period; the time index did not imply one, "
                    "so name it with period="
                )
            return "fourier", int(self.period)
        if self.period is None:
            return None, None
        return ("monthly", 12) if int(self.period) == 12 else ("fourier", int(self.period))

    def scaler_for(self, values: Array) -> Scaler:
        """The fitted scaler: the override when given, else the automatic choice."""
        if self.scaler is not None:
            return self.scaler.fit(values)
        return auto_scaler(values)

    # -- the model -----------------------------------------------------------

    def build(
        self,
        name: str,
        *,
        obs: Array | None,
        parents: Mapping[str, Array],
        T: int,
        horizon: int,
        mask: npt.NDArray[np.bool_],
    ) -> Any:
        """Register every site and return ``mu``, the mean path of length ``T + horizon``.

        Runs inside a ``pm.Model`` context. When ``horizon`` is positive the
        model also carries the fresh forecast innovations (``z_level_fc``,
        ``z_slope_fc``) and the predictive site ``y_fc``, which is what the
        posterior-predictive pass reads back.
        """
        pm = require_pymc()
        from pytensor.scan.basic import scan

        if T < 2:
            raise ValueError(f"a structural mechanism needs at least 2 steps, got T={T}")
        if horizon < 0:
            raise ValueError(f"horizon must be >= 0, got {horizon}")
        n_total = int(T) + int(horizon)
        site = self.site

        def named(label: str, value: Any) -> Any:
            """Register a deterministic, but only in a forecasting model.

            A fitting model deliberately records none of these. A deterministic
            stored in the trace and free of volatile inputs is read back *from*
            the trace by the posterior-predictive pass rather than recomputed, so
            a fit-time ``season_factor`` of length T would be poured into a
            forecast-time slot of length T + H. Keeping the fit trace to its free
            variables makes that impossible instead of merely unlikely.
            """
            return pm.Deterministic(site(name, label), value) if horizon > 0 else value

        phi = pm.Beta(site(name, "phi"), alpha=8.0, beta=2.0)
        sigma_level = pm.HalfNormal(site(name, "sigma_level"), sigma=0.5)
        sigma_slope = pm.HalfNormal(site(name, "sigma_slope"), sigma=0.05)
        sigma_obs = pm.HalfNormal(site(name, "sigma_obs"), sigma=0.5)
        level0 = pm.Normal(site(name, "level0"), mu=1.0, sigma=0.5)
        slope0 = pm.Normal(site(name, "slope0"), mu=0.0, sigma=0.1)

        # Train innovations are replayed from the posterior; forecast innovations
        # are drawn fresh, which is the whole trick behind an honest band. The
        # two are concatenated before the scan, so the recursion never learns
        # which half of its own history it is walking.
        z_level = pm.Normal(site(name, "z_level"), mu=0.0, sigma=1.0, shape=T)
        z_slope = pm.Normal(site(name, "z_slope"), mu=0.0, sigma=1.0, shape=T)
        if horizon > 0:
            z_level = pm.math.concatenate(
                [z_level, pm.Normal(site(name, "z_level_fc"), mu=0.0, sigma=1.0, shape=horizon)]
            )
            z_slope = pm.math.concatenate(
                [z_slope, pm.Normal(site(name, "z_slope_fc"), mu=0.0, sigma=1.0, shape=horizon)]
            )

        def step(
            zl_t: Any,
            zs_t: Any,
            level_prev: Any,
            slope_prev: Any,
            phi_: Any,
            sl_: Any,
            ss_: Any,
        ) -> tuple[Any, Any]:
            level_t = level_prev + slope_prev + sl_ * zl_t
            slope_t = phi_ * slope_prev + ss_ * zs_t
            return level_t, slope_t

        states = scan(
            fn=step,
            sequences=[z_level, z_slope],
            outputs_info=[level0, slope0],
            non_sequences=[phi, sigma_level, sigma_slope],
            strict=True,
            return_updates=False,
        )
        trend = named("trend", states[0])

        reg = pm.math.zeros(n_total)
        for base in sorted(parents):
            if base.endswith("_fc"):
                raise ValueError(
                    f"parent {base!r} of {name!r} ends in '_fc', which is the suffix that "
                    "marks a freshly simulated site; rename the variable"
                )
            values = parent_path(parents[base], node=name, base=base, n_total=n_total)
            beta = pm.Normal(site(name, f"beta_{base}"), mu=0.0, sigma=self.reg_scale)
            contribution = named(f"reg_{base}", beta * values)
            reg = reg + contribution
        reg = named("reg", reg)

        kind, period = self.seasonal_form()
        season = pm.math.zeros(n_total)
        factor = pm.math.ones(n_total)
        if kind == "monthly":
            raw = pm.Normal(site(name, "season_raw"), mu=0.0, sigma=0.5, shape=12)
            # 12 * softmax is exactly one at raw = 0, so a zeroed season is a
            # neutral season and nothing has to be renormalized to switch it off.
            factor = (12.0 * pm.math.softmax(raw))[np.arange(n_total) % 12]
        elif kind == "fourier":
            if period is None:  # pragma: no cover - seasonal_form resolves it or raises
                raise ValueError(f"{name!r}: Fourier seasonality resolved without a period")
            coefs = pm.Normal(site(name, "fourier"), mu=0.0, sigma=0.3, shape=2 * self.n_harmonics)
            season = pm.math.dot(_fourier_design(n_total, period, self.n_harmonics), coefs)
        factor = named("season_factor", factor)
        season = named("season", season)

        mu = named("mu", (trend + reg + season) * factor)

        nu: Any = None
        if self.robust:
            # Two degrees of freedom is the boundary where the variance stops
            # existing, so the shift keeps the tail heavy but the moments real.
            nu = named("nu", 2.0 + pm.Gamma(site(name, "nu_raw"), alpha=2.0, beta=0.1))

        if obs is not None:
            observed = np.asarray(obs, dtype=np.float64)
            keep = np.flatnonzero(np.asarray(mask, dtype=bool))
            if observed.shape != (T,):
                raise ValueError(f"obs has shape {observed.shape}, expected ({T},)")
            if keep.size == 0:
                raise ValueError(f"{name!r} has no observed steps to fit")
            # Index down to the observed steps in numpy, before anything reaches
            # the model: a masked step contributes no term, so whatever sits in
            # its slot cannot touch the posterior.
            self._likelihood(name, mu[keep], sigma_obs, observed[keep], nu)

        if horizon > 0:
            self._likelihood(name, mu[T:], sigma_obs, None, nu, shape=horizon)
        return mu

    def _likelihood(
        self,
        name: str,
        mu: Any,
        sigma: Any,
        observed: Array | None,
        nu: Any,
        *,
        shape: int | None = None,
    ) -> Any:
        """The observation model, once for the fitted steps and once for the horizon."""
        pm = require_pymc()

        site = self.site(name, "obs" if observed is not None else "y_fc")
        kwargs: dict[str, Any] = {"mu": mu, "sigma": sigma}
        if observed is not None:
            kwargs["observed"] = observed
        else:
            kwargs["shape"] = shape
        if self.robust:
            return pm.StudentT(site, nu=nu, **kwargs)
        return pm.Normal(site, **kwargs)


def _fourier_design(n_steps: int, period: int, n_harmonics: int) -> Array:
    """The ``(n_steps, 2 * n_harmonics)`` seasonal design, columns ``[sin_k, cos_k]``.

    Time runs 0, 1, ... from the first observation, so the horizon rows continue
    the same phase and the season does not jump at the forecast origin.
    """
    t = np.arange(n_steps, dtype=np.float64)
    columns: list[Array] = []
    for k in range(1, n_harmonics + 1):
        angle = 2.0 * np.pi * k * t / float(period)
        columns.append(np.sin(angle))
        columns.append(np.cos(angle))
    return np.stack(columns, axis=1)
