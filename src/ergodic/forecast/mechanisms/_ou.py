# SPDX-License-Identifier: Apache-2.0
"""Causal Ornstein-Uhlenbeck: the drivers move where the series is heading.

The structural mechanism regresses the *level* on its drivers, so a driver's
effect lands the moment it moves and leaves the moment it stops. Plenty of series
do not behave like that. Inventory days, a margin, a utilization rate, a price
index against its competitors: they sit near an equilibrium, and what a driver
changes is the equilibrium, not the value. The value then walks toward the new
equilibrium at its own speed.

That is what this mechanism says, in one line of scaled-space arithmetic:

    mu_LR_t   = mu0 + sum_b beta_b parent_b[t]
    level_t   = level_{t-1} + kappa (mu_LR_t - level_{t-1}) + sigma_level z_t

``kappa`` is the reversion rate, on ``(0, 1)`` by construction with a
``Beta(2, 2)`` prior: at 0 the level ignores its attractor entirely and at 1 it
lands on it in one step, and a business series is almost always in between. The
half-life of a shock is ``ln(2) / -ln(1 - kappa)``, which is the number worth
quoting to whoever asks how long a price change takes to show up.

The difference from a level regression is the whole point and it is testable:
hold a driver two units above where it was, and this mechanism's forecast walks
to ``mu0 + 2 beta`` and stays, where a level regression would step there at once.
The transient is the answer to "when", not just "how much".

Everything else is shared with the structural workhorse: the same optional
season, the same Normal or Student-t likelihood, the same non-centered
innovations, the same naming discipline. Two site names are added, ``kappa`` and
``mu0``, and there is no slope state at all: an attractor with its own trend is a
different model and is not this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._mechanism import TemporalMechanism, auto_scaler, parent_path, require_pymc
from .._scalers import Scaler
from ._structural import _fourier_design

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

    Array = npt.NDArray[np.float64]

__all__ = ["CausalOU"]

_SEASONALITIES = ("auto", "monthly", "fourier")


@dataclass(frozen=True)
class CausalOU(TemporalMechanism):
    """A mean-reverting level whose long-run attractor the drivers set.

    Options:

    - ``period``: the season length, as in
      :class:`~ergodic.forecast.StructuralMechanism`. ``None`` is no season.
    - ``seasonality``: ``"auto"`` (read the period), ``"monthly"``, ``"fourier"``,
      or ``None``.
    - ``n_harmonics``: how many sine and cosine pairs the Fourier form uses.
    - ``robust``: a Student-t likelihood rather than a Normal one.
    - ``reg_scale``: the prior scale on each attractor coefficient, on the scaled
      series. Smaller than the structural mechanism's by default (0.2 against
      1.0), because a coefficient here moves the equilibrium of a series that is
      scaled to sit near one, and an equilibrium that swings by a whole unit of
      the series is a very large claim.
    - ``scaler``: override the automatic choice.

    The priors, on the scaled series: ``kappa ~ Beta(2, 2)``,
    ``mu0 ~ Normal(1, 0.5)``, ``sigma_level ~ HalfNormal(0.3)``,
    ``sigma_obs ~ HalfNormal(0.5)``, ``level0 ~ Normal(1, 0.5)``.
    """

    period: int | None = None
    seasonality: str | None = "auto"
    n_harmonics: int = 2
    robust: bool = False
    reg_scale: float = 0.2
    scaler: Scaler | None = None

    def __post_init__(self) -> None:
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
        if self.reg_scale <= 0:
            raise ValueError(f"reg_scale must be positive, got {self.reg_scale}")
        if self.scaler is not None and not isinstance(self.scaler, Scaler):
            raise TypeError(f"scaler must be a Scaler, got {type(self.scaler).__name__}")

    # -- resolution ----------------------------------------------------------

    def seasonal_form(self) -> tuple[str | None, int | None]:
        """The seasonal form actually used, as ``(kind, period)``. Same rule as S2's."""
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
        """Register every site and return ``mu``, the mean path of length ``T + horizon``."""
        pm = require_pymc()
        from pytensor.scan.basic import scan

        if T < 2:
            raise ValueError(f"a causal OU mechanism needs at least 2 steps, got T={T}")
        if horizon < 0:
            raise ValueError(f"horizon must be >= 0, got {horizon}")
        n_total = int(T) + int(horizon)
        site = self.site

        def named(label: str, value: Any) -> Any:
            """A deterministic, but only in a forecasting model (S2's audit rule 4)."""
            return pm.Deterministic(site(name, label), value) if horizon > 0 else value

        kappa = pm.Beta(site(name, "kappa"), alpha=2.0, beta=2.0)
        mu0 = pm.Normal(site(name, "mu0"), mu=1.0, sigma=0.5)
        sigma_level = pm.HalfNormal(site(name, "sigma_level"), sigma=0.3)
        sigma_obs = pm.HalfNormal(site(name, "sigma_obs"), sigma=0.5)
        level0 = pm.Normal(site(name, "level0"), mu=1.0, sigma=0.5)

        z_level = pm.Normal(site(name, "z_level"), mu=0.0, sigma=1.0, shape=T)
        if horizon > 0:
            z_level = pm.math.concatenate(
                [z_level, pm.Normal(site(name, "z_level_fc"), mu=0.0, sigma=1.0, shape=horizon)]
            )

        # The drivers move the attractor, not the level. reg is registered under
        # the same names the structural mechanism uses, so a decomposition reads
        # the same way; what it means here is the shift in the equilibrium rather
        # than a term added to the mean.
        reg = pm.math.zeros(n_total)
        for base in sorted(parents):
            if base.endswith("_fc"):
                raise ValueError(
                    f"parent {base!r} of {name!r} ends in '_fc', which is the suffix that "
                    "marks a freshly simulated site; rename the variable"
                )
            values = parent_path(parents[base], node=name, base=base, n_total=n_total)
            beta = pm.Normal(site(name, f"beta_{base}"), mu=0.0, sigma=self.reg_scale)
            reg = reg + named(f"reg_{base}", beta * values)
        reg = named("reg", reg)
        attractor = named("attractor", mu0 + reg)

        def step(zl_t: Any, target_t: Any, level_prev: Any, kappa_: Any, sl_: Any) -> Any:
            return level_prev + kappa_ * (target_t - level_prev) + sl_ * zl_t

        levels = scan(
            fn=step,
            sequences=[z_level, attractor],
            outputs_info=[level0],
            non_sequences=[kappa, sigma_level],
            strict=True,
            return_updates=False,
        )
        trend = named("trend", levels)

        kind, period = self.seasonal_form()
        season = pm.math.zeros(n_total)
        factor = pm.math.ones(n_total)
        if kind == "monthly":
            raw = pm.Normal(site(name, "season_raw"), mu=0.0, sigma=0.5, shape=12)
            factor = (12.0 * pm.math.softmax(raw))[np.arange(n_total) % 12]
        elif kind == "fourier":
            if period is None:  # pragma: no cover - seasonal_form resolves it or raises
                raise ValueError(f"{name!r}: Fourier seasonality resolved without a period")
            coefs = pm.Normal(site(name, "fourier"), mu=0.0, sigma=0.3, shape=2 * self.n_harmonics)
            season = pm.math.dot(_fourier_design(n_total, period, self.n_harmonics), coefs)
        factor = named("season_factor", factor)
        season = named("season", season)

        mu = named("mu", (trend + season) * factor)

        nu: Any = None
        if self.robust:
            nu = named("nu", 2.0 + pm.Gamma(site(name, "nu_raw"), alpha=2.0, beta=0.1))

        if obs is not None:
            observed = np.asarray(obs, dtype=np.float64)
            keep = np.flatnonzero(np.asarray(mask, dtype=bool))
            if observed.shape != (T,):
                raise ValueError(f"obs has shape {observed.shape}, expected ({T},)")
            if keep.size == 0:
                raise ValueError(f"{name!r} has no observed steps to fit")
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
