# SPDX-License-Identifier: Apache-2.0
"""Bass diffusion: the S-curve a new product traces out.

A new product has no history to extrapolate, so a trend model has nothing to say
about it. What it has instead is a shape, and the shape has been the same since
Bass wrote it down in 1969: a few people adopt because they heard about it
(``p``, the coefficient of innovation), and everyone else adopts because they saw
someone else adopt (``q``, the coefficient of imitation). Two numbers and a
market size give the whole curve.

In cumulative-fraction form, with ``F`` the share of the market that has adopted:

    dF_t = (p + q F_{t-1}) (1 - F_{t-1})
    F_t  = F_{t-1} + dF_t

and the series itself is incremental adoption, ``M dF_t``, for a market of size
``M``. The peak lands at ``ln(q / p) / (p + q)``, which is the number a launch
plan is usually asking for.

Three things this is honest about. It is **pure Bass plus noise**: there is no
trend state, no innovation on the level, so a launch that deviates from the curve
is charged to the likelihood rather than absorbed. It is **deterministic given
its parameters**, so the forecast band is parameter uncertainty and observation
noise and nothing else. And ``M`` for a curve that has not turned over yet is
famously weakly identified, which shows up as a wide posterior rather than as a
wrong number: read the interval, not the median, in the first few periods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._mechanism import TemporalMechanism, require_pymc
from .._scalers import RatioScaler, Scaler

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    import numpy.typing as npt

    Array = npt.NDArray[np.float64]

__all__ = ["Bass"]


@dataclass(frozen=True)
class Bass(TemporalMechanism):
    """The Bass diffusion curve for a new product's adoption.

    Options:

    - ``launch_index``: the step the product launches at. Everything before it is
      exactly zero adoption, not a small one, so a series that carries a run of
      pre-launch zeros is modeled rather than fought.
    - ``p_prior``, ``q_prior``: ``Beta`` parameters for innovation and imitation.
      The defaults sit where the diffusion literature's meta-analyses put them:
      innovation small (``Beta(2, 50)``, mean about 0.04) and imitation an order
      of magnitude larger (``Beta(2, 5)``, mean about 0.29), which is what makes
      the curve S-shaped rather than exponential-decaying.
    - ``m_prior``: ``(mu, sigma)`` of the LogNormal on the market size, on the
      scaled series. Left unset it is anchored at the total already adopted, with
      ``sigma = 1`` around it, which is a factor of seven either way at two
      standard deviations: enough that the likelihood decides, and anchored so
      the prior means the same thing for a series in units and one in millions.
    - ``scaler``: override the automatic choice.

    Parents are not read. A driver that moves adoption is a real thing to want and
    it is a different model (the hybrid correction the plan defers), so passing
    one raises rather than being ignored.
    """

    launch_index: int = 0
    p_prior: tuple[float, float] = (2.0, 50.0)
    q_prior: tuple[float, float] = (2.0, 5.0)
    m_prior: tuple[float, float] | None = None
    scaler: Scaler | None = None

    def __post_init__(self) -> None:
        if int(self.launch_index) < 0:
            raise ValueError(f"launch_index must be >= 0, got {self.launch_index}")
        for label, prior in (("p_prior", self.p_prior), ("q_prior", self.q_prior)):
            if len(tuple(prior)) != 2 or min(prior) <= 0:
                raise ValueError(f"{label} must be two positive Beta parameters, got {prior!r}")
        if self.m_prior is not None and (len(tuple(self.m_prior)) != 2 or self.m_prior[1] <= 0):
            raise ValueError(
                f"m_prior must be (mu, sigma) of a LogNormal with sigma > 0, got {self.m_prior!r}"
            )
        if self.scaler is not None and not isinstance(self.scaler, Scaler):
            raise TypeError(f"scaler must be a Scaler, got {type(self.scaler).__name__}")

    # -- resolution ----------------------------------------------------------

    def scaler_for(self, values: Array) -> Scaler:
        """Divide by the series' own **mean**, not its median.

        The automatic choice would take the median of the positives, and on a
        diffusion curve that is the wrong anchor by an order of magnitude or two:
        most of a completed S-curve is tail, so its median sits near zero and
        dividing by it would put the peak in the hundreds and the market size in
        the thousands. The mean is the anchor that matches what the parameter
        means, because the quantity of interest here is the *area* under the
        curve: divide by the mean and the market size lands near the number of
        steps, whatever the series is measured in.

        The floor comes off too. A ratio scaler normally clips from below to keep
        a positive likelihood away from zero, and a diffusion tail genuinely is
        zero, so clipping it would invent adoption that never happened.
        """
        if self.scaler is not None:
            return self.scaler.fit(values)
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        positive = finite[finite > 0.0]
        anchor = float(np.mean(positive)) if positive.size else 1.0
        return RatioScaler(floor=0.0, median=anchor if abs(anchor) > 1e-8 else 1.0)

    # -- the decomposition ---------------------------------------------------

    def baseline_label(self) -> str:
        """Nothing here is a trend: the whole path is the diffusion curve."""
        return "Diffusion"

    def factors(self, node: str, components: Mapping[str, Array]) -> tuple[str, ...] | None:
        """No factors at all, which is a real answer rather than a refusal.

        Pure diffusion has no drivers and no season, so the whole mean path is
        baseline. Saying so beats falling through to replay, which would spend a
        sampler to discover the same thing.
        """
        if self.site(node, "mu") not in components:
            return None
        return ()

    def compose(
        self,
        node: str,
        components: Mapping[str, Array],
        posterior: Mapping[str, Array],
        *,
        active: Collection[str],
    ) -> Array:
        """The adoption path itself, whatever is switched on, since nothing can be off."""
        return np.asarray(components[self.site(node, "mu")], dtype=np.float64)

    def peak_step(self, p: float, q: float) -> float:
        """Where the incremental-adoption curve peaks: ``ln(q / p) / (p + q)``, plus the launch."""
        if p <= 0 or q <= 0:
            raise ValueError(f"p and q must be positive, got p={p}, q={q}")
        return float(self.launch_index + np.log(q / p) / (p + q))

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
        """Register ``p``, ``q``, ``M`` and return ``mu``, the incremental-adoption path."""
        pm = require_pymc()
        from pytensor.scan.basic import scan

        if T < 2:
            raise ValueError(f"a Bass mechanism needs at least 2 steps, got T={T}")
        if horizon < 0:
            raise ValueError(f"horizon must be >= 0, got {horizon}")
        if parents:
            raise NotImplementedError(
                f"{name!r} carries a Bass mechanism, which is pure diffusion: it has no slot "
                f"for the driver(s) {sorted(parents)}. A driver that moves adoption is the "
                "hybrid form, which is deferred; until then model the driver separately or "
                "give this node a StructuralMechanism or CausalOU."
            )
        n_total = int(T) + int(horizon)
        launch = int(self.launch_index)
        if launch >= T:
            raise ValueError(
                f"launch_index={launch} is at or past the end of the {T}-step training "
                "window, so no observed step carries any adoption to fit"
            )
        site = self.site

        def named(label: str, value: Any) -> Any:
            """A deterministic, but only in a forecasting model (S2's audit rule 4)."""
            return pm.Deterministic(site(name, label), value) if horizon > 0 else value

        p = pm.Beta(site(name, "p"), alpha=self.p_prior[0], beta=self.p_prior[1])
        q = pm.Beta(site(name, "q"), alpha=self.q_prior[0], beta=self.q_prior[1])
        mu_m, sigma_m = self._market_prior(obs, mask)
        market = pm.LogNormal(site(name, "M"), mu=mu_m, sigma=sigma_m)
        sigma_obs = pm.HalfNormal(site(name, "sigma_obs"), sigma=0.5)

        active = np.zeros(n_total, dtype=np.float64)
        active[launch:] = 1.0

        def step(on_t: Any, cumulative_prev: Any, p_: Any, q_: Any) -> Any:
            increment = on_t * (p_ + q_ * cumulative_prev) * (1.0 - cumulative_prev)
            return cumulative_prev + increment

        cumulative = scan(
            fn=step,
            sequences=[active],
            outputs_info=[pm.math.zeros(())],
            non_sequences=[p, q],
            strict=True,
            return_updates=False,
        )
        # The increment is the difference of the cumulative path, and the first
        # step's increment is the whole of it, since nothing had adopted before.
        increments = pm.math.concatenate([cumulative[:1], cumulative[1:] - cumulative[:-1]])
        named("cumulative", cumulative)
        mu = named("mu", market * increments)

        if obs is not None:
            observed = np.asarray(obs, dtype=np.float64)
            keep = np.flatnonzero(np.asarray(mask, dtype=bool))
            if observed.shape != (T,):
                raise ValueError(f"obs has shape {observed.shape}, expected ({T},)")
            if keep.size == 0:
                raise ValueError(f"{name!r} has no observed steps to fit")
            pm.Normal(site(name, "obs"), mu=mu[keep], sigma=sigma_obs, observed=observed[keep])

        if horizon > 0:
            pm.Normal(site(name, "y_fc"), mu=mu[T:], sigma=sigma_obs, shape=horizon)
        return mu

    def _market_prior(self, obs: Array | None, mask: npt.NDArray[np.bool_]) -> tuple[float, float]:
        """The LogNormal parameters for ``M``, anchored on what has already been adopted."""
        if self.m_prior is not None:
            return float(self.m_prior[0]), float(self.m_prior[1])
        anchor = 1.0
        if obs is not None:
            observed = np.asarray(obs, dtype=np.float64)[np.asarray(mask, dtype=bool)]
            total = float(np.nansum(observed))
            if total > 0.0:
                anchor = total
        return float(np.log(anchor)), 1.0
