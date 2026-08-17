# SPDX-License-Identifier: Apache-2.0
"""The temporal mechanism: one node's structural equation through time.

A :class:`TemporalMechanism` is to a forecast what
:class:`~ergodic.scm.Mechanism` is to a structural causal model. It owns one
node: the scale it works on, the states it carries, the priors over them, and
the likelihood that ties them to the data. It never sees raw pandas, only a
scaled numpy array and the sampler context, so the same mechanism serves a
series in dollars and a series in units.

Two rules hold the contract together. Every stochastic input is its own named
random variable, so the mean path is a deterministic function of parameters,
innovations, and parents; that is the explicit-noise discipline the SCM merge
will need later. And every site a mechanism registers is prefixed with the node
name, so a posterior can be replayed into a second model by name alone.

:func:`assign_mechanisms` is the schema dispatch: a continuous column gets the
:class:`~ergodic.forecast.mechanisms.StructuralMechanism` with its seasonal
period read off the time index, and anything else raises rather than guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ._scalers import IdentityScaler, RatioScaler, Scaler, ZScoreScaler

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    import numpy.typing as npt

    from ..data import TimeSeriesData

    Array = npt.NDArray[np.float64]

__all__ = [
    "SEASON",
    "ScenarioMechanism",
    "TemporalMechanism",
    "assign_mechanisms",
    "auto_scaler",
    "infer_period",
    "is_symbolic",
    "parent_path",
    "regression_keys",
    "require_pymc",
]

#: The factor key for seasonality in the attribution contract. Driver factors are
#: named by their design key, so a driver column called ``"season"`` would collide
#: with this one; :meth:`TemporalMechanism.factors` refuses that rather than
#: silently merging the two bars.
SEASON = "season"


def require_pymc() -> Any:
    """Import pymc, or say how to get it.

    The engine is imported inside the methods that sample, never at module
    import, so the bare install of the package never loads it and never pays for
    it. This is the one place that turns its absence into a sentence a user can
    act on.
    """
    try:
        import pymc
    except ImportError as exc:  # pragma: no cover - exercised by import interception
        raise ImportError(
            "the forecast pillar samples with pymc, which the bare install does not "
            "carry: pip install 'ergodic[forecast]' (and jax plus numpyro for the fast "
            "sampler path)"
        ) from exc
    return pymc


def auto_scaler(values: Array) -> Scaler:
    """Pick and fit a scaler for one series: ratio when positive, z-score when signed.

    A series that never goes at or below zero is multiplicative in character, so
    dividing by its median puts it near one and leaves the proportions intact. A
    series that changes sign has no such median, so it gets centered and
    standardized instead. Only the finite entries count, which is what keeps a
    masked step from moving the scale.
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot scale a series with no finite values")
    if bool(np.all(finite > 0.0)):
        return RatioScaler().fit(finite)
    return ZScoreScaler().fit(finite)


def regression_keys(node: str, components: Mapping[str, Array]) -> tuple[str, ...]:
    """The design keys behind a node's ``{node}_reg_{key}`` deterministics, sorted.

    This is the naming contract read backwards: a forecast model registers one
    such deterministic per driver, so the drivers of a fitted node can be
    recovered from what it recorded, without asking the graph again.
    """
    prefix = f"{node}_reg_"
    keys = sorted(str(name)[len(prefix) :] for name in components if str(name).startswith(prefix))
    if SEASON in keys:
        raise ValueError(
            f"{node!r} has a driver whose design key is {SEASON!r}, which is the key "
            "seasonality uses in an attribution, so the two bars would be the same bar. "
            "Rename the column."
        )
    return tuple(keys)


def is_symbolic(value: Any) -> bool:
    """True when ``value`` is a pytensor variable rather than data.

    Duck-typed on purpose: this module is imported by the bare install, which has
    no pytensor to check against. A pytensor variable carries both a ``type`` and
    an ``owner``; a numpy array carries neither.
    """
    return hasattr(value, "type") and hasattr(value, "owner")


def parent_path(values: Any, *, node: str, base: str, n_total: int) -> Any:
    """Validate one parent column and hand it back in a form the model can multiply.

    A parent arrives lag-aligned over ``T + horizon`` steps, and it arrives in one
    of two forms. Data (a numpy array) is the ordinary case: a driver whose future
    is known, or the whole training window. A **tensor** is the staged and joint
    case: a driver whose future is a random variable of this very model, either a
    frozen parent's injected ``{base}_fc`` draws or an upstream node's own mean
    path. Both are length ``T + horizon`` and nothing downstream needs to know
    which one it got.
    """
    if is_symbolic(values):
        shape = tuple(values.type.shape)
        if len(shape) != 1 or (shape[0] is not None and int(shape[0]) != n_total):
            raise ValueError(
                f"parent {base!r} of {node!r} is a tensor of shape {shape}, expected "
                f"({n_total},) = T + horizon"
            )
        return values
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (n_total,):
        raise ValueError(
            f"parent {base!r} of {node!r} has shape {array.shape}, expected "
            f"({n_total},) = T + horizon; a parent arrives lag-aligned and with its "
            "future already resolved"
        )
    return array


@dataclass(frozen=True)
class TemporalMechanism:
    """The contract: own a scale, build a mean path, register named sites.

    Subclasses implement :meth:`build`, which runs inside a ``pm.Model`` context
    and returns the mean path as a tensor of length ``T + horizon``. The
    arguments it reads:

    - ``obs``: the scaled observations, shape ``(T,)``, with whatever the masked
      steps happen to hold. A mechanism must index down to the observed steps
      before it writes a likelihood; masked steps never enter, and nothing is
      imputed on the way in.
    - ``parents``: design name to scaled, lag-aligned path of length
      ``T + horizon``. Either a numpy array (a driver whose future is known) or a
      tensor of this model (a frozen parent's injected draws, or an upstream
      node's own mean path in a joint fit). :func:`parent_path` validates both
      and a mechanism should route every parent through it. The design name is
      the parent's variable name when it drives this node at one lag, and
      ``{base}_lag{k}`` for the extra lags when it drives at several.
    - ``mask``: boolean ``(T,)``, True where the series was observed.
    - ``horizon``: zero for a fitting model, ``H`` for a forecasting one. A
      forecasting model adds fresh innovation sites whose names end in ``_fc``,
      which is how the posterior-predictive pass tells replay from simulation.

    Every registered name goes through :meth:`site`, so the naming contract has
    exactly one implementation.
    """

    def site(self, node: str, site: str) -> str:
        """The name of one site: ``"sales"`` and ``"phi"`` give ``"sales_phi"``.

        Every mechanism registers through this, and nothing renames a site
        afterwards. Posterior replay, frozen-parent injection, and attribution
        zeroing all match by name.
        """
        return f"{node}_{site}"

    def predictive_site(self, node: str) -> str:
        """The site holding the horizon's predictive draws in a forecasting model."""
        return self.site(node, "y_fc")

    def mean_site(self, node: str) -> str:
        """The deterministic holding the mean path, length ``T + horizon``.

        Registered by a forecasting model only, never by a fitting one, which is
        why in-sample fitted values are read off the forecast model's first ``T``
        rows rather than out of the trace.
        """
        return self.site(node, "mu")

    def scaler_for(self, values: Array) -> Scaler:
        """The fitted scaler this mechanism wants for ``values``."""
        return auto_scaler(values)

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
        """Register this node's sites and return its mean path, length ``T + horizon``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement build(); a temporal mechanism "
            "has to register its own sites and return its mean path"
        )

    # -- the decomposition contract (what .explain() reads) -------------------

    def baseline_label(self) -> str:
        """What to call the part of the forecast that is nobody's driver."""
        return "Level and trend"

    def factors(self, node: str, components: Mapping[str, Array]) -> tuple[str, ...] | None:
        """The switchable pieces of this node's mean path, in composition order.

        ``None``, the default, means this mechanism cannot be taken apart from its
        own deterministics, so :meth:`~ergodic.forecast.ForecastResult.explain`
        falls back to the replay engine: it zeroes a factor's sites in a copy of
        the trace and forecasts again. A mechanism that *can* be decomposed
        returns its factor keys instead, which costs no sampling at all.

        The keys are :data:`SEASON` for the seasonal term and the design key for
        each driver, ordered seasonality first and then drivers by name. The order
        is what ``method="sequential"`` walks. An empty tuple is a real answer: it
        says the whole mean path is baseline, which is what a pure diffusion curve
        is.

        Read the ``components`` rather than the mechanism's own settings, because
        the question is what this *forecast* recorded, and a subclass that hides
        its deterministics has to fall through to replay.
        """
        return None

    def compose(
        self,
        node: str,
        components: Mapping[str, Array],
        posterior: Mapping[str, Array],
        *,
        active: Collection[str],
    ) -> Array:
        """The mean path with only ``active`` factors switched on, ``(S, T + H)``.

        The identity elements are the pinned parameterization: a season switched
        off is a factor of exactly one and an additive term of exactly zero, and a
        driver switched off contributes zero. ``compose`` with every factor active
        must reproduce the model's own ``mu`` deterministic, which is not an
        assumption: :meth:`~ergodic.forecast.ForecastResult.explain` measures the
        gap and reports it as ``closure_error``, so a composition rule that drifts
        away from ``build`` shows up as a number rather than as silence.

        ``posterior`` carries the fitted sites for a mechanism whose composition
        needs a parameter as well as the recorded pieces (the reversion rate of a
        mean-reverting level, say).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement compose(); a mechanism that "
            "returns factors() has to be able to rebuild its own mean path"
        )


@dataclass(frozen=True)
class ScenarioMechanism(TemporalMechanism):
    """A node whose future is given rather than modeled.

    This is the resolved role: nothing is sampled and nothing is fitted, because
    the path comes from outside. An exogenous driver with a known future (a
    published rate, a planned price schedule, a calendar) is the case it exists
    for. The scaler is the identity, since a supplied path arrives in the units
    the caller means.

    ``values`` may pin the future path at construction time. Leaving it ``None``
    means the path arrives at forecast time through ``scenario=``, which the
    driver sprint wires up.
    """

    values: tuple[float, ...] | None = None

    def path(self) -> Array:
        """The pinned future path as an array; raises when nothing was pinned."""
        if self.values is None:
            raise ValueError(
                "this ScenarioMechanism has no pinned path; supply one at "
                "construction (ScenarioMechanism(values=...)) or through scenario="
            )
        return np.asarray(self.values, dtype=np.float64)

    def scaler_for(self, values: Array) -> Scaler:
        """The identity: a supplied path is already on the scale the caller means."""
        return IdentityScaler()

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
        """Refuse: a resolved node has no model to build."""
        raise NotImplementedError(
            f"{name!r} carries a ScenarioMechanism, which is resolved rather than "
            "sampled: its path is supplied, so there is nothing to fit. Give it a "
            "StructuralMechanism if you want it modeled."
        )


# -- seasonal period inference ------------------------------------------------

# Median spacing in days to seasonal period. The bands are wide enough for real
# calendars (a month is 28 to 31 days, a quarter 89 to 92, and a week is exact
# only until a daylight-saving boundary moves it by an hour).
_PERIOD_BY_SPACING: tuple[tuple[float, float, int], ...] = (
    (6.0, 8.0, 52),
    (28.0, 31.0, 12),
    (89.0, 93.0, 4),
)


def infer_period(data: TimeSeriesData) -> int | None:
    """The seasonal period implied by the time index, or ``None`` when unclear.

    Monthly spacing gives 12, quarterly 4, weekly 52. Anything else, including a
    plain integer step index, gives ``None``, which means no seasonal term unless
    the caller names a period. Guessing a period from data that does not carry
    one is how a forecast grows a season it never had.
    """
    times = np.asarray(data.times)
    if times.size < 3 or times.dtype.kind != "M":
        return None
    spacing = np.diff(times.astype("datetime64[s]").astype(np.int64)) / 86400.0
    if spacing.size == 0:
        return None
    median = float(np.median(spacing))
    for low, high, period in _PERIOD_BY_SPACING:
        if low <= median <= high:
            return period
    return None


# -- schema dispatch ----------------------------------------------------------


def assign_mechanisms(
    data: TimeSeriesData,
    mechanisms: Mapping[str, TemporalMechanism] | None = None,
    *,
    robust: bool = False,
) -> dict[str, TemporalMechanism]:
    """One mechanism per modeled column: the caller's choice, else the schema's.

    A column named in ``mechanisms`` keeps the mechanism it was given, whatever
    the schema says. Every other continuous column gets a
    :class:`~ergodic.forecast.mechanisms.StructuralMechanism` with the period
    :func:`infer_period` read off the time index. A column that is not continuous
    raises: counts and categories need their own mechanisms, and a Gaussian trend
    model fitted to a count series is worse than a refusal.

    ``robust`` reaches the auto-assigned mechanisms only. A mechanism the caller
    built keeps its own settings, because an explicit object outranks a flag.
    """
    from .mechanisms import StructuralMechanism

    given = dict(mechanisms or {})
    unknown = sorted(set(given) - set(data.variables))
    if unknown:
        raise KeyError(
            f"mechanisms names columns the data does not model: {unknown}; "
            f"it holds {list(data.variables)}"
        )
    for node, mech in given.items():
        if not isinstance(mech, TemporalMechanism):
            raise TypeError(
                f"mechanisms[{node!r}] is a {type(mech).__name__}, not a TemporalMechanism; "
                "pass a mechanism instance (StructuralMechanism(...), ScenarioMechanism(...))"
            )

    period = infer_period(data)
    out: dict[str, TemporalMechanism] = {}
    for node in data.variables:
        if node in given:
            out[node] = given[node]
            continue
        column = data.schema[node]
        if not column.is_continuous:
            raise NotImplementedError(
                f"column {node!r} is {column.kind.value}, and count and categorical "
                "mechanisms are a reserved seam; write a custom TemporalMechanism or "
                "exclude the column"
            )
        out[node] = StructuralMechanism(period=period, robust=robust)
    return out
