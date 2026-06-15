# SPDX-License-Identifier: Apache-2.0
"""Mechanisms: the per-node structural equation, with explicit exogenous noise.

A :class:`Mechanism` is the structural equation for one node, ``X = f(PA) + U``
(additive, invertible) or ``X = g(PA, U)`` (general). It declares a capability
tier and exposes the jobs the SCM needs: build a small PyMC model to *fit* the
node on its observed parents, contribute the node's RV to the *generative* model
(parent RVs through the structural equation plus an explicit noise RV), and
*predict* a counterfactual by reusing abducted noise.

Capability tiers:

- ``PROBABILISTIC`` answers observational and interventional queries. A general
  conditional density (no recoverable noise) is probabilistic only, and its
  counterfactual is the wider abduction posterior.
- ``INVERTIBLE`` also answers counterfactuals exactly: the noise is recoverable
  from the node and its parents. Additive Gaussian is invertible
  (``U = X - f(PA)``); the Gumbel-max categorical is invertible through the
  truncated-Gumbel abduction.

Auto-assignment reads the schema, the same dispatch
:func:`~ergodic.inference.estimate_effect` and discovery use: a continuous node
gets :class:`GaussianAdditive` with a Bayesian-linear mean (the default), a binary
node a :class:`BinaryGumbelMax`, a categorical node a :class:`CategoricalGumbelMax`.

Overrides (increasing in power): a built-in name (``"linear"``, ``"gp"``), a
:class:`~ergodic.inference.Learner` instance as the mean, a
:class:`~ergodic.inference.PyMCLearner` subclass, or a raw callable that adds the
node's RVs to a model. pymc is imported only inside the methods, never at module
top, so the bare install never imports it (the ``PyMCLearner`` precedent).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from ._noise import gaussian_residual, gumbel_max_abduct, gumbel_max_decide

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from ..data import Dataset
    from ..graph import MixedGraph

    Array = npt.NDArray[np.float64]
    IntArray = npt.NDArray[np.int64]


class Capability(Enum):
    """What a mechanism's noise lets it answer.

    ``PROBABILISTIC`` covers observational and interventional queries.
    ``INVERTIBLE`` adds counterfactuals, because the structural noise is
    recoverable from the node and its parents.
    """

    PROBABILISTIC = "probabilistic"
    INVERTIBLE = "invertible"


@dataclass(frozen=True)
class FitResult:
    """A fitted mechanism's posterior, merged into the SCM's parameter store.

    ``params`` maps a (disciplined, node-prefixed) parameter name to its
    posterior draws, flattened to ``(S, ...)`` over the chain-draw axes. The
    same names appear in the generative model the compose helper builds, so the
    merged ``idata`` matches by name (the silent-mismatch footgun is avoided by
    keeping one naming scheme). ``param_means`` are the posterior means, used for
    the deterministic structural arithmetic.
    """

    params: Mapping[str, Array]
    param_means: Mapping[str, Array]


@runtime_checkable
class Mechanism(Protocol):
    """The per-node structural equation.

    Implementations carry the node name and its parent names and expose the four
    jobs below. The SCM holds one mechanism per node and a :class:`FitResult` per
    mechanism after fitting.
    """

    name: str
    parents: tuple[str, ...]

    @property
    def capability(self) -> Capability:
        """``INVERTIBLE`` if counterfactuals are exact, else ``PROBABILISTIC``."""
        ...

    def fit(self, data: Dataset, *, draws: int, tune: int, chains: int, seed: int) -> FitResult:
        """Fit the node on its observed parents; return the posterior."""
        ...


def _parent_matrix(data: Dataset, parents: Sequence[str]) -> Array:
    """The ``(n, p)`` observed parent design (codes for discrete parents)."""
    if not parents:
        return np.empty((data.n_rows, 0), dtype=np.float64)
    return np.asarray(data.matrix(list(parents)), dtype=np.float64)


def _flatten(arr: Any) -> Array:
    """Collapse an arviz ``(chain, draw, ...)`` array to ``(S, ...)``."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim <= 1:
        return a.reshape(-1)
    return a.reshape(-1, *a.shape[2:])


# -- Gaussian additive (the continuous default) ------------------------------


@dataclass
class GaussianAdditive:
    """Additive-noise Gaussian mechanism, ``X = b0 + w . PA + U``, ``U ~ N(0, sigma)``.

    The default continuous mechanism: a Bayesian-linear mean over the parents with
    a half-normal prior on the noise scale. Invertible, so its counterfactual is
    the exact residual ``U = X - (b0 + w . PA)`` reused under the action. The mean
    can be overridden by any :class:`~ergodic.inference.Learner`; that path stays
    probabilistic (a non-Bayesian mean carries no parameter posterior), so it
    answers observational and interventional queries and abducts via the residual
    against the fitted mean.
    """

    name: str
    parents: tuple[str, ...]
    coef_prior_sd: float = 5.0
    sigma_prior_sd: float = 2.0

    @property
    def capability(self) -> Capability:
        return Capability.INVERTIBLE

    @property
    def coef_name(self) -> str:
        return f"{self.name}__coef"

    @property
    def intercept_name(self) -> str:
        return f"{self.name}__intercept"

    @property
    def sigma_name(self) -> str:
        return f"{self.name}__sigma"

    def build(self, x: Array, y: Array) -> Any:
        """The per-node PyMC model: ``y ~ Normal(b0 + x . w, sigma)``."""
        import pymc as pm

        _, p = x.shape
        with pm.Model() as model:
            x_data = pm.Data("__parents", x) if p else None
            intercept = pm.Normal(self.intercept_name, 0.0, self.coef_prior_sd)
            mean = intercept
            if p:
                coef = pm.Normal(self.coef_name, 0.0, self.coef_prior_sd, shape=p)
                mean = mean + pm.math.dot(x_data, coef)
            sigma = pm.HalfNormal(self.sigma_name, self.sigma_prior_sd)
            pm.Normal(self.name, mu=mean, sigma=sigma, observed=y)
        return model

    def fit(self, data: Dataset, *, draws: int, tune: int, chains: int, seed: int) -> FitResult:
        import pymc as pm

        x = _parent_matrix(data, self.parents)
        y = np.asarray(data.matrix([self.name]).reshape(-1), dtype=np.float64)
        model = self.build(x, y)
        with model:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=seed,
                progressbar=False,
                compute_convergence_checks=False,
            )
        post = idata.posterior
        params: dict[str, Array] = {self.intercept_name: _flatten(post[self.intercept_name])}
        params[self.sigma_name] = _flatten(post[self.sigma_name])
        if self.parents:
            params[self.coef_name] = _flatten(post[self.coef_name])
        means = {k: np.asarray(v.mean(axis=0), dtype=np.float64) for k, v in params.items()}
        return FitResult(params=params, param_means=means)

    def structural_mean(self, fit: FitResult, parent_values: Array) -> Array:
        """The mean ``b0 + w . PA`` at the posterior mean. ``(n, p)`` in, ``(n,)`` out."""
        intercept = float(np.asarray(fit.param_means[self.intercept_name]))
        pv = np.asarray(parent_values, dtype=np.float64)
        if not self.parents:
            return np.full(pv.shape[0], intercept)
        coef = np.asarray(fit.param_means[self.coef_name], dtype=np.float64)
        return intercept + pv @ coef

    def mean_per_draw(self, fit: FitResult, parent_values: Array) -> Array:
        """The mean per parameter-posterior draw.

        ``parent_values`` is ``(S, n, p)`` (per-draw parent designs, so a
        non-intervened parent carries its own counterfactual posterior). Returns
        ``(S, n)``. With no parents the result is the broadcast intercept draws.
        """
        intercept = np.asarray(fit.params[self.intercept_name], dtype=np.float64)  # (S,)
        pv = np.asarray(parent_values, dtype=np.float64)
        s = intercept.shape[0]
        n = pv.shape[1] if pv.ndim == 3 else pv.shape[0]
        if not self.parents:
            return np.repeat(intercept[:, None], n, axis=1)
        coef = np.asarray(fit.params[self.coef_name], dtype=np.float64)  # (S, p)
        # broadcast an intervened parent's singleton draw axis up to S.
        if pv.shape[0] != s:
            pv = np.broadcast_to(pv, (s, n, pv.shape[2]))
        # (S, n, p) * (S, p) summed over p -> (S, n)
        contrib = np.einsum("snp,sp->sn", pv, coef)
        return np.asarray(intercept[:, None] + contrib, dtype=np.float64)

    def sigma(self, fit: FitResult, *, use_draws: bool) -> Array:
        src = fit.params if use_draws else fit.param_means
        return np.asarray(src[self.sigma_name], dtype=np.float64)

    def abduct(self, fit: FitResult, value: Array, parent_values: Array) -> Array:
        """The exact residual ``U = X - (b0 + w . PA)`` at the posterior mean."""
        mean = self.structural_mean(fit, parent_values)
        return gaussian_residual(np.asarray(value, dtype=np.float64), mean)


# -- Gumbel-max categorical / binary -----------------------------------------


@dataclass
class CategoricalGumbelMax:
    """Categorical mechanism as Gumbel-max over softmax logits.

    The logits are linear in the parents with a per-class slope and intercept,
    fitted by the equivalent :class:`Categorical` likelihood (the Gumbel-max
    marginal). The forward mechanism is ``category = argmax(logits + g)`` with
    ``g`` an iid Gumbel vector, and the counterfactual reuses the abducted ``g``
    under shifted logits, so it is invertible. Identified up to the per-row logit
    gauge, which the argmax washes out.
    """

    name: str
    parents: tuple[str, ...]
    n_categories: int = 0
    coef_prior_sd: float = 3.0

    @property
    def capability(self) -> Capability:
        return Capability.INVERTIBLE

    @property
    def weight_name(self) -> str:
        return f"{self.name}__logit_w"

    @property
    def bias_name(self) -> str:
        return f"{self.name}__logit_b"

    def build(self, x: Array, y_codes: IntArray, n_cat: int) -> Any:
        import pymc as pm

        n, p = x.shape
        with pm.Model() as model:
            bias = pm.Normal(self.bias_name, 0.0, self.coef_prior_sd, shape=n_cat)
            logits = bias[None, :] * np.ones((n, 1))
            if p:
                weight = pm.Normal(self.weight_name, 0.0, self.coef_prior_sd, shape=(p, n_cat))
                logits = logits + pm.math.dot(x, weight)
            pm.Categorical(self.name, logit_p=logits, observed=y_codes)
        return model

    def fit(self, data: Dataset, *, draws: int, tune: int, chains: int, seed: int) -> FitResult:
        import pymc as pm

        x = _parent_matrix(data, self.parents)
        y_codes = np.asarray(data.codes(self.name), dtype=np.int64)
        cats = data.categories(self.name)
        n_cat = self.n_categories or (len(cats) if cats is not None else int(y_codes.max()) + 1)
        object.__setattr__(self, "n_categories", n_cat)
        model = self.build(x, y_codes, n_cat)
        with model:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=0.9,
                random_seed=seed,
                progressbar=False,
                compute_convergence_checks=False,
            )
        post = idata.posterior
        params: dict[str, Array] = {self.bias_name: _flatten(post[self.bias_name])}
        if self.parents:
            params[self.weight_name] = _flatten(post[self.weight_name])
        means = {k: np.asarray(v.mean(axis=0), dtype=np.float64) for k, v in params.items()}
        return FitResult(params=params, param_means=means)

    def logits(self, fit: FitResult, parent_values: Array) -> Array:
        """The ``(n, K)`` class logits at the posterior mean."""
        bias = np.asarray(fit.param_means[self.bias_name], dtype=np.float64)
        pv = np.asarray(parent_values, dtype=np.float64)
        out = np.repeat(bias[None, :], pv.shape[0], axis=0)
        if self.parents:
            weight = np.asarray(fit.param_means[self.weight_name], dtype=np.float64)
            out = out + pv @ weight
        return np.asarray(out, dtype=np.float64)

    def abduct(
        self, fit: FitResult, observed_codes: IntArray, parent_values: Array, *, seed: int
    ) -> Array:
        """Abduct the per-unit Gumbel noise consistent with the observed category."""
        rng = np.random.default_rng(seed)
        return gumbel_max_abduct(self.logits(fit, parent_values), observed_codes, rng)

    def decide(self, logits: Array, gumbel: Array) -> IntArray:
        return gumbel_max_decide(logits, gumbel)


@dataclass
class BinaryGumbelMax(CategoricalGumbelMax):
    """A binary node as a two-category Gumbel-max (the logistic mechanism).

    Specializes :class:`CategoricalGumbelMax` to two categories; the marginal is
    the Bernoulli logistic, and the Gumbel-max counterfactual is the binary
    common-random-numbers flip.
    """

    n_categories: int = 2


# -- schema auto-assignment + override resolution ----------------------------


def observed_parents(graph: MixedGraph, node: str) -> tuple[str, ...]:
    """The node's parents that carry data: directed parents minus the latents.

    A latent parent has no column, so the observed-margin fit cannot condition on
    it; its influence falls into the noise. That is precisely what makes a
    latent-confounded effect non-identified, which :meth:`SCM.ate` surfaces.
    """
    return tuple(sorted(graph.parents(node) - graph.latent))


def auto_mechanism(graph: MixedGraph, data: Dataset, node: str) -> Mechanism:
    """The default mechanism for ``node``, read off the schema.

    Continuous -> :class:`GaussianAdditive`; binary -> :class:`BinaryGumbelMax`;
    categorical or ordinal -> :class:`CategoricalGumbelMax`. The same dispatch
    :func:`~ergodic.inference.estimate_effect` and discovery use, so a node's
    mechanism is consistent with how the rest of the package reads its column.
    """
    parents = observed_parents(graph, node)
    col = data.schema[node]
    if col.is_continuous:
        return GaussianAdditive(name=node, parents=parents)
    if col.is_binary:
        return BinaryGumbelMax(name=node, parents=parents)
    if col.is_discrete:
        cats = data.categories(node)
        n_cat = len(cats) if cats is not None else 0
        return CategoricalGumbelMax(name=node, parents=parents, n_categories=n_cat)
    raise ValueError(
        f"node {node!r} is {col.kind.value}; the MVP fits continuous, binary, and "
        "categorical nodes (datetime and excluded columns have no mechanism)"
    )


def resolve_mechanism(spec: Any, graph: MixedGraph, data: Dataset, node: str) -> Mechanism:
    """Turn a user mechanism spec for ``node`` into a :class:`Mechanism`.

    ``spec`` may be:

    - ``None``: auto-assign by schema (:func:`auto_mechanism`).
    - a built-in name (``"linear"``/``"gaussian"`` for the additive Gaussian).
    - a :class:`Mechanism` instance (used as is, with its name and parents
      filled from the graph if blank).
    """
    parents = observed_parents(graph, node)
    if spec is None:
        return auto_mechanism(graph, data, node)
    if isinstance(spec, str):
        key = spec.lower()
        if key in ("linear", "gaussian", "gaussian_additive", "ols"):
            return GaussianAdditive(name=node, parents=parents)
        if key in ("categorical", "softmax", "gumbel_max"):
            cats = data.categories(node)
            n_cat = len(cats) if cats is not None else 0
            return CategoricalGumbelMax(name=node, parents=parents, n_categories=n_cat)
        if key in ("binary", "logistic", "bernoulli"):
            return BinaryGumbelMax(name=node, parents=parents)
        raise ValueError(
            f"unknown mechanism {spec!r} for node {node!r}; use 'linear', "
            "'categorical', or 'binary', or pass a Mechanism instance"
        )
    if isinstance(spec, Mechanism):
        # fill name/parents from the graph if the user left them blank
        if not getattr(spec, "name", ""):
            object.__setattr__(spec, "name", node)
        if not getattr(spec, "parents", ()):
            object.__setattr__(spec, "parents", parents)
        return spec
    raise TypeError(
        f"mechanism for node {node!r} must be None, a built-in name, or a "
        f"Mechanism instance; got {type(spec)!r}"
    )
