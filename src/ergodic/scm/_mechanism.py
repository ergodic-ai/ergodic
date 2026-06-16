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

Overrides in the MVP are a built-in name (``"linear"``/``"gaussian"``/``"ols"`` for
the additive Gaussian, ``"categorical"``/``"softmax"`` for the Gumbel-max
categorical, ``"binary"``/``"logistic"`` for the binary Gumbel-max) or a
:class:`Mechanism` instance used as is. Richer mean overrides -- a
:class:`~ergodic.inference.Learner` instance as the mean, GP/BART built-ins, a
:class:`~ergodic.inference.PyMCLearner` subclass, and a raw callable that adds the
node's RVs to a model -- are planned post-MVP. pymc is imported only inside the
methods, never at module top, so the bare install never imports it (the
``PyMCLearner`` precedent).

Mechanisms are frozen value objects, matching the repo convention: filling a
node's name/parents from the graph returns a new instance rather than mutating the
spec, so one mechanism can be reused across SCMs without cross-corruption.
"""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
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

    ``n_categories`` carries a categorical node's fit-resolved class count (0 for a
    continuous node). It lives here, not on the mechanism, so a mechanism stays an
    immutable spec and ``fit`` does not mutate ``self``; query-time consumers read
    the resolved count off the fit.
    """

    params: Mapping[str, Array]
    param_means: Mapping[str, Array]
    n_categories: int = 0


@runtime_checkable
class Mechanism(Protocol):
    """The per-node structural equation.

    Implementations carry the node name and its parent names and expose the four
    jobs below. The SCM holds one mechanism per node and a :class:`FitResult` per
    mechanism after fitting. ``name`` and ``parents`` are read-only: mechanisms are
    frozen value objects, so filling a blank node name returns a new instance
    (:func:`resolve_mechanism`) rather than mutating in place.
    """

    @property
    def name(self) -> str:
        """The node this mechanism is the structural equation for."""
        ...

    @property
    def parents(self) -> tuple[str, ...]:
        """The node's observed parents, in the mechanism's design order."""
        ...

    @property
    def capability(self) -> Capability:
        """``INVERTIBLE`` if counterfactuals are exact, else ``PROBABILISTIC``."""
        ...

    def fit(self, data: Dataset, *, draws: int, tune: int, chains: int, seed: int) -> FitResult:
        """Fit the node on its observed parents; return the posterior."""
        ...

    def generative_rv(self, nodes: Mapping[str, Any], n: int) -> Any:
        """Add this node's RV to the active generative PyMC model, return the RV.

        ``nodes`` maps each already-built parent name to its RV; the structural
        equation reads the parents from it. Names every parameter RV with the same
        scheme :meth:`fit` used, so the merged posterior matches by name. Called
        inside a ``pm.Model()`` context by the compose helper.
        """
        ...


def _parent_matrix(data: Dataset, parents: Sequence[str]) -> Array:
    """The ``(n, p)`` observed parent design (codes for discrete parents).

    One column per parent: a continuous parent's value, a discrete parent's
    integer code. The Gumbel-max/Gaussian mean turns this into the encoded design
    via :func:`encode_parent_design` (one-hot for a discrete parent); this is the
    raw, pre-encoding matrix both views start from.
    """
    if not parents:
        return np.empty((data.n_rows, 0), dtype=np.float64)
    return np.asarray(data.matrix(list(parents)), dtype=np.float64)


def parent_cardinalities(data: Dataset, parents: Sequence[str]) -> tuple[int | None, ...]:
    """The per-parent category count: ``None`` for a continuous/datetime parent, ``k``
    for a discrete one. Drives the one-hot encoding so the design is reproducible
    from the mechanism spec alone (no dataset needed at query time)."""
    cards: list[int | None] = []
    for p in parents:
        col = data.schema[p]
        cards.append(col.cardinality if col.is_discrete else None)
    return tuple(cards)


def encoded_width(parent_cards: Sequence[int | None]) -> int:
    """The encoded-design column count: 1 per continuous parent, ``k-1`` per discrete
    parent (drop-first dummy, the :func:`~ergodic.inference._estimator._covariate_matrix`
    convention). A binary parent contributes a single 0/1 column."""
    width = 0
    for card in parent_cards:
        width += 1 if card is None else max(card - 1, 0)
    return width


def encode_parent_design(raw: Array, parent_cards: Sequence[int | None]) -> Array:
    """Encode the ``(n, p)`` raw parent matrix into the ``(n, P)`` model design.

    A continuous parent's column passes through; a discrete parent's integer-code
    column expands to a drop-first one-hot block (dummies for codes ``1..k-1``).
    This is the single encoding convention shared by every view -- the per-node
    fit, the generative model, and the counterfactual arithmetic -- so a coefficient
    means the same thing everywhere. Matches the estimation pillar's
    ``_covariate_matrix`` exactly (continuous as is, discrete one-hot drop-first).
    """
    arr = np.asarray(raw, dtype=np.float64)
    n = arr.shape[0]
    if not parent_cards:
        return np.empty((n, 0), dtype=np.float64)
    blocks: list[Array] = []
    for j, card in enumerate(parent_cards):
        col = arr[:, j]
        if card is None:
            blocks.append(col[:, None])
        elif card > 1:
            codes = col.astype(np.int64)
            onehot = (codes[:, None] == np.arange(card)[None, :]).astype(np.float64)
            blocks.append(onehot[:, 1:])  # drop first (the reference category)
    if not blocks:
        return np.empty((n, 0), dtype=np.float64)
    return np.asarray(np.hstack(blocks), dtype=np.float64)


def encode_parents_pt(parent_rvs: Sequence[Any], parent_cards: Sequence[int | None]) -> Any:
    """The pytensor twin of :func:`encode_parent_design`, for the generative model.

    ``parent_rvs`` is the list of parent node RVs (each shape ``(n,)``) in design
    order; returns a ``(n, P)`` pytensor design with the *identical* drop-first
    one-hot layout. Single-sourcing the layout (this and the numpy encoder run off
    the same ``parent_cards``) is what keeps the fit view and the generative view in
    lockstep, so ``sample_posterior_predictive`` never matches a coefficient to a
    differently shaped design.
    """
    # pm.math re-exports the pytensor ops and is the typed-Any surface the rest of
    # the SCM uses (pytensor.tensor's own ops trip mypy's no-untyped-call); this
    # function only runs inside a generative-model build, so pymc is already in play.
    import pymc as pm

    blocks: list[Any] = []
    for rv, card in zip(parent_rvs, parent_cards, strict=True):
        if card is None:
            blocks.append(rv[:, None])
        elif card > 1:
            dummies = [pm.math.eq(rv, k) for k in range(1, card)]  # drop-first one-hot
            blocks.append(pm.math.stack(dummies, axis=1).astype("float64"))
    if not blocks:
        # an empty (n, 0) design; n is inferred from the first parent if any exist.
        n = parent_rvs[0].shape[0] if parent_rvs else 0
        return pm.math.zeros((n, 0))
    return pm.math.concatenate(blocks, axis=1)


def _flatten(arr: Any) -> Array:
    """Collapse an arviz ``(chain, draw, ...)`` array to ``(S, ...)``."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim <= 1:
        return a.reshape(-1)
    return a.reshape(-1, *a.shape[2:])


# -- Gaussian additive (the continuous default) ------------------------------


@dataclass(frozen=True)
class GaussianAdditive:
    """Additive-noise Gaussian mechanism, ``X = b0 + w . PA + U``, ``U ~ N(0, sigma)``.

    The default continuous mechanism: a Bayesian-linear mean over the parents with
    a half-normal prior on the noise scale. Invertible, so its counterfactual is
    the exact residual ``U = X - (b0 + w . PA)`` reused under the action. A
    :class:`~ergodic.inference.Learner`-as-mean override (carrying no parameter
    posterior, abducting via the residual against the fitted mean) is planned
    post-MVP; the MVP fits the Bayesian-linear mean.
    """

    name: str
    parents: tuple[str, ...]
    parent_cards: tuple[int | None, ...] = ()
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

    @property
    def _design_width(self) -> int:
        """The encoded coefficient count, ``len(parents)`` when every parent is
        continuous (no ``parent_cards``), else the one-hot-expanded width."""
        if not self.parents:
            return 0
        if not self.parent_cards:
            return len(self.parents)
        return encoded_width(self.parent_cards)

    def build(self, x: Array, y: Array) -> Any:
        """The per-node PyMC model: ``y ~ Normal(b0 + x_encoded . w, sigma)``.

        ``x`` is the raw parent matrix (codes for discrete parents); it is encoded
        to the one-hot design here, the same encoding the generative model uses.
        """
        import pymc as pm

        x_enc = encode_parent_design(x, self.parent_cards) if self.parents else x
        _, p = x_enc.shape
        with pm.Model() as model:
            x_data = pm.Data("__parents", x_enc) if p else None
            intercept = pm.Normal(self.intercept_name, 0.0, self.coef_prior_sd)
            mean = intercept
            if p:
                coef = pm.Normal(self.coef_name, 0.0, self.coef_prior_sd, shape=p)
                mean = mean + pm.math.dot(x_data, coef)
            sigma = pm.HalfNormal(self.sigma_name, self.sigma_prior_sd)
            pm.Normal(self.name, mu=mean, sigma=sigma, observed=y)
        return model

    def generative_rv(self, nodes: Mapping[str, Any], n: int) -> Any:
        """Add this node's RV to the active generative model: ``Normal(b0 + PA . w, sigma)``.

        ``nodes`` maps each (already-built) parent name to its RV. Parents are
        encoded by the shared pytensor encoder, so the design matches the fit view
        column for column; the RV names match the merged posterior exactly, so
        ``sample_posterior_predictive`` reads the fitted draws.
        """
        import pymc as pm
        import pytensor.tensor as pt

        intercept = pm.Normal(self.intercept_name, 0.0, self.coef_prior_sd)
        mean: Any = intercept * pt.ones((n,))
        if self.parents:
            design = encode_parents_pt([nodes[p] for p in self.parents], self.parent_cards)
            coef = pm.Normal(self.coef_name, 0.0, self.coef_prior_sd, shape=self._design_width)
            mean = mean + pm.math.dot(design, coef)
        sigma = pm.HalfNormal(self.sigma_name, self.sigma_prior_sd)
        return pm.Normal(self.name, mu=mean, sigma=sigma, shape=(n,))

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
        """The mean ``b0 + w . encode(PA)`` at the posterior mean.

        ``parent_values`` is the ``(n, p)`` *raw* parent matrix (codes for discrete
        parents); it is one-hot encoded internally, so the coefficients line up with
        the fit. ``(n, p)`` in, ``(n,)`` out.
        """
        intercept = float(np.asarray(fit.param_means[self.intercept_name]))
        pv = np.asarray(parent_values, dtype=np.float64)
        if not self.parents:
            return np.full(pv.shape[0], intercept)
        design = encode_parent_design(pv, self.parent_cards)
        coef = np.asarray(fit.param_means[self.coef_name], dtype=np.float64)
        return intercept + design @ coef

    def mean_per_draw(self, fit: FitResult, parent_values: Array) -> Array:
        """The mean per parameter-posterior draw.

        ``parent_values`` is the ``(S, n, p)`` *raw* per-draw parent matrix (a
        non-intervened parent carries its own counterfactual posterior; codes for a
        discrete parent). It is one-hot encoded per draw, so the encoded design
        matches the ``(S, P)`` coefficient draws. Returns ``(S, n)``. With no parents
        the result is the broadcast intercept draws.
        """
        intercept = np.asarray(fit.params[self.intercept_name], dtype=np.float64)  # (S,)
        pv = np.asarray(parent_values, dtype=np.float64)
        s = intercept.shape[0]
        n = pv.shape[1] if pv.ndim == 3 else pv.shape[0]
        if not self.parents:
            return np.repeat(intercept[:, None], n, axis=1)
        coef = np.asarray(fit.params[self.coef_name], dtype=np.float64)  # (S, P)
        # broadcast an intervened parent's singleton draw axis up to S, then encode
        # each draw's raw (n, p) slice to its (n, P) one-hot design.
        if pv.shape[0] != s:
            pv = np.broadcast_to(pv, (s, n, pv.shape[2]))
        design = np.stack(
            [encode_parent_design(pv[d], self.parent_cards) for d in range(s)], axis=0
        )
        # (S, n, P) * (S, P) summed over P -> (S, n)
        contrib = np.einsum("snp,sp->sn", design, coef)
        return np.asarray(intercept[:, None] + contrib, dtype=np.float64)

    def sigma(self, fit: FitResult, *, use_draws: bool) -> Array:
        src = fit.params if use_draws else fit.param_means
        return np.asarray(src[self.sigma_name], dtype=np.float64)

    def abduct(self, fit: FitResult, value: Array, parent_values: Array) -> Array:
        """The exact residual ``U = X - (b0 + w . PA)`` at the posterior mean."""
        mean = self.structural_mean(fit, parent_values)
        return gaussian_residual(np.asarray(value, dtype=np.float64), mean)


# -- Gumbel-max categorical / binary -----------------------------------------


@dataclass(frozen=True)
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
    parent_cards: tuple[int | None, ...] = ()
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

    @property
    def _design_width(self) -> int:
        """The encoded weight rows: ``len(parents)`` for all-continuous parents (no
        ``parent_cards``), else the one-hot-expanded width."""
        if not self.parents:
            return 0
        if not self.parent_cards:
            return len(self.parents)
        return encoded_width(self.parent_cards)

    def build(self, x: Array, y_codes: IntArray, n_cat: int) -> Any:
        import pymc as pm

        x_enc = encode_parent_design(x, self.parent_cards) if self.parents else x
        n, p = x_enc.shape
        with pm.Model() as model:
            bias = pm.Normal(self.bias_name, 0.0, self.coef_prior_sd, shape=n_cat)
            logits = bias[None, :] * np.ones((n, 1))
            if p:
                weight = pm.Normal(self.weight_name, 0.0, self.coef_prior_sd, shape=(p, n_cat))
                logits = logits + pm.math.dot(x_enc, weight)
            pm.Categorical(self.name, logit_p=logits, observed=y_codes)
        return model

    def generative_rv(self, nodes: Mapping[str, Any], n: int) -> Any:
        """Add this node's categorical RV to the active generative model.

        ``logits = bias + encode(PA) . weight`` (same per-class linear form and same
        one-hot parent encoding the fit uses), then ``pm.Categorical`` samples the
        node forward (the Gumbel-max marginal). The RV names match the merged
        posterior, so ``sample_posterior_predictive`` reads the fitted logits. The
        node's resolved class count must be passed in (it lives on the fit, not the
        spec).
        """
        import pymc as pm
        import pytensor.tensor as pt

        if self.n_categories <= 0:
            raise ValueError(
                f"categorical node {self.name!r} has an unresolved class count; the "
                "generative model needs n_categories (read it off the fit)"
            )
        bias = pm.Normal(self.bias_name, 0.0, self.coef_prior_sd, shape=self.n_categories)
        logits: Any = bias[None, :] * pt.ones((n, 1))
        if self.parents:
            design = encode_parents_pt([nodes[p] for p in self.parents], self.parent_cards)
            weight = pm.Normal(
                self.weight_name,
                0.0,
                self.coef_prior_sd,
                shape=(self._design_width, self.n_categories),
            )
            logits = logits + pm.math.dot(design, weight)
        return pm.Categorical(self.name, logit_p=logits, shape=(n,))

    def fit(self, data: Dataset, *, draws: int, tune: int, chains: int, seed: int) -> FitResult:
        import pymc as pm

        x = _parent_matrix(data, self.parents)
        y_codes = np.asarray(data.codes(self.name), dtype=np.int64)
        cats = data.categories(self.name)
        n_cat = self.n_categories or (len(cats) if cats is not None else int(y_codes.max()) + 1)
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
        return FitResult(params=params, param_means=means, n_categories=n_cat)

    def logits(self, fit: FitResult, parent_values: Array) -> Array:
        """The ``(n, K)`` class logits at the posterior mean.

        ``parent_values`` is the ``(n, p)`` *raw* parent matrix (codes for discrete
        parents); it is one-hot encoded internally so the weight rows line up.
        """
        bias = np.asarray(fit.param_means[self.bias_name], dtype=np.float64)
        pv = np.asarray(parent_values, dtype=np.float64)
        out = np.repeat(bias[None, :], pv.shape[0], axis=0)
        if self.parents:
            design = encode_parent_design(pv, self.parent_cards)
            weight = np.asarray(fit.param_means[self.weight_name], dtype=np.float64)
            out = out + design @ weight
        return np.asarray(out, dtype=np.float64)

    def logits_per_draw(self, fit: FitResult, parent_values: Array) -> Array:
        """The ``(S, n, K)`` class logits per parameter-posterior draw.

        The per-draw twin of :meth:`logits`: ``bias`` is ``(S, K)`` and ``weight``
        ``(S, P, K)``, so each draw's logits read ``bias[d] + encode(PA)[d] . w[d]``.
        ``parent_values`` is the ``(S, n, p)`` *raw* per-draw parent matrix (an
        intervened/categorical parent's singleton draw axis broadcasts up to ``S``,
        and is one-hot encoded per draw so the design matches the ``(S, P, K)`` weight
        draws). With no parents the result is the broadcast bias draws. Used by the
        counterfactual to spread a categorical ancestor's decision across draws, so
        its logit-parameter posterior enters the counterfactual interval (the
        posterior-mean collapse it replaces understated the width).
        """
        bias = np.asarray(fit.params[self.bias_name], dtype=np.float64)  # (S, K)
        s, k = bias.shape
        pv = np.asarray(parent_values, dtype=np.float64)
        n = pv.shape[1] if pv.ndim == 3 else pv.shape[0]
        if not self.parents:
            return np.broadcast_to(bias[:, None, :], (s, n, k)).copy()
        weight = np.asarray(fit.params[self.weight_name], dtype=np.float64)  # (S, P, K)
        if pv.ndim == 2:
            pv = pv[None, :, :]
        if pv.shape[0] != s:
            pv = np.broadcast_to(pv, (s, n, pv.shape[2]))
        design = np.stack(
            [encode_parent_design(pv[d], self.parent_cards) for d in range(s)], axis=0
        )  # (S, n, P)
        # (S, n, P) @ (S, P, K) batched over the draw axis -> (S, n, K)
        contrib = np.einsum("snp,spk->snk", design, weight)
        return np.asarray(bias[:, None, :] + contrib, dtype=np.float64)

    def abduct(
        self, fit: FitResult, observed_codes: IntArray, parent_values: Array, *, seed: int
    ) -> Array:
        """Abduct the per-unit Gumbel noise consistent with the observed category."""
        rng = np.random.default_rng(seed)
        return gumbel_max_abduct(self.logits(fit, parent_values), observed_codes, rng)

    def decide(self, logits: Array, gumbel: Array) -> IntArray:
        return gumbel_max_decide(logits, gumbel)


@dataclass(frozen=True)
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
    cards = parent_cardinalities(data, parents)
    col = data.schema[node]
    if col.is_continuous:
        return GaussianAdditive(name=node, parents=parents, parent_cards=cards)
    if col.is_binary:
        return BinaryGumbelMax(name=node, parents=parents, parent_cards=cards)
    if col.is_discrete:
        cats = data.categories(node)
        n_cat = len(cats) if cats is not None else 0
        return CategoricalGumbelMax(
            name=node, parents=parents, parent_cards=cards, n_categories=n_cat
        )
    raise ValueError(
        f"node {node!r} is {col.kind.value}; the MVP fits continuous, binary, and "
        "categorical nodes (datetime and excluded columns have no mechanism)"
    )


def resolve_mechanism(spec: Any, graph: MixedGraph, data: Dataset, node: str) -> Mechanism:
    """Turn a user mechanism spec for ``node`` into a :class:`Mechanism`.

    ``spec`` may be:

    - ``None``: auto-assign by schema (:func:`auto_mechanism`).
    - a built-in name (``"linear"``/``"gaussian"``/``"ols"`` for the additive
      Gaussian, ``"categorical"``/``"softmax"`` for the Gumbel-max categorical,
      ``"binary"``/``"logistic"`` for the binary Gumbel-max).
    - a :class:`Mechanism` instance. Mechanisms are frozen, so this returns a copy
      with its name and parents filled from the graph when blank (via
      :func:`dataclasses.replace`); the caller's instance is never mutated, so one
      mechanism can be reused across SCMs.
    """
    parents = observed_parents(graph, node)
    cards = parent_cardinalities(data, parents)
    if spec is None:
        return auto_mechanism(graph, data, node)
    if isinstance(spec, str):
        key = spec.lower()
        if key in ("linear", "gaussian", "gaussian_additive", "ols"):
            return GaussianAdditive(name=node, parents=parents, parent_cards=cards)
        if key in ("categorical", "softmax", "gumbel_max"):
            cats = data.categories(node)
            n_cat = len(cats) if cats is not None else 0
            return CategoricalGumbelMax(
                name=node, parents=parents, parent_cards=cards, n_categories=n_cat
            )
        if key in ("binary", "logistic", "bernoulli"):
            return BinaryGumbelMax(name=node, parents=parents, parent_cards=cards)
        raise ValueError(
            f"unknown mechanism {spec!r} for node {node!r}; use 'linear', "
            "'categorical', or 'binary', or pass a Mechanism instance"
        )
    if isinstance(spec, Mechanism):
        # fill name/parents from the graph when blank, returning a NEW instance:
        # mechanisms are frozen value objects, so reusing one across SCMs must not
        # cross-corrupt it. dataclasses.replace is the immutable copy-with-changes.
        # Filling parents also fills parent_cards, so the one-hot encoding stays
        # consistent for an instance mechanism left blank.
        changes: dict[str, Any] = {}
        if not getattr(spec, "name", ""):
            changes["name"] = node
        if not getattr(spec, "parents", ()):
            changes["parents"] = parents
            if not getattr(spec, "parent_cards", ()):
                changes["parent_cards"] = cards
        if not changes:
            return spec
        if is_dataclass(spec) and not isinstance(spec, type):
            return replace(spec, **changes)
        raise TypeError(
            f"mechanism for node {node!r} is missing its name/parents and is not a "
            "dataclass, so it cannot be filled immutably; pass a dataclass Mechanism "
            "(GaussianAdditive/CategoricalGumbelMax/BinaryGumbelMax) with name and "
            "parents set, or leave name/parents blank on a dataclass mechanism"
        )
    raise TypeError(
        f"mechanism for node {node!r} must be None, a built-in name, or a "
        f"Mechanism instance; got {type(spec)!r}"
    )
