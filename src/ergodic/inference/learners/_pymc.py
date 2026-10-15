# SPDX-License-Identifier: Apache-2.0
"""Bayesian learners backed by a user-defined PyMC model.

PyMC is an optional dependency. Importing this module never imports pymc; the
``import pymc`` lives inside the methods, so the zero-dependency install keeps
working and only a learner that is actually fitted needs pymc present.

Subclass :class:`PyMCLearner` (regression) or :class:`PyMCClassifier`
(classification) and implement ``build``. The convention, so the base can serve
predictions at new data:

- put the design in a ``pm.Data`` container named ``"X"``,
- name the mean node ``"mu"`` (a regression mean) or ``"p"`` (a class-1
  probability), shaped by the data container, and
- attach the likelihood with ``observed=y``.

The base owns sampling, the posterior-predictive of the mean node (the smooth
mean, not the noisy outcome), and cloning through ``get_params`` / ``set_params``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

_SAMPLER_KEYS = ("draws", "tune", "chains", "target_accept", "random_state", "progressbar")


class PyMCLearner:
    """A Bayesian regression learner over a PyMC model.

    Implement :meth:`build` to return a ``pymc.Model``. ``predict`` is the
    posterior predictive mean and ``predict_draws`` the ``(S, n)`` posterior of the
    mean node, so the learner drops into a regression slot and also drives the
    posterior path.
    """

    _mean_node = "mu"
    _data_name = "X"
    #: A user-written PyMC model has no generic place for row weights, so a
    #: weighted slot (the R-learner's final stage) refuses this learner up front.
    supports_sample_weight: ClassVar[bool] = False

    def __init__(
        self,
        *,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        target_accept: float = 0.9,
        random_state: int | None = None,
        progressbar: bool = False,
        **build_kwargs: Any,
    ) -> None:
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.random_state = random_state
        self.progressbar = progressbar
        self.build_kwargs = build_kwargs
        self._model: Any = None
        self._idata: Any = None

    def build(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> Any:
        """Return a ``pymc.Model``. Subclasses implement this."""
        raise NotImplementedError("subclass PyMCLearner and implement build")

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64] | None = None,
    ) -> PyMCLearner:
        """Build the model with :meth:`build` and sample its posterior with NUTS.

        Raises ``TypeError`` when ``sample_weight`` is given: a PyMC model has no
        generic place for row weights (``supports_sample_weight`` is ``False``). Needs pymc.

        Returns:
            The fitted learner itself.
        """
        import pymc as pm

        if sample_weight is not None:
            raise TypeError(
                f"{type(self).__name__}.fit takes no sample_weight: a PyMC model has no "
                "generic place for row weights. Write them into build() yourself, or "
                "use a learner that accepts them"
            )
        x = np.asarray(X, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64).reshape(-1)
        self._model = self.build(x, target)
        with self._model:
            self._idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                target_accept=self.target_accept,
                random_seed=self.random_state,
                progressbar=self.progressbar,
            )
        return self

    def _mean_draws(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        import pymc as pm

        if self._idata is None:
            raise RuntimeError(f"{type(self).__name__}.predict called before fit")
        x = np.asarray(X, dtype=np.float64)
        with self._model:
            pm.set_data({self._data_name: x})
            posterior = pm.sample_posterior_predictive(
                self._idata,
                var_names=[self._mean_node],
                progressbar=self.progressbar,
                random_seed=self.random_state,
            )
        arr = np.asarray(
            posterior.posterior_predictive[self._mean_node].to_numpy(), dtype=np.float64
        )
        return arr.reshape(-1, arr.shape[-1])

    def predict_draws(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return ``(S, n)`` posterior draws of the mean node at the rows of ``X``.

        ``S`` is ``chains * draws``. For a :class:`PyMCClassifier` the mean node is
        ``"p"``, so these are probability draws.

        Raises:
            RuntimeError: The learner has not been fitted.
        """
        return self._mean_draws(X)

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the posterior mean of the mean node at the rows of ``X``."""
        return np.asarray(self._mean_draws(X).mean(axis=0), dtype=np.float64)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the sampler settings and the extra ``build`` keyword arguments."""
        params = {key: getattr(self, key) for key in _SAMPLER_KEYS}
        params.update(self.build_kwargs)
        return params

    def set_params(self, **params: Any) -> PyMCLearner:
        """Set sampler settings or ``build`` keyword arguments by name; return the learner."""
        for key, value in params.items():
            if key in _SAMPLER_KEYS:
                setattr(self, key, value)
            else:
                self.build_kwargs[key] = value
        return self


class PyMCClassifier(PyMCLearner):
    """A Bayesian classification learner over a PyMC model.

    Name the probability node ``"p"`` (the per-row P(class=1)). Exposes
    ``predict_proba`` and ``predict_proba_draws``, so it fills a propensity or a
    mediator slot and drives the posterior path there.
    """

    _mean_node = "p"

    def predict_proba_draws(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return ``(S, n)`` posterior draws of P(y = 1) at the rows of ``X``.

        Raises:
            RuntimeError: The learner has not been fitted.
        """
        return self._mean_draws(X)

    def predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return class probabilities as an ``(n, 2)`` array from the posterior mean of P(y = 1)."""
        p = self._mean_draws(X).mean(axis=0)
        return np.asarray(np.column_stack([1.0 - p, p]), dtype=np.float64)

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the 0/1 label, 1 where the posterior mean of P(y = 1) is at least 0.5."""
        return np.asarray(self.predict_proba(X)[:, 1] >= 0.5, dtype=np.float64)
