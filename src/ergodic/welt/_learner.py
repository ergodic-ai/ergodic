# SPDX-License-Identifier: Apache-2.0
"""A Welt tabular model in an estimator's nuisance slot.

:class:`WeltLearner` satisfies ergodic's ``Learner`` protocol: ``fit(X, y)``
uploads the training rows and fits a Welt predictor, ``predict(X)`` asks it for
predictions, and a classification learner adds ``predict_proba``. Every fit is a
paid Welt job, and cross-fitting fits once per fold and slot, so the learner and
all its clones share one fit budget (``max_fits``) that stops a run before it
fits more than you allowed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ._client import matrix_rows, using

if TYPE_CHECKING:
    from typing import Self

    import numpy.typing as npt

_TASKS = ("regression", "classification")
_TARGET = "__ergodic_target__"


class _FitBudget:
    """The fit count a learner and its clones share, with the cap they share."""

    def __init__(self, max_fits: int | None) -> None:
        self.max_fits = max_fits
        self.used = 0

    def charge(self, model: str) -> None:
        if self.max_fits is not None and self.used >= self.max_fits:
            raise RuntimeError(
                f"WeltLearner reached max_fits={self.max_fits} paid Welt fits of {model!r} "
                "and stopped before fitting again. Cross-fitting fits once per fold and "
                "slot, and a bootstrapped standard error refits for every resample; use "
                "fewer folds or an estimator with an analytic standard error (dml), or "
                "raise max_fits if you mean to pay for more"
            )
        self.used += 1


def _features(X: Any) -> npt.NDArray[np.float64]:
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"X must be a 2-D matrix; got shape {matrix.shape}")
    if matrix.shape[1] == 0:
        raise ValueError(
            "WeltLearner needs at least one feature column, and this slot has none (an "
            "empty adjustment set). Leave this slot to the default learner"
        )
    if matrix.shape[0] == 0:
        raise ValueError("X has no rows")
    return matrix


def _label(value: Any) -> Any:
    """A JSON-ready class label; an integral float such as ``1.0`` becomes ``1``."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


class WeltLearner:
    """A Welt tabular model as an ergodic learner.

    Use it wherever ergodic takes a learner, for example
    ``estimate_effect(graph, data, "X", "Y", strategy="dml",
    learner=WeltLearner("tabicl-v2"))``. A slot that needs probabilities (a
    propensity) takes a classification learner; pass a slot mapping such as
    ``{"outcome": WeltLearner(), "propensity": WeltLearner(task="classification")}``.

    Each fit uploads the training rows as a new Welt dataset and fits a predictor:
    one paid fit. Each predict is a paid prediction, sent in batches no larger than
    the model's prediction-row limit. DML with ``n_folds=k`` fits ``2 * k`` times.
    The learner and its clones share one budget of ``max_fits`` fits.

    Args:
        model: Any Welt classification or regression model id; Welt's catalogue
            (``welt.Client().models()``) lists them with their row and column limits.
        task: ``"regression"`` or ``"classification"``. Only a classification
            learner has ``predict_proba``.
        client: A ``welt.Client`` to use as is, shared with every clone. ``None``
            opens a short-lived ``welt.Client()`` per request.
        max_fits: The most paid fits this learner and its clones may make together;
            ``None`` removes the cap. The default 10 covers DML with five folds.
        configuration: The Welt model configuration.
        seed: The fit seed Welt records with each predictor.
        timeout: Seconds to wait for each fit or pending prediction. A timeout
            leaves the job running in Welt.

    Attributes:
        predictor_id_: The fitted Welt predictor's id.
        model_version_: The exact model version Welt fitted.
        classes_: The class labels, sorted (classification only).
    """

    def __init__(
        self,
        model: str = "tabicl-v2",
        task: str = "regression",
        *,
        client: Any = None,
        max_fits: int | None = 10,
        configuration: str = "default",
        seed: int = 0,
        timeout: float = 600.0,
    ) -> None:
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a Welt model id such as 'tabicl-v2'")
        if task not in _TASKS:
            raise ValueError(f"task must be 'regression' or 'classification'; got {task!r}")
        if max_fits is not None and (not isinstance(max_fits, int) or max_fits < 1):
            raise ValueError("max_fits must be a positive integer or None")
        self.model = model
        self.task = task
        self.client = client
        self.max_fits = max_fits
        self.configuration = configuration
        self.seed = seed
        self.timeout = timeout
        self._budget = _FitBudget(max_fits)
        self._prediction_rows: int | None = None

    # -- scikit-learn surface -------------------------------------------------

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """The constructor arguments, as scikit-learn's ``get_params`` returns them.

        Args:
            deep: Accepted for scikit-learn compatibility; there are no nested
                estimators.

        Returns:
            A dict of every constructor argument.
        """
        return {
            "model": self.model,
            "task": self.task,
            "client": self.client,
            "max_fits": self.max_fits,
            "configuration": self.configuration,
            "seed": self.seed,
            "timeout": self.timeout,
        }

    def set_params(self, **params: Any) -> Self:
        """Set constructor arguments; a new ``max_fits`` starts a fresh budget.

        Args:
            **params: Constructor arguments by name.

        Returns:
            This learner.

        Raises:
            ValueError: A name is not a constructor argument, or a value is invalid.
        """
        merged = {**self.get_params(), **params}
        unknown = set(params) - set(self.get_params())
        if unknown:
            raise ValueError(f"WeltLearner has no parameter(s) {sorted(unknown)}")
        budget = self._budget
        WeltLearner.__init__(self, **merged)
        if "max_fits" not in params:
            self._budget = budget
        return self

    def __sklearn_clone__(self) -> WeltLearner:
        """An unfitted copy that shares this learner's client and fit budget.

        ergodic clones a learner for every fold, arm and resample. Sharing the
        budget makes ``max_fits`` cap the whole run, and sharing the client keeps
        one connection instead of copying it.

        Returns:
            A new unfitted :class:`WeltLearner` with the same arguments.
        """
        twin = WeltLearner(**self.get_params())
        twin._budget = self._budget
        twin._prediction_rows = self._prediction_rows
        return twin

    @property
    def fits_used(self) -> int:
        """Paid fits made so far by this learner and its clones together."""
        return self._budget.used

    def __repr__(self) -> str:
        return f"WeltLearner(model={self.model!r}, task={self.task!r}, max_fits={self.max_fits})"

    # -- fit and predict ------------------------------------------------------

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64] | None = None,
    ) -> Self:
        """Fit a Welt predictor on these rows: one paid fit.

        Args:
            X: The feature matrix, one row per unit and at least one column.
            y: The target, one value per row: numbers for regression, labels for
                classification.
            sample_weight: Must be ``None``. Welt fits take no row weights, and a
                weight is never silently dropped.

        Returns:
            This learner, fitted.

        Raises:
            ValueError: ``sample_weight`` was given, ``X`` has no columns or rows,
                or ``y`` does not match ``X``. Nothing is uploaded.
            RuntimeError: The shared ``max_fits`` budget is spent.
        """
        if sample_weight is not None:
            raise ValueError(
                "WeltLearner does not take sample_weight: Welt fits have no row weights, "
                "and ergodic never drops a weight silently. Use a native or scikit-learn "
                "learner in this slot"
            )
        matrix = _features(X)
        target = np.asarray(y)
        if target.ndim != 1 or target.shape[0] != matrix.shape[0]:
            raise ValueError("y must be 1-D with one value per row of X")
        if self.task == "regression":
            labels: list[Any] = [float(value) for value in target.astype(np.float64)]
            if not np.all(np.isfinite(labels)):
                raise ValueError("regression targets must be finite")
        else:
            labels = [_label(value) for value in target]
        columns = [f"x{i}" for i in range(matrix.shape[1])]
        rows = [[*row, label] for row, label in zip(matrix_rows(matrix), labels, strict=True)]
        self._budget.charge(self.model)
        for name in ("predictor_id_", "model_version_", "classes_", "_order"):
            self.__dict__.pop(name, None)
        with using(self.client) as client:
            dataset = client.upload(
                columns=[*columns, _TARGET], rows=rows, name="ergodic fit", target=_TARGET
            )
            job = client.submit_fit(
                dataset["id"],
                model=self.model,
                task=self.task,
                configuration=self.configuration,
                seed=self.seed,
            )
            predictor = job.result(timeout=self.timeout)
            if self._prediction_rows is None:
                self._prediction_rows = self._limit(client)
        if predictor.get("task") != self.task:
            raise ValueError("Welt fitted a predictor for a different task")
        self.predictor_id_: str = predictor["id"]
        self.model_version_: str = predictor["model_version"]
        self._n_features = matrix.shape[1]
        if self.task == "classification":
            classes = np.asarray(predictor["classes"])
            self._order = np.argsort(classes, kind="stable")
            self.classes_ = classes[self._order]
        return self

    def _limit(self, client: Any) -> int | None:
        """The model's prediction-row limit from Welt's catalogue, if it states one."""
        for profile in client.model(self.model).get("task_profiles", []):
            if profile.get("task") == self.task:
                limit = (profile.get("limits") or {}).get("max_prediction_rows")
                return int(limit) if limit else None
        return None

    def _predict(self, X: npt.NDArray[np.float64], probabilities: bool) -> list[dict[str, Any]]:
        if not hasattr(self, "predictor_id_"):
            raise RuntimeError("WeltLearner is not fitted; call fit first")
        matrix = _features(X)
        if matrix.shape[1] != self._n_features:
            raise ValueError(f"X has {matrix.shape[1]} columns; the fit had {self._n_features}")
        columns = [f"x{i}" for i in range(matrix.shape[1])]
        rows = matrix_rows(matrix)
        size = self._prediction_rows or len(rows)
        from welt import PredictionPendingError

        answers: list[dict[str, Any]] = []
        with using(self.client) as client:
            for start in range(0, len(rows), size):
                batch = rows[start : start + size]
                try:
                    answer = client.predict(
                        self.predictor_id_,
                        columns=columns,
                        rows=batch,
                        probabilities=probabilities,
                    )
                except PredictionPendingError as pending:
                    if pending.job_id is None:
                        raise
                    answer = client.job(pending.job_id).result(timeout=self.timeout)
                answers.append(answer)
        return answers

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[Any]:
        """Predict with the fitted Welt predictor: one paid prediction per batch.

        Args:
            X: Rows with the same columns as the fit.

        Returns:
            Numbers in the target's units for regression, class labels for
            classification.
        """
        answers = self._predict(X, probabilities=False)
        values = [value for answer in answers for value in answer["predictions"]]
        if self.task == "regression":
            return np.asarray(values, dtype=np.float64)
        return np.asarray(values)

    def _predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        answers = self._predict(X, probabilities=True)
        blocks = []
        for answer in answers:
            if answer.get("probabilities") is None:
                raise ValueError(f"Welt model {self.model!r} returned no class probabilities")
            blocks.append(np.asarray(answer["probabilities"], dtype=np.float64))
        return np.vstack(blocks)[:, self._order]

    @property
    def predict_proba(self) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Class probabilities, one column per class in ``classes_`` order.

        Only a classification learner has this; ergodic reads its presence to tell a
        classifier from a regressor.

        Raises:
            AttributeError: The learner's task is regression.
        """
        if self.task != "classification":
            raise AttributeError("predict_proba exists only for task='classification'")
        return self._predict_proba
