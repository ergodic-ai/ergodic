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
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ._client import matrix_rows, using

if TYPE_CHECKING:
    from typing import Self

    import numpy.typing as npt

_TASKS = ("regression", "classification")
_TARGET = "__ergodic_target__"


class _FitBudget:
    """What a learner and its clones share: the fit cap and count, and catalogue limits."""

    def __init__(self, max_fits: int | None) -> None:
        self.max_fits = max_fits
        self.used = 0
        self.prediction_rows: dict[tuple[str, str], int | None] = {}

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
    one paid fit. Both stay in your Welt workspace afterwards; nothing here deletes
    them (``welt.Client().predictors()`` and ``.datasets()`` list them). Each predict
    is a paid prediction, sent in batches no larger than the model's prediction-row
    limit. DML with ``n_folds=k`` fits ``2 * k`` times.

    A slot with no feature columns (an empty adjustment set) has nothing for a model
    to learn: its conditional mean is the training mean, or the class frequencies for
    classification. The learner fits that locally, with no upload, no paid fit and no
    charge to ``max_fits``, and ``predictor_id_`` is ``None``.

    The learner and its clones share one budget of ``max_fits`` fits. It caps paid
    fits only, not predictions: each fit is followed by one prediction call per
    batch of held-out rows (DML on 8,000 rows with five folds and a 100-row limit
    makes about 160). Fits take no row weights, so ``supports_sample_weight`` is
    ``False`` and ergodic refuses this learner in a weighted slot (the R-learner's
    ``final``) before fitting anything.

    Args:
        model: Any Welt classification or regression model id; none is checked
            here. The default follows what Welt serves: ``"tabicl-v2"`` is available
            today, while ``"tabpfn-3.5-fast"`` is still planned. See which models run
            now, with their row, column and prediction limits, with
            ``[(m["id"], p["task"], p["limits"]) for m in welt.Client().models() for p
            in m["task_profiles"] if p["execution_available"]]``, ``welt models list`` on
            the command line, or the Models page of the Welt SDK docs.
        task: ``"regression"`` or ``"classification"``. Only a classification
            learner has ``predict_proba``.
        client: A ``welt.Client`` to use as is, shared with every clone. ``None``
            opens a short-lived ``welt.Client()`` per request.
        max_fits: The most paid fits this learner and its clones may make together;
            ``None`` removes the cap. The default 10 covers DML with five folds.
            Predictions are not counted.
        configuration: The Welt model configuration.
        seed: The fit seed Welt records with each predictor.
        timeout: Seconds to wait for each fit or pending prediction. A timeout
            leaves the job running in Welt.

    Attributes:
        predictor_id_: The fitted Welt predictor's id; ``None`` after a no-feature
            (local) fit.
        model_version_: The exact model version Welt fitted; ``None`` after a
            no-feature fit.
        classes_: The class labels, sorted (classification only).
        supports_sample_weight: Always ``False``; read by ergodic's weighted slots.
    """

    supports_sample_weight: ClassVar[bool] = False

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
            X: The feature matrix, one row per unit. With no columns the fit is the
                local training mean (or class frequencies) and costs nothing.
            y: The target, one value per row: numbers for regression, labels for
                classification.
            sample_weight: Must be ``None``. Welt fits take no row weights, and a
                weight is never silently dropped.

        Returns:
            This learner, fitted.

        Raises:
            ValueError: ``sample_weight`` was given, ``X`` has no rows, or ``y``
                does not match ``X``. Nothing is uploaded.
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
        for name in ("predictor_id_", "model_version_", "classes_", "_order", "_local"):
            self.__dict__.pop(name, None)
        if matrix.shape[1] == 0:
            return self._fit_local(labels)
        columns = [f"x{i}" for i in range(matrix.shape[1])]
        rows = [[*row, label] for row, label in zip(matrix_rows(matrix), labels, strict=True)]
        with using(self.client) as client:
            # The free catalogue read comes first, so a failed lookup costs no fit
            # and cannot strand a paid predictor.
            key = (self.model, self.task)
            if key not in self._budget.prediction_rows:
                self._budget.prediction_rows[key] = self._limit(client)
            self._budget.charge(self.model)
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
        if predictor.get("task") != self.task:
            raise ValueError("Welt fitted a predictor for a different task")
        self.predictor_id_: str | None = predictor["id"]
        self.model_version_: str | None = predictor["model_version"]
        self._n_features = matrix.shape[1]
        if self.task == "classification":
            classes = np.asarray(predictor["classes"])
            self._order = np.argsort(classes, kind="stable")
            self.classes_ = classes[self._order]
        return self

    def _fit_local(self, labels: list[Any]) -> Self:
        """The no-feature fit: the training mean, or the class frequencies, kept here."""
        self.predictor_id_ = None
        self.model_version_ = None
        self._n_features = 0
        if self.task == "regression":
            self._local: npt.NDArray[Any] = np.asarray([np.mean(labels)], dtype=np.float64)
            return self
        classes, counts = np.unique(np.asarray(labels), return_counts=True)
        self.classes_ = classes
        self._order = np.arange(len(classes))
        self._local = counts / counts.sum()
        return self

    def _limit(self, client: Any) -> int | None:
        """The model's prediction-row limit from Welt's catalogue, if it states one."""
        for profile in client.model(self.model).get("task_profiles", []):
            if profile.get("task") == self.task:
                limit = (profile.get("limits") or {}).get("max_prediction_rows")
                return int(limit) if limit else None
        return None

    def _checked(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if not hasattr(self, "predictor_id_"):
            raise RuntimeError("WeltLearner is not fitted; call fit first")
        matrix = _features(X)
        if matrix.shape[1] != self._n_features:
            raise ValueError(f"X has {matrix.shape[1]} columns; the fit had {self._n_features}")
        return matrix

    def _predict(self, X: npt.NDArray[np.float64], probabilities: bool) -> list[dict[str, Any]]:
        matrix = self._checked(X)
        columns = [f"x{i}" for i in range(matrix.shape[1])]
        rows = matrix_rows(matrix)
        size = self._budget.prediction_rows.get((self.model, self.task)) or len(rows)
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
        if "_local" in self.__dict__:
            n = self._checked(X).shape[0]
            if self.task == "regression":
                return np.full(n, self._local[0], dtype=np.float64)
            return np.full(n, self.classes_[int(np.argmax(self._local))])
        answers = self._predict(X, probabilities=False)
        values = [value for answer in answers for value in answer["predictions"]]
        if self.task == "regression":
            return np.asarray(values, dtype=np.float64)
        return np.asarray(values)

    def _predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if "_local" in self.__dict__:
            return np.tile(self._local, (self._checked(X).shape[0], 1))
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
