# SPDX-License-Identifier: Apache-2.0
"""ergodic.welt against a fake Welt API served through httpx.MockTransport.

No test here touches the network: every ``welt.Client`` gets a mock transport.
The fake server does real arithmetic where a test needs it (an OLS "model"), so
an estimate through WeltLearner can be held to the same estimate with ergodic's
own OLS learner on the same folds.
"""

from __future__ import annotations

import json
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("welt")
httpx = pytest.importorskip("httpx")

import welt  # noqa: E402

import ergodic  # noqa: E402
from ergodic import dag, estimate_effect, tabular  # noqa: E402
from ergodic.discovery import DiscoveryResult, bootstrap, discover  # noqa: E402
from ergodic.inference import Learner  # noqa: E402
from ergodic.welt import (  # noqa: E402
    WeltGraphNotDAGError,
    WeltLearner,
    load_research_dataset,
    upload_dataset,
)
from ergodic.welt import discover as welt_discover  # noqa: E402

HAS_FORMAT = getattr(ergodic, "GRAPH_FORMAT", None) == "ergodic.graph/1"
needs_format = pytest.mark.skipif(
    not HAS_FORMAT, reason="welt-client's to_ergodic needs ergodic.GRAPH_FORMAT (API-140)"
)


class FakeWelt:
    """A stateful stand-in for the Welt API routes ergodic.welt uses."""

    def __init__(
        self,
        *,
        graph_type: str = "dag",
        prediction_rows: int | None = 100,
        catalogue_graph: dict[str, str] | None = None,
        pending_predictions: bool = False,
    ) -> None:
        self.graph_type = graph_type
        # When set, every prediction answers 504 prediction_timeout with a job to poll.
        self.pending_predictions = pending_predictions
        # Native graph types the fake catalogue states, by model id (none by default,
        # as Welt's catalogue does not state them today).
        self.catalogue_graph = catalogue_graph or {}
        self.prediction_rows = prediction_rows
        self.datasets: dict[str, dict[str, Any]] = {}
        self.predictors: dict[str, dict[str, Any]] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str]] = []
        self.predict_batches: list[int] = []

    def client(self) -> Any:
        return welt.Client(
            base_url="https://welt.test",
            api_key="wk_fixture_only",
            transport=httpx.MockTransport(self.handle),
        )

    def count(self, method: str, path_start: str) -> int:
        return sum(1 for m, p in self.calls if m == method and p.startswith(path_start))

    def handle(self, request: Any) -> Any:
        path = request.url.path
        self.calls.append((request.method, path))
        body: dict[str, Any] = json.loads(request.content) if request.content else {}
        if request.method == "POST" and path == "/v1/datasets":
            ident = f"ds_{len(self.datasets) + 1}"
            self.datasets[ident] = body
            return httpx.Response(201, json={"id": ident, "name": body["name"]})
        if request.method == "POST" and path == "/v1/discoveries":
            return self._discover(body)
        if request.method == "POST" and path == "/v1/fits":
            return self._fit(body)
        if request.method == "GET" and path == "/v1/models":
            return httpx.Response(200, json={"items": []})
        if request.method == "GET" and path.startswith("/v1/models/"):
            limits = {} if self.prediction_rows is None else {"max_prediction_rows": 100}
            profiles = [
                {"task": "regression", "limits": limits},
                {"task": "classification", "limits": limits},
            ]
            ident = path.rsplit("/", 1)[1]
            if ident in self.catalogue_graph:
                profiles = [
                    {"task": "causal_discovery", "native_graph_type": self.catalogue_graph[ident]}
                ]
            model = {"id": ident, "task_profiles": profiles}
            return httpx.Response(200, json=model)
        if request.method == "GET" and path.startswith("/v1/predictors/"):
            return httpx.Response(200, json=self.predictors[path.rsplit("/", 1)[1]]["public"])
        if request.method == "POST" and path.endswith("/predict"):
            return self._predict(path.split("/")[3], body)
        if request.method == "GET" and path.startswith("/v1/jobs/"):
            parts = path.split("/")
            job = self.jobs[parts[3]]
            if len(parts) == 4:
                return httpx.Response(200, json=job["status"])
            if parts[-1] == "scores":
                return httpx.Response(200, json=job["scores"])
            return httpx.Response(200, json=job["result"])
        raise AssertionError(f"unexpected request {request.method} {path}")

    def _discover(self, body: dict[str, Any]) -> Any:
        dataset = self.datasets[body["dataset_id"]]
        assert dataset["target"] is None
        names = dataset["columns"]
        job = f"job_d{len(self.jobs) + 1}"
        # A chain over the uploaded order, plus one isolate when there is room.
        edges = [
            {"u": a, "v": b, "u_mark": "tail", "v_mark": "arrow"} for a, b in pairwise(names[:-1])
        ]
        values = [[0.0 if i == j else 0.25 for j in range(len(names))] for i in range(len(names))]
        result = {
            "id": f"causal_{job}",
            "workspace_id": "ws_fixture",
            "dataset_id": body["dataset_id"],
            "model_version": f"{body['model_id']}-fixture-1",
            "configuration_version": "fixture-config",
            "variables": names,
            "native_graph_type": self.graph_type,
            "edge_marks": edges,
            "score_semantics": "native_directed_edge_probability",
            "score_reference": f"scores_{job}",
            "decoder": {"name": "fixture-decoder"},
            "assumptions": ["fixture only"],
            "constraints_applied": body["constraints"],
            "diagnostics": {"job_id": job, "seed": body["seed"]},
        }
        scores = {
            "result_id": result["id"],
            "score_reference": result["score_reference"],
            "variables": names,
            "score_axes": "source_row_target_column",
            "score_semantics": "native_directed_edge_probability",
            "values": values,
        }
        self.jobs[job] = {
            "status": {"id": job, "status": "succeeded", "operation": "discover"},
            "result": result,
            "scores": scores,
        }
        return httpx.Response(202, json={"id": job})

    def _fit(self, body: dict[str, Any]) -> Any:
        dataset = self.datasets[body["dataset_id"]]
        target = dataset["target"]
        index = dataset["columns"].index(target)
        rows = np.asarray(
            [[v for k, v in enumerate(r) if k != index] for r in dataset["rows"]], dtype=float
        )
        labels = [r[index] for r in dataset["rows"]]
        ident = f"pred_{len(self.predictors) + 1}"
        public: dict[str, Any] = {
            "id": ident,
            "task": body["task"],
            "model_id": body["model_id"],
            "model_version": f"{body['model_id']}-fixture-1",
            "columns": [c for c in dataset["columns"] if c != target],
        }
        state: dict[str, Any] = {"public": public}
        if body["task"] == "regression":
            design = np.column_stack([np.ones(len(rows)), rows])
            state["coef"] = np.linalg.lstsq(design, np.asarray(labels, dtype=float), rcond=None)[0]
        else:
            # Classes come back in reverse order, so the learner must reorder them.
            classes = sorted(set(labels), reverse=True)
            public["classes"] = classes
            state["proba"] = [labels.count(c) / len(labels) for c in classes]
        self.predictors[ident] = state
        job = f"job_f{len(self.jobs) + 1}"
        self.jobs[job] = {
            "status": {"id": job, "status": "succeeded", "operation": "fit", "predictor_id": ident}
        }
        return httpx.Response(202, json={"id": job})

    def _predict(self, ident: str, body: dict[str, Any]) -> Any:
        answer = self._answer(ident, body)
        if not self.pending_predictions:
            return httpx.Response(200, json=answer)
        job = f"job_p{len(self.jobs) + 1}"
        self.jobs[job] = {
            "status": {"id": job, "status": "succeeded", "operation": "predict"},
            "result": answer,
        }
        error = {"code": "prediction_timeout", "message": "Continue polling.", "job_id": job}
        return httpx.Response(504, json={"error": error})

    def _answer(self, ident: str, body: dict[str, Any]) -> dict[str, Any]:
        rows = np.asarray(body["rows"], dtype=float)
        self.predict_batches.append(len(rows))
        if self.prediction_rows is not None:
            assert len(rows) <= self.prediction_rows
        state = self.predictors[ident]
        if "coef" in state:
            design = np.column_stack([np.ones(len(rows)), rows])
            return {"predictions": (design @ state["coef"]).tolist()}
        proba = [state["proba"]] * len(rows)
        top = state["public"]["classes"][int(np.argmax(state["proba"]))]
        return {
            "predictions": [top] * len(rows),
            "probabilities": proba if body["probabilities"] else None,
        }


@pytest.fixture(autouse=True)
def _no_ambient_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WELT_WORKSPACE", raising=False)
    monkeypatch.delenv("WELT_BASE_URL", raising=False)


def continuous(n: int = 200, seed: int = 0) -> ergodic.TabularData:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = 0.8 * z + rng.normal(size=n)
    y = 1.5 * x + 0.7 * z + rng.normal(size=n)
    w = rng.normal(size=n)
    return tabular(pd.DataFrame({"Z": z, "X": x, "Y": y, "W": w}))


# -- discovery --------------------------------------------------------------


@needs_format
def test_welt_discovery_returns_the_native_dag_with_provenance() -> None:
    fake = FakeWelt()
    data = continuous()
    with fake.client() as client:
        result = welt_discover(data, "arrow", client=client, seed=7)
    assert isinstance(result, DiscoveryResult)
    assert result.method == "welt:arrow"
    assert result.n == data.n_rows
    assert result.graph.kind.value == "dag"
    # Every variable survives, the isolate W included, with native direction.
    assert result.graph.nodes == frozenset({"Z", "X", "Y", "W"})
    matrix, order = result.graph.to_adjacency(order=["Z", "X", "Y", "W"])
    assert order == ["Z", "X", "Y", "W"]
    assert matrix.tolist() == [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    d = result.diagnostics
    assert d["welt_job_id"] == "job_d1"
    assert d["welt_result_id"] == "causal_job_d1"
    assert d["welt_dataset_id"] == "ds_1"
    assert d["welt_model"] == "arrow"
    assert d["welt_model_version"] == "arrow-fixture-1"
    assert d["welt_configuration_version"] == "fixture-config"
    assert d["welt_seed"] == 7
    scores = d["welt_scores"]
    assert isinstance(scores, pd.DataFrame)
    assert list(scores.index) == list(scores.columns) == ["Z", "X", "Y", "W"]
    assert scores.index.name == "source" and scores.columns.name == "target"
    assert fake.datasets["ds_1"]["columns"] == ["Z", "X", "Y", "W"]
    assert fake.datasets["ds_1"]["target"] is None


@needs_format
def test_discover_facade_routes_welt_methods_lazily() -> None:
    fake = FakeWelt()
    with fake.client() as client:
        result = discover(continuous(), method="welt:arrow", client=client)
    assert result.method == "welt:arrow"
    assert fake.count("POST", "/v1/discoveries") == 1


@pytest.mark.parametrize("model", ["cdfm", "avici"])
def test_models_known_to_return_directed_graphs_are_refused_before_paying(model: str) -> None:
    fake = FakeWelt(graph_type="directed")
    with fake.client() as client, pytest.raises(WeltGraphNotDAGError) as error:
        welt_discover(continuous(), model, client=client)
    assert error.value.model == model
    assert error.value.native_graph_type == "directed"
    assert error.value.result is None and error.value.scores is None
    message = str(error.value)
    assert "'directed'" in message and "nothing was uploaded or submitted" in message
    assert "welt.Client().discover" in message
    # Only the free catalogue read happened: no upload, no discovery job.
    assert fake.count("POST", "/v1/datasets") == 0
    assert fake.count("POST", "/v1/discoveries") == 0


def test_the_catalogue_stated_graph_type_wins_over_the_built_in_list() -> None:
    fake = FakeWelt(catalogue_graph={"new-causal": "directed"})
    with fake.client() as client, pytest.raises(WeltGraphNotDAGError) as error:
        welt_discover(continuous(), "new-causal", client=client)
    assert error.value.result is None
    assert fake.count("POST", "/v1/discoveries") == 0


def test_a_result_that_is_not_a_declared_dag_is_refused_after_the_run() -> None:
    """The second guard: a model with no known graph type runs, and its result is checked."""
    fake = FakeWelt(graph_type="directed")
    with fake.client() as client, pytest.raises(WeltGraphNotDAGError) as error:
        welt_discover(continuous(), "unlisted-causal", client=client)
    # The paid result is not lost: the error carries it and its scores.
    assert error.value.native_graph_type == "directed"
    assert error.value.result.native_graph_type == "directed"
    assert error.value.result.job_id == "job_d1"
    assert error.value.scores is not None
    assert list(error.value.scores.columns) == ["Z", "X", "Y", "W"]
    assert "does not project" in str(error.value)
    assert fake.count("POST", "/v1/discoveries") == 1


def test_discovery_refusals_happen_before_any_request() -> None:
    fake = FakeWelt()
    frame = continuous().to_pandas()
    with fake.client() as client:
        with pytest.raises(ValueError, match="domain knowledge"):
            welt_discover(continuous(), client=client, knowledge=ergodic.DomainKnowledge())
        labelled = tabular(frame.assign(G=np.where(frame["X"] > 0, "a", "b")))
        with pytest.raises(ValueError, match="continuous"):
            welt_discover(labelled, client=client)
        holed = frame.copy()
        holed.loc[3, "Y"] = np.nan
        with pytest.raises(ValueError, match="complete rows"):
            welt_discover(tabular(holed), client=client)
        with pytest.raises(ValueError, match="no test or score"):
            discover(continuous(), method="welt:arrow", test="linear", client=client)
        with pytest.raises(ValueError, match="paid Welt jobs"):
            bootstrap(continuous(), method="welt:arrow", n_bootstrap=5, client=client)
    assert fake.calls == []


# -- the learner --------------------------------------------------------------


def test_welt_learner_satisfies_the_learner_protocol_by_task() -> None:
    regression = WeltLearner("tabicl-v2")
    classification = WeltLearner("tabicl-v2", task="classification")
    assert isinstance(regression, Learner)
    assert not hasattr(regression, "predict_proba")
    assert hasattr(classification, "predict_proba")
    with pytest.raises(ValueError, match="task"):
        WeltLearner(task="forecasting")
    with pytest.raises(ValueError, match="max_fits"):
        WeltLearner(max_fits=0)
    # Any model id passes to Welt unvalidated; Welt's catalogue is the authority.
    assert WeltLearner("any-future-model").model == "any-future-model"


def test_sample_weight_is_refused_not_ignored() -> None:
    fake = FakeWelt()
    with fake.client() as client:
        learner = WeltLearner(client=client)
        with pytest.raises(ValueError, match="sample_weight"):
            learner.fit(np.ones((4, 1)), np.arange(4.0), sample_weight=np.ones(4))
    assert fake.calls == []
    assert learner.fits_used == 0


def test_an_empty_feature_matrix_is_refused_before_a_fit() -> None:
    with pytest.raises(ValueError, match="at least one feature"):
        WeltLearner(client=object()).fit(np.empty((5, 0)), np.arange(5.0))


def test_regression_fit_and_batched_predict() -> None:
    fake = FakeWelt()
    rng = np.random.default_rng(1)
    X = rng.normal(size=(250, 2))
    y = 2.0 + X @ np.array([1.0, -3.0])
    with fake.client() as client:
        learner = WeltLearner("tabicl-v2", client=client).fit(X, y)
        predicted = learner.predict(X)
    np.testing.assert_allclose(predicted, y, atol=1e-9)
    assert learner.predictor_id_ == "pred_1"
    assert learner.model_version_ == "tabicl-v2-fixture-1"
    # 250 rows go up in batches no larger than the catalogue's prediction limit.
    assert fake.predict_batches == [100, 100, 50]
    assert fake.datasets["ds_1"]["target"] == "__ergodic_target__"


def test_classification_probabilities_follow_sorted_classes() -> None:
    fake = FakeWelt()
    X = np.arange(20.0).reshape(-1, 1)
    y = np.array([0.0] * 15 + [1.0] * 5)
    with fake.client() as client:
        learner = WeltLearner(task="classification", client=client).fit(X, y)
        proba = learner.predict_proba(X)
    assert learner.classes_.tolist() == [0, 1]
    np.testing.assert_allclose(proba, [[0.75, 0.25]] * 20)
    # Float 0/1 treatments go up as integer labels.
    assert {row[-1] for row in fake.datasets["ds_1"]["rows"]} == {0, 1}


def test_clones_share_the_client_and_one_fit_budget() -> None:
    from sklearn.base import clone

    from ergodic.inference._learner import _fresh

    sentinel = object()
    learner = WeltLearner("tabicl-v2", client=sentinel, max_fits=3)
    for twin in (_fresh(learner), clone(learner)):
        assert isinstance(twin, WeltLearner)
        assert twin is not learner
        assert twin.client is sentinel
        assert twin.get_params() == learner.get_params()
        assert twin._budget is learner._budget
    learner.set_params(max_fits=5)
    assert learner.max_fits == 5 and learner.fits_used == 0


def test_dml_through_welt_matches_ols_on_the_same_folds() -> None:
    """The fake server fits OLS, so DML must reproduce ergodic's OLS estimate exactly."""
    fake = FakeWelt()
    data = continuous(n=200)
    graph = dag(["Z -> X", "Z -> Y", "X -> Y"])
    native = estimate_effect(
        graph, data, "X", "Y", strategy="dml", learner="ols", random_state=3, n_folds=2
    )
    with fake.client() as client:
        learner = WeltLearner("tabicl-v2", client=client)
        hosted = estimate_effect(
            graph, data, "X", "Y", strategy="dml", learner=learner, random_state=3, n_folds=2
        )
    assert hosted.value == pytest.approx(native.value, rel=1e-9)
    assert hosted.std_error == pytest.approx(native.std_error, rel=1e-9)
    # Two slots times two folds: four paid fits, counted on the shared budget.
    assert learner.fits_used == 4
    assert fake.count("POST", "/v1/fits") == 4
    assert hosted.kind == "analytic"
    # The clones share one catalogue lookup for the prediction-row limit.
    assert fake.count("GET", "/v1/models/") == 1


def test_the_fit_budget_stops_a_bootstrap_fan_out_before_paying() -> None:
    fake = FakeWelt()
    data = continuous(n=60)
    graph = dag(["Z -> X", "Z -> Y", "X -> Y"])
    with fake.client() as client:
        learner = WeltLearner("tabicl-v2", client=client, max_fits=3)
        with pytest.raises(RuntimeError, match="max_fits=3"):
            estimate_effect(
                graph, data, "X", "Y", strategy="dml", learner=learner, n_folds=2, random_state=0
            )
    assert fake.count("POST", "/v1/fits") == 3
    assert learner.fits_used == 3


def test_a_classification_slot_fills_a_propensity() -> None:
    fake = FakeWelt()
    rng = np.random.default_rng(4)
    z = rng.normal(size=120)
    t = (rng.normal(size=120) + z > 0).astype(float)
    y = 2.0 * t + z + rng.normal(size=120)
    data = tabular(pd.DataFrame({"Z": z, "T": t, "Y": y}), types={"T": "categorical"})
    graph = dag(["Z -> T", "Z -> Y", "T -> Y"])
    with fake.client() as client:
        learners = {
            "outcome": WeltLearner(client=client),
            "propensity": WeltLearner(task="classification", client=client),
        }
        estimate = estimate_effect(
            graph, data, "T", "Y", strategy="aipw", learner=learners, random_state=0
        )
    assert np.isfinite(estimate.value)
    assert fake.count("POST", "/v1/fits") >= 2


# -- the data bridge ----------------------------------------------------------


def test_upload_dataset_sends_variables_in_order() -> None:
    fake = FakeWelt()
    frame = pd.DataFrame({"id": ["a", "b", "c"], "A": [1.5, np.nan, 3.0], "B": [1.0, 2.0, 3.0]})
    data = tabular(frame, exclude=["id"])
    with fake.client() as client:
        record = upload_dataset(data, target="B", client=client, name="mine")
        with pytest.raises(ValueError, match="target"):
            upload_dataset(data, target="id", client=client)
    assert record["id"] == "ds_1"
    sent = fake.datasets["ds_1"]
    assert sent["columns"] == ["A", "B"]
    assert sent["rows"] == [[1.5, 1.0], [None, 2.0], [3.0, 3.0]]
    assert sent["target"] == "B" and sent["name"] == "mine"
    with pytest.raises(TypeError, match="tabular"):
        upload_dataset(frame, client=object())  # type: ignore[arg-type]


def test_load_research_dataset_returns_tabular_data() -> None:
    class Research:
        def __init__(self) -> None:
            self.asked: list[tuple[str, str | None]] = []

        def load_research_dataset(self, dataset_id: str, *, version: str | None) -> pd.DataFrame:
            self.asked.append((dataset_id, version))
            return pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": ["x", "y", "x"]})

    research = Research()
    data = load_research_dataset("rd_1", version="v2", client=research)
    assert isinstance(data, ergodic.TabularData)
    assert data.variables == ("a", "b")
    assert data.schema["b"].is_discrete
    assert research.asked == [("rd_1", "v2")]


def test_without_a_client_each_call_opens_and_closes_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeWelt()
    opened: list[Any] = []
    real = welt.Client

    def make() -> Any:
        client = real(
            base_url="https://welt.test",
            api_key="wk_fixture_only",
            transport=httpx.MockTransport(fake.handle),
        )
        opened.append(client)
        return client

    monkeypatch.setattr(welt, "Client", make)
    learner = WeltLearner("tabicl-v2")
    learner.fit(np.arange(10.0).reshape(-1, 1), np.arange(10.0))
    learner.predict(np.arange(3.0).reshape(-1, 1))
    upload_dataset(continuous(n=5))
    assert len(opened) == 3
    assert all(client.http.is_closed for client in opened)


def test_ergodic_welt_names_the_extra_when_welt_client_is_missing() -> None:
    import subprocess
    import sys

    probe = (
        "import sys; sys.modules['welt'] = None\n"
        "try:\n"
        "    import ergodic.welt\n"
        "except ImportError as error:\n"
        "    print(error)\n"
    )
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert "pip install 'ergodic[welt]'" in done.stdout


def test_a_pending_prediction_is_awaited_through_its_job() -> None:
    fake = FakeWelt(pending_predictions=True)
    X = np.arange(30.0).reshape(-1, 1)
    y = 1.0 + 2.0 * X[:, 0]
    with fake.client() as client:
        learner = WeltLearner(client=client).fit(X, y)
        predicted = learner.predict(X)
    np.testing.assert_allclose(predicted, y, atol=1e-9)
    assert fake.count("GET", "/v1/jobs/job_p") >= 2  # status, then the result


def test_a_weighted_slot_refuses_a_welt_learner_before_any_fit() -> None:
    """R-learner's final stage is fit with sample_weight; WeltLearner says it can't."""
    from ergodic import cate

    fake = FakeWelt()
    rng = np.random.default_rng(5)
    z = rng.normal(size=80)
    t = (rng.normal(size=80) + z > 0).astype(float)
    y = 2.0 * t + z + rng.normal(size=80)
    data = tabular(pd.DataFrame({"Z": z, "T": t, "Y": y}), types={"T": "categorical"})
    graph = dag(["Z -> T", "Z -> Y", "T -> Y"])
    assert WeltLearner.supports_sample_weight is False
    with fake.client() as client:
        learners = {
            "outcome": WeltLearner(client=client),
            "propensity": WeltLearner(task="classification", client=client),
            "final": WeltLearner(client=client),
        }
        with pytest.raises(TypeError, match="sample_weight"):
            cate(graph, data, "T", "Y", method="r_learner", learner=learners)
    assert fake.calls == []


def test_welt_methods_refuse_local_only_arguments_before_any_request() -> None:
    fake = FakeWelt()
    with fake.client() as client:
        with pytest.raises(ValueError, match="alpha"):
            discover(continuous(), method="welt:arrow", alpha=0.01, client=client)
        with pytest.raises(ValueError, match="max_cond_size"):
            discover(continuous(), method="welt:arrow", max_cond_size=2, client=client)
    assert fake.calls == []
