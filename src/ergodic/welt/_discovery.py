# SPDX-License-Identifier: Apache-2.0
"""Welt causal discovery as an ergodic discovery method.

One run uploads the table, submits one Welt discovery job, reads the native result
and its edge scores, and converts a declared native DAG with
``welt.CausalResult.to_ergodic()``. Nothing is thresholded, oriented or projected
here: a result Welt does not declare a DAG is refused, with the native result
attached so the paid run is not lost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..discovery import DiscoveryResult
from ._client import frame_rows, using
from ._data import _require_tabular

if TYPE_CHECKING:
    import pandas as pd

    from ..data import TabularData
    from ..knowledge import DomainKnowledge


class WeltGraphNotDAGError(ValueError):
    """A Welt model returned a native graph that is not a declared DAG.

    ergodic converts only declared DAGs and never projects one into a DAG. The
    paid result stays in your Welt workspace, and this error carries it.

    Attributes:
        result: The native ``welt.CausalResult`` (variables, edge marks, provenance).
        scores: The edge-score matrix, rows the source and columns the target.
    """

    def __init__(self, result: Any, scores: pd.DataFrame) -> None:
        self.result = result
        self.scores = scores
        super().__init__(
            f"Welt model returned a native {result.native_graph_type!r} graph, not a "
            "declared DAG; ergodic does not project it. The result stays in your Welt "
            f"workspace (job {result.job_id!r}); read it with "
            f"welt.Client().causal_result({result.job_id!r}) or from this error's "
            ".result and .scores"
        )


def _score_frame(scores: dict[str, Any]) -> pd.DataFrame:
    import pandas as pd

    names = list(scores["variables"])
    frame = pd.DataFrame(scores["values"], index=names, columns=names, dtype=float)
    frame.index.name = "source"
    frame.columns.name = "target"
    return frame


def discover(
    data: TabularData,
    model: str = "arrow",
    *,
    seed: int = 0,
    timeout: float = 600.0,
    knowledge: DomainKnowledge | None = None,
    client: Any = None,
) -> DiscoveryResult:
    """Learn a DAG with a Welt causal discovery model.

    Uploads the dataset's variables (continuous, no missing values), runs one
    discovery job and waits for it. This is one paid Welt job; the uploaded
    dataset and the native result stay in your workspace until you delete them.
    ``ergodic.discover(data, method="welt:<model>")`` calls this.

    Welt's models and limits come from its catalogue (``welt.Client().models()``).
    Today ``arrow`` declares native DAGs; ``cdfm`` and ``avici`` return native
    directed graphs that may hold cycles or reciprocal edges, which this refuses
    with :class:`WeltGraphNotDAGError`.

    Args:
        data: Observational rows; every variable must be continuous and complete.
        model: The Welt causal discovery model id, for example ``"arrow"``.
        seed: The run's seed, recorded in the result's provenance.
        timeout: Seconds to wait for the job. A timeout leaves the job running in
            Welt; reconnect with ``welt.Client().job(job_id)``.
        knowledge: Must be ``None``. Welt discovery takes no constraints yet.
        client: A ``welt.Client`` to use as is; ``None`` opens a short-lived one.

    Returns:
        A :class:`~ergodic.discovery.DiscoveryResult` whose ``graph`` is the native
        DAG with every variable, ``method`` is ``"welt:<model>"``, and whose
        ``diagnostics`` hold the Welt provenance: ``welt_job_id``,
        ``welt_result_id``, ``welt_dataset_id``, ``welt_model``,
        ``welt_model_version``, ``welt_configuration_version``, ``welt_decoder``,
        ``welt_seed`` and ``welt_scores`` (the native edge scores as a dataframe,
        rows the source and columns the target; not calibrated confidence).

    Raises:
        TypeError: ``data`` is not tabular.
        ValueError: ``knowledge`` was given, or a variable is not continuous or has
            missing values. Nothing is uploaded.
        WeltGraphNotDAGError: The model returned a graph that is not a declared DAG.
    """
    data = _require_tabular(data)
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a Welt model id such as 'arrow'")
    if knowledge is not None:
        raise ValueError(
            "Welt discovery takes no domain knowledge yet; discover without it, then "
            "check the graph against your knowledge"
        )
    variables = list(data.variables)
    not_continuous = [name for name in variables if not data.schema[name].is_continuous]
    if not_continuous:
        raise ValueError(
            f"Welt discovery reads continuous variables; {not_continuous} are not. "
            "Exclude them or retype them with data.with_type(...)"
        )
    incomplete = [name for name, count in data.missing().items() if count and name in variables]
    if incomplete:
        raise ValueError(f"Welt discovery needs complete rows; {incomplete} have missing values")
    columns, rows = frame_rows(data.to_pandas()[variables])
    with using(client) as welt_client:
        dataset = welt_client.upload(
            columns=columns, rows=rows, name="ergodic discovery", target=None
        )
        result = welt_client.discover(dataset["id"], model=model, seed=seed, timeout=timeout)
        scores = _score_frame(welt_client.causal_scores(result.job_id))
    if result.native_graph_type != "dag":
        raise WeltGraphNotDAGError(result, scores)
    graph = result.to_ergodic()
    diagnostics: dict[str, Any] = {
        "welt_job_id": result.job_id,
        "welt_result_id": result.id,
        "welt_dataset_id": result.dataset_id,
        "welt_model": model,
        "welt_model_version": result.model_version,
        "welt_configuration_version": result.configuration_version,
        "welt_decoder": result.to_dict()["decoder"],
        "welt_seed": seed,
        "welt_scores": scores,
    }
    return DiscoveryResult(
        graph=graph,
        method=f"welt:{model}",
        n=data.n_rows,
        diagnostics=diagnostics,
    )
