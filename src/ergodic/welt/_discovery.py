# SPDX-License-Identifier: Apache-2.0
"""Welt causal discovery as an ergodic discovery method.

One run uploads the table, submits one Welt discovery job, reads the native result
and its edge scores, and converts a declared native DAG with
``welt.CausalResult.to_ergodic()``. Nothing is thresholded, oriented or projected
here by default. A model known to return non-DAG graphs is refused before submitting,
and a result Welt does not declare a DAG is refused with the native result attached so
the paid run is not lost.

With an explicit ``decoder=`` the native edge scores become an
:class:`~ergodic.discovery.EdgeProbabilities` and a named local decoder
(:func:`ergodic.decode`) turns them into a DAG labelled ``decoded`` (DEC-079). That is
the only path by which a non-DAG model (AVICI, CDFM) reaches an ergodic DAG.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from ..discovery import Decoder, DecodeResult, DiscoveryResult, EdgeProbabilities, decode
from ..discovery._decode import check_knowledge
from ._client import frame_rows, using
from ._data import _require_tabular

if TYPE_CHECKING:
    import pandas as pd

    from ..data import TabularData
    from ..knowledge import DomainKnowledge


# Native graph types of the Welt causal models, as their adapters declare them
# (welt-api causal-results contract). Used only when Welt's catalogue does not state
# a model's native graph type itself; a model missing from both is submitted, and
# the result's own declaration is checked afterwards.
_KNOWN_NATIVE_GRAPH = {"arrow": "dag", "cdfm": "directed", "avici": "directed"}


class WeltGraphNotDAGError(ValueError):
    """A Welt model's native graph is not a declared DAG, so ergodic refuses it.

    ergodic converts only declared DAGs and never projects or relabels a directed
    graph as one. When the model is known to return such graphs, the refusal comes
    before anything is uploaded or paid for, and ``result`` is ``None``. When a
    finished result turns out not to be a declared DAG, the paid result stays in
    your Welt workspace and this error carries it.

    Attributes:
        model: The Welt model id.
        native_graph_type: The model's native graph type, for example ``"directed"``.
        result: The native ``welt.CausalResult``, or ``None`` if nothing was submitted.
        scores: Its edge-score matrix (rows the source, columns the target), or
            ``None`` if nothing was submitted.
    """

    def __init__(
        self,
        model: str,
        native_graph_type: str,
        result: Any = None,
        scores: pd.DataFrame | None = None,
    ) -> None:
        self.model = model
        self.native_graph_type = native_graph_type
        self.result = result
        self.scores = scores
        direct = (
            f"To read its native graph and scores, call welt.Client().discover(dataset_id, "
            f"model={model!r}) and use the CausalResult and client.causal_scores directly"
        )
        if result is None:
            message = (
                f"Welt model {model!r} returns native {native_graph_type!r} graphs, not "
                "declared DAGs, and ergodic never projects one into a DAG; nothing was "
                f"uploaded or submitted. {direct}"
            )
        else:
            message = (
                f"Welt model {model!r} returned a native {native_graph_type!r} graph, not "
                "a declared DAG; ergodic does not project it. The result stays in your "
                f"Welt workspace (job {result.job_id!r}): read it from this error's .result "
                f"and .scores, or with welt.Client().causal_result({result.job_id!r})"
            )
        super().__init__(message)


def native_graph_type(model: str, client: Any) -> str | None:
    """The native graph type a Welt causal model returns, before running it.

    Welt's catalogue entry wins when it states one (``native_graph_type`` on the
    model or its ``causal_discovery`` task profile); otherwise the type the model's
    adapter is known to declare; otherwise ``None`` (unknown).
    """
    entry = client.model(model)
    profiles = [
        profile
        for profile in entry.get("task_profiles") or []
        if profile.get("task") == "causal_discovery"
    ]
    for source in (*profiles, entry):
        stated = source.get("native_graph_type")
        if isinstance(stated, str) and stated:
            return stated
    return _KNOWN_NATIVE_GRAPH.get(model)


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
    decoder: str | Decoder | None = None,
    client: Any = None,
) -> DiscoveryResult:
    """Learn a DAG with a Welt causal discovery model.

    Uploads the dataset's variables (continuous, no missing values), runs one
    discovery job and waits for it. This is one paid Welt job; the uploaded
    dataset and the native result stay in your workspace until you delete them.
    ``ergodic.discover(data, method="welt:<model>")`` calls this.

    Welt's models and limits come from its catalogue (``welt.Client().models()``).
    Today ``arrow`` declares native DAGs. ``cdfm`` and ``avici`` return native
    directed graphs that may hold cycles or reciprocal edges, so they are refused
    with :class:`WeltGraphNotDAGError` before anything is uploaded or paid for
    (the catalogue's stated graph type wins over that built-in list). A result
    that is still not a declared DAG is refused too, with the paid result attached.

    ``decoder=`` opts into local decoding (DEC-079). The run then accepts any model,
    reads the native edge scores (rows the source, columns the target) as
    :class:`~ergodic.discovery.EdgeProbabilities` of kind ``"independent"`` (each
    score a separate directed-edge marginal, uncalibrated) with the Welt provenance,
    and decodes them with :func:`ergodic.decode`. The returned graph is that decoded
    DAG, not the model's native graph, and its diagnostics say so. For the greedy
    decoder without an explicit threshold, the model's native threshold is used when
    Welt records one (``decoder.parameters.threshold``: 0.5 for AVICI, the adaptive
    threshold for CDFM), so greedy keeps what the model's own rule keeps and only
    drops edges that close a cycle or a reciprocal pair. On a DAG model (``arrow``)
    the decoder replaces the native DAG with one decoded from the same scores, read
    the same independent way. Any other choice can be made locally, with no new run,
    from ``result.diagnostics["edge_probabilities"]``.

    Args:
        data: Observational rows; every variable must be continuous and complete.
        model: The Welt causal discovery model id, for example ``"arrow"``.
        seed: The run's seed, recorded in the result's provenance.
        timeout: Seconds to wait for the job. A timeout leaves the job running in
            Welt; reconnect with ``welt.Client().job(job_id)``.
        knowledge: Must be ``None`` without a decoder (Welt discovery takes no
            constraints yet). With a decoder it constrains the local decoding only;
            Welt still runs unconstrained. It is checked before anything is uploaded.
        decoder: ``None`` (native DAGs only), a decoder id such as ``"greedy"``, or
            a :class:`~ergodic.discovery.Decoder` with its settings (a sampler needs
            ``Decoder("pair_sampler", samples=..., seed=...)``).
        client: A ``welt.Client`` to use as is; ``None`` opens a short-lived one.

    Returns:
        A :class:`~ergodic.discovery.DiscoveryResult` whose ``graph`` is the native
        DAG with every variable, ``method`` is ``"welt:<model>"``, and whose
        ``diagnostics`` hold the Welt provenance: ``welt_job_id``,
        ``welt_result_id``, ``welt_dataset_id``, ``welt_model``,
        ``welt_model_version``, ``welt_configuration_version``, ``welt_decoder``,
        ``welt_seed`` and ``welt_scores`` (the native edge scores as a dataframe,
        rows the source and columns the target; not calibrated confidence).
        With a decoder it is a :class:`~ergodic.discovery.DecodeResult` (still a
        ``DiscoveryResult``, ``method`` ``"welt:<model>"``) whose ``graph`` is the
        decoded DAG; ``diagnostics`` add the decode labels (``decoded`` ``True``,
        ``native`` ``False``, ``decoder``, ``parameters``, ``threshold_source``,
        ``rejected_cycles``, ``note``, ...), ``welt_native_graph_type``,
        ``welt_native_edges`` (the native directed edges as pairs) and
        ``edge_probabilities``.

    Raises:
        TypeError: ``data`` is not tabular.
        ValueError: ``knowledge`` was given without a decoder (or names unknown
            variables or a confounded pair), the decoder settings are invalid, or a
            variable is not continuous or has missing values. Nothing is uploaded.
        WeltGraphNotDAGError: The model's native graph is not a declared DAG, either
            known before submitting (nothing is paid) or found in the result.
    """
    data = _require_tabular(data)
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a Welt model id such as 'arrow'")
    config = _decoder_config(decoder)
    if knowledge is not None and config is None:
        raise ValueError(
            "Welt discovery takes no domain knowledge yet; discover without it, then "
            "check the graph against your knowledge (or pass decoder= to apply it "
            "while decoding the edge scores locally)"
        )
    variables = list(data.variables)
    if knowledge is not None:
        check_knowledge(variables, knowledge)
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
        # cdfm and avici are refused here, before anything is paid for, unless the
        # caller opted into an explicit local decoder (DEC-079).
        if config is None:
            expected = native_graph_type(model, welt_client)
            if expected is not None and expected != "dag":
                raise WeltGraphNotDAGError(model, expected)
        dataset = welt_client.upload(
            columns=columns, rows=rows, name="ergodic discovery", target=None
        )
        result = welt_client.discover(dataset["id"], model=model, seed=seed, timeout=timeout)
        scores = _score_frame(welt_client.causal_scores(result.job_id))
    if config is not None:
        return _decoded(data.n_rows, model, seed, result, scores, config, knowledge)
    if result.native_graph_type != "dag":
        raise WeltGraphNotDAGError(model, result.native_graph_type, result, scores)
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


def _decoder_config(decoder: str | Decoder | None) -> Decoder | None:
    """The decoder settings, checked before any request."""
    if decoder is None or isinstance(decoder, Decoder):
        return decoder
    if isinstance(decoder, str):
        try:
            return Decoder(decoder)
        except ValueError as error:
            raise ValueError(
                f"{error}; pass decoder=ergodic.discovery.Decoder({decoder!r}, ...) with "
                "its settings"
            ) from None
    raise TypeError(
        f"decoder must be a decoder id, an ergodic.discovery.Decoder or None, "
        f"got {type(decoder).__name__}"
    )


def _welt_provenance(model: str, seed: int, result: Any) -> dict[str, Any]:
    payload = result.to_dict()
    return {
        "source": "welt",
        "welt_model": model,
        "welt_model_version": result.model_version,
        "welt_configuration_version": result.configuration_version,
        "welt_job_id": result.job_id,
        "welt_result_id": result.id,
        "welt_dataset_id": result.dataset_id,
        "welt_seed": seed,
        "native_graph_type": result.native_graph_type,
        "native_decoder": payload["decoder"],
        "score_semantics": payload.get("score_semantics"),
    }


def _decoded(
    n: int,
    model: str,
    seed: int,
    result: Any,
    scores: pd.DataFrame,
    config: Decoder,
    knowledge: DomainKnowledge | None,
) -> DecodeResult:
    """Decode a Welt result's edge scores locally, keeping the Welt provenance."""
    probabilities = EdgeProbabilities.from_frame(
        scores, kind="independent", provenance=_welt_provenance(model, seed, result)
    )
    native_threshold = (result.to_dict()["decoder"].get("parameters") or {}).get("threshold")
    threshold = config.threshold
    threshold_source = "decoder" if threshold is not None else None
    if config.method == "greedy_v1" and threshold is None and native_threshold is not None:
        # Welt allows a native threshold of 1 (nothing passes); the decoder's cut is
        # strict and below 1, so the largest float under 1 reads the same.
        threshold = min(float(native_threshold), math.nextafter(1.0, 0.0))
        threshold_source = "welt_native"
    decoded = decode(
        probabilities,
        config.method,
        threshold=threshold,
        knowledge=knowledge,
        samples=config.samples,
        seed=config.seed,
        max_nodes=config.max_nodes,
    )
    native_edges = tuple(
        (edge["u"], edge["v"]) if edge["v_mark"] == "arrow" else (edge["v"], edge["u"])
        for edge in result.to_dict()["edge_marks"]
    )
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
        "welt_native_graph_type": result.native_graph_type,
        "welt_native_edges": native_edges,
        **decoded.diagnostics,
        "threshold_source": threshold_source,
        "edge_probabilities": probabilities,
    }
    return DecodeResult(
        graph=decoded.graph,
        method=f"welt:{model}",
        n=n,
        diagnostics=diagnostics,
        decoder=decoded.decoder,
        candidates=decoded.candidates,
        probabilities=probabilities,
    )
