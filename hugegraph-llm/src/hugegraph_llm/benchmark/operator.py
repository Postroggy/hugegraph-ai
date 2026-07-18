# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Atomic benchmark evaluation operator.

``evaluate`` is the single entry point for embedding benchmark evaluation
into external pipelines (e.g. a graph-extraction workflow's quality gate).
The caller supplies gold and candidate as two separate in-memory lists —
the natural shape for a pipeline that has standard annotations on one side
and extractor output on the other — and the operator owns pairing them,
deriving the schema, building the LLM-Judge client, and running the metric
suite.

Deliberately bounded responsibilities:

  * pairs ``gold`` + ``candidate`` by ``sample_id`` and assembles samples;
  * derives the benchmark schema from gold only;
  * builds the LLM-Judge client from ``llm_settings`` (``.env``);
  * runs the extraction metric suite via ``ExtractionRunner``;
  * returns a ``BenchmarkResult``.

It does NOT: convert data formats (use ``car_pipeline.build_extraction_inputs``
for car-format data), save / load baselines, compare runs, render reports,
or reconfigure global logging. It raises standard exceptions
(``TypeError`` / ``ValueError``) rather than ``SystemExit`` so a host process
can catch failures without being killed.

Only the ``extraction`` dimension is supported for now. ``retrieval`` and
``answer`` will adopt the same gold+candidate shape in a follow-up.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from hugegraph_llm.benchmark.llm_judge.client import create_judge_llm
from hugegraph_llm.benchmark.metrics.dimensions import get_dimension
from hugegraph_llm.benchmark.metrics.registry import MetricRegistry
from hugegraph_llm.benchmark.models.result import BenchmarkResult
from hugegraph_llm.benchmark.runners.extraction_runner import ExtractionRunner

logger = logging.getLogger(__name__)

# Full default metric set for extraction, run when ``metrics`` is omitted.
_EXTRACTION_METRICS = [
    "entity_f1", "triple_f1", "schema_validity", "property_f1",
    "semantic_entity_f1", "semantic_triple_f1", "extraction_faithfulness",
]

# dimensions.py keys answer-quality metrics under the "generation" domain; the
# operator surfaces them as the "answer" dimension (matching AnswerRunner).
_DOMAIN_TO_DIMENSION = {
    "extraction": "extraction",
    "retrieval": "retrieval",
    "generation": "answer",
}

SUPPORTED_DIMENSIONS = ("extraction",)


def _infer_dimension(metrics: List[str]) -> str:
    """Return the single dimension shared by every metric in ``metrics``.

    Raises ``ValueError`` if a metric is unsupported or the list spans more
    than one dimension — cross-dimension evaluation is not allowed because
    each dimension's runner expects a different data schema.
    """
    dims = set()
    for m in metrics:
        domain, _ = get_dimension(m)
        dim = _DOMAIN_TO_DIMENSION.get(domain)
        if dim is None:
            raise ValueError(
                f"metric {m!r} does not belong to a supported dimension "
                f"(supported: extraction / retrieval / answer)"
            )
        dims.add(dim)
    if len(dims) != 1:
        raise ValueError(
            f"metrics span multiple dimensions {sorted(dims)}; pass metrics "
            f"from a single dimension only"
        )
    return dims.pop()


def _infer_primary_key(vertex: Dict[str, Any]) -> str:
    """Infer a vertex's primary key: the property whose value equals its name.

    Car-pipeline vertices carry ``name = properties[name_property]`` (see
    GraphVertex._fill_name_from_property), so the property whose value matches
    ``name`` is the primary key — this works without hard-coding the car
    ``_NAME_PROPERTY`` map. Falls back to ``"name"`` when no match is found.
    """
    name = vertex.get("name")
    props = vertex.get("properties") or {}
    if name:
        for key, value in props.items():
            if value == name:
                return key
    return "name"


def _derive_schema(gold: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Derive the benchmark schema ``{vertexlabels, edgelabels}`` from gold.

    Only gold vertices/edges contribute, so the schema encodes the
    human-annotated type universe; candidate types absent from gold are then
    flagged by ``schema_validity`` rather than being silently self-endorsed.
    """
    vertex_types: Dict[str, str] = {}
    edge_types: Dict[str, Tuple[str, str]] = {}
    for sample in gold:
        for vertex in sample.get("vertices", []):
            label = vertex.get("label")
            if label and label not in vertex_types:
                vertex_types[label] = _infer_primary_key(vertex)
        for edge in sample.get("edges", []):
            label = edge.get("label")
            if label and label not in edge_types:
                edge_types[label] = (edge.get("outVLabel", ""), edge.get("inVLabel", ""))
    vertexlabels = [{"name": name, "primary_keys": [pk]} for name, pk in vertex_types.items()]
    edgelabels = [
        {"name": name, "source_label": src, "target_label": tgt}
        for name, (src, tgt) in edge_types.items()
    ]
    return {"vertexlabels": vertexlabels, "edgelabels": edgelabels}


def _pair_extraction_samples(
    gold: List[Dict[str, Any]], candidate: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Pair gold/candidate by ``sample_id`` into extraction samples.

    Gold-driven: every gold sample becomes a benchmark sample. A gold sample
    with no matching candidate gets empty ``candidate_*`` lists so the
    benchmark surfaces "extraction produced nothing" cases. Candidates with
    no matching gold are dropped (no gold → nothing to score against).
    """
    candidate_by_id = {c.get("sample_id"): c for c in candidate}
    samples: List[Dict[str, Any]] = []
    for g in gold:
        sample_id = g.get("sample_id")
        if not sample_id:
            continue
        c = candidate_by_id.get(sample_id, {})
        samples.append(
            {
                "sample_id": sample_id,
                "input_text": g.get("input_text", ""),
                "gold_vertices": g.get("vertices", []),
                "gold_edges": g.get("edges", []),
                "candidate_vertices": c.get("vertices", []),
                "candidate_edges": c.get("edges", []),
            }
        )
    return samples


def evaluate(
    gold: List[Dict[str, Any]],
    candidate: List[Dict[str, Any]],
    *,
    dimension: str = "extraction",
    metrics: Optional[List[str]] = None,
    language: str = "zh",
    max_workers: int = 20,
    llm: Optional[Any] = None,
) -> BenchmarkResult:
    """Evaluate extraction ``candidate`` against ``gold``; return a ``BenchmarkResult``.

    Args:
        gold: list of per-sample gold dicts in benchmark format —
            ``{sample_id, vertices, edges, input_text?}`` where vertices are
            ``{label, name, properties}`` and edges are
            ``{label, outV, inV, outVLabel, inVLabel, properties}``.
            Use ``car_pipeline.build_extraction_inputs`` to convert car-format
            gold/candidate (the pipeline's in-memory shape) into this format.
        candidate: list of per-sample candidate dicts, same shape as gold
            (minus ``input_text``). Paired with gold by ``sample_id``.
        dimension: evaluation dimension. Only ``"extraction"`` is supported
            for now.
        metrics: explicit metric list. When omitted, the full extraction
            suite (``_EXTRACTION_METRICS``) is run.
        language: language code for normalization / LLM-Judge prompts
            (``"en"`` or ``"zh"``).
        max_workers: sample-level concurrency for LLM-Judge metrics.
        llm: optional LLM-Judge client for tests. When omitted, a client is
            built from ``llm_settings`` (``.env``). External callers should
            not pass this — the operator owns LLM client creation.

    Returns:
        Aggregated ``BenchmarkResult``.

    Raises:
        TypeError: ``gold`` / ``candidate`` are not lists.
        ValueError: ``dimension`` is not ``"extraction"``; ``gold`` is empty;
            metrics are unknown / cross-dimension / non-extraction; or
            LLM-Judge metrics requested but no LLM client is available.
    """
    if dimension != "extraction":
        raise ValueError(
            f"only dimension='extraction' is supported for now; got {dimension!r}. "
            f"retrieval/answer will adopt the same gold+candidate shape later."
        )
    if not isinstance(gold, list) or not isinstance(candidate, list):
        raise TypeError("gold and candidate must be lists of per-sample dicts")
    if not gold:
        raise ValueError("gold must contain at least one sample")

    # Resolve metrics: explicit list must be extraction-only; default = full set.
    if metrics is not None:
        selected = list(metrics)
        inferred = _infer_dimension(selected)
        if inferred != "extraction":
            raise ValueError(
                f"`metrics` belong to dimension {inferred!r}, not 'extraction'"
            )
    else:
        selected = list(_EXTRACTION_METRICS)
    unknown = [m for m in selected if MetricRegistry.get(m) is None]
    if unknown:
        raise ValueError(f"unknown metric(s): {', '.join(unknown)}")

    # Pair gold+candidate, derive schema from gold, assemble runner data.
    samples = _pair_extraction_samples(gold, candidate)
    schema = _derive_schema(gold)
    data = {"schema": schema, "samples": samples}

    # Build the LLM-Judge client (self-owned). ``create_judge_llm`` returns
    # (None, {}) on failure instead of raising; surface a clear error so the
    # caller knows to configure ``.env`` or drop LLM-Judge metrics.
    llm_meta: Dict[str, Any] = {}
    if llm is None:
        llm, llm_meta = create_judge_llm()
    llm_metric_names = [m for m in selected if MetricRegistry.get(m).requires_llm]
    if llm is None and llm_metric_names:
        raise ValueError(
            f"LLM-Judge metric(s) {llm_metric_names} require a configured LLM "
            f"client (set OPENAI_CHAT_* in .env), or pass offline metrics only"
        )

    runner = ExtractionRunner(max_workers=max_workers)
    logger.info(
        "operator.evaluate: dimension=extraction metrics=%s language=%s llm=%s",
        selected, language, "enabled" if llm else "offline",
    )
    result = runner.run(data=data, metrics=selected, language=language, llm=llm)
    if llm_meta:
        result.metadata.update(llm_meta)
    # Pairing summary so the caller can see candidate coverage at a glance:
    # how many gold chunks found a candidate, which gold chunks had no
    # candidate (extraction produced nothing /漏抽), and which candidates had
    # no gold to score against. Reported only when non-empty to avoid noise.
    gold_ids = {g.get("sample_id") for g in gold if g.get("sample_id")}
    cand_ids = {c.get("sample_id") for c in candidate if c.get("sample_id")}
    result.metadata["matched_count"] = len(gold_ids & cand_ids)
    missing = sorted(gold_ids - cand_ids)
    extra = sorted(cand_ids - gold_ids)
    if missing:
        result.metadata["missing_candidate_chunks"] = missing
    if extra:
        result.metadata["extra_candidate_chunks"] = extra
    return result
