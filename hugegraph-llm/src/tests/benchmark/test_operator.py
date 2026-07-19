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

"""Tests for the benchmark operator (``evaluate``) and the car adapter."""

from unittest.mock import patch

import pytest

from hugegraph_llm.benchmark.datasets.car_pipeline import build_extraction_inputs
from hugegraph_llm.benchmark.operator import (
    SUPPORTED_DIMENSIONS,
    _derive_schema,
    _infer_dimension,
    _infer_primary_key,
    _pair_extraction_samples,
    evaluate,
)

pytestmark = pytest.mark.unit


class _MockLLM:
    """LLM-Judge stand-in returning empty-result JSON (no real API call).

    Returns a JSON object with every key any LLM-Judge metric might look for
    (verdicts / matches / classifications / tp / fp / fn / score / statements /
    facts), all empty/zero. Metrics then score 0 (e.g. empty matches → F1=0)
    rather than None — a None would mean "LLM call failed" and the metric would
    be excluded from overall, which hermetic tests don't want.
    """

    def generate(self, prompt="", messages=None, **kw):
        return (
            '{"verdicts": [], "matches": [], "classifications": [], '
            '"tp": [], "fp": [], "fn": [], "score": 0, '
            '"statements": [], "facts": []}'
        )


class _FailingLLM:
    """LLM-Judge stand-in that always fails (raises ``LLMPermanentError`` so
    ``retry_llm_call`` short-circuits without retrying — keeps the test fast).
    """

    def __init__(self):
        from hugegraph_llm.benchmark.llm_judge.exceptions import LLMPermanentError
        self._err = LLMPermanentError("simulated permanent failure")

    def generate(self, prompt="", messages=None, **kw):
        raise self._err


@pytest.fixture(autouse=True)
def _stub_judge_client():
    """Keep tests hermetic: ``evaluate()`` won't build a real LLM client."""
    with patch(
        "hugegraph_llm.benchmark.operator.create_judge_llm",
        return_value=(None, {}),
    ) as mock:
        yield mock


def _gold_candidate_pair():
    """Minimal benchmark-format gold + candidate (same chunk, exact match)."""
    vertex = {"label": "Component", "name": "制动主缸", "properties": {"comp_name": "制动主缸"}}
    gold = [{"sample_id": "c1", "vertices": [vertex], "edges": [], "input_text": "制动主缸负责制动。"}]
    candidate = [{"sample_id": "c1", "vertices": [dict(vertex)], "edges": []}]
    return gold, candidate


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_non_list_gold():
    with pytest.raises(TypeError, match="must be lists"):
        evaluate("not a list", [], dimension="extraction")


def test_rejects_non_extraction_dimension():
    gold, cand = _gold_candidate_pair()
    with pytest.raises(ValueError, match="only dimension='extraction'"):
        evaluate(gold, cand, dimension="retrieval")


def test_rejects_empty_gold():
    with pytest.raises(ValueError, match="gold must contain"):
        evaluate([], [], dimension="extraction")


def test_rejects_cross_dimension_metrics():
    gold, cand = _gold_candidate_pair()
    with pytest.raises(ValueError, match="span multiple dimensions"):
        evaluate(gold, cand, metrics=["entity_f1", "recall_at_k"])


def test_rejects_non_extraction_metrics():
    gold, cand = _gold_candidate_pair()
    with pytest.raises(ValueError, match="not 'extraction'"):
        evaluate(gold, cand, metrics=["recall_at_k"])


def test_rejects_unknown_metric():
    gold, cand = _gold_candidate_pair()
    with pytest.raises(ValueError, match="unknown metric|does not belong"):
        evaluate(gold, cand, metrics=["entity_f1", "totally_made_up"])


def test_supported_dimensions_extraction_only():
    assert SUPPORTED_DIMENSIONS == ("extraction",)


def test_infer_dimension_maps_generation_to_answer():
    assert _infer_dimension(["token_f1"]) == "answer"
    assert _infer_dimension(["entity_f1"]) == "extraction"


# ---------------------------------------------------------------------------
# Pairing / schema helpers
# ---------------------------------------------------------------------------


def test_pair_missing_candidate_yields_empty_graph():
    gold = [{"sample_id": "c1", "vertices": [{"label": "X", "name": "a", "properties": {}}], "edges": []}]
    samples = _pair_extraction_samples(gold, [])
    assert len(samples) == 1
    assert samples[0]["candidate_vertices"] == []
    assert samples[0]["gold_vertices"] == gold[0]["vertices"]


def test_pair_drops_candidate_without_gold():
    gold = [{"sample_id": "c1", "vertices": [], "edges": []}]
    candidate = [{"sample_id": "c2", "vertices": [{"label": "X", "name": "a", "properties": {}}], "edges": []}]
    samples = _pair_extraction_samples(gold, candidate)
    assert len(samples) == 1
    assert samples[0]["sample_id"] == "c1"


def test_derive_schema_uses_gold_primary_key():
    gold = [
        {
            "sample_id": "c1",
            "vertices": [{"label": "Component", "name": "x", "properties": {"comp_name": "x"}}],
            "edges": [],
        }
    ]
    schema = _derive_schema(gold)
    assert {"name": "Component", "primary_keys": ["comp_name"]} in schema["vertexlabels"]


def test_infer_primary_key_falls_back_to_name():
    vertex = {"label": "X", "name": "a", "properties": {"other": "b"}}
    assert _infer_primary_key(vertex) == "name"


# ---------------------------------------------------------------------------
# evaluate (extraction)
# ---------------------------------------------------------------------------


def test_extraction_offline_metrics():
    gold, cand = _gold_candidate_pair()
    result = evaluate(gold, cand, metrics=["entity_f1", "triple_f1"], language="en")
    assert result.metadata["mode"] == "extraction"
    assert "entity_f1" in result.overall
    assert len(result.samples) == 1


def test_extraction_full_with_mock_llm():
    gold, cand = _gold_candidate_pair()
    result = evaluate(gold, cand, language="en", llm=_MockLLM())
    assert result.metadata["mode"] == "extraction"
    # full default set includes the LLM-judged metrics
    assert "extraction_faithfulness" in result.overall
    assert "semantic_entity_f1" in result.overall


def test_llm_failure_returns_none_not_zero():
    """LLM call failure → LLM metric None (excluded from overall), not 0.

    Failed LLM-Judge metrics must return None so compute_overall skips them,
    not 0 which would pollute the mean. Non-LLM metrics stay scored.
    """
    gold, cand = _gold_candidate_pair()
    result = evaluate(gold, cand, language="en", llm=_FailingLLM())
    # LLM-Judge metrics that actually call the LLM → None, excluded from overall
    assert "extraction_faithfulness" not in result.overall
    assert "semantic_entity_f1" not in result.overall
    # non-LLM metrics still scored normally
    assert "entity_f1" in result.overall
    assert result.overall["entity_f1"] > 0
    # skipped_metrics records the failed LLM metrics
    skipped = result.metadata.get("skipped_metrics", [])
    assert "extraction_faithfulness" in skipped
    assert "semantic_entity_f1" in skipped


def test_llm_failure_reported_in_markdown():
    """LLM failure → markdown report surfaces failure rate + missing metrics."""
    from hugegraph_llm.benchmark.reporters.markdown_reporter import MarkdownReporter

    gold, cand = _gold_candidate_pair()
    result = evaluate(gold, cand, language="en", llm=_FailingLLM())
    report = MarkdownReporter.report(result)
    assert "LLM 调用失败率" in report
    assert "缺失 metric" in report
    assert "extraction_faithfulness" in report  # listed in the missing-metric block


def test_llm_metric_without_client_raises():
    gold, cand = _gold_candidate_pair()
    with pytest.raises(ValueError, match="require a configured LLM client"):
        evaluate(gold, cand, metrics=["semantic_entity_f1"])


def test_injected_llm_skips_client_creation(_stub_judge_client):
    gold, cand = _gold_candidate_pair()
    result = evaluate(
        gold, cand, metrics=["entity_f1", "semantic_entity_f1"], language="en", llm=_MockLLM()
    )
    _stub_judge_client.assert_not_called()
    assert "semantic_entity_f1" in result.overall


# ---------------------------------------------------------------------------
# Pairing summary in metadata (caller-facing coverage info)
# ---------------------------------------------------------------------------


def test_pairing_summary_reports_missing_candidate():
    gold = [
        {"sample_id": "c1", "vertices": [], "edges": []},
        {"sample_id": "c2", "vertices": [], "edges": []},
        {"sample_id": "c3", "vertices": [], "edges": []},
    ]
    candidate = [
        {"sample_id": "c1", "vertices": [], "edges": []},
        {"sample_id": "c2", "vertices": [], "edges": []},
    ]
    result = evaluate(gold, candidate, metrics=["entity_f1"], language="en")
    assert result.metadata["matched_count"] == 2
    assert result.metadata["missing_candidate_chunks"] == ["c3"]
    assert "extra_candidate_chunks" not in result.metadata


def test_pairing_summary_reports_extra_candidate():
    gold = [{"sample_id": "c1", "vertices": [], "edges": []}]
    candidate = [
        {"sample_id": "c1", "vertices": [], "edges": []},
        {"sample_id": "c2", "vertices": [], "edges": []},
    ]
    result = evaluate(gold, candidate, metrics=["entity_f1"], language="en")
    assert result.metadata["matched_count"] == 1
    assert result.metadata["extra_candidate_chunks"] == ["c2"]
    assert "missing_candidate_chunks" not in result.metadata


# ---------------------------------------------------------------------------
# car adapter (build_extraction_inputs)
# ---------------------------------------------------------------------------


def test_adapter_converts_car_to_benchmark():
    gold_car = [
        {
            "chunk_id": "c1",
            "vertices": [{"type": "Component", "name": "制动主缸", "properties": {"comp_name": "制动主缸"}}],
            "edges": [],
        }
    ]
    cand_car = [
        {
            "chunk_id": "c1",
            "vertices": [{"type": "Component", "name": "制动主缸", "properties": {"comp_name": "制动主缸"}}],
            "edges": [],
        }
    ]
    gold_list, cand_list = build_extraction_inputs(gold_car, cand_car, chunk_texts={"c1": "原文"})
    assert gold_list[0]["sample_id"] == "c1"
    assert gold_list[0]["vertices"][0]["label"] == "Component"  # type → label
    assert gold_list[0]["input_text"] == "原文"
    assert cand_list[0]["sample_id"] == "c1"
    assert "input_text" not in cand_list[0]


def test_adapter_then_evaluate_end_to_end():
    """car adapter → operator (offline metrics), the intended call shape."""
    gold_car = [
        {
            "chunk_id": "c1",
            "vertices": [{"type": "Component", "name": "制动主缸", "properties": {"comp_name": "制动主缸"}}],
            "edges": [],
        }
    ]
    cand_car = [
        {
            "chunk_id": "c1",
            "vertices": [{"type": "Component", "name": "制动主缸", "properties": {"comp_name": "制动主缸"}}],
            "edges": [],
        }
    ]
    gold_list, cand_list = build_extraction_inputs(
        gold_car, cand_car, chunk_texts={"c1": "制动主缸负责制动。"}
    )
    result = evaluate(gold_list, cand_list, metrics=["entity_f1"], language="zh")
    assert result.metadata["mode"] == "extraction"
    assert "entity_f1" in result.overall
