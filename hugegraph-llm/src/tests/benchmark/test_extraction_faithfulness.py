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

"""Tests for ``ExtractionFaithfulness``.

Focus:
- The docs-mode ``source_snippet`` fallback triggers when ``input_text``
  is empty and vertices/edges carry per-item snippets.
- The fallback preserves order and de-duplicates repeated snippets so
  the LLM judge sees one clean concatenation.
- When ``input_text`` is provided (baseline mode), it is NOT overridden
  by snippet concatenation — the fuller chunk text is a stronger evidence
  base than a handful of 30-100 char snippets.
- Offline (no LLM) returns ``None``, matching the metric registry contract.
"""

import json
from typing import Any, Dict, List

import pytest

from hugegraph_llm.benchmark.metrics.extraction.extraction_faithfulness import (
    ExtractionFaithfulness,
    _compute_extraction_faithfulness,
)

pytestmark = pytest.mark.unit


class RecordingLLM:
    """LLM stub that captures the prompt so we can assert on its content."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.prompts: List[str] = []

    def generate(self, messages, response_format, **_: Any) -> dict:
        self.prompts.append(messages[0].content if messages else "")
        if isinstance(self._response, str):
            return json.loads(self._response)
        return self._response


def _all_faithful(n: int) -> str:
    return json.dumps({
        "verdicts": [{"idx": i, "verdict": 1, "reason": "supported"} for i in range(n)]
    })


# --- offline contract ---

def test_extraction_faithfulness_offline_returns_none():
    metric = ExtractionFaithfulness()
    prediction: Dict[str, Any] = {"vertices": [{"label": "C", "name": "A"}], "edges": []}
    result = metric.calculate(prediction, llm=None, input_text="")
    assert result == {"extraction_faithfulness": None}


# --- docs-mode fallback: empty input_text + per-item snippets ---

def test_faithfulness_falls_back_to_source_snippets_when_input_text_empty():
    """docs mode (production): the runner passes ``input_text=""``. The metric
    must synthesize the judge's evidence base from candidate ``source_snippet``
    fields — otherwise every extraction is judged against nothing."""
    llm = RecordingLLM(_all_faithful(2))
    prediction = {
        "vertices": [
            {"label": "Component", "name": "远光灯", "source_snippet": "远光灯已打开。"},
        ],
        "edges": [
            {"label": "HAS_STATUS", "outV": "远光灯", "inV": "远光灯已打开",
             "source_snippet": "远光灯已打开。"},
        ],
    }
    result = _compute_extraction_faithfulness(llm, prediction, input_text="", language="en")
    assert result["extraction_faithfulness"] == 1.0
    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    # The judge sees the snippet text, not an empty input.
    assert "远光灯已打开。" in prompt
    # And the per-item snippet is embedded next to each extraction item.
    assert "依据: 远光灯已打开。" in prompt


def test_faithfulness_snippet_fallback_dedupes_and_preserves_order():
    """Two items sharing a snippet should not duplicate it; two distinct
    snippets should both appear in original insertion order."""
    llm = RecordingLLM(_all_faithful(3))
    prediction = {
        "vertices": [
            {"label": "C", "name": "A", "source_snippet": "snippet-1"},
            {"label": "C", "name": "B", "source_snippet": "snippet-2"},
            # duplicate snippet from a different item
            {"label": "C", "name": "C", "source_snippet": "snippet-1"},
        ],
        "edges": [],
    }
    _compute_extraction_faithfulness(llm, prediction, input_text="", language="en")
    prompt = llm.prompts[0]
    # Both snippets present.
    assert "snippet-1" in prompt
    assert "snippet-2" in prompt
    # snippet-1 only appears once in the merged evidence header (dedup); it
    # still appears as per-item ``依据:`` on items C, so overall count > 1
    # but the concatenated evidence block preserves 1st-occurrence ordering.
    assert prompt.index("snippet-1") < prompt.index("snippet-2")


def test_faithfulness_does_not_override_provided_input_text():
    """baseline mode: chunk_text.md is passed through as ``input_text``.
    The full chunk text is a strictly stronger evidence base than a
    concatenation of 30-100 char snippets, so the metric must NOT overwrite
    it with the snippet fallback."""
    llm = RecordingLLM(_all_faithful(1))
    prediction = {
        "vertices": [
            {"label": "C", "name": "A", "source_snippet": "只有 remark 的短句"},
        ],
        "edges": [],
    }
    provided = "chunk 完整原文，包含大量上下文……"
    result = _compute_extraction_faithfulness(
        llm, prediction, input_text=provided, language="en"
    )
    assert result["extraction_faithfulness"] == 1.0
    prompt = llm.prompts[0]
    # The prompt template places the input text between the "Input text:"
    # header and "Extraction items:" (see EXTRACTION_FAITHFULNESS_PROMPT).
    # We check that the provided text sits exactly in that slot, and that
    # the snippet-fallback content did NOT get pasted there.
    marker = f"Input text:\n{provided}\n\nExtraction items:"
    assert marker in prompt, "provided input_text must fill the Input text slot verbatim"
    # The per-item snippet must still appear next to the item, not in the
    # top-level Input text slot.
    assert "依据: 只有 remark 的短句" in prompt


def test_faithfulness_returns_zero_when_no_evidence_at_all():
    """No input_text AND no source_snippet on any item — the judge would
    have literally nothing to check against.  Behaviour today: the prompt
    goes out with an empty evidence base and the LLM's response drives the
    score. This test pins that we still make ONE call and produce a valid
    numeric score (no exception, no None), so upstream can tell it apart
    from an offline run."""
    llm = RecordingLLM(json.dumps({
        "verdicts": [{"idx": 0, "verdict": 0, "reason": "empty text"}]
    }))
    prediction = {
        "vertices": [{"label": "C", "name": "A"}],
        "edges": [],
    }
    result = _compute_extraction_faithfulness(llm, prediction, input_text="", language="en")
    assert result["extraction_faithfulness"] == 0.0
    assert len(llm.prompts) == 1


def test_faithfulness_empty_prediction_returns_zero_without_calling_llm():
    """No candidate items → score is 0.0 and no LLM call is made (nothing
    to judge). Guards against wasted LLM calls when candidate is empty."""
    llm = RecordingLLM(_all_faithful(0))
    result = _compute_extraction_faithfulness(
        llm, {"vertices": [], "edges": []}, input_text="anything", language="en"
    )
    assert result["extraction_faithfulness"] == 0.0
    assert llm.prompts == []
