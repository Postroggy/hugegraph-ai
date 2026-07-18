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

"""Extraction faithfulness — LLM judges whether each extracted item has textual support.

Unlike the F1 metrics, this is a GT-free metric: it only needs the candidate
extraction results and the original input text. The LLM judge checks each
vertex and edge for support in the source document.

Reference: deepeval FaithfulnessMetric (claims-vs-truths NLI pattern),
ragas NLIStatementPrompt (per-statement entailment verdict).
"""

import logging
from typing import Any, Dict, List, Optional

from hugegraph_llm.benchmark.llm_judge.judge_utils import (
    parse_json_response as _parse_json_response,
)
from hugegraph_llm.benchmark.llm_judge.judge_utils import (
    retry_llm_call,
)
from hugegraph_llm.benchmark.llm_judge.prompts import get_prompt
from hugegraph_llm.benchmark.metrics.base import BaseMetric
from hugegraph_llm.benchmark.metrics.extraction import _edge_in, _edge_out
from hugegraph_llm.benchmark.metrics.registry import MetricRegistry

logger = logging.getLogger(__name__)

def _format_item(
    idx: int,
    item: Dict[str, Any],
    item_type: str,
) -> str:
    """Format a single vertex or edge as a prompt line (with source_snippet if present)."""
    if item_type == "vertex":
        label = item.get("label", "")
        name = item.get("name")
        if not name and isinstance(item.get("properties"), dict):
            name = item["properties"].get("name", "")
        line = f'[{idx}] {{"type": "vertex", "label": "{label}", "name": "{name}"}}'
    else:
        # edge
        out_v = str(_edge_out(item) or "?")
        label = str(item.get("label", "") or "?")
        in_v = str(_edge_in(item) or "?")
        line = (
            f'[{idx}] {{"type": "edge", "label": "{label}", '
            f'"source": "{out_v}", "target": "{in_v}"}}'
        )
    snippet = str(item.get("source_snippet") or "").strip()
    if snippet:
        line += f"\n    依据: {snippet}"
    return line


def _compute_extraction_faithfulness(
    llm: Any,
    prediction: Any,
    input_text: str,
    language: str = "en",
) -> Dict[str, Optional[float]]:
    """Core: judge each candidate vertex/edge for faithfulness to input text."""
    if llm is None:
        return {
            "extraction_faithfulness": None,
        }

    # prediction may be a composite dict from the runner:
    # {"vertices": [...], "edges": [...]}
    if isinstance(prediction, dict):
        vertices = prediction.get("vertices", prediction.get("candidate_vertices", []))
        edges = prediction.get("edges", prediction.get("candidate_edges", []))
    elif isinstance(prediction, list):
        vertices = prediction
        edges = []
    else:
        return {
            "extraction_faithfulness": None,
        }

    items: List[str] = []
    idx = 0
    for v in vertices:
        items.append(_format_item(idx, v, "vertex"))
        idx += 1
    for e in edges:
        items.append(_format_item(idx, e, "edge"))
        idx += 1

    if not items:
        return {
            "extraction_faithfulness": 0.0,
        }

    text = input_text or ""
    # docs 模式：无 chunk 原文时，用候选 item 自带的 source_snippet 合并作为核对依据。
    # 顺序保留 + 去重（同一条 snippet 可能被多个 item 引用），避免噪声重复。
    # 注意：input_text 非空时不覆盖 —— baseline 模式的 chunk_text.md 是完整 chunk
    # 原文（含上下文），比 30-100 字的 snippet 判分依据更强，不能被 snippet 拼接降级。
    # 相应地，car_pipeline._input_text_for 在无 chunk_text.md 时必须返回空串，
    # 而不是合成 "doc_name | heading_path | section" 这类元信息串，否则本 fallback
    # 永远不会触发。
    if not text:
        snippets: List[str] = []
        for it in list(vertices) + list(edges):
            s = str(it.get("source_snippet") or "").strip()
            if s and s not in snippets:
                snippets.append(s)
        if snippets:
            text = "\n".join(snippets)

    prompt = get_prompt("EXTRACTION_FAITHFULNESS_PROMPT", language).format(
        input_text=text,
        items="\n".join(items),
    )

    verdicts: List[Dict[str, Any]] = []
    try:
        response = retry_llm_call(llm, prompt)
        data = _parse_json_response(response)
        if data and isinstance(data.get("verdicts"), list):
            verdicts = data["verdicts"]
    except Exception as e:
        logger.warning("Extraction faithfulness judgment failed: %s", e)

    total = len(items)
    # 按 idx 对齐 verdicts（prompt 要求每个 verdict 带 idx）。LLM 可能漏/多/重复
    # 返回，不能假设 verdicts 顺序或数量 == items：必须按 idx 索引，缺判的 item
    # 保守计为不忠实（verdict=0），并对数量不一致告警，避免 faithful/total 失真。
    verdicts_by_idx: Dict[int, Dict[str, Any]] = {}
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        idx = v.get("idx")
        if isinstance(idx, int):
            # 重复 idx 取第一个（LLM 一对多 artefact），其余忽略。
            if idx not in verdicts_by_idx:
                verdicts_by_idx[idx] = v
        else:
            logger.warning(
                "Extraction faithfulness: verdict missing valid idx, skipped: %r", v
            )
    judged = len(verdicts_by_idx)
    if judged != total:
        missing = sum(1 for idx in range(total) if idx not in verdicts_by_idx)
        out_of_range = sum(1 for idx in verdicts_by_idx if idx < 0 or idx >= total)
        logger.warning(
            "Extraction faithfulness: LLM returned %d verdicts for %d items "
            "(%d missing, %d out-of-range); missing items counted as unfaithful "
            "(verdict=0), out-of-range idx ignored.",
            judged,
            total,
            missing,
            out_of_range,
        )

    faithful = sum(
        1
        for idx in range(total)
        if verdicts_by_idx.get(idx, {}).get("verdict") in (1, "1", True)
    )

    score = faithful / total if total > 0 else 0.0

    return {
        "extraction_faithfulness": round(score, 4),
    }


@MetricRegistry.register
class ExtractionFaithfulness(BaseMetric):
    """GT-free faithfulness check: does each extracted item have textual support?

    Uses an LLM judge to check whether each candidate vertex/edge is supported
    by the original input text. This metric does NOT require gold annotations —
    it only needs the candidate extraction and the source document.

    Requires ``llm`` in kwargs and ``input_text`` in kwargs.
    Returns ``None`` when no LLM is available.

    Registered name: ``extraction_faithfulness``
    """

    name: str = "extraction_faithfulness"
    requires_llm: bool = True

    def calculate(
        self,
        prediction: Any,
        reference: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Optional[float]]:
        """Calculate extraction faithfulness.

        Args:
            prediction: Candidate vertices/edges. Accepts either a composite
                       dict `{"vertices": [...], "edges": [...]}` (from the
                       runner) or a flat list of vertices.
            reference: Unused (GT-free metric).
            **kwargs: Must contain ``llm`` and ``input_text``.
                     Optional ``language`` ("en" or "zh").

        Returns:
            Dict with extraction_faithfulness (0-1).
        """
        llm = kwargs.get("llm")
        if llm is None:
            return {
                "extraction_faithfulness": None,
            }

        input_text = kwargs.get("input_text", "")
        language = kwargs.get("language", "en")
        return _compute_extraction_faithfulness(llm, prediction, input_text, language)
