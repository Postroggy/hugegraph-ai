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

"""Semantic entity F1 via LLM-based semantic matching.

Unlike :class:`EntityF1` which uses exact (label, name) matching, this metric
uses an LLM judge to determine whether candidate entities are *semantically*
equivalent to gold entities — allowing for synonym normalization, abbreviation
expansion, and phrasing variation (e.g. "制动液" ↔ "制动液检查/更换").

Reference: car33 评分规则.md §4.1 (entity normalization rules),
ragas ContextEntityRecall (LLM entity extraction pattern).
"""

import json
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
from hugegraph_llm.benchmark.metrics.registry import MetricRegistry

logger = logging.getLogger(__name__)

def _format_vertex(idx: int, vertex: Dict[str, Any]) -> str:
    """Format a single vertex as an indexed prompt line (``idx`` is the global index)."""
    label = vertex.get("label", "")
    name = vertex.get("name")
    if not name and isinstance(vertex.get("properties"), dict):
        name = vertex["properties"].get("name", "")
    name = str(name or "")
    return f"[{idx}] {{\"label\": \"{label}\", \"name\": \"{name}\"}}"


def _compute_semantic_entity_pr_f1(
    llm: Any,
    prediction: List[Dict[str, Any]],
    reference: List[Dict[str, Any]],
    language: str = "en",
) -> Dict[str, Optional[float]]:
    """Core: call LLM to match entities, then compute precision/recall/F1.

    Returns None values when no LLM is available.
    """
    if llm is None:
        return {
            "semantic_entity_precision": None,
            "semantic_entity_recall": None,
            "semantic_entity_f1": None,
            "semantic_entity_redundancy": None,
        }

    if not prediction or not reference:
        return {
            "semantic_entity_precision": 0.0,
            "semantic_entity_recall": 0.0,
            "semantic_entity_f1": 0.0,
            "semantic_entity_redundancy": None,
        }

    # 按 label 分桶后逐桶 LLM 匹配（保留全局 idx）。匹配规则本就要求 label
    # 相等，分桶无损（不裁切数据，每个实体都进 LLM）；避免一次性 O(n*m) prompt
    # 在大 chunk（100+ 实体）时让 LLM hallucination。每桶只含同 label 的
    # gold/candidate，prompt 小、判断准。matches 的 idx 始终是全局 idx，桶间
    # 合并后 dedup/计数与原逻辑一致。
    gold_by_label: Dict[str, List[tuple]] = {}
    cand_by_label: Dict[str, List[tuple]] = {}
    for i, v in enumerate(reference):
        gold_by_label.setdefault(str(v.get("label", "")), []).append((i, v))
    for i, v in enumerate(prediction):
        cand_by_label.setdefault(str(v.get("label", "")), []).append((i, v))

    matches: List[List[int]] = []
    attempts = 0
    successes = 0
    for label, gold_bucket in gold_by_label.items():
        cand_bucket = cand_by_label.get(label)
        if not cand_bucket:
            continue  # 该 label 无 candidate，无匹配可判
        attempts += 1
        # 桶内用局部 idx（0-based）给 LLM，符合 prompt example 的索引习惯；
        # LLM 返回 [[local_cand, local_gold], ...]，这里映射回全局 idx
        # （gold_bucket/cand_bucket 每项是 (global_idx, vertex)）。
        gold_lines = [_format_vertex(li, v) for li, (_gi, v) in enumerate(gold_bucket)]
        cand_lines = [_format_vertex(li, v) for li, (_ci, v) in enumerate(cand_bucket)]
        prompt = get_prompt("ENTITY_SEMANTIC_MATCH_PROMPT", language).format(
            gold_entities="\n".join(gold_lines),
            candidate_entities="\n".join(cand_lines),
        )
        try:
            response = retry_llm_call(llm, prompt)
            data = _parse_json_response(response)
            if data and isinstance(data.get("matches"), list):
                for m in data["matches"]:
                    if not (isinstance(m, list) and len(m) == 2):
                        continue
                    lc, lg = m[0], m[1]
                    if (
                        isinstance(lc, int)
                        and isinstance(lg, int)
                        and 0 <= lc < len(cand_bucket)
                        and 0 <= lg < len(gold_bucket)
                    ):
                        matches.append([cand_bucket[lc][0], gold_bucket[lg][0]])
            successes += 1
        except Exception as e:
            logger.warning("Semantic entity matching failed for label=%s: %s", label, e)

    # 全部 label 的 LLM 调用都失败 → 返回 None（避免 0 污染均值；部分失败
    # 仍按保守计分——失败桶 matched=0，成功桶照常算）。
    if attempts > 0 and successes == 0:
        return {
            "semantic_entity_precision": None,
            "semantic_entity_recall": None,
            "semantic_entity_f1": None,
            "semantic_entity_redundancy": None,
        }

    # Enforce 1:1 matching (each candidate/gold at most once). The LLM may
    # return duplicate or many-to-one pairs, which would let matched exceed
    # gold/cand counts and inflate recall/precision above 1.0.
    #
    # NOTE: dedup is greedy in LLM output order — the first pair claiming a
    # candidate/gold wins. Without per-pair confidence we can't do optimal
    # assignment (e.g. Hungarian), so matched/redundancy depend on the order
    # the LLM emits pairs. Acceptable for a diagnostic metric; revisit if
    # confidence scores become available.
    seen_cand = set()
    seen_gold = set()
    deduped: List[List[int]] = []
    redundant_cands = set()  # candidates whose target gold was already taken
    for m in matches:
        c, g = m[0], m[1]
        if c in seen_cand or g in seen_gold:
            # Dropped pair. A candidate reaching for a gold already claimed by
            # another candidate is a semantic duplicate of that winner
            # (extractor over-split). A candidate already matched elsewhere
            # (c in seen_cand) is an LLM one-to-many artefact, not extractor
            # redundancy — excluded from the redundancy numerator.
            if g in seen_gold and c not in seen_cand:
                redundant_cands.add(c)
            continue
        seen_cand.add(c)
        seen_gold.add(g)
        deduped.append(m)
    matches = deduped

    gold_count = len(reference)
    cand_count = len(prediction)
    matched = len(matches)
    # Redundancy: share of candidates that are semantic duplicates — each
    # pointed at a gold already claimed by another candidate, so 1:1 dedup
    # dropped it. High redundancy => the extractor emitted many duplicate
    # entities. Counts distinct candidates (not dropped pairs) so a single
    # candidate matched to several golds can't inflate it.
    redundancy = len(redundant_cands) / cand_count if cand_count > 0 else 0.0

    precision = matched / cand_count if cand_count > 0 else 0.0
    recall = matched / gold_count if gold_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "semantic_entity_precision": round(precision, 4),
        "semantic_entity_recall": round(recall, 4),
        "semantic_entity_f1": round(f1, 4),
        "semantic_entity_redundancy": round(redundancy, 4),
    }


@MetricRegistry.register
class SemanticEntityF1(BaseMetric):
    """Entity-level F1 using LLM-based semantic matching.

    Unlike :class:`EntityF1` (exact string match), this metric asks an LLM
    judge to determine semantic equivalence between candidate and gold
    entities, allowing synonym normalization, abbreviation expansion, and
    phrasing variation.

    Requires ``llm`` in kwargs. Returns ``None`` for all scores when no
    LLM is available (offline mode).

    Registered name: ``semantic_entity_f1``
    """

    name: str = "semantic_entity_f1"
    requires_llm: bool = True

    @classmethod
    def is_higher_is_better(cls, score_name: str) -> Optional[bool]:
        """Declare direction for this metric's scores.

        ``semantic_entity_redundancy`` is lower-is-better and must be declared
        explicitly — the registry defaults to higher-is-better, which would
        invert its direction in compare-mode regression detection. The F1 /
        precision / recall scores return ``None`` so they fall through to the
        registry default (higher-is-better).
        """
        if score_name == "semantic_entity_redundancy":
            return False
        return None

    def calculate(
        self,
        prediction: Any,
        reference: Any,
        **kwargs: Any,
    ) -> Dict[str, Optional[float]]:
        """Calculate semantic entity precision, recall, and F1.

        Args:
            prediction: List of candidate vertex dicts.
            reference: List of gold vertex dicts.
            **kwargs: Must contain ``llm``. Optional ``language`` ("en" or "zh").

        Returns:
            Dict with semantic_entity_precision, semantic_entity_recall,
            semantic_entity_f1.
        """
        llm = kwargs.get("llm")
        if llm is None:
            return {
                "semantic_entity_precision": None,
                "semantic_entity_recall": None,
                "semantic_entity_f1": None,
                "semantic_entity_redundancy": None,
            }

        pred = prediction if isinstance(prediction, list) else []
        ref = reference if isinstance(reference, list) else []
        language = kwargs.get("language", "en")
        return _compute_semantic_entity_pr_f1(llm, pred, ref, language)
