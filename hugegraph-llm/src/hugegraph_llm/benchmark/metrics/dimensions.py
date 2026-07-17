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

"""Metric → dimension classification for analytical reporting.

Every metric belongs to one top-level benchmark domain (extraction /
retrieval / generation) and one fine-grained sub-dimension. The reporter
uses this mapping to roll changes up from "metric-level noise" to
"dimension-level signal" — e.g. "relation-extraction regressed" is far
more legible than "triple_f1 -0.07".

This module is purely a presentation aid (dimension labels for grouping).
It does NOT influence regression verdicts — every metric is judged the
same way in ``compare()``, regardless of its dimension. Unknown metrics
fall back to ``("Other", "Other")`` so new metrics still render.
"""

from typing import Tuple

# (top-level domain, sub-dimension)
#
# Top-level domains correspond to the three benchmark scenarios:
#   extraction — knowledge-graph construction (entity/relation/property/schema)
#   retrieval  — context recall for answering
#   generation — answer quality in the ablation runner
_METRIC_DIMENSIONS: dict[str, Tuple[str, str]] = {
    # --- extraction: entity ---
    "entity_precision": ("extraction", "实体识别"),
    "entity_recall": ("extraction", "实体识别"),
    "entity_f1": ("extraction", "实体识别"),
    # --- extraction: relation ---
    "triple_precision": ("extraction", "关系抽取"),
    "triple_recall": ("extraction", "关系抽取"),
    "triple_f1": ("extraction", "关系抽取"),
    # --- extraction: property ---
    "property_precision": ("extraction", "属性抽取"),
    "property_recall": ("extraction", "属性抽取"),
    "property_f1": ("extraction", "属性抽取"),
    # --- extraction: schema compliance ---
    "schema_validity": ("extraction", "Schema 合规"),
    "type_constraint_pass": ("extraction", "Schema 合规"),
    "required_property_fill": ("extraction", "Schema 合规"),
    "illegal_edge_rate": ("extraction", "Schema 合规"),
    # --- extraction: graph structure ---
    "graph_structure": ("extraction", "图结构"),
    "density": ("extraction", "图结构"),
    "clustering_coefficient": ("extraction", "图结构"),
    "largest_component_ratio": ("extraction", "图结构"),
    "num_nodes": ("extraction", "图结构"),
    "num_edges": ("extraction", "图结构"),
    "num_components": ("extraction", "图结构"),
    # --- extraction: syntax / conflict / temporal / load / semantic ---
    "syntax_validity": ("extraction", "语法/冲突/时序"),
    "json_parse_rate": ("extraction", "语法/冲突/时序"),
    "conflict_detection": ("extraction", "语法/冲突/时序"),
    "conflict_rate": ("extraction", "语法/冲突/时序"),
    "num_conflicts": ("extraction", "语法/冲突/时序"),
    "temporal_validity": ("extraction", "语法/冲突/时序"),
    "temporal_valid_rate": ("extraction", "语法/冲突/时序"),
    "num_temporal_attrs": ("extraction", "语法/冲突/时序"),
    "load_to_db_success": ("extraction", "语法/冲突/时序"),
    # --- extraction: LLM-based semantic metrics ---
    # semantic entity/triple sit under 实体识别/关系抽取 (same axis as their
    # exact-match counterparts); faithfulness gets its own sub-dimension
    # ``抽取忠实度`` — it is a GT-free "does this claim have textual
    # support?" check, not a candidate-vs-gold match, so grouping it
    # under a generic "语义匹配" label was confusing.
    "semantic_entity_precision": ("extraction", "实体识别"),
    "semantic_entity_recall": ("extraction", "实体识别"),
    "semantic_entity_f1": ("extraction", "实体识别"),
    "semantic_triple_precision": ("extraction", "关系抽取"),
    "semantic_triple_recall": ("extraction", "关系抽取"),
    "semantic_triple_f1": ("extraction", "关系抽取"),
    "extraction_faithfulness": ("extraction", "抽取忠实度"),
    # --- retrieval ---
    "recall_at_k": ("retrieval", "召回"),
    "hit_at_k": ("retrieval", "命中"),
    "mrr": ("retrieval", "排序"),
    "context_precision": ("retrieval", "上下文质量"),
    "context_relevancy": ("retrieval", "上下文质量"),
    "evidence_recall_llm": ("retrieval", "上下文质量"),
    # retrieval: metric variants produced by some runners (hit_any@k /
    # hit_all@k / recall@k). Listed explicitly because the prefix fallback
    # below keys on ``recall`` without the ``@`` suffix.
    "recall@1": ("retrieval", "召回"),
    "recall@5": ("retrieval", "召回"),
    "recall@10": ("retrieval", "召回"),
    "hit_any@1": ("retrieval", "命中"),
    "hit_any@5": ("retrieval", "命中"),
    "hit_any@10": ("retrieval", "命中"),
    "hit_all@1": ("retrieval", "命中"),
    "hit_all@5": ("retrieval", "命中"),
    "hit_all@10": ("retrieval", "命中"),
    # --- generation ---
    "token_f1": ("generation", "词面匹配"),
    "exact_match": ("generation", "词面匹配"),
    "rouge_l": ("generation", "词面匹配"),
    "answer_correctness": ("generation", "语义正确"),
    "faithfulness": ("generation", "语义正确"),
    "coverage": ("generation", "覆盖度"),
}

# Prefix-based fallback so newly added metrics in a known family still
# resolve to the right sub-dimension without an explicit entry.
_PREFIX_FALLBACK: Tuple[Tuple[str, Tuple[str, str]], ...] = (
    ("entity_", ("extraction", "实体识别")),
    ("triple_", ("extraction", "关系抽取")),
    ("property_", ("extraction", "属性抽取")),
    ("recall", ("retrieval", "召回")),
)


def get_dimension(metric_name: str) -> Tuple[str, str]:
    """Return ``(top_level_domain, sub_dimension)`` for a metric.

    Falls back to prefix matching, then to ``("Other", "Other")`` so
    unknown metrics still render rather than disappearing from the report.
    """
    exact = _METRIC_DIMENSIONS.get(metric_name)
    if exact is not None:
        return exact
    for prefix, dim in _PREFIX_FALLBACK:
        if metric_name.startswith(prefix):
            return dim
    return ("Other", "Other")


def domain_label(domain: str) -> str:
    """Map an internal domain key to a human-readable label."""
    return {
        "extraction": "图提取",
        "retrieval": "检索",
        "generation": "生成回答",
        "Other": "其他",
    }.get(domain, domain)


# Short human-readable description per score — 含义 only. The computation
# method (EM / LLM / Schema 检查) is conveyed by the report's table labels
# and the 关于本次测评 section, not repeated in each description. Unknown
# scores fall back to "" so the report still renders for new metrics.
_METRIC_DESCRIPTIONS: dict[str, str] = {
    # entity — EM
    "entity_precision": "候选实体命中 gold 占比",
    "entity_recall": "gold 实体被召回占比",
    "entity_f1": "实体 F1（P/R 调和）",
    # triple — EM
    "triple_precision": "候选三元组命中 gold 占比",
    "triple_recall": "gold 三元组被召回占比",
    "triple_f1": "关系 F1（P/R 调和）",
    # property — EM
    "property_precision": "候选属性命中 gold 占比",
    "property_recall": "gold 属性被召回占比",
    "property_f1": "属性 F1（P/R 调和）",
    # schema — Schema 检查（合规校验，非匹配）
    "schema_validity": "Schema 合规综合",
    "type_constraint_pass": "类型落在 gold schema 内占比",
    "required_property_fill": "主键属性填充完整度",
    "illegal_edge_rate": "违反 schema 的边占比",
    # semantic — LLM
    "semantic_entity_precision": "候选实体语义命中 gold 占比",
    "semantic_entity_recall": "gold 实体被语义召回占比",
    "semantic_entity_f1": "实体语义 F1（P/R 调和）",
    "semantic_triple_precision": "候选三元组语义命中 gold 占比",
    "semantic_triple_recall": "gold 三元组被语义召回占比",
    "semantic_triple_f1": "三元组语义 F1（P/R 调和）",
    # LLM judge
    "extraction_faithfulness": "抽取忠实度",
    # retrieval
    "recall_at_k": "Top-K 召回率",
    "hit_at_k": "Top-K 命中率",
    "mrr": "平均倒数排名",
    # generation — EM
    "token_f1": "答案 token F1",
    "exact_match": "完全匹配率",
    "rouge_l": "ROUGE-L",
}


def metric_description(metric_name: str) -> str:
    """Return a short Chinese description for a score, or ``""`` if unknown."""
    return _METRIC_DESCRIPTIONS.get(metric_name, "")


# Display order for sub-dimensions within a report, following the natural
# extraction pipeline (entity -> relation -> property -> schema -> ...).
# Sub-dimensions not listed sort last, then alphabetically.
_SUBDIM_DISPLAY_ORDER: dict[str, int] = {
    "实体识别": 0,
    "关系抽取": 1,
    "属性抽取": 2,
    "Schema 合规": 3,
    "图结构": 4,
    "语法/冲突/时序": 5,
    "抽取忠实度": 6,
}


def subdim_order(subdim: str) -> int:
    """Return the display priority for a sub-dimension (lower = earlier)."""
    return _SUBDIM_DISPLAY_ORDER.get(subdim, 99)
