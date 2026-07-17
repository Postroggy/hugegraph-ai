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

"""Markdown reporter for benchmark results.

Report layout (inverted-pyramid, designed for PR/Issue comments):

  1. 概览 (TL;DR)       — analyst-style summary, no BLOCK verdict
  2. 分析               — programmatic roll-up: domain / sub-dimension /
                          question-type clustering / concentration
  3. 指标总览            — only changed metrics in a table; flat ones folded
  4. 退化样例 / 改进样例  — per-sample rows sorted by severity
  5. 证据层              — failures + full metrics + metadata, all folded

The compare-mode report is driven by ``ComparisonResult.analyze()``; the
single-run report reuses the same section scaffolding without comparison.
"""

from typing import Any, Dict, List, Optional, Tuple

# Import metrics to trigger self-registration before querying directions.
from hugegraph_llm.benchmark import metrics  # noqa: F401
from hugegraph_llm.benchmark.baseline.compare import ComparisonResult
from hugegraph_llm.benchmark.metrics.dimensions import (
    domain_label,
    get_dimension,
    metric_description,
    subdim_order,
)
from hugegraph_llm.benchmark.metrics.registry import MetricRegistry
from hugegraph_llm.benchmark.models.result import BenchmarkResult

_VERDICT_SYMBOL = {"regressed": "🔴", "improved": "🟢", "unchanged": "—"}
_VERDICT_LABEL = {"regressed": "退化", "improved": "改进", "unchanged": "持平"}

# Mode → plain-language explainer bullets for single-run reports, so a reader
# landing on the report understands the task, the data, and how to read the
# numbers without external context.
_MODE_ABOUT: Dict[str, List[str]] = {
    "extraction": [
        "**任务**：图提取（extraction）—— 比较 candidate 抽取的图与 gold 标注的标准图，评估知识图谱抽取质量。",
        "**数据**：按 chunk 配对；gold = 人工标注，candidate = 被测系统输出。",
        "**评分方式**（各指标「说明」列标注了计算方式标签，含义如下）：",
        "  - **EM**（Exact Match）：字符串精确匹配，归一化后字面相等才算命中 —— 用于实体 / 关系 / 属性",
        "  - **LLM**（LLM as Judge）：模型判定语义等价 / 是否无幻觉 —— 用于语义 F1、抽取忠实度",
        "  - **Schema 检查**：校验类型 / 属性 / 边端点是否符合 gold 推导的 schema —— 用于 Schema 合规（独立于匹配的合规性校验）",
        "**如何读数**：",
        "  - 分数已映射到 0–100（满分 100）",
        "  - 「方向」列：↑ 越高越好、↓ 越低越好",
        "  - EM 下「发动机」vs「引擎」算未命中，F1 偏低不代表抽错，可能是同义表述差异（LLM 类指标能补救）",
    ],
}


def _fmt(value: float) -> str:
    """Format a 0-1 score as a 0-100 percentage (2 decimals)."""
    return f"{value * 100:.2f}"


def _fmt_delta(value: float) -> str:
    """Format a 0-1 delta as a signed 0-100 percentage."""
    return f"+{value * 100:.2f}" if value > 0 else f"{value * 100:.2f}"


# Splits a sub-dimension's metrics into separate tables by computation method
# (EM vs LLM-judged semantic F1), so e.g. 实体识别 shows an EM table and an
# "语义 F1（LLM）" table side by side.
_METHOD_ORDER: Dict[str, int] = {"EM": 0, "规则匹配": 1, "语义 F1（LLM）": 2}

# Fixed column order for the evidence-layer wide table: group by sub-dimension
# (entity -> relation -> schema -> faithfulness), EM before LLM within each.
_WIDE_TABLE_ORDER = [
    "entity_f1", "entity_precision", "entity_recall",
    "semantic_entity_f1", "semantic_entity_precision", "semantic_entity_recall", "semantic_entity_redundancy",
    "triple_f1", "triple_precision", "triple_recall",
    "semantic_triple_f1", "semantic_triple_precision", "semantic_triple_recall", "semantic_triple_redundancy",
    "type_constraint_pass", "required_property_fill", "illegal_edge_rate",
    "extraction_faithfulness",
]


def _metric_method_group(metric: str) -> str:
    """Return the computation-method group used for splitting tables.

    ``extraction_faithfulness`` is a sub-dimension of its own (抽取忠实度),
    so no method label is needed and it stays under a single-table section.
    """
    if metric.startswith("semantic_"):
        return "语义 F1（LLM）"
    if metric in {"type_constraint_pass", "required_property_fill", "illegal_edge_rate", "schema_validity"}:
        return "规则匹配"
    return "EM"


def _direction_symbol(metric_name: str) -> str:
    return "↑" if MetricRegistry.is_higher_is_better(metric_name) else "↓"


def _direction_label(metric_name: str) -> str:
    """Human-readable direction, e.g. '↑ 越高越好' / '↓ 越低越好'."""
    return "↑ 越高越好" if MetricRegistry.is_higher_is_better(metric_name) else "↓ 越低越好"


# ---------------------------------------------------------------------------
# Section 1 — 概览 (TL;DR)
# ---------------------------------------------------------------------------


def _section_overview(
    result: BenchmarkResult,
    analysis: Optional[Dict[str, Any]],
    comparison: Optional[ComparisonResult] = None,
) -> List[str]:
    """Analyst-style overview. States what happened, never whether to merge."""
    lines: List[str] = ["## 📊 概览", ""]

    if analysis is None:
        # Single-run: headline numbers + input-data context so the reader
        # knows what was evaluated and where the data came from.
        lines.append(f"- 样例数：{len(result.samples)}")
        lines.append(f"- 指标数：{sum(1 for m in result.overall if not m.endswith('_redundancy'))}")
        data_path = result.metadata.get("data_path")
        data_meta = result.metadata.get("data_meta") or {}
        if data_path:
            lines.append(f"- 数据文件：`{data_path}`")
        # Prefer the split gold_source / candidate_source pair (car_pipeline
        # emits both); fall back to the legacy single ``source`` field for
        # older dataset adapters.
        gold_source = data_meta.get("gold_source")
        candidate_source = data_meta.get("candidate_source")
        if gold_source:
            lines.append(f"- Gold 数据：`{gold_source}`")
        if candidate_source and candidate_source != gold_source:
            lines.append(f"- Candidate 数据：`{candidate_source}`")
        if not (gold_source or candidate_source) and data_meta.get("source"):
            lines.append(f"- 数据来源：`{data_meta['source']}`")
        if data_meta.get("candidate"):
            lines.append(f"- 评测对象：{data_meta['candidate']}")
        if data_meta.get("gold_count") is not None:
            lines.append(
                f"- gold/candidate 配对：{data_meta.get('matched_count')}/{data_meta.get('gold_count')}"
            )
            missing = data_meta.get("missing_candidate_chunks") or []
            empty = data_meta.get("empty_candidate_chunks") or []
            if missing:
                lines.append(f"  - 缺 candidate 文件（{len(missing)}）：{', '.join(missing)}")
            if empty:
                lines.append(f"  - candidate 空图（{len(empty)}）：{', '.join(empty)}")
        # candidate 冗余率（LLM 判定的语义重复占比，反映抽取是否过碎）
        ent_red = result.overall.get("semantic_entity_redundancy")
        tri_red = result.overall.get("semantic_triple_redundancy")
        if ent_red is not None or tri_red is not None:
            parts = []
            if ent_red is not None:
                parts.append(f"实体 {_fmt(ent_red)}")
            if tri_red is not None:
                parts.append(f"三元组 {_fmt(tri_red)}")
            lines.append(f"- candidate 平均冗余率：{' / '.join(parts)}")
        errors = result.metadata.get("error_count") or len(result.metadata.get("errors", []))
        if errors:
            lines.append(f"- 失败样例：{errors}")
        lines.append("")
        return lines

    counts = analysis["counts"]
    total = sum(counts.values())
    lines.append(
        f"- 指标变化：{counts['regressed']} 退化 / {counts['improved']} 改进 / "
        f"{counts['unchanged']} 持平（共 {total}）"
    )

    # Per-domain one-liners, only for domains that actually moved.
    domain_lines: List[str] = []
    for domain, slot in sorted(analysis["by_domain"].items()):
        if slot["regressed"] == 0 and slot["improved"] == 0:
            continue
        parts = []
        if slot["regressed"]:
            parts.append(f"{slot['regressed']} 退化")
        if slot["improved"]:
            parts.append(f"{slot['improved']} 改进")
        worst = slot["worst_delta"]
        tail = f"，最严重 { _fmt_delta(worst)}" if worst < 0 else ""
        domain_lines.append(f"- {domain_label(domain)}：{' / '.join(parts)}{tail}")
    lines.extend(domain_lines)

    # Sample-level headline. Report both directions explicitly so an empty
    # 🟢 side is not ambiguous with "collapsed and hidden".
    n_reg = analysis["concentration"]["regressed_samples"]
    n_imp = len(comparison.improved_samples) if comparison is not None else 0
    lines.append(f"- 样例级：{n_reg} 个退化样例 / {n_imp} 个改进样例")
    lines.append("")
    return lines


def _section_about(result: BenchmarkResult) -> List[str]:
    """Mode-specific explainer so a reader understands the task at a glance.

    Rendered for both single-run and compare reports: tells the reader what
    the evaluation does, how the data is structured, and how to read the
    numbers — without requiring external context.
    """
    mode = result.metadata.get("mode", "")
    bullets = _MODE_ABOUT.get(mode)
    if not bullets:
        return []
    lines = ["## 关于本次测评", ""]
    for b in bullets:
        # Bullets already carrying a list marker (incl. nested "  - ") are
        # emitted as-is so sub-bullets keep their indentation.
        if b.lstrip().startswith("- "):
            lines.append(b)
        else:
            lines.append(f"- {b}")
    lines.append("")
    return lines


def _short_commit(value: Any) -> str:
    """Render a git commit hash short (7 chars) for compact tables."""
    s = str(value or "")
    return s[:7] if s and s != "N/A" else "—"


def _run_ctx_row(label: str, base_val: Any, cand_val: Any) -> str:
    """Row for the Baseline / Candidate side-by-side context table.

    Marks a mismatch with `⚠` so the reader spots data/model swaps that
    could confound the diff. ``None``/missing sides render as "—".
    Timestamps deliberately don't get the ⚠ marker — two runs never
    finish at exactly the same time, so it would be pure noise.
    """
    b = "—" if base_val in (None, "", "N/A") else str(base_val)
    c = "—" if cand_val in (None, "", "N/A") else str(cand_val)
    show_warn = label not in _NO_WARN_ROWS
    diff = " ⚠ 不一致" if show_warn and b != c and b != "—" and c != "—" else ""
    return f"| {label} | {b} | {c}{diff} |"


# Rows in 运行上下文 that always differ between two runs — muting the
# ⚠ marker here keeps the useful signal (Gold dataset / commit / model
# swaps) from being drowned out.
# - ``运行时间``: never matches by definition.
# - ``数据文件``: the adapter's output path; the dataset identity is
#   expressed by 数据来源 / Gold 数据 / Candidate 数据, so a divergent
#   file path alone is not a real inconsistency.
# - ``Candidate 数据``: comparing two candidate extractions against the
#   same gold is the whole point of compare-mode, so different candidate
#   sources are the norm rather than an anomaly. A ⚠ on Gold 数据 IS
#   meaningful (means baseline vs candidate used different eval data)
#   and stays warn-eligible.
_NO_WARN_ROWS = {"运行时间", "数据文件", "Candidate 数据"}


def _section_run_context(comparison: ComparisonResult) -> List[str]:
    """Compare-mode: baseline vs candidate side-by-side environment table.

    Answers "was it the same data / same commit / same model?" — without
    this a reviewer cannot tell if the delta comes from the change under
    test or from a data/model swap.
    """
    b_meta = comparison.baseline_metadata or {}
    c_meta = comparison.candidate_metadata or {}
    b_dm = b_meta.get("data_meta") or {}
    c_dm = c_meta.get("data_meta") or {}
    lines = ["## 🔧 运行上下文", ""]
    lines.append("| 项 | Baseline | Candidate |")
    lines.append("|----|----------|-----------|")
    lines.append(_run_ctx_row("运行时间", b_meta.get("timestamp"), c_meta.get("timestamp")))
    lines.append(
        _run_ctx_row(
            "Git Commit",
            _short_commit(b_meta.get("git_commit")),
            _short_commit(c_meta.get("git_commit")),
        )
    )
    lines.append(_run_ctx_row("Judge 模型", b_meta.get("model"), c_meta.get("model")))
    lines.append(_run_ctx_row("Temperature / Seed",
                              f"{b_meta.get('temperature')}/{b_meta.get('seed')}"
                              if b_meta.get("temperature") is not None else None,
                              f"{c_meta.get('temperature')}/{c_meta.get('seed')}"
                              if c_meta.get("temperature") is not None else None))
    lines.append(_run_ctx_row("数据文件", b_meta.get("data_path"), c_meta.get("data_path")))
    lines.append(_run_ctx_row("Gold 数据", b_dm.get("gold_source"), c_dm.get("gold_source")))
    lines.append(_run_ctx_row("Candidate 数据", b_dm.get("candidate_source"), c_dm.get("candidate_source")))
    lines.append(_run_ctx_row("评测对象", b_dm.get("candidate"), c_dm.get("candidate")))
    # gold/candidate pairing on both sides if reported by adapter.
    b_pair = f"{b_dm.get('matched_count')}/{b_dm.get('gold_count')}" if b_dm.get("gold_count") is not None else None
    c_pair = f"{c_dm.get('matched_count')}/{c_dm.get('gold_count')}" if c_dm.get("gold_count") is not None else None
    lines.append(_run_ctx_row("gold/candidate 配对", b_pair, c_pair))
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Section 2 — 分析
# ---------------------------------------------------------------------------


def _section_analysis(result: BenchmarkResult, comparison: Optional[ComparisonResult]) -> List[str]:
    """Programmatic analysis bullets — dimension / sub-dim / type / concentration."""
    if comparison is None:
        return []
    analysis = comparison.analyze()

    lines: List[str] = ["## 🔍 分析", ""]
    bullets: List[str] = []

    # (a) Which sub-dimension regressed most? Strongest localized signal.
    worst_subdim = _worst_subdimension(analysis)
    if worst_subdim:
        name, slot = worst_subdim
        bullets.append(
            f"退化集中在 **{_prettify_subdim_key(name)}**（{slot['regressed']}/{slot['total']} 指标退化，"
            f"最严重 {_fmt_delta(slot['worst_delta'])}）"
        )

    # (b) Question-type clustering — is the regression pinned to one tier?
    by_qt = analysis["by_question_type"]
    if by_qt:
        dominant_qt, dominant_n = max(by_qt.items(), key=lambda kv: kv[1])
        total_reg = analysis["concentration"]["regressed_samples"]
        if total_reg and dominant_n / max(total_reg, 1) >= 0.5 and len(by_qt) < total_reg:
            bullets.append(
                f"退化扎堆在 **{dominant_qt}** 类型（{dominant_n}/{total_reg}），"
                f"建议回归测试聚焦该类型"
            )

    # (c) Concentration — outlier-driven vs systemic.
    conc = analysis["concentration"]
    n_reg_samples = conc["regressed_samples"]
    n_reg_metrics = analysis["counts"]["regressed"]
    if n_reg_samples and n_reg_metrics:
        if n_reg_samples == 1:
            bullets.append("退化为单一样例驱动（个案），非系统性回归")
        elif conc["max_metrics_per_sample"] >= 3:
            bullets.append(
                f"最严重样例一次丢失 {conc['max_metrics_per_sample']} 个指标，"
                "关注是否存在结构性破坏"
            )

    # (d) Direction consistency within a sub-dimension — noise vs real signal.
    inconsistent = _direction_inconsistency(analysis)
    if inconsistent:
        names = "、".join(inconsistent[:3])
        bullets.append(f"部分维度指标方向不一致（{names}），可能为评测噪音而非真实变化")

    if bullets:
        for b in bullets:
            lines.append(f"- {b}")
    else:
        lines.append("- 无显著结构性变化")
    lines.append("")
    return lines


def _worst_subdimension(analysis: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Pick the sub-dimension worst-hit: most regressions, then most-negative delta."""
    candidates = [
        (name, slot)
        for name, slot in analysis["by_subdimension"].items()
        if slot["regressed"] > 0
    ]
    if not candidates:
        return None
    # Most regressions first; ties broken by the most-negative worst_delta.
    candidates.sort(key=lambda kv: (-kv[1]["regressed"], kv[1]["worst_delta"]))
    return candidates[0]


def _direction_inconsistency(analysis: Dict[str, Any]) -> List[str]:
    """Sub-dimensions where metrics move in opposite directions (noise hint)."""
    out: List[str] = []
    for name, slot in analysis["by_subdimension"].items():
        if slot["regressed"] and slot["improved"] and slot["total"] >= 2:
            out.append(_prettify_subdim_key(name))
    return out


def _prettify_subdim_key(key: str) -> str:
    """Turn ``"extraction / 关系抽取"`` → ``"图提取 / 关系抽取"``.

    ``ComparisonResult.analyze()`` keys ``by_subdimension`` with the raw
    domain slug (English), which is unfriendly in a Chinese report.
    """
    if " / " not in key:
        return key
    domain, subdim = key.split(" / ", 1)
    return f"{domain_label(domain)} / {subdim}"


# ---------------------------------------------------------------------------
# Section 3 — 指标总览
# ---------------------------------------------------------------------------


def _section_metrics(result: BenchmarkResult, comparison: Optional[ComparisonResult]) -> List[str]:
    """Changed-metric table up top; unchanged metrics folded below."""
    if comparison is None:
        # Single run: the domain (e.g. 图提取) becomes the section heading,
        # with each sub-dimension as a sub-heading — no generic 指标总览 wrapper.
        return _render_single_run_metrics(result)

    lines: List[str] = ["## 指标总览", ""]
    lines.append("> **Δ 怎么读**：Δ 是 candidate 相对 baseline 的变化，已按指标方向"
                 "翻号；**负数 = 变差**（无论指标本身是「越高越好」还是「越低越好」）。")
    lines.append(">")
    lines.append("> **方向列（↑/↓）**：只影响原始 Baseline / Candidate 数值的解读；对 Δ 的正负判断无影响。")
    lines.append("")

    analysis = comparison.analyze()
    verdicts = analysis["metric_verdicts"]

    # Group verdicts by top-level domain first — a compare report may
    # contain multiple modes (图提取 + 检索 + 生成回答) and each domain
    # deserves its own subsection so a reader can drill into the relevant
    # one without wading through unrelated metrics.
    by_domain: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for metric, v in verdicts.items():
        domain, _ = get_dimension(metric)
        by_domain.setdefault(domain, []).append((metric, v))

    # Stable domain order: extraction first, retrieval, generation, then Other.
    domain_order = ["extraction", "retrieval", "generation"]
    ordered_domains = [d for d in domain_order if d in by_domain] + \
                      [d for d in sorted(by_domain.keys()) if d not in domain_order]

    for domain in ordered_domains:
        domain_verdicts = dict(by_domain[domain])
        # Skip a domain only when it has no movement at all (all unchanged).
        # Otherwise we surface both regressions and the untouched context.
        domain_lines = _render_metrics_for_domain(domain, domain_verdicts, comparison)
        if domain_lines:
            lines.append(f"### {domain_label(domain)}")
            lines.append("")
            lines.extend(domain_lines)
    return lines


def _render_metrics_for_domain(
    domain: str,
    verdicts: Dict[str, Dict[str, Any]],
    comparison: ComparisonResult,
) -> List[str]:
    """Sub-dimension rollup + per-metric table for one domain.

    Returns [] when every metric in the domain is unchanged so a
    completely quiet domain (e.g. 检索 in an extraction-only compare)
    doesn't take up section space.
    """
    changed = [(m, v) for m, v in verdicts.items() if v["verdict"] != "unchanged"]
    flat = [(m, v) for m, v in verdicts.items() if v["verdict"] == "unchanged"]
    if not changed and not flat:
        return []

    lines: List[str] = []

    # Sub-dimension rollup — one summary row per sub-dim within the domain.
    subdim_totals: Dict[str, Dict[str, Any]] = {}
    for metric, v in verdicts.items():
        _, subdim = get_dimension(metric)
        slot = subdim_totals.setdefault(
            subdim,
            {"base_sum": 0.0, "cand_sum": 0.0, "n": 0, "worst_delta": 0.0,
             "regressed": 0, "improved": 0, "unchanged": 0},
        )
        slot["base_sum"] += comparison.baseline_overall.get(metric, 0.0)
        slot["cand_sum"] += comparison.candidate_overall.get(metric, 0.0)
        slot["n"] += 1
        slot[v["verdict"]] = slot.get(v["verdict"], 0) + 1
        if v["semantic_delta"] < slot["worst_delta"]:
            slot["worst_delta"] = v["semantic_delta"]

    moved_subdims = [(s, slot) for s, slot in subdim_totals.items()
                     if slot["regressed"] or slot["improved"]]
    if moved_subdims:
        moved_subdims.sort(
            key=lambda kv: (subdim_order(kv[0]), kv[1]["worst_delta"])
        )
        lines.append("**按子维度汇总**")
        lines.append("")
        lines.append("| 子维度 | 指标数 | Baseline 均值 | Candidate 均值 | 最严重 Δ | 退化/改进 |")
        lines.append("|--------|--------|--------------|----------------|---------|-----------|")
        for name, slot in moved_subdims:
            base_avg = slot["base_sum"] / slot["n"] if slot["n"] else 0.0
            cand_avg = slot["cand_sum"] / slot["n"] if slot["n"] else 0.0
            reg = slot.get("regressed", 0)
            imp = slot.get("improved", 0)
            lines.append(
                f"| {name} | {slot['n']} | {_fmt(base_avg)} | {_fmt(cand_avg)} "
                f"| {_fmt_delta(slot['worst_delta'])} | {reg} 退化 / {imp} 改进 |"
            )
        lines.append("")

    # Changed-metric detail — worst regression first.
    changed.sort(key=lambda mv: mv[1]["semantic_delta"])
    if changed:
        lines.append("**逐指标明细**")
        lines.append("")
        lines.append("| 指标 | 子维度 | 方向 | Baseline | Candidate | Δ | 判定 |")
        lines.append("|------|-------|------|----------|-----------|-----|------|")
        for metric, v in changed:
            _, subdim = get_dimension(metric)
            base_val = comparison.baseline_overall.get(metric, 0.0)
            cand_val = comparison.candidate_overall.get(metric, 0.0)
            lines.append(
                f"| {metric} | {subdim} | {_direction_symbol(metric)} "
                f"| {_fmt(base_val)} | {_fmt(cand_val)} "
                f"| {_fmt_delta(v['semantic_delta'])} | {_VERDICT_SYMBOL[v['verdict']]} {_VERDICT_LABEL[v['verdict']]} |"
            )
        lines.append("")

    if flat:
        lines.append(f"<details><summary>未显著变化的指标（{len(flat)}）</summary>")
        lines.append("")
        lines.append("| 指标 | 子维度 | 方向 | Baseline | Candidate | Δ |")
        lines.append("|------|-------|------|----------|-----------|-----|")
        for metric, v in sorted(flat, key=lambda mv: mv[0]):
            _, subdim = get_dimension(metric)
            base_val = comparison.baseline_overall.get(metric, 0.0)
            cand_val = comparison.candidate_overall.get(metric, 0.0)
            lines.append(
                f"| {metric} | {subdim} | {_direction_symbol(metric)} "
                f"| {_fmt(base_val)} | {_fmt(cand_val)} | {_fmt_delta(v['semantic_delta'])} |"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return lines


def _render_single_run_metrics(result: BenchmarkResult) -> List[str]:
    """Single-run metrics: domain as the section heading, sub-dimensions nested."""
    lines: List[str] = []
    # Group by domain -> sub-dimension -> [metrics], so the domain (图提取)
    # heads the section and each sub-dimension (实体识别 / 关系抽取 / ...) is a
    # sub-heading with its own table.
    by_domain: Dict[str, Dict[str, List[str]]] = {}
    for metric in sorted(result.overall.keys()):
        if metric.endswith("_redundancy"):
            continue  # 诊断指标（冗余率），不进指标总览子表，在概览 + 证据层展示
        domain, subdim = get_dimension(metric)
        by_domain.setdefault(domain, {}).setdefault(subdim, []).append(metric)

    for domain in sorted(by_domain.keys()):
        lines.append(f"## {domain_label(domain)}")
        lines.append("")
        subdims = by_domain[domain]
        for subdim in sorted(subdims.keys(), key=lambda s: (subdim_order(s), s)):
            lines.append(f"### {subdim}")
            lines.append("")
            # Split into separate tables by computation method when a
            # sub-dimension mixes EM and LLM metrics (e.g. 实体识别).
            groups: Dict[str, List[str]] = {}
            for metric in subdims[subdim]:
                groups.setdefault(_metric_method_group(metric), []).append(metric)
            multi = len(groups) > 1
            for group_name in sorted(groups.keys(), key=lambda g: _METHOD_ORDER.get(g, 99)):
                if multi:
                    lines.append(f"**{group_name}**")
                    lines.append("")
                lines.append("| 指标 | 说明 | 方向 | 得分 |")
                lines.append("|------|------|------|------|")
                for metric in groups[group_name]:
                    lines.append(
                        f"| {metric} | {metric_description(metric)} | {_direction_label(metric)} "
                        f"| {_fmt(result.overall[metric])} |"
                    )
                lines.append("")
    return lines


def _section_by_type(result: BenchmarkResult) -> List[str]:
    """Single-run: metrics grouped by question-type tier (``result.by_type``).

    Mirrors the domain grouping but keyed by ``SampleResult.question_type``,
    so tiered runs (e.g. GraphRAG-Benchmark) can be read per question type
    instead of collapsing to one overall number. Empty on untiered runs.
    """
    by_type = result.by_type or {}
    if not by_type:
        return []
    lines: List[str] = ["## 按问题类型", ""]
    for tier in sorted(by_type.keys()):
        metrics = by_type[tier]
        if not metrics:
            continue
        lines.append(f"### {tier}")
        lines.append("")
        lines.append("| 指标 | 方向 | 得分 |")
        lines.append("|------|------|------|")
        for metric in sorted(metrics.keys()):
            lines.append(f"| {metric} | {_direction_symbol(metric)} | {_fmt(metrics[metric])} |")
        lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Section 4 — 样例
# ---------------------------------------------------------------------------


def _section_samples(
    title: str,
    symbol: str,
    entries: List[Dict[str, Any]],
    change_key: str,
    limit: int = 5,
) -> List[str]:
    """Render regressed/improved samples: top-N rows + folded detail.

    Each entry becomes ONE row (sample_id + worst metric + counts) so a human
    can scan dozens of samples; the per-metric breakdown is folded.

    Always renders the section header, even when empty — so a reader can
    tell "no regressions" apart from "hidden in a fold" at a glance.
    """
    if not entries:
        # Explicit empty state — critical for compare reports because a
        # missing 🟢 改进样例 section otherwise reads as "collapsed", not
        # "genuinely none".
        return [f"## {symbol} {title}（0）", "", "- 无该方向的样例。", ""]

    lines: List[str] = [f"## {symbol} {title}（{len(entries)}）", ""]

    # Detect whether ``question_type`` is meaningful for this run. In an
    # extraction compare (no tiering) it's always None, and a column of
    # bare "—" is confusing — drop the column altogether. In retrieval /
    # generation runs with GraphRAG-style tiers, keep it.
    has_qtypes = any(e.get("question_type") for e in entries)

    # Flatten to find the worst metric per sample, then sort samples by it.
    summarized: List[Dict[str, Any]] = []
    for entry in entries:
        changes = entry.get(change_key, {})
        if not changes:
            continue
        # worst = most negative semantic delta (regression) or most positive (improvement)
        worst_metric, worst_delta = min(changes.items(), key=lambda kv: kv[1]) \
            if change_key == "regressions" else max(changes.items(), key=lambda kv: kv[1])
        base_m = entry.get("baseline_metrics", {})
        cand_m = entry.get("candidate_metrics", {})
        summarized.append(
            {
                "sample_id": entry["sample_id"],
                "question_type": entry.get("question_type"),
                "worst_metric": worst_metric,
                "worst_delta": worst_delta,
                "worst_baseline": base_m.get(worst_metric),
                "worst_candidate": cand_m.get(worst_metric),
                "n_metrics": len(changes),
            }
        )
    summarized.sort(key=lambda r: r["worst_delta"])  # worst first

    # Column set depends on tier availability. Baseline / Candidate cells
    # come from the sample's actual metric values so the top table shows
    # *what changed*, not just *by how much*.
    if has_qtypes:
        lines.append("| Sample | 最严重指标 | Baseline | Candidate | Δ | 涉及指标数 | 类型 |")
        lines.append("|--------|-----------|----------|-----------|-----|-----------|------|")
    else:
        lines.append("| Sample | 最严重指标 | Baseline | Candidate | Δ | 涉及指标数 |")
        lines.append("|--------|-----------|----------|-----------|-----|-----------|")
    for row in summarized[:limit]:
        base_cell = _fmt(row["worst_baseline"]) if isinstance(row["worst_baseline"], (int, float)) else "—"
        cand_cell = _fmt(row["worst_candidate"]) if isinstance(row["worst_candidate"], (int, float)) else "—"
        base_row = (
            f"| {row['sample_id']} | {row['worst_metric']} "
            f"| {base_cell} | {cand_cell} | {_fmt_delta(row['worst_delta'])} "
            f"| {row['n_metrics']} |"
        )
        if has_qtypes:
            qt = row["question_type"] or "—"
            base_row += f" {qt} |"
        lines.append(base_row)
    if len(summarized) > limit:
        pad = "| " * (7 if has_qtypes else 6)
        lines.append(f"| ... | 还有 {len(summarized) - limit} 个样例见下方明细 {pad}")
    lines.append("")

    # Folded per-sample detail. One table per sample — inside the table
    # each row is a metric that moved on that sample, worst first. This
    # reads much better than a flat sample × metric list: the reviewer's
    # actual question is "what happened to sample X?", so grouping the
    # rows under a per-sample sub-header keeps context together.
    lines.append("<details><summary>逐样本明细</summary>")
    lines.append("")
    # Preserve top-level worst-first sample order.
    order_by_sid = {row["sample_id"]: i for i, row in enumerate(summarized)}
    ordered_entries = sorted(
        entries,
        key=lambda e: order_by_sid.get(e["sample_id"], 999),
    )
    for entry in ordered_entries:
        sid = entry["sample_id"]
        base_m = entry.get("baseline_metrics", {})
        cand_m = entry.get("candidate_metrics", {})
        # Rows within one sample: worst delta first for regressions,
        # best delta first for improvements.
        items = list(entry.get(change_key, {}).items())
        items.sort(key=lambda kv: kv[1], reverse=(change_key != "regressions"))
        if not items:
            continue
        qt = entry.get("question_type")
        header_tail = f"（类型：{qt}）" if qt else ""
        lines.append(f"#### `{sid}`{header_tail}")
        lines.append("")
        lines.append("| 指标 | 方向 | Baseline | Candidate | Δ |")
        lines.append("|------|------|----------|-----------|-----|")
        for metric, diff in items:
            lines.append(
                f"| {metric} | {_direction_symbol(metric)} "
                f"| {_fmt(base_m.get(metric, 0.0))} "
                f"| {_fmt(cand_m.get(metric, 0.0))} | {_fmt_delta(diff)} |"
            )
        lines.append("")
    lines.append("</details>")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Section 5 — 证据层
# ---------------------------------------------------------------------------


def _section_evidence(
    result: BenchmarkResult,
    comparison: Optional[ComparisonResult] = None,
) -> List[str]:
    """Failures + full per-sample metrics + metadata, all folded.

    In compare mode the wide per-sample table renders Baseline / Candidate
    rows for every sample so the reviewer can eyeball the shift without
    reopening the source JSON. In single-run mode it stays a single-row
    per-sample summary as before.
    """
    lines: List[str] = ["## 证据层", ""]
    errors = result.metadata.get("errors", [])
    if errors:
        lines.append("<details><summary>失败样例（{}）</summary>".format(len(errors)))
        lines.append("")
        lines.append("| Sample | Metric | Error |")
        lines.append("|--------|--------|-------|")
        for entry in errors:
            sid = entry.get("sample_id", "N/A")
            metric = entry.get("metric", "N/A")
            err = str(entry.get("error", "")).replace("|", "\\|").replace("\n", " ")
            if len(err) > 120:
                err = err[:117] + "..."
            lines.append(f"| {sid} | {metric} | {err} |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # Full per-sample metrics: wide table so each sample is one row, folded.
    samples = result.samples
    metric_keys = sorted(
        result.overall.keys(),
        key=lambda m: (_WIDE_TABLE_ORDER.index(m) if m in _WIDE_TABLE_ORDER else 999, m),
    )
    if samples and metric_keys:
        if comparison is None:
            # Single run — one row per sample.
            lines.append("<details><summary>逐样例指标明细（{} 个样例）</summary>".format(len(samples)))
            lines.append("")
            lines.append("| Sample | " + " | ".join(metric_keys) + " |")
            lines.append("|--------|" + "|".join(["------" for _ in metric_keys]) + "|")
            for s in samples:
                cells = []
                for k in metric_keys:
                    v = s.metrics.get(k)
                    cells.append(_fmt(v) if isinstance(v, (int, float)) else "N/A")
                lines.append("| " + str(s.sample_id) + " | " + " | ".join(cells) + " |")
            lines.append("")
            lines.append("</details>")
            lines.append("")
        else:
            # Compare — two rows per sample (Baseline / Candidate) so the
            # reviewer can spot per-cell shifts. Only samples that showed
            # some change need to be rendered; unchanged samples are folded
            # by the sample sections above.
            reg_by_sid = {e["sample_id"]: e for e in comparison.regressed_samples}
            imp_by_sid = {e["sample_id"]: e for e in comparison.improved_samples}
            all_sids = set(reg_by_sid) | set(imp_by_sid)
            if all_sids:
                lines.append(
                    "<details><summary>逐样例指标明细（Baseline vs Candidate，{} 个样例）</summary>".format(len(all_sids))
                )
                lines.append("")
                lines.append("| Sample | 侧 | " + " | ".join(metric_keys) + " |")
                lines.append("|--------|----|" + "|".join(["------" for _ in metric_keys]) + "|")
                for sid in sorted(all_sids):
                    entry = reg_by_sid.get(sid) or imp_by_sid.get(sid) or {}
                    base_m = entry.get("baseline_metrics", {})
                    cand_m = entry.get("candidate_metrics", {})
                    for label, m in (("Baseline", base_m), ("Candidate", cand_m)):
                        cells = []
                        for k in metric_keys:
                            v = m.get(k)
                            cells.append(_fmt(v) if isinstance(v, (int, float)) else "N/A")
                        lines.append(
                            "| " + sid + " | " + label + " | " + " | ".join(cells) + " |"
                        )
                lines.append("")
                lines.append("</details>")
                lines.append("")
        lines.append("> **冗余率**（列 `semantic_entity_redundancy` / `semantic_triple_redundancy`）")
        lines.append("> - 含义：candidate 实体/三元组中，被 LLM 判定语义重复、一对一去重时丢弃的占比")
        lines.append("> - 解读：冗余率高 = 抽取系统产出大量语义重复实体/关系（抽取过碎），属抽取侧问题、非匹配错误")
        lines.append("")

    # Metadata footer. Compare mode already prints the environment table
    # up-top; here we just keep the run-timing details for the candidate
    # side so 证据层 stays self-contained.
    meta = result.metadata
    lines.append("<details><summary>元数据</summary>")
    lines.append("")
    if comparison is not None:
        b = comparison.baseline_metadata or {}
        c = comparison.candidate_metadata or {}
        lines.append(f"- Baseline · Timestamp: {b.get('timestamp', 'N/A')} · Commit: {_short_commit(b.get('git_commit'))}")
        lines.append(f"- Candidate · Timestamp: {c.get('timestamp', 'N/A')} · Commit: {_short_commit(c.get('git_commit'))}")
    else:
        lines.append(f"- Timestamp: {meta.get('timestamp', 'N/A')}")
        lines.append(f"- Git Commit: {meta.get('git_commit', 'N/A')}")
        lines.append(f"- Model: {meta.get('model', 'N/A')}")
        if meta.get("temperature") is not None:
            lines.append(f"- Temperature: {meta.get('temperature')}  Seed: {meta.get('seed')}")
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class MarkdownReporter:
    """Generate a Markdown string from benchmark results.

    Output is designed to be pasted into PR / Issue comments.
    """

    @staticmethod
    def report(
        result: BenchmarkResult,
        comparison: Optional[ComparisonResult] = None,
    ) -> str:
        """Build a Markdown report.

        Args:
            result: The (candidate) benchmark result to report.
            comparison: Optional comparison against a baseline.

        Returns:
            A complete Markdown document as a string.
        """
        analysis = comparison.analyze() if comparison else None
        lines: List[str] = []

        lines.append("# Benchmark Report")
        lines.append("")
        lines.extend(_section_overview(result, analysis, comparison))
        # `关于本次测评` is useful whichever mode we're in — reviewers coming
        # to a compare report also need to know what EM vs LLM means and
        # which direction is good, otherwise the `Δ` column is a mystery.
        lines.extend(_section_about(result))
        # Compare mode: right after the TL;DR, spell out what was compared
        # against what. Without this the reviewer can't tell if the delta
        # comes from the change under test, a data swap, or a judge swap.
        if comparison is not None:
            lines.extend(_section_run_context(comparison))
        lines.extend(_section_analysis(result, comparison))
        lines.extend(_section_metrics(result, comparison))
        if not comparison:
            lines.extend(_section_by_type(result))

        if comparison:
            lines.extend(
                _section_samples("退化样例", "🔴", comparison.regressed_samples, "regressions")
            )
            lines.extend(
                _section_samples("改进样例", "🟢", comparison.improved_samples, "improvements")
            )
        lines.extend(_section_evidence(result, comparison))

        return "\n".join(lines)
