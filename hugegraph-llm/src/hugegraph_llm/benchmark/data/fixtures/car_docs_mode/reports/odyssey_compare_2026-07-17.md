# Benchmark Report

## 📊 概览

- 指标变化：8 退化 / 0 改进 / 11 持平（共 19）
- 图提取：8 退化，最严重 -86.67
- 样例级：3 个退化样例 / 0 个改进样例

## 关于本次测评

- **任务**：图提取（extraction）—— 比较 candidate 抽取的图与 gold 标注的标准图，评估知识图谱抽取质量。
- **数据**：按 chunk 配对；gold = 人工标注，candidate = 被测系统输出。
- **评分方式**（各指标「说明」列标注了计算方式标签，含义如下）：
  - **EM**（Exact Match）：字符串精确匹配，归一化后字面相等才算命中 —— 用于实体 / 关系 / 属性
  - **LLM**（LLM as Judge）：模型判定语义等价 / 是否无幻觉 —— 用于语义 F1、抽取忠实度
  - **Schema 检查**：校验类型 / 属性 / 边端点是否符合 gold 推导的 schema —— 用于 Schema 合规（独立于匹配的合规性校验）
- **如何读数**：
  - 分数已映射到 0–100（满分 100）
  - 「方向」列：↑ 越高越好、↓ 越低越好
  - EM 下「发动机」vs「引擎」算未命中，F1 偏低不代表抽错，可能是同义表述差异（LLM 类指标能补救）

## 🔧 运行上下文

| 项 | Baseline | Candidate |
|----|----------|-----------|
| 运行时间 | 2026-07-17T13:51:42 | 2026-07-17T13:52:32 |
| Git Commit | 6ee4c53 | 6ee4c53 |
| Judge 模型 | DeepSeek-V4-Pro | DeepSeek-V4-Pro |
| Temperature / Seed | 0.0/42 | 0.0/42 |
| 数据文件 | /tmp/benchmark_odyssey/car_pipeline_extraction.json | /tmp/benchmark_odyssey/car_pipeline_extraction_regressed.json |
| Gold 数据 | hugegraph-llm/src/hugegraph_llm/benchmark/data/fixtures/car_docs_mode/gold | hugegraph-llm/src/hugegraph_llm/benchmark/data/fixtures/car_docs_mode/gold |
| Candidate 数据 | hugegraph-llm/src/hugegraph_llm/benchmark/data/fixtures/car_docs_mode/docs_run | hugegraph-llm/src/hugegraph_llm/benchmark/data/fixtures/car_docs_mode/docs_run_regressed |
| 评测对象 | pipeline | pipeline |
| gold/candidate 配对 | 3/3 | 3/3 |

## 🔍 分析

- 退化集中在 **图提取 / 关系抽取**（3/6 指标退化，最严重 -86.67）
- 最严重样例一次丢失 7 个指标，关注是否存在结构性破坏

## 指标总览

> **Δ 怎么读**：Δ 是 candidate 相对 baseline 的变化，已按指标方向翻号；**负数 = 变差**（无论指标本身是「越高越好」还是「越低越好」）。
>
> **方向列（↑/↓）**：只影响原始 Baseline / Candidate 数值的解读；对 Δ 的正负判断无影响。

### 图提取

**按子维度汇总**

| 子维度 | 指标数 | Baseline 均值 | Candidate 均值 | 最严重 Δ | 退化/改进 |
|--------|--------|--------------|----------------|---------|-----------|
| 实体识别 | 6 | 45.52 | 32.54 | -38.89 | 3 退化 / 0 改进 |
| 关系抽取 | 6 | 42.41 | 10.95 | -86.67 | 3 退化 / 0 改进 |
| Schema 合规 | 3 | 70.37 | 77.78 | -22.22 | 1 退化 / 0 改进 |
| 抽取忠实度 | 1 | 95.24 | 78.33 | -16.91 | 1 退化 / 0 改进 |

**逐指标明细**

| 指标 | 子维度 | 方向 | Baseline | Candidate | Δ | 判定 |
|------|-------|------|----------|-----------|-----|------|
| semantic_triple_recall | 关系抽取 | ↑ | 100.00 | 13.33 | -86.67 | 🔴 退化 |
| semantic_triple_f1 | 关系抽取 | ↑ | 82.22 | 19.05 | -63.17 | 🔴 退化 |
| semantic_entity_recall | 实体识别 | ↑ | 88.89 | 50.00 | -38.89 | 🔴 退化 |
| semantic_triple_precision | 关系抽取 | ↑ | 72.22 | 33.33 | -38.89 | 🔴 退化 |
| semantic_entity_f1 | 实体识别 | ↑ | 90.91 | 61.90 | -29.01 | 🔴 退化 |
| illegal_edge_rate | Schema 合规 | ↓ | 11.11 | 33.33 | -22.22 | 🔴 退化 |
| extraction_faithfulness | 抽取忠实度 | ↑ | 95.24 | 78.33 | -16.91 | 🔴 退化 |
| semantic_entity_precision | 实体识别 | ↑ | 93.33 | 83.33 | -10.00 | 🔴 退化 |

<details><summary>未显著变化的指标（11）</summary>

| 指标 | 子维度 | 方向 | Baseline | Candidate | Δ |
|------|-------|------|----------|-----------|-----|
| entity_f1 | 实体识别 | ↑ | 0.00 | 0.00 | 0.00 |
| entity_precision | 实体识别 | ↑ | 0.00 | 0.00 | 0.00 |
| entity_recall | 实体识别 | ↑ | 0.00 | 0.00 | 0.00 |
| property_f1 | 属性抽取 | ↑ | 0.00 | 0.00 | 0.00 |
| property_precision | 属性抽取 | ↑ | 0.00 | 0.00 | 0.00 |
| property_recall | 属性抽取 | ↑ | 0.00 | 0.00 | 0.00 |
| required_property_fill | Schema 合规 | ↑ | 100.00 | 100.00 | 0.00 |
| triple_f1 | 关系抽取 | ↑ | 0.00 | 0.00 | 0.00 |
| triple_precision | 关系抽取 | ↑ | 0.00 | 0.00 | 0.00 |
| triple_recall | 关系抽取 | ↑ | 0.00 | 0.00 | 0.00 |
| type_constraint_pass | Schema 合规 | ↑ | 100.00 | 100.00 | 0.00 |

</details>

## 🔴 退化样例（3）

| Sample | 最严重指标 | Baseline | Candidate | Δ | 涉及指标数 |
|--------|-----------|----------|-----------|-----|-----------|
| 奥德赛用户手册2023款--e9c06954_s0046 | semantic_triple_recall | 100.00 | 0.00 | -100.00 | 7 |
| 奥德赛用户手册2023款--e9c06954_s0105 | semantic_triple_recall | 100.00 | 0.00 | -100.00 | 7 |
| 奥德赛用户手册2023款--e9c06954_s0060 | semantic_triple_recall | 100.00 | 40.00 | -60.00 | 5 |

<details><summary>逐样本明细</summary>

#### `奥德赛用户手册2023款--e9c06954_s0046`

| 指标 | 方向 | Baseline | Candidate | Δ |
|------|------|----------|-----------|-----|
| semantic_triple_recall | ↑ | 100.00 | 0.00 | -100.00 |
| semantic_triple_f1 | ↑ | 66.67 | 0.00 | -66.67 |
| semantic_triple_precision | ↑ | 50.00 | 0.00 | -50.00 |
| semantic_entity_recall | ↑ | 66.67 | 33.33 | -33.34 |
| semantic_entity_f1 | ↑ | 72.73 | 44.44 | -28.29 |
| extraction_faithfulness | ↑ | 100.00 | 75.00 | -25.00 |
| semantic_entity_precision | ↑ | 80.00 | 66.67 | -13.33 |

#### `奥德赛用户手册2023款--e9c06954_s0105`

| 指标 | 方向 | Baseline | Candidate | Δ |
|------|------|----------|-----------|-----|
| semantic_triple_recall | ↑ | 100.00 | 0.00 | -100.00 |
| semantic_triple_f1 | ↑ | 80.00 | 0.00 | -80.00 |
| semantic_triple_precision | ↑ | 66.67 | 0.00 | -66.67 |
| illegal_edge_rate | ↓ | 33.33 | 100.00 | -66.67 |
| extraction_faithfulness | ↑ | 85.71 | 60.00 | -25.71 |
| semantic_entity_recall | ↑ | 100.00 | 75.00 | -25.00 |
| semantic_entity_f1 | ↑ | 100.00 | 85.71 | -14.29 |

#### `奥德赛用户手册2023款--e9c06954_s0060`

| 指标 | 方向 | Baseline | Candidate | Δ |
|------|------|----------|-----------|-----|
| semantic_triple_recall | ↑ | 100.00 | 40.00 | -60.00 |
| semantic_entity_recall | ↑ | 100.00 | 41.67 | -58.33 |
| semantic_entity_f1 | ↑ | 100.00 | 55.56 | -44.44 |
| semantic_triple_f1 | ↑ | 100.00 | 57.14 | -42.86 |
| semantic_entity_precision | ↑ | 100.00 | 83.33 | -16.67 |

</details>

## 🟢 改进样例（0）

- 无该方向的样例。

## 证据层

<details><summary>逐样例指标明细（Baseline vs Candidate，3 个样例）</summary>

| Sample | 侧 | entity_f1 | entity_precision | entity_recall | semantic_entity_f1 | semantic_entity_precision | semantic_entity_recall | semantic_entity_redundancy | triple_f1 | triple_precision | triple_recall | semantic_triple_f1 | semantic_triple_precision | semantic_triple_recall | semantic_triple_redundancy | type_constraint_pass | required_property_fill | illegal_edge_rate | extraction_faithfulness | property_f1 | property_precision | property_recall |
|--------|----|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| 奥德赛用户手册2023款--e9c06954_s0046 | Baseline | 0.00 | 0.00 | 0.00 | 72.73 | 80.00 | 66.67 | 0.00 | 0.00 | 0.00 | 0.00 | 66.67 | 50.00 | 100.00 | 0.00 | 100.00 | 100.00 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 |
| 奥德赛用户手册2023款--e9c06954_s0046 | Candidate | 0.00 | 0.00 | 0.00 | 44.44 | 66.67 | 33.33 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.00 | 100.00 | 0.00 | 75.00 | 0.00 | 0.00 | 0.00 |
| 奥德赛用户手册2023款--e9c06954_s0060 | Baseline | 0.00 | 0.00 | 0.00 | 100.00 | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.00 | 100.00 | 100.00 | 0.00 | 100.00 | 100.00 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 |
| 奥德赛用户手册2023款--e9c06954_s0060 | Candidate | 0.00 | 0.00 | 0.00 | 55.56 | 83.33 | 41.67 | 0.00 | 0.00 | 0.00 | 0.00 | 57.14 | 100.00 | 40.00 | 0.00 | 100.00 | 100.00 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 |
| 奥德赛用户手册2023款--e9c06954_s0105 | Baseline | 0.00 | 0.00 | 0.00 | 100.00 | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 80.00 | 66.67 | 100.00 | 0.00 | 100.00 | 100.00 | 33.33 | 85.71 | 0.00 | 0.00 | 0.00 |
| 奥德赛用户手册2023款--e9c06954_s0105 | Candidate | 0.00 | 0.00 | 0.00 | 85.71 | 100.00 | 75.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.00 | 100.00 | 100.00 | 60.00 | 0.00 | 0.00 | 0.00 |

</details>

> **冗余率**（列 `semantic_entity_redundancy` / `semantic_triple_redundancy`）
> - 含义：candidate 实体/三元组中，被 LLM 判定语义重复、一对一去重时丢弃的占比
> - 解读：冗余率高 = 抽取系统产出大量语义重复实体/关系（抽取过碎），属抽取侧问题、非匹配错误

<details><summary>元数据</summary>

- Baseline · Timestamp: 2026-07-17T13:51:42 · Commit: 6ee4c53
- Candidate · Timestamp: 2026-07-17T13:52:32 · Commit: 6ee4c53

</details>
