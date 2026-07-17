# Benchmark Report

## 📊 概览

- 样例数：3
- 指标数：19
- 数据文件：`/tmp/benchmark_odyssey/car_pipeline_extraction.json`
- Gold 数据：`hugegraph-llm/src/hugegraph_llm/benchmark/data/fixtures/car_docs_mode/gold`
- Candidate 数据：`hugegraph-llm/src/hugegraph_llm/benchmark/data/fixtures/car_docs_mode/docs_run`
- 评测对象：pipeline
- gold/candidate 配对：3/3
- candidate 平均冗余率：实体 0.00 / 三元组 0.00

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

## 图提取

### 实体识别

**EM**

| 指标 | 说明 | 方向 | 得分 |
|------|------|------|------|
| entity_f1 | 实体 F1（P/R 调和） | ↑ 越高越好 | 0.00 |
| entity_precision | 候选实体命中 gold 占比 | ↑ 越高越好 | 0.00 |
| entity_recall | gold 实体被召回占比 | ↑ 越高越好 | 0.00 |

**语义 F1（LLM）**

| 指标 | 说明 | 方向 | 得分 |
|------|------|------|------|
| semantic_entity_f1 | 实体语义 F1（P/R 调和） | ↑ 越高越好 | 90.91 |
| semantic_entity_precision | 候选实体语义命中 gold 占比 | ↑ 越高越好 | 93.33 |
| semantic_entity_recall | gold 实体被语义召回占比 | ↑ 越高越好 | 88.89 |

### 关系抽取

**EM**

| 指标 | 说明 | 方向 | 得分 |
|------|------|------|------|
| triple_f1 | 关系 F1（P/R 调和） | ↑ 越高越好 | 0.00 |
| triple_precision | 候选三元组命中 gold 占比 | ↑ 越高越好 | 0.00 |
| triple_recall | gold 三元组被召回占比 | ↑ 越高越好 | 0.00 |

**语义 F1（LLM）**

| 指标 | 说明 | 方向 | 得分 |
|------|------|------|------|
| semantic_triple_f1 | 三元组语义 F1（P/R 调和） | ↑ 越高越好 | 82.22 |
| semantic_triple_precision | 候选三元组语义命中 gold 占比 | ↑ 越高越好 | 72.22 |
| semantic_triple_recall | gold 三元组被语义召回占比 | ↑ 越高越好 | 100.00 |

### 属性抽取

| 指标 | 说明 | 方向 | 得分 |
|------|------|------|------|
| property_f1 | 属性 F1（P/R 调和） | ↑ 越高越好 | 0.00 |
| property_precision | 候选属性命中 gold 占比 | ↑ 越高越好 | 0.00 |
| property_recall | gold 属性被召回占比 | ↑ 越高越好 | 0.00 |

### Schema 合规

| 指标 | 说明 | 方向 | 得分 |
|------|------|------|------|
| illegal_edge_rate | 违反 schema 的边占比 | ↓ 越低越好 | 11.11 |
| required_property_fill | 主键属性填充完整度 | ↑ 越高越好 | 100.00 |
| type_constraint_pass | 类型落在 gold schema 内占比 | ↑ 越高越好 | 100.00 |

### 抽取忠实度

| 指标 | 说明 | 方向 | 得分 |
|------|------|------|------|
| extraction_faithfulness | 抽取忠实度 | ↑ 越高越好 | 95.24 |

## 证据层

<details><summary>逐样例指标明细（3 个样例）</summary>

| Sample | entity_f1 | entity_precision | entity_recall | semantic_entity_f1 | semantic_entity_precision | semantic_entity_recall | semantic_entity_redundancy | triple_f1 | triple_precision | triple_recall | semantic_triple_f1 | semantic_triple_precision | semantic_triple_recall | semantic_triple_redundancy | type_constraint_pass | required_property_fill | illegal_edge_rate | extraction_faithfulness | property_f1 | property_precision | property_recall |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| 奥德赛用户手册2023款--e9c06954_s0046 | 0.00 | 0.00 | 0.00 | 72.73 | 80.00 | 66.67 | 0.00 | 0.00 | 0.00 | 0.00 | 66.67 | 50.00 | 100.00 | 0.00 | 100.00 | 100.00 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 |
| 奥德赛用户手册2023款--e9c06954_s0060 | 0.00 | 0.00 | 0.00 | 100.00 | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.00 | 100.00 | 100.00 | 0.00 | 100.00 | 100.00 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 |
| 奥德赛用户手册2023款--e9c06954_s0105 | 0.00 | 0.00 | 0.00 | 100.00 | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 80.00 | 66.67 | 100.00 | 0.00 | 100.00 | 100.00 | 33.33 | 85.71 | 0.00 | 0.00 | 0.00 |

</details>

> **冗余率**（列 `semantic_entity_redundancy` / `semantic_triple_redundancy`）
> - 含义：candidate 实体/三元组中，被 LLM 判定语义重复、一对一去重时丢弃的占比
> - 解读：冗余率高 = 抽取系统产出大量语义重复实体/关系（抽取过碎），属抽取侧问题、非匹配错误

<details><summary>元数据</summary>

- Timestamp: 2026-07-17T16:07:46
- Git Commit: 4947e42
- Model: DeepSeek-V4-Pro
- Temperature: 0.0  Seed: 42

</details>
