# 实体消歧包

本目录是 car graph 的离线实体消歧流水线。它处理本地抽取结果 JSON，
不会直接读写 HugeGraph。

推荐顺序：

```text
抽取结果 extracted_entities/extracted_relations
-> disambiguation 生成 merged_entities/merged_relations
-> convert 转 HugeGraph 导入格式
-> 导入 HugeGraph
```

## 输入输出

默认输入：

```text
car_graph_pipeline/output/version/v2_fault_split/
  extracted_entities.json
  extracted_relations.json
```

默认输出：

```text
car_graph_pipeline/output/version/v3_disambiguated/
  candidates.json
  merge_decisions.json
  merged_entities.json
  merged_relations.json
  merge_log.json
  validation_result.json
  run.log
```

实际批量运行时建议显式传入目录：

```bash
python -m car_graph_pipeline.disambiguation.run \
  --input-dir /path/to/extracted_version \
  --output-dir /path/to/disambiguated_version
```

只跑单个阶段：

```bash
python -m car_graph_pipeline.disambiguation.run --phase 1 --input-dir ... --output-dir ...
python -m car_graph_pipeline.disambiguation.run --phase 2 --input-dir ... --output-dir ...
python -m car_graph_pipeline.disambiguation.run --phase 3 --input-dir ... --output-dir ...
python -m car_graph_pipeline.disambiguation.run --phase 4 --input-dir ... --output-dir ...
```

## 关键策略

- Phase 1：按 `(实体类型, vehicle_model)` 分组生成候选对，优先使用实体
  `properties.vehicle_model`，兜底解析 `entity_id` 中的车型。
- Phase 1：同组内用 embedding 相似度和编辑距离筛选候选实体对。
- Phase 2：用 LLM 判断候选对是否合并，默认关闭 thinking。
- Phase 3：使用 Union-Find 执行合并，迁移边端点，去除自环和重复边。
- Phase 4：检查 entity_id 唯一性、悬挂边、实体/关系减少比例、跨车型合并风险。

`VehicleBrand` 和 `VehicleModel` 默认不参与消歧；它们仍然保留为图里的实体点，
其他实体和关系通过 `vehicle_brand` / `vehicle_model` 属性做车型范围隔离。

## 配置

默认配置在 `config.yaml`。本地覆盖可创建被 git 忽略的
`config.local.yaml`，也可以使用环境变量覆盖，例如：

```bash
export CAR_GRAPH_DISAMBIGUATION_LLM_MODEL=DeepSeek-V4-Flash
export CAR_GRAPH_DISAMBIGUATION_LLM_CONCURRENCY=5
export CAR_GRAPH_DISAMBIGUATION_COSINE_THRESHOLD=0.85
```

LLM key 不写入本目录，仍从 `LLM_API_KEY` / `OPENAI_API_KEY` 或
`car_graph_pipeline/.env` 读取。

## 依赖

独立运行本包时可安装：

```bash
pip install -r car_graph_pipeline/disambiguation/requirements.txt
```
