你现在进入 goal 模式。

目标：完成 `/Users/lzj/proj/car_graph/all_doc/docs_tasks/extract_task_5` 中汽车说明书的 GraphRAG 抽取准备、chunk 抽取、chunk review/self-review、doc raw 合并、程序化校验和结果汇总。

你只负责当前 task 目录：
`/Users/lzj/proj/car_graph/all_doc/docs_tasks/extract_task_5`

不要读取、修改或写入其他 `extract_task_*` 目录。

## 关键路径

- 原始文档目录：`/Users/lzj/proj/car_graph/all_doc/docs_tasks/extract_task_5/task_5_doc`
- 本任务代码目录：`/Users/lzj/proj/car_graph/all_doc/docs_tasks/extract_task_5/task_5_code`
- 本任务输出目录：`/Users/lzj/proj/car_graph/all_doc/docs_tasks/extract_task_5/task_5_output`
- 本任务 manifest：`/Users/lzj/proj/car_graph/all_doc/docs_tasks/extract_task_5/task_5_output/task_manifest.json`
- 车型映射文件：`/Users/lzj/proj/car_graph/all_doc/docs_tasks/extract_task_5/task_5_output/vehicle_scope_map.json`
- 车型映射待审文件：`/Users/lzj/proj/car_graph/all_doc/docs_tasks/extract_task_5/task_5_output/scope_review_needed.json`
- 本任务 few-shot 示例：`/Users/lzj/proj/car_graph/all_doc/docs_tasks/extract_task_5/task_5_output/examples/car_manual_graph_extract_few_shots.v1.json`

可参考但不要直接修改的原始项目代码：

- `/Users/lzj/proj/car_graph/car_graph_pipeline/extract/semantic_agent_extract_v3.py`
- `/Users/lzj/proj/car_graph/car_graph_pipeline/extract/agent_extract_v2.py`
- `/Users/lzj/proj/car_graph/car_graph_pipeline/extract/chunk_gate_extraction_sop.md`
- `/Users/lzj/proj/car_graph/car_graph_pipeline/extract/few_shot_examples/car_manual_graph_extract_few_shots.v1.json`
- `/Users/lzj/proj/car_graph/car_schema_pure_with_vehicle_scope.groovy`

## 先读取

开始前只读取这些文件/目录做上下文：

1. `task_5_output/task_manifest.json`
2. `task_5_code/README_CODE.md`
3. `/Users/lzj/proj/car_graph/car_graph_pipeline/extract/chunk_gate_extraction_sop.md`
4. `task_5_output/README_OUTPUT_LAYOUT.md`
5. `task_5_output/examples/car_manual_graph_extract_few_shots.v1.json`

确认当前 task 有 204 个文档后，直接开始执行，不要停留在方案阶段。

## 强约束

- 所有代码改造都放在 `task_5_code`，不要直接修改原始 `car_graph_pipeline` 代码。
- 所有结果都放在 `task_5_output`，不要写入其他 task 输出目录。
- 不要写 HugeGraph，不要写当前 `car_graph` 图。
- 不要覆盖 v3 结果目录 `output/new_doc_agent_v3_semantic`。
- 每个文档独立使用 `task_5_output/doc_runs/<doc_run_id>/`。
- 每个 chunk 抽取完必须立即落盘，不允许最后凭记忆写整份 doc raw。
- 最终每个文档的 doc raw 必须由固定代码从 `chunk_final/*.final.json` 合并生成。
- 不要把参考答案、旧评测结果或其他 task 的结果当作抽取依据。
- 遇到 blocked chunk，不要卡死整个 task；记录 blocked，继续后续文档。

## 已预创建目录

task 级目录：

- `task_5_output/semantic_chunks`
- `task_5_output/risk_classification`
- `task_5_output/examples`
- `task_5_output/doc_runs`
- `task_5_output/doc_raw_results`
- `task_5_output/validated_results`
- `task_5_output/analysis`
- `task_5_output/blocked`
- `task_5_output/checkpoints`
- `task_5_output/merged`
- `task_5_output/locks`
- `task_5_output/prompts`
- `task_5_output/run_logs`

每个文档目录：

- `task_5_output/doc_runs/<doc_run_id>/source_info.json`
- `task_5_output/doc_runs/<doc_run_id>/chunks`
- `task_5_output/doc_runs/<doc_run_id>/chunk_drafts`
- `task_5_output/doc_runs/<doc_run_id>/chunk_reviews`
- `task_5_output/doc_runs/<doc_run_id>/chunk_self_reviews`
- `task_5_output/doc_runs/<doc_run_id>/chunk_final`
- `task_5_output/doc_runs/<doc_run_id>/doc_raw`
- `task_5_output/doc_runs/<doc_run_id>/state`
- `task_5_output/doc_runs/<doc_run_id>/logs`
- `task_5_output/doc_runs/<doc_run_id>/intents`
- `task_5_output/doc_runs/<doc_run_id>/scratch`

不要自行发明新的主目录。确实需要新目录时，放在当前 doc_run 的 `scratch/` 或 task 级 `analysis/` 下。

## 已提供的 task 内固定代码

`task_5_code` 中已经提供以下固定脚本。优先运行/小修这些脚本，不要从零重写，也不要改原始 `car_graph_pipeline` 代码：

1. `task_pipeline.py`
   - 统一 CLI 入口，支持 `scope`、`prepare_chunks`、`classify_risk`、`init_state`、`merge_doc_raw`、`validate_doc_raw`、`summarize`、`all_prep`。

2. `prepare_chunks.py`
   - 参考 `semantic_agent_extract_v3.py` 的 clean/split/chunk 思路。
   - 输入当前 task 的 `task_5_doc`。
   - 输出每个文档的 `doc_runs/<doc_run_id>/chunks/chunks.json`。

3. `classify_chunk_risk.py`
   - 给每个 chunk 生成 risk。
   - 输出 `doc_runs/<doc_run_id>/chunks/risk.json`。
   - task 级汇总输出 `risk_classification/<doc_run_id>.risk.json`。
   - 同时给出 `recommended_doc_flow` 和每个 chunk 的 `recommended_flow`。`recommended_doc_flow` 只表示当前 doc 是否需要配 reviewer，不表示把同一文档拆成两套 extractor 上下文。

4. `init_doc_state.py`
   - 根据 chunks、risk、vehicle scope 初始化 `doc_runs/<doc_run_id>/state/state.json`。

5. `merge_doc_raw.py`
   - 从 `chunk_final/*.final.json` 合并文档 raw。
   - 输出 `doc_runs/<doc_run_id>/doc_raw/<doc_run_id>.raw.json`。
   - 同步复制到 `doc_raw_results/<doc_run_id>.raw.json`。

6. `validate_doc_raw.py`
   - 兼容 raw 方言并做 schema/端点/关系方向/车型属性检查。
   - 输出 `validated_results/<doc_run_id>.validated.json`。

7. `summarize_progress.py`
   - 更新 `analysis/progress.json` 和 `analysis/progress_report.md`。

常用命令示例：

```bash
cd /Users/lzj/proj/car_graph/all_doc/docs_tasks/extract_task_5/task_5_code
python3 task_pipeline.py scope
python3 task_pipeline.py all_prep --limit 1
python3 task_pipeline.py merge_doc_raw --doc-run-id '<doc_run_id>'
python3 task_pipeline.py validate_doc_raw --doc-run-id '<doc_run_id>'
python3 task_pipeline.py summarize
```

## few-shot 示例要求

抽取和 review 前必须读取：

`task_5_output/examples/car_manual_graph_extract_few_shots.v1.json`

该文件当前状态为 `agent_reviewed_pending_user_review`，可以作为风格、覆盖和 reviewer 判定参考，但不能把示例事实迁移到当前文档。用户确认后才能改为 `approved`。

重点参考示例覆盖：

- `operation_steps`：操作步骤、控制件、被操作部件、警告。
- `operation_plus_spec_warning`：加油步骤、材料、规格、状态/故障。
- `spec_table`：表格参数拆成带条件的 Specification。
- `warning_status_table`：警告灯、状态、故障、处理动作。
- `menu_button_path`：按钮/菜单路径/显示模式。
- `mixed_operations`：同一 chunk 多任务拆分，例如应急启动和电池更换不能混成一个 Operation。

## 车型映射要求

抽取前必须先为当前 task 生成或补全：

`task_5_output/vehicle_scope_map.json`

格式：

```json
{
  "A5L用户手册.md": {
    "vehicle_brand": "奥迪",
    "vehicle_model": "A5L",
    "confidence": "high",
    "source": "filename"
  }
}
```

规则：

- `vehicle_model` 默认从文件名 stem 规范化得到。
- `vehicle_brand` 优先从明确品牌词、文件名前缀、常见品牌词典推断。
- 不确定的项写入 `scope_review_needed.json`，不要静默乱填。
- 合并/校验阶段必须统一覆盖每个实体和关系的 `vehicle_brand` / `vehicle_model`。

## 批处理与锁

不要一次性把 204 个文档全部塞进同一轮工作。

推荐每批处理 1-2 个 doc-run：

1. 从 `task_manifest.json` 中选择 status 为 `pending` 的 doc-run。
2. 为每个 doc-run 创建 lock：
   - `task_5_output/locks/<doc_run_id>.lock`
3. 如果 lock 已存在且未超时，跳过该 doc-run。
4. 每完成一个 doc-run，更新：
   - `checkpoints/completed_docs.json`
   - `analysis/progress.json`
5. 每批完成后更新：
   - `analysis/progress_report.md`

如果会话中断，新会话根据 lock、checkpoint、state 恢复。

## 文档内统一 extractor 与阻塞式 reviewer gate

先对每个 doc 的所有 chunk 做风险分类。风险分类不用于把同一个 doc 拆给不同 extractor，而是用于决定每个 chunk 是否需要 reviewer gate。

每个 doc 必须只有一个 extractor agent。这个 extractor 负责整篇文档的完整上下文、术语、别名和实体消歧，并按 chunk 顺序抽取。

不要在同一个 doc 内拆成多个 extractor。否则会造成术语、别名、实体消歧和上下文记忆割裂。

reviewer 是当前 doc 的 gate worker，不负责抽取，不写 final。reviewer 只审状态为 `awaiting_review` 的 chunk。

高风险和中风险 chunk 必须进入阻塞式 reviewer gate：

- `operation`
- `warning`
- `spec_table`
- 含按钮、开关、旋钮、菜单、显示屏。
- 含指示灯、警告灯、故障、蜂鸣器、闪烁、点亮、异常。
- 含油液、胎压、容量、维护周期、规格、单位数值。
- 含安全带、气囊、儿童座椅、制动、驻车、辅助驾驶。

低风险 chunk 可由同一个 extractor 自审后直接 final：

- 背景介绍。
- 目录、版权、章节过渡。
- 无操作、无参数、无警告、无状态、无部件密集信息。

risk 输出格式：

```json
{
  "chunk_id": "...",
  "risk_level": "high|medium|low",
  "risk_score": 0,
  "reasons": [],
  "recommended_flow": "reviewer_gate|self_review"
}
```

doc 级 risk 输出还包含：

```json
{
  "recommended_doc_flow": "single_extractor_selective_reviewer|single_agent_self_review"
}
```

## 阻塞式 reviewer gate

当 `recommended_doc_flow=single_extractor_selective_reviewer` 时，整个 doc 使用一个 extractor agent 顺序抽取所有 chunk，并配一个 reviewer agent 审核需要 gate 的 chunk。

对 `recommended_flow=reviewer_gate` 的 chunk：

- extractor 写 `chunk_drafts/<chunk_id>.attempt<N>.json`
- extractor 更新 `state.json`：`phase=awaiting_review`，并写入 `draft_path`
- extractor 必须等待 reviewer 结果，不能提前抽下一个 chunk
- reviewer 写 `chunk_reviews/<chunk_id>.attempt<N>.review.json`
- reviewer pass：状态改为 `passed`，extractor 写 `chunk_final/<chunk_id>.final.json`，然后推进下一个 chunk
- reviewer fail：状态改为 `needs_fix`，extractor 只修当前 chunk 并再次提交 reviewer
- max attempts 后写 blocked，记录原因，然后 extractor 才能推进后续 chunk/doc

也就是说，高风险/中风险 chunk 是同步 gate：同一个 extractor 的 chunk 序列必须停在当前 chunk，直到 reviewer pass/fail/block 处理完毕。

## reviewer 无响应兜底

extractor 等待 reviewer 时不能无限卡住。对 `phase=awaiting_review` 的 chunk：

- 默认等待 reviewer 的轮询间隔不得低于一次正常 reviewer 响应时间，建议 `fallback_poll_seconds=180` 秒。
- 如果连续 3 次轮询都没有新的 review 文件，也没有 state 变化，extractor 可以执行结构化 self-review 兜底。
- 兜底 self-review 必须写入 `chunk_self_reviews/<chunk_id>.attempt<N>.fallback_self_review.json`。
- 兜底通过后，extractor 才能写 `chunk_final/<chunk_id>.final.json`，并在 final 中标记 `review_status=fallback_self_review_pass`、`finalized_from_attempt=N`。
- 兜底不通过时，extractor 只修当前 chunk；达到 `max_attempts` 后写 blocked，然后继续后续 chunk/doc。
- 兜底不是跳过 reviewer gate，而是 reviewer 无响应时的超时降级；如果 reviewer 已返回 fail/block，必须优先按 reviewer 结果处理。

主会话不要频繁介入 doc 内部流程。主会话只做健康检查：

- 默认每 15 分钟检查一次 active doc-run 的 `state.json`、draft/review/final 数量和最近更新时间。
- 仅在 stale、blocked、JSON 损坏、agent 退出或长时间无文件变化时介入。
- 不读取 chunk 正文，不审阅 draft/review 内容，避免主会话上下文膨胀。

如果当前环境不方便长时间维护双 agent 自动状态机，可以由主会话按 SOP 调度，但仍必须按文件落盘。

## 单 agent 自审

当 `recommended_doc_flow=single_agent_self_review` 时，整个 doc 使用一个 extractor/self-review agent 顺序处理所有 chunk。

对 `recommended_flow=self_review` 的 chunk：

- 写 `chunk_drafts/<chunk_id>.attempt1.json`
- 写 `chunk_self_reviews/<chunk_id>.attempt1.review.json`
- pass 后写 `chunk_final/<chunk_id>.final.json`

self review 必须是结构化 JSON，不允许只在内心检查。

## 中间结果 JSON 标准字段

Entity：

```json
{
  "type": "Component",
  "name": "...",
  "aliases": [],
  "properties": {},
  "chunk_id": "...",
  "heading_path": "...",
  "source_section": "...",
  "line_start": 1,
  "line_end": 2,
  "source_snippet": "...",
  "evidence": "..."
}
```

Relation：

```json
{
  "type": "OPERATES_ON",
  "source_type": "Operation",
  "source_name": "...",
  "target_type": "Component",
  "target_name": "...",
  "properties": {},
  "chunk_id": "...",
  "heading_path": "...",
  "source_section": "...",
  "line_start": 1,
  "line_end": 2,
  "source_snippet": "...",
  "evidence": "..."
}
```

## Schema 要求

使用：

`/Users/lzj/proj/car_graph/car_schema_pure_with_vehicle_scope.groovy`

实体/关系类型和关系方向参考：

`/Users/lzj/proj/car_graph/car_graph_pipeline/extract/agent_extract_v2.py`

最终每个实体和关系都必须有：

- `vehicle_brand`
- `vehicle_model`

抽取阶段可以缺失或写错，合并/校验代码必须统一覆盖。

## 合并代码必须兼容 raw 方言

至少兼容：

- `label -> type`
- `entity_type -> type`
- 从 `properties.<name_prop>` 补 `name`
- 从 `source/target` 或 `source_entity_id/target_entity_id` 解析关系端点
- 顶层属性归入 `properties`
- 强制覆盖 `vehicle_brand` / `vehicle_model`

## 进度与交付

定期更新：

- `analysis/progress.json`
- `analysis/progress_report.md`
- `analysis/validation_summary.json`

最终交付：

- `doc_raw_results/*.raw.json`
- `validated_results/*.validated.json`
- `analysis/validation_summary.json`
- `analysis/progress_report.md`
- `blocked/*.blocked.json` 如有

## 完成标准

当前 task 的所有 doc-run 都满足以下之一：

1. `completed`：已有 chunks、risk、chunk_final、doc raw、validated result。
2. `blocked`：已有 blocked 文件，记录 chunk、attempt、原因、待人工处理建议。

完成后给出简短报告：

- total_docs
- completed_docs
- blocked_docs
- total_chunks
- completed_chunks
- blocked_chunks
- 主要 blocked 原因
- 输出目录路径

现在开始：先检查当前 task 目录、manifest、vehicle_scope_map 状态，然后直接开始实现/抽取。
