# task_5_code

本目录保存 `extract_task_5` 独立使用的抽取/切分/合并代码。不要直接修改原始 `car_graph_pipeline` 代码。

已提供固定脚本：

- `task_pipeline.py`: 统一 CLI 入口。
- `prepare_chunks.py`: 生成 semantic chunks。
- `classify_chunk_risk.py`: chunk 风险分类，并给出文档级 `recommended_doc_flow`。
- `init_doc_state.py`: 初始化 doc-run state。
- `merge_doc_raw.py`: 从 `chunk_final` 合并文档 raw。
- `validate_doc_raw.py`: 程序化校验 doc raw。
- `summarize_progress.py`: 汇总进度。

常用命令：

```bash
cd /Users/lzj/proj/car_graph/all_doc/docs_tasks/extract_task_5/task_5_code
python3 task_pipeline.py scope
python3 task_pipeline.py all_prep --limit 5
python3 task_pipeline.py merge_doc_raw --doc-run-id '<doc_run_id>'
python3 task_pipeline.py validate_doc_raw --doc-run-id '<doc_run_id>'
python3 task_pipeline.py summarize
```

设计原则：主会话只做调度和状态管理，不读完整大文档正文；实际抽取交给 doc 级子 agent。双 agent / 单 agent 的选择是文档级选择，不在同一文档内混用两套上下文。

运行策略补充：

- extractor 等待 reviewer 时不能无限卡住。`awaiting_review` 连续 3 次轮询无 review 文件且 state 无变化时，允许写结构化 fallback self-review 并继续。
- fallback 轮询间隔不得低于一次正常 reviewer 响应时间，建议 180 秒；不能用过短轮询制造假超时。
- 主会话只做 15 分钟级健康检查：看 `state.json`、draft/review/final 数量和最近更新时间；除 stale、blocked、JSON 损坏、agent 退出外，不介入 chunk 内容。
