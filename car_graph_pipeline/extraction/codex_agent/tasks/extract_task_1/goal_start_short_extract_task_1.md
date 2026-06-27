你现在进入 goal 模式。

这是短版启动 prompt。最高优先级、最完整的 goal 任务命令在：
`/Users/lzj/proj/car_graph/all_doc/docs_tasks/extract_task_1/goal_mode_prompt_extract_task_1.md`

你必须先读取并遵守该文件；如果本短版与该文件冲突，以该文件为准。

目标：执行 `/Users/lzj/proj/car_graph/all_doc/docs_tasks/extract_task_1` 的汽车说明书 GraphRAG 抽取任务。

你只负责这个目录：
`/Users/lzj/proj/car_graph/all_doc/docs_tasks/extract_task_1`

原始文档目录：`/Users/lzj/proj/car_graph/all_doc/docs_tasks/extract_task_1/task_1_doc`

不要读取、修改或写入其他 `extract_task_*` 目录。不要写 HugeGraph，不要覆盖 `output/new_doc_agent_v3_semantic`。

先读取这些本地文件，再开始执行。第 1 个文件是最高优先级详细任务命令：

1. `/Users/lzj/proj/car_graph/all_doc/docs_tasks/extract_task_1/goal_mode_prompt_extract_task_1.md`
2. `/Users/lzj/proj/car_graph/all_doc/docs_tasks/extract_task_1/task_1_output/task_manifest.json`
3. `/Users/lzj/proj/car_graph/all_doc/docs_tasks/extract_task_1/task_1_code/README_CODE.md`
4. `/Users/lzj/proj/car_graph/all_doc/docs_tasks/extract_task_1/task_1_output/README_OUTPUT_LAYOUT.md`
5. `/Users/lzj/proj/car_graph/all_doc/docs_tasks/extract_task_1/task_1_output/examples/car_manual_graph_extract_few_shots.v1.json`
6. `/Users/lzj/proj/car_graph/car_graph_pipeline/extract/chunk_gate_extraction_sop.md`

执行原则以第 1 个完整 prompt 为准。不要停留在方案阶段。

关键硬约束：

- 每个 doc 一个 extractor，负责全部 chunk 顺序抽取。
- 每个 chunk 抽完必须立即落盘，不允许最后凭记忆写整份结果。
- 低风险 chunk：extractor 自审后写 final。
- 高/中风险 chunk：进入 reviewer gate；extractor 必须等待 reviewer pass/fail/block 后才能继续下一个 chunk。
- 每个文档最终 raw 必须由固定代码从 `chunk_final/*.final.json` 合并。
- blocked chunk 记录到本 task 输出目录后继续后续文档。

常用起步命令：

```bash
cd /Users/lzj/proj/car_graph/all_doc/docs_tasks/extract_task_1/task_1_code
python3 task_pipeline.py scope
python3 task_pipeline.py all_prep --limit 1
python3 task_pipeline.py summarize
```
