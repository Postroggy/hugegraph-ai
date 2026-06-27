# Codex-Agent 抽取资产

本目录保存从 `/Users/lzj/proj/car_graph/all_doc/docs_tasks` 复制出的
Codex-agent 抽取流程资产。

## 内容

- `goal_mode_prompt_template.md`：共享的 goal 模式 prompt 模板。
- `tasks/extract_task_*/goal_mode_prompt_*.md`：每个任务的完整 prompt。
- `tasks/extract_task_*/goal_start_short_*.md`：每个任务的短启动 prompt。
- `tasks/extract_task_*/task_*_code/`：每个任务的辅助脚本。

这里不复制原始文档和任务输出。那些大文件和运行态文件仍保留在原
`all_doc/docs_tasks` 工作区。

## 预期用法

当一个 Codex 会话负责文档级抽取调度时，可按以下流程使用这些资产：

1. 切分 chunk；
2. 判断 chunk 或文档风险；
3. 让文档级 extractor 生成每个 chunk 的 final 文件；
4. 合并 chunk final 为单文档 raw 结果；
5. 校验单文档 raw 结果。

这套流程目前是策略参考和操作手册，尚未接入 HugeGraph AI API。

