# 抽取模块

本目录保存两套相互独立的汽车手册图谱抽取方案。

## 模块

- `llm_api/`：基于 LLM API 的抽取方案，包含 LLM review/repair 和可续跑的
  chunk 级中间产物设计。
- `codex_agent/`：基于 Codex-agent 的抽取方案，保存 prompt 和辅助脚本。

两套方案的 raw 输出在概念上保持一致：

```json
{
  "entities": [],
  "relations": []
}
```

raw 结果刻意与 HugeGraph 存储格式分离。需要生成可入库的顶点/边文件时，
使用 `../convert`。

