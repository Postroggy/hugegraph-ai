# 汽车图谱流水线

本包保存汽车场景 GraphRAG 相关的抽取、转换和后续接入 HugeGraph AI 的资产。
它刻意与 `hugegraph_llm` 解耦：当前抽取策略可以独立运行，后续再通过
HugeGraph AI 的服务层或 Flow 适配层调用本包。

## 目录结构

```text
car_graph_pipeline/
  config.py           # car-graph 共享配置，key 从环境变量或本地 .env 读取
  extraction/
    llm_api/       # LLM API 抽取方案；原始文档和输出都是本地运行时文件
    codex_agent/   # Codex-agent 抽取方案代码和 prompt
  convert/         # raw 抽取结果转 HugeGraph 顶点/边格式
```

## 抽取方案

### `extraction/llm_api`

这是从原 `/Users/lzj/proj/car_graph/llm_api_task` 工作区复制整理出的
API 驱动抽取方案，包含：

- `code/extract_llm_api.py`：抽取、review、repair、校验、合并和断点续跑主流程。
- `config.yaml`：可提交的默认运行配置，包含模型、并发、重试、token、review、
  chunk 等参数。
- `settings.py`：配置加载器，负责合并 `config.yaml`、被忽略的
  `config.local.yaml` 和环境变量。
- `prompts/`：抽取 prompt 使用的 few-shot 示例。
- `md_output/`：本地原始 markdown 手册目录，git 中只保留 README。
- `output/`：本地运行输出目录，git 中只保留 README。

该方案输出的文档 raw 结果形如：

```json
{
  "entities": [],
  "relations": []
}
```

它不会直接写入 HugeGraph。

为保证提交包干净，复制过来的 `md_output/`、`output/` 和
`output_benchmark/` 产物已经从本包清理。源数据仍保留在原
`/Users/lzj/proj/car_graph/llm_api_task` 工作区：

```text
/Users/lzj/proj/car_graph/llm_api_task/md_output
/Users/lzj/proj/car_graph/llm_api_task/output
/Users/lzj/proj/car_graph/llm_api_task/output_benchmark
```

本流程通过 `car_graph_pipeline.config` 读取密钥和共享服务地址。配置仍采用
环境变量或本地 `.env` 加载方式，不要在 Python 文件中硬编码 API key。

### `extraction/codex_agent`

该目录保存从 `/Users/lzj/proj/car_graph/all_doc/docs_tasks` 复制出的
Codex-agent 抽取 SOP、任务 prompt 和辅助脚本。这里不复制原始文档和任务输出，
只保留可复用的流程资产。

## 格式转换

`convert/` 是从 car-graph raw 抽取结果到 HugeGraph 顶点/边文件的适配层。
抽取模块只决定有哪些事实；转换模块负责把这些事实序列化成图存储格式。

其中：

- `convert/to_hugegraph_v2.py`：较新的车型作用域转换器，适配带
  `vehicle_model` 的 schema。
- `convert/to_hugegraph.py`：旧版无车型作用域转换器，保留作参考和兼容。

## 与 HugeGraph AI 的接入边界

后续接入 HugeGraph AI 时，建议新增一层薄适配层或 Flow：

1. HugeGraph AI API/Flow 接收图抽取请求；
2. 适配层调用这里的汽车领域抽取流程；
3. 本包负责车型作用域归一化和格式校验；
4. 需要入库时再把 raw 结果转换为 HugeGraph 顶点/边格式。

不要把本包伪装成 `hugegraph_llm` 包，也不要在抽取流程中直接修改 HugeGraph AI
的全局设置。
