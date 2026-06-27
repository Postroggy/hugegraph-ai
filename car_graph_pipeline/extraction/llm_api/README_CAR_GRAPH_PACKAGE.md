# LLM API 抽取包说明

本目录是原 `/Users/lzj/proj/car_graph/llm_api_task` 的包内复制版，用来在
`new_hugegraph_ai/ico/hugegraph-ai/car_graph_pipeline` 下独立保存 car-graph
的 LLM API 抽取方案。当前它不依赖 `hugegraph_llm`，后续可以再由
HugeGraph AI 的服务层或 Flow 适配层调用。

## 是否完整

当前 `llm_api/` 包包含 LLM API 抽取方案需要的主要内容：

- `code/extract_llm_api.py`：主 CLI，包含文档发现、chunk 切分、流水线并发、
  LLM 调用、review/repair、JSON/Pydantic 校验、chunk final 落盘、doc raw
  合并和断点续跑。
- `config.yaml`：可提交的默认运行参数，包含模型、并发、重试、token、
  review 阈值和 chunk 大小。
- `settings.py`：配置加载器，负责合并 `config.yaml`、本地
  `config.local.yaml` 和环境变量。
- `requirements.txt`：运行依赖。
- `prompts/compact_few_shots.md`：抽取 prompt 使用的 few-shot 示例。
- `md_output/`：源 markdown 手册目录，git 中只保留 README，不提交原始文档。
- `output/`：运行时产物目录，git 中只保留 README，实际抽取输出会在运行时生成。

未复制原 `llm_api_task/.venv`、`.env` 和 Python 缓存目录。为了保持提交包干净，
复制过来的 `md_output/` 原始文档、历史 `output/` 和 `output_benchmark/` 产物也已清理；
源数据仍保留在原 `/Users/lzj/proj/car_graph/llm_api_task` 目录：

```text
/Users/lzj/proj/car_graph/llm_api_task/md_output
/Users/lzj/proj/car_graph/llm_api_task/output
/Users/lzj/proj/car_graph/llm_api_task/output_benchmark
```

## 运行配置

密钥不在本包代码中配置。`extract_llm_api.py` 从
`car_graph_pipeline.config` 读取：

- `LLM_API_KEY`
- `LLM_BASE_URL`

推荐在 HugeGraph AI checkout 根目录创建本地 `.env`：

```text
LLM_API_KEY=你的本地key
LLM_BASE_URL=https://oneapi-comate.baidu-int.com/v1
```

该 `.env` 已被仓库 `.gitignore` 忽略，不应提交。也可以直接使用环境变量：

```bash
export LLM_API_KEY='你的本地key'
export LLM_BASE_URL='https://oneapi-comate.baidu-int.com/v1'
```

运行参数默认值集中在 `config.yaml`。`settings.py` 的合并优先级是：

```text
CLI 参数 > 环境变量 > config.local.yaml > config.yaml > 代码兜底默认值
```

`config.local.yaml` 可用于本地覆盖，已经被 `.gitignore` 忽略，不要提交。
也可以用环境变量覆盖：

```bash
export CAR_GRAPH_LLM_MODEL='DeepSeek-V4-Flash'
export CAR_GRAPH_LLM_WORKERS=8
export CAR_GRAPH_LLM_API_RETRIES=4
export CAR_GRAPH_LLM_MAX_TOKENS_EXTRACT=40000
export CAR_GRAPH_LLM_MAX_TOKENS_REVIEW=40000
export CAR_GRAPH_LLM_ADAPTIVE_TOKEN_STEPS='40000,50000,70000'
export CAR_GRAPH_CHUNK_TARGET_CHARS=2000
export CAR_GRAPH_CHUNK_MAX_CHARS=2800
export CAR_GRAPH_CHUNK_OVERLAP_CHARS=200
```

CLI 参数优先级高于默认配置，例如 `--workers`、
`--max-tokens-extract`、`--max-tokens-review` 会覆盖对应默认值。

## 运行命令

从 `new_hugegraph_ai/ico/hugegraph-ai` 根目录执行。

先安装依赖：

```bash
python3 -m pip install -r car_graph_pipeline/extraction/llm_api/requirements.txt
```

试跑 1 个文档、最多 5 个 chunk：

```bash
python3 car_graph_pipeline/extraction/llm_api/code/extract_llm_api.py \
  --limit-docs 1 \
  --limit-chunks-per-doc 5 \
  --workers 2 \
  --review-policy auto
```

全量抽取：

```bash
python3 car_graph_pipeline/extraction/llm_api/code/extract_llm_api.py \
  --workers 8 \
  --review-policy auto
```

只切分不调用 LLM：

```bash
python3 car_graph_pipeline/extraction/llm_api/code/extract_llm_api.py --prepare-only
```

按文件名过滤：

```bash
python3 car_graph_pipeline/extraction/llm_api/code/extract_llm_api.py \
  --doc "帝豪GS" \
  --workers 5
```

## 输出约定

每个文档的合并 raw 结果写入：

```text
output/doc_raw_results/<doc_run_id>.json
```

`output/` 已在 `.gitignore` 中忽略，除了 `output/README.md` 外不会随代码提交。

格式为：

```json
{
  "entities": [],
  "relations": []
}
```

这还不是 HugeGraph 批量入库格式。需要入库时，用
`car_graph_pipeline/convert` 转成 vertex/edge 文件。

## 与 HugeGraph AI 的边界

当前这个包是 car-graph 的独立抽取策略包，不直接调用 HugeGraph AI 内部类。
后续接入时建议新增薄适配层：

1. HugeGraph AI API/Flow 接收请求；
2. 适配层调用本包抽取；
3. 本包输出 raw `entities/relations`；
4. convert/normalizer 转成 HugeGraph 可导入格式；
5. 由 HugeGraph AI 或导入脚本负责写图。
