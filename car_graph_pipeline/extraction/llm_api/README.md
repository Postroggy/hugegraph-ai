# LLM API 汽车手册图谱抽取任务

源文档目录：`car_graph_pipeline/extraction/llm_api/md_output`

> 该目录在 git 中只保留 README，不提交原始文档。原始文档仍保存在
> `/Users/lzj/proj/car_graph/llm_api_task/md_output`。运行前请复制或软链接需要的
> markdown 手册到本目录。

代码：`car_graph_pipeline/extraction/llm_api/code/extract_llm_api.py`

输出：`car_graph_pipeline/extraction/llm_api/output`

## 当前方案

- 使用 `DeepSeek-V4-Flash`
- 并发默认 `5`，当前可先用 `--workers 8` 做提速试跑
- 抽取默认 `max_tokens=40000`，review 默认 `max_tokens=40000`
- 每次 API 调用只抽一个 flat chunk，不再使用“大块上下文 + 小块抽取”的两层切分
- flat chunk 按章节/段落边界单层聚合，目标约 `2000` 字，最大约 `2800` 字；允许为保留完整小节上下浮动，相邻 chunk 带少量 overlap
- 默认 `--review-policy auto`：extract draft -> JSON/Pydantic 校验 -> 图 schema 程序校验 -> LLM review -> final
- review 按 `action_tag` 分类处理：`pass` 85 分以上直接通过，`minor_accept` 80 分以上直接用草稿，`supplement` 合并增量，`corrected_full` 使用全量修正版，只有 `repair_required` 才重抽
- review 评分按 schema 合法性、证据可靠性、高价值覆盖、关系语义、简洁去噪五项判断；`quality_score` 表示采用 review 动作后的最终质量，80 分以上即可收口
- 同一 chunk 最多连续重抽 2 次；若仍需重抽，优先保存已出现的最高分结构合法结果，没有合法结果则标记为 `deferred_repair_no_legal_candidate` 供后续集中修复
- 并发是流水线式：worker 处理完一个 chunk 才会补上下一个；单个 chunk 校验失败会在同一 worker 内最多尝试 4 轮，不会提前释放名额
- 单个 chunk 如果 extract/review 输出接近当前 `max_tokens`，只对该 chunk 下一轮升档：默认 `40000`，再到 `50000`，仍接近上限再到 `70000`；其他 chunk 默认仍是 `40000`
- 保留 `--review-policy always` 做全审对照，`--review-policy never` 做纯 extract 快速对照
- 抽取和 review 都要求输出严格 JSON；extract 必须是 `{ "entities": [...], "relations": [...] }`
- 不写 HugeGraph

## 配置

运行参数默认值集中在 `config.yaml`，包括模型名、并发数、重试次数、
token 上限、review 阈值和 chunk 大小。`settings.py` 会按以下优先级合并配置：

```text
CLI 参数 > 环境变量 > config.local.yaml > config.yaml > 代码兜底默认值
```

`config.local.yaml` 是本地覆盖文件，已被 `.gitignore` 忽略，不要提交。

默认值可以通过环境变量覆盖，例如：

```bash
export CAR_GRAPH_LLM_MODEL='DeepSeek-V4-Flash'
export CAR_GRAPH_LLM_WORKERS=8
export CAR_GRAPH_LLM_API_RETRIES=4
export CAR_GRAPH_CHUNK_TARGET_CHARS=2000
export CAR_GRAPH_CHUNK_MAX_CHARS=2800
```

LLM key 不写在代码里，由 `car_graph_pipeline.config` 从环境变量或本地
`.env` 读取。推荐在 `new_hugegraph_ai/ico/hugegraph-ai/.env` 中放：

```text
LLM_API_KEY=你的本地key
LLM_BASE_URL=https://oneapi-comate.baidu-int.com/v1
```

`.env` 已被仓库 `.gitignore` 忽略，不要提交。

## 目录

- `output/context_chunks`: 每个文档的 flat chunk 基础切分结果，历史目录名保留为 context_chunks，但不再向 LLM 发送额外大上下文
- `output/chunk_tasks`: 每个文档的 flat chunk 抽取任务列表
- `output/doc_runs/<doc_run_id>/chunk_extract`: 每个 chunk 的抽取草稿
- `output/doc_runs/<doc_run_id>/chunk_review`: 每个 chunk 的 LLM review
- `output/doc_runs/<doc_run_id>/chunk_final`: 每个 chunk 的最终结果
- `output/doc_runs/<doc_run_id>/doc_raw`: 单文档合并结果
- `output/doc_raw_results`: 全局单文档合并结果副本
- `output/failed`: 失败 chunk 汇总
- `output/logs`: 运行日志

## 常用命令

如果当前 Python 环境没有 OpenAI SDK，先安装依赖：

```bash
python3 -m pip install -r car_graph_pipeline/extraction/llm_api/requirements.txt
```

先试跑 1 个文档、最多 5 个 chunk：

```bash
python3 car_graph_pipeline/extraction/llm_api/code/extract_llm_api.py --limit-docs 1 --limit-chunks-per-doc 5 --workers 2 --review-policy auto
```

全量抽取：

```bash
python3 car_graph_pipeline/extraction/llm_api/code/extract_llm_api.py --workers 8 --review-policy auto
```

全量抽取会复用已有 `chunk_final`，中断后可直接再次执行同一命令续跑。

默认续跑规则：

- 不加 `--force` 时，已经存在 `output/doc_runs/<doc>/chunk_final/*.final.json` 的 chunk 会自动跳过
- 中断后再次执行同一命令，会只提交尚未生成 final 的 chunk
- 如需从整篇文档的某个全局 chunk 序号开始，可加 `--start-chunk-index N`
- 如需只测试某个全局 chunk，可加 `--chunk-index N`，可重复传多个
- `--force` 会忽略已有 final 并重跑选中的 chunk

如遇到超长表格/复杂章节导致输出被截断，可进一步加大输出余量：

```bash
python3 car_graph_pipeline/extraction/llm_api/code/extract_llm_api.py --workers 8 --review-policy auto --max-tokens-extract 40000 --max-tokens-review 40000
```

只切分不调用 LLM：

```bash
python3 car_graph_pipeline/extraction/llm_api/code/extract_llm_api.py --prepare-only
```

只测试单个内容丰富 chunk：

```bash
python3 car_graph_pipeline/extraction/llm_api/code/extract_llm_api.py --doc '长安UNI-T用户手册' --chunk-index 2 --workers 1
```

按文件名过滤：

```bash
python3 car_graph_pipeline/extraction/llm_api/code/extract_llm_api.py --doc "帝豪GS" --workers 5
```

重跑已有 final：

```bash
python3 car_graph_pipeline/extraction/llm_api/code/extract_llm_api.py --doc "帝豪GS" --force
```
