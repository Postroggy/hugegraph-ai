# Car Graph Pipeline

This package keeps the car-domain GraphRAG assets that are being prepared for
integration with HugeGraph AI. It is intentionally isolated from
`hugegraph_llm`: the current extraction strategies can run without importing
HugeGraph AI internals, and later service adapters can call into this package.

## Layout

```text
car_graph_pipeline/
  config.py           # shared car-graph config, keys loaded from env or .env
  extraction/
    llm_api/       # LLM API extraction workflow; docs/outputs are local only
    codex_agent/   # Codex-agent workflow code and prompts only
  convert/         # Raw extraction result -> HugeGraph vertex/edge adapters
```

## Extraction Strategies

### `extraction/llm_api`

This is the current API-driven extraction workflow copied from the original
`/Users/lzj/proj/car_graph/llm_api_task` workspace. It includes:

- `code/extract_llm_api.py`: extraction, review, repair, validation, merge, and
  resume logic.
- `config.yaml`: committed default runtime settings for model, concurrency,
  retry, token, review, and chunk parameters.
- `settings.py`: config loader that merges `config.yaml`, ignored
  `config.local.yaml`, and environment variable overrides.
- `prompts/`: compact few-shot prompt assets.
- `md_output/`: local source markdown manual directory, ignored except for its
  README.
- `output/`: runtime output directory, ignored except for its README.

This workflow writes raw document results shaped as:

```json
{
  "entities": [],
  "relations": []
}
```

It does not write HugeGraph directly.

Historical copied `md_output/`, `output/`, and `output_benchmark/` artifacts
were removed from this package copy before commit. The source copies remain
under the original `/Users/lzj/proj/car_graph/llm_api_task` workspace:

```text
/Users/lzj/proj/car_graph/llm_api_task/md_output
/Users/lzj/proj/car_graph/llm_api_task/output
/Users/lzj/proj/car_graph/llm_api_task/output_benchmark
```

The workflow imports `car_graph_pipeline.config` for secrets and shared service
addresses. The copied config keeps the same environment-variable based secret
loading pattern; do not hardcode API keys in Python files.

### `extraction/codex_agent`

This keeps the Codex-agent extraction SOP and task code copied from
`/Users/lzj/proj/car_graph/all_doc/docs_tasks`. It includes task prompts and
task code only. Source documents and task outputs are intentionally not copied
here, so the package stays focused on reusable workflow assets.

## Conversion

`convert/` is the adapter layer from car-graph raw extraction results to
HugeGraph-friendly vertex/edge files. Keep this separate from extraction:
extraction decides what facts exist, while conversion decides how those facts
are serialized for graph storage.

The newer vehicle-scoped converter is `convert/to_hugegraph_v2.py`. The older
non-scoped converter is kept for reference and compatibility.

## Integration Boundary

For HugeGraph AI integration, prefer adding a thin service or flow adapter that:

1. accepts a graph-extract request from HugeGraph AI,
2. calls the car-domain extraction workflow here,
3. normalizes and validates car scope fields,
4. converts raw results to HugeGraph vertex/edge format when import is needed.

Do not import this package as `hugegraph_llm`, and do not mutate HugeGraph AI
global settings from the extraction workflow.
