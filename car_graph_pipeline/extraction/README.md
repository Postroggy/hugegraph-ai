# Extraction

This directory stores two independent car manual graph extraction strategies.

## Modules

- `llm_api/`: API-driven extraction with LLM review/repair and resumable
  chunk-level artifacts.
- `codex_agent/`: prompt and helper-code assets for Codex-agent based manual
  extraction.

Both strategies produce raw car-graph results in the same conceptual shape:

```json
{
  "entities": [],
  "relations": []
}
```

The raw shape is deliberately separate from HugeGraph storage format. Use
`../convert` when vertex/edge import files are needed.

