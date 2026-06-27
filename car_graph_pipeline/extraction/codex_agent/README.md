# Codex-Agent Extraction Assets

This directory stores the reusable Codex-agent extraction workflow assets copied
from `/Users/lzj/proj/car_graph/all_doc/docs_tasks`.

## Contents

- `goal_mode_prompt_template.md`: shared goal-mode prompt template.
- `tasks/extract_task_*/goal_mode_prompt_*.md`: task-specific full prompts.
- `tasks/extract_task_*/goal_start_short_*.md`: short launch prompts.
- `tasks/extract_task_*/task_*_code/`: task helper scripts.

Source documents and task outputs are intentionally not copied here. The
original task workspaces keep those large and run-specific files.

## Intended Use

Use these assets when a Codex session is responsible for document-level
extraction orchestration:

1. prepare chunks,
2. classify risk,
3. let the document-level extractor produce per-chunk final files,
4. merge chunk finals into one document raw result,
5. validate the document raw result.

This workflow is a strategy reference and operator playbook. It is not wired
into HugeGraph AI APIs yet.

