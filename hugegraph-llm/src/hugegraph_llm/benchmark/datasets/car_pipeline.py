# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Adapter: convert car-pipeline extraction artifacts into benchmark extraction format.

Supports three candidate sources:

1. **Pipeline candidate** (chunks-mode run output):
      ``<run_output_dir>/doc_runs/<doc_run_id>/chunk_final/<chunk_id>.final.json``
   Blocked chunks fall back to ``blocked/*.blocked.json`` →
   ``best_observed_candidate``.
   Default ``run_output_dir = artifacts/workflow_runs`` (the pipeline's
   ``--output-dir`` default).

2. **Docs-mode candidate** (docs-mode run output):
   Same layout as chunks-mode above, but chunk_ids carry the docs-mode
   naming ``<doc>--<sha256>_s<idx>``.  Pairing relies on the caller providing
   matching docs-mode gold; the adapter itself can only pair what is available.

3. **Baseline dataset** (flat layout with gold + candidate in the same leaf dir):
      ``<baseline_dir>/<doc>/<chunk_id>_flat/manual_result_full_recall.json``  (gold)
      ``<baseline_dir>/<doc>/<chunk_id>_flat/<candidate_name>.json``            (candidate)
   Candidate name is configurable, e.g. ``api_result.json`` or
   ``codex_agent_workflow_result_gpt54_full_recall_20260701.json``.

All three sources share the same car-pipeline graph schema — vertex
``{type, name, properties}`` and edge
``{type, source_type, source_name, target_type, target_name, properties}`` —
and are mapped losslessly to the benchmark's HugeGraph-native extraction format:
vertex ``{label, name, properties}`` and edge
``{label, outV, outVLabel, inV, inVLabel, properties}``.

Typical usage (pipeline chunks mode)::

    python -m hugegraph_llm.benchmark.datasets.car_pipeline \\
        --gold-dir data/dataset \\
        --run-output-dir artifacts/workflow_runs/2026-07-01_run

Typical usage (baseline dataset)::

    python -m hugegraph_llm.benchmark.datasets.car_pipeline \\
        --mode baseline \\
        --baseline-dir ~/Downloads/baseline_trash \\
        --candidate api_result.json

Then benchmark::

    hugegraph-benchmark run --mode extraction --data <output>.json --offline
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# car-pipeline vertex type -> its identifying property (mirrors the pipeline's
# schema/enums.py ``_NAME_PROPERTY``). Used to set ``vertexlabels.primary_keys``
# so the benchmark's ``schema_validity`` metric scores candidate vertices against
# the correct key rather than a guessed one.
_CAR_NAME_PROPERTY: Dict[str, str] = {
    "VehicleBrand": "brand_name",
    "VehicleModel": "model_name",
    "VehicleSystem": "system_name",
    "Component": "comp_name",
    "Function": "func_name",
    "Status": "status_name",
    "Fault": "fault_name",
    "Operation": "op_name",
    "MaintenanceItem": "maint_name",
    "Specification": "spec_name",
    "Material": "material_name",
    "ImageAsset": "image_id",
}

# Defaults match the car pipeline's own conventions.
DEFAULT_GOLD_DIR = Path("data/dataset")
DEFAULT_RUN_OUTPUT_DIR = Path("artifacts/workflow_runs")
OUTPUT_DIR = Path(__file__).resolve().parents[4] / "benchmark_data" / "external"

_FINAL_SUFFIX = ".final.json"
_BLOCKED_SUFFIX = ".blocked.json"
_GOLD_FILENAME = "manual_result_full_recall.json"


# ---------------------------------------------------------------------------
# Field mapping: car-pipeline graph -> benchmark extraction graph
# ---------------------------------------------------------------------------

def car_vertex_to_benchmark(vertex: Dict[str, Any]) -> Dict[str, Any]:
    """Map a car-pipeline vertex to benchmark format (``type`` -> ``label``)."""
    record: Dict[str, Any] = {
        "label": vertex.get("type", ""),
        "name": vertex.get("name", ""),
        "properties": dict(vertex.get("properties") or {}),
    }
    snippet = vertex.get("source_snippet")
    if snippet:
        record["source_snippet"] = snippet
    return record


def car_edge_to_benchmark(edge: Dict[str, Any]) -> Dict[str, Any]:
    """Map a car-pipeline edge to benchmark format.

    ``type`` -> ``label``, ``source_name`` -> ``outV``, ``source_type`` ->
    ``outVLabel``, ``target_name`` -> ``inV``, ``target_type`` -> ``inVLabel``.
    Endpoint labels are preserved when present; absent ones are omitted so the
    record stays lossless rather than carrying empty placeholders.
    """
    record: Dict[str, Any] = {
        "label": edge.get("type", ""),
        "outV": edge.get("source_name", ""),
        "inV": edge.get("target_name", ""),
        "properties": dict(edge.get("properties") or {}),
    }
    source_type = edge.get("source_type")
    target_type = edge.get("target_type")
    if source_type:
        record["outVLabel"] = source_type
    if target_type:
        record["inVLabel"] = target_type
    snippet = edge.get("source_snippet")
    if snippet:
        record["source_snippet"] = snippet
    return record


def _car_graph_to_benchmark(
    vertices: List[Dict[str, Any]], edges: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return (
        [car_vertex_to_benchmark(v) for v in vertices if isinstance(v, dict)],
        [car_edge_to_benchmark(e) for e in edges if isinstance(e, dict)],
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping unreadable JSON %s: %s", path, exc)
        return None


def load_gold(gold_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Index gold records by ``chunk_id``: ``{chunk_id: {data, dir}}``."""
    gold: Dict[str, Dict[str, Any]] = {}
    for path in sorted(Path(gold_dir).rglob(_GOLD_FILENAME)):
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        chunk_id = data.get("chunk_id") or path.parent.name.replace("_flat", "")
        gold[chunk_id] = {"data": data, "dir": path.parent}
    return gold


# -- Pipeline candidate loader (chunks-mode + docs-mode) --

def load_candidates_from_pipeline(run_output_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Index candidate graphs from pipeline ``chunk_final/`` artifacts.

    Searches ``**/chunk_final/*.final.json``, which covers both chunks-mode
    (``batch_chunks/doc_runs/doc_run_id/chunk_final/``) and docs-mode
    (``<run>/doc_runs/<doc>--<hash>/chunk_final/``) layouts.

    Fallback for blocked chunks: ``**/blocked/*.blocked.json`` →
    ``best_observed_candidate.draft_path`` (best-effort; skipped if the
    path references an absolute location from a different machine).
    """
    candidates: Dict[str, Dict[str, Any]] = {}

    for final_dir in sorted(Path(run_output_dir).rglob("chunk_final")):
        if not final_dir.is_dir():
            continue
        for fp in sorted(final_dir.glob(f"*{_FINAL_SUFFIX}")):
            data = _read_json(fp)
            if not isinstance(data, dict):
                continue
            chunk_id = fp.name[: -len(_FINAL_SUFFIX)]
            candidates[chunk_id] = {
                "vertices": data.get("vertices", []),
                "edges": data.get("edges", []),
            }

    # Fallback: blocked chunks' best observed candidate.
    for blocked_dir in sorted(Path(run_output_dir).rglob("blocked")):
        if not blocked_dir.is_dir():
            continue
        for fp in sorted(blocked_dir.glob(f"*{_BLOCKED_SUFFIX}")):
            chunk_id = fp.name[: -len(_BLOCKED_SUFFIX)]
            if chunk_id in candidates:
                continue
            data = _read_json(fp)
            if not isinstance(data, dict):
                continue
            best = data.get("best_observed_candidate") or {}
            draft_path = best.get("draft_path")
            if not draft_path:
                continue
            draft = _read_json(Path(draft_path))
            if not isinstance(draft, dict):
                continue
            candidates[chunk_id] = {
                "vertices": draft.get("vertices", []),
                "edges": draft.get("edges", []),
            }
    return candidates


# -- Baseline dataset loader (gold + candidate in the same leaf dir) --

def load_baseline(
    baseline_dir: Path, candidate_name: str
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Load gold + candidate from a flat baseline dataset.

    Each leaf directory ``<doc>/<chunk_id>_flat/`` must contain both
    ``manual_result_full_recall.json`` (gold) and ``<candidate_name>.json``
    (candidate).  Chunk directories missing either file are skipped silently
    to support incomplete datasets.

    Args:
        baseline_dir: Root directory containing per-doc subdirectories.
        candidate_name: Candidate JSON filename stem (no extension), e.g.
            ``api_result`` or ``codex_agent_workflow_result_gpt54_full_recall_20260701``.

    Returns:
        ``(gold_index, candidate_index)``, both keyed by ``chunk_id``.
        ``gold_index`` values are ``{"data": ..., "dir": path}``;
        ``candidate_index`` values are ``{"vertices": [...], "edges": [...]}``.
    """
    gold: Dict[str, Dict[str, Any]] = {}
    candidates: Dict[str, Dict[str, Any]] = {}

    # Discover leaf directories that contain a gold file.
    for gold_path in sorted(Path(baseline_dir).rglob(_GOLD_FILENAME)):
        leaf_dir = gold_path.parent
        gold_data = _read_json(gold_path)
        if not isinstance(gold_data, dict):
            continue
        chunk_id = gold_data.get("chunk_id") or leaf_dir.name.replace("_flat", "")
        gold[chunk_id] = {"data": gold_data, "dir": leaf_dir}

        # Candidate file — exact match first, then glob for convenience
        # (e.g. "codex_*gpt54*.json" for date-stamped filenames).
        candidate_path = leaf_dir / f"{candidate_name}.json"
        if not candidate_path.is_file() and ("*" in candidate_name or "?" in candidate_name):
            matches = sorted(leaf_dir.glob(f"{candidate_name}.json"))
            candidate_path = matches[0] if matches else candidate_path

        cand_data = _read_json(candidate_path) if candidate_path.is_file() else None
        if isinstance(cand_data, dict):
            candidates[chunk_id] = {
                "vertices": cand_data.get("vertices", []),
                "edges": cand_data.get("edges", []),
            }

    return gold, candidates


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _input_text_for(gold_entry: Dict[str, Any]) -> str:
    """Return ``chunk_text.md`` beside the gold file; else empty string.

    When no chunk text is available (e.g. docs-mode candidates paired against
    gold entries lacking a ``chunk_text.md`` sibling), returning ``""`` lets
    ``extraction_faithfulness`` fall back to per-item ``source_snippet``
    values.  A synthesized metadata string
    (``doc_name | heading_path | section``) would carry no factual content
    yet be non-empty, which would silently defeat that fallback and turn
    the LLM judge's ``input_text`` into noise.
    """
    chunk_text = gold_entry["dir"] / "chunk_text.md"
    if chunk_text.is_file():
        return chunk_text.read_text(encoding="utf-8").strip()
    return ""


def _build_schema(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Derive benchmark schema ``{vertexlabels, edgelabels}`` from the gold graphs only.

    Only gold vertices/edges contribute, so the schema encodes the
    human-annotated type universe. Candidate types absent from gold are then
    flagged by ``schema_validity.type_constraint_pass`` (and candidate edge
    labels by ``illegal_edge_rate``) rather than being silently self-endorsed
    by inclusion in the schema.
    """
    vertex_types: Dict[str, str] = {}
    edge_types: Dict[str, Tuple[str, str]] = {}
    for sample in samples:
        for vertex in sample["gold_vertices"]:
            label = vertex.get("label")
            if label and label not in vertex_types:
                vertex_types[label] = _CAR_NAME_PROPERTY.get(label, "name")
        for edge in sample["gold_edges"]:
            edge_label = edge.get("label")
            if edge_label and edge_label not in edge_types:
                edge_types[edge_label] = (edge.get("outVLabel", ""), edge.get("inVLabel", ""))
    vertexlabels = [{"name": name, "primary_keys": [pk]} for name, pk in vertex_types.items()]
    edgelabels = [
        {"name": name, "source_label": src, "target_label": tgt}
        for name, (src, tgt) in edge_types.items()
    ]
    return {"vertexlabels": vertexlabels, "edgelabels": edgelabels}


def _save(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_samples(
    gold: Dict[str, Dict[str, Any]],
    candidates: Dict[str, Dict[str, Any]],
    subset_size: Optional[int],
) -> Tuple[List[Dict[str, Any]], int, List[str], List[str]]:
    """Pair gold and candidate entries by chunk_id, applying field mapping.

    Gold chunks without a matching candidate are still emitted (with empty
    ``candidate_*``) so the benchmark can surface "extraction produced nothing"
    cases.  Returns ``(samples, matched_count, missing_chunks, empty_chunks)``
    where ``missing_chunks`` are gold chunks with no candidate file and
    ``empty_chunks`` are those whose candidate graph is empty — both are
    surfaced in the report so the reader can locate exactly where data is
    missing.
    """
    samples: List[Dict[str, Any]] = []
    matched = 0
    missing_chunks: List[str] = []
    empty_chunks: List[str] = []
    for chunk_id, gold_entry in sorted(gold.items()):
        gold_vertices, gold_edges = _car_graph_to_benchmark(
            gold_entry["data"].get("vertices", []),
            gold_entry["data"].get("edges", []),
        )
        if chunk_id in candidates:
            candidate = candidates[chunk_id]
            candidate_vertices, candidate_edges = _car_graph_to_benchmark(
                candidate.get("vertices", []),
                candidate.get("edges", []),
            )
            if candidate_vertices or candidate_edges:
                matched += 1
            else:
                empty_chunks.append(chunk_id)
        else:
            missing_chunks.append(chunk_id)
            candidate_vertices, candidate_edges = [], []
        samples.append(
            {
                "sample_id": chunk_id,
                "input_text": _input_text_for(gold_entry),
                "gold_vertices": gold_vertices,
                "gold_edges": gold_edges,
                "candidate_vertices": candidate_vertices,
                "candidate_edges": candidate_edges,
            }
        )
        if subset_size and len(samples) >= subset_size:
            break
    return samples, matched, missing_chunks, empty_chunks


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def prepare_from_pipeline(
    gold_dir: Path = DEFAULT_GOLD_DIR,
    run_output_dir: Path = DEFAULT_RUN_OUTPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    subset_size: Optional[int] = None,
    output_name: str = "car_pipeline_extraction.json",
) -> Path:
    """Convert pipeline run artifacts (chunks-mode or docs-mode) to benchmark format.

    Gold and candidate are loaded from separate directory hierarchies and
    paired by ``chunk_id``.

    Args:
        gold_dir: Root of gold annotations.
        run_output_dir: Root of the pipeline run output
            (``artifacts/workflow_runs/<run_id>``).
        output_dir: Where to write the benchmark JSON.
        subset_size: Limit to the first N paired samples.
        output_name: Output filename (default: ``car_pipeline_extraction.json``).

    Returns:
        Path to the generated benchmark JSON file.
    """
    gold = load_gold(Path(gold_dir))
    candidates = load_candidates_from_pipeline(Path(run_output_dir))

    if not gold:
        raise FileNotFoundError(f"No gold annotations found under {gold_dir}")

    samples, matched, missing_chunks, empty_chunks = _build_samples(gold, candidates, subset_size)
    result = {
        "schema": _build_schema(samples),
        "samples": samples,
        "meta": {
            "gold_source": str(Path(gold_dir)),
            "candidate_source": str(Path(run_output_dir)),
            "candidate": "pipeline",
            "gold_count": len(gold),
            "matched_count": matched,
            "sample_count": len(samples),
            "missing_candidate_chunks": missing_chunks,
            "empty_candidate_chunks": empty_chunks,
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_name
    _save(result, out_path)

    logger.info(
        "car_pipeline (pipeline mode): %d samples (gold=%d, candidate-matched=%d) -> %s",
        len(samples),
        len(gold),
        matched,
        out_path,
    )
    return out_path


def prepare_from_baseline(
    baseline_dir: Path,
    candidate_name: str,
    output_dir: Path = OUTPUT_DIR,
    subset_size: Optional[int] = None,
) -> Path:
    """Convert a flat baseline dataset to benchmark format.

    Gold and candidate are loaded from the same leaf directory.  Each leaf
    directory must contain both ``manual_result_full_recall.json`` and
    ``<candidate_name>.json``.

    Args:
        baseline_dir: Root of the baseline dataset (containing per-doc dirs).
        candidate_name: Candidate JSON filename stem (no ``.json``), e.g.
            ``api_result`` or ``codex_agent_workflow_result_gpt54_full_recall_20260701``.
            Wildcards (``*``, ``?``) are supported.
        output_dir: Where to write the benchmark JSON.
        subset_size: Limit to the first N paired samples.

    Returns:
        Path to the generated benchmark JSON file.
    """
    gold, candidates = load_baseline(Path(baseline_dir), candidate_name)

    if not gold:
        raise FileNotFoundError(f"No gold annotations found under {baseline_dir}")

    samples, matched, missing_chunks, empty_chunks = _build_samples(gold, candidates, subset_size)
    result = {
        "schema": _build_schema(samples),
        "samples": samples,
        "meta": {
            "gold_source": str(Path(baseline_dir)),
            "candidate_source": str(Path(baseline_dir)),
            "candidate": candidate_name,
            "gold_count": len(gold),
            "matched_count": matched,
            "sample_count": len(samples),
            "missing_candidate_chunks": missing_chunks,
            "empty_candidate_chunks": empty_chunks,
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"car_baseline_{candidate_name}.json"
    _save(result, out_path)

    logger.info(
        "car_pipeline (baseline mode): %d samples (gold=%d, candidate=%s matched=%d) -> %s",
        len(samples),
        len(gold),
        candidate_name,
        matched,
        out_path,
    )
    return out_path


# ---------------------------------------------------------------------------
# In-memory adapter (car → benchmark) for operator.evaluate
# ---------------------------------------------------------------------------


def build_extraction_inputs(
    gold: List[Dict[str, Any]],
    candidate: List[Dict[str, Any]],
    chunk_texts: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert car-format gold/candidate into benchmark format for ``evaluate``.

    The car pipeline's in-memory structures — extractor draft (candidate) and
    manual annotation (gold) — are both ``{chunk_id, vertices, edges}`` in car
    format (vertex ``{type, name, properties}``, edge
    ``{type, source_type, source_name, target_type, target_name, properties}``).
    This adapter maps them losslessly to the benchmark's HugeGraph-native
    shape (vertex ``{label, name, properties}``, edge
    ``{label, outV, inV, outVLabel, inVLabel}``) so the operator can consume
    them without knowing the car schema. ``ChunkPayload.text`` is passed via
    ``chunk_texts`` and becomes ``input_text`` for ``extraction_faithfulness``.

    No pairing is performed here — the operator pairs gold/candidate by
    ``sample_id``. Callers with both sides in memory pass them straight through.

    Args:
        gold: list of ``{chunk_id, vertices, edges}`` (car format).
        candidate: list of ``{chunk_id, vertices, edges}`` (car format).
        chunk_texts: optional ``chunk_id →原文 text``; when present for a
            gold chunk it is attached as ``input_text`` on the gold item.

    Returns:
        ``(gold_list, candidate_list)`` in benchmark format, each item keyed
        by ``sample_id`` (= ``chunk_id``). ``gold_list`` items additionally
        carry ``input_text`` when a matching chunk text was supplied.
    """
    chunk_texts = chunk_texts or {}
    gold_list: List[Dict[str, Any]] = []
    for g in gold:
        chunk_id = g.get("chunk_id") or g.get("sample_id") or ""
        vertices, edges = _car_graph_to_benchmark(g.get("vertices", []), g.get("edges", []))
        item: Dict[str, Any] = {"sample_id": chunk_id, "vertices": vertices, "edges": edges}
        text = chunk_texts.get(chunk_id, "")
        if text:
            item["input_text"] = text
        gold_list.append(item)
    candidate_list: List[Dict[str, Any]] = []
    for c in candidate:
        chunk_id = c.get("chunk_id") or c.get("sample_id") or ""
        vertices, edges = _car_graph_to_benchmark(c.get("vertices", []), c.get("edges", []))
        candidate_list.append({"sample_id": chunk_id, "vertices": vertices, "edges": edges})
    return gold_list, candidate_list


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="car-pipeline-adapter",
        description="Convert car-pipeline extraction artifacts to benchmark extraction format.",
    )

    subparsers = parser.add_subparsers(dest="mode", help="Candidate source mode")

    # -- pipeline mode (default)
    pipeline_parser = subparsers.add_parser("pipeline", help="Pipeline run output (chunks-mode or docs-mode)")
    pipeline_parser.add_argument(
        "--gold-dir", type=Path, default=DEFAULT_GOLD_DIR,
        help=f"Gold annotations root (default: {DEFAULT_GOLD_DIR})",
    )
    pipeline_parser.add_argument(
        "--run-output-dir", type=Path, default=DEFAULT_RUN_OUTPUT_DIR,
        help=f"Candidate run output root (default: {DEFAULT_RUN_OUTPUT_DIR})",
    )
    pipeline_parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    pipeline_parser.add_argument(
        "--subset-size", type=int, default=None,
        help="Limit to the first N gold chunks.",
    )

    # -- baseline mode
    baseline_parser = subparsers.add_parser("baseline", help="Flat baseline dataset (gold+candidate same dir)")
    baseline_parser.add_argument(
        "--baseline-dir", type=Path, required=True,
        help="Baseline dataset root (contains per-doc directories).",
    )
    baseline_parser.add_argument(
        "--candidate", type=str, required=True,
        dest="candidate_name",
        help="Candidate JSON filename stem (no .json), e.g. 'api_result'.",
    )
    baseline_parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    baseline_parser.add_argument(
        "--subset-size", type=int, default=None,
        help="Limit to the first N gold chunks.",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)

    if args.mode == "baseline":
        out_path = prepare_from_baseline(
            baseline_dir=args.baseline_dir,
            candidate_name=args.candidate_name,
            output_dir=args.output_dir,
            subset_size=args.subset_size,
        )
    else:
        # pipeline mode (default for backward compatibility)
        gold_dir = getattr(args, "gold_dir", DEFAULT_GOLD_DIR)
        run_dir = getattr(args, "run_output_dir", DEFAULT_RUN_OUTPUT_DIR)
        out_path = prepare_from_pipeline(
            gold_dir=gold_dir,
            run_output_dir=run_dir,
            output_dir=args.output_dir,
            subset_size=args.subset_size,
        )

    print(f"Written: {out_path}")
    return 0


# Backward compatibility alias — callers that imported ``prepare_car_pipeline``
# still work without code changes.
prepare_car_pipeline = prepare_from_pipeline


if __name__ == "__main__":
    raise SystemExit(main())
