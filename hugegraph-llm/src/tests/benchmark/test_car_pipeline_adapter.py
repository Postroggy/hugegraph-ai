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

"""Tests for the car-pipeline benchmark adapter.

Focus:
- ``car_vertex_to_benchmark`` / ``car_edge_to_benchmark`` preserve
  ``source_snippet`` — needed by ``ExtractionFaithfulness`` in docs mode
  where the source markdown is not available to the benchmark.
- ``_input_text_for`` returns an empty string when no ``chunk_text.md``
  sibling exists, so the faithfulness metric's snippet-merge fallback
  can trigger.  A synthesized metadata string (``doc_name | heading_path
  | section``) is factually inert and would silently defeat that fallback,
  so it MUST NOT be returned.
"""

import json
from pathlib import Path

import pytest

from hugegraph_llm.benchmark.datasets.car_pipeline import (
    _build_samples,
    _input_text_for,
    car_edge_to_benchmark,
    car_vertex_to_benchmark,
    load_baseline,
    load_candidates_from_pipeline,
)

pytestmark = pytest.mark.unit


# --- field mapping preserves source_snippet (docs-mode contract) ---

def test_car_vertex_to_benchmark_preserves_source_snippet():
    """docs-mode candidates carry ``source_snippet`` per vertex; the adapter
    must not drop it — faithfulness relies on it when chunk_text.md is
    absent (which is always true in production docs mode)."""
    car_vertex = {
        "type": "Component",
        "name": "远光灯",
        "aliases": [],
        "properties": {"comp_name": "远光灯", "component_type": "灯光部件"},
        "source_snippet": "远光灯已打开。",
    }
    record = car_vertex_to_benchmark(car_vertex)
    assert record["label"] == "Component"
    assert record["name"] == "远光灯"
    assert record["source_snippet"] == "远光灯已打开。"


def test_car_vertex_to_benchmark_omits_empty_snippet():
    """Absent / empty ``source_snippet`` must not leave a placeholder key —
    downstream code checks ``item.get('source_snippet')`` for truthiness."""
    record = car_vertex_to_benchmark({"type": "Component", "name": "X", "properties": {}})
    assert "source_snippet" not in record


def test_car_edge_to_benchmark_preserves_source_snippet():
    car_edge = {
        "type": "HAS_STATUS",
        "source_type": "Component",
        "source_name": "远光灯",
        "target_type": "Status",
        "target_name": "远光灯已打开",
        "properties": {},
        "source_snippet": "远光灯已打开。",
    }
    record = car_edge_to_benchmark(car_edge)
    assert record["label"] == "HAS_STATUS"
    assert record["outV"] == "远光灯"
    assert record["inV"] == "远光灯已打开"
    assert record["outVLabel"] == "Component"
    assert record["inVLabel"] == "Status"
    assert record["source_snippet"] == "远光灯已打开。"


# --- _input_text_for: no chunk_text.md → empty string, NOT metadata ---

def test_input_text_for_reads_chunk_text_md_when_present(tmp_path: Path):
    """baseline mode: chunk_text.md is the full chunk source — return it verbatim."""
    (tmp_path / "chunk_text.md").write_text("这是完整的 chunk 原文内容。\n", encoding="utf-8")
    entry = {"dir": tmp_path, "data": {"doc_name": "d", "heading_path": "h"}}
    assert _input_text_for(entry) == "这是完整的 chunk 原文内容。"


def test_input_text_for_returns_empty_when_no_chunk_text_md(tmp_path: Path):
    """docs mode (production): no chunk_text.md sibling — the adapter MUST
    return an empty string so ``extraction_faithfulness`` falls back to
    merging per-item ``source_snippet`` values.  A synthesized
    ``doc_name | heading_path | section`` string is factually inert
    (position, not content) and would silently poison the LLM judge."""
    entry = {
        "dir": tmp_path,
        "data": {
            "doc_name": "五菱缤果用户手册",
            "heading_path": "车门自动落锁 || 自动解锁",
            "source_section": "自动解锁",
        },
    }
    assert _input_text_for(entry) == ""


# --- load_candidates_from_pipeline handles docs-mode chunk_ids and top-level meta ---

def test_load_candidates_docs_mode_naming(tmp_path: Path):
    """docs-mode chunk_ids are ``<doc>--<sha256>_s<idx>`` and files may or may
    not carry a top-level metadata header; the loader must pick both up
    from any ``**/chunk_final/*.final.json`` under the run output dir."""
    run_dir = tmp_path / "workflow_runs" / "run_a"
    doc_final = run_dir / "doc_runs" / "五菱缤果用户手册--4cb5ed37" / "chunk_final"
    doc_final.mkdir(parents=True)

    # (1) File with top-level metadata header (docs-mode after apply_chunk fills gaps)
    (doc_final / "五菱缤果用户手册--4cb5ed37_s0004.final.json").write_text(
        json.dumps({
            "chunk_id": "五菱缤果用户手册--4cb5ed37_s0004",
            "heading_path": "车门自动落锁 || 自动解锁",
            "vertices": [
                {"type": "Component", "name": "远光灯", "aliases": [],
                 "properties": {"comp_name": "远光灯"}, "source_snippet": "远光灯已打开。"}
            ],
            "edges": [],
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    # (2) File without top-level metadata (initial docs-mode extractor output)
    (doc_final / "五菱缤果用户手册--4cb5ed37_s0001.final.json").write_text(
        json.dumps({
            "vertices": [
                {"type": "Specification", "name": "车辆类型",
                 "properties": {"spec_name": "车辆类型", "value_text": "纯电动轿车"},
                 "source_snippet": "车辆类型：纯电动轿车"}
            ],
            "edges": [],
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    candidates = load_candidates_from_pipeline(run_dir)
    assert set(candidates) == {
        "五菱缤果用户手册--4cb5ed37_s0001",
        "五菱缤果用户手册--4cb5ed37_s0004",
    }
    # source_snippet is retained on the raw dict for downstream mapping.
    v = candidates["五菱缤果用户手册--4cb5ed37_s0001"]["vertices"][0]
    assert v["source_snippet"] == "车辆类型：纯电动轿车"


# --- end-to-end: _build_samples in a docs-like scenario ---

def test_build_samples_docs_like_input_text_is_empty(tmp_path: Path):
    """Simulate a docs-mode pairing (no chunk_text.md): the resulting sample's
    ``input_text`` must be empty so the runner triggers the snippet fallback.
    ``source_snippet`` on candidate items must survive the field mapping."""
    gold_dir = tmp_path / "gold_ctx" / "五菱缤果--4cb5ed37_s0001"
    gold_dir.mkdir(parents=True)
    # Deliberately NO chunk_text.md written — this is what docs-mode looks like.
    gold_payload = {
        "chunk_id": "五菱缤果--4cb5ed37_s0001",
        "doc_name": "五菱缤果用户手册",
        "heading_path": "车辆参数",
        "source_section": "车辆参数",
        "vertices": [
            {"type": "Specification", "name": "车辆类型",
             "properties": {"spec_name": "车辆类型", "value_text": "纯电动轿车"}},
        ],
        "edges": [],
    }
    gold = {
        "五菱缤果--4cb5ed37_s0001": {
            "data": gold_payload,
            "dir": gold_dir,
        }
    }
    candidates = {
        "五菱缤果--4cb5ed37_s0001": {
            "vertices": [
                {"type": "Specification", "name": "车辆类型",
                 "properties": {"spec_name": "车辆类型", "value_text": "纯电动轿车"},
                 "source_snippet": "车辆类型：纯电动轿车"},
            ],
            "edges": [],
        }
    }
    samples, matched, missing, empty = _build_samples(gold, candidates, subset_size=None)
    assert matched == 1
    assert missing == []
    assert empty == []
    assert len(samples) == 1
    sample = samples[0]
    # docs-mode contract: NO input_text; faithfulness must use snippet fallback.
    assert sample["input_text"] == ""
    # source_snippet survived the car→benchmark mapping on candidate items.
    assert sample["candidate_vertices"][0]["source_snippet"] == "车辆类型：纯电动轿车"


# --- baseline mode still works (regression guard for chunk_text.md path) ---

def test_load_baseline_reads_chunk_text_md(tmp_path: Path):
    """The baseline layout keeps chunk_text.md next to the gold file; the
    adapter must still return that text as ``input_text``."""
    leaf = tmp_path / "L90" / "L90_ctx001_flat"
    leaf.mkdir(parents=True)
    (leaf / "manual_result_full_recall.json").write_text(
        json.dumps({"chunk_id": "L90_ctx001", "vertices": [], "edges": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    (leaf / "api_result.json").write_text(
        json.dumps({"vertices": [{"type": "Component", "name": "A", "properties": {}}], "edges": []}),
        encoding="utf-8",
    )
    (leaf / "chunk_text.md").write_text("chunk 完整原文", encoding="utf-8")

    gold, candidates = load_baseline(tmp_path, "api_result")
    assert set(gold) == {"L90_ctx001"}
    assert set(candidates) == {"L90_ctx001"}

    samples, matched, _, _ = _build_samples(gold, candidates, subset_size=None)
    assert matched == 1
    assert samples[0]["input_text"] == "chunk 完整原文"
