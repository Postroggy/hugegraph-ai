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

"""Real-LLM integration tests for the benchmark operator.

These tests make real LLM-Judge API calls against the project's configured
endpoint (``.env`` → oneapi-comate → DeepSeek-V4-Pro). They are skipped by
default to keep the suite hermetic; opt in with ``BENCHMARK_REAL_LLM=1``::

    BENCHMARK_REAL_LLM=1 .venv/bin/python -m pytest \\
        hugegraph-llm/src/tests/benchmark/test_operator_integration.py -v

They validate the full LLM call chain end-to-end — operator / CLI →
``create_judge_llm`` → ``JudgeLLM.generate`` → metric → real API → parsed
score — covering the code paths changed when extracting the judge client
and routing the CLI through ``evaluate()``. The offline / mock unit tests
in ``test_operator.py`` do not exercise these paths.
"""

import json
import os
import subprocess
import sys

import pytest

# Classification only (markers are registered in pyproject.toml). Skipping is
# driven by the BENCHMARK_REAL_LLM env var below, not by these markers.
pytestmark = [pytest.mark.external, pytest.mark.slow]

_REAL_LLM = pytest.mark.skipif(
    not os.environ.get("BENCHMARK_REAL_LLM"),
    reason="set BENCHMARK_REAL_LLM=1 to run real-LLM integration tests",
)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_FIXTURE = os.path.join(
    _REPO,
    "hugegraph-llm", "src", "hugegraph_llm", "benchmark", "data", "fixtures",
    "car_docs_mode", "benchmark_inputs", "car_pipeline_extraction.json",
)


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@_REAL_LLM
def test_real_llm_operator_extraction_full():
    """evaluate(gold, candidate) builds its own client and runs every extraction LLM-Judge metric."""
    from hugegraph_llm.benchmark.operator import evaluate

    data = _load(_FIXTURE)
    # fixture is pre-assembled benchmark format; split into gold + candidate
    # (a real caller uses build_extraction_inputs from car-format data).
    gold = [
        {
            "sample_id": s["sample_id"],
            "vertices": s.get("gold_vertices", []),
            "edges": s.get("gold_edges", []),
            "input_text": s.get("input_text", ""),
        }
        for s in data["samples"]
    ]
    candidate = [
        {
            "sample_id": s["sample_id"],
            "vertices": s.get("candidate_vertices", []),
            "edges": s.get("candidate_edges", []),
        }
        for s in data["samples"]
    ]
    result = evaluate(gold, candidate, language="zh")

    assert result.metadata["mode"] == "extraction"
    assert result.metadata.get("error_count", 0) == 0
    # LLM-Judge metrics must return real scores (not None) — this proves the
    # create_judge_llm → JudgeLLM.generate → retry_llm_call → real API chain
    # works after the client was extracted out of cli.py.
    for metric in ("semantic_entity_f1", "semantic_triple_f1", "extraction_faithfulness"):
        assert result.overall.get(metric) is not None, f"{metric} was None"
    # the operator attaches the judge model to metadata
    assert result.metadata.get("model"), "judge model missing from metadata"


@_REAL_LLM
def test_real_llm_cli_extraction(tmp_path):
    """CLI run → runner with the CLI-built client; stdout stays JSON-clean."""
    out = tmp_path / "cli_real.json"
    result = subprocess.run(
        [
            sys.executable, "-m", "hugegraph_llm.benchmark", "run",
            "--mode", "extraction",
            "--data", _FIXTURE,
            "--language", "zh",
            "--metrics", "entity_f1,semantic_entity_f1,extraction_faithfulness",
            "--format", "json",
            "--output", str(out),
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"stderr tail: {result.stderr[-500:]}"
    # logs must go to stderr, not stdout — the _handle_run logging setup keeps
    # stdout clean even though evaluate() imports llm_settings (llm logger).
    assert "INFO" not in result.stdout
    assert "llm:" not in result.stdout

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["meta"]["mode"] == "extraction"
    assert data["meta"].get("error_count", 0) == 0
    assert data["overall"].get("semantic_entity_f1") is not None
    # the CLI-injected client's model lands in metadata
    assert data["meta"].get("model"), "judge model missing from CLI metadata"
