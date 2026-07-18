# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runner for single-answer evaluation (e.g. graph_vector answer)."""

import logging
from typing import Any, Dict, List

from hugegraph_llm.benchmark.models.result import BenchmarkResult, SampleResult
from hugegraph_llm.benchmark.runners.base_runner import BaseRunner

logger = logging.getLogger(__name__)


class AnswerRunner(BaseRunner):
    """Run answer-quality evaluation for a single answer field.

    Expected data format::

        {
            "samples": [
                {
                    "sample_id": "ans_001",
                    "question": "...",
                    "gold_answer": "...",
                    "answer": "...",
                    "retrieved_contexts": ["..."]  # optional
                }
            ]
        }

    The candidate answer is read from the ``answer`` field. Callers whose
    pipeline produces a differently-named field (e.g. ``graph_vector_answer``)
    should normalize it to ``answer`` before evaluation rather than relying
    on a per-runner key override.
    """

    def __init__(self, max_workers: int = 20) -> None:
        super().__init__(max_workers=max_workers)

    def run(
        self,
        data: Any,
        metrics: List[str],
        language: str = "en",
        llm: Any = None,
    ) -> BenchmarkResult:
        """Execute single-answer benchmark.

        Args:
            data: Benchmark data — an in-memory dict (operator path) or
                a path to a JSON file (CLI path).
            metrics: List of metric names to evaluate.
            language: Language code ('en' or 'zh').
            llm: Optional LLM instance for LLM-based metrics.

        Returns:
            Aggregated BenchmarkResult.
        """
        self._errors.clear()
        source = data
        data = self._resolve_data(data)

        samples = data.get("samples", [])

        metric_instances = self._create_metric_instances(metrics)

        result = self._create_result(
            mode="answer",
            language=language,
            metrics=metrics,
            data_path=source if isinstance(source, str) else None,
        )

        def process_sample(sample: Dict[str, Any]) -> SampleResult:
            sample_id = sample.get("sample_id", "unknown")
            sample_result = SampleResult(
                sample_id=sample_id,
                question_type=sample.get("question_type"),
            )
            prediction = sample.get("answer", "")
            reference = sample.get("gold_answer", "")
            context = sample.get("retrieved_contexts", [])

            for name, metric in metric_instances.items():
                scores = self._run_metric_safe(
                    metric=metric,
                    prediction=prediction,
                    reference=reference,
                    sample_id=sample_id,
                    language=language,
                    question=sample.get("question", ""),
                    context=context,
                    llm=llm,
                )
                sample_result.metrics.update(scores)
            return sample_result

        for sample_result in self._run_samples_concurrent(samples, process_sample):
            result.samples.append(sample_result)

        self._finalize_result(result)
        return result
