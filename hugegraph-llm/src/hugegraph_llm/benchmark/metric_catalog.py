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

"""Single source of truth for which metrics each benchmark mode runs.

Centralizes the per-mode metric sets and shared validation helpers that were
previously duplicated across ``cli.py`` (``_DEFAULT_METRICS``,
``_MODE_ALLOWED_METRICS``, ``_unknown_metrics``, ``_llm_metrics``,
``_select_metrics``) and ``operator.py`` (``_EXTRACTION_METRICS``).

Deliberate split kept intact:

  * ``default_metrics(mode)`` — offline-friendly subset. The CLI uses this when
    ``--metrics`` is omitted, so the tool runs without an LLM client.
  * ``allowed_metrics(mode)`` — full set including opt-in LLM-Judge metrics.
    The operator's default (it expects a configured LLM) and the upper bound
    for CLI ``--metrics`` selections.

Mode-membership validation (a metric must belong to the requested mode) lives
here; dimension-consistency validation (operator: all metrics share one
dimension) stays in ``operator._infer_dimension`` — a different rule, not
collapsed. This module is neutral: it returns lists and never prints or calls
``SystemExit``; callers own error presentation.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

# Importing the metrics package runs metrics/__init__.py, which imports every
# metric subpackage so each metric self-registers via MetricRegistry. Done here
# (not in cli.py / operator.py) so any catalog caller gets metrics registered
# automatically — unknown_metrics / llm_metrics query MetricRegistry.
import hugegraph_llm.benchmark.metrics  # noqa: F401
from hugegraph_llm.benchmark.metrics.registry import MetricRegistry


@dataclass(frozen=True)
class ModeMetrics:
    """Per-mode metric set: offline-friendly defaults + opt-in LLM-Judge metrics."""

    defaults: Tuple[str, ...]
    opt_in: Tuple[str, ...]

    @property
    def all(self) -> Tuple[str, ...]:
        return self.defaults + self.opt_in


_MODES: Dict[str, ModeMetrics] = {
    "extraction": ModeMetrics(
        defaults=("entity_f1", "triple_f1", "schema_validity"),
        opt_in=(
            "property_f1",
            "semantic_entity_f1",
            "semantic_triple_f1",
            "extraction_faithfulness",
        ),
    ),
    "retrieval": ModeMetrics(
        defaults=("recall_at_k", "hit_at_k", "mrr"),
        opt_in=("context_precision", "context_relevancy", "evidence_recall_llm"),
    ),
    "answer": ModeMetrics(
        defaults=("token_f1", "exact_match", "rouge_l"),
        opt_in=("answer_correctness", "faithfulness", "coverage"),
    ),
}


def default_metrics(mode: str) -> List[str]:
    """Offline-friendly default metrics for *mode*.

    Used by the CLI when ``--metrics`` is omitted (keeps the tool runnable
    without an LLM client).
    """
    return list(_MODES[mode].defaults)


def allowed_metrics(mode: str) -> List[str]:
    """Full allow-list for *mode* (defaults + opt-in LLM-Judge metrics).

    The operator's default metric set, and the upper bound for CLI
    ``--metrics`` selections.
    """
    return list(_MODES[mode].all)


def unknown_metrics(metrics: List[str]) -> List[str]:
    """Metrics in *metrics* not registered in :class:`MetricRegistry`."""
    available = set(MetricRegistry.list_metrics())
    return [m for m in metrics if m not in available]


def llm_metrics(metrics: List[str]) -> List[str]:
    """Metrics in *metrics* that require an LLM-Judge client."""
    out: List[str] = []
    for m in metrics:
        cls = MetricRegistry.get(m)
        if cls is not None and cls.requires_llm:
            out.append(m)
    return out


def metrics_not_in_mode(metrics: List[str], mode: str) -> List[str]:
    """Metrics in *metrics* not in *mode*'s allow-list (mode-mismatch).

    The mode-level membership check: a metric is invalid for a mode if it
    isn't in that mode's allow-list, regardless of whether it is registered
    for another mode. Callers present the error (CLI: ``not valid for <mode>
    mode``).
    """
    allowed = set(_MODES[mode].all)
    return [m for m in metrics if m not in allowed]
