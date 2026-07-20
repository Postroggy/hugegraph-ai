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

"""Faithfulness metric using LLM-based statement decomposition and NLI.

Measures whether the answer is faithful to the provided context by:
1. Decomposing the answer into atomic statements.
2. Verifying each statement against the context via NLI.

Reference: RAGAS faithfulness implementation.
"""

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

from hugegraph_llm.benchmark.llm_judge.judge_utils import clean_contexts, retry_llm_call
from hugegraph_llm.benchmark.llm_judge.message import Message
from hugegraph_llm.benchmark.llm_judge.prompts import get_prompt
from hugegraph_llm.benchmark.llm_judge.schemas import NLIVerdictsResult, StatementListResult
from hugegraph_llm.benchmark.metrics.base import BaseMetric
from hugegraph_llm.benchmark.metrics.registry import MetricRegistry

logger = logging.getLogger(__name__)


def _decompose_statements(llm: Any, question: str, answer: str, language: str = "en") -> List[str]:
    """Decompose an answer into atomic statements using LLM."""
    prompt = get_prompt("STATEMENT_DECOMPOSE_PROMPT", language).format(question=question, answer=answer)
    try:
        data = retry_llm_call(
            llm, [Message(prompt)], response_format=StatementListResult
        )
        if data and isinstance(data.get("statements"), list):
            return [str(s) for s in data["statements"] if s]
    except Exception as e:
        logger.warning("Statement decomposition failed: %s", e)

    # Fallback: treat entire answer as single statement
    return [answer] if answer else []


def _verify_statements(llm: Any, context: str, statements: List[str], language: str = "en") -> Optional[int]:
    """Verify statements against context, return count of supported ones.

    Returns ``None`` when the LLM call fails or the response can't be parsed,
    so the caller surfaces a missing score rather than a misleading 0.
    """
    if not statements:
        return 0

    stmt_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(statements))
    prompt = get_prompt("NLI_STATEMENT_PROMPT", language).format(context=context, statements=stmt_text)
    try:
        data = retry_llm_call(
            llm, [Message(prompt)], response_format=NLIVerdictsResult
        )
    except Exception as e:
        logger.warning("NLI verification LLM call failed: %s", e)
        return None
    if not data or not isinstance(data.get("verdicts"), list):
        logger.warning("NLI verification: failed to parse verdicts")
        return None
    verdicts = data["verdicts"]
    if len(verdicts) != len(statements):
        logger.warning(
            "NLI verification: expected %d verdicts, got %d",
            len(statements),
            len(verdicts),
        )
        return None
    if any(
        not isinstance(verdict, dict)
        or verdict.get("verdict") not in {"Yes", "No"}
        or not str(verdict.get("reason", "")).strip()
        for verdict in verdicts
    ) or Counter(verdict.get("statement") for verdict in verdicts) != Counter(statements):
        logger.warning("NLI verification: verdict statements do not match input")
        return None
    supported = sum(1 for verdict in verdicts if verdict.get("verdict") == "Yes")
    return supported


@MetricRegistry.register
class Faithfulness(BaseMetric):
    """Faithfulness metric: measures answer grounding in context.

    Requires ``llm`` and ``context`` in kwargs. Returns None when
    no LLM is available (offline mode).

    Registered name: ``faithfulness``
    """

    name: str = "faithfulness"
    requires_llm: bool = True

    def calculate(
        self,
        prediction: Any,
        reference: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Optional[float]]:
        """Calculate faithfulness score.

        Args:
            prediction: Answer text (str).
            reference: Unused.
            **kwargs: Must contain ``llm`` and ``context`` (List[str]).

        Returns:
            Dict with ``faithfulness`` key (float 0-1 or None).
        """
        llm = kwargs.get("llm")
        if llm is None:
            return {"faithfulness": None}

        answer = str(prediction or "")
        contexts = clean_contexts(kwargs.get("context", []))
        question = kwargs.get("question", "")
        language = kwargs.get("language", "en")

        if not contexts:
            return {"faithfulness": 0.0}

        combined_context = "\n\n".join(contexts)

        if not answer:
            # Vacuous truth: an empty answer has no statements to verify,
            # so it's trivially faithful (GraphRAG-Benchmark convention).
            return {"faithfulness": 1.0}

        statements = _decompose_statements(llm, question, answer, language)
        if not statements:
            # Failed to decompose a non-empty answer → cannot evaluate
            return {"faithfulness": None}

        supported = _verify_statements(llm, combined_context, statements, language)
        if supported is None:
            return {"faithfulness": None}
        score = supported / len(statements)

        return {"faithfulness": round(score, 4)}
