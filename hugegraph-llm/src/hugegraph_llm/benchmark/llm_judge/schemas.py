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

"""Output schemas for LLM-Judge metrics (OpenAI Structured Outputs).

Each metric passes one of these pydantic models as ``response_format`` so the
model's reply is guaranteed to adhere to the schema (``strict: true``). The
SDK's ``beta.chat.completions.parse`` deserializes the reply into the model;
``JudgeLLM.generate`` returns ``.model_dump()``.

Strict-mode constraints (all fields required, ``additionalProperties: false``)
are satisfied automatically — no Optional fields, no extra properties.
"""

from typing import List

from pydantic import BaseModel


class FaithfulnessVerdict(BaseModel):
    """One item's faithfulness verdict from extraction_faithfulness."""

    idx: int
    verdict: int  # 1 = faithful, 0 = unfaithful


class FaithfulnessResult(BaseModel):
    verdicts: List[FaithfulnessVerdict]


class MatchResult(BaseModel):
    """semantic_entity_f1 / semantic_triple_f1: pairs of [candidate_idx, gold_idx]."""

    matches: List[List[int]]


class ContextPrecisionResult(BaseModel):
    verdict: str  # "yes" or "no"


class ContextRelevancyResult(BaseModel):
    score: int  # 0, 1, or 2


class AttributedItem(BaseModel):
    """One fact/statement with an attribution flag."""

    statement: str
    attributed: int  # 0 or 1


class ClassificationResult(BaseModel):
    """evidence_recall_llm / coverage check: per-item attribution list."""

    classifications: List[AttributedItem]


class CorrectnessResult(BaseModel):
    """answer_correctness: TP / FP / FN statement lists."""

    tp: List[str]
    fp: List[str]
    fn: List[str]


class StatementListResult(BaseModel):
    """faithfulness decompose: atomic statements."""

    statements: List[str]


class FactsResult(BaseModel):
    """coverage fact extract: atomic facts from the reference answer."""

    facts: List[str]


class NLIVerdict(BaseModel):
    """faithfulness NLI verify: one statement + yes/no entailment."""

    statement: str
    verdict: str  # "yes" or "no"


class NLIVerdictsResult(BaseModel):
    verdicts: List[NLIVerdict]
