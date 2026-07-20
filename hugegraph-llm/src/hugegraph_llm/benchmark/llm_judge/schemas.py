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

from typing import Annotated, List, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt


class JudgeSchema(BaseModel):
    """Strict base model shared by all LLM-Judge outputs."""

    model_config = ConfigDict(extra="forbid")


BinaryFlag = Literal[0, 1]
BinaryVerdict = Literal["Yes", "No"]
RelevancyScore = Literal[0, 1, 2]
NonEmptyText = Annotated[str, Field(min_length=1)]


class MatchPair(JudgeSchema):
    """One candidate-to-gold match with explicit list indices."""

    candidate_index: NonNegativeInt = Field(description="0-based candidate index")
    gold_index: NonNegativeInt = Field(description="0-based gold/reference index")


class ClassifiedStatement(JudgeSchema):
    """A statement assigned to a correctness bucket."""

    statement: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class FaithfulnessVerdict(JudgeSchema):
    """One item's faithfulness verdict from extraction_faithfulness."""

    idx: NonNegativeInt = Field(description="0-based extraction item index")
    verdict: BinaryVerdict = Field(description="Yes when faithful, otherwise No")
    reason: str = Field(min_length=1, description="Brief support or contradiction rationale")


class FaithfulnessResult(JudgeSchema):
    verdicts: List[FaithfulnessVerdict]


class MatchResult(JudgeSchema):
    """Semantic entity/triple matches with explicit candidate and gold indices."""

    matches: List[MatchPair]
    reasoning: str = Field(min_length=1, description="Brief explanation of matching decisions")


class ContextPrecisionResult(JudgeSchema):
    verdict: BinaryVerdict


class ContextRelevancyResult(JudgeSchema):
    score: RelevancyScore


class AttributedItem(JudgeSchema):
    """One fact/statement with an attribution flag and rationale."""

    statement: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    attributed: BinaryFlag


class ClassificationResult(JudgeSchema):
    """evidence_recall_llm / coverage check: per-item attribution list."""

    classifications: List[AttributedItem]


class CorrectnessResult(JudgeSchema):
    """answer_correctness: TP / FP / FN statement lists."""

    tp: List[ClassifiedStatement]
    fp: List[ClassifiedStatement]
    fn: List[ClassifiedStatement]


class StatementListResult(JudgeSchema):
    """faithfulness decompose: atomic statements."""

    statements: List[NonEmptyText]


class FactsResult(JudgeSchema):
    """coverage fact extract: atomic facts from the reference answer."""

    facts: List[NonEmptyText]


class NLIVerdict(JudgeSchema):
    """faithfulness NLI verify: one statement + yes/no entailment."""

    statement: str = Field(min_length=1)
    reason: str = Field(min_length=1, description="Brief context-based rationale")
    verdict: BinaryVerdict


class NLIVerdictsResult(JudgeSchema):
    verdicts: List[NLIVerdict]
