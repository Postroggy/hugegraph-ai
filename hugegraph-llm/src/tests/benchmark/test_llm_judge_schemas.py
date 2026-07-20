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

"""Contract tests for LLM-Judge Structured Outputs schemas."""

import pytest
from pydantic import ValidationError

from hugegraph_llm.benchmark.llm_judge.schemas import (
    ClassificationResult,
    ContextPrecisionResult,
    ContextRelevancyResult,
    CorrectnessResult,
    FaithfulnessResult,
    MatchResult,
    NLIVerdictsResult,
)

pytestmark = pytest.mark.unit


def test_faithfulness_schema_requires_reason_and_binary_verdict():
    result = FaithfulnessResult.model_validate(
        {"verdicts": [{"idx": 0, "verdict": "Yes", "reason": "supported"}]}
    )

    assert result.verdicts[0].reason == "supported"
    with pytest.raises(ValidationError):
        FaithfulnessResult.model_validate({"verdicts": [{"idx": 0, "verdict": "Maybe", "reason": "bad"}]})


def test_judgment_schemas_use_explicit_enums():
    assert ContextPrecisionResult.model_validate({"verdict": "Yes"}).verdict == "Yes"
    assert ContextRelevancyResult.model_validate({"score": 2}).score == 2
    assert ClassificationResult.model_validate(
        {"classifications": [{"statement": "fact", "reason": "supported", "attributed": 1}]}
    ).classifications[0].attributed == 1

    with pytest.raises(ValidationError):
        ContextPrecisionResult.model_validate({"verdict": "maybe"})
    with pytest.raises(ValidationError):
        ContextRelevancyResult.model_validate({"score": 3})


def test_match_schema_is_self_describing_and_requires_reasoning():
    result = MatchResult.model_validate(
        {
            "matches": [{"candidate_index": 1, "gold_index": 0}],
            "reasoning": "same entity",
        }
    )

    assert result.matches[0].candidate_index == 1
    assert result.matches[0].gold_index == 0
    with pytest.raises(ValidationError):
        MatchResult.model_validate({"matches": [[1, 0]], "reasoning": "legacy pair"})


def test_correctness_and_nli_items_preserve_reasons():
    correctness = CorrectnessResult.model_validate(
        {
            "tp": [{"statement": "A", "reason": "matched"}],
            "fp": [],
            "fn": [],
        }
    )
    nli = NLIVerdictsResult.model_validate(
        {"verdicts": [{"statement": "A", "reason": "context says A", "verdict": "Yes"}]}
    )

    assert correctness.tp[0].reason == "matched"
    assert nli.verdicts[0].verdict == "Yes"


def test_all_nested_models_forbid_extra_fields():
    with pytest.raises(ValidationError):
        FaithfulnessResult.model_validate(
            {"verdicts": [{"idx": 0, "verdict": "Yes", "reason": "ok", "extra": True}]}
        )
