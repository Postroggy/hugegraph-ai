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

"""Tests for semantic entity/triple F1: 1:1 dedup of LLM matches, the
redundancy diagnostic, and lower-is-better direction declaration."""

import json

import pytest

from hugegraph_llm.benchmark.metrics.extraction.semantic_entity_f1 import SemanticEntityF1
from hugegraph_llm.benchmark.metrics.extraction.semantic_triple_f1 import SemanticTripleF1
from hugegraph_llm.benchmark.metrics.registry import MetricRegistry

pytestmark = pytest.mark.unit


class FakeLLM:
    """Returns a single canned JSON response — semantic metrics call once."""

    def __init__(self, response):
        self._response = response

    def generate(self, prompt='', **kwargs):
        return self._response


# --- semantic entity: 1:1 dedup caps P/R at 1.0 ---

def test_semantic_entity_duplicate_pairs_do_not_inflate_scores():
    """The LLM returning the same pair repeatedly must not push precision/recall
    past 1.0 — 1:1 dedup keeps one, redundancy stays 0 (not over-split)."""
    metric = SemanticEntityF1()
    fake = FakeLLM(json.dumps({'matches': [[0, 0], [0, 0], [0, 0]]}))
    r = metric.calculate([{'label': 'car', 'name': 'A'}], [{'label': 'car', 'name': 'A'}], llm=fake)
    assert r['semantic_entity_precision'] == 1.0
    assert r['semantic_entity_recall'] == 1.0
    assert r['semantic_entity_redundancy'] == 0.0


def test_semantic_entity_many_candidates_one_gold():
    """3 candidates all matching the same gold: dedup keeps 1, the other 2 are
    redundant (extractor over-split)."""
    metric = SemanticEntityF1()
    fake = FakeLLM(json.dumps({'matches': [[0, 0], [1, 0], [2, 0]]}))
    pred = [{'label': 'car', 'name': n} for n in ('Alice', 'Alis', 'Alyce')]
    ref = [{'label': 'car', 'name': 'Alice'}]
    r = metric.calculate(pred, ref, llm=fake)
    assert r['semantic_entity_precision'] == round(1 / 3, 4)
    assert r['semantic_entity_recall'] == 1.0
    assert r['semantic_entity_redundancy'] == round(2 / 3, 4)
    assert r['semantic_entity_precision'] <= 1.0
    assert r['semantic_entity_recall'] <= 1.0


def test_semantic_entity_one_candidate_many_gold_does_not_inflate_redundancy():
    """One candidate matched to two golds is an LLM one-to-many artefact, not
    extractor over-split — redundancy must be 0. The old
    ``(raw_matched - matched) / cand`` numerator would wrongly yield 1.0 here."""
    metric = SemanticEntityF1()
    fake = FakeLLM(json.dumps({'matches': [[0, 0], [0, 1]]}))
    pred = [{'label': 'car', 'name': 'Alice'}]
    ref = [{'label': 'car', 'name': 'A'}, {'label': 'car', 'name': 'B'}]
    r = metric.calculate(pred, ref, llm=fake)
    assert r['semantic_entity_redundancy'] == 0.0
    assert r['semantic_entity_precision'] == 1.0
    assert r['semantic_entity_recall'] == 0.5


def test_semantic_entity_offline_returns_none():
    metric = SemanticEntityF1()
    r = metric.calculate([{'label': 'car', 'name': 'A'}], [{'label': 'car', 'name': 'A'}], llm=None)
    assert r['semantic_entity_f1'] is None
    assert r['semantic_entity_redundancy'] is None


# --- semantic triple: 1:1 dedup ---

def test_semantic_triple_many_candidates_one_gold():
    metric = SemanticTripleF1()
    fake = FakeLLM(json.dumps({'matches': [[0, 0], [1, 0]]}))
    pred = [{'outV': 'A', 'label': 'knows', 'inV': 'B'}, {'outV': 'A', 'label': 'knows', 'inV': 'B'}]
    ref = [{'outV': 'A', 'label': 'knows', 'inV': 'B'}]
    r = metric.calculate(pred, ref, llm=fake)
    assert r['semantic_triple_redundancy'] == 0.5
    assert r['semantic_triple_precision'] == 0.5
    assert r['semantic_triple_recall'] == 1.0


def test_semantic_triple_one_candidate_many_gold_no_redundancy():
    metric = SemanticTripleF1()
    fake = FakeLLM(json.dumps({'matches': [[0, 0], [0, 1]]}))
    pred = [{'outV': 'A', 'label': 'knows', 'inV': 'B'}]
    ref = [{'outV': 'A', 'label': 'knows', 'inV': 'B'}, {'outV': 'A', 'label': 'likes', 'inV': 'B'}]
    r = metric.calculate(pred, ref, llm=fake)
    assert r['semantic_triple_redundancy'] == 0.0


def test_semantic_triple_offline_returns_none():
    metric = SemanticTripleF1()
    r = metric.calculate([{'outV': 'A', 'label': 'knows', 'inV': 'B'}], [{'outV': 'A', 'label': 'knows', 'inV': 'B'}], llm=None)
    assert r['semantic_triple_redundancy'] is None


# --- direction declaration (P0 fix) ---

def test_redundancy_direction_is_lower_is_better():
    assert SemanticEntityF1.is_higher_is_better('semantic_entity_redundancy') is False
    assert SemanticTripleF1.is_higher_is_better('semantic_triple_redundancy') is False
    # registry-level lookup — used by compare + reporter direction arrows.
    assert MetricRegistry.is_higher_is_better('semantic_entity_redundancy') is False
    assert MetricRegistry.is_higher_is_better('semantic_triple_redundancy') is False


def test_semantic_f1_falls_through_to_default_higher_is_better():
    # The metric returns None for f1/precision/recall; registry default is True.
    assert SemanticEntityF1.is_higher_is_better('semantic_entity_f1') is None
    assert MetricRegistry.is_higher_is_better('semantic_entity_f1') is True
    assert MetricRegistry.is_higher_is_better('semantic_triple_f1') is True
