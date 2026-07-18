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

"""Factory for the OpenAI-compatible LLM client used by LLM-Judge metrics.

Creates a client from ``llm_settings`` (project ``.env``). Generation
parameters (temperature/seed) are fixed so judge results are reproducible.

This module deliberately does NOT configure global logging — that is the
caller's concern. The CLI forces logs to stderr to keep stdout JSON-clean;
the operator leaves logging to the host process and never reconfigures it.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)


class JudgeLLM:
    """Thin wrapper exposing ``generate(prompt=...)`` over chat completions.

    Uses standard OpenAI messages format and non-streaming chat completions.
    Any object with a compatible ``generate(prompt=...) -> str`` method can
    substitute for this class — the metrics only depend on that contract
    (see ``hugegraph_llm.benchmark.llm_judge.judge_utils.retry_llm_call``).
    """

    def __init__(self, client: Any, model: str, temperature: float, seed: int, max_tokens: int) -> None:
        self._c = client
        self._m = model
        self._temperature = temperature
        self._seed = seed
        self._max_tokens = max_tokens

    def generate(self, prompt: str = "", messages: Optional[List[Dict[str, str]]] = None, **kw: Any) -> str:
        msgs = messages or [{"role": "user", "content": prompt}]
        response = self._c.chat.completions.create(
            model=self._m,
            messages=msgs,
            temperature=self._temperature,
            max_tokens=kw.get("max_tokens", self._max_tokens),
            seed=self._seed,
            # Disable reasoning/thinking for judge calls. Some chat models on
            # the gateway (e.g. DeepSeek-V4-Pro) emit a reasoning_content by
            # default, which wastes tokens and can leak into the parsed
            # output. Provider-specific field; verified to disable thinking
            # on DeepSeek-V4-Pro.
            extra_body={"thinking": {"type": "disabled"}},
        )
        return response.choices[0].message.content


def create_judge_llm(settings: Optional[Any] = None) -> Tuple[Optional[JudgeLLM], Dict[str, Any]]:
    """Create the LLM-Judge client + reproducibility metadata.

    Reads endpoint / model / credentials from ``llm_settings`` (project
    ``.env``) unless ``settings`` is provided (used by tests to inject a
    fake config). Generation parameters are fixed: ``temperature=0.0``,
    ``seed=42`` — deterministic judge output across runs.

    Args:
        settings: Optional LLM settings object exposing
            ``openai_chat_api_key`` / ``openai_chat_api_base`` /
            ``openai_chat_language_model`` / ``openai_chat_tokens``.
            When omitted, ``llm_settings`` is imported from
            ``hugegraph_llm.config``.

    Returns:
        ``(llm, metadata)`` where ``metadata`` has ``model`` /
        ``temperature`` / ``seed``. On failure, ``(None, {})`` plus a
        warning log — callers can then fall back to offline mode, in which
        LLM-Judge metrics return ``None`` scores.
    """
    try:
        from hugegraph_llm.config import llm_settings

        cfg = settings if settings is not None else llm_settings
        model = getattr(cfg, "openai_chat_language_model", None) or "gpt-4.1-mini"
        client = OpenAI(
            api_key=getattr(cfg, "openai_chat_api_key", None) or "",
            base_url=getattr(cfg, "openai_chat_api_base", None),
        )
        temperature = 0.0
        seed = 42
        max_tokens = getattr(cfg, "openai_chat_tokens", None) or 2048

        llm = JudgeLLM(client, model, temperature, seed, max_tokens)
        logger.info(
            "LLM client: OpenAI-compatible, model=%s, temperature=%s, seed=%s",
            model,
            temperature,
            seed,
        )
        meta = {"model": model, "temperature": temperature, "seed": seed}
        return llm, meta
    except Exception as e:
        logger.warning("LLM client creation failed: %s. LLM-Judge metrics will be skipped.", e)
        return None, {}
