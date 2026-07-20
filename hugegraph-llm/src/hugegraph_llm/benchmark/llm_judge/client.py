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
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from pydantic import BaseModel

from hugegraph_llm.benchmark.llm_judge.exceptions import (
    LLMPermanentError,
    LLMTransientError,
)
from hugegraph_llm.benchmark.llm_judge.message import Message

logger = logging.getLogger(__name__)

# openai SDK exception → abstract LLM error mapping. Done here — the only place
# that imports openai — so retry_llm_call / metrics stay provider-agnostic: a
# provider swap only touches this file.
_PERMANENT_OPENAI_ERRORS = (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    ConflictError,
    UnprocessableEntityError,
)
_TRANSIENT_OPENAI_ERRORS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)

# Default extra_body for JudgeLLM: disable reasoning/thinking on judge calls.
# Some gateway models (e.g. DeepSeek-V4-Pro) emit reasoning_content by default,
# which wastes tokens and can leak into parsed output. Override per-instance
# (JudgeLLM(extra_body=...)) or per-call (generate(extra_body=...)).
_DEFAULT_EXTRA_BODY = {"thinking": {"type": "disabled"}}


class JudgeLLM:
    """Thin wrapper exposing ``generate(prompt=...)`` over chat completions.

    Uses standard OpenAI messages format and non-streaming chat completions.
    Any object with a compatible ``generate(prompt=...) -> str`` method can
    substitute for this class — the metrics only depend on that contract
    (see ``hugegraph_llm.benchmark.llm_judge.judge_utils.retry_llm_call``).

    Provider exceptions raised by the underlying client are translated to
    abstract ``LLMTransientError`` / ``LLMPermanentError`` here, so retry logic
    and metrics stay provider-agnostic.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        temperature: float,
        seed: int,
        max_tokens: int,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._c = client
        self._m = model
        self._temperature = temperature
        self._seed = seed
        self._max_tokens = max_tokens
        # Provider-specific params (e.g. disabling thinking/reasoning). Defaults
        # to disabling thinking; override per-instance or per-call.
        self._extra_body = extra_body if extra_body is not None else dict(_DEFAULT_EXTRA_BODY)

    def generate(
        self,
        messages: List[Message],
        response_format: type[BaseModel],
        **kw: Any,
    ) -> Optional[dict]:
        """Call the LLM with Structured Outputs (``response_format`` json_schema).

        Args:
            messages: Chat messages (system / user / assistant turns).
            response_format: pydantic model defining the expected JSON schema.
                Passed to ``beta.chat.completions.parse`` so the reply is
                guaranteed to adhere to the schema (strict mode).
            **kw: Per-call overrides (e.g. ``extra_body``, ``max_tokens``).

        Returns:
            The parsed reply as a dict (``model_dump()``), or ``None`` if the
            model returned no parseable structured content.

        Raises:
            LLMPermanentError: Auth / 4xx / refusal — retrying won't help.
            LLMTransientError: Rate limit / timeout / 5xx / truncated (length).
        """
        msgs = [m.to_dict() for m in messages]
        extra_body = kw.get("extra_body", self._extra_body)
        create_kwargs: Dict[str, Any] = {
            "model": self._m,
            "messages": msgs,
            "temperature": self._temperature,
            "max_tokens": kw.get("max_tokens", self._max_tokens),
            "seed": self._seed,
            "response_format": response_format,
        }
        if extra_body:
            create_kwargs["extra_body"] = extra_body
        try:
            response = self._c.beta.chat.completions.parse(**create_kwargs)
        except _PERMANENT_OPENAI_ERRORS as e:
            raise LLMPermanentError(f"LLM call failed (permanent {type(e).__name__}): {e}") from e
        except _TRANSIENT_OPENAI_ERRORS as e:
            raise LLMTransientError(f"LLM call failed (transient {type(e).__name__}): {e}") from e
        except APIStatusError as e:
            # APIStatusError covers 4xx/5xx not matched above; classify by code.
            if e.status_code >= 500:
                raise LLMTransientError(f"LLM call failed (HTTP {e.status_code}): {e}") from e
            raise LLMPermanentError(f"LLM call failed (HTTP {e.status_code}): {e}") from e
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMTransientError(
                f"LLM response truncated (finish_reason=length, "
                f"max_tokens={create_kwargs['max_tokens']})"
            )
        msg = choice.message
        refusal = getattr(msg, "refusal", None)
        if refusal:
            raise LLMPermanentError(f"LLM refused to answer: {refusal}")
        if msg.parsed is None:
            logger.warning("LLM returned no parsed structured output")
            return None
        return msg.parsed.model_dump()


def create_judge_llm(
    settings: Optional[Any] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[JudgeLLM], Dict[str, Any]]:
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
        if settings is not None:
            cfg = settings
        else:
            # 自己读 hugegraph-llm/.env，避免 import hugegraph_llm.config
            # （它链式依赖 pyhugegraph/hugegraph-python-client，与 judge client 无关）。
            import os
            from pathlib import Path
            from types import SimpleNamespace

            from dotenv import load_dotenv

            _env_path = Path(__file__).resolve().parents[4] / ".env"
            if _env_path.is_file():
                load_dotenv(_env_path)
            cfg = SimpleNamespace(
                openai_chat_api_key=os.environ.get("OPENAI_CHAT_API_KEY"),
                openai_chat_api_base=os.environ.get("OPENAI_CHAT_API_BASE"),
                openai_chat_language_model=os.environ.get("OPENAI_CHAT_LANGUAGE_MODEL"),
                openai_chat_tokens=os.environ.get("OPENAI_CHAT_TOKENS"),
            )
        model = getattr(cfg, "openai_chat_language_model", None) or "gpt-4.1-mini"
        client = OpenAI(
            api_key=getattr(cfg, "openai_chat_api_key", None) or "",
            base_url=getattr(cfg, "openai_chat_api_base", None),
        )
        temperature = 0.0
        seed = 42
        max_tokens = int(getattr(cfg, "openai_chat_tokens", None) or 2048)

        llm = JudgeLLM(client, model, temperature, seed, max_tokens, extra_body=extra_body)
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
