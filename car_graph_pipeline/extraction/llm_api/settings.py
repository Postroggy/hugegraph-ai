"""Settings loader for the LLM API extraction workflow.

Precedence:
CLI arguments in `extract_llm_api.py` > environment variables >
config.local.yaml > config.yaml > dataclass defaults.

Secrets are intentionally not loaded here. LLM credentials come from
`car_graph_pipeline.config`, which reads environment variables or an ignored
local `.env`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, MutableMapping


SETTINGS_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SETTINGS_DIR / "config.yaml"
LOCAL_CONFIG_PATH = SETTINGS_DIR / "config.local.yaml"


@dataclass(frozen=True)
class ExtractionSettings:
    model: str = "DeepSeek-V4-Flash"
    default_workers: int = 5
    request_timeout: int = 900
    max_tokens_extract: int = 40000
    max_tokens_review: int = 40000
    adaptive_token_steps: list[int] = field(default_factory=lambda: [40000, 50000, 70000])
    temperature: float = 0.0
    api_retries: int = 4
    max_review_retries: int = 3
    review_token_threshold: int = 36000
    direct_pass_score: int = 85
    min_accept_score: int = 80
    max_repair_attempts: int = 2
    chunk_target_chars: int = 2000
    chunk_max_chars: int = 2800
    chunk_overlap_chars: int = 200
    min_context_chars: int = 1000
    max_context_chars: int = 2800
    small_chunk_target: int = 1000
    small_chunk_max: int = 1400
    small_chunk_min: int = 180


def _parse_scalar(raw: str) -> object:
    value = raw.strip()
    if value == "":
        return ""
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value[0:1] in {"'", '"'} and value[-1:] == value[0]:
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _read_simple_yaml(path: Path) -> dict[str, object]:
    """Read the limited YAML subset used by config.yaml.

    Supported forms are top-level sections, two-space indented scalar values,
    and two-space indented lists of scalar values.
    """
    if not path.exists():
        return {}

    data: dict[str, object] = {}
    current_section: str | None = None
    current_key: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        if not line.startswith(" "):
            if not line.endswith(":"):
                raise ValueError(f"unsupported top-level YAML line in {path}: {raw_line}")
            current_section = line[:-1].strip()
            data.setdefault(current_section, {})
            current_key = None
            continue

        if current_section is None:
            raise ValueError(f"YAML value appears before section in {path}: {raw_line}")

        stripped = line.strip()
        section = data.setdefault(current_section, {})
        if not isinstance(section, dict):
            raise ValueError(f"YAML section is not a mapping in {path}: {current_section}")

        if stripped.startswith("- "):
            if current_key is None:
                raise ValueError(f"YAML list item appears before key in {path}: {raw_line}")
            values = section.setdefault(current_key, [])
            if not isinstance(values, list):
                raise ValueError(f"YAML key is not a list in {path}: {current_key}")
            values.append(_parse_scalar(stripped[2:]))
            continue

        if ":" not in stripped:
            raise ValueError(f"unsupported YAML line in {path}: {raw_line}")
        key, raw_value = stripped.split(":", 1)
        current_key = key.strip()
        if raw_value.strip() == "":
            section[current_key] = []
        else:
            section[current_key] = _parse_scalar(raw_value)

    return data


def _deep_merge(base: MutableMapping[str, object], override: Mapping[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        old_value = merged.get(key)
        if isinstance(old_value, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(old_value, value)
        else:
            merged[key] = value
    return merged


def _get_section(data: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = data.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"config section must be a mapping: {name}")
    return value


def _as_int(value: object) -> int:
    return int(value)


def _as_float(value: object) -> float:
    return float(value)


def _as_int_list(value: object) -> list[int]:
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [int(item) for item in value]
    raise ValueError(f"expected int list, got {type(value).__name__}")


def _env_int(env: Mapping[str, str], name: str, current: int) -> int:
    raw = env.get(name)
    return current if raw is None or raw.strip() == "" else int(raw)


def _env_float(env: Mapping[str, str], name: str, current: float) -> float:
    raw = env.get(name)
    return current if raw is None or raw.strip() == "" else float(raw)


def _env_int_list(env: Mapping[str, str], name: str, current: list[int]) -> list[int]:
    raw = env.get(name)
    return current if raw is None or raw.strip() == "" else _as_int_list(raw)


def load_settings(
    *,
    env: Mapping[str, str] | None = None,
    config_path: Path = CONFIG_PATH,
    local_config_path: Path = LOCAL_CONFIG_PATH,
) -> ExtractionSettings:
    merged = _deep_merge(_read_simple_yaml(config_path), _read_simple_yaml(local_config_path))
    llm = _get_section(merged, "llm")
    review = _get_section(merged, "review")
    chunk = _get_section(merged, "chunk")

    defaults = ExtractionSettings()
    cfg = ExtractionSettings(
        model=str(llm.get("model", defaults.model)),
        default_workers=_as_int(llm.get("workers", defaults.default_workers)),
        request_timeout=_as_int(llm.get("request_timeout", defaults.request_timeout)),
        max_tokens_extract=_as_int(llm.get("max_tokens_extract", defaults.max_tokens_extract)),
        max_tokens_review=_as_int(llm.get("max_tokens_review", defaults.max_tokens_review)),
        adaptive_token_steps=_as_int_list(llm.get("adaptive_token_steps", defaults.adaptive_token_steps)),
        temperature=_as_float(llm.get("temperature", defaults.temperature)),
        api_retries=_as_int(llm.get("api_retries", defaults.api_retries)),
        max_review_retries=_as_int(review.get("max_review_retries", defaults.max_review_retries)),
        review_token_threshold=_as_int(review.get("review_token_threshold", defaults.review_token_threshold)),
        direct_pass_score=_as_int(review.get("direct_pass_score", defaults.direct_pass_score)),
        min_accept_score=_as_int(review.get("min_accept_score", defaults.min_accept_score)),
        max_repair_attempts=_as_int(review.get("max_repair_attempts", defaults.max_repair_attempts)),
        chunk_target_chars=_as_int(chunk.get("target_chars", defaults.chunk_target_chars)),
        chunk_max_chars=_as_int(chunk.get("max_chars", defaults.chunk_max_chars)),
        chunk_overlap_chars=_as_int(chunk.get("overlap_chars", defaults.chunk_overlap_chars)),
        min_context_chars=_as_int(chunk.get("min_context_chars", defaults.min_context_chars)),
        max_context_chars=_as_int(chunk.get("max_context_chars", chunk.get("max_chars", defaults.max_context_chars))),
        small_chunk_target=_as_int(chunk.get("small_chunk_target", defaults.small_chunk_target)),
        small_chunk_max=_as_int(chunk.get("small_chunk_max", defaults.small_chunk_max)),
        small_chunk_min=_as_int(chunk.get("small_chunk_min", defaults.small_chunk_min)),
    )

    env_values = os.environ if env is None else env
    return ExtractionSettings(
        model=env_values.get("CAR_GRAPH_LLM_MODEL", cfg.model),
        default_workers=_env_int(env_values, "CAR_GRAPH_LLM_WORKERS", cfg.default_workers),
        request_timeout=_env_int(env_values, "CAR_GRAPH_LLM_REQUEST_TIMEOUT", cfg.request_timeout),
        max_tokens_extract=_env_int(env_values, "CAR_GRAPH_LLM_MAX_TOKENS_EXTRACT", cfg.max_tokens_extract),
        max_tokens_review=_env_int(env_values, "CAR_GRAPH_LLM_MAX_TOKENS_REVIEW", cfg.max_tokens_review),
        adaptive_token_steps=_env_int_list(env_values, "CAR_GRAPH_LLM_ADAPTIVE_TOKEN_STEPS", cfg.adaptive_token_steps),
        temperature=_env_float(env_values, "CAR_GRAPH_LLM_TEMPERATURE", cfg.temperature),
        api_retries=_env_int(env_values, "CAR_GRAPH_LLM_API_RETRIES", cfg.api_retries),
        max_review_retries=_env_int(env_values, "CAR_GRAPH_LLM_MAX_REVIEW_RETRIES", cfg.max_review_retries),
        review_token_threshold=_env_int(env_values, "CAR_GRAPH_LLM_REVIEW_TOKEN_THRESHOLD", cfg.review_token_threshold),
        direct_pass_score=_env_int(env_values, "CAR_GRAPH_LLM_DIRECT_PASS_SCORE", cfg.direct_pass_score),
        min_accept_score=_env_int(env_values, "CAR_GRAPH_LLM_MIN_ACCEPT_SCORE", cfg.min_accept_score),
        max_repair_attempts=_env_int(env_values, "CAR_GRAPH_LLM_MAX_REPAIR_ATTEMPTS", cfg.max_repair_attempts),
        chunk_target_chars=_env_int(env_values, "CAR_GRAPH_CHUNK_TARGET_CHARS", cfg.chunk_target_chars),
        chunk_max_chars=_env_int(env_values, "CAR_GRAPH_CHUNK_MAX_CHARS", cfg.chunk_max_chars),
        chunk_overlap_chars=_env_int(env_values, "CAR_GRAPH_CHUNK_OVERLAP_CHARS", cfg.chunk_overlap_chars),
        min_context_chars=_env_int(env_values, "CAR_GRAPH_MIN_CONTEXT_CHARS", cfg.min_context_chars),
        max_context_chars=_env_int(env_values, "CAR_GRAPH_MAX_CONTEXT_CHARS", cfg.max_context_chars),
        small_chunk_target=_env_int(env_values, "CAR_GRAPH_SMALL_CHUNK_TARGET", cfg.small_chunk_target),
        small_chunk_max=_env_int(env_values, "CAR_GRAPH_SMALL_CHUNK_MAX", cfg.small_chunk_max),
        small_chunk_min=_env_int(env_values, "CAR_GRAPH_SMALL_CHUNK_MIN", cfg.small_chunk_min),
    )


SETTINGS = load_settings()

MODEL = SETTINGS.model
DEFAULT_WORKERS = SETTINGS.default_workers
REQUEST_TIMEOUT = SETTINGS.request_timeout
MAX_TOKENS_EXTRACT = SETTINGS.max_tokens_extract
MAX_TOKENS_REVIEW = SETTINGS.max_tokens_review
ADAPTIVE_TOKEN_STEPS = SETTINGS.adaptive_token_steps
TEMPERATURE = SETTINGS.temperature
API_RETRIES = SETTINGS.api_retries
MAX_REVIEW_RETRIES = SETTINGS.max_review_retries
REVIEW_CYCLES = MAX_REVIEW_RETRIES + 1
REVIEW_TOKEN_THRESHOLD = SETTINGS.review_token_threshold
DIRECT_PASS_SCORE = SETTINGS.direct_pass_score
MIN_ACCEPT_SCORE = SETTINGS.min_accept_score
MAX_REPAIR_ATTEMPTS = SETTINGS.max_repair_attempts

CHUNK_TARGET_CHARS = SETTINGS.chunk_target_chars
CHUNK_MAX_CHARS = SETTINGS.chunk_max_chars
CHUNK_OVERLAP_CHARS = SETTINGS.chunk_overlap_chars
MIN_CONTEXT_CHARS = SETTINGS.min_context_chars
MAX_CONTEXT_CHARS = SETTINGS.max_context_chars
SMALL_CHUNK_TARGET = SETTINGS.small_chunk_target
SMALL_CHUNK_MAX = SETTINGS.small_chunk_max
SMALL_CHUNK_MIN = SETTINGS.small_chunk_min

