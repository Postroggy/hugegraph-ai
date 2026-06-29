"""Settings and runtime context for offline entity disambiguation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping

from car_graph_pipeline.config import (
    EMBEDDING_MAX_CHARS as BASE_EMBEDDING_MAX_CHARS,
    EMBEDDING_MODEL as BASE_EMBEDDING_MODEL,
    EMBEDDING_TIMEOUT as BASE_EMBEDDING_TIMEOUT,
    EMBEDDING_URL as BASE_EMBEDDING_URL,
    LLM_BASE_URL as BASE_LLM_BASE_URL,
    LLM_DISAMBIGUATE_MODEL,
    LLM_MAX_TOKENS as BASE_LLM_MAX_TOKENS,
    LLM_TIMEOUT as BASE_LLM_TIMEOUT,
)


SETTINGS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SETTINGS_DIR.parent
DEFAULT_VERSION_ROOT = PACKAGE_ROOT / "output" / "version"
CONFIG_PATH = SETTINGS_DIR / "config.yaml"
LOCAL_CONFIG_PATH = SETTINGS_DIR / "config.local.yaml"


@dataclass(frozen=True)
class DisambiguationSettings:
    input_version: str = "v2_fault_split"
    output_version: str = "v3_disambiguated"
    cosine_similarity_threshold: float = 0.85
    edit_distance_ratio_threshold: float = 0.3
    llm_batch_size: int = 8
    skip_types: frozenset[str] = frozenset({"VehicleBrand", "VehicleModel"})
    max_entity_reduction_pct: float = 20.0
    max_relation_reduction_pct: float = 15.0
    embedding_url: str = BASE_EMBEDDING_URL
    embedding_model: str = BASE_EMBEDDING_MODEL
    embedding_dim: int = 1024
    embedding_max_chars: int = BASE_EMBEDDING_MAX_CHARS
    embedding_concurrency: int = 5
    embedding_batch_size: int = 16
    embedding_timeout: int = BASE_EMBEDDING_TIMEOUT
    llm_base_url: str = BASE_LLM_BASE_URL
    llm_model: str = LLM_DISAMBIGUATE_MODEL
    llm_concurrency: int = 5
    llm_timeout: int = BASE_LLM_TIMEOUT
    llm_max_tokens: int = BASE_LLM_MAX_TOKENS
    llm_temperature: float = 0.1
    llm_retries: int = 4
    disable_thinking: bool = True


@dataclass(frozen=True)
class DisambiguationContext:
    settings: DisambiguationSettings
    input_dir: Path
    output_dir: Path

    @property
    def input_entities_path(self) -> Path:
        return self.input_dir / "extracted_entities.json"

    @property
    def input_relations_path(self) -> Path:
        return self.input_dir / "extracted_relations.json"

    @property
    def candidates_path(self) -> Path:
        return self.output_dir / "candidates.json"

    @property
    def decisions_path(self) -> Path:
        return self.output_dir / "merge_decisions.json"

    @property
    def partial_decisions_path(self) -> Path:
        return self.output_dir / "merge_decisions_partial.json"

    @property
    def merged_entities_path(self) -> Path:
        return self.output_dir / "merged_entities.json"

    @property
    def merged_relations_path(self) -> Path:
        return self.output_dir / "merged_relations.json"


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


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_str_set(value: object) -> frozenset[str]:
    if isinstance(value, str):
        return frozenset(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, list):
        return frozenset(str(item).strip() for item in value if str(item).strip())
    raise ValueError(f"expected string list, got {type(value).__name__}")


def _env_int(env: Mapping[str, str], name: str, current: int) -> int:
    raw = env.get(name)
    return current if raw is None or raw.strip() == "" else int(raw)


def _env_float(env: Mapping[str, str], name: str, current: float) -> float:
    raw = env.get(name)
    return current if raw is None or raw.strip() == "" else float(raw)


def _env_bool(env: Mapping[str, str], name: str, current: bool) -> bool:
    raw = env.get(name)
    return current if raw is None or raw.strip() == "" else _as_bool(raw)


def load_settings(
    *,
    env: Mapping[str, str] | None = None,
    config_path: Path = CONFIG_PATH,
    local_config_path: Path = LOCAL_CONFIG_PATH,
) -> DisambiguationSettings:
    merged = _deep_merge(_read_simple_yaml(config_path), _read_simple_yaml(local_config_path))
    paths = _get_section(merged, "paths")
    thresholds = _get_section(merged, "thresholds")
    embedding = _get_section(merged, "embedding")
    llm = _get_section(merged, "llm")
    validation = _get_section(merged, "validation")

    defaults = DisambiguationSettings()
    cfg = DisambiguationSettings(
        input_version=str(paths.get("input_version", defaults.input_version)),
        output_version=str(paths.get("output_version", defaults.output_version)),
        cosine_similarity_threshold=float(thresholds.get("cosine_similarity", defaults.cosine_similarity_threshold)),
        edit_distance_ratio_threshold=float(thresholds.get("edit_distance_ratio", defaults.edit_distance_ratio_threshold)),
        llm_batch_size=int(thresholds.get("llm_batch_size", defaults.llm_batch_size)),
        skip_types=_as_str_set(thresholds.get("skip_types", list(defaults.skip_types))),
        max_entity_reduction_pct=float(validation.get("max_entity_reduction_pct", defaults.max_entity_reduction_pct)),
        max_relation_reduction_pct=float(validation.get("max_relation_reduction_pct", defaults.max_relation_reduction_pct)),
        embedding_url=str(embedding.get("url", defaults.embedding_url)),
        embedding_model=str(embedding.get("model", defaults.embedding_model)),
        embedding_dim=int(embedding.get("dim", defaults.embedding_dim)),
        embedding_max_chars=int(embedding.get("max_chars", defaults.embedding_max_chars)),
        embedding_concurrency=int(embedding.get("concurrency", defaults.embedding_concurrency)),
        embedding_batch_size=int(embedding.get("batch_size", defaults.embedding_batch_size)),
        embedding_timeout=int(embedding.get("timeout", defaults.embedding_timeout)),
        llm_base_url=str(llm.get("base_url", defaults.llm_base_url)),
        llm_model=str(llm.get("model", defaults.llm_model)),
        llm_concurrency=int(llm.get("concurrency", defaults.llm_concurrency)),
        llm_timeout=int(llm.get("timeout", defaults.llm_timeout)),
        llm_max_tokens=int(llm.get("max_tokens", defaults.llm_max_tokens)),
        llm_temperature=float(llm.get("temperature", defaults.llm_temperature)),
        llm_retries=int(llm.get("retries", defaults.llm_retries)),
        disable_thinking=_as_bool(llm.get("disable_thinking", defaults.disable_thinking)),
    )

    env_values = os.environ if env is None else env
    return DisambiguationSettings(
        input_version=env_values.get("CAR_GRAPH_DISAMBIGUATION_INPUT_VERSION", cfg.input_version),
        output_version=env_values.get("CAR_GRAPH_DISAMBIGUATION_OUTPUT_VERSION", cfg.output_version),
        cosine_similarity_threshold=_env_float(env_values, "CAR_GRAPH_DISAMBIGUATION_COSINE_THRESHOLD", cfg.cosine_similarity_threshold),
        edit_distance_ratio_threshold=_env_float(env_values, "CAR_GRAPH_DISAMBIGUATION_EDIT_THRESHOLD", cfg.edit_distance_ratio_threshold),
        llm_batch_size=_env_int(env_values, "CAR_GRAPH_DISAMBIGUATION_LLM_BATCH_SIZE", cfg.llm_batch_size),
        skip_types=_as_str_set(env_values.get("CAR_GRAPH_DISAMBIGUATION_SKIP_TYPES", ",".join(sorted(cfg.skip_types)))),
        max_entity_reduction_pct=_env_float(env_values, "CAR_GRAPH_DISAMBIGUATION_MAX_ENTITY_REDUCTION_PCT", cfg.max_entity_reduction_pct),
        max_relation_reduction_pct=_env_float(env_values, "CAR_GRAPH_DISAMBIGUATION_MAX_RELATION_REDUCTION_PCT", cfg.max_relation_reduction_pct),
        embedding_url=env_values.get("CAR_GRAPH_EMBEDDING_URL", cfg.embedding_url),
        embedding_model=env_values.get("CAR_GRAPH_EMBEDDING_MODEL", cfg.embedding_model),
        embedding_dim=_env_int(env_values, "CAR_GRAPH_EMBEDDING_DIM", cfg.embedding_dim),
        embedding_max_chars=_env_int(env_values, "CAR_GRAPH_EMBEDDING_MAX_CHARS", cfg.embedding_max_chars),
        embedding_concurrency=_env_int(env_values, "CAR_GRAPH_EMBEDDING_CONCURRENCY", cfg.embedding_concurrency),
        embedding_batch_size=_env_int(env_values, "CAR_GRAPH_EMBEDDING_BATCH_SIZE", cfg.embedding_batch_size),
        embedding_timeout=_env_int(env_values, "CAR_GRAPH_EMBEDDING_TIMEOUT", cfg.embedding_timeout),
        llm_base_url=env_values.get("CAR_GRAPH_DISAMBIGUATION_LLM_BASE_URL", cfg.llm_base_url),
        llm_model=env_values.get("CAR_GRAPH_DISAMBIGUATION_LLM_MODEL", cfg.llm_model),
        llm_concurrency=_env_int(env_values, "CAR_GRAPH_DISAMBIGUATION_LLM_CONCURRENCY", cfg.llm_concurrency),
        llm_timeout=_env_int(env_values, "CAR_GRAPH_DISAMBIGUATION_LLM_TIMEOUT", cfg.llm_timeout),
        llm_max_tokens=_env_int(env_values, "CAR_GRAPH_DISAMBIGUATION_LLM_MAX_TOKENS", cfg.llm_max_tokens),
        llm_temperature=_env_float(env_values, "CAR_GRAPH_DISAMBIGUATION_LLM_TEMPERATURE", cfg.llm_temperature),
        llm_retries=_env_int(env_values, "CAR_GRAPH_DISAMBIGUATION_LLM_RETRIES", cfg.llm_retries),
        disable_thinking=_env_bool(env_values, "CAR_GRAPH_DISAMBIGUATION_DISABLE_THINKING", cfg.disable_thinking),
    )


def make_context(
    *,
    settings: DisambiguationSettings | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    input_version: str | None = None,
    output_version: str | None = None,
) -> DisambiguationContext:
    cfg = settings or load_settings()
    in_version = input_version or cfg.input_version
    out_version = output_version or cfg.output_version
    resolved_input = Path(input_dir) if input_dir else DEFAULT_VERSION_ROOT / in_version
    resolved_output = Path(output_dir) if output_dir else DEFAULT_VERSION_ROOT / out_version
    return DisambiguationContext(
        settings=cfg,
        input_dir=resolved_input.expanduser().resolve(),
        output_dir=resolved_output.expanduser().resolve(),
    )


SETTINGS = load_settings()
