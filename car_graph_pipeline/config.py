"""统一配置 — 所有阶段共享的路径、服务地址、模型参数。"""
import os
import re
from pathlib import Path

# ============================================================================
# 路径
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = PROJECT_ROOT / "doc"
OUTPUT_DIR = PROJECT_ROOT / "output"
VERSION_DIR = OUTPUT_DIR / "version"
INDEX_DIR = PROJECT_ROOT / "graphrag_index" / "car_graph"

# 当前数据版本（可通过 CLI 覆盖）
CURRENT_VERSION = "v3_disambiguated"


def _load_local_env_config() -> dict[str, str]:
    """Load ignored local .env values without hardcoding secrets in Python."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return {}
    config: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([\w]+)\s*[:=]\s*(.+)$", line)
        if not match:
            continue
        key = match.group(1)
        value = match.group(2).strip().strip('"').strip("'").strip("\u201c\u201d\u2018\u2019")
        config[key] = value
    return config


def _normalize_openai_base_url(raw_url: str) -> str:
    """Normalize legacy gateway base URLs for OpenAI-compatible SDK calls."""
    url = raw_url.rstrip("/")
    if url and not url.endswith("/v1"):
        return f"{url}/v1"
    return url


_LOCAL_ENV = _load_local_env_config()

def get_version_dir(version: str = None) -> Path:
    """获取指定版本的数据目录。"""
    v = version or CURRENT_VERSION
    return VERSION_DIR / v

# ============================================================================
# HugeGraph Server
# ============================================================================
HUGEGRAPH_URL = "http://10.103.169.46:8080"
HUGEGRAPH_GRAPH = "car_graph"
HUGEGRAPH_GRAPHSPACE = "socgs"
HUGEGRAPH_USER = "soc_dp"
HUGEGRAPH_PWD = os.getenv("HUGEGRAPH_PWD", "")
HUGEGRAPH_AUTH = (HUGEGRAPH_USER, HUGEGRAPH_PWD)

# 派生
HUGEGRAPH_API_PREFIX = f"{HUGEGRAPH_URL}/graphspaces/{HUGEGRAPH_GRAPHSPACE}/graphs/{HUGEGRAPH_GRAPH}/graph"
HUGEGRAPH_GREMLIN_URL = f"{HUGEGRAPH_URL}/gremlin"
HUGEGRAPH_GREMLIN_ALIASES = {
    "graph": f"{HUGEGRAPH_GRAPHSPACE}-{HUGEGRAPH_GRAPH}",
    "g": f"__g_{HUGEGRAPH_GRAPHSPACE}-{HUGEGRAPH_GRAPH}",
}

# ============================================================================
# Embedding 服务
# ============================================================================
EMBEDDING_URL = "http://10.45.151.13:8000/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 1024
EMBEDDING_MAX_CHARS = 2048
EMBEDDING_BATCH_SIZE = 50  # 同步批量大小
EMBEDDING_ASYNC_BATCH_SIZE = 16  # 异步批量大小
EMBEDDING_ASYNC_CONCURRENCY = 5  # 异步并发数
EMBEDDING_TIMEOUT = 600  # 秒

# ============================================================================
# LLM 服务
# ============================================================================
LLM_BASE_URL = _normalize_openai_base_url(
    os.getenv("LLM_BASE_URL") or _LOCAL_ENV.get("LLM_BASE_URL", "https://oneapi-comate.baidu-int.com/v1")
)
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or _LOCAL_ENV.get("LLM_API_KEY") or _LOCAL_ENV.get("key", "")
LLM_EXTRACT_KEY = LLM_API_KEY

# 各阶段使用的模型
LLM_EXTRACT_MODEL = "gpt-5.4"  # 抽取
LLM_DISAMBIGUATE_MODEL = "DeepSeek-V4-Flash"  # 消歧
LLM_QA_MODEL = "gpt-5.5"  # 问答
LLM_QA_FALLBACK_MODELS = ["gpt-5.4", "GLM-5.1", "GLM-5-Turbo", "Claude Sonnet 4.6"]

LLM_CONCURRENCY = 5
LLM_TIMEOUT = 120
LLM_MAX_TOKENS = 4096

# ============================================================================
# 消歧阈值
# ============================================================================
COSINE_SIMILARITY_THRESHOLD = 0.85
EDIT_DISTANCE_RATIO_THRESHOLD = 0.3
DISAMBIGUATE_BATCH_SIZE = 8

# 不参与消歧的类型
SKIP_TYPES = {"VehicleBrand", "VehicleModel"}

# ============================================================================
# 图导入
# ============================================================================
VERTEX_BATCH_SIZE = 500
EDGE_BATCH_SIZE = 500

# ============================================================================
# GraphRAG 评测
# ============================================================================
MAX_DEEP = 2
MAX_GRAPH_ITEMS = 50
EDGE_LIMIT_PER_LABEL = 10
SEMANTIC_TOPK = 10
SUBSUMPTION_DEDUP = True
ZERO_HOP = False
SAVE_TRACE = True
