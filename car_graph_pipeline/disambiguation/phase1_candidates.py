"""Phase 1: 候选对发现 — 同 (type, vehicle_model) 分组内向量相似度 + 编辑距离。"""

import asyncio
import json
import logging
from collections import defaultdict
from typing import Any

from .settings import DisambiguationContext, make_context

logger = logging.getLogger(__name__)


def get_model_from_id(entity_id: str, etype: str) -> str | None:
    """从 entity_id 提取车型名。"""
    parts = entity_id.split("::")
    if etype in ("VehicleBrand", "VehicleModel"):
        return None
    if len(parts) >= 3:
        return parts[1]
    return None


def get_vehicle_model(entity: dict) -> str | None:
    """优先从属性读取车型, 兼容旧版 entity_id 编码。"""
    props = entity.get("properties", {})
    model = props.get("vehicle_model") or entity.get("vehicle_model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return get_model_from_id(entity.get("entity_id", ""), entity.get("type", ""))


def build_embed_text(entity: dict) -> str:
    """根据实体类型构建 embedding 文本。"""
    etype = entity["type"]
    name = entity.get("name", "")
    props = entity.get("properties", {})

    if etype == "Function":
        trigger = props.get("trigger_condition", "")
        return f"{name} {trigger}".strip()
    elif etype == "Operation":
        steps = props.get("steps", [])
        step0 = steps[0] if steps else ""
        if isinstance(step0, dict):
            step0 = step0.get("description", str(step0))
        return f"{name} {step0}".strip()
    elif etype == "Component":
        comp_type = props.get("component_type", "")
        location = props.get("location", "")
        return f"{name} {comp_type} {location}".strip()
    elif etype == "Status":
        status_type = props.get("status_type", "")
        return f"{name} {status_type}".strip()
    elif etype == "Fault":
        fault_type = props.get("fault_type", "")
        return f"{name} {fault_type}".strip()
    elif etype == "Specification":
        spec_type = props.get("spec_type", "")
        return f"{name} {spec_type}".strip()
    else:
        return name


async def batch_embed(
    texts: list[str],
    session: Any,
    semaphore: asyncio.Semaphore,
    ctx: DisambiguationContext,
) -> list[Any]:
    """批量获取 embedding, 支持并发控制和指数退避重试。"""
    import aiohttp
    import numpy as np

    cfg = ctx.settings
    results = [None] * len(texts)

    async def _embed_batch(batch_texts: list[str], indices: list[int]):
        async with semaphore:
            payload = {
                "model": cfg.embedding_model,
                "input": [t[: cfg.embedding_max_chars] for t in batch_texts],
            }
            for attempt in range(4):
                try:
                    async with session.post(
                        cfg.embedding_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=cfg.embedding_timeout),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            embeddings = data["embeddings"]
                            for i, idx in enumerate(indices):
                                results[idx] = np.array(embeddings[i], dtype=np.float32)
                            return
                        elif resp.status == 429:
                            wait = 5 * (2**attempt)
                            logger.warning(f"Embed rate limited, waiting {wait}s...")
                            await asyncio.sleep(wait)
                        else:
                            err_text = await resp.text()
                            logger.warning(f"Embed API error {resp.status} (attempt {attempt + 1}): {err_text[:100]}")
                            await asyncio.sleep(2 * (attempt + 1))
                except asyncio.TimeoutError:
                    wait = 3 * (attempt + 1)
                    logger.warning(f"Embed timeout (attempt {attempt + 1}), retry in {wait}s")
                    await asyncio.sleep(wait)
                except (aiohttp.ClientError, aiohttp.ServerDisconnectedError) as e:
                    wait = 2 * (attempt + 1)
                    logger.warning(f"Embed connection error (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(wait)
            # 所有重试失败后用零向量
            logger.error(f"Embed batch failed after 4 attempts ({len(batch_texts)} texts)")
            for idx in indices:
                results[idx] = np.zeros(cfg.embedding_dim, dtype=np.float32)

    # 分窗口执行，避免一次性创建全部协程
    window_size = cfg.embedding_concurrency * 2
    for win_start in range(0, len(texts), cfg.embedding_batch_size * window_size):
        tasks = []
        for i in range(
            win_start,
            min(win_start + cfg.embedding_batch_size * window_size, len(texts)),
            cfg.embedding_batch_size,
        ):
            batch = texts[i : i + cfg.embedding_batch_size]
            indices = list(range(i, i + len(batch)))
            tasks.append(_embed_batch(batch, indices))
        await asyncio.gather(*tasks)

    return results


def edit_distance(s1: str, s2: str) -> int:
    """计算编辑距离。"""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def cosine_similarity(a: Any, b: Any) -> float:
    """计算余弦相似度。"""
    import numpy as np

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def is_substring(a: str, b: str) -> bool:
    """判断 a 是否是 b 的子串或反之。"""
    return (a in b or b in a) and a != b


async def find_candidates_async(ctx: DisambiguationContext) -> list[dict]:
    """异步执行候选对发现。"""
    import aiohttp

    cfg = ctx.settings
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Phase 1: 加载数据...")
    with open(ctx.input_entities_path, encoding="utf-8") as f:
        entities = json.load(f)
    with open(ctx.input_relations_path, encoding="utf-8") as f:
        relations = json.load(f)

    # 按 (type, vehicle_model) 分组，避免跨车型消歧。
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in entities:
        etype = e["type"]
        if etype in cfg.skip_types:
            continue
        model = get_vehicle_model(e)
        if model:
            groups[(etype, model)].append(e)

    logger.info(f"  分组数: {len(groups)}")
    total_entities_in_groups = sum(len(v) for v in groups.values())
    logger.info(f"  参与消歧实体数: {total_entities_in_groups}")

    # 构建边索引（用于后续 LLM 判断时提供关联边信息）
    edge_index: dict[str, list[dict]] = defaultdict(list)
    for r in relations:
        edge_index[r["source_entity_id"]].append(r)
        edge_index[r["target_entity_id"]].append(r)

    # 保存边索引供后续 phase2 使用
    # (不在这里保存大文件，phase2 自己读)

    # 对每个分组做 embedding + 相似度计算
    all_candidates = []
    semaphore = asyncio.Semaphore(cfg.embedding_concurrency)

    connector = aiohttp.TCPConnector(limit=cfg.embedding_concurrency, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        for (etype, model), group_entities in sorted(groups.items(), key=lambda x: -len(x[1])):
            if len(group_entities) < 2:
                continue

            logger.info(f"  处理 ({etype}, {model}): {len(group_entities)} 实体")

            # 构建 embed 文本
            texts = [build_embed_text(e) for e in group_entities]

            # 批量 embedding
            embeddings = await batch_embed(texts, session, semaphore, ctx)

            # 计算两两相似度（只保留超过阈值的）
            n = len(group_entities)
            for i in range(n):
                if embeddings[i] is None:
                    continue
                for j in range(i + 1, n):
                    if embeddings[j] is None:
                        continue
                    sim = cosine_similarity(embeddings[i], embeddings[j])
                    if sim < cfg.cosine_similarity_threshold:
                        continue

                    name_a = group_entities[i].get("name", "")
                    name_b = group_entities[j].get("name", "")

                    # 辅助信号
                    reasons = [f"cosine={sim:.4f}"]
                    if is_substring(name_a, name_b):
                        reasons.append("substring")
                    max_len = max(len(name_a), len(name_b))
                    if max_len > 0:
                        ed_ratio = edit_distance(name_a, name_b) / max_len
                        if ed_ratio < cfg.edit_distance_ratio_threshold:
                            reasons.append(f"edit_dist_ratio={ed_ratio:.3f}")

                    all_candidates.append(
                        {
                            "entity_a_id": group_entities[i]["entity_id"],
                            "entity_b_id": group_entities[j]["entity_id"],
                            "entity_a_name": name_a,
                            "entity_b_name": name_b,
                            "type": etype,
                            "model": model,
                            "similarity": round(sim, 4),
                            "reasons": reasons,
                        }
                    )

    logger.info(f"  候选对总数: {len(all_candidates)}")

    # 按相似度降序排序
    all_candidates.sort(key=lambda x: -x["similarity"])

    # 保存中间结果
    with open(ctx.candidates_path, "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, ensure_ascii=False, indent=2)
    logger.info(f"  候选对已保存: {ctx.candidates_path}")

    return all_candidates


def find_candidates(ctx: DisambiguationContext | None = None) -> list[dict]:
    """同步包装。"""
    return asyncio.run(find_candidates_async(ctx or make_context()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    candidates = find_candidates()
    logger.info("完成: %s 个候选对", len(candidates))
