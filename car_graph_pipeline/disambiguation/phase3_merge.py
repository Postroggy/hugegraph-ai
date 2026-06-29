"""Phase 3: 执行合并 — Union-Find + 属性合并 + 边迁移。"""

import copy
import json
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from .settings import DisambiguationContext, make_context

logger = logging.getLogger(__name__)


class UnionFind:
    """Union-Find 用于处理传递性合并。"""

    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.winner: Dict[str, str] = {}  # root -> winner_id

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def merge(self, loser: str, winner: str):
        root_l = self.find(loser)
        root_w = self.find(winner)
        if root_l != root_w:
            self.parent[root_l] = root_w
            self.winner[root_w] = winner

    def get_winner(self, x: str) -> str:
        root = self.find(x)
        return self.winner.get(root, root)


def merge_entity(winner: dict, loser: dict, strategy: str = "union") -> dict:
    """合并两个实体的属性，返回合并后的实体。"""
    merged = copy.deepcopy(winner)
    winner_props = merged.get("properties", {})
    loser_props = loser.get("properties", {})

    if strategy == "union":
        for key, value in loser_props.items():
            if key in {"vehicle_brand", "vehicle_model"}:
                continue
            if key not in winner_props or not winner_props[key]:
                winner_props[key] = value
            elif isinstance(value, list) and isinstance(winner_props[key], list):
                # 列表取并集
                existing = set(json.dumps(x, ensure_ascii=False) for x in winner_props[key])
                for item in value:
                    if json.dumps(item, ensure_ascii=False) not in existing:
                        winner_props[key].append(item)
    elif strategy == "winner_priority":
        for key, value in loser_props.items():
            if key in {"vehicle_brand", "vehicle_model"}:
                continue
            if key not in winner_props:
                winner_props[key] = value

    merged["properties"] = winner_props
    return merged


def execute_merge(
    decisions: List[dict],
    ctx: DisambiguationContext | None = None,
) -> Tuple[List[dict], List[dict], dict]:
    """
    执行合并：
    1. 构建 Union-Find 处理传递性
    2. 合并实体属性
    3. 迁移边 + 去重 + 自环检测

    Returns: (merged_entities, merged_relations, merge_log)
    """
    context = ctx or make_context()
    context.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Phase 3: 加载数据...")
    with open(context.input_entities_path, "r", encoding="utf-8") as f:
        entities = json.load(f)
    with open(context.input_relations_path, "r", encoding="utf-8") as f:
        relations = json.load(f)

    entity_map = {e["entity_id"]: e for e in entities}

    # 只处理 decision == "merge" 的
    merge_decisions = [d for d in decisions if d["decision"] == "merge"]
    logger.info(f"  合并决策: {len(merge_decisions)} 对")

    if not merge_decisions:
        logger.info("  无需合并，直接输出")
        merge_log = {
            "total_merge_decisions": 0,
            "unique_merge_groups": 0,
            "entities_removed": 0,
            "entities_before": len(entities),
            "entities_after": len(entities),
            "relations_before": len(relations),
            "relations_after": len(relations),
            "edges_remapped": 0,
            "self_loops_removed": 0,
            "merge_details": [],
        }
        with open(context.merged_entities_path, "w", encoding="utf-8") as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        with open(context.merged_relations_path, "w", encoding="utf-8") as f:
            json.dump(relations, f, ensure_ascii=False, indent=2)
        with open(context.output_dir / "merge_log.json", "w", encoding="utf-8") as f:
            json.dump(merge_log, f, ensure_ascii=False, indent=2)
        return entities, relations, merge_log

    # Phase 3.1: 构建 Union-Find
    uf = UnionFind()
    for dec in merge_decisions:
        a_id = dec["entity_a_id"]
        b_id = dec["entity_b_id"]
        winner_id = dec.get("winner_id")

        # 确定 winner 和 loser
        if winner_id == a_id:
            uf.merge(b_id, a_id)
        elif winner_id == b_id:
            uf.merge(a_id, b_id)
        else:
            # winner_id 未指定或无效，选属性更多的
            a_props = len(entity_map.get(a_id, {}).get("properties", {}))
            b_props = len(entity_map.get(b_id, {}).get("properties", {}))
            if a_props >= b_props:
                uf.merge(b_id, a_id)
            else:
                uf.merge(a_id, b_id)

    # 计算最终合并映射: loser_id -> winner_id
    # 找出所有被涉及的实体
    involved_ids: Set[str] = set()
    for dec in merge_decisions:
        involved_ids.add(dec["entity_a_id"])
        involved_ids.add(dec["entity_b_id"])

    # 对于每个涉及的 ID，找到其最终 winner
    id_remap: Dict[str, str] = {}  # old_id -> final_winner_id
    losers: Set[str] = set()
    winner_groups: Dict[str, List[str]] = defaultdict(list)  # winner -> [losers]

    for eid in involved_ids:
        final_winner = uf.get_winner(eid)
        if final_winner != eid:
            id_remap[eid] = final_winner
            losers.add(eid)
            winner_groups[final_winner].append(eid)

    logger.info(f"  最终合并组: {len(winner_groups)} 组, 删除 {len(losers)} 个实体")

    # Phase 3.2: 合并实体属性
    # 找到每个决策对应的 strategy
    strategy_map: Dict[Tuple[str, str], str] = {}
    for dec in merge_decisions:
        key = (dec["entity_a_id"], dec["entity_b_id"])
        strategy_map[key] = dec.get("property_merge", "union")
        # 也加反向
        strategy_map[(dec["entity_b_id"], dec["entity_a_id"])] = dec.get("property_merge", "union")

    for winner_id, loser_ids in winner_groups.items():
        if winner_id not in entity_map:
            continue
        for loser_id in loser_ids:
            if loser_id not in entity_map:
                continue
            strategy = strategy_map.get((loser_id, winner_id), "union")
            entity_map[winner_id] = merge_entity(entity_map[winner_id], entity_map[loser_id], strategy)

    # Phase 3.3: 构建新实体列表（排除 losers）
    merged_entities = [e for e in entities if e["entity_id"] not in losers]
    # 更新 winner 实体的属性
    for i, e in enumerate(merged_entities):
        if e["entity_id"] in entity_map and e["entity_id"] in winner_groups:
            merged_entities[i] = entity_map[e["entity_id"]]

    # Phase 3.4: 边迁移 + 去重 + 自环检测
    merged_relations = []
    seen_edges: Set[Tuple[str, str, str]] = set()
    self_loops_removed = 0
    edges_remapped = 0

    for rel in relations:
        src = rel["source_entity_id"]
        tgt = rel["target_entity_id"]

        # 替换 loser → winner
        new_src = id_remap.get(src, src)
        new_tgt = id_remap.get(tgt, tgt)

        if new_src != src or new_tgt != tgt:
            edges_remapped += 1

        # 自环检测
        if new_src == new_tgt:
            self_loops_removed += 1
            continue

        # 去重
        edge_key = (new_src, rel["type"], new_tgt)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        new_rel = {**rel, "source_entity_id": new_src, "target_entity_id": new_tgt}
        merged_relations.append(new_rel)

    logger.info(f"  边迁移: {edges_remapped} 条, 自环移除: {self_loops_removed}, 去重后: {len(merged_relations)}")

    # 构建合并日志
    merge_log = {
        "total_merge_decisions": len(merge_decisions),
        "unique_merge_groups": len(winner_groups),
        "entities_removed": len(losers),
        "entities_before": len(entities),
        "entities_after": len(merged_entities),
        "relations_before": len(relations),
        "relations_after": len(merged_relations),
        "edges_remapped": edges_remapped,
        "self_loops_removed": self_loops_removed,
        "merge_details": [
            {
                "winner": winner_id,
                "losers": loser_ids,
                "group_size": len(loser_ids) + 1,
            }
            for winner_id, loser_ids in winner_groups.items()
        ],
    }

    # 保存结果
    with open(context.merged_entities_path, "w", encoding="utf-8") as f:
        json.dump(merged_entities, f, ensure_ascii=False, indent=2)
    with open(context.merged_relations_path, "w", encoding="utf-8") as f:
        json.dump(merged_relations, f, ensure_ascii=False, indent=2)
    with open(context.output_dir / "merge_log.json", "w", encoding="utf-8") as f:
        json.dump(merge_log, f, ensure_ascii=False, indent=2)

    logger.info(f"  结果已保存到 {context.output_dir}/")
    logger.info(f"  实体: {len(entities)} → {len(merged_entities)} (-{len(losers)})")
    logger.info(f"  关系: {len(relations)} → {len(merged_relations)} (-{len(relations) - len(merged_relations)})")

    return merged_entities, merged_relations, merge_log


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    context = make_context()
    dec_path = context.decisions_path
    if not dec_path.exists():
        print("请先运行 Phase 2 生成 merge_decisions.json")
        exit(1)
    with open(dec_path, "r", encoding="utf-8") as f:
        decisions = json.load(f)
    print(f"加载 {len(decisions)} 个决策")
    entities, relations, log = execute_merge(decisions, context)
    print(f"\n完成: {log['entities_after']} 实体, {log['relations_after']} 关系")
