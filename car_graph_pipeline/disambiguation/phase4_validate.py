"""Phase 4: 验证 — 检查合并结果的正确性。"""

import json
import logging
from collections import Counter
from typing import List

from .phase1_candidates import get_vehicle_model
from .settings import DisambiguationContext, make_context

logger = logging.getLogger(__name__)

def validate(
    merged_entities: List[dict],
    merged_relations: List[dict],
    ctx: DisambiguationContext | None = None,
) -> dict:
    """执行所有验证检查。"""
    context = ctx or make_context()
    cfg = context.settings
    logger.info("Phase 4: 验证...")
    results = {}

    # 原始数据
    with open(context.input_entities_path, "r", encoding="utf-8") as f:
        orig_entities = json.load(f)
    with open(context.input_relations_path, "r", encoding="utf-8") as f:
        orig_relations = json.load(f)

    entity_ids = {e["entity_id"] for e in merged_entities}

    # Check 1: entity_id 唯一性
    id_counts = Counter(e["entity_id"] for e in merged_entities)
    duplicates = {k: v for k, v in id_counts.items() if v > 1}
    results["entity_id_unique"] = len(duplicates) == 0
    if duplicates:
        logger.error(f"  ❌ entity_id 重复: {len(duplicates)} 个")
        for eid, cnt in list(duplicates.items())[:5]:
            logger.error(f"    {eid}: {cnt}")
    else:
        logger.info("  ✅ entity_id 唯一性通过")

    # Check 2: 悬挂边检查
    dangling_src = 0
    dangling_tgt = 0
    for rel in merged_relations:
        if rel["source_entity_id"] not in entity_ids:
            dangling_src += 1
        if rel["target_entity_id"] not in entity_ids:
            dangling_tgt += 1

    # 对比原始数据的悬挂边
    orig_ids = {e["entity_id"] for e in orig_entities}
    orig_dangling = sum(
        1 for r in orig_relations
        if r["source_entity_id"] not in orig_ids or r["target_entity_id"] not in orig_ids
    )
    total_dangling = dangling_src + dangling_tgt
    results["dangling_edges"] = total_dangling
    results["dangling_edges_original"] = orig_dangling
    results["dangling_not_increased"] = total_dangling <= orig_dangling

    if total_dangling <= orig_dangling:
        logger.info(f"  ✅ 悬挂边: {orig_dangling} → {total_dangling}")
    else:
        logger.error(f"  ❌ 悬挂边增加: {orig_dangling} → {total_dangling} (+{total_dangling - orig_dangling})")

    # Check 3: 实体数量合理性（不应减少超过20%）
    reduction_pct = 0.0
    if orig_entities:
        reduction_pct = (len(orig_entities) - len(merged_entities)) / len(orig_entities) * 100
    results["entity_reduction_pct"] = round(reduction_pct, 2)
    results["entity_reduction_reasonable"] = reduction_pct < cfg.max_entity_reduction_pct

    if results["entity_reduction_reasonable"]:
        logger.info(f"  ✅ 实体减少: {reduction_pct:.1f}% ({len(orig_entities)} → {len(merged_entities)})")
    else:
        logger.warning(f"  ⚠️ 实体减少过多: {reduction_pct:.1f}%")

    # Check 4: 边数量合理性（不应减少超过15%）
    edge_reduction_pct = 0.0
    if orig_relations:
        edge_reduction_pct = (len(orig_relations) - len(merged_relations)) / len(orig_relations) * 100
    results["relation_reduction_pct"] = round(edge_reduction_pct, 2)
    results["relation_reduction_reasonable"] = edge_reduction_pct < cfg.max_relation_reduction_pct

    if results["relation_reduction_reasonable"]:
        logger.info(f"  ✅ 关系减少: {edge_reduction_pct:.1f}% ({len(orig_relations)} → {len(merged_relations)})")
    else:
        logger.warning(f"  ⚠️ 关系减少过多: {edge_reduction_pct:.1f}%")

    # Check 5: 非品牌/车型实体必须保持单一车型范围。
    cross_model_issues = 0
    for e in merged_entities:
        etype = e["type"]
        if etype in cfg.skip_types:
            continue
        prop_model = e.get("properties", {}).get("vehicle_model")
        id_model = get_vehicle_model(e)
        if prop_model and id_model and prop_model != id_model:
            logger.error(f"  ❌ 车型不一致: {e['entity_id']} prop={prop_model} id={id_model}")
            cross_model_issues += 1
    results["cross_model_issues"] = cross_model_issues
    results["vehicle_scope_consistent"] = cross_model_issues == 0
    if cross_model_issues == 0:
        logger.info("  ✅ 车型范围一致性通过")
    else:
        logger.error(f"  ❌ 车型范围不一致: {cross_model_issues} 个")

    # 总结
    all_pass = (
        results["entity_id_unique"]
        and results["dangling_not_increased"]
        and results["entity_reduction_reasonable"]
        and results["relation_reduction_reasonable"]
        and results["vehicle_scope_consistent"]
    )
    results["all_pass"] = all_pass

    if all_pass:
        logger.info("\n  ✅✅✅ 全部验证通过!")
    else:
        logger.warning("\n  ⚠️ 部分验证未通过，请检查")

    # 保存验证结果
    with open(context.output_dir / "validation_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    context = make_context()
    entities_path = context.merged_entities_path
    relations_path = context.merged_relations_path
    if not entities_path.exists():
        print("请先运行 Phase 3")
        exit(1)
    with open(entities_path, "r") as f:
        entities = json.load(f)
    with open(relations_path, "r") as f:
        relations = json.load(f)
    results = validate(entities, relations, context)
    print(f"\n验证完成: {'✅ 全部通过' if results['all_pass'] else '⚠️ 部分失败'}")
