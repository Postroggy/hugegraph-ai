"""格式转换: 将 extracted_entities/relations.json 转换为 HugeGraph 导入格式。"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from ..config import OUTPUT_DIR, get_version_dir

logger = logging.getLogger(__name__)

# ============================================================================
# Schema 定义
# ============================================================================

VERTEX_LABELS = [
    {
        "id": 1,
        "name": "VehicleBrand",
        "properties": ["entity_id", "brand_name", "country", "description", "source_doc"],
        "primary_keys": ["entity_id"],
    },
    {
        "id": 2,
        "name": "VehicleModel",
        "properties": ["entity_id", "model_name", "year", "series", "fuel_type", "config", "description", "source_doc"],
        "primary_keys": ["entity_id"],
    },
    {
        "id": 3,
        "name": "VehicleSystem",
        "properties": ["entity_id", "system_name", "system_type", "description", "source_doc"],
        "primary_keys": ["entity_id"],
    },
    {
        "id": 4,
        "name": "Component",
        "properties": ["entity_id", "comp_name", "component_type", "location", "description", "source_doc"],
        "primary_keys": ["entity_id"],
    },
    {
        "id": 5,
        "name": "Function",
        "properties": [
            "entity_id",
            "func_name",
            "function_type",
            "trigger_condition",
            "alert_method",
            "warnings",
            "description",
            "source_doc",
        ],
        "primary_keys": ["entity_id"],
    },
    {
        "id": 6,
        "name": "Status",
        "properties": ["entity_id", "status_name", "status_type", "perceivable_way", "description", "source_doc"],
        "primary_keys": ["entity_id"],
    },
    {
        "id": 7,
        "name": "Fault",
        "properties": [
            "entity_id",
            "fault_name",
            "fault_type",
            "severity",
            "drivable",
            "risk_desc",
            "description",
            "source_doc",
        ],
        "primary_keys": ["entity_id"],
    },
    {
        "id": 8,
        "name": "Operation",
        "properties": [
            "entity_id",
            "op_name",
            "operation_type",
            "difficulty",
            "steps",
            "precondition",
            "warnings",
            "description",
            "source_doc",
        ],
        "primary_keys": ["entity_id"],
    },
    {
        "id": 9,
        "name": "MaintenanceItem",
        "properties": ["entity_id", "maint_name", "item_type", "interval", "warnings", "description", "source_doc"],
        "primary_keys": ["entity_id"],
    },
    {
        "id": 10,
        "name": "Specification",
        "properties": [
            "entity_id",
            "spec_name",
            "value_text",
            "value_num",
            "unit",
            "condition_note",
            "description",
            "source_doc",
        ],
        "primary_keys": ["entity_id"],
    },
    {
        "id": 11,
        "name": "Material",
        "properties": ["entity_id", "material_name", "material_type", "spec", "brand", "description", "source_doc"],
        "primary_keys": ["entity_id"],
    },
]

NAME_PROPERTY_MAP = {
    "VehicleBrand": "brand_name",
    "VehicleModel": "model_name",
    "VehicleSystem": "system_name",
    "Component": "comp_name",
    "Function": "func_name",
    "Status": "status_name",
    "Fault": "fault_name",
    "Operation": "op_name",
    "MaintenanceItem": "maint_name",
    "Specification": "spec_name",
    "Material": "material_name",
}

VERTEX_PROPERTIES_MAP = {v["name"]: v["properties"] for v in VERTEX_LABELS}
VERTEX_LABEL_INDEX = {v["name"]: v["id"] for v in VERTEX_LABELS}


def convert_entity_to_vertex(entity: dict) -> dict:
    """将 entity 转换为 HugeGraph vertex 格式。"""
    label = entity["type"]
    name_key = NAME_PROPERTY_MAP.get(label)
    allowed_props = set(VERTEX_PROPERTIES_MAP.get(label, []))

    props = {}
    entity_id = entity["entity_id"].replace("::", "__")
    props["entity_id"] = entity_id

    if name_key:
        props[name_key] = entity.get("name", "")

    raw_props = entity.get("properties", {})
    for k, v in raw_props.items():
        if k in allowed_props and v is not None:
            if isinstance(v, (list, dict)):
                props[k] = json.dumps(v, ensure_ascii=False)
            else:
                props[k] = v

    if entity.get("source_doc"):
        props["source_doc"] = entity["source_doc"]
    if entity.get("description"):
        props["description"] = entity["description"]
    elif raw_props.get("description"):
        props["description"] = raw_props["description"]

    return {
        "id": f"{VERTEX_LABEL_INDEX[label]}:{entity_id}",
        "label": label,
        "type": "vertex",
        "properties": props,
    }


def convert_relation_to_edge(relation: dict, entity_id_map: dict) -> dict | None:
    """将 relation 转换为 HugeGraph edge 格式。"""
    source_id = relation.get("source_entity_id", "").replace("::", "__")
    target_id = relation.get("target_entity_id", "").replace("::", "__")

    if source_id not in entity_id_map or target_id not in entity_id_map:
        return None

    source_label = relation.get("source_type", "")
    target_label = relation.get("target_type", "")
    source_label_idx = VERTEX_LABEL_INDEX.get(source_label)
    target_label_idx = VERTEX_LABEL_INDEX.get(target_label)

    if not source_label_idx or not target_label_idx:
        return None

    return {
        "label": relation["type"],
        "type": "edge",
        "outV": f"{source_label_idx}:{source_id}",
        "outVLabel": source_label,
        "inV": f"{target_label_idx}:{target_id}",
        "inVLabel": target_label,
        "properties": {},
    }


def run_convert(version: str | None = None, output_dir: Path | None = None):
    """执行格式转换。

    Args:
        version: 数据版本 (如 v3_disambiguated), 默认使用 config.CURRENT_VERSION
        output_dir: 输出目录, 默认为 OUTPUT_DIR
    """
    ver_dir = get_version_dir(version)
    out_dir = output_dir or OUTPUT_DIR

    # 尝试多种文件名（消歧输出用 merged_*, 抽取输出用 extracted_*）
    entities_path = ver_dir / "merged_entities.json"
    if not entities_path.exists():
        entities_path = ver_dir / "extracted_entities.json"
    relations_path = ver_dir / "merged_relations.json"
    if not relations_path.exists():
        relations_path = ver_dir / "extracted_relations.json"

    if not entities_path.exists():
        logger.error(f"找不到实体文件: {ver_dir}")
        return False
    if not relations_path.exists():
        logger.error(f"找不到关系文件: {ver_dir}")
        return False

    logger.info(f"读取数据: {ver_dir}")
    with open(entities_path, encoding="utf-8") as f:
        entities = json.load(f)
    with open(relations_path, encoding="utf-8") as f:
        relations = json.load(f)

    logger.info(f"  实体数: {len(entities)}, 关系数: {len(relations)}")

    # 转换顶点
    vertices = []
    entity_id_map = {}
    skipped = 0
    for entity in entities:
        if "type" not in entity or "entity_id" not in entity:
            skipped += 1
            continue
        if entity["type"] not in VERTEX_LABEL_INDEX:
            skipped += 1
            continue
        vertex = convert_entity_to_vertex(entity)
        vertices.append(vertex)
        entity_id_map[entity["entity_id"].replace("::", "__")] = True

    logger.info(f"  顶点: {len(vertices)} (跳过 {skipped})")

    # 转换边
    edges = []
    skipped_edges = 0
    for relation in relations:
        edge = convert_relation_to_edge(relation, entity_id_map)
        if edge:
            edges.append(edge)
        else:
            skipped_edges += 1

    logger.info(f"  边: {len(edges)} (跳过 {skipped_edges})")

    # 写入输出
    out_dir.mkdir(parents=True, exist_ok=True)
    vertices_path = out_dir / "hugegraph_vertices.json"
    edges_path = out_dir / "hugegraph_edges.json"

    with open(vertices_path, "w", encoding="utf-8") as f:
        json.dump(vertices, f, ensure_ascii=False, indent=2)
    with open(edges_path, "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)

    logger.info(f"  输出: {vertices_path}, {edges_path}")

    # 统计
    v_counter = Counter(v["label"] for v in vertices)
    e_counter = Counter(e["label"] for e in edges)
    logger.info(f"  顶点类型: {dict(v_counter.most_common())}")
    logger.info(f"  边类型: {dict(e_counter.most_common())}")

    return True
