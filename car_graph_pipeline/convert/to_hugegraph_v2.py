#!/usr/bin/env python3
"""转换新文档抽取结果为 HugeGraph 导入格式 (含 vehicle_model 属性)。

输入: output/new_doc_output/merged_entities.json, merged_relations.json
输出: output/new_doc_output/hugegraph_vertices.json, hugegraph_edges.json, hugegraph_schema.json
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "output" / "new_doc_output"
logger = logging.getLogger(__name__)

# Schema — 每个 label 都新增 vehicle_model 属性
VERTEX_LABELS = [
    {
        "id": 1,
        "name": "VehicleBrand",
        "properties": ["entity_id", "brand_name", "country", "vehicle_model", "description", "source_doc"],
        "primary_keys": ["entity_id"],
        "nullable_keys": ["brand_name", "country", "vehicle_model", "description", "source_doc"],
    },
    {
        "id": 2,
        "name": "VehicleModel",
        "properties": [
            "entity_id",
            "model_name",
            "year",
            "series",
            "fuel_type",
            "config",
            "vehicle_model",
            "description",
            "source_doc",
        ],
        "primary_keys": ["entity_id"],
        "nullable_keys": [
            "model_name",
            "year",
            "series",
            "fuel_type",
            "config",
            "vehicle_model",
            "description",
            "source_doc",
        ],
    },
    {
        "id": 3,
        "name": "VehicleSystem",
        "properties": ["entity_id", "system_name", "system_type", "vehicle_model", "description", "source_doc"],
        "primary_keys": ["entity_id"],
        "nullable_keys": ["system_name", "system_type", "vehicle_model", "description", "source_doc"],
    },
    {
        "id": 4,
        "name": "Component",
        "properties": [
            "entity_id",
            "comp_name",
            "component_type",
            "location",
            "vehicle_model",
            "description",
            "source_doc",
        ],
        "primary_keys": ["entity_id"],
        "nullable_keys": ["comp_name", "component_type", "location", "vehicle_model", "description", "source_doc"],
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
            "vehicle_model",
            "description",
            "source_doc",
        ],
        "primary_keys": ["entity_id"],
        "nullable_keys": [
            "func_name",
            "function_type",
            "trigger_condition",
            "alert_method",
            "warnings",
            "vehicle_model",
            "description",
            "source_doc",
        ],
    },
    {
        "id": 6,
        "name": "Status",
        "properties": [
            "entity_id",
            "status_name",
            "status_type",
            "perceivable_way",
            "vehicle_model",
            "description",
            "source_doc",
        ],
        "primary_keys": ["entity_id"],
        "nullable_keys": [
            "status_name",
            "status_type",
            "perceivable_way",
            "vehicle_model",
            "description",
            "source_doc",
        ],
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
            "vehicle_model",
            "description",
            "source_doc",
        ],
        "primary_keys": ["entity_id"],
        "nullable_keys": [
            "fault_name",
            "fault_type",
            "severity",
            "drivable",
            "risk_desc",
            "vehicle_model",
            "description",
            "source_doc",
        ],
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
            "vehicle_model",
            "description",
            "source_doc",
        ],
        "primary_keys": ["entity_id"],
        "nullable_keys": [
            "op_name",
            "operation_type",
            "difficulty",
            "steps",
            "precondition",
            "warnings",
            "vehicle_model",
            "description",
            "source_doc",
        ],
    },
    {
        "id": 9,
        "name": "MaintenanceItem",
        "properties": [
            "entity_id",
            "maint_name",
            "item_type",
            "interval",
            "warnings",
            "vehicle_model",
            "description",
            "source_doc",
        ],
        "primary_keys": ["entity_id"],
        "nullable_keys": [
            "maint_name",
            "item_type",
            "interval",
            "warnings",
            "vehicle_model",
            "description",
            "source_doc",
        ],
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
            "vehicle_model",
            "description",
            "source_doc",
        ],
        "primary_keys": ["entity_id"],
        "nullable_keys": [
            "spec_name",
            "value_text",
            "value_num",
            "unit",
            "condition_note",
            "vehicle_model",
            "description",
            "source_doc",
        ],
    },
    {
        "id": 11,
        "name": "Material",
        "properties": [
            "entity_id",
            "material_name",
            "material_type",
            "spec",
            "brand",
            "vehicle_model",
            "description",
            "source_doc",
        ],
        "primary_keys": ["entity_id"],
        "nullable_keys": [
            "material_name",
            "material_type",
            "spec",
            "brand",
            "vehicle_model",
            "description",
            "source_doc",
        ],
    },
]

EDGE_LABELS = [
    {
        "name": "HAS_MODEL",
        "source_label": "VehicleBrand",
        "target_label": "VehicleModel",
        "properties": ["vehicle_model"],
    },
    {
        "name": "HAS_SYSTEM",
        "source_label": "VehicleModel",
        "target_label": "VehicleSystem",
        "properties": ["vehicle_model"],
    },
    {
        "name": "HAS_COMPONENT",
        "source_label": "VehicleModel",
        "target_label": "Component",
        "properties": ["vehicle_model"],
    },
    {
        "name": "HAS_FUNCTION",
        "source_label": "VehicleModel",
        "target_label": "Function",
        "properties": ["vehicle_model"],
    },
    {
        "name": "BELONGS_TO",
        "source_label": "Component",
        "target_label": "VehicleSystem",
        "properties": ["vehicle_model"],
    },
    {"name": "ACTIVATES", "source_label": "Component", "target_label": "Function", "properties": ["vehicle_model"]},
    {"name": "OPERATED_BY", "source_label": "Function", "target_label": "Operation", "properties": ["vehicle_model"]},
    {"name": "OPERATES_ON", "source_label": "Operation", "target_label": "Component", "properties": ["vehicle_model"]},
    {"name": "HAS_STATUS", "source_label": "Component", "target_label": "Status", "properties": ["vehicle_model"]},
    {
        "name": "SYSTEM_HAS_STATUS",
        "source_label": "VehicleSystem",
        "target_label": "Status",
        "properties": ["vehicle_model"],
    },
    {"name": "CAUSED_BY", "source_label": "Status", "target_label": "Fault", "properties": ["vehicle_model"]},
    {"name": "LEADS_TO", "source_label": "Fault", "target_label": "Fault", "properties": ["vehicle_model"]},
    {"name": "AFFECTS", "source_label": "Fault", "target_label": "VehicleSystem", "properties": ["vehicle_model"]},
    {"name": "RESOLVED_BY", "source_label": "Status", "target_label": "Operation", "properties": ["vehicle_model"]},
    {
        "name": "FAULT_RESOLVED_BY",
        "source_label": "Fault",
        "target_label": "Operation",
        "properties": ["vehicle_model"],
    },
    {"name": "HAS_SPEC", "source_label": "Component", "target_label": "Specification", "properties": ["vehicle_model"]},
    {
        "name": "MODEL_HAS_SPEC",
        "source_label": "VehicleModel",
        "target_label": "Specification",
        "properties": ["vehicle_model"],
    },
    {"name": "REQUIRES", "source_label": "Operation", "target_label": "Material", "properties": ["vehicle_model"]},
    {
        "name": "APPLICABLE_TO",
        "source_label": "MaintenanceItem",
        "target_label": "VehicleModel",
        "properties": ["vehicle_model"],
    },
    {
        "name": "MAINT_HAS_SPEC",
        "source_label": "MaintenanceItem",
        "target_label": "Specification",
        "properties": ["vehicle_model"],
    },
    {
        "name": "MAINT_REQUIRES",
        "source_label": "MaintenanceItem",
        "target_label": "Material",
        "properties": ["vehicle_model"],
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

VERTEX_PROPERTIES_MAP = {v["name"]: set(v["properties"]) for v in VERTEX_LABELS}
VERTEX_LABEL_INDEX = {v["name"]: v["id"] for v in VERTEX_LABELS}


def convert_entity_to_vertex(entity: dict) -> dict:
    label = entity["type"]
    name_key = NAME_PROPERTY_MAP.get(label)
    allowed_props = VERTEX_PROPERTIES_MAP.get(label, set())

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
                props[k] = str(v) if not isinstance(v, str) else v

    if entity.get("source_doc"):
        props["source_doc"] = entity["source_doc"]

    return {
        "id": f"{VERTEX_LABEL_INDEX[label]}:{entity_id}",
        "label": label,
        "type": "vertex",
        "properties": props,
    }


def convert_relation_to_edge(relation: dict, entity_id_set: set) -> dict | None:
    source_id = relation.get("source_entity_id", "").replace("::", "__")
    target_id = relation.get("target_entity_id", "").replace("::", "__")

    if not source_id or not target_id:
        return None
    if source_id not in entity_id_set or target_id not in entity_id_set:
        return None

    source_label = relation.get("source_type", "")
    target_label = relation.get("target_type", "")
    source_idx = VERTEX_LABEL_INDEX.get(source_label)
    target_idx = VERTEX_LABEL_INDEX.get(target_label)

    if not source_idx or not target_idx:
        return None

    edge_props = {}
    vehicle_model = relation.get("properties", {}).get("vehicle_model", "")
    if vehicle_model:
        edge_props["vehicle_model"] = vehicle_model

    return {
        "label": relation["type"],
        "type": "edge",
        "outV": f"{source_idx}:{source_id}",
        "outVLabel": source_label,
        "inV": f"{target_idx}:{target_id}",
        "inVLabel": target_label,
        "properties": edge_props,
    }


def run_convert():
    entities_path = OUTPUT_DIR / "merged_entities.json"
    relations_path = OUTPUT_DIR / "merged_relations.json"

    with open(entities_path, encoding="utf-8") as f:
        entities = json.load(f)
    with open(relations_path, encoding="utf-8") as f:
        relations = json.load(f)

    logger.info("输入: %s 实体, %s 关系", len(entities), len(relations))

    # 转换顶点
    vertices = []
    entity_id_set = set()
    skipped = 0
    for entity in entities:
        if entity["type"] not in VERTEX_LABEL_INDEX:
            skipped += 1
            continue
        vertex = convert_entity_to_vertex(entity)
        vertices.append(vertex)
        entity_id_set.add(entity["entity_id"].replace("::", "__"))

    logger.info("  顶点: %s (跳过 %s)", len(vertices), skipped)

    # 转换边
    edges = []
    skipped_edges = 0
    for relation in relations:
        edge = convert_relation_to_edge(relation, entity_id_set)
        if edge:
            edges.append(edge)
        else:
            skipped_edges += 1

    logger.info("  边: %s (跳过 %s)", len(edges), skipped_edges)

    # 保存
    vertices_path = OUTPUT_DIR / "hugegraph_vertices.json"
    edges_path = OUTPUT_DIR / "hugegraph_edges.json"
    vertices_path.write_text(json.dumps(vertices, ensure_ascii=False, indent=2), encoding="utf-8")
    edges_path.write_text(json.dumps(edges, ensure_ascii=False, indent=2), encoding="utf-8")

    # 保存 schema
    schema = {
        "propertykeys": _build_property_keys(),
        "vertexlabels": VERTEX_LABELS,
        "edgelabels": EDGE_LABELS,
    }
    schema_path = OUTPUT_DIR / "hugegraph_schema.json"
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    # 统计
    v_counter = Counter(v["label"] for v in vertices)
    e_counter = Counter(e["label"] for e in edges)
    logger.info("  顶点类型: %s", dict(v_counter.most_common(5)))
    logger.info("  边类型: %s", dict(e_counter.most_common(5)))
    logger.info("  输出: %s", vertices_path)
    logger.info("  输出: %s", edges_path)
    logger.info("  输出: %s", schema_path)


def _build_property_keys():
    all_props = set()
    for vl in VERTEX_LABELS:
        all_props.update(vl["properties"])
    for el in EDGE_LABELS:
        all_props.update(el.get("properties", []))
    return [{"name": p, "data_type": "TEXT", "cardinality": "SINGLE"} for p in sorted(all_props)]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_convert()
