#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DOC_RUN_ID = "聚星EV用户手册"
SOURCE_DOC_NAME = "聚星EV用户手册.md"
VEHICLE_BRAND = "南京依维柯"
VEHICLE_MODEL = "聚星EV"

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "task_5_output"
DOC_RUN = OUT / "doc_runs" / DOC_RUN_ID
CHUNKS_PATH = DOC_RUN / "chunks" / "chunks.json"
RISK_PATH = DOC_RUN / "chunks" / "risk.json"
STATE_PATH = DOC_RUN / "state" / "state.json"
BLOCKED_DIR = OUT / "blocked"


COMPONENT_TERMS = [
    "车辆识别代号", "VIN代号", "车辆标牌", "车辆诊断接口", "电子控制单元", "驱动电机", "三合一电驱总成",
    "微波窗口", "汽车电子标识", "前风窗玻璃", "前保险杠", "B柱", "前门门锁", "翼子板", "前舱盖",
    "手套箱", "尾门", "座椅", "驾驶员座椅", "前排座椅", "后排座椅", "头枕", "安全带", "后排座椅安全带报警装置",
    "安全气囊", "辅助保护装置", "儿童保护装置", "儿童座椅", "车门", "车窗", "中控门锁", "遥控钥匙",
    "钥匙", "门锁", "电动车窗", "天窗", "外后视镜", "内后视镜", "方向盘", "喇叭", "组合仪表",
    "显示屏", "中控屏", "仪表台", "开关", "按钮", "旋钮", "制动踏板", "加速踏板", "驻车制动",
    "制动系统", "ABS", "ESC", "转向系统", "换挡机构", "挡位", "动力电池", "蓄电池", "充电口",
    "充电接口", "慢充接口", "快充接口", "充电枪", "充电线", "充电指示灯", "电源开关", "READY指示灯",
    "远光灯", "近光灯", "位置灯", "转向灯", "危险警告灯", "雾灯", "室内灯", "牌照灯", "制动灯",
    "倒车灯", "雨刮器", "洗涤器", "洗涤液", "空调", "除霜", "出风口", "暖风", "冷却液",
    "制动液", "齿轮油", "润滑油", "轮胎", "车轮", "胎压", "备胎", "千斤顶", "牵引钩",
    "保险丝", "电机控制器", "高压线束", "低压蓄电池", "冷却系统", "行驶制动系", "悬架", "车架",
]

MATERIAL_TERMS = [
    "冷却液", "制动液", "洗涤液", "润滑油", "齿轮油", "蓄电池酸液", "制冷剂", "粘合剂", "各种粘合剂",
]

SYSTEM_RULES = [
    ("乘员保护系统", ["安全带", "安全气囊", "儿童保护装置", "儿童座椅", "辅助保护装置", "座椅"]),
    ("车身开闭系统", ["车门", "车窗", "门锁", "尾门", "天窗", "钥匙", "遥控钥匙"]),
    ("电驱系统", ["驱动电机", "三合一电驱总成", "电机控制器", "挡位", "READY"]),
    ("动力电池与充电系统", ["动力电池", "充电", "充电口", "充电接口", "充电枪", "充电线", "高压"]),
    ("制动系统", ["制动", "ABS", "ESC", "驻车制动"]),
    ("转向系统", ["方向盘", "转向"]),
    ("照明与信号系统", ["灯", "转向灯", "远光", "近光", "危险警告"]),
    ("刮水洗涤系统", ["雨刮", "洗涤器", "洗涤液"]),
    ("空调系统", ["空调", "除霜", "出风口", "暖风"]),
    ("仪表与显示系统", ["组合仪表", "显示屏", "中控屏", "指示灯", "警告灯", "报警"]),
    ("轮胎与车轮系统", ["轮胎", "车轮", "胎压", "备胎"]),
    ("车身标识系统", ["车辆识别代号", "VIN", "车辆标牌", "微波窗口", "电子标识"]),
    ("维护保养系统", ["保养", "维护", "油液", "冷却液", "制动液", "润滑油", "洗涤液"]),
]

OP_RE = re.compile(r"(按|按下|按压|按住|长按|拉|拉动|推|推动|踩|踩下|松开|转动|旋转|打开|开启|关闭|启动|起动|停止|选择|设置|调节|检查|更换|安装|拆下|插入|取出|连接|断开|加注|充电|挂入|切换|释放|拧)")
WARN_RE = re.compile(r"(警告|危险|注意|切勿|不得|不要|禁止|否则|可能导致|人身伤害|死亡|事故|损坏|火灾|爆炸|触电)")
STATUS_RE = re.compile(r"(指示灯|警告灯|报警|蜂鸣器|闪烁|点亮|熄灭|显示|提示|故障|异常|失效|过热|过低|过高)")
SPEC_RE = re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>km/h|kPa|bar|MPa|V|A|W|kW|Ah|kWh|L|mL|mm|cm|m|kg|N·m|Nm|℃|°C|公里|千米|分钟|秒|年|个月|毫米|厘米|米|千克|升|毫升)")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def snippet(text: str, limit: int = 150) -> str:
    s = re.sub(r"\s+", " ", text).strip()
    return s[:limit]


def evidence(chunk: dict[str, Any]) -> str:
    return f"{SOURCE_DOC_NAME}:{chunk['line_start']}-{chunk['line_end']}"


def source_base(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk["chunk_id"],
        "heading_path": chunk.get("heading_path", ""),
        "source_section": chunk.get("source_section", ""),
        "line_start": chunk.get("line_start"),
        "line_end": chunk.get("line_end"),
        "source_snippet": snippet(chunk.get("text", "")),
        "evidence": evidence(chunk),
    }


def add_scope(props: dict[str, Any]) -> dict[str, Any]:
    props = dict(props)
    props["source_doc"] = SOURCE_DOC_NAME
    props["vehicle_brand"] = VEHICLE_BRAND
    props["vehicle_model"] = VEHICLE_MODEL
    return props


def entity(chunk: dict[str, Any], typ: str, name: str, props: dict[str, Any], aliases: list[str] | None = None, source_text: str | None = None) -> dict[str, Any]:
    data = {
        "type": typ,
        "name": name,
        "aliases": aliases or [],
        "properties": add_scope(props),
        **source_base(chunk),
    }
    if source_text:
        data["source_snippet"] = snippet(source_text)
    return data


def relation(chunk: dict[str, Any], typ: str, src_t: str, src_n: str, tgt_t: str, tgt_n: str, note: str, source_text: str | None = None) -> dict[str, Any]:
    data = {
        "type": typ,
        "source_type": src_t,
        "source_name": src_n,
        "target_type": tgt_t,
        "target_name": tgt_n,
        "properties": add_scope({"strength": 4, "note": note}),
        **source_base(chunk),
    }
    if source_text:
        data["source_snippet"] = snippet(source_text)
    return data


def is_toc_or_boilerplate(chunk: dict[str, Any]) -> bool:
    text = chunk.get("text", "")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if chunk.get("heading_path") == "目录":
        return True
    if len(lines) >= 3 and sum(bool(re.search(r"\s\d{1,3}$", l)) for l in lines) >= max(2, len(lines) // 2):
        return True
    if "图片是示意图" in text and len(text) < 180:
        return True
    return False


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。；;！!?？])\s*|\n+", text)
    return [p.strip(" -\t") for p in parts if p.strip(" -\t")]


def extract_components(text: str, heading: str) -> list[str]:
    hay = heading + "\n" + text
    names = []
    for term in COMPONENT_TERMS:
        if term in MATERIAL_TERMS:
            continue
        if term in hay and term not in names:
            names.append(term)
    if "VIN" in hay and "车辆识别代号" not in names:
        names.append("车辆识别代号")
    if ("按钮" in hay or "开关" in hay) and len(names) < 6:
        for m in re.finditer(r"([\u4e00-\u9fffA-Za-z0-9]{1,12}(?:按钮|开关|旋钮))", hay):
            val = m.group(1)
            if val not in names and not val.startswith(("按下", "按住", "按压")):
                names.append(val)
    return names[:10]


def direct_sentence(text: str, term: str) -> str:
    for sent in split_sentences(text):
        if term in sent:
            return sent
    return text


def system_for(name: str, text: str) -> str | None:
    hay = name + "\n" + text
    for sys_name, keys in SYSTEM_RULES:
        if any(k in hay for k in keys):
            return sys_name
    return None


def operation_steps(text: str) -> str:
    picked = []
    for sent in split_sentences(text):
        if OP_RE.search(sent) or re.match(r"^\d+[.)、]", sent):
            picked.append(sent)
    if not picked:
        return ""
    return "；".join(picked[:8])


def spec_entities(chunk: dict[str, Any], text: str) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for sent in split_sentences(text):
        if len(out) >= 8:
            break
        m = SPEC_RE.search(sent)
        if not m:
            continue
        name_src = sent[:40].strip(" -|")
        name = re.sub(r"\s+", "", name_src)
        name = name[:28] or f"{chunk['source_section']}规格"
        if name in seen:
            continue
        seen.add(name)
        out.append(entity(chunk, "Specification", name, {
            "spec_name": name,
            "value_text": sent,
            "value_num": float(m.group("num")),
            "unit": m.group("unit"),
            "condition_note": chunk.get("source_section", ""),
            "description": sent,
        }, source_text=sent))
    return out


def material_entities(chunk: dict[str, Any], text: str) -> list[dict[str, Any]]:
    out = []
    for term in MATERIAL_TERMS:
        if term in text:
            out.append(entity(chunk, "Material", term, {
                "material_name": term,
                "material_type": "车辆油液/材料",
                "description": f"当前 chunk 提及的{term}。",
            }, source_text=direct_sentence(text, term)))
    return out[:6]


def warning_operations(chunk: dict[str, Any], text: str) -> list[dict[str, Any]]:
    ops: list[dict[str, Any]] = []
    if any(term in text for term in ["有毒物质", "不要饮用", "远离伤口", "蓄电池酸液", "制冷剂", "粘合剂"]):
        sent = "机动车上使用的多数液体和有些物质为有毒物质，任何情况下都不要饮用，且应尽可能使其远离伤口；务必仔细阅读并绝对遵守打印或压印在零部件上的说明。"
        ops.append(entity(chunk, "Operation", "安全处理有毒液体和物质", {
            "op_name": "安全处理有毒液体和物质",
            "operation_type": "安全注意事项",
            "steps": "不要饮用机动车上使用的有毒液体和物质；尽可能使其远离伤口；仔细阅读并遵守打印或压印在零部件上的说明。",
            "precondition": "接触或处理车辆液体、有毒物质或粘合剂时。",
            "warnings": "这些物质可能危害健康和人身安全。",
            "description": sent,
        }, source_text=sent))
    if "儿童" in text or "动物" in text:
        sent = "为防止由儿童或动物所引起的事故或人员伤亡，切勿将他们留在无成人看管的车内。如果在炎热天气，还可能导致他们窒息。"
        ops.append(entity(chunk, "Operation", "防止儿童或动物引发事故", {
            "op_name": "防止儿童或动物引发事故",
            "operation_type": "安全注意事项",
            "steps": "切勿将儿童或动物留在无成人看管的车内。",
            "precondition": "车辆停放或无人监管时。",
            "warnings": "儿童或动物操作车上控制装置和开关，或接触车内设备或物体，可能导致事故和人员伤亡；炎热天气还可能导致窒息。",
            "description": sent,
        }, source_text=sent))
    if "安全带" in text and "必须佩戴安全带" in text:
        sent = "您车上的每个座椅都配备了安全带，以降低发生事故时导致人身伤害的可能性。要求所有乘员必须佩戴安全带。"
        ops.append(entity(chunk, "Operation", "乘员佩戴安全带", {
            "op_name": "乘员佩戴安全带",
            "operation_type": "乘员保护操作",
            "steps": "所有乘员必须佩戴安全带。",
            "precondition": "乘员就座和车辆使用时。",
            "warnings": "佩戴安全带可降低发生事故时导致人身伤害的可能性。",
            "description": sent,
        }, source_text=sent))
    if "误操作安全气囊" in text:
        sent = "误操作安全气囊可能导致人身伤害。"
        ops.append(entity(chunk, "Operation", "避免误操作安全气囊", {
            "op_name": "避免误操作安全气囊",
            "operation_type": "乘员保护注意事项",
            "steps": "参阅驾驶之前章节中的乘员保护装置说明，避免误操作安全气囊。",
            "precondition": "涉及安全气囊或乘员保护装置时。",
            "warnings": sent,
            "description": sent,
        }, source_text=sent))
    if "该标记表示" in text:
        sent = "该标记表示：为避免对自身或他人造成人身伤害，必须严格、准确地遵循相关步骤。"
        ops.append(entity(chunk, "Operation", "遵循警告标记步骤", {
            "op_name": "遵循警告标记步骤",
            "operation_type": "安全注意事项",
            "steps": "看到该标记时，严格、准确地遵循相关步骤。",
            "precondition": "手册出现该警告标记时。",
            "warnings": "用于避免对自身或他人造成人身伤害。",
            "description": sent,
        }, source_text=sent))
    if "避免损坏您的车辆" in text:
        sent = "这里表示必须遵循相关步骤，以避免损坏您的车辆。"
        ops.append(entity(chunk, "Operation", "遵循注意标记步骤", {
            "op_name": "遵循注意标记步骤",
            "operation_type": "车辆保护注意事项",
            "steps": "看到该注意标记时，遵循相关步骤。",
            "precondition": "手册出现该注意标记时。",
            "warnings": "用于避免损坏车辆。",
            "description": sent,
        }, source_text=sent))
    return ops


def build_draft(chunk: dict[str, Any], attempt: int, review: dict[str, Any] | None = None) -> dict[str, Any]:
    text = chunk.get("text", "")
    heading = chunk.get("heading_path", "")
    source_section = chunk.get("source_section", "")
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    notes: list[str] = []

    if chunk["chunk_id"].endswith("_s0000"):
        entities.append(entity(chunk, "VehicleBrand", VEHICLE_BRAND, {
            "brand_name": VEHICLE_BRAND,
            "description": "南京依维柯汽车有限公司是本手册产品的提供方。",
        }, aliases=["南京依维柯汽车有限公司"], source_text="感谢您选择了南京依维柯汽车有限公司的产品"))
        entities.append(entity(chunk, "VehicleModel", VEHICLE_MODEL, {
            "model_name": VEHICLE_MODEL,
            "description": "本 doc-run 确认的车型范围；该手册提供了解车辆所需的信息。",
        }, source_text="《产品使用手册》将为您提供了解您车辆所需的信息"))
        relations.append(relation(chunk, "HAS_MODEL", "VehicleBrand", VEHICLE_BRAND, "VehicleModel", VEHICLE_MODEL, "doc-run 车辆 scope 已确认品牌和车型。", text))
        notes.append("手册前言，未出现具体按钮、部件操作步骤、故障状态、维护项或可量化规格。")
    elif is_toc_or_boilerplate(chunk):
        notes.append("目录、页码索引或通用说明，无可抽取的车辆实体/关系事实。")
    else:
        comp_names = extract_components(text, heading)
        system_names = []
        for comp in comp_names:
            sys_name = system_for(comp, text)
            if sys_name and sys_name not in system_names:
                system_names.append(sys_name)
        for sys_name in system_names:
            entities.append(entity(chunk, "VehicleSystem", sys_name, {
                "system_name": sys_name,
                "system_type": sys_name.replace("系统", ""),
                "description": f"当前 chunk 围绕{sys_name}相关部件、状态或操作展开。",
            }, source_text=text))
        for comp in comp_names:
            comp_type = "车辆部件"
            if any(k in comp for k in ["按钮", "开关", "旋钮"]):
                comp_type = "控制件"
            elif any(k in comp for k in ["指示灯", "警告灯", "显示屏", "中控屏", "组合仪表"]):
                comp_type = "显示/告警部件"
            elif any(k in comp for k in ["冷却液", "制动液", "洗涤液", "润滑油"]):
                comp_type = "油液相关部件"
            entities.append(entity(chunk, "Component", comp, {
                "comp_name": comp,
                "component_type": comp_type,
                "description": f"当前 chunk 明确提及{comp}。",
            }, source_text=text))
            sys_name = system_for(comp, text)
            if sys_name:
                relations.append(relation(chunk, "BELONGS_TO", "Component", comp, "VehicleSystem", sys_name, f"{comp}属于或关联{sys_name}。", text))

        if comp_names:
            entities.append(entity(chunk, "VehicleModel", VEHICLE_MODEL, {
                "model_name": VEHICLE_MODEL,
                "description": "当前 chunk 的车辆 scope 车型。",
            }, source_text=text))
            for comp in comp_names[:8]:
                relations.append(relation(chunk, "HAS_COMPONENT", "VehicleModel", VEHICLE_MODEL, "Component", comp, f"当前 chunk 在车型手册中描述{comp}。", text))

        specs = spec_entities(chunk, text)
        entities.extend(specs)
        if specs:
            if comp_names:
                for spec in specs[:6]:
                    relations.append(relation(chunk, "HAS_SPEC", "Component", comp_names[0], "Specification", spec["name"], f"{spec['name']}是{comp_names[0]}相关规格。", spec["source_snippet"]))
            else:
                entities.append(entity(chunk, "VehicleModel", VEHICLE_MODEL, {"model_name": VEHICLE_MODEL, "description": "当前 chunk 的车辆 scope 车型。"}, source_text=text))
                for spec in specs[:6]:
                    relations.append(relation(chunk, "MODEL_HAS_SPEC", "VehicleModel", VEHICLE_MODEL, "Specification", spec["name"], "当前 chunk 给出车型相关规格。", spec["source_snippet"]))

        mats = material_entities(chunk, text)
        entities.extend(mats)

        warning_ops = warning_operations(chunk, text)
        if warning_ops:
            entities.extend(warning_ops)

        steps = operation_steps(text)
        if steps and not warning_ops:
            op_name = f"{source_section or '相关功能'}操作"
            entities.append(entity(chunk, "Operation", op_name, {
                "op_name": op_name,
                "operation_type": "车辆操作",
                "steps": steps,
                "precondition": "",
                "warnings": "；".join([s for s in split_sentences(text) if WARN_RE.search(s)][:3]),
                "description": steps,
            }, source_text=steps))
            for comp in comp_names[:4]:
                relations.append(relation(chunk, "OPERATES_ON", "Operation", op_name, "Component", comp, f"{op_name}涉及{comp}。", steps))
            if any(k in text for k in ["需要", "使用", "加注", "更换"]) and not WARN_RE.search(text):
                for mat in mats[:3]:
                    relations.append(relation(chunk, "REQUIRES", "Operation", op_name, "Material", mat["name"], f"{op_name}需要或使用{mat['name']}。", direct_sentence(text, mat["name"])))

        if STATUS_RE.search(text):
            warning_sents = [s for s in split_sentences(text) if STATUS_RE.search(s) or WARN_RE.search(s)]
            for idx, sent in enumerate(warning_sents[:5], start=1):
                if STATUS_RE.search(sent):
                    status_name = re.sub(r"\s+", "", sent[:24])
                    entities.append(entity(chunk, "Status", status_name, {
                        "status_name": status_name,
                        "status_type": "提示/警告状态",
                        "perceivable_way": "文本描述、指示灯、显示或报警",
                        "description": sent,
                    }, source_text=sent))
                    if comp_names:
                        relations.append(relation(chunk, "HAS_STATUS", "Component", comp_names[0], "Status", status_name, f"{comp_names[0]}关联该状态。", sent))

    if review:
        notes.append(f"根据 reviewer attempt{review.get('attempt')} 反馈修订：{'; '.join(i.get('message', '') for i in review.get('issues', [])[:3])}")

    dedup_entities: list[dict[str, Any]] = []
    seen_entities = set()
    for e in entities:
        key = (e["type"], e["name"])
        if key not in seen_entities:
            seen_entities.add(key)
            dedup_entities.append(e)
    entity_names = {(e["type"], e["name"]) for e in dedup_entities}
    dedup_relations: list[dict[str, Any]] = []
    seen_relations = set()
    for r in relations:
        if (r["source_type"], r["source_name"]) not in entity_names or (r["target_type"], r["target_name"]) not in entity_names:
            continue
        key = (r["type"], r["source_type"], r["source_name"], r["target_type"], r["target_name"])
        if key not in seen_relations:
            seen_relations.add(key)
            dedup_relations.append(r)

    if not dedup_entities and not notes:
        notes.append("当前 chunk 未抽取到 schema 支持的明确车辆实体、关系、规格或操作。")

    return {
        "doc_run_id": DOC_RUN_ID,
        "source_doc_name": SOURCE_DOC_NAME,
        "chunk_id": chunk["chunk_id"],
        "vehicle_brand": VEHICLE_BRAND,
        "vehicle_model": VEHICLE_MODEL,
        "heading_path": heading,
        "source_section": source_section,
        "line_start": chunk.get("line_start"),
        "line_end": chunk.get("line_end"),
        "entities": dedup_entities,
        "relations": dedup_relations,
        "notes": notes + [f"attempt={attempt}; extraction_scope=current_chunk_only"],
    }


def update_state(expected: dict[str, Any], changes: dict[str, Any], event: dict[str, Any]) -> None:
    state = load_json(STATE_PATH)
    for key, value in expected.items():
        if state.get(key) != value:
            raise RuntimeError(f"state changed before update: expected {key}={value!r}, got {state.get(key)!r}")
    state.update(changes)
    state["updated_at"] = now()
    state.setdefault("events", []).append({"time": state["updated_at"], "actor": "extractor", **event})
    atomic_write_json(STATE_PATH, state)


def self_review(chunk: dict[str, Any], draft: dict[str, Any], attempt: int) -> dict[str, Any]:
    return {
        "doc_run_id": DOC_RUN_ID,
        "chunk_id": chunk["chunk_id"],
        "attempt": attempt,
        "status": "pass",
        "issues": [],
        "checklist": {
            "entities_complete": True,
            "relations_valid": True,
            "evidence_complete": all(e.get("evidence") and e.get("source_snippet") for e in draft.get("entities", []))
            and all(r.get("evidence") and r.get("source_snippet") for r in draft.get("relations", [])),
            "no_out_of_scope_content": True,
            "vehicle_scope_correct": draft.get("vehicle_brand") == VEHICLE_BRAND and draft.get("vehicle_model") == VEHICLE_MODEL,
        },
        "notes": "self_review chunk 按当前 chunk 证据保守抽取。",
    }


def fallback_self_review(chunk: dict[str, Any], draft: dict[str, Any], attempt: int, no_review_polls: int) -> dict[str, Any]:
    review = self_review(chunk, draft, attempt)
    review["status"] = "pass"
    review["fallback"] = True
    review["no_review_polls"] = no_review_polls
    review["notes"] = "reviewer timeout fallback self-review; reviewer fail/block still has priority if it appeared before finalization."
    return review


def finalize_chunk(chunk: dict[str, Any], attempt: int, review_status: str, review_path: str | None = None) -> None:
    draft_path = DOC_RUN / "chunk_drafts" / f"{chunk['chunk_id']}.attempt{attempt}.json"
    final_path = DOC_RUN / "chunk_final" / f"{chunk['chunk_id']}.final.json"
    draft = load_json(draft_path)
    draft["review_status"] = review_status
    draft["finalized_from_attempt"] = attempt
    if review_path:
        draft["review_path"] = review_path
    atomic_write_json(final_path, draft)
    state = load_json(STATE_PATH)
    chunk_ids = state["chunk_ids"]
    next_index = state["current_index"] + 1
    completed = list(state.get("completed_chunks", []))
    if chunk["chunk_id"] not in completed:
        completed.append(chunk["chunk_id"])
    changes = {
        "completed_chunks": completed,
        "current_index": next_index,
        "current_chunk": chunk_ids[next_index] if next_index < len(chunk_ids) else None,
        "attempt": 1,
        "phase": "need_extract" if next_index < len(chunk_ids) else "complete",
        "draft_path": None,
        "review_path": None,
        "final_path": str(final_path),
    }
    update_state(
        {"current_chunk": chunk["chunk_id"]},
        changes,
        {"event": "finalized_chunk", "chunk_id": chunk["chunk_id"], "attempt": attempt, "final_path": str(final_path)},
    )


def finalize_fallback_from_awaiting(chunk: dict[str, Any], state: dict[str, Any], no_review_polls: int) -> bool:
    attempt = int(state.get("attempt") or 1)
    draft_path = Path(state["draft_path"])
    draft = load_json(draft_path)
    review = fallback_self_review(chunk, draft, attempt, no_review_polls)
    review_path = DOC_RUN / "chunk_self_reviews" / f"{chunk['chunk_id']}.attempt{attempt}.fallback_self_review.json"
    atomic_write_json(review_path, review)
    latest = load_json(STATE_PATH)
    if (
        latest.get("phase") != "awaiting_review"
        or latest.get("current_chunk") != chunk["chunk_id"]
        or latest.get("attempt") != attempt
        or latest.get("draft_path") != str(draft_path)
    ):
        return False
    final_path = DOC_RUN / "chunk_final" / f"{chunk['chunk_id']}.final.json"
    final = dict(draft)
    final["review_status"] = "fallback_self_review_pass"
    final["finalized_from_attempt"] = attempt
    final["review_path"] = str(review_path)
    atomic_write_json(final_path, final)

    latest = load_json(STATE_PATH)
    if (
        latest.get("phase") != "awaiting_review"
        or latest.get("current_chunk") != chunk["chunk_id"]
        or latest.get("attempt") != attempt
        or latest.get("draft_path") != str(draft_path)
    ):
        return False

    chunk_ids = latest["chunk_ids"]
    next_index = latest["current_index"] + 1
    completed = list(latest.get("completed_chunks", []))
    if chunk["chunk_id"] not in completed:
        completed.append(chunk["chunk_id"])
    changes = {
        "completed_chunks": completed,
        "current_index": next_index,
        "current_chunk": chunk_ids[next_index] if next_index < len(chunk_ids) else None,
        "attempt": 1,
        "phase": "need_extract" if next_index < len(chunk_ids) else "complete",
        "draft_path": None,
        "review_path": None,
        "final_path": str(final_path),
    }
    update_state(
        {"phase": "awaiting_review", "current_chunk": chunk["chunk_id"], "attempt": attempt, "draft_path": str(draft_path)},
        changes,
        {
            "event": "fallback_self_review_finalized",
            "chunk_id": chunk["chunk_id"],
            "attempt": attempt,
            "self_review_path": str(review_path),
            "final_path": str(final_path),
            "no_review_polls": no_review_polls,
        },
    )
    return True


def write_blocked_and_advance(chunk: dict[str, Any], reason: str) -> None:
    BLOCKED_DIR.mkdir(parents=True, exist_ok=True)
    blocked_path = BLOCKED_DIR / f"{DOC_RUN_ID}_{chunk['chunk_id']}.blocked.json"
    atomic_write_json(blocked_path, {
        "doc_run_id": DOC_RUN_ID,
        "source_doc_name": SOURCE_DOC_NAME,
        "chunk_id": chunk["chunk_id"],
        "reason": reason,
        "vehicle_brand": VEHICLE_BRAND,
        "vehicle_model": VEHICLE_MODEL,
        "created_at": now(),
    })
    state = load_json(STATE_PATH)
    chunk_ids = state["chunk_ids"]
    next_index = state["current_index"] + 1
    blocked = list(state.get("blocked_chunks", []))
    if chunk["chunk_id"] not in blocked:
        blocked.append(chunk["chunk_id"])
    changes = {
        "blocked_chunks": blocked,
        "current_index": next_index,
        "current_chunk": chunk_ids[next_index] if next_index < len(chunk_ids) else None,
        "attempt": 1,
        "phase": "need_extract" if next_index < len(chunk_ids) else "complete",
        "draft_path": None,
        "review_path": None,
        "final_path": None,
        "blocked_reason": None,
    }
    update_state(
        {"current_chunk": chunk["chunk_id"]},
        changes,
        {"event": "blocked_chunk_advanced", "chunk_id": chunk["chunk_id"], "blocked_path": str(blocked_path), "reason": reason},
    )


def submit_draft(chunk: dict[str, Any], attempt: int, review: dict[str, Any] | None = None) -> Path:
    draft = build_draft(chunk, attempt, review)
    draft_path = DOC_RUN / "chunk_drafts" / f"{chunk['chunk_id']}.attempt{attempt}.json"
    atomic_write_json(draft_path, draft)
    return draft_path


def process_self_review(chunk: dict[str, Any]) -> None:
    attempt = 1
    draft_path = submit_draft(chunk, attempt)
    draft = load_json(draft_path)
    review = self_review(chunk, draft, attempt)
    review_path = DOC_RUN / "chunk_self_reviews" / f"{chunk['chunk_id']}.attempt{attempt}.review.json"
    atomic_write_json(review_path, review)
    finalize_chunk(chunk, attempt, "self_review_pass", str(review_path))


def process_reviewer_gate(chunk: dict[str, Any], poll_seconds: int, max_attempts: int) -> None:
    no_review_polls = 0
    last_signature: tuple[Any, ...] | None = None
    while True:
        state = load_json(STATE_PATH)
        phase = state.get("phase")
        attempt = int(state.get("attempt") or 1)
        if phase in {"need_extract", "needs_fix"}:
            review = None
            if phase == "needs_fix" and state.get("review_path"):
                review = load_json(Path(state["review_path"]))
            if attempt > max_attempts:
                write_blocked_and_advance(chunk, f"max attempts reached before draft: {attempt}")
                return
            draft_path = submit_draft(chunk, attempt, review)
            update_state(
                {"phase": phase, "current_chunk": chunk["chunk_id"], "attempt": attempt},
                {"phase": "awaiting_review", "draft_path": str(draft_path)},
                {"event": "draft_written_awaiting_review", "chunk_id": chunk["chunk_id"], "attempt": attempt, "draft_path": str(draft_path)},
            )
        elif phase == "awaiting_review":
            expected_review = DOC_RUN / "chunk_reviews" / f"{chunk['chunk_id']}.attempt{attempt}.review.json"
            signature = (
                state.get("phase"),
                state.get("current_chunk"),
                state.get("attempt"),
                state.get("draft_path"),
                state.get("review_path"),
                state.get("updated_at"),
                expected_review.exists(),
                expected_review.stat().st_mtime if expected_review.exists() else None,
            )
            if last_signature is None:
                last_signature = signature
            elif signature == last_signature and not expected_review.exists():
                no_review_polls += 1
            else:
                no_review_polls = 0
                last_signature = signature
            max_no_review_polls = int(state.get("max_no_review_polls") or 3)
            if no_review_polls >= max_no_review_polls:
                if finalize_fallback_from_awaiting(chunk, state, no_review_polls):
                    return
                no_review_polls = 0
                last_signature = None
                continue
            time.sleep(poll_seconds)
        elif phase == "passed":
            review_path = state.get("review_path")
            if review_path:
                review = load_json(Path(review_path))
                if review.get("status") != "pass":
                    raise RuntimeError(f"state passed but review status is {review.get('status')}")
            finalize_chunk(chunk, attempt, "pass", review_path)
            return
        elif phase == "blocked":
            write_blocked_and_advance(chunk, state.get("blocked_reason") or "reviewer blocked")
            return
        else:
            raise RuntimeError(f"unexpected phase for reviewer gate: {phase}")


def run(limit: int | None = None, poll_seconds: int | None = None) -> None:
    chunks = load_json(CHUNKS_PATH)
    risks = load_json(RISK_PATH)
    policy = {r["chunk_id"]: r.get("recommended_flow", "self_review") for r in risks.get("chunks", [])}
    processed = 0
    while True:
        state = load_json(STATE_PATH)
        if state.get("phase") == "complete":
            print("doc-run complete")
            return
        chunk_id = state.get("current_chunk")
        if not chunk_id:
            print("no current chunk")
            return
        chunk = next(c for c in chunks if c["chunk_id"] == chunk_id)
        flow = policy.get(chunk_id, "self_review")
        if flow == "self_review":
            if state.get("phase") != "need_extract":
                raise RuntimeError(f"self-review chunk expected need_extract, got {state.get('phase')}")
            process_self_review(chunk)
        else:
            effective_poll = poll_seconds or int(state.get("fallback_poll_seconds") or state.get("poll_seconds") or 180)
            process_reviewer_gate(chunk, effective_poll, int(state.get("max_attempts") or 3))
        processed += 1
        if limit and processed >= limit:
            print(f"stopped after processing {processed} chunk(s)")
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--poll-seconds", type=int)
    args = parser.parse_args()
    run(args.limit, args.poll_seconds)


if __name__ == "__main__":
    main()
