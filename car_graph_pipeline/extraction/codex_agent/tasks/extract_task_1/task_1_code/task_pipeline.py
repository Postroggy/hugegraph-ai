#!/usr/bin/env python3
"""Task-local extraction pipeline utilities.

This script is intentionally self-contained so each extract_task can run in
isolation. It prepares semantic chunks, classifies document-level risk, creates
doc-run state files, merges chunk finals, validates raw outputs, and summarizes
progress. It does not call LLMs and does not write HugeGraph.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENTITY_NAME_PROPS = {
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

VALID_ENTITY_TYPES = set(ENTITY_NAME_PROPS)

RELATION_DIRECTION = {
    "HAS_MODEL": ("VehicleBrand", "VehicleModel"),
    "HAS_SYSTEM": ("VehicleModel", "VehicleSystem"),
    "HAS_COMPONENT": ("VehicleModel", "Component"),
    "HAS_FUNCTION": ("VehicleModel", "Function"),
    "BELONGS_TO": ("Component", "VehicleSystem"),
    "ACTIVATES": ("Component", "Function"),
    "OPERATED_BY": ("Function", "Operation"),
    "OPERATES_ON": ("Operation", "Component"),
    "HAS_STATUS": ("Component", "Status"),
    "SYSTEM_HAS_STATUS": ("VehicleSystem", "Status"),
    "CAUSED_BY": ("Status", "Fault"),
    "LEADS_TO": ("Fault", "Fault"),
    "AFFECTS": ("Fault", "VehicleSystem"),
    "RESOLVED_BY": ("Status", "Operation"),
    "FAULT_RESOLVED_BY": ("Fault", "Operation"),
    "HAS_SPEC": ("Component", "Specification"),
    "MODEL_HAS_SPEC": ("VehicleModel", "Specification"),
    "REQUIRES": ("Operation", "Material"),
    "APPLICABLE_TO": ("MaintenanceItem", "VehicleModel"),
    "MAINT_HAS_SPEC": ("MaintenanceItem", "Specification"),
    "MAINT_REQUIRES": ("MaintenanceItem", "Material"),
}

TARGET_CHARS = 760
MAX_CHARS = 1200
MIN_KEEP_CHARS = 60
NEIGHBOR_CHARS = 320

OPERATION_RE = re.compile(
    r"(步骤|操作|方法|开启|打开|关闭|启动|起动|按下|按住|长按|拉动|推动|转动|旋转|调节|选择|设置|"
    r"连接|配对|拆下|安装|更换|加注|检查|释放|踩下|挂入|切换|取出|插入|拧下|拧紧|上提|按压)"
)
WARNING_RE = re.compile(r"^(#*\s*)?(危险|警告|小心|注意|提示|重要|告诫)\b|(?:否则|切勿|不得|可能导致)")
SPEC_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:km/h|kPa|bar|MPa|L|mL|W|V|A|mm|cm|N·m|Nm|公里|千米|分钟|秒|年|个月|"
    r"公里/小时|升|毫升|℃|°C)|R\d{2}|W-\d+|0W|5W|API|ACEA|DOT|冷态|容量|规格|型号|压力|胎压)"
)
STATUS_RE = re.compile(r"(指示灯|警告灯|提示灯|报警|蜂鸣器|闪烁|点亮|显示|故障|失效|异常|警告信息)")
MENU_RE = re.compile(r"(\[OK\]|设置|菜单|显示屏|信息屏|中控屏|进入.+项目|选择.+项目|返回上一层)")
MAINT_RE = re.compile(r"(机油|冷却液|制动液|蓄电池|轮胎气压|保养|更换周期|油液|润滑剂|容量|胎压)")
SAFETY_RE = re.compile(r"(安全带|气囊|儿童座椅|ABS|ESP|VDC|AEB|IEB|制动|驻车|辅助驾驶)")

BRAND_HINTS = {
    "奥迪": ["奥迪", "Audi", "A5", "A6", "A8", "Q3", "Q5", "Q6"],
    "广汽埃安": ["AION", "埃安"],
    "奔驰": ["奔驰", "AMG", "Mercedes", "EQB", "EQE", "EQS", "GLB", "CLA", "CLS", "CLE", "A级", "E级", "v-class"],
    "日产": [
        "日产",
        "Nissan",
        "ARIYA",
        "Altima",
        "GT-R",
        "Lannia",
        "Murano",
        "NV200",
        "Note",
        "Quest",
        "Tiida",
        "X-Trail",
        "轩逸",
    ],
    "丰田": [
        "丰田",
        "Toyota",
        "Aygo",
        "C-HR",
        "RAV4",
        "YARiS",
        "HIACE",
        "MIRAI",
        "SUPRA",
        "普拉多",
        "普锐斯",
        "埃尔法",
        "亚洲狮",
    ],
    "林肯": ["林肯", "Lincoln", "Aviator", "Corsair", "MKC", "MKX", "MKZ", "Nautilus", "Navigator", "Zephyr"],
    "本田": ["本田", "Honda", "CR-V", "CIIMO", "LIFE", "VE-1"],
    "保时捷": ["保时捷", "Porsche", "Boxster", "Cayenne", "Cayman", "Macan", "Panamera", "Taycan"],
    "凯迪拉克": ["凯迪拉克", "Cadillac", "CT5", "XT6", "SLS", "IQ锐歌"],
    "马自达": ["马自达", "Mazda", "CX-3", "CX-4", "Mazda6", "Mazda8"],
    "哈弗": ["哈弗", "H2", "H4", "H6", "H7", "Dagou", "大狗", "神兽"],
    "比亚迪": ["比亚迪", "BYD", "F3", "e2", "e6"],
    "大众": ["大众", "Volkswagen", "CC", "ID.4", "ID.6", "Magotan", "Polo", "T-ROC", "迈腾"],
    "特斯拉": ["Tesla", "Model 3", "Model S", "Model X", "Model Y"],
    "上汽大通": ["上汽大通", "MAXUS", "大通"],
    "五菱": ["五菱", "宏光", "凯捷", "佳辰"],
    "别克": ["别克", "GL8", "世纪", "君威", "凯越"],
    "雪铁龙": ["雪铁龙", "Citroen", "世嘉", "AirCross", "C4L"],
}


def task_paths() -> dict[str, Path]:
    code_dir = Path(__file__).resolve().parent
    task_dir = code_dir.parent
    m = re.search(r"extract_task_(\d+)$", task_dir.name)
    if not m:
        raise RuntimeError(f"Cannot infer task number from {task_dir}")
    n = m.group(1)
    return {
        "task_dir": task_dir,
        "code_dir": code_dir,
        "doc_dir": task_dir / f"task_{n}_doc",
        "out_dir": task_dir / f"task_{n}_output",
        "task_num": Path(n),
    }


def load_manifest(out_dir: Path) -> list[dict[str, Any]]:
    return json.loads((out_dir / "task_manifest.json").read_text(encoding="utf-8"))


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def safe_name(name: str) -> str:
    name = re.sub(r"\.md$", "", name)
    return re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", name)[:160]


def clean_text(text: str) -> str:
    text = re.sub(r"!\[.*?\]\([^)]*\)", "", text)
    text = re.sub(r"\$\\bullet\$", "-", text)
    text = re.sub(r"\$\\triangleright\$", "▶", text)
    text = re.sub(r"\$\\Leftrightarrow\$", "⟺", text)
    text = re.sub(r"\$[^$]{1,30}\$", "", text)
    text = re.sub(r"www\.carobook\.com", "", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def heading_level(line: str) -> int | None:
    m = re.match(r"^(#{1,4})\s+(.+?)\s*$", line)
    return len(m.group(1)) if m else None


def heading_title(line: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", line).strip()


def is_table_line(line: str) -> bool:
    return ("|" in line and len(line.split("|")) >= 3) or "<table" in line or "<td" in line


def is_list_line(line: str) -> bool:
    return bool(re.match(r"^\s*(?:[-*+•▶]|\d+[.)、]|[（(]?\d+[）)]|[a-zA-Z][.)])\s+", line))


def is_toc_like(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    td_numbers = len(re.findall(r"<td>\s*\d{1,3}\s*</td>", text))
    td_page_refs = len(re.findall(r"第\s*\d+\s*页", text))
    if "<table" in text and re.search(r"(目录|概览|检索|索引)", text[:300]) and td_numbers + td_page_refs >= 6:
        return True
    if sum(1 for line in lines if line.startswith("# ")) > 8:
        return True
    page_refs = sum(1 for line in lines if re.search(r"(第\s*\d+\s*页|\s+\d{1,3}$)", line))
    short_refs = sum(1 for line in lines if len(line) <= 28 and re.search(r"(\.{2,}|·+|\s)\d{1,3}$", line))
    if len(lines) >= 6 and (page_refs + short_refs) / max(1, len(lines)) > 0.35:
        return True
    return len(lines) > 20 and page_refs / max(1, len(lines)) > 0.55


def block_type(text: str, heading_path: str, has_table: bool, has_list: bool) -> str:
    head = heading_path + "\n" + text[:160]
    if WARNING_RE.search(text.strip()) or WARNING_RE.search(heading_path):
        return "warning"
    if has_table:
        return "spec_table"
    if OPERATION_RE.search(head) or MENU_RE.search(head) or has_list:
        return "operation"
    if SPEC_RE.search(text) or MAINT_RE.search(text):
        return "spec"
    return "normal"


def split_blocks(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    headings: dict[int, str] = {}
    blocks: list[dict[str, Any]] = []
    buffer: list[str] = []
    start_line = 1
    has_table = False
    has_list = False

    def heading_path() -> str:
        return " > ".join(headings[i] for i in sorted(headings) if headings[i])

    def flush(end_line: int) -> None:
        nonlocal buffer, start_line, has_table, has_list
        raw = "\n".join(buffer).strip()
        if not raw:
            buffer = []
            has_table = False
            has_list = False
            return
        non_heading = [line for line in raw.splitlines() if line.strip() and heading_level(line.strip()) is None]
        if not non_heading:
            buffer = []
            has_table = False
            has_list = False
            return
        hp = heading_path()
        btype = block_type(raw, hp, has_table, has_list)
        keep = len(raw) >= MIN_KEEP_CHARS or btype in {"operation", "warning", "spec_table", "spec"}
        if keep and not is_toc_like(raw):
            blocks.append(
                {
                    "text": raw,
                    "heading_path": hp,
                    "source_section": hp.split(" > ")[-1] if hp else "",
                    "line_start": start_line,
                    "line_end": end_line,
                    "block_type": btype,
                }
            )
        buffer = []
        has_table = False
        has_list = False

    for i, line in enumerate(lines, start=1):
        level = heading_level(line)
        if level is not None:
            flush(i - 1)
            for old in list(headings):
                if old >= level:
                    del headings[old]
            headings[level] = heading_title(line)
            buffer = [line]
            start_line = i
            continue
        blank = not line.strip()
        table = is_table_line(line)
        listed = is_list_line(line)
        if blank:
            if buffer and not has_table and not has_list:
                flush(i - 1)
            elif buffer:
                buffer.append(line)
            continue
        if buffer and ((table and not has_table) or (listed and not has_list and len("\n".join(buffer)) > 200)):
            flush(i - 1)
            start_line = i
        if not buffer:
            start_line = i
        buffer.append(line)
        has_table = has_table or table
        has_list = has_list or listed
    flush(len(lines))
    return blocks


def split_large_block(block: dict[str, Any]) -> list[dict[str, Any]]:
    text = block["text"]
    if len(text) <= MAX_CHARS:
        return [block]
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        out: list[dict[str, Any]] = []
        buf: list[str] = []
        for line in lines:
            candidate = "\n".join([*buf, line])
            if buf and len(candidate) > MAX_CHARS:
                item = dict(block)
                item["text"] = "\n".join(buf).strip()
                out.append(item)
                buf = [line]
            else:
                buf.append(line)
        if buf:
            item = dict(block)
            item["text"] = "\n".join(buf).strip()
            out.append(item)
        return out
    parts = re.split(r"(?<=。|；|;|！|!|？|\?)\s*", text)
    out = []
    buf = ""
    for part in parts:
        if not part:
            continue
        if buf and len(buf) + len(part) > MAX_CHARS:
            item = dict(block)
            item["text"] = buf.strip()
            out.append(item)
            buf = part
        else:
            buf = (buf + part) if buf else part
    if buf.strip():
        item = dict(block)
        item["text"] = buf.strip()
        out.append(item)
    return out


def build_chunks(doc_path: Path, doc_run_id: str, scope: dict[str, str]) -> list[dict[str, Any]]:
    text = clean_text(doc_path.read_text(encoding="utf-8"))
    blocks: list[dict[str, Any]] = []
    for block in split_blocks(text):
        blocks.extend(split_large_block(block))

    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        joined = "\n\n".join(b["text"] for b in current).strip()
        if not joined:
            current = []
            return
        types = Counter(b["block_type"] for b in current)
        heading_values = []
        for b in current:
            hp = b["heading_path"]
            if hp and hp not in heading_values:
                heading_values.append(hp)
        heading = " || ".join(heading_values[-4:])
        chunk = {
            "chunk_id": f"{safe_name(doc_run_id)}_s{len(chunks):04d}",
            "doc_name": doc_path.name,
            "doc_run_id": doc_run_id,
            "vehicle_brand": scope.get("vehicle_brand", ""),
            "vehicle_model": scope.get("vehicle_model", ""),
            "heading_path": heading,
            "source_section": heading.split(" > ")[-1] if heading else "",
            "chunk_type": types.most_common(1)[0][0],
            "line_start": min(b["line_start"] for b in current),
            "line_end": max(b["line_end"] for b in current),
            "text": joined,
            "char_count": len(joined),
            "block_count": len(current),
            "blocks": [
                {
                    "heading_path": b["heading_path"],
                    "line_start": b["line_start"],
                    "line_end": b["line_end"],
                    "block_type": b["block_type"],
                    "char_count": len(b["text"]),
                }
                for b in current
            ],
        }
        if not is_toc_like(joined):
            chunks.append(chunk)
        current = []

    for block in blocks:
        text_len = len(block["text"])
        high_value = block["block_type"] in {"operation", "warning", "spec_table", "spec"}
        current_len = len("\n\n".join(b["text"] for b in current))
        current_types = {b["block_type"] for b in current}
        compatible = (
            not current
            or block["block_type"] in current_types
            or (high_value and current_types <= {"operation", "warning", "spec_table", "spec"})
        )
        if current and (current_len + text_len > TARGET_CHARS or not compatible):
            flush()
        current.append(block)
        current_len = len("\n\n".join(b["text"] for b in current))
        if current_len >= TARGET_CHARS or (high_value and current_len >= MAX_CHARS):
            flush()
    flush()

    for i, chunk in enumerate(chunks):
        chunk["context_before"] = chunks[i - 1]["text"][-NEIGHBOR_CHARS:] if i > 0 else ""
        chunk["context_after"] = chunks[i + 1]["text"][:NEIGHBOR_CHARS] if i + 1 < len(chunks) else ""
    return chunks


def normalize_vehicle_model_name(doc_name: str) -> str:
    stem = re.sub(r"[\u200b\ufeff]", "", doc_name).strip()
    stem = re.sub(r"(?:\.(?:md|pdf))+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\s+copy$", "", stem, flags=re.IGNORECASE).strip()
    stem = re.sub(r"^\d{4}款", "", stem).strip(" -_")
    stem = re.sub(
        r"(?:使用说明书|用户手册|说明书|操作手册)\s*(?:copy)?\s*(?:\d{4}款?)?$",
        "",
        stem,
        flags=re.IGNORECASE,
    ).strip(" -_")
    if "-" in stem:
        left, right = stem.split("-", 1)
        if re.search(r"[\u4e00-\u9fff]", left) and re.search(r"[A-Za-z]", right):
            stem = left.strip(" -_")
    return stem or re.sub(r"(?:\.(?:md|pdf))+$", "", doc_name, flags=re.IGNORECASE)


def infer_scope(doc_name: str) -> dict[str, str]:
    stem = normalize_vehicle_model_name(doc_name)
    brand = ""
    for candidate, hints in BRAND_HINTS.items():
        if any(hint and hint in doc_name for hint in hints):
            brand = candidate
            break
    confidence = "high" if brand else "low"
    return {
        "vehicle_brand": brand or "UNKNOWN_BRAND",
        "vehicle_model": stem,
        "confidence": confidence,
        "source": "filename_brand_hints" if brand else "filename_model_only",
    }


def scope_needs_refresh(doc_name: str, scope: dict[str, Any]) -> bool:
    model = str(scope.get("vehicle_model") or "")
    if any(token in model for token in ("使用说明书", "用户手册", "说明书", "\u200b", "\ufeff")):
        return True
    if model.endswith(".") or re.search(r"\s+copy$", model, flags=re.IGNORECASE):
        return True
    return model != normalize_vehicle_model_name(doc_name) and scope.get("source", "").startswith("filename_")


def ensure_scope_map() -> dict[str, Any]:
    paths = task_paths()
    out_dir = paths["out_dir"]
    scope_path = out_dir / "vehicle_scope_map.json"
    review_path = out_dir / "scope_review_needed.json"
    manifest = load_manifest(out_dir)
    existing = json.loads(scope_path.read_text(encoding="utf-8")) if scope_path.exists() else {}
    review_needed = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else []
    review_by_doc = {
        item.get("doc_name") or item.get("source_doc_name"): item for item in review_needed if isinstance(item, dict)
    }
    changed = False
    for item in manifest:
        doc_name = item["source_doc_name"]
        if doc_name not in existing or scope_needs_refresh(doc_name, existing[doc_name]):
            scope = infer_scope(doc_name)
            existing[doc_name] = scope
            changed = True
        if existing[doc_name].get("confidence") != "high" and doc_name not in review_by_doc:
            review_by_doc[doc_name] = {
                "doc_name": doc_name,
                **existing[doc_name],
                "reason": "low confidence vehicle scope",
            }
    if changed:
        atomic_write_json(scope_path, existing)
    atomic_write_json(review_path, list(review_by_doc.values()))
    return existing


def risk_for_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    text = f"{chunk.get('heading_path', '')}\n{chunk.get('text', '')}"
    score = 0
    reasons: list[str] = []

    def add(points: int, reason: str) -> None:
        nonlocal score
        score += points
        reasons.append(reason)

    ctype = chunk.get("chunk_type")
    if ctype == "spec_table" or "<table" in text or "<td" in text:
        add(4, "table_or_spec_table")
    if ctype == "operation" or OPERATION_RE.search(text):
        add(3, "operation_words")
    if ctype == "warning" or WARNING_RE.search(text):
        add(3, "warning_words")
    if STATUS_RE.search(text):
        add(3, "status_fault_words")
    if SPEC_RE.search(text):
        add(3, "numeric_unit_or_spec")
    if MAINT_RE.search(text):
        add(3, "maintenance_material_words")
    if MENU_RE.search(text):
        add(3, "menu_button_display_words")
    if SAFETY_RE.search(text):
        add(2, "safety_system_words")
    if chunk.get("char_count", 0) > 900:
        add(1, "long_chunk")
    if is_toc_like(chunk.get("text", "")):
        score -= 5
        reasons.append("toc_like")

    if score >= 4:
        level = "high"
    elif score >= 2:
        level = "medium"
    else:
        level = "low"
    return {
        "chunk_id": chunk["chunk_id"],
        "risk_level": level,
        "risk_score": score,
        "reasons": reasons,
        "recommended_flow": "reviewer_gate" if level in {"high", "medium"} else "self_review",
    }


def doc_flow_from_risks(risks: list[dict[str, Any]]) -> str:
    high = sum(1 for r in risks if r["risk_level"] == "high")
    medium = sum(1 for r in risks if r["risk_level"] == "medium")
    # Keep one extractor context for the whole document. Risk controls whether
    # individual chunks must pass a blocking reviewer gate before proceeding.
    if high or medium:
        return "single_extractor_selective_reviewer"
    return "single_agent_self_review"


def prepare_chunks(args: argparse.Namespace) -> None:
    paths = task_paths()
    out_dir = paths["out_dir"]
    scope_map = ensure_scope_map()
    manifest = load_manifest(out_dir)
    selected = select_manifest(manifest, args.limit, args.doc_run_id)
    task_summary = []
    for item in selected:
        doc_path = Path(item["source_doc_path"])
        doc_run_dir = Path(item["doc_run_dir"])
        scope = scope_map.get(item["source_doc_name"], infer_scope(item["source_doc_name"]))
        chunks = build_chunks(doc_path, item["doc_run_id"], scope)
        chunks_path = doc_run_dir / "chunks" / "chunks.json"
        atomic_write_json(chunks_path, chunks)
        atomic_write_json(out_dir / "semantic_chunks" / f"{item['doc_run_id']}.chunks.json", chunks)
        task_summary.append({"doc_run_id": item["doc_run_id"], "chunks": len(chunks), "chunks_path": str(chunks_path)})
        logger.info("prepared %s: chunks=%s", item["doc_run_id"], len(chunks))
    atomic_write_json(out_dir / "analysis" / "last_prepare_chunks.json", task_summary)


def classify_risk(args: argparse.Namespace) -> None:
    paths = task_paths()
    out_dir = paths["out_dir"]
    manifest = load_manifest(out_dir)
    selected = select_manifest(manifest, args.limit, args.doc_run_id)
    docs = []
    for item in selected:
        doc_run_dir = Path(item["doc_run_dir"])
        chunks_path = doc_run_dir / "chunks" / "chunks.json"
        if not chunks_path.exists():
            logger.warning("missing chunks, skip: %s", item["doc_run_id"])
            continue
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        risks = [risk_for_chunk(c) for c in chunks]
        doc_flow = doc_flow_from_risks(risks)
        out = {
            "doc_run_id": item["doc_run_id"],
            "source_doc_name": item["source_doc_name"],
            "chunk_count": len(chunks),
            "risk_counts": dict(Counter(r["risk_level"] for r in risks)),
            "recommended_doc_flow": doc_flow,
            "chunks": risks,
        }
        atomic_write_json(doc_run_dir / "chunks" / "risk.json", out)
        atomic_write_json(out_dir / "risk_classification" / f"{item['doc_run_id']}.risk.json", out)
        docs.append(out)
        logger.info("risk %s: %s -> %s", item["doc_run_id"], out["risk_counts"], doc_flow)
    atomic_write_json(out_dir / "analysis" / "last_risk_classification.json", docs)


def init_doc_state(args: argparse.Namespace) -> None:
    paths = task_paths()
    out_dir = paths["out_dir"]
    scope_map = ensure_scope_map()
    manifest = load_manifest(out_dir)
    selected = select_manifest(manifest, args.limit, args.doc_run_id)
    for item in selected:
        doc_run_dir = Path(item["doc_run_dir"])
        chunks_path = doc_run_dir / "chunks" / "chunks.json"
        risk_path = doc_run_dir / "chunks" / "risk.json"
        if not chunks_path.exists():
            logger.warning("missing chunks, skip state: %s", item["doc_run_id"])
            continue
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        risk = (
            json.loads(risk_path.read_text(encoding="utf-8"))
            if risk_path.exists()
            else {"recommended_doc_flow": "single_agent_self_review"}
        )
        scope = scope_map.get(item["source_doc_name"], infer_scope(item["source_doc_name"]))
        state = {
            "doc_run_id": item["doc_run_id"],
            "source_doc_name": item["source_doc_name"],
            "source_doc_path": item["source_doc_path"],
            **scope,
            "chunks_path": str(chunks_path),
            "risk_path": str(risk_path),
            "chunk_ids": [c["chunk_id"] for c in chunks],
            "recommended_doc_flow": risk.get("recommended_doc_flow", "single_agent_self_review"),
            "chunk_review_policy": {
                r["chunk_id"]: r.get("recommended_flow", "self_review") for r in risk.get("chunks", [])
            },
            "current_index": 0,
            "current_chunk": chunks[0]["chunk_id"] if chunks else None,
            "attempt": 1,
            "phase": "need_extract" if chunks else "complete",
            "draft_path": None,
            "review_path": None,
            "final_path": None,
            "max_attempts": 3,
            "poll_seconds": 180,
            "review_timeout_polls": 3,
            "review_timeout_fallback": "extractor_self_review",
            "orchestrator_check_minutes": 15,
            "completed_chunks": [],
            "blocked_chunks": [],
            "events": [],
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        atomic_write_json(doc_run_dir / "state" / "state.json", state)
        logger.info("state %s: %s chunks=%s", item["doc_run_id"], state["recommended_doc_flow"], len(chunks))


def normalize_entity(raw: dict[str, Any], scope: dict[str, str], fallback: dict[str, Any]) -> dict[str, Any] | None:
    entity = dict(raw)
    if "type" not in entity and entity.get("label"):
        entity["type"] = entity["label"]
    if "type" not in entity and entity.get("entity_type"):
        entity["type"] = entity["entity_type"]
    etype = entity.get("type")
    props = entity.get("properties") if isinstance(entity.get("properties"), dict) else {}
    for key, value in list(entity.items()):
        if key not in {"type", "label", "entity_type", "name", "aliases", "properties", "entity_id"} and value not in (
            None,
            "",
            [],
            {},
        ):
            props.setdefault(key, value)
    name_prop = ENTITY_NAME_PROPS.get(etype)
    if not entity.get("name") and name_prop and props.get(name_prop):
        entity["name"] = props[name_prop]
    if not entity.get("name") and entity.get("entity_id"):
        entity["name"] = str(entity["entity_id"])
    if etype not in VALID_ENTITY_TYPES or not entity.get("name"):
        return None
    props["vehicle_brand"] = scope["vehicle_brand"]
    props["vehicle_model"] = scope["vehicle_model"]
    if name_prop:
        props.setdefault(name_prop, entity["name"])
    entity["properties"] = props
    for key in ("chunk_id", "heading_path", "source_section", "line_start", "line_end", "source_snippet", "evidence"):
        entity.setdefault(key, fallback.get(key))
    return entity


def normalize_relation(
    raw: dict[str, Any], scope: dict[str, str], fallback: dict[str, Any], entity_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    rel = dict(raw)
    if "type" not in rel and rel.get("label"):
        rel["type"] = rel["label"]
    if "type" not in rel and rel.get("relation_type"):
        rel["type"] = rel["relation_type"]
    props = rel.get("properties") if isinstance(rel.get("properties"), dict) else {}
    for key in ("strength", "confidence", "note", "description"):
        if rel.get(key) not in (None, "", [], {}):
            props.setdefault(key, rel[key])
    props["vehicle_brand"] = scope["vehicle_brand"]
    props["vehicle_model"] = scope["vehicle_model"]
    rel["properties"] = props

    src_id = rel.get("source_entity_id") or rel.get("source")
    tgt_id = rel.get("target_entity_id") or rel.get("target")
    if src_id and (not rel.get("source_type") or not rel.get("source_name")):
        src = entity_by_id.get(str(src_id))
        if src:
            rel.setdefault("source_type", src.get("type"))
            rel.setdefault("source_name", src.get("name"))
            rel.setdefault("source_entity_id", str(src_id))
    if tgt_id and (not rel.get("target_type") or not rel.get("target_name")):
        tgt = entity_by_id.get(str(tgt_id))
        if tgt:
            rel.setdefault("target_type", tgt.get("type"))
            rel.setdefault("target_name", tgt.get("name"))
            rel.setdefault("target_entity_id", str(tgt_id))
    for key in ("chunk_id", "heading_path", "source_section", "line_start", "line_end", "source_snippet", "evidence"):
        rel.setdefault(key, fallback.get(key))
    if rel.get("type") not in RELATION_DIRECTION:
        return None
    return rel


def merge_doc_raw(args: argparse.Namespace) -> None:
    paths = task_paths()
    out_dir = paths["out_dir"]
    scope_map = ensure_scope_map()
    manifest = load_manifest(out_dir)
    selected = select_manifest(manifest, args.limit, args.doc_run_id)
    summaries = []
    for item in selected:
        doc_run_dir = Path(item["doc_run_dir"])
        scope = scope_map.get(item["source_doc_name"], infer_scope(item["source_doc_name"]))
        finals = sorted((doc_run_dir / "chunk_final").glob("*.final.json"))
        chunk_results = []
        entities = []
        relations = []
        errors = []
        for fp in finals:
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"bad final json {fp}: {exc}")
                continue
            chunk_results.append(data)
            fallback = {
                k: data.get(k)
                for k in (
                    "chunk_id",
                    "heading_path",
                    "source_section",
                    "line_start",
                    "line_end",
                    "source_snippet",
                    "evidence",
                )
            }
            entity_by_id = {}
            chunk_entities = []
            for raw_e in data.get("entities", []) or []:
                ent = normalize_entity(raw_e, scope, fallback)
                if not ent:
                    errors.append(
                        f"{fp.name}: invalid entity {raw_e.get('type') or raw_e.get('label')} {raw_e.get('name')}"
                    )
                    continue
                chunk_entities.append(ent)
                if ent.get("entity_id"):
                    entity_by_id[str(ent["entity_id"])] = ent
            entities.extend(chunk_entities)
            for raw_r in data.get("relations", []) or []:
                rel = normalize_relation(raw_r, scope, fallback, entity_by_id)
                if not rel:
                    errors.append(f"{fp.name}: invalid relation {raw_r.get('type') or raw_r.get('label')}")
                    continue
                relations.append(rel)
        raw = {
            "doc_run_id": item["doc_run_id"],
            "source_doc_name": item["source_doc_name"],
            **scope,
            "chunk_results": chunk_results,
            "entities": entities,
            "relations": relations,
            "merge_errors": errors,
        }
        local_path = doc_run_dir / "doc_raw" / f"{item['doc_run_id']}.raw.json"
        task_path = out_dir / "doc_raw_results" / f"{item['doc_run_id']}.raw.json"
        atomic_write_json(local_path, raw)
        atomic_write_json(task_path, raw)
        summaries.append(
            {
                "doc_run_id": item["doc_run_id"],
                "final_chunks": len(finals),
                "entities": len(entities),
                "relations": len(relations),
                "errors": errors,
            }
        )
        logger.info(
            "merged %s: finals=%s entities=%s relations=%s errors=%s",
            item["doc_run_id"],
            len(finals),
            len(entities),
            len(relations),
            len(errors),
        )
    atomic_write_json(out_dir / "analysis" / "last_merge_doc_raw.json", summaries)


def validate_doc_raw(args: argparse.Namespace) -> None:
    paths = task_paths()
    out_dir = paths["out_dir"]
    manifest = load_manifest(out_dir)
    selected = select_manifest(manifest, args.limit, args.doc_run_id)
    summaries = []
    for item in selected:
        raw_path = out_dir / "doc_raw_results" / f"{item['doc_run_id']}.raw.json"
        if not raw_path.exists():
            summaries.append({"doc_run_id": item["doc_run_id"], "status": "missing_raw", "errors": [str(raw_path)]})
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        errors = list(raw.get("merge_errors", []) or [])
        entities = raw.get("entities", []) or []
        relations = raw.get("relations", []) or []
        names = {(e.get("type"), norm(e.get("name"))) for e in entities}
        for ent in entities:
            if ent.get("type") not in VALID_ENTITY_TYPES:
                errors.append(f"invalid entity type: {ent.get('type')} {ent.get('name')}")
            props = ent.get("properties") if isinstance(ent.get("properties"), dict) else {}
            if not props.get("vehicle_brand") or not props.get("vehicle_model"):
                errors.append(f"missing vehicle scope entity: {ent.get('type')} {ent.get('name')}")
        for rel in relations:
            rtype = rel.get("type")
            expected = RELATION_DIRECTION.get(rtype)
            if not expected:
                errors.append(f"invalid relation type: {rtype}")
                continue
            if (rel.get("source_type"), rel.get("target_type")) != expected:
                errors.append(f"wrong relation direction: {rtype} {rel.get('source_type')}->{rel.get('target_type')}")
            if (rel.get("source_type"), norm(rel.get("source_name"))) not in names:
                errors.append(f"missing source endpoint: {rtype} {rel.get('source_type')} {rel.get('source_name')}")
            if (rel.get("target_type"), norm(rel.get("target_name"))) not in names:
                errors.append(f"missing target endpoint: {rtype} {rel.get('target_type')} {rel.get('target_name')}")
            props = rel.get("properties") if isinstance(rel.get("properties"), dict) else {}
            if not props.get("vehicle_brand") or not props.get("vehicle_model"):
                errors.append(f"missing vehicle scope relation: {rtype}")
        status = "ok" if not errors else "partial"
        validated = {**raw, "validation_status": status, "validation_errors": errors}
        atomic_write_json(out_dir / "validated_results" / f"{item['doc_run_id']}.validated.json", validated)
        summaries.append(
            {
                "doc_run_id": item["doc_run_id"],
                "status": status,
                "entity_count": len(entities),
                "relation_count": len(relations),
                "error_count": len(errors),
                "errors": errors[:50],
            }
        )
        logger.info("validated %s: %s errors=%s", item["doc_run_id"], status, len(errors))
    atomic_write_json(out_dir / "analysis" / "validation_summary.json", summaries)


def norm(text: Any) -> str:
    s = str(text or "").lower()
    s = re.sub(r"\s+", "", s)
    return re.sub(r"[·•:：;；,，。.!！?？/\\|`'\"“”‘’（）()\[\]{}<>《》_-]+", "", s)


def summarize_progress(_args: argparse.Namespace) -> None:
    paths = task_paths()
    out_dir = paths["out_dir"]
    manifest = load_manifest(out_dir)
    total_docs = len(manifest)
    completed = 0
    blocked_docs = 0
    total_chunks = 0
    completed_chunks = 0
    blocked_chunks = 0
    rows = []
    for item in manifest:
        doc_run_dir = Path(item["doc_run_dir"])
        chunks_path = doc_run_dir / "chunks" / "chunks.json"
        chunks = json.loads(chunks_path.read_text(encoding="utf-8")) if chunks_path.exists() else []
        finals = list((doc_run_dir / "chunk_final").glob("*.final.json"))
        blocked = list((out_dir / "blocked").glob(f"{item['doc_run_id']}*.blocked.json"))
        validated = out_dir / "validated_results" / f"{item['doc_run_id']}.validated.json"
        total_chunks += len(chunks)
        completed_chunks += len(finals)
        blocked_chunks += len(blocked)
        if validated.exists():
            completed += 1
        if blocked:
            blocked_docs += 1
        rows.append(
            {
                "doc_run_id": item["doc_run_id"],
                "chunks": len(chunks),
                "finals": len(finals),
                "validated": validated.exists(),
                "blocked": len(blocked),
            }
        )
    progress = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_docs": total_docs,
        "completed_docs": completed,
        "blocked_docs": blocked_docs,
        "pending_docs": total_docs - completed - blocked_docs,
        "total_chunks": total_chunks,
        "completed_chunks": completed_chunks,
        "blocked_chunks": blocked_chunks,
        "docs": rows,
    }
    atomic_write_json(out_dir / "analysis" / "progress.json", progress)
    report = [
        "# Progress Report",
        "",
        f"- Updated: {progress['updated_at']}",
        f"- Total docs: {total_docs}",
        f"- Completed docs: {completed}",
        f"- Blocked docs: {blocked_docs}",
        f"- Pending docs: {progress['pending_docs']}",
        f"- Total chunks: {total_chunks}",
        f"- Completed chunks: {completed_chunks}",
        f"- Blocked chunks: {blocked_chunks}",
        "",
    ]
    (out_dir / "analysis" / "progress_report.md").write_text("\n".join(report), encoding="utf-8")
    logger.info("%s", json.dumps({k: progress[k] for k in progress if k != "docs"}, ensure_ascii=False, indent=2))


def select_manifest(manifest: list[dict[str, Any]], limit: int | None, doc_run_id: str | None) -> list[dict[str, Any]]:
    if doc_run_id:
        selected = [m for m in manifest if m["doc_run_id"] == doc_run_id or m["source_doc_name"] == doc_run_id]
        if not selected:
            raise SystemExit(f"doc_run_id not found: {doc_run_id}")
        return selected
    return manifest[:limit] if limit else manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Task-local GraphRAG extraction utility")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in [
        "scope",
        "prepare_chunks",
        "classify_risk",
        "init_state",
        "merge_doc_raw",
        "validate_doc_raw",
        "summarize",
    ]:
        p = sub.add_parser(name)
        p.add_argument("--limit", type=int)
        p.add_argument("--doc-run-id")
    p_all = sub.add_parser("all_prep")
    p_all.add_argument("--limit", type=int)
    p_all.add_argument("--doc-run-id")
    args = parser.parse_args()
    if args.command == "scope":
        scope = ensure_scope_map()
        logger.info("scope entries: %s", len(scope))
    elif args.command == "prepare_chunks":
        prepare_chunks(args)
    elif args.command == "classify_risk":
        classify_risk(args)
    elif args.command == "init_state":
        init_doc_state(args)
    elif args.command == "merge_doc_raw":
        merge_doc_raw(args)
    elif args.command == "validate_doc_raw":
        validate_doc_raw(args)
    elif args.command == "summarize":
        summarize_progress(args)
    elif args.command == "all_prep":
        ensure_scope_map()
        prepare_chunks(args)
        classify_risk(args)
        init_doc_state(args)
        summarize_progress(args)


if __name__ == "__main__":
    main()
