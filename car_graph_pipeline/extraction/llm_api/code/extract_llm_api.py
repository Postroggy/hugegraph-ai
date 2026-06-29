#!/usr/bin/env python3
"""LLM API based vehicle-manual graph extraction.

This package-local copy is intentionally isolated under
car_graph_pipeline/extraction/llm_api: source docs live in md_output and every
generated artifact lives in output. It never writes HugeGraph.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import logging
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

TASK_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_IMPORT_ROOT = TASK_ROOT.parents[2]
DOC_DIR = TASK_ROOT / "md_output"
OUTPUT_DIR = TASK_ROOT / "output"
PROMPT_DIR = TASK_ROOT / "prompts"
FEW_SHOT_PATH = PROMPT_DIR / "compact_few_shots.md"

sys.path.insert(0, str(PACKAGE_IMPORT_ROOT))
from car_graph_pipeline.config import LLM_API_KEY, LLM_BASE_URL  # noqa: E402
from car_graph_pipeline.extraction.llm_api.settings import (  # noqa: E402
    ADAPTIVE_TOKEN_STEPS,
    API_RETRIES,
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_CHARS,
    CHUNK_TARGET_CHARS,
    DEFAULT_WORKERS,
    DIRECT_PASS_SCORE,
    MAX_CONTEXT_CHARS,
    MAX_REPAIR_ATTEMPTS,
    MAX_TOKENS_EXTRACT,
    MAX_TOKENS_REVIEW,
    MIN_ACCEPT_SCORE,
    MIN_CONTEXT_CHARS,
    MODEL,
    REQUEST_TIMEOUT,
    REVIEW_CYCLES,
    SMALL_CHUNK_MAX,
    SMALL_CHUNK_MIN,
    SMALL_CHUNK_TARGET,
    TEMPERATURE,
)

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

REQUIRED_PROPERTIES = {
    "VehicleBrand": ["brand_name"],
    "VehicleModel": ["model_name"],
    "VehicleSystem": ["system_name", "system_type"],
    "Component": ["comp_name", "component_type"],
    "Function": ["func_name", "function_type"],
    "Status": ["status_name", "status_type"],
    "Fault": ["fault_name", "fault_type"],
    "Operation": ["op_name", "operation_type"],
    "MaintenanceItem": ["maint_name", "item_type"],
    "Specification": ["spec_name"],
    "Material": ["material_name", "material_type"],
}

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

VALID_RELATION_TYPES = set(RELATION_DIRECTION)

BRAND_HINTS = {
    "奥迪": ["奥迪", "Audi", "A5", "A6", "A8", "Q3", "Q5", "Q6"],
    "广汽埃安": ["AION", "埃安"],
    "奔驰": ["奔驰", "AMG", "Mercedes", "EQB", "EQE", "EQS", "GLB", "CLA", "CLS", "CLE", "A级", "E级"],
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
        "X-Trail",
        "轩逸",
        "奇骏",
    ],
    "丰田": [
        "丰田",
        "Toyota",
        "C-HR",
        "RAV4",
        "YARiS",
        "HIACE",
        "SUPRA",
        "普拉多",
        "普锐斯",
        "埃尔法",
        "亚洲狮",
        "凯美瑞",
    ],
    "林肯": ["林肯", "Lincoln", "Aviator", "Corsair", "MKC", "MKX", "MKZ", "Nautilus", "Navigator", "Zephyr"],
    "本田": [
        "本田",
        "Honda",
        "CR-V",
        "XR-V",
        "HR-V",
        "CIIMO",
        "LIFE",
        "VE-1",
        "飞度",
        "奥德赛",
        "皓影",
        "锋范",
        "凌派",
    ],
    "保时捷": ["保时捷", "Porsche", "Boxster", "Cayenne", "Cayman", "Macan", "Panamera", "Taycan"],
    "凯迪拉克": ["凯迪拉克", "Cadillac", "CT4", "CT5", "CT6", "XT4", "XT5", "XT6", "SLS", "IQ锐歌"],
    "马自达": ["马自达", "Mazda", "CX-3", "CX-4", "CX-5", "CX-30", "Mazda6", "Mazda8", "昂克赛拉"],
    "哈弗": ["哈弗", "H2", "H4", "H5", "H6", "H7", "H9", "Dagou", "大狗", "神兽", "枭龙", "猛龙"],
    "比亚迪": [
        "比亚迪",
        "BYD",
        "F0",
        "F3",
        "G3",
        "G5",
        "G6",
        "e2",
        "e6",
        "秦",
        "唐",
        "宋",
        "元",
        "海鸥",
        "海豚",
        "海豹",
        "汉",
    ],
    "大众": [
        "大众",
        "Volkswagen",
        "CC",
        "ID.3",
        "ID.4",
        "ID.6",
        "Magotan",
        "Polo",
        "T-ROC",
        "迈腾",
        "帕萨特",
        "途观",
        "途昂",
        "朗逸",
        "高尔夫",
        "宝来",
        "速腾",
    ],
    "特斯拉": ["Tesla", "Model 3", "Model S", "Model X", "Model Y"],
    "上汽大通": ["上汽大通", "MAXUS", "大通"],
    "五菱": ["五菱", "宏光", "凯捷", "佳辰", "荣光", "征程", "缤果"],
    "别克": ["别克", "GL8", "世纪", "君威", "凯越", "英朗", "昂科"],
    "雪铁龙": ["雪铁龙", "Citroen", "世嘉", "AirCross", "C4L", "C5", "C6", "凡尔赛"],
    "吉利": ["吉利", "帝豪", "博越", "星越", "星瑞", "银河", "几何"],
    "宝马": ["宝马", "BMW", "MINI"],
    "福特": ["福特", "Ford", "锐界", "探险者", "福克斯", "EVOS"],
    "红旗": ["红旗"],
    "蔚来": ["蔚来", "ES6", "ES7", "EC6", "EC7", "ET7"],
    "小鹏": ["小鹏"],
    "坦克": ["坦克"],
    "长安": ["长安", "UNI"],
    "名爵": ["名爵", "MG"],
    "雷克萨斯": ["雷克萨斯", "Lexus"],
    "起亚": ["起亚", "Kia"],
    "北京": ["北京", "北京越野", "北京现代"],
    "奇瑞": ["奇瑞", "瑞虎", "艾瑞泽", "iCAR"],
    "路虎": ["路虎", "Land Rover"],
    "捷豹": ["捷豹", "Jaguar", "F-TYPE"],
    "斯巴鲁": ["斯巴鲁", "Subaru", "BRZ"],
    "现代": ["现代", "Hyundai"],
    "雪佛兰": ["雪佛兰", "科鲁兹", "科帕奇", "迈锐宝", "赛欧"],
    "标致": ["标致", "Peugeot"],
    "宝骏": ["宝骏"],
    "荣威": ["荣威"],
    "问界": ["问界"],
    "魏牌": ["魏牌"],
    "领克": ["领克"],
    "雷诺": ["雷诺", "Renault"],
}

DIRECT_BRAND_ALIASES = {
    "雷克萨斯": "雷克萨斯",
    "起亚": "起亚",
    "北京越野": "北京",
    "北京现代": "现代",
    "北京": "北京",
    "奇瑞": "奇瑞",
    "路虎": "路虎",
    "捷豹": "捷豹",
    "斯巴鲁": "斯巴鲁",
    "现代": "现代",
    "雪佛兰": "雪佛兰",
    "标致": "标致",
    "宝骏": "宝骏",
    "荣威": "荣威",
    "名爵": "名爵",
    "MG": "名爵",
    "A5L": "奥迪",
    "A6L": "奥迪",
    "Q5L": "奥迪",
    "Q3": "奥迪",
    "Q6": "奥迪",
    "问界": "问界",
    "魏牌": "魏牌",
    "领克": "领克",
    "雷诺": "雷诺",
    "蔚来": "蔚来",
    "小鹏": "小鹏",
    "坦克": "坦克",
    "长安": "长安",
    "宝马": "宝马",
    "奔驰": "奔驰",
    "奥迪": "奥迪",
    "大众": "大众",
    "丰田": "丰田",
    "本田": "本田",
    "日产": "日产",
    "马自达": "马自达",
    "凯迪拉克": "凯迪拉克",
    "福特": "福特",
    "比亚迪": "比亚迪",
    "哈弗": "哈弗",
    "吉利": "吉利",
    "五菱": "五菱",
    "别克": "别克",
    "雪铁龙": "雪铁龙",
    "保时捷": "保时捷",
    "红旗": "红旗",
    "上汽大通": "上汽大通",
}


SCHEMA_PROMPT = """你是汽车知识图谱抽取专家。严格按 schema 从汽车用户手册文本中抽取实体和关系，只输出 JSON。

节点类型和属性：
- VehicleBrand: brand_name, country
- VehicleModel: model_name, year, series, fuel_type, config
- VehicleSystem: system_name, system_type, vehicle_brand, vehicle_model
- Component: comp_name, component_type, location, vehicle_brand, vehicle_model
- Function: func_name, function_type, trigger_condition, alert_method, warnings, vehicle_brand, vehicle_model
- Status: status_name, status_type, perceivable_way, vehicle_brand, vehicle_model
- Fault: fault_name, fault_type, severity, drivable, risk_desc, vehicle_brand, vehicle_model
- Operation: op_name, operation_type, difficulty, steps, precondition, warnings, vehicle_brand, vehicle_model
- MaintenanceItem: maint_name, item_type, interval, warnings, vehicle_brand, vehicle_model
- Specification: spec_name, value_text, value_num, unit, condition_note, vehicle_brand, vehicle_model
- Material: material_name, material_type, spec, brand, vehicle_brand, vehicle_model

关系方向：
HAS_MODEL(VehicleBrand->VehicleModel), HAS_SYSTEM(VehicleModel->VehicleSystem),
HAS_COMPONENT(VehicleModel->Component), HAS_FUNCTION(VehicleModel->Function),
BELONGS_TO(Component->VehicleSystem), ACTIVATES(Component->Function),
OPERATED_BY(Function->Operation), OPERATES_ON(Operation->Component),
HAS_STATUS(Component->Status), SYSTEM_HAS_STATUS(VehicleSystem->Status),
CAUSED_BY(Status->Fault), LEADS_TO(Fault->Fault), AFFECTS(Fault->VehicleSystem),
RESOLVED_BY(Status->Operation), FAULT_RESOLVED_BY(Fault->Operation),
HAS_SPEC(Component->Specification), MODEL_HAS_SPEC(VehicleModel->Specification),
REQUIRES(Operation->Material), APPLICABLE_TO(MaintenanceItem->VehicleModel),
MAINT_HAS_SPEC(MaintenanceItem->Specification), MAINT_REQUIRES(MaintenanceItem->Material)

所有关系 properties 均包含固定范围字段：vehicle_brand, vehicle_model。

关键抽取规则：
1. 唯一事实来源是“待抽取小块”。只抽取文本明确提到的，不推测。
2. 输出必须是一个 JSON 对象，顶层只能包含 entities 和 relations 两个数组；不要输出 Markdown，
   不要输出单个实体对象、单个关系对象或多个分散 JSON 对象。
3. properties 只填文本中明确提到、对召回有用的业务属性，以及固定范围字段
   vehicle_brand、vehicle_model；不要输出 source_doc、chunk_id、heading_path、line_start、line_end。
4. source_snippet 可选但建议填写不超过 80 个中文字符的连续原文短证据；不要输出 evidence 字段，程序会统一补元数据。
5. 固定范围字段必须使用当前文档范围中的 vehicle_brand 和 vehicle_model，不要从文本另行推断；
   程序会在后处理时统一检查并覆盖错误值。VehicleBrand、VehicleModel 这两类点本身不需要
   vehicle_brand/vehicle_model 属性。
6. 不要在每个 chunk 里重复抽 VehicleBrand、VehicleModel、HAS_MODEL。只有当前文本明确讲品牌/车型本身时才抽；
   即使未抽出，程序也会保证文档级 VehicleBrand、VehicleModel 和 HAS_MODEL 存在。
7. 不要为了给 Component/Function 强行连 VehicleModel 而输出大量 HAS_COMPONENT/HAS_FUNCTION；
   这类全局承接关系可由后处理按 vehicle_model 属性补。
8. 警告灯是 Component，“灯亮/闪烁/显示提示”是 Status，根本原因或风险是 Fault。
9. Function 是车辆能力，Operation 是人的操作步骤；步骤、前提、警告要尽量保留。
10. Material 是耗材/工具/油液，Component 是车上的组成部件。
11. 表格/参数不能泛化；不同条件的数值拆成不同 Specification，并在 condition_note 保留条件。
12. 目录、页码索引、纯跳转列表、控制灯总览里只有“名称→页码”的条目时，不要逐行生成实体和关系；
    只抽有明确含义/状态/操作/故障说明的内容。
13. 每个 chunk 通常不要超过 25 个实体、35 条关系。若文本是长表格或密集列表，
    优先抽高价值、可回答问题的信息，不要机械地为每行造点造边。
14. 输出示例中的“当前文档车品牌/当前文档车型号”只是占位说明，
    真实输出时必须替换为当前文档范围里给定的具体 vehicle_brand 和 vehicle_model。
15. CAUSED_BY/LEADS_TO 这类因果边必须有直接因果证据，例如同一句或同一条注意事项中明确出现
    “导致、引起、造成、可能发生、从而”等表达；不要把相邻的独立警告项、注意事项或操作后果强行串成因果链。

输出格式：
必须输出一个 JSON 对象，示例：
{
  "entities": [
    {
      "type": "Component",
      "name": "远光灯",
      "aliases": [],
      "properties": {
        "comp_name": "远光灯",
        "component_type": "灯光部件",
        "description": "远光灯指示相关部件",
        "vehicle_brand": "当前文档车品牌",
        "vehicle_model": "当前文档车型号"
      },
      "source_snippet": "远光灯已打开。"
    },
    {
      "type": "Status",
      "name": "远光灯已打开",
      "aliases": [],
      "properties": {
        "status_name": "远光灯已打开",
        "status_type": "指示灯状态",
        "perceivable_way": "仪表指示灯显示",
        "vehicle_brand": "当前文档车品牌",
        "vehicle_model": "当前文档车型号"
      },
      "source_snippet": "远光灯已打开。"
    }
  ],
  "relations": [
    {
      "type": "HAS_STATUS",
      "source_type": "Component",
      "source_name": "远光灯",
      "target_type": "Status",
      "target_name": "远光灯已打开",
      "properties": {
        "note": "远光灯对应远光灯已打开状态",
        "vehicle_brand": "当前文档车品牌",
        "vehicle_model": "当前文档车型号"
      },
      "source_snippet": "远光灯已打开。"
    }
  ]
}
"""


REVIEW_PROMPT = """你是汽车知识图谱抽取 reviewer。你要对一个 chunk 的抽取草稿做完整度、正确性、schema 和证据审查。

Review 标准：
1. 唯一事实来源是“待抽取小块”。大块上下文只用于理解、命名统一和实体消歧，不可从背景新增实体或关系。
2. 如果草稿中的实体/关系/source_snippet/evidence 只来自“大块上下文”而不是“待抽取小块”，必须判为 unsupported_evidence。
3. 检查是否漏掉高价值实体：操作步骤、按钮/开关/灯/屏、系统、功能、状态、故障、保养项、规格、材料。
4. 检查关系方向是否符合 schema，尤其 Operation->Component、Status->Fault、Component->System。
5. 检查 source_snippet 是否能支撑实体/关系；没有证据的要删除。
6. 重点检查 CAUSED_BY/LEADS_TO：只有同一句或同一条注意事项中存在明确因果表达时才保留；
   相邻但独立的警告、注意事项、风险描述不能互相连成因果边。
7. 审核尺度不要过严：应用你选择的 action 后，如果最终质量能达到 80 分以上即可收口；
   85 分以上直接通过，不要因为少量低价值遗漏要求重抽或重写全量。
8. 你会拿到和 extractor 相同的大块上下文，以及完全相同的“待抽取小块”。
9. quality_score 必须按以下维度给“最终采用结果”打分，不是给原始草稿打分：
   - schema 合法性 25 分：实体/关系类型正确，关系方向正确，端点完整。
   - 证据可靠性 25 分：实体/关系都有待抽取 chunk 内的直接证据，不跨背景补事实。
   - 高价值覆盖 25 分：覆盖操作步骤、按钮/开关/灯/屏、系统、功能、状态、故障、规格、材料等可用于问答召回的信息。
   - 关系语义 15 分：BELONGS_TO、OPERATES_ON、HAS_STATUS、CAUSED_BY/LEADS_TO 等关系语义准确，不把并列事项硬连。
   - 简洁去噪 10 分：不抽目录页码、服务网点通讯录、低价值重复项，不机械展开全局车型承接边。
10. 优先在 review 阶段解决问题，尽量不要要求 extractor 重抽：
   - pass：草稿最终质量 85 分以上，可直接保存，supplement_output/corrected_output 均为 null。
   - minor_accept：草稿有小问题但不影响召回，最终质量 80 分以上，直接保存草稿。
   - supplement：草稿主体正确，只缺少少量高价值点/边；只在 supplement_output 给增量，不要重写全量。
     仅遗漏/补充信息时必须用 supplement，不要用 corrected_full。
   - corrected_full：只有当草稿存在需要删除或替换的错误内容时才用，
     例如关系方向错、端点错、无证据实体/关系、错误因果边；给一份全量 corrected_output。
   - repair_required：抽取很烂、事实大量不支持、主体错乱、或无法通过全量修正解决；只有这种情况才要求重抽。
11. 如果 action_tag 是 supplement 或 corrected_full，必须保证对应输出是完整合法的 {entities, relations} JSON 对象。
12. quality_score 表示应用 supplement_output/corrected_output 之后的最终结果质量，不是原始草稿质量；
    若你给 supplement 或 corrected_full，且修正后结果可用，quality_score 应反映修正后的质量，通常不应低于 80。
13. 不要为了低价值遗漏输出 corrected_full；能直接通过就 pass/minor_accept，
    只缺少少量高价值事实就 supplement，只有需要删除错误项或修正方向/端点时才 corrected_full。

只输出 JSON：
通过示例：
{
  "action_tag": "pass",
  "passed": true,
  "quality_score": 92,
  "issue_tag": "pass",
  "severity": "pass",
  "reasons": [],
  "feedback_for_extractor": "",
  "supplement_output": null,
  "corrected_output": null
}
不通过示例：
{
  "action_tag": "repair_required",
  "passed": false,
  "quality_score": 55,
  "issue_tag": "missing_entities",
  "severity": "major",
  "reasons": ["待抽取小块中有明确的警告灯状态，但草稿未抽取 Status。"],
  "feedback_for_extractor": "请只依据待抽取小块补充警告灯 Component、Status 及 HAS_STATUS 关系。",
  "supplement_output": null,
  "corrected_output": null
}
增量补充示例：
{
  "action_tag": "supplement",
  "passed": true,
  "quality_score": 86,
  "issue_tag": "missing_minor_facts",
  "severity": "minor",
  "reasons": ["仅缺少少量高价值关系，已在 supplement_output 中补齐。"],
  "feedback_for_extractor": "",
  "supplement_output": {
    "entities": [],
    "relations": []
  },
  "corrected_output": null
}
全量修正示例：
{
  "action_tag": "corrected_full",
  "passed": true,
  "quality_score": 88,
  "issue_tag": "minor_relation_errors",
  "severity": "minor",
  "reasons": ["少量关系方向错误，已在 corrected_output 中修正。"],
  "feedback_for_extractor": "",
  "supplement_output": null,
  "corrected_output": {
    "entities": [],
    "relations": []
  }
}
"""


@dataclass
class Section:
    title: str
    heading_path: str
    text: str
    line_start: int
    line_end: int


class ExtractEntityModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    source_snippet: str | None = None


class ExtractRelationModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    source_type: str
    source_name: str
    target_type: str
    target_name: str
    properties: dict[str, Any] = Field(default_factory=dict)
    source_snippet: str | None = None


class ExtractionOutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[ExtractEntityModel]
    relations: list[ExtractRelationModel]


class ReviewOutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_tag: Literal["pass", "minor_accept", "supplement", "corrected_full", "repair_required"]
    passed: bool
    quality_score: int = Field(ge=0, le=100)
    issue_tag: str
    severity: Literal["pass", "minor", "major"]
    reasons: list[str] = Field(default_factory=list)
    feedback_for_extractor: str = ""
    supplement_output: ExtractionOutputModel | None = None
    corrected_output: ExtractionOutputModel | None = None


def pydantic_error_messages(exc: ValidationError) -> list[str]:
    messages = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ()))
        messages.append(f"{loc or '<root>'}: {err.get('msg', 'validation error')}")
    return messages


def validate_extraction_pydantic(result: Any) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        model = ExtractionOutputModel.model_validate(result)
    except ValidationError as exc:
        return None, pydantic_error_messages(exc)
    return model.model_dump(mode="python"), []


def validate_review_pydantic(result: Any) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        model = ReviewOutputModel.model_validate(result)
    except ValidationError as exc:
        return None, pydantic_error_messages(exc)
    return model.model_dump(mode="python"), []


_thread_local = threading.local()


def setup_logging() -> logging.Logger:
    (OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "logs" / f"extract_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("llm_api_extract")


logger = setup_logging()


def key_fingerprint(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:10]


def get_client():
    client = getattr(_thread_local, "client", None)
    if client is None:
        from openai import OpenAI

        client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            timeout=REQUEST_TIMEOUT,
            max_retries=0,
        )
        _thread_local.client = client
    return client


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_few_shots() -> str:
    if FEW_SHOT_PATH.exists():
        return FEW_SHOT_PATH.read_text(encoding="utf-8").strip()
    return ""


def safe_stem(name: str) -> str:
    stem = re.sub(r"\.md$", "", name, flags=re.IGNORECASE)
    stem = re.sub(r"[^\w\u4e00-\u9fff.+ -]+", "_", stem)
    return stem.strip().replace("/", "_")[:180]


def clean_text(text: str) -> str:
    text = re.sub(r"!\[.*?\]\([^)]*\)", "", text)
    text = re.sub(r"\$\\bullet\$", "-", text)
    text = re.sub(r"\$\\triangleright\$", "▶", text)
    text = re.sub(r"\$\\Leftrightarrow\$", "⟺", text)
    text = re.sub(r"\$[^$]{1,40}\$", "", text)
    text = re.sub(r"www\.carobook\.com", "", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def heading_level(line: str) -> int | None:
    m = re.match(r"^(#{1,4})\s+(.+?)\s*$", line)
    return len(m.group(1)) if m else None


def heading_title(line: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", line).strip()


def low_value_chunk_reason(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "empty"
    head = text[:500]
    if re.search(r"(版权所有|未经.*许可|翻印|复制|保留修改|定稿日期|公司可能随时改进产品)", head) and len(text) < 1600:
        return "front_matter"
    sentence_marks = len(re.findall(r"[。！？；]", text))
    h1_count = sum(1 for line in lines if line.startswith("# "))
    page_ref_lines = sum(
        1 for line in lines if re.search(r"(第\s*\d+\s*页|→\s*第?\s*\d+\s*页|\s+\d{1,4}$|<td>\s*\d{1,4}\s*</td>)", line)
    )
    page_ref_ratio = page_ref_lines / max(len(lines), 1)
    first_lines = lines[: min(80, len(lines))]
    first_page_ref_lines = sum(
        1 for line in first_lines if re.search(r"(第\s*\d+\s*页|→\s*第?\s*\d+\s*页|\s+\d{1,4}$)", line)
    )
    first_page_ref_ratio = first_page_ref_lines / max(len(first_lines), 1)
    toc_like = sum(1 for line in lines if re.match(r"^.{2,45}\s+\.?\s*\d{1,4}$", line))
    toc_ratio = toc_like / max(len(lines), 1)
    heading_text = " ".join(line.lstrip("# ").strip() for line in lines[:12] if line.startswith("#"))
    index_heading = bool(re.search(r"(目录|索引|页码|一览|总览)", heading_text + " " + head[:120]))
    spec_table = bool(
        re.search(
            r"(技术数据|规格|输出功率|扭矩|机油|加注量|空车重量|允许总重量|轮胎|胎压|kPa|bar|kg|kW|Nm|mm|Ah|V|升|L/100)",
            text,
        )
    )
    dealer_directory = bool(
        re.search(r"(服务网点通讯录|销售服务商全称|服务站地址|特约服务站|服务有限公司|汽车销售服务有限公司)", text)
    )
    html_cells = text.count("<td") + text.count("<tr")

    if dealer_directory and html_cells >= 80 and sentence_marks <= 12:
        return "dealer_service_directory"

    if (
        len(lines) >= 12
        and toc_ratio > 0.55
        and sentence_marks <= max(4, int(toc_like * 0.2))
        and not (spec_table and "<td" in text)
    ):
        return "toc_like_lines"
    if len(lines) >= 12 and page_ref_ratio > 0.65 and not (spec_table and "<td" in text):
        return "page_index_lines"
    if len(first_lines) >= 30 and first_page_ref_ratio > 0.65 and not (spec_table and "<td" in "\n".join(first_lines)):
        return "toc_prefix_page_refs"
    if index_heading and len(lines) >= 10 and page_ref_ratio > 0.35 and sentence_marks <= 12:
        return "index_heading_page_refs"
    if h1_count > 12 and sentence_marks < max(8, h1_count):
        return "many_headings_index"
    page_refs = sum(1 for line in lines if re.search(r"(第\s*\d+\s*页|\s+\d{1,4}$|<td>\s*\d{1,4}\s*</td>)", line))
    if (
        len(lines) >= 20
        and page_refs / len(lines) > 0.65
        and sentence_marks <= 8
        and not (spec_table and "<td" in text)
    ):
        return "dense_page_refs"
    arrow_refs = len(re.findall(r"→\s*第?\s*\d+\s*页|第\s*\d+\s*页", text))
    if index_heading and arrow_refs >= 20 and sentence_marks < max(10, arrow_refs * 0.2):
        return "arrow_page_index"
    table_refs = len(re.findall(r"<tr>|</td>|→\s*第?\s*\d+\s*页", text))
    if table_refs >= 100 and arrow_refs >= 30 and sentence_marks <= 30:
        return "page_navigation_table"
    if index_heading and table_refs >= 60 and sentence_marks < 20:
        return "html_index_table"
    return None


def is_low_value_chunk(text: str) -> bool:
    return low_value_chunk_reason(text) is not None


def split_sections(text: str) -> list[Section]:
    lines = text.splitlines()
    headings: dict[int, str] = {}
    sections: list[Section] = []
    buf: list[str] = []
    start = 1

    def current_path() -> str:
        return " > ".join(headings[i] for i in sorted(headings) if headings[i])

    def flush(end_line: int) -> None:
        nonlocal buf
        raw = "\n".join(buf).strip()
        if raw and len(raw) >= 50:
            hp = current_path()
            title = hp.split(" > ")[-1] if hp else ""
            sections.append(Section(title=title, heading_path=hp, text=raw, line_start=start, line_end=end_line))
        buf = []

    for idx, line in enumerate(lines, start=1):
        level = heading_level(line)
        if level is not None:
            flush(idx - 1)
            for old in list(headings):
                if old >= level:
                    del headings[old]
            headings[level] = heading_title(line)
            start = idx
            buf = [line]
        else:
            if not buf:
                start = idx
            buf.append(line)
    flush(len(lines))

    if not sections:
        sections.append(Section(title="", heading_path="", text=text, line_start=1, line_end=len(lines)))
    return sections


def split_oversized_section(section: Section) -> list[Section]:
    if len(section.text) <= MAX_CONTEXT_CHARS:
        return [section]
    hard_limit = MAX_CONTEXT_CHARS

    def hard_split(text: str) -> list[str]:
        if len(text) <= hard_limit:
            return [text]
        lines = text.splitlines()
        if len(lines) > 1:
            pieces: list[str] = []
            buf: list[str] = []
            for line in lines:
                if len(line) > hard_limit:
                    if buf:
                        pieces.append("\n".join(buf).strip())
                        buf = []
                    pieces.extend(
                        line[i : i + hard_limit].strip()
                        for i in range(0, len(line), hard_limit)
                        if line[i : i + hard_limit].strip()
                    )
                    continue
                candidate = "\n".join([*buf, line])
                if buf and len(candidate) > hard_limit:
                    pieces.append("\n".join(buf).strip())
                    buf = [line]
                else:
                    buf.append(line)
            if buf:
                pieces.append("\n".join(buf).strip())
            return [p for p in pieces if p]
        return [
            text[i : i + hard_limit].strip()
            for i in range(0, len(text), hard_limit)
            if text[i : i + hard_limit].strip()
        ]

    paras = re.split(r"\n\s*\n+", section.text)
    out: list[Section] = []
    buf: list[str] = []
    part = 1
    for para in paras:
        para = para.strip()
        if not para:
            continue
        if len(para) > hard_limit:
            if buf:
                text = "\n\n".join(buf).strip()
                out.append(
                    Section(
                        title=f"{section.title} part {part}",
                        heading_path=section.heading_path,
                        text=text,
                        line_start=section.line_start,
                        line_end=section.line_end,
                    )
                )
                part += 1
                buf = []
            for piece in hard_split(para):
                out.append(
                    Section(
                        title=f"{section.title} part {part}",
                        heading_path=section.heading_path,
                        text=piece,
                        line_start=section.line_start,
                        line_end=section.line_end,
                    )
                )
                part += 1
            continue
        candidate = "\n\n".join([*buf, para])
        if buf and len(candidate) > hard_limit:
            text = "\n\n".join(buf).strip()
            out.append(
                Section(
                    title=f"{section.title} part {part}",
                    heading_path=section.heading_path,
                    text=text,
                    line_start=section.line_start,
                    line_end=section.line_end,
                )
            )
            part += 1
            buf = [para]
        else:
            buf.append(para)
    if buf:
        out.append(
            Section(
                title=f"{section.title} part {part}" if part > 1 else section.title,
                heading_path=section.heading_path,
                text="\n\n".join(buf).strip(),
                line_start=section.line_start,
                line_end=section.line_end,
            )
        )
    return out


def is_navigation_overview_section(section: Section) -> bool:
    heading = f"{section.heading_path} > {section.title}"
    return bool(re.search(r"(目录|索引|一览图|控制灯总览|符号概览|功能概览|快速访问.*概览)", heading))


def build_context_chunks(sections: list[Section], doc_name: str, scope: dict[str, str]) -> list[dict[str, Any]]:
    expanded: list[Section] = []
    for sec in sections:
        expanded.extend(split_oversized_section(sec))

    groups: list[list[Section]] = []
    buf: list[Section] = []
    for sec in expanded:
        if is_navigation_overview_section(sec):
            if buf:
                groups.append(buf)
                buf = []
            groups.append([sec])
            continue
        candidate_len = len("\n\n".join(s.text for s in [*buf, sec]))
        cur_len = len("\n\n".join(s.text for s in buf)) if buf else 0
        if buf and (
            (cur_len >= CHUNK_TARGET_CHARS and candidate_len > CHUNK_TARGET_CHARS) or candidate_len > MAX_CONTEXT_CHARS
        ):
            groups.append(buf)
            buf = []
        buf.append(sec)
    if buf:
        if groups and sum(len(s.text) for s in buf) < MIN_CONTEXT_CHARS:
            prev_len = len("\n\n".join(s.text for s in groups[-1]))
            cur_len = len("\n\n".join(s.text for s in buf))
            if prev_len + cur_len <= MAX_CONTEXT_CHARS - 10:
                groups[-1].extend(buf)
            else:
                groups.append(buf)
        else:
            groups.append(buf)

    def overlap_tail(text: str) -> str:
        if CHUNK_OVERLAP_CHARS <= 0 or len(text) <= CHUNK_OVERLAP_CHARS:
            return ""
        tail = text[-CHUNK_OVERLAP_CHARS:]
        paragraph_pos = tail.find("\n\n")
        if paragraph_pos != -1 and paragraph_pos < len(tail) - 80:
            tail = tail[paragraph_pos + 2 :]
        sentence_pos = max(tail.find("。"), tail.find("."), tail.find("；"), tail.find(";"))
        if 0 <= sentence_pos < len(tail) - 80:
            tail = tail[sentence_pos + 1 :]
        return tail.strip()

    contexts = []
    prev_text = ""
    prev_was_navigation = False
    for i, group in enumerate(groups):
        group_is_navigation = all(is_navigation_overview_section(sec) for sec in group)
        base_text = "\n\n".join(sec.text for sec in group).strip()
        prefix = "" if prev_was_navigation or group_is_navigation else (overlap_tail(prev_text) if i > 0 else "")
        text = f"{prefix}\n\n{base_text}".strip() if prefix else base_text
        headings = []
        for sec in group:
            if sec.heading_path and sec.heading_path not in headings:
                headings.append(sec.heading_path)
        contexts.append(
            {
                "context_id": f"{safe_stem(doc_name)}_ctx{i:03d}",
                "doc_name": doc_name,
                "vehicle_brand": scope["vehicle_brand"],
                "vehicle_model": scope["vehicle_model"],
                "context_index": i,
                "heading_paths": headings,
                "line_start": min(sec.line_start for sec in group),
                "line_end": max(sec.line_end for sec in group),
                "char_count": len(text),
                "target_chars": CHUNK_TARGET_CHARS,
                "max_chars": MAX_CONTEXT_CHARS,
                "overlap_chars": len(prefix),
                "text": text,
            }
        )
        prev_text = base_text
        prev_was_navigation = group_is_navigation
    return contexts


def split_small_chunks(context: dict[str, Any]) -> list[dict[str, Any]]:
    paras = [p.strip() for p in re.split(r"\n\s*\n+", context["text"]) if p.strip()]
    chunks: list[dict[str, Any]] = []
    buf: list[str] = []

    def infer_heading(text: str) -> tuple[str, str]:
        titles = []
        for line in text.splitlines():
            m = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
            if m:
                title = clean_text(m.group(1))
                if title and title not in titles:
                    titles.append(title)
        if titles:
            return " || ".join(titles[-4:]), titles[-1]
        fallback = context.get("heading_paths") or []
        heading_path = " || ".join(fallback[-4:])
        source_section = fallback[-1].split(" > ")[-1] if fallback else ""
        return heading_path, source_section

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        text = "\n\n".join(buf).strip()
        if len(text) >= SMALL_CHUNK_MIN and not is_low_value_chunk(text):
            idx = len(chunks)
            heading_path, source_section = infer_heading(text)
            chunks.append(
                {
                    "chunk_id": f"{context['context_id']}_c{idx:03d}",
                    "context_id": context["context_id"],
                    "doc_name": context["doc_name"],
                    "vehicle_brand": context["vehicle_brand"],
                    "vehicle_model": context["vehicle_model"],
                    "context_index": context["context_index"],
                    "chunk_index": idx,
                    "heading_path": heading_path,
                    "source_section": source_section,
                    "line_start": context["line_start"],
                    "line_end": context["line_end"],
                    "text": text,
                    "char_count": len(text),
                }
            )
        buf = []

    for para in paras:
        if len(para) > SMALL_CHUNK_MAX:
            flush()
            parts = re.split(r"(?<=。|；|;|！|!|？|\?)\s*", para)
            part_buf = ""
            for part in parts:
                if not part:
                    continue
                if part_buf and len(part_buf) + len(part) > SMALL_CHUNK_TARGET:
                    buf = [part_buf]
                    flush()
                    part_buf = part
                else:
                    part_buf = part_buf + part if part_buf else part
            if part_buf:
                buf = [part_buf]
                flush()
            continue
        candidate = "\n\n".join([*buf, para])
        if buf and len(candidate) > SMALL_CHUNK_TARGET:
            flush()
        buf.append(para)
        if len("\n\n".join(buf)) >= SMALL_CHUNK_TARGET:
            flush()
    flush()
    if len(chunks) >= 2 and chunks[-1]["char_count"] < SMALL_CHUNK_MIN * 2:
        prev = chunks[-2]
        tail = chunks[-1]
        merged_text = f"{prev['text']}\n\n{tail['text']}".strip()
        if len(merged_text) <= SMALL_CHUNK_MAX:
            heading_parts: list[str] = []
            for value in (prev.get("heading_path"), tail.get("heading_path")):
                for part in str(value or "").split(" || "):
                    part = part.strip()
                    if part and part not in heading_parts:
                        heading_parts.append(part)
            prev["text"] = merged_text
            prev["char_count"] = len(merged_text)
            prev["line_end"] = tail["line_end"]
            prev["heading_path"] = " || ".join(heading_parts[-4:])
            prev["source_section"] = tail.get("source_section") or prev.get("source_section", "")
            chunks.pop()
    for idx, chunk in enumerate(chunks):
        chunk["chunk_index"] = idx
        chunk["chunk_id"] = f"{context['context_id']}_c{idx:03d}"
    return chunks


def normalize_vehicle_model_name(doc_name: str) -> str:
    stem = re.sub(r"[\u200b\ufeff]", "", doc_name).strip()
    stem = re.sub(r"(?:\.(?:md|pdf))+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\s+copy$", "", stem, flags=re.IGNORECASE).strip()
    stem = re.sub(r"^\d{4}款", "", stem).strip(" -_")
    stem = re.sub(
        r"(?:使用说明书|用户手册|说明书|操作手册)\s*(?:copy)?\s*(?:\d{4}款?)?$", "", stem, flags=re.IGNORECASE
    )
    stem = stem.strip(" -_")
    if "-" in stem:
        left, right = stem.split("-", 1)
        if (
            re.search(r"[\u4e00-\u9fff]", left)
            and not re.search(r"[A-Za-z0-9]", left)
            and re.search(r"[A-Za-z]", right)
        ):
            stem = left.strip(" -_")
    return stem or re.sub(r"(?:\.(?:md|pdf))+$", "", doc_name, flags=re.IGNORECASE)


def infer_scope(doc_name: str) -> dict[str, str]:
    model = normalize_vehicle_model_name(doc_name)
    brand = ""
    upper_name = doc_name.upper()
    for alias, candidate in sorted(DIRECT_BRAND_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True):
        if re.fullmatch(r"[A-Za-z0-9. _-]+", alias):
            a = alias.upper()
            matched = upper_name.startswith(a) or bool(
                re.search(rf"(?<![A-Z0-9]){re.escape(a)}(?![A-Z0-9])", upper_name)
            )
        else:
            matched = alias in doc_name
        if matched:
            brand = candidate
            break
    if not brand:
        for candidate, hints in BRAND_HINTS.items():
            for hint in sorted(hints, key=len, reverse=True):
                if not hint:
                    continue
                if re.search(r"[\u4e00-\u9fff]", hint):
                    if len(hint) >= 2 and hint in doc_name:
                        brand = candidate
                        break
                else:
                    h = hint.upper()
                    if len(h) < 3:
                        continue
                    if upper_name.startswith(h) or re.search(rf"(?<![A-Z0-9]){re.escape(h)}(?![A-Z0-9])", upper_name):
                        brand = candidate
                        break
            if brand:
                break
    return {
        "vehicle_brand": brand or "UNKNOWN_BRAND",
        "vehicle_model": model,
        "confidence": "high" if brand else "low",
        "source": "filename_brand_hints" if brand else "filename_model_only",
    }


def read_doc_text(doc_path: Path) -> str:
    try:
        return doc_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("decode with replacement: %s", doc_path.name)
        return doc_path.read_bytes().decode("utf-8", errors="replace")


def generate_entity_id(entity_type: str, name: str, vehicle_model: str) -> str:
    prefix = {
        "VehicleBrand": "brand",
        "VehicleModel": "model",
        "VehicleSystem": "system",
        "Component": "comp",
        "Function": "func",
        "Status": "status",
        "Fault": "fault",
        "Operation": "op",
        "MaintenanceItem": "maint",
        "Specification": "spec",
        "Material": "mat",
    }.get(entity_type, entity_type.lower())
    if entity_type == "VehicleBrand":
        return f"{prefix}::{name}"
    if entity_type == "VehicleModel":
        return f"{prefix}::{name}"
    return f"{prefix}::{vehicle_model}::{name}"


def entity_name(entity: dict[str, Any]) -> str:
    name = str(entity.get("name") or "").strip()
    if name:
        return name
    props = entity.get("properties") or {}
    prop = ENTITY_NAME_PROPS.get(str(entity.get("type") or ""))
    return str(props.get(prop) or "").strip()


def ensure_metadata(item: dict[str, Any], chunk: dict[str, Any], scope: dict[str, str]) -> None:
    item["chunk_id"] = chunk["chunk_id"]
    item["heading_path"] = chunk.get("heading_path", "")
    item["source_section"] = chunk.get("source_section", "")
    item["line_start"] = chunk.get("line_start")
    item["line_end"] = chunk.get("line_end")
    item["evidence"] = f"{chunk['doc_name']}:{chunk.get('line_start')}-{chunk.get('line_end')}"
    item.setdefault("source_snippet", chunk["text"][:160])
    if isinstance(item.get("source_snippet"), str) and len(item["source_snippet"]) > 160:
        item["source_snippet"] = item["source_snippet"][:160]
    props = item.setdefault("properties", {})
    is_relation = item.get("type") in VALID_RELATION_TYPES
    if not is_relation:
        props["source_doc"] = chunk["doc_name"]
    else:
        props.pop("source_doc", None)
    if is_relation or item.get("type") not in {"VehicleBrand", "VehicleModel"}:
        props["vehicle_brand"] = scope["vehicle_brand"]
        props["vehicle_model"] = scope["vehicle_model"]


def validate_and_normalize(
    result: dict[str, Any], chunk: dict[str, Any], scope: dict[str, str]
) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(result, dict):
        return {"entities": [], "relations": []}, ["result is not object"], warnings

    entities: list[dict[str, Any]] = []
    known: set[tuple[str, str]] = set()
    for ent in result.get("entities") or []:
        if not isinstance(ent, dict):
            errors.append("entity is not object")
            continue
        etype = str(ent.get("type") or "").strip()
        name = entity_name(ent)
        if etype not in VALID_ENTITY_TYPES:
            errors.append(f"invalid entity type: {etype}")
            continue
        if not name:
            errors.append(f"missing entity name for {etype}")
            continue
        if etype == "VehicleBrand":
            name = scope["vehicle_brand"]
        elif etype == "VehicleModel":
            name = scope["vehicle_model"]
        ent["type"] = etype
        ent["name"] = name
        props = ent.setdefault("properties", {})
        if etype in {"VehicleBrand", "VehicleModel"}:
            props.pop("vehicle_brand", None)
            props.pop("vehicle_model", None)
        name_prop = ENTITY_NAME_PROPS[etype]
        if etype in {"VehicleBrand", "VehicleModel"}:
            props[name_prop] = name
        else:
            props.setdefault(name_prop, name)
        for required in REQUIRED_PROPERTIES.get(etype, []):
            if not props.get(required):
                warnings.append(f"{etype}({name}) missing property {required}")
        ensure_metadata(ent, chunk, scope)
        ent["entity_id"] = generate_entity_id(etype, name, scope["vehicle_model"])
        known.add((etype, name))
        entities.append(ent)

    relations: list[dict[str, Any]] = []

    def add_stub_entity(etype: str, name: str, snippet: str = "") -> None:
        if not etype or not name or etype not in VALID_ENTITY_TYPES or (etype, name) in known:
            return
        ent = {
            "type": etype,
            "name": name,
            "aliases": [],
            "properties": {ENTITY_NAME_PROPS[etype]: name},
            "source_snippet": snippet,
            "auto_stub": True,
        }
        ensure_metadata(ent, chunk, scope)
        ent["entity_id"] = generate_entity_id(etype, name, scope["vehicle_model"])
        known.add((etype, name))
        entities.append(ent)

    for rel in result.get("relations") or []:
        if not isinstance(rel, dict):
            errors.append("relation is not object")
            continue
        rtype = str(rel.get("type") or "").strip()
        src_type = str(rel.get("source_type") or "").strip()
        tgt_type = str(rel.get("target_type") or "").strip()
        src_name = str(rel.get("source_name") or "").strip()
        tgt_name = str(rel.get("target_name") or "").strip()
        if rtype not in VALID_RELATION_TYPES:
            errors.append(f"invalid relation type: {rtype}")
            continue
        expected = RELATION_DIRECTION[rtype]
        if (src_type, tgt_type) != expected:
            errors.append(
                f"relation direction error {rtype}: {src_type}->{tgt_type}, expected {expected[0]}->{expected[1]}"
            )
            continue
        if not src_name or not tgt_name:
            errors.append(f"relation missing endpoint: {rtype}")
            continue
        rel["type"] = rtype
        rel["source_type"] = src_type
        rel["source_name"] = src_name
        rel["target_type"] = tgt_type
        rel["target_name"] = tgt_name
        ensure_metadata(rel, chunk, scope)
        rel["source_entity_id"] = generate_entity_id(src_type, src_name, scope["vehicle_model"])
        rel["target_entity_id"] = generate_entity_id(tgt_type, tgt_name, scope["vehicle_model"])
        if (src_type, src_name) not in known:
            add_stub_entity(src_type, src_name, rel.get("source_snippet", ""))
            warnings.append(f"relation source auto-stubbed: {src_type}({src_name})")
        if (tgt_type, tgt_name) not in known:
            add_stub_entity(tgt_type, tgt_name, rel.get("source_snippet", ""))
            warnings.append(f"relation target auto-stubbed: {tgt_type}({tgt_name})")
        relations.append(rel)

    if not entities and chunk.get("char_count", 0) > 250:
        warnings.append("no entities extracted")
    return {"entities": entities, "relations": relations}, errors, warnings


def merge_normalized_results(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    entities: dict[tuple[str, str], dict[str, Any]] = {}
    relations: dict[tuple[str, str, str], dict[str, Any]] = {}

    for ent in [*(base.get("entities") or []), *(extra.get("entities") or [])]:
        key = (str(ent.get("type") or ""), entity_name(ent))
        if not key[0] or not key[1]:
            continue
        if key not in entities:
            entities[key] = ent
            continue
        old = entities[key]
        old_props = old.setdefault("properties", {})
        for prop_key, value in (ent.get("properties") or {}).items():
            if value and (not old_props.get(prop_key) or len(str(value)) > len(str(old_props.get(prop_key, "")))):
                old_props[prop_key] = value
        old_aliases = old.setdefault("aliases", [])
        for alias in ent.get("aliases") or []:
            if alias and alias not in old_aliases:
                old_aliases.append(alias)
        if len(str(ent.get("source_snippet") or "")) > len(str(old.get("source_snippet") or "")):
            old["source_snippet"] = ent.get("source_snippet")

    for rel in [*(base.get("relations") or []), *(extra.get("relations") or [])]:
        src = rel.get("source_entity_id") or f"{rel.get('source_type')}::{rel.get('source_name')}"
        tgt = rel.get("target_entity_id") or f"{rel.get('target_type')}::{rel.get('target_name')}"
        key = (str(rel.get("type") or ""), str(src), str(tgt))
        if key[0] and key[1] and key[2] and key not in relations:
            relations[key] = rel

    return {"entities": list(entities.values()), "relations": list(relations.values())}


def normalize_review_extraction_output(
    payload: Any,
    chunk: dict[str, Any],
    scope: dict[str, str],
    field_name: str,
) -> tuple[dict[str, Any], list[str], list[str]]:
    output = coerce_extraction_output(payload)
    model_output, pydantic_errors = validate_extraction_pydantic(output)
    if model_output is None:
        return (
            {"entities": [], "relations": []},
            [f"pydantic {field_name} validation failed: {msg}" for msg in pydantic_errors],
            [],
        )
    return validate_and_normalize(model_output, chunk, scope)


def can_save_partial_review_output(result: dict[str, Any], errors: list[str], quality_score: int) -> bool:
    if quality_score < MIN_ACCEPT_SCORE:
        return False
    if not result.get("entities"):
        return False
    if not errors:
        return True
    droppable_prefixes = (
        "relation direction error ",
        "relation missing endpoint:",
        "invalid relation type:",
    )
    return all(any(error.startswith(prefix) for prefix in droppable_prefixes) for error in errors)


def usage_completion_tokens(usage: dict[str, Any]) -> int:
    val = usage.get("completion_tokens") if isinstance(usage, dict) else 0
    return int(val) if isinstance(val, (int, float)) else 0


def compact_for_match(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    text = re.sub(r"[\s|　]+", "", text)
    text = re.sub(r"[，。；：、,.():（）\\[\\]【】<>《》\"'“”‘’`]", "", text)
    return text.lower()


def near_token_limit(usage: dict[str, Any], max_tokens: int) -> bool:
    completion = usage_completion_tokens(usage)
    if completion <= 0:
        return False
    return completion >= max(max_tokens - 3_000, int(max_tokens * 0.9))


def next_token_limit(current: int) -> int | None:
    for limit in ADAPTIVE_TOKEN_STEPS:
        if current < limit:
            return limit
    return None


def token_limit_feedback(stage: str, usage: dict[str, Any], old_limit: int, new_limit: int) -> str:
    return json.dumps(
        {
            "issue_tag": "output_truncated",
            "severity": "major",
            "reasons": [
                f"{stage} completion_tokens={usage_completion_tokens(usage)} "
                f"接近本轮 max_tokens={old_limit}，疑似输出被截断或过长。"
            ],
            "feedback_for_extractor": (
                f"上一轮输出接近 max_tokens，本轮已把当前 chunk 的 {stage} max_tokens 提高到 {new_limit}。"
                "请继续只输出完整 JSON，不要输出解释或 Markdown；不要重复无证据实体。"
            ),
        },
        ensure_ascii=False,
    )


def result_counts(result: dict[str, Any]) -> tuple[int, int]:
    return len(result.get("entities", []) or []), len(result.get("relations", []) or [])


def result_summary(result: dict[str, Any], limit: int = 12) -> dict[str, Any]:
    entities = result.get("entities", []) or []
    relations = result.get("relations", []) or []
    return {
        "entity_count": len(entities),
        "relation_count": len(relations),
        "sample_entities": [
            {"type": ent.get("type"), "name": ent.get("name")} for ent in entities[:limit] if isinstance(ent, dict)
        ],
        "sample_relations": [
            {
                "type": rel.get("type"),
                "source": f"{rel.get('source_type')}:{rel.get('source_name')}",
                "target": f"{rel.get('target_type')}:{rel.get('target_name')}",
            }
            for rel in relations[:limit]
            if isinstance(rel, dict)
        ],
    }


def review_summary(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"type": type(result).__name__}
    corrected = result.get("corrected_output")
    supplement = result.get("supplement_output")
    summary = {
        "action_tag": result.get("action_tag"),
        "passed": result.get("passed"),
        "quality_score": result.get("quality_score"),
        "issue_tag": result.get("issue_tag"),
        "severity": result.get("severity"),
        "reasons": (result.get("reasons") or [])[:8]
        if isinstance(result.get("reasons"), list)
        else result.get("reasons"),
        "feedback_for_extractor": str(result.get("feedback_for_extractor") or "")[:800],
    }
    if isinstance(supplement, dict):
        summary["supplement_output_summary"] = result_summary(coerce_extraction_output(supplement))
    if isinstance(corrected, dict):
        summary["corrected_output_summary"] = result_summary(coerce_extraction_output(corrected))
    return summary


def over_extracted_reason(result: dict[str, Any], chunk: dict[str, Any]) -> str | None:
    entity_count, relation_count = result_counts(result)
    total = entity_count + relation_count
    char_count = int(chunk.get("char_count") or 0)
    if entity_count > 45 or relation_count > 70 or total > 110:
        return f"too_many_items entities={entity_count} relations={relation_count}"
    if char_count <= 1800 and total > 100:
        return f"too_dense_for_short_chunk chars={char_count} entities={entity_count} relations={relation_count}"
    if char_count <= 1200 and total > 80:
        return f"too_dense_for_tiny_chunk chars={char_count} entities={entity_count} relations={relation_count}"
    return None


def over_extraction_feedback(reason: str, result: dict[str, Any]) -> str:
    return json.dumps(
        {
            "issue_tag": "over_extracted",
            "severity": "major",
            "reasons": [
                reason,
                "上一轮输出明显过长，通常是把目录/页码索引/全局车型承接边机械展开了。",
            ],
            "feedback_for_extractor": (
                "请重新抽取同一个 chunk，但只保留当前 chunk 中对问答召回有价值的局部事实；"
                "不要重复 VehicleBrand、VehicleModel、HAS_MODEL；"
                "不要为每个部件强行补 VehicleModel 的 HAS_COMPONENT/HAS_FUNCTION；"
                "目录、页码索引、只有名称和页码的总览表不要逐行造点造边；"
                "本轮必须控制在 35 个实体、55 条关系以内，优先保留操作、状态、故障、规格和关键部件。"
            ),
            "previous_extract_summary": result_summary(result),
        },
        ensure_ascii=False,
    )


def source_evidence_warnings(result: dict[str, Any], chunk: dict[str, Any]) -> list[str]:
    warnings = []
    text = chunk.get("text", "")
    compact_text = compact_for_match(text)
    for kind in ("entities", "relations"):
        for idx, item in enumerate(result.get(kind, []) or []):
            snippet = str(item.get("source_snippet") or "").strip()
            compact_snippet = compact_for_match(snippet)
            if compact_snippet and len(compact_snippet) >= 8 and compact_snippet not in compact_text:
                warnings.append(f"{kind}[{idx}] source_snippet not found in chunk")
            if kind == "relations" and item.get("type") in {"CAUSED_BY", "LEADS_TO"} and compact_snippet:
                source_name = compact_for_match(str(item.get("source_name") or ""))
                target_name = compact_for_match(str(item.get("target_name") or ""))
                if len(source_name) >= 4 and source_name not in compact_snippet:
                    warnings.append(f"causal_relation[{idx}] source endpoint not supported by snippet")
                if len(target_name) >= 4 and target_name not in compact_snippet:
                    warnings.append(f"causal_relation[{idx}] target endpoint not supported by snippet")
    return warnings


def compact_result_for_prompt(result: dict[str, Any]) -> dict[str, Any]:
    compact = {"entities": [], "relations": []}
    for ent in result.get("entities", []) or []:
        props = dict(ent.get("properties") or {})
        for key in ("source_doc", "vehicle_brand", "vehicle_model", "entity_id"):
            props.pop(key, None)
        item = {
            "type": ent.get("type"),
            "name": ent.get("name"),
            "properties": {k: v for k, v in props.items() if v not in ("", None, [], {})},
        }
        snippet = ent.get("source_snippet")
        if snippet:
            item["source_snippet"] = snippet
        compact["entities"].append(item)
    for rel in result.get("relations", []) or []:
        props = dict(rel.get("properties") or {})
        for key in ("source_doc", "vehicle_brand", "vehicle_model"):
            props.pop(key, None)
        item = {
            "type": rel.get("type"),
            "source_type": rel.get("source_type"),
            "source_name": rel.get("source_name"),
            "target_type": rel.get("target_type"),
            "target_name": rel.get("target_name"),
        }
        if props:
            item["properties"] = {k: v for k, v in props.items() if v not in ("", None, [], {})}
        snippet = rel.get("source_snippet")
        if snippet:
            item["source_snippet"] = snippet
        compact["relations"].append(item)
    return compact


def risk_reasons(
    draft: dict[str, Any],
    code_errors: list[str],
    code_warnings: list[str],
    extract_usage: dict[str, Any],
    chunk: dict[str, Any],
    extract_max_tokens: int,
) -> list[str]:
    reasons = []
    entities = draft.get("entities", [])
    relations = draft.get("relations", [])
    completion = usage_completion_tokens(extract_usage)
    if code_errors:
        reasons.append("schema_code_errors")
    if near_token_limit(extract_usage, extract_max_tokens):
        reasons.append(f"completion_tokens_near_limit completion={completion} max_tokens={extract_max_tokens}")
    tr_count = str(chunk.get("text", "")).count("<tr>")
    td_count = str(chunk.get("text", "")).count("<td")
    if tr_count > 20 or td_count > 80:
        reasons.append(f"large_table tr={tr_count} td={td_count}")
    if not entities and chunk.get("char_count", 0) > 250:
        reasons.append("no_entities")
    over_reason = over_extracted_reason(draft, chunk)
    if over_reason:
        reasons.append(over_reason)
    if entities and len(relations) > max(30, len(entities) * 2.5):
        reasons.append(f"relation_entity_ratio relations={len(relations)} entities={len(entities)}")
    evidence_warnings = source_evidence_warnings(draft, chunk)
    if any(w.startswith("causal_relation[") for w in evidence_warnings):
        reasons.append("causal_relation_evidence_suspicious")
    if len(evidence_warnings) >= 3:
        reasons.append("source_evidence_suspicious")
    return reasons


def extraction_format_errors(result: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(result, dict):
        return ["extract result must be a JSON object"]
    if "entities" not in result:
        errors.append("missing top-level field: entities")
    elif not isinstance(result.get("entities"), list):
        errors.append("top-level field entities must be a list")
    if "relations" not in result:
        errors.append("missing top-level field: relations")
    elif not isinstance(result.get("relations"), list):
        errors.append("top-level field relations must be a list")
    return errors


def review_format_errors(result: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(result, dict):
        return ["review result must be a JSON object"]
    action_tag = result.get("action_tag")
    if action_tag not in {"pass", "minor_accept", "supplement", "corrected_full", "repair_required"}:
        errors.append("review field action_tag must be pass/minor_accept/supplement/corrected_full/repair_required")
    if not isinstance(result.get("passed"), bool):
        errors.append("review field passed must be boolean")
    quality_score = result.get("quality_score")
    if not isinstance(quality_score, int) or not 0 <= quality_score <= 100:
        errors.append("review field quality_score must be integer 0-100")
    severity = result.get("severity")
    if severity not in {"pass", "minor", "major"}:
        errors.append("review field severity must be pass/minor/major")
    issue_tag = result.get("issue_tag")
    if not isinstance(issue_tag, str) or not issue_tag:
        errors.append("review field issue_tag must be a non-empty string")
    reasons = result.get("reasons")
    if reasons is not None and not isinstance(reasons, list):
        errors.append("review field reasons must be a list when present")
    feedback = result.get("feedback_for_extractor")
    if feedback is not None and not isinstance(feedback, str):
        errors.append("review field feedback_for_extractor must be a string when present")
    supplement = result.get("supplement_output")
    if supplement is not None and not isinstance(supplement, dict):
        errors.append("review field supplement_output must be null or an object")
    corrected = result.get("corrected_output")
    if corrected is not None and not isinstance(corrected, dict):
        errors.append("review field corrected_output must be null or an object")
    if action_tag == "supplement" and supplement is None:
        errors.append("supplement action requires supplement_output")
    if action_tag == "corrected_full" and corrected is None:
        errors.append("corrected_full action requires corrected_output")
    if action_tag in {"pass", "minor_accept"} and (supplement is not None or corrected is not None):
        errors.append("pass/minor_accept action must not include supplement_output or corrected_output")
    return errors


def parse_json_response(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as direct_error:
        first_error = direct_error

    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        fenced = m.group(1).strip()
        try:
            return json.loads(fenced)
        except json.JSONDecodeError as fenced_error:
            first_error = fenced_error

    # If the response itself starts like JSON but direct parsing failed, it is
    # usually truncated. Do not salvage a nested entity object from a broken root.
    if text.startswith(("{", "[")):
        raise first_error

    start = text.find("{")
    while start != -1:
        end = find_balanced_json_object_end(text, start)
        if end != -1:
            try:
                candidate = json.loads(text[start : end + 1])
                if isinstance(candidate, dict) and (
                    {"entities", "relations"} & set(candidate)
                    or {"passed", "issue_tag", "corrected_output"} & set(candidate)
                ):
                    return candidate
            except json.JSONDecodeError as balanced_error:
                first_error = balanced_error
        start = text.find("{", start + 1)
    raise first_error


def coerce_extraction_output(result: Any) -> Any:
    """Recover a few common non-fatal extraction shapes.

    The model occasionally returns a single entity/relation object or a bare
    list. Old extraction code tolerated partial cleaned results; this keeps the
    new pipeline from wasting retries on recoverable wrapping mistakes.
    """
    if isinstance(result, dict) and ("entities" in result or "relations" in result):
        return {"entities": result.get("entities") or [], "relations": result.get("relations") or []}
    if isinstance(result, dict):
        if {"type", "source_type", "source_name", "target_type", "target_name"}.issubset(result):
            return {"entities": [], "relations": [result]}
        if "type" in result and ("name" in result or "properties" in result):
            return {"entities": [result], "relations": []}
    if isinstance(result, list):
        entities: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        for item in result:
            if not isinstance(item, dict):
                continue
            if {"type", "source_type", "source_name", "target_type", "target_name"}.issubset(item):
                relations.append(item)
            elif "type" in item and ("name" in item or "properties" in item):
                entities.append(item)
        if entities or relations:
            return {"entities": entities, "relations": relations}
    return result


def find_balanced_json_object_end(text: str, start: int) -> int:
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def call_llm(stage: str, messages: list[dict[str, str]], max_tokens: int) -> tuple[str, dict[str, Any]]:
    last_error = ""
    fp = key_fingerprint(LLM_API_KEY)
    for attempt in range(API_RETRIES):
        started = time.time()
        try:
            resp = get_client().chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": "disabled"}},
            )
            if not getattr(resp, "choices", None):
                raise RuntimeError(f"LLM response missing choices; response_type={type(resp).__name__}")
            usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
            usage["finish_reason"] = getattr(resp.choices[0], "finish_reason", None)
            logger.info(
                "LLM call ok stage=%s key_fp=%s model=%s max_tokens=%s thinking=disabled duration=%.1fs usage=%s",
                stage,
                fp,
                MODEL,
                max_tokens,
                time.time() - started,
                usage,
            )
            content = resp.choices[0].message.content or ""
            return content, usage
        except Exception as exc:
            last_error = str(exc)
            if "401" in last_error or "unauthorized" in last_error.lower() or "令牌状态不可用" in last_error:
                logger.error(
                    "LLM 401 stage=%s key_fp=%s model=%s max_tokens=%s thinking=disabled; fail fast. error=%s",
                    stage,
                    fp,
                    MODEL,
                    max_tokens,
                    last_error[:240],
                )
                raise RuntimeError(last_error) from exc
            wait = min(90, (2**attempt) * 3 + random.random() * 3)
            if "429" in last_error or "rate" in last_error.lower():
                wait = min(180, (2**attempt) * 8 + random.random() * 5)
            logger.warning(
                "LLM call failed stage=%s key_fp=%s attempt=%s wait=%.1fs error=%s",
                stage,
                fp,
                attempt + 1,
                wait,
                last_error[:240],
            )
            time.sleep(wait)
    raise RuntimeError(last_error)


def build_extract_messages(context: dict[str, Any], chunk: dict[str, Any], feedback: str = "") -> list[dict[str, str]]:
    few_shots = read_few_shots()
    if chunk.get("is_flat_chunk"):
        context_block = ""
        extract_heading = "## 待抽取 chunk / 唯一抽取区"
        scope_note = (
            "当前是 flat chunk 模式：下面整段 chunk text 都是唯一抽取区，可能包含多个 heading/小节；"
            "heading_path/source_section 只是摘要，不是抽取边界。"
        )
    else:
        context_block = f"""## 大块上下文 / 禁止抽取区
下面内容只用于统一实体命名、别名、章节语义和指代消歧。
严禁从下面内容抽取实体、关系、source_snippet 或 evidence。
context_id: {context["context_id"]}
heading_paths: {" || ".join(context.get("heading_paths") or [])}
line_range: {context["line_start"]}-{context["line_end"]}

{context["text"]}

"""
        extract_heading = "## 待抽取小块 / 唯一抽取区"
        scope_note = "只有待抽取小块可以作为事实来源；大块上下文只是命名和消歧背景。"
    user = f"""## 参考 few-shot
{few_shots}

## 当前文档范围
source_doc: {chunk["doc_name"]}
vehicle_brand: {chunk["vehicle_brand"]}
vehicle_model: {chunk["vehicle_model"]}

{context_block}{extract_heading}
{scope_note}
只能从下面内容抽取实体和关系。所有 source_snippet/evidence 必须来自下面内容。
chunk_id: {chunk["chunk_id"]}
heading_path: {chunk.get("heading_path", "")}
source_section: {chunk.get("source_section", "")}
line_range: {chunk.get("line_start")}-{chunk.get("line_end")}

{chunk["text"]}
"""
    if feedback:
        user += f"\n\n## 上轮 reviewer 反馈（本次补抽必须修正）\n{feedback}\n"
    user += (
        "\n再次强调：只抽取“唯一抽取区”中的事实；flat chunk 模式下整个 chunk text 都是唯一抽取区。"
        "只输出一个合法 JSON 对象，不要输出解释、Markdown 或多个分散对象。"
    )
    return [{"role": "system", "content": SCHEMA_PROMPT}, {"role": "user", "content": user}]


def build_review_messages(
    context: dict[str, Any],
    chunk: dict[str, Any],
    draft: dict[str, Any],
    code_errors: list[str],
    code_warnings: list[str],
    risk_reasons_for_review: list[str] | None = None,
) -> list[dict[str, str]]:
    few_shots = read_few_shots()
    if chunk.get("is_flat_chunk"):
        context_block = ""
        extract_heading = "## 待抽取 chunk / 唯一抽取区"
        scope_note = (
            "当前是 flat chunk 模式：下面整段 chunk text 都是唯一抽取区，可能包含多个 heading/小节；"
            "heading_path/source_section 只是摘要，不是抽取边界。"
            "review 时不得把同一 chunk text 内的其他小节误判为大块上下文。"
        )
    else:
        context_block = f"""## 大块上下文 / 禁止抽取区
下面内容只用于理解、命名统一和实体消歧。review 时不能要求补充只在这里出现、但未在待抽取小块出现的事实。
context_id: {context["context_id"]}
heading_paths: {" || ".join(context.get("heading_paths") or [])}

{context["text"]}

"""
        extract_heading = "## 待抽取小块 / 唯一抽取区"
        scope_note = "只有待抽取小块可以作为事实来源；大块上下文只是命名和消歧背景。"
    user = f"""## 参考 few-shot
{few_shots}

## 当前文档范围
source_doc: {chunk["doc_name"]}
vehicle_brand: {chunk["vehicle_brand"]}
vehicle_model: {chunk["vehicle_model"]}

{context_block}{extract_heading}
{scope_note}
只有下面内容可以作为抽取事实、source_snippet 和 evidence 的依据。
chunk_id: {chunk["chunk_id"]}
heading_path: {chunk.get("heading_path", "")}
line_range: {chunk.get("line_start")}-{chunk.get("line_end")}

{chunk["text"]}

## 抽取草稿
{json.dumps(compact_result_for_prompt(draft), ensure_ascii=False, indent=2)}

## 程序校验信息
errors: {json.dumps(code_errors, ensure_ascii=False)}
warnings: {json.dumps(code_warnings, ensure_ascii=False)}
risk_reasons: {json.dumps(risk_reasons_for_review or [], ensure_ascii=False)}

请 review。只有“唯一抽取区”可以作为事实依据；flat chunk 模式下整个 chunk text 都属于唯一抽取区。
必须优先使用 pass、minor_accept、supplement 或 corrected_full 在 review 阶段解决问题；
只有抽取主体严重错乱且无法修正时才输出 repair_required。
只输出 JSON，且 action_tag 必须放在顶层第一个字段。
"""
    return [{"role": "system", "content": REVIEW_PROMPT}, {"role": "user", "content": user}]


def process_chunk_impl(
    run_dir: Path,
    context: dict[str, Any],
    chunk: dict[str, Any],
    force: bool = False,
    max_tokens_extract: int = MAX_TOKENS_EXTRACT,
    max_tokens_review: int = MAX_TOKENS_REVIEW,
    review_policy: str = "auto",
) -> dict[str, Any]:
    scope = {"vehicle_brand": chunk["vehicle_brand"], "vehicle_model": chunk["vehicle_model"]}
    final_path = run_dir / "chunk_final" / f"{chunk['chunk_id']}.final.json"
    if final_path.exists() and not force:
        data = json.loads(final_path.read_text(encoding="utf-8"))
        return {
            "status": "skipped",
            "chunk_id": chunk["chunk_id"],
            "entities": len(data.get("entities", [])),
            "relations": len(data.get("relations", [])),
        }

    feedback = ""
    last_payload: dict[str, Any] | None = None
    current_extract_max = max_tokens_extract
    current_review_max = max_tokens_review
    repair_attempts = 0
    best_candidate: dict[str, Any] | None = None

    def remember_candidate(
        result: dict[str, Any],
        status: str,
        score: int,
        review: dict[str, Any] | None,
        code_warnings_for_candidate: list[str],
        extract_usage_for_candidate: dict[str, Any],
        review_usage_for_candidate: dict[str, Any] | None,
        risk_reasons_for_candidate: list[str],
        max_tokens_for_candidate: dict[str, int],
    ) -> None:
        nonlocal best_candidate
        if not result.get("entities"):
            return
        candidate = {
            **result,
            "chunk_id": chunk["chunk_id"],
            "status": status,
            "review_policy": review_policy,
            "risk_reasons": risk_reasons_for_candidate,
            "review": review,
            "quality_score": score,
            "code_warnings": code_warnings_for_candidate,
            "usage": {"extract": extract_usage_for_candidate},
            "max_tokens_used": max_tokens_for_candidate,
        }
        if review_usage_for_candidate is not None:
            candidate["usage"]["review"] = review_usage_for_candidate
        if best_candidate is None or score > int(best_candidate.get("quality_score") or -1):
            best_candidate = candidate

    def save_best_candidate(status: str, reason: str) -> dict[str, Any] | None:
        if best_candidate is None:
            return None
        final = dict(best_candidate)
        final["status"] = status
        final["selected_after_repair_attempts"] = repair_attempts
        final["selection_reason"] = reason
        atomic_write_json(final_path, final)
        return {
            "status": final["status"],
            "chunk_id": chunk["chunk_id"],
            "entities": len(final["entities"]),
            "relations": len(final["relations"]),
        }

    def should_stop_repair() -> dict[str, Any] | None:
        nonlocal repair_attempts
        repair_attempts += 1
        if repair_attempts < MAX_REPAIR_ATTEMPTS:
            return None
        saved = save_best_candidate("final_best_legal_after_retries", "max_repair_attempts_reached")
        if saved is not None:
            return saved
        deferred = {
            "chunk_id": chunk["chunk_id"],
            "doc_name": chunk["doc_name"],
            "status": "deferred_repair_no_legal_candidate",
            "review_policy": review_policy,
            "repair_attempts": repair_attempts,
            "last_payload": last_payload,
        }
        atomic_write_json(run_dir / "failed" / f"{chunk['chunk_id']}.failed.json", deferred)
        atomic_write_json(OUTPUT_DIR / "failed" / f"{chunk['chunk_id']}.failed.json", deferred)
        return {
            "status": "deferred_repair_no_legal_candidate",
            "chunk_id": chunk["chunk_id"],
            "entities": 0,
            "relations": 0,
        }

    for cycle in range(1, REVIEW_CYCLES + 1):
        extract_messages = build_extract_messages(context, chunk, feedback)
        extract_content, extract_usage = call_llm("extract", extract_messages, current_extract_max)
        try:
            extract_raw = parse_json_response(extract_content)
        except Exception as exc:
            invalid_payload = {
                "chunk_id": chunk["chunk_id"],
                "cycle": cycle,
                "model": MODEL,
                "max_tokens": current_extract_max,
                "usage": extract_usage,
                "format_errors": [f"extract JSON parse failed: {exc}"],
                "raw_content": extract_content,
            }
            atomic_write_json(
                run_dir / "chunk_extract" / f"{chunk['chunk_id']}.cycle{cycle}.extract_invalid.json", invalid_payload
            )
            upgraded = (
                next_token_limit(current_extract_max) if near_token_limit(extract_usage, current_extract_max) else None
            )
            if upgraded:
                feedback = token_limit_feedback("extract", extract_usage, current_extract_max, upgraded)
                current_extract_max = upgraded
                last_payload = invalid_payload
                continue
            feedback = json.dumps(
                {
                    "issue_tag": "bad_schema",
                    "severity": "major",
                    "reasons": invalid_payload["format_errors"],
                    "feedback_for_extractor": (
                        "上轮输出不是合法 JSON，必须只输出包含 entities 和 relations 两个数组字段的 JSON 对象。"
                    ),
                    "previous_extract_file": str(
                        run_dir / "chunk_extract" / f"{chunk['chunk_id']}.cycle{cycle}.extract_invalid.json"
                    ),
                },
                ensure_ascii=False,
            )
            last_payload = invalid_payload
            continue
        extract_raw = coerce_extraction_output(extract_raw)
        extract_model, extract_fmt_errors = validate_extraction_pydantic(extract_raw)
        if extract_model is not None:
            extract_raw = extract_model
        else:
            extract_fmt_errors = [f"pydantic extraction validation failed: {msg}" for msg in extract_fmt_errors]
        extract_fmt_errors.extend(extraction_format_errors(extract_raw))
        if extract_fmt_errors:
            invalid_payload = {
                "chunk_id": chunk["chunk_id"],
                "cycle": cycle,
                "model": MODEL,
                "max_tokens": current_extract_max,
                "usage": extract_usage,
                "format_errors": extract_fmt_errors,
                "raw": extract_raw,
            }
            atomic_write_json(
                run_dir / "chunk_extract" / f"{chunk['chunk_id']}.cycle{cycle}.extract_invalid.json", invalid_payload
            )
            upgraded = (
                next_token_limit(current_extract_max) if near_token_limit(extract_usage, current_extract_max) else None
            )
            if upgraded:
                feedback = token_limit_feedback("extract", extract_usage, current_extract_max, upgraded)
                current_extract_max = upgraded
                last_payload = invalid_payload
                continue
            feedback = json.dumps(
                {
                    "issue_tag": "bad_schema",
                    "severity": "major",
                    "reasons": extract_fmt_errors,
                    "feedback_for_extractor": (
                        '上轮抽取 JSON 顶层格式不符合要求。必须输出 {"entities": [...], "relations": [...]}，'
                        "两个字段都必须是数组。"
                    ),
                    "previous_extract_file": str(
                        run_dir / "chunk_extract" / f"{chunk['chunk_id']}.cycle{cycle}.extract_invalid.json"
                    ),
                },
                ensure_ascii=False,
            )
            last_payload = invalid_payload
            continue
        draft, code_errors, code_warnings = validate_and_normalize(extract_raw, chunk, scope)
        extract_payload = {
            "chunk_id": chunk["chunk_id"],
            "cycle": cycle,
            "model": MODEL,
            "max_tokens": current_extract_max,
            "usage": extract_usage,
            "raw": extract_raw,
            "normalized": draft,
            "code_errors": code_errors,
            "code_warnings": code_warnings,
        }
        atomic_write_json(run_dir / "chunk_extract" / f"{chunk['chunk_id']}.cycle{cycle}.extract.json", extract_payload)

        upgraded = (
            next_token_limit(current_extract_max) if near_token_limit(extract_usage, current_extract_max) else None
        )
        if upgraded:
            feedback = token_limit_feedback("extract", extract_usage, current_extract_max, upgraded)
            current_extract_max = upgraded
            last_payload = {
                "draft": draft,
                "code_errors": code_errors,
                "code_warnings": code_warnings,
                "usage": extract_usage,
            }
            continue

        reasons_for_review = risk_reasons(draft, code_errors, code_warnings, extract_usage, chunk, current_extract_max)
        if review_policy == "always":
            reasons_for_review = ["review_policy_always", *reasons_for_review]
        if review_policy == "never":
            if code_errors:
                failed = {
                    "chunk_id": chunk["chunk_id"],
                    "doc_name": chunk["doc_name"],
                    "status": "failed_extract_validation",
                    "review_policy": review_policy,
                    "code_errors": code_errors,
                    "code_warnings": code_warnings,
                    "usage": {"extract": extract_usage},
                    "max_tokens_used": {"extract": current_extract_max},
                }
                atomic_write_json(run_dir / "failed" / f"{chunk['chunk_id']}.failed.json", failed)
                atomic_write_json(OUTPUT_DIR / "failed" / f"{chunk['chunk_id']}.failed.json", failed)
                return {
                    "status": "failed_extract_validation",
                    "chunk_id": chunk["chunk_id"],
                    "entities": 0,
                    "relations": 0,
                }
            final = {
                **draft,
                "chunk_id": chunk["chunk_id"],
                "status": "final_extract_passed",
                "review_policy": review_policy,
                "risk_reasons": reasons_for_review,
                "code_warnings": code_warnings,
                "usage": {"extract": extract_usage},
                "max_tokens_used": {"extract": current_extract_max},
            }
            atomic_write_json(final_path, final)
            return {
                "status": final["status"],
                "chunk_id": chunk["chunk_id"],
                "entities": len(final["entities"]),
                "relations": len(final["relations"]),
            }

        review_messages = build_review_messages(context, chunk, draft, code_errors, code_warnings, reasons_for_review)
        review_content, review_usage = call_llm("review", review_messages, current_review_max)
        try:
            review_raw = parse_json_response(review_content)
        except Exception as exc:
            invalid_review_payload = {
                "chunk_id": chunk["chunk_id"],
                "cycle": cycle,
                "model": MODEL,
                "max_tokens": current_review_max,
                "usage": review_usage,
                "format_errors": [f"review JSON parse failed: {exc}"],
                "raw_content": review_content,
            }
            atomic_write_json(
                run_dir / "chunk_review" / f"{chunk['chunk_id']}.cycle{cycle}.review_invalid.json",
                invalid_review_payload,
            )
            upgraded = (
                next_token_limit(current_review_max) if near_token_limit(review_usage, current_review_max) else None
            )
            if upgraded:
                feedback = token_limit_feedback("review", review_usage, current_review_max, upgraded)
                current_review_max = upgraded
                last_payload = {
                    "draft": draft,
                    "review": invalid_review_payload,
                    "code_errors": code_errors,
                    "code_warnings": code_warnings,
                }
                continue
            feedback = json.dumps(
                {
                    "issue_tag": "bad_schema",
                    "severity": "major",
                    "reasons": invalid_review_payload["format_errors"],
                    "feedback_for_extractor": (
                        "reviewer 上轮输出不是合法 JSON，本轮请重新抽取并确保抽取 JSON 严格符合 schema。"
                    ),
                    "previous_extract_summary": result_summary(draft),
                    "previous_review_file": str(
                        run_dir / "chunk_review" / f"{chunk['chunk_id']}.cycle{cycle}.review_invalid.json"
                    ),
                },
                ensure_ascii=False,
            )
            last_payload = {
                "draft": draft,
                "review": invalid_review_payload,
                "code_errors": code_errors,
                "code_warnings": code_warnings,
            }
            continue
        review_model, review_fmt_errors = validate_review_pydantic(review_raw)
        if review_model is not None:
            review_raw = review_model
        else:
            review_fmt_errors = [f"pydantic review validation failed: {msg}" for msg in review_fmt_errors]
        review_fmt_errors.extend(review_format_errors(review_raw))
        if review_fmt_errors:
            invalid_review_payload = {
                "chunk_id": chunk["chunk_id"],
                "cycle": cycle,
                "model": MODEL,
                "max_tokens": current_review_max,
                "usage": review_usage,
                "format_errors": review_fmt_errors,
                "review": review_raw,
            }
            atomic_write_json(
                run_dir / "chunk_review" / f"{chunk['chunk_id']}.cycle{cycle}.review_invalid.json",
                invalid_review_payload,
            )
            upgraded = (
                next_token_limit(current_review_max) if near_token_limit(review_usage, current_review_max) else None
            )
            if upgraded:
                feedback = token_limit_feedback("review", review_usage, current_review_max, upgraded)
                current_review_max = upgraded
                last_payload = {
                    "draft": draft,
                    "review": invalid_review_payload,
                    "code_errors": code_errors,
                    "code_warnings": code_warnings,
                }
                continue
            feedback = json.dumps(
                {
                    "issue_tag": "bad_schema",
                    "severity": "major",
                    "reasons": review_fmt_errors,
                    "feedback_for_extractor": (
                        "reviewer 上轮格式不符合要求，本轮请重新抽取；抽取输出仍必须是标准 entities/relations JSON。"
                    ),
                    "previous_extract_summary": result_summary(draft),
                    "previous_review_summary": review_summary(review_raw),
                    "previous_review_file": str(
                        run_dir / "chunk_review" / f"{chunk['chunk_id']}.cycle{cycle}.review_invalid.json"
                    ),
                },
                ensure_ascii=False,
            )
            last_payload = {
                "draft": draft,
                "review": invalid_review_payload,
                "code_errors": code_errors,
                "code_warnings": code_warnings,
            }
            continue
        review_payload = {
            "chunk_id": chunk["chunk_id"],
            "cycle": cycle,
            "model": MODEL,
            "max_tokens": current_review_max,
            "usage": review_usage,
            "review": review_raw,
        }
        atomic_write_json(run_dir / "chunk_review" / f"{chunk['chunk_id']}.cycle{cycle}.review.json", review_payload)
        upgraded = next_token_limit(current_review_max) if near_token_limit(review_usage, current_review_max) else None
        if upgraded:
            feedback = token_limit_feedback("review", review_usage, current_review_max, upgraded)
            current_review_max = upgraded
            last_payload = {
                "draft": draft,
                "review": review_raw,
                "code_errors": code_errors,
                "code_warnings": code_warnings,
            }
            continue

        action_tag = str(review_raw.get("action_tag") or "") if isinstance(review_raw, dict) else ""
        passed = bool(review_raw.get("passed")) if isinstance(review_raw, dict) else False
        quality_score = int(review_raw.get("quality_score") or 0) if isinstance(review_raw, dict) else 0
        severity = str(review_raw.get("severity") or "") if isinstance(review_raw, dict) else ""
        if not code_errors:
            remember_candidate(
                draft,
                "final_review_candidate",
                quality_score,
                review_raw,
                code_warnings,
                extract_usage,
                review_usage,
                reasons_for_review,
                {"extract": current_extract_max, "review": current_review_max},
            )

        if action_tag == "pass" and passed and quality_score >= DIRECT_PASS_SCORE and not code_errors:
            final_status = "final_review_passed"
        elif action_tag == "minor_accept" and passed and quality_score >= MIN_ACCEPT_SCORE and not code_errors:
            final_status = "final_review_minor_accepted"
        else:
            final_status = ""
        if final_status:
            final = {
                **draft,
                "chunk_id": chunk["chunk_id"],
                "status": final_status,
                "review_policy": review_policy,
                "risk_reasons": reasons_for_review,
                "review": review_raw,
                "quality_score": quality_score,
                "code_warnings": code_warnings,
                "usage": {"extract": extract_usage, "review": review_usage},
                "max_tokens_used": {"extract": current_extract_max, "review": current_review_max},
            }
            atomic_write_json(final_path, final)
            return {
                "status": final["status"],
                "chunk_id": chunk["chunk_id"],
                "entities": len(final["entities"]),
                "relations": len(final["relations"]),
            }

        if action_tag == "supplement" and passed and quality_score >= MIN_ACCEPT_SCORE:
            supplement_norm, supplement_errors, supplement_warnings = normalize_review_extraction_output(
                review_raw.get("supplement_output"), chunk, scope, "supplement_output"
            )
            if not supplement_errors:
                merged = merge_normalized_results(draft, supplement_norm)
                remember_candidate(
                    merged,
                    "final_review_supplemented",
                    quality_score,
                    review_raw,
                    [*code_warnings, *supplement_warnings],
                    extract_usage,
                    review_usage,
                    reasons_for_review,
                    {"extract": current_extract_max, "review": current_review_max},
                )
                final = {
                    **merged,
                    "chunk_id": chunk["chunk_id"],
                    "status": "final_review_supplemented",
                    "review_policy": review_policy,
                    "risk_reasons": reasons_for_review,
                    "review": review_raw,
                    "quality_score": quality_score,
                    "code_warnings": [*code_warnings, *supplement_warnings],
                    "usage": {"extract": extract_usage, "review": review_usage},
                    "max_tokens_used": {"extract": current_extract_max, "review": current_review_max},
                }
                atomic_write_json(final_path, final)
                return {
                    "status": final["status"],
                    "chunk_id": chunk["chunk_id"],
                    "entities": len(final["entities"]),
                    "relations": len(final["relations"]),
                }
            feedback = json.dumps(
                {
                    "issue_tag": "bad_review_supplement",
                    "severity": "major",
                    "reasons": supplement_errors,
                    "feedback_for_extractor": (
                        "reviewer supplement_output 未通过校验。请重新抽取，优先输出一份完整且简洁的合法 JSON。"
                    ),
                    "previous_extract_summary": result_summary(draft),
                    "previous_review_summary": review_summary(review_raw),
                },
                ensure_ascii=False,
            )
            last_payload = {
                "draft": draft,
                "review": review_raw,
                "supplement_errors": supplement_errors,
                "code_errors": code_errors,
                "code_warnings": code_warnings,
            }
            stop = should_stop_repair()
            if stop is not None:
                return stop
            continue

        if action_tag == "corrected_full" and passed and quality_score >= MIN_ACCEPT_SCORE:
            corrected_norm, corrected_errors, corrected_warnings = normalize_review_extraction_output(
                review_raw.get("corrected_output"), chunk, scope, "corrected_output"
            )
            if not corrected_errors:
                remember_candidate(
                    corrected_norm,
                    "final_review_corrected",
                    quality_score,
                    review_raw,
                    corrected_warnings,
                    extract_usage,
                    review_usage,
                    reasons_for_review,
                    {"extract": current_extract_max, "review": current_review_max},
                )
                final = {
                    **corrected_norm,
                    "chunk_id": chunk["chunk_id"],
                    "status": "final_review_corrected",
                    "review_policy": review_policy,
                    "risk_reasons": reasons_for_review,
                    "review": review_raw,
                    "quality_score": quality_score,
                    "code_warnings": corrected_warnings,
                    "usage": {"extract": extract_usage, "review": review_usage},
                    "max_tokens_used": {"extract": current_extract_max, "review": current_review_max},
                }
                atomic_write_json(final_path, final)
                return {
                    "status": final["status"],
                    "chunk_id": chunk["chunk_id"],
                    "entities": len(final["entities"]),
                    "relations": len(final["relations"]),
                }
            if can_save_partial_review_output(corrected_norm, corrected_errors, quality_score):
                remember_candidate(
                    corrected_norm,
                    "final_review_corrected_partial",
                    quality_score,
                    review_raw,
                    [*corrected_warnings, *corrected_errors],
                    extract_usage,
                    review_usage,
                    reasons_for_review,
                    {"extract": current_extract_max, "review": current_review_max},
                )
                final = {
                    **corrected_norm,
                    "chunk_id": chunk["chunk_id"],
                    "status": "final_review_corrected_partial",
                    "review_policy": review_policy,
                    "risk_reasons": reasons_for_review,
                    "review": review_raw,
                    "quality_score": quality_score,
                    "code_warnings": [*corrected_warnings, *corrected_errors],
                    "usage": {"extract": extract_usage, "review": review_usage},
                    "max_tokens_used": {"extract": current_extract_max, "review": current_review_max},
                }
                atomic_write_json(final_path, final)
                return {
                    "status": final["status"],
                    "chunk_id": chunk["chunk_id"],
                    "entities": len(final["entities"]),
                    "relations": len(final["relations"]),
                }
            feedback = json.dumps(
                {
                    "issue_tag": "bad_review_correction",
                    "severity": "major",
                    "reasons": corrected_errors,
                    "feedback_for_extractor": (
                        "reviewer corrected_output 未通过校验。请重新抽取，确保关系方向、实体类型和端点完整。"
                    ),
                    "previous_extract_summary": result_summary(draft),
                    "previous_review_summary": review_summary(review_raw),
                },
                ensure_ascii=False,
            )
            last_payload = {
                "draft": draft,
                "review": review_raw,
                "corrected_errors": corrected_errors,
                "code_errors": code_errors,
                "code_warnings": code_warnings,
            }
            stop = should_stop_repair()
            if stop is not None:
                return stop
            continue

        reasons = review_raw.get("reasons") if isinstance(review_raw, dict) else []
        feedback_for_extractor = review_raw.get("feedback_for_extractor") if isinstance(review_raw, dict) else ""
        issue_tag = review_raw.get("issue_tag") if isinstance(review_raw, dict) else "other"
        if action_tag != "repair_required" and not code_errors and passed and quality_score >= MIN_ACCEPT_SCORE:
            final = {
                **draft,
                "chunk_id": chunk["chunk_id"],
                "status": "final_review_minor_accepted",
                "review_policy": review_policy,
                "risk_reasons": reasons_for_review,
                "review": review_raw,
                "quality_score": quality_score,
                "code_warnings": code_warnings,
                "usage": {"extract": extract_usage, "review": review_usage},
                "max_tokens_used": {"extract": current_extract_max, "review": current_review_max},
            }
            atomic_write_json(final_path, final)
            return {
                "status": final["status"],
                "chunk_id": chunk["chunk_id"],
                "entities": len(final["entities"]),
                "relations": len(final["relations"]),
            }
        feedback = json.dumps(
            {
                "issue_tag": issue_tag,
                "severity": severity,
                "reasons": reasons,
                "feedback_for_extractor": feedback_for_extractor,
                "code_errors": code_errors,
                "code_warnings": code_warnings,
                "previous_extract_summary": result_summary(draft),
                "previous_review_summary": review_summary(review_raw),
                "previous_extract_file": str(
                    run_dir / "chunk_extract" / f"{chunk['chunk_id']}.cycle{cycle}.extract.json"
                ),
                "previous_review_file": str(run_dir / "chunk_review" / f"{chunk['chunk_id']}.cycle{cycle}.review.json"),
            },
            ensure_ascii=False,
        )
        last_payload = {
            "draft": draft,
            "review": review_raw,
            "code_errors": code_errors,
            "code_warnings": code_warnings,
        }
        stop = should_stop_repair()
        if stop is not None:
            return stop

    saved = save_best_candidate("final_best_legal_after_retries", "review_cycles_exhausted")
    if saved is not None:
        return saved

    failed = {
        "chunk_id": chunk["chunk_id"],
        "doc_name": chunk["doc_name"],
        "status": "deferred_repair_no_legal_candidate",
        "review_policy": review_policy,
        "repair_attempts": repair_attempts,
        "last_payload": last_payload,
    }
    atomic_write_json(run_dir / "failed" / f"{chunk['chunk_id']}.failed.json", failed)
    atomic_write_json(OUTPUT_DIR / "failed" / f"{chunk['chunk_id']}.failed.json", failed)
    return {
        "status": "deferred_repair_no_legal_candidate",
        "chunk_id": chunk["chunk_id"],
        "entities": 0,
        "relations": 0,
    }


def process_chunk(
    run_dir: Path,
    context: dict[str, Any],
    chunk: dict[str, Any],
    force: bool = False,
    max_tokens_extract: int = MAX_TOKENS_EXTRACT,
    max_tokens_review: int = MAX_TOKENS_REVIEW,
    review_policy: str = "auto",
) -> dict[str, Any]:
    try:
        return process_chunk_impl(run_dir, context, chunk, force, max_tokens_extract, max_tokens_review, review_policy)
    except Exception as exc:
        failed = {
            "chunk_id": chunk["chunk_id"],
            "doc_name": chunk["doc_name"],
            "status": "exception",
            "error": str(exc),
        }
        logger.exception("chunk failed: %s", chunk["chunk_id"])
        atomic_write_json(run_dir / "failed" / f"{chunk['chunk_id']}.failed.json", failed)
        atomic_write_json(OUTPUT_DIR / "failed" / f"{chunk['chunk_id']}.failed.json", failed)
        return {"status": "failed_exception", "chunk_id": chunk["chunk_id"], "entities": 0, "relations": 0}


def prepare_doc(doc_path: Path) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    scope = infer_scope(doc_path.name)
    doc_run_id = safe_stem(doc_path.name)
    run_dir = OUTPUT_DIR / "doc_runs" / doc_run_id
    for sub in ["chunk_extract", "chunk_review", "chunk_final", "doc_raw", "state", "failed"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    text = clean_text(read_doc_text(doc_path))
    sections = split_sections(text)
    contexts = build_context_chunks(sections, doc_path.name, scope)
    chunks: list[dict[str, Any]] = []
    skipped_chunks: list[dict[str, Any]] = []
    for ctx in contexts:
        low_value_reason = low_value_chunk_reason(ctx["text"])
        if low_value_reason:
            skipped_chunks.append(
                {
                    "chunk_id": f"{ctx['context_id']}_flat",
                    "context_id": ctx["context_id"],
                    "doc_name": ctx["doc_name"],
                    "reason": low_value_reason,
                    "context_index": ctx["context_index"],
                    "line_start": ctx["line_start"],
                    "line_end": ctx["line_end"],
                    "char_count": len(ctx["text"]),
                    "heading_paths": ctx.get("heading_paths") or [],
                    "text_preview": ctx["text"][:300],
                }
            )
            continue
        heading_paths = ctx.get("heading_paths") or []
        heading_path = " || ".join(heading_paths[-4:])
        source_section = heading_paths[-1].split(" > ")[-1] if heading_paths else ""
        chunks.append(
            {
                "chunk_id": f"{ctx['context_id']}_flat",
                "context_id": ctx["context_id"],
                "doc_name": ctx["doc_name"],
                "vehicle_brand": ctx["vehicle_brand"],
                "vehicle_model": ctx["vehicle_model"],
                "context_index": ctx["context_index"],
                "chunk_index": 0,
                "heading_path": heading_path,
                "source_section": source_section,
                "line_start": ctx["line_start"],
                "line_end": ctx["line_end"],
                "text": ctx["text"],
                "char_count": len(ctx["text"]),
                "target_chars": CHUNK_TARGET_CHARS,
                "max_chars": CHUNK_MAX_CHARS,
                "overlap_chars": ctx.get("overlap_chars", 0),
                "is_flat_chunk": True,
            }
        )

    for doc_chunk_index, chunk in enumerate(chunks):
        chunk["doc_chunk_index"] = doc_chunk_index

    atomic_write_json(
        run_dir / "state" / "source_info.json",
        {
            "doc_name": doc_path.name,
            "doc_path": str(doc_path),
            "scope": scope,
            "chunk_mode": "single_flat_chunk_2000_with_overlap",
            "chunk_target_chars": CHUNK_TARGET_CHARS,
            "chunk_max_chars": CHUNK_MAX_CHARS,
            "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
            "skipped_chunk_count": len(skipped_chunks),
        },
    )
    atomic_write_json(run_dir / "state" / "context_chunks.json", contexts)
    atomic_write_json(run_dir / "state" / "flat_chunks.json", chunks)
    atomic_write_json(run_dir / "state" / "skipped_chunks.json", skipped_chunks)
    atomic_write_json(OUTPUT_DIR / "context_chunks" / f"{doc_run_id}.contexts.json", contexts)
    atomic_write_json(OUTPUT_DIR / "chunk_tasks" / f"{doc_run_id}.chunks.json", chunks)
    atomic_write_json(OUTPUT_DIR / "chunk_tasks" / f"{doc_run_id}.skipped_chunks.json", skipped_chunks)
    return run_dir, contexts, chunks, scope


def final_path_for_chunk(run_dir: Path, chunk: dict[str, Any]) -> Path:
    return run_dir / "chunk_final" / f"{chunk['chunk_id']}.final.json"


def cleanup_chunk_artifacts(run_dir: Path, chunk: dict[str, Any]) -> None:
    chunk_id = chunk["chunk_id"]
    for sub in ("chunk_extract", "chunk_review", "chunk_final", "failed"):
        for path in (run_dir / sub).glob(f"{chunk_id}*"):
            if path.is_file():
                path.unlink()
    global_failed = OUTPUT_DIR / "failed" / f"{chunk_id}.failed.json"
    if global_failed.exists():
        global_failed.unlink()


def select_chunks_for_run(
    run_dir: Path,
    chunks: list[dict[str, Any]],
    limit_chunks_per_doc: int,
    force: bool,
    start_chunk_index: int,
    chunk_indices: list[int] | None,
) -> tuple[list[dict[str, Any]], dict[str, int | list[int]]]:
    selected = chunks
    if chunk_indices:
        wanted = set(chunk_indices)
        selected = [chunk for chunk in selected if int(chunk.get("doc_chunk_index", -1)) in wanted]
    elif start_chunk_index > 0:
        selected = [chunk for chunk in selected if int(chunk.get("doc_chunk_index", -1)) >= start_chunk_index]
    if limit_chunks_per_doc:
        selected = selected[:limit_chunks_per_doc]

    selected_count = len(selected)
    skipped_existing = 0
    pending = []
    for chunk in selected:
        if force:
            cleanup_chunk_artifacts(run_dir, chunk)
            pending.append(chunk)
        else:
            if final_path_for_chunk(run_dir, chunk).exists():
                skipped_existing += 1
            else:
                pending.append(chunk)
    selected = pending

    return selected, {
        "selected_before_resume": selected_count,
        "skipped_existing_final": skipped_existing,
        "pending_to_run": len(selected),
        "start_chunk_index": start_chunk_index,
        "chunk_indices": chunk_indices or [],
    }


def merge_doc_results(
    run_dir: Path, doc_name: str, scope: dict[str, str], active_chunk_ids: set[str] | None = None
) -> dict[str, Any]:
    entities: dict[str, dict[str, Any]] = {}
    relations_seen: set[tuple[str, str, str]] = set()
    relations: list[dict[str, Any]] = []
    statuses = Counter()
    usage = Counter()

    brand_entity_id = generate_entity_id("VehicleBrand", scope["vehicle_brand"], scope["vehicle_model"])
    model_entity_id = generate_entity_id("VehicleModel", scope["vehicle_model"], scope["vehicle_model"])
    entities[brand_entity_id] = {
        "type": "VehicleBrand",
        "name": scope["vehicle_brand"],
        "aliases": [],
        "properties": {
            "brand_name": scope["vehicle_brand"],
            "source_doc": doc_name,
        },
        "entity_id": brand_entity_id,
        "auto_scope": True,
    }
    entities[model_entity_id] = {
        "type": "VehicleModel",
        "name": scope["vehicle_model"],
        "aliases": [],
        "properties": {
            "model_name": scope["vehicle_model"],
            "source_doc": doc_name,
        },
        "entity_id": model_entity_id,
        "auto_scope": True,
    }

    for final_path in sorted((run_dir / "chunk_final").glob("*.final.json")):
        data = json.loads(final_path.read_text(encoding="utf-8"))
        if active_chunk_ids is not None and data.get("chunk_id") not in active_chunk_ids:
            continue
        statuses[data.get("status", "unknown")] += 1
        for side in ("extract", "review"):
            for key, val in (data.get("usage", {}).get(side, {}) or {}).items():
                if isinstance(val, int):
                    usage[f"{side}_{key}"] += val
        for ent in data.get("entities", []):
            eid = ent.get("entity_id") or generate_entity_id(ent["type"], entity_name(ent), scope["vehicle_model"])
            ent["entity_id"] = eid
            if ent.get("type") not in {"VehicleBrand", "VehicleModel"}:
                ent.setdefault("properties", {})["vehicle_brand"] = scope["vehicle_brand"]
                ent.setdefault("properties", {})["vehicle_model"] = scope["vehicle_model"]
            if eid not in entities:
                entities[eid] = ent
            else:
                old = entities[eid]
                old_props = old.setdefault("properties", {})
                for key, value in (ent.get("properties") or {}).items():
                    if value and (not old_props.get(key) or len(str(value)) > len(str(old_props.get(key, "")))):
                        old_props[key] = value
                old.setdefault("mentions", []).append(
                    {
                        "chunk_id": ent.get("chunk_id"),
                        "source_snippet": ent.get("source_snippet"),
                        "evidence": ent.get("evidence"),
                    }
                )
        for rel in data.get("relations", []):
            props = rel.setdefault("properties", {})
            props["vehicle_brand"] = scope["vehicle_brand"]
            props["vehicle_model"] = scope["vehicle_model"]
            src = rel.get("source_entity_id") or generate_entity_id(
                rel["source_type"], rel["source_name"], scope["vehicle_model"]
            )
            tgt = rel.get("target_entity_id") or generate_entity_id(
                rel["target_type"], rel["target_name"], scope["vehicle_model"]
            )
            key = (rel["type"], src, tgt)
            if key in relations_seen:
                continue
            relations_seen.add(key)
            rel["source_entity_id"] = src
            rel["target_entity_id"] = tgt
            relations.append(rel)

    has_model_key = ("HAS_MODEL", brand_entity_id, model_entity_id)
    if has_model_key not in relations_seen:
        relations_seen.add(has_model_key)
        relations.insert(
            0,
            {
                "type": "HAS_MODEL",
                "source_type": "VehicleBrand",
                "source_name": scope["vehicle_brand"],
                "target_type": "VehicleModel",
                "target_name": scope["vehicle_model"],
                "source_entity_id": brand_entity_id,
                "target_entity_id": model_entity_id,
                "properties": {
                    "vehicle_brand": scope["vehicle_brand"],
                    "vehicle_model": scope["vehicle_model"],
                    "note": "document scope canonical vehicle model",
                },
                "auto_scope": True,
            },
        )

    for ent in entities.values():
        props = ent.setdefault("properties", {})
        if ent.get("type") == "VehicleBrand":
            ent["name"] = scope["vehicle_brand"]
            props["brand_name"] = scope["vehicle_brand"]
            props.pop("vehicle_brand", None)
            props.pop("vehicle_model", None)
        elif ent.get("type") == "VehicleModel":
            ent["name"] = scope["vehicle_model"]
            props["model_name"] = scope["vehicle_model"]
            props.pop("vehicle_brand", None)
            props.pop("vehicle_model", None)
        else:
            props["vehicle_brand"] = scope["vehicle_brand"]
            props["vehicle_model"] = scope["vehicle_model"]

    merged = {
        "source_doc": doc_name,
        "vehicle_brand": scope["vehicle_brand"],
        "vehicle_model": scope["vehicle_model"],
        "model_name": scope["vehicle_model"],
        "extract_method": "llm_api_flat_chunk_extract_auto_review",
        "llm_model": MODEL,
        "entities": list(entities.values()),
        "relations": relations,
        "stats": {
            "entities": len(entities),
            "relations": len(relations),
            "final_chunks": sum(statuses.values()),
            "statuses": dict(statuses),
            "usage": dict(usage),
        },
    }
    doc_run_id = safe_stem(doc_name)
    atomic_write_json(run_dir / "doc_raw" / f"{doc_run_id}.json", merged)
    atomic_write_json(OUTPUT_DIR / "doc_raw_results" / f"{doc_run_id}.json", merged)
    return merged


def run_doc(
    doc_path: Path,
    workers: int,
    limit_chunks_per_doc: int,
    force: bool,
    prepare_only: bool,
    max_tokens_extract: int,
    max_tokens_review: int,
    start_chunk_index: int,
    chunk_indices: list[int] | None,
    review_policy: str,
) -> dict[str, Any]:
    run_dir, contexts, chunks, scope = prepare_doc(doc_path)
    context_by_id = {ctx["context_id"]: ctx for ctx in contexts}
    all_chunks = list(chunks)
    total_chunks = len(chunks)
    if prepare_only:
        logger.info(
            "doc=%s contexts=%s chunks=%s scope=%s/%s",
            doc_path.name,
            len(contexts),
            len(chunks),
            scope["vehicle_brand"],
            scope["vehicle_model"],
        )
        return {"doc_name": doc_path.name, "prepared": True, "contexts": len(contexts), "chunks": len(chunks)}

    chunks, selection = select_chunks_for_run(
        run_dir, chunks, limit_chunks_per_doc, force, start_chunk_index, chunk_indices
    )
    logger.info(
        "doc=%s contexts=%s total_chunks=%s pending_chunks=%s skipped_existing=%s scope=%s/%s",
        doc_path.name,
        len(contexts),
        total_chunks,
        len(chunks),
        selection["skipped_existing_final"],
        scope["vehicle_brand"],
        scope["vehicle_model"],
    )

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [
            pool.submit(
                process_chunk,
                run_dir,
                context_by_id[chunk["context_id"]],
                chunk,
                force,
                max_tokens_extract,
                max_tokens_review,
                review_policy,
            )
            for chunk in chunks
        ]
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            results.append(res)
            logger.info(
                "chunk_done doc=%s chunk=%s status=%s e=%s r=%s",
                doc_path.name,
                res["chunk_id"],
                res["status"],
                res.get("entities"),
                res.get("relations"),
            )

    merged = merge_doc_results(run_dir, doc_path.name, scope, {chunk["chunk_id"] for chunk in all_chunks})
    summary = {
        "doc_name": doc_path.name,
        "contexts": len(contexts),
        "chunks": total_chunks,
        "selected_before_resume": selection["selected_before_resume"],
        "skipped_existing_final": selection["skipped_existing_final"],
        "pending_to_run": selection["pending_to_run"],
        "start_chunk_index": selection["start_chunk_index"],
        "chunk_indices": selection["chunk_indices"],
        "review_policy": review_policy,
        "chunk_statuses": dict(Counter(r["status"] for r in results)),
        "entities": len(merged["entities"]),
        "relations": len(merged["relations"]),
    }
    atomic_write_json(run_dir / "state" / "summary.json", summary)
    return summary


def select_docs(doc_filter: list[str] | None, limit_docs: int) -> list[Path]:
    docs = sorted(DOC_DIR.glob("*.md"), key=lambda p: p.name)
    if doc_filter:
        docs = [p for p in docs if any(f in p.name for f in doc_filter)]
    if limit_docs:
        docs = docs[:limit_docs]
    return docs


def write_manifest(docs: list[Path]) -> None:
    manifest = []
    scope_map = {}
    for idx, path in enumerate(docs):
        scope = infer_scope(path.name)
        scope_map[path.name] = scope
        manifest.append({"index": idx, "doc_name": path.name, "path": str(path), **scope})
    atomic_write_json(OUTPUT_DIR / "manifests" / "run_manifest.json", manifest)
    atomic_write_json(OUTPUT_DIR / "manifests" / "vehicle_scope_map.json", scope_map)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run car_graph_pipeline llm_api extraction with LLM review.")
    parser.add_argument("--doc", action="append", help="filename substring filter, can be repeated")
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--limit-chunks-per-doc", type=int, default=0)
    parser.add_argument(
        "--start-chunk-index", type=int, default=0, help="doc-level zero-based chunk index to start from"
    )
    parser.add_argument(
        "--chunk-index", action="append", type=int, help="run only the given doc-level chunk index; can be repeated"
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-tokens-extract", type=int, default=MAX_TOKENS_EXTRACT)
    parser.add_argument("--max-tokens-review", type=int, default=MAX_TOKENS_REVIEW)
    parser.add_argument(
        "--review-policy",
        choices=["auto", "always", "never"],
        default="auto",
        help="auto reviews only risky chunks; always reviews every valid extract; never skips LLM review",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-run chunks even when final exists")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.prepare_only and importlib.util.find_spec("openai") is None:
        logger.error(
            "Missing dependency: openai. Install with: python3 -m pip install -r "
            "car_graph_pipeline/extraction/llm_api/requirements.txt"
        )
        return 2
    for sub in [
        "context_chunks",
        "chunk_tasks",
        "doc_runs",
        "doc_raw_results",
        "validated_results",
        "failed",
        "logs",
        "manifests",
    ]:
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    docs = select_docs(args.doc, args.limit_docs)
    write_manifest(docs)
    logger.info(
        "selected_docs=%s workers=%s prepare_only=%s review_policy=%s",
        len(docs),
        args.workers,
        args.prepare_only,
        args.review_policy,
    )
    all_summaries = []
    for idx, doc_path in enumerate(docs, start=1):
        logger.info("[%s/%s] start %s", idx, len(docs), doc_path.name)
        try:
            summary = run_doc(
                doc_path,
                workers=args.workers,
                limit_chunks_per_doc=args.limit_chunks_per_doc,
                force=args.force,
                prepare_only=args.prepare_only,
                max_tokens_extract=args.max_tokens_extract,
                max_tokens_review=args.max_tokens_review,
                start_chunk_index=args.start_chunk_index,
                chunk_indices=args.chunk_index,
                review_policy=args.review_policy,
            )
            all_summaries.append(summary)
            atomic_write_json(OUTPUT_DIR / "manifests" / "latest_progress.json", all_summaries)
        except Exception as exc:
            logger.exception("doc failed: %s", doc_path.name)
            atomic_write_json(
                OUTPUT_DIR / "failed" / f"{safe_stem(doc_path.name)}.doc_failed.json",
                {"doc_name": doc_path.name, "error": str(exc)},
            )
    atomic_write_json(OUTPUT_DIR / "manifests" / "latest_summary.json", all_summaries)
    logger.info("done docs=%s", len(all_summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
