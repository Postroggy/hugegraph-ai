"""Phase 2: LLM 判断合并决策 — 批量调用 DeepSeek-V4-Flash (openai SDK + pydantic 验证)。"""

import asyncio
import json
import logging
import re
from collections import defaultdict

from openai import AsyncOpenAI
from pydantic import BaseModel, field_validator

from car_graph_pipeline.config import LLM_API_KEY

from .settings import DisambiguationContext, make_context

logger = logging.getLogger(__name__)


# === Pydantic 模型用于验证 LLM 输出格式 ===


class MergeDecisionItem(BaseModel):
    """LLM 返回的单个合并决策。"""

    pair_index: int
    decision: str
    winner_id: str | None = None
    property_merge: str = "union"
    reason: str = ""

    @field_validator("decision")
    @classmethod
    def validate_decision(cls, v):
        if v not in ("merge", "keep_separate"):
            raise ValueError(f"decision 必须是 'merge' 或 'keep_separate', got '{v}'")
        return v

    @field_validator("property_merge")
    @classmethod
    def validate_property_merge(cls, v):
        if v not in ("union", "winner_priority"):
            return "union"  # 容错：非法值降级为 union
        return v


# Schema 类型说明（给 LLM 提供上下文）
TYPE_SCHEMA = {
    "Function": (
        "车辆功能（如座椅按摩、车道偏离预警）。"
        "属性: func_name, function_type, trigger_condition, alert_method, warnings"
    ),
    "Operation": "操作步骤（如通过按钮开启座椅按摩）。属性: op_name, steps(操作步骤列表), prerequisites(前置条件)",
    "Component": "车辆部件（如座椅按摩按钮、水温报警灯）。属性: comp_name, component_type, location",
    "Status": (
        "车辆状态（如警告灯亮起、功能已开启）。"
        "属性: status_name, status_type(正常/报警/异常), perceivable_way(感知方式)"
    ),
    "Fault": "故障（如轮胎泄气、冷却液不足）。属性: fault_name, severity, fault_type",
    "VehicleSystem": "车辆系统（如冷却系统、智能互联系统）。属性: system_name, system_type",
    "Specification": "技术规格（如机油容量、轮胎气压）。属性: spec_name, spec_type, value, unit",
    "MaintenanceItem": "保养项目（如机油更换、轮胎检查）。属性: maint_name, interval, warnings",
    "Material": "材料/耗材（如5W-30全合成机油）。属性: material_name, material_type, specification",
}

SYSTEM_PROMPT = """你是知识图谱实体消歧专家。你的任务是判断同一车型下同一类型内的疑似重复实体对是否应该合并。

## 判断标准:
- **合并 (merge)**: 两个实体指的是完全相同的概念/事物，只是命名方式不同（如"座椅按摩"和"座椅按摩功能"）
- **不合并 (keep_separate)**: 两个实体虽然名称相似，但指的是不同的具体事物/层级/粒度

## 特别注意:
- Operation 类型：如果两个操作的 steps 不同（一个是开启，一个是开启+关闭），倾向于 keep_separate
- Component 类型：如果 location 不同，即使名称相似也应 keep_separate
- Function 类型：如果 trigger_condition 完全不同，可能是不同的功能变体
- 合并时选择属性更完整的那个作为 winner

## 输出要求:
对每对实体输出一个 JSON 对象，必须包含以下字段:
- pair_index: 候选对序号（从1开始）
- decision: "merge" 或 "keep_separate"
- winner_id: 合并时保留哪个的 entity_id（keep_separate 时填 null）
- property_merge: "union"（默认）或 "winner_priority"
- reason: 简短理由（中文，20字以内）

请输出一个 JSON 数组，包含所有候选对的判断结果。不要输出其他任何内容。"""


def build_user_prompt(batch: list[dict], edge_index: dict[str, list[dict]]) -> str:
    """构建 user prompt, 包含候选对信息和关联边。"""
    lines = []
    etype = batch[0]["type"]
    model = batch[0]["model"]
    lines.append(f"## 类型: {etype} | 车型: {model}")
    lines.append(f"## Schema: {TYPE_SCHEMA.get(etype, etype)}")
    lines.append("")
    lines.append("## 候选对:")

    for idx, cand in enumerate(batch, 1):
        a_id = cand["entity_a_id"]
        b_id = cand["entity_b_id"]
        a_props = cand.get("entity_a_props", {})
        b_props = cand.get("entity_b_props", {})

        # 边摘要
        a_edges = edge_index.get(a_id, [])
        b_edges = edge_index.get(b_id, [])
        a_edge_summary = _summarize_edges(a_edges, a_id)
        b_edge_summary = _summarize_edges(b_edges, b_id)

        lines.append(f"\n### 对 {idx} (相似度: {cand['similarity']})")
        lines.append(f"- Entity A: `{a_id}`")
        lines.append(f"  name: {cand['entity_a_name']}")
        lines.append(f"  properties: {json.dumps(a_props, ensure_ascii=False)[:300]}")
        lines.append(f"  关联边({len(a_edges)}): {a_edge_summary}")
        lines.append(f"- Entity B: `{b_id}`")
        lines.append(f"  name: {cand['entity_b_name']}")
        lines.append(f"  properties: {json.dumps(b_props, ensure_ascii=False)[:300]}")
        lines.append(f"  关联边({len(b_edges)}): {b_edge_summary}")

    return "\n".join(lines)


def _summarize_edges(edges: list[dict], entity_id: str, max_show: int = 5) -> str:
    """边摘要: 显示关联的边类型和对端实体名。"""
    if not edges:
        return "无"
    summaries = []
    for e in edges[:max_show]:
        if e["source_entity_id"] == entity_id:
            summaries.append(f"-[{e['type']}]-> {e['target_entity_id'].split('::')[-1]}")
        else:
            summaries.append(f"<-[{e['type']}]- {e['source_entity_id'].split('::')[-1]}")
    suffix = f" (+{len(edges) - max_show}条)" if len(edges) > max_show else ""
    return "; ".join(summaries) + suffix


def extract_json_from_response(text: str) -> list:
    """从 LLM 回复中提取 JSON 数组。"""
    text = text.strip()
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 尝试从 code block 中提取
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # 尝试找 [ ... ]
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    logger.warning(f"无法解析 LLM 返回: {text[:200]}")
    return []


def validate_decisions(raw_decisions: list, batch_size: int) -> list[dict]:
    """使用 pydantic 验证 LLM 返回的决策格式。"""
    validated = []
    for item in raw_decisions:
        try:
            dec = MergeDecisionItem(**item)
            if 1 <= dec.pair_index <= batch_size:
                validated.append(dec.model_dump())
        except Exception as e:
            logger.debug(f"  决策格式校验失败: {e}, raw={item}")
    return validated


async def call_llm(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
    ctx: DisambiguationContext,
) -> str:
    """调用 LLM API (openai SDK), 带指数退避重试。"""
    cfg = ctx.settings
    async with semaphore:
        for attempt in range(cfg.llm_retries):
            try:
                kwargs = {
                    "model": cfg.llm_model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": cfg.llm_temperature,
                    "max_tokens": cfg.llm_max_tokens,
                }
                if cfg.disable_thinking:
                    kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                response = await client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate" in err_str.lower():
                    wait = 5 * (2**attempt)  # 5, 10, 20, 40
                    logger.warning(f"Rate limited, waiting {wait}s... (attempt {attempt + 1})")
                    await asyncio.sleep(wait)
                elif "timeout" in err_str.lower() or "connect" in err_str.lower():
                    wait = 3 * (attempt + 1)
                    logger.warning(f"Connection error, retry in {wait}s (attempt {attempt + 1}): {err_str[:100]}")
                    await asyncio.sleep(wait)
                else:
                    wait = 2 * (attempt + 1)
                    logger.warning(f"LLM error (attempt {attempt + 1}): {err_str[:150]}")
                    await asyncio.sleep(wait)
    return ""


async def judge_candidates_async(candidates: list[dict], ctx: DisambiguationContext) -> list[dict]:
    """异步批量调用 LLM 判断候选对。"""
    if not candidates:
        logger.info("Phase 2: 无候选对需要判断")
        return []
    cfg = ctx.settings
    ctx.output_dir.mkdir(parents=True, exist_ok=True)

    # 加载实体和关系，构建索引
    logger.info("Phase 2: 加载实体和关系数据...")
    with open(ctx.input_entities_path, encoding="utf-8") as f:
        entities = json.load(f)
    with open(ctx.input_relations_path, encoding="utf-8") as f:
        relations = json.load(f)

    entity_map = {e["entity_id"]: e for e in entities}
    edge_index: dict[str, list[dict]] = defaultdict(list)
    for r in relations:
        edge_index[r["source_entity_id"]].append(r)
        edge_index[r["target_entity_id"]].append(r)

    # 给候选对附加属性信息
    for cand in candidates:
        a = entity_map.get(cand["entity_a_id"], {})
        b = entity_map.get(cand["entity_b_id"], {})
        cand["entity_a_props"] = a.get("properties", {})
        cand["entity_b_props"] = b.get("properties", {})

    # 按 (type, model) 分组后分批
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for cand in candidates:
        grouped[(cand["type"], cand["model"])].append(cand)

    # 构建所有批次
    batches = []
    for group_cands in grouped.values():
        for i in range(0, len(group_cands), cfg.llm_batch_size):
            batch = group_cands[i : i + cfg.llm_batch_size]
            batches.append(batch)

    logger.info(f"  总批次: {len(batches)} (候选对 {len(candidates)}, batch_size={cfg.llm_batch_size})")

    # 检查已完成的批次（断点续跑）
    all_decisions = []
    start_batch = 0
    if ctx.partial_decisions_path.exists():
        with open(ctx.partial_decisions_path, encoding="utf-8") as f:
            saved = json.load(f)
        all_decisions = saved.get("decisions", [])
        start_batch = saved.get("completed_batches", 0)
        if start_batch > 0:
            logger.info(f"  断点续跑: 从第 {start_batch} 批次继续 (已有 {len(all_decisions)} 决策)")

    # 构建 OpenAI 客户端
    client = AsyncOpenAI(
        api_key=LLM_API_KEY,
        base_url=cfg.llm_base_url,
        timeout=cfg.llm_timeout,
    )

    # 并发调用 LLM
    semaphore = asyncio.Semaphore(cfg.llm_concurrency)
    completed = start_batch
    failed = 0
    validation_errors = 0

    async def _process_batch(batch: list[dict], batch_idx: int) -> list[dict]:
        nonlocal completed, failed, validation_errors
        prompt = build_user_prompt(batch, edge_index)
        response = await call_llm(client, prompt, semaphore, ctx)
        if not response:
            failed += 1
            logger.warning(f"  批次 {batch_idx} 失败: {batch[0]['type']}/{batch[0]['model']} ({len(batch)} 对)")
            return []

        raw_decisions = extract_json_from_response(response)
        if not raw_decisions:
            failed += 1
            return []

        # Pydantic 格式验证
        validated = validate_decisions(raw_decisions, len(batch))
        if len(validated) < len(raw_decisions):
            validation_errors += len(raw_decisions) - len(validated)

        results = []
        for dec in validated:
            idx = dec["pair_index"] - 1
            if 0 <= idx < len(batch):
                results.append(
                    {
                        "entity_a_id": batch[idx]["entity_a_id"],
                        "entity_b_id": batch[idx]["entity_b_id"],
                        "decision": dec["decision"],
                        "winner_id": dec.get("winner_id"),
                        "property_merge": dec.get("property_merge", "union"),
                        "reason": dec.get("reason", ""),
                        "similarity": batch[idx]["similarity"],
                    }
                )
        completed += 1
        if completed % 50 == 0:
            merge_total = sum(1 for d in all_decisions + results if d.get("decision") == "merge")
            logger.info(
                "  进度: %s/%s 批次完成 (merge=%s)",
                completed,
                len(batches),
                merge_total,
            )
        return results

    # 分窗口并发执行，每窗保存一次中间结果
    window_size = cfg.llm_concurrency * 3
    remaining_batches = batches[start_batch:]

    for win_start in range(0, len(remaining_batches), window_size):
        window = remaining_batches[win_start : win_start + window_size]
        tasks = [_process_batch(batch, start_batch + win_start + i) for i, batch in enumerate(window)]
        results_nested = await asyncio.gather(*tasks)
        for results in results_nested:
            all_decisions.extend(results)

        # 保存中间结果（断点续跑用）
        with open(ctx.partial_decisions_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "completed_batches": start_batch + win_start + len(window),
                    "total_batches": len(batches),
                    "decisions": all_decisions,
                },
                f,
                ensure_ascii=False,
            )

    logger.info(f"  LLM 判断完成: {completed} 成功, {failed} 失败, 格式错误: {validation_errors}")
    merge_count = sum(1 for d in all_decisions if d["decision"] == "merge")
    keep_count = sum(1 for d in all_decisions if d["decision"] == "keep_separate")
    logger.info(f"  决策: {merge_count} merge, {keep_count} keep_separate")

    # 保存最终结果
    with open(ctx.decisions_path, "w", encoding="utf-8") as f:
        json.dump(all_decisions, f, ensure_ascii=False, indent=2)
    logger.info(f"  决策已保存: {ctx.decisions_path}")

    # 清理中间文件
    if ctx.partial_decisions_path.exists():
        ctx.partial_decisions_path.unlink()

    return all_decisions


def judge_candidates(candidates: list[dict], ctx: DisambiguationContext | None = None) -> list[dict]:
    """同步包装。"""
    return asyncio.run(judge_candidates_async(candidates, ctx or make_context()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # 加载 Phase 1 的输出
    context = make_context()
    cand_path = context.candidates_path
    if not cand_path.exists():
        raise SystemExit("请先运行 Phase 1 生成 candidates.json")
    with open(cand_path, encoding="utf-8") as f:
        candidates = json.load(f)
    logger.info("加载 %s 个候选对", len(candidates))
    decisions = judge_candidates(candidates, context)
    logger.info("完成: %s 个决策", len(decisions))
