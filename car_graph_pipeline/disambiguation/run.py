"""Offline entity disambiguation pipeline entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Literal

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from car_graph_pipeline.disambiguation.phase1_candidates import find_candidates
from car_graph_pipeline.disambiguation.phase2_llm_judge import judge_candidates
from car_graph_pipeline.disambiguation.phase3_merge import execute_merge
from car_graph_pipeline.disambiguation.phase4_validate import validate
from car_graph_pipeline.disambiguation.settings import DisambiguationContext, make_context

logger = logging.getLogger("car_graph_pipeline.disambiguation")


def setup_logging(ctx: DisambiguationContext) -> None:
    """配置日志: 同时输出到控制台和当前 output_dir/run.log。"""
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = ctx.output_dir / "run.log"
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    root = logging.getLogger()
    root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def run_phase1(ctx: DisambiguationContext) -> list[dict]:
    logger.info("=" * 60)
    logger.info("Phase 1: 候选对发现")
    logger.info("=" * 60)
    t0 = time.time()
    candidates = find_candidates(ctx)
    logger.info("Phase 1 完成: %s 个候选对, 耗时 %.1fs", len(candidates), time.time() - t0)
    return candidates


def run_phase2(ctx: DisambiguationContext, candidates: list[dict] | None = None) -> list[dict]:
    logger.info("=" * 60)
    logger.info("Phase 2: LLM 判断合并决策")
    logger.info("=" * 60)
    if candidates is None:
        if not ctx.candidates_path.exists():
            raise FileNotFoundError(f"candidates.json 不存在，请先运行 Phase 1: {ctx.candidates_path}")
        candidates = json.loads(ctx.candidates_path.read_text(encoding="utf-8"))
        logger.info("  从文件加载 %s 个候选对", len(candidates))
    t0 = time.time()
    decisions = judge_candidates(candidates, ctx)
    merge_count = sum(1 for d in decisions if d["decision"] == "merge")
    logger.info("Phase 2 完成: %s merge / %s total, 耗时 %.1fs", merge_count, len(decisions), time.time() - t0)
    return decisions


def run_phase3(ctx: DisambiguationContext, decisions: list[dict] | None = None) -> tuple[list[dict], list[dict], dict]:
    logger.info("=" * 60)
    logger.info("Phase 3: 执行合并")
    logger.info("=" * 60)
    if decisions is None:
        if not ctx.decisions_path.exists():
            raise FileNotFoundError(f"merge_decisions.json 不存在，请先运行 Phase 2: {ctx.decisions_path}")
        decisions = json.loads(ctx.decisions_path.read_text(encoding="utf-8"))
        logger.info("  从文件加载 %s 个决策", len(decisions))
    t0 = time.time()
    entities, relations, merge_log = execute_merge(decisions, ctx)
    logger.info(
        "Phase 3 完成: %s 实体, %s 关系, 耗时 %.1fs",
        len(entities),
        len(relations),
        time.time() - t0,
    )
    return entities, relations, merge_log


def run_phase4(
    ctx: DisambiguationContext,
    entities: list[dict] | None = None,
    relations: list[dict] | None = None,
) -> dict:
    logger.info("=" * 60)
    logger.info("Phase 4: 验证")
    logger.info("=" * 60)
    if entities is None:
        if not ctx.merged_entities_path.exists():
            raise FileNotFoundError(f"merged_entities.json 不存在，请先运行 Phase 3: {ctx.merged_entities_path}")
        entities = json.loads(ctx.merged_entities_path.read_text(encoding="utf-8"))
        relations = json.loads(ctx.merged_relations_path.read_text(encoding="utf-8"))
    assert relations is not None
    t0 = time.time()
    results = validate(entities, relations, ctx)
    logger.info("Phase 4 完成: %s, 耗时 %.1fs", "全部通过" if results["all_pass"] else "部分失败", time.time() - t0)
    return results


def run_disambiguation(
    *,
    phase: Literal[1, 2, 3, 4] | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    input_version: str | None = None,
    output_version: str | None = None,
) -> bool:
    """运行离线实体消歧流水线。"""
    ctx = make_context(
        input_dir=input_dir,
        output_dir=output_dir,
        input_version=input_version,
        output_version=output_version,
    )
    setup_logging(ctx)
    logger.info("实体消歧流水线启动")
    logger.info("  输入: %s", ctx.input_dir)
    logger.info("  输出: %s", ctx.output_dir)

    total_start = time.time()
    candidates: list[dict] | None = None
    decisions: list[dict] | None = None
    entities: list[dict] | None = None
    relations: list[dict] | None = None

    if phase is None or phase == 1:
        candidates = run_phase1(ctx)
        if phase == 1:
            return True

    if phase is None or phase == 2:
        decisions = run_phase2(ctx, candidates if phase is None else None)
        if phase == 2:
            return True

    if phase is None or phase == 3:
        entities, relations, _ = run_phase3(ctx, decisions if phase is None else None)
        if phase == 3:
            return True

    if phase is None or phase == 4:
        results = run_phase4(ctx, entities if phase is None else None, relations if phase is None else None)
        if not results["all_pass"]:
            return False

    logger.info("全流程完成, 总耗时 %.1fs", time.time() - total_start)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="car graph 离线实体消歧流水线")
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="只运行指定 phase，默认全量运行",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None, help="包含 extracted_entities.json/extracted_relations.json 的目录"
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="消歧输出目录")
    parser.add_argument(
        "--input-version",
        default=None,
        help="使用 car_graph_pipeline/output/version/<input-version> 作为输入",
    )
    parser.add_argument(
        "--output-version",
        default=None,
        help="使用 car_graph_pipeline/output/version/<output-version> 作为输出",
    )
    args = parser.parse_args()
    ok = run_disambiguation(
        phase=args.phase,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        input_version=args.input_version,
        output_version=args.output_version,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
