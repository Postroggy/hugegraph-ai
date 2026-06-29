from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT))


class DisambiguationTest(unittest.TestCase):
    def test_settings_env_override(self) -> None:
        from car_graph_pipeline.disambiguation.settings import load_settings

        cfg = load_settings(
            env={
                "CAR_GRAPH_DISAMBIGUATION_LLM_CONCURRENCY": "9",
                "CAR_GRAPH_DISAMBIGUATION_COSINE_THRESHOLD": "0.91",
                "CAR_GRAPH_DISAMBIGUATION_SKIP_TYPES": "VehicleBrand,VehicleModel,Material",
            }
        )

        self.assertEqual(cfg.llm_concurrency, 9)
        self.assertEqual(cfg.cosine_similarity_threshold, 0.91)
        self.assertIn("Material", cfg.skip_types)

    def test_vehicle_model_prefers_property_over_entity_id(self) -> None:
        from car_graph_pipeline.disambiguation.phase1_candidates import get_vehicle_model

        entity = {
            "type": "Component",
            "entity_id": "comp::旧车型::按钮",
            "properties": {"vehicle_model": "新车型"},
        }

        self.assertEqual(get_vehicle_model(entity), "新车型")

    def test_merge_and_validate_local_json(self) -> None:
        from car_graph_pipeline.disambiguation.phase3_merge import execute_merge
        from car_graph_pipeline.disambiguation.phase4_validate import validate
        from car_graph_pipeline.disambiguation.settings import DisambiguationSettings, make_context

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            entities = [
                {
                    "entity_id": "comp::ModelA::座椅按摩按钮",
                    "type": "Component",
                    "name": "座椅按摩按钮",
                    "properties": {
                        "component_type": "button",
                        "vehicle_brand": "理想",
                        "vehicle_model": "ModelA",
                    },
                },
                {
                    "entity_id": "comp::ModelA::座椅按摩按键",
                    "type": "Component",
                    "name": "座椅按摩按键",
                    "properties": {
                        "location": "中控台",
                        "vehicle_brand": "错误品牌",
                        "vehicle_model": "错误车型",
                    },
                },
                {
                    "entity_id": "func::ModelA::座椅按摩功能",
                    "type": "Function",
                    "name": "座椅按摩功能",
                    "properties": {
                        "function_type": "comfort",
                        "vehicle_brand": "理想",
                        "vehicle_model": "ModelA",
                    },
                },
                {
                    "entity_id": "comp::ModelB::座椅按摩按钮",
                    "type": "Component",
                    "name": "座椅按摩按钮",
                    "properties": {
                        "component_type": "button",
                        "vehicle_brand": "其他",
                        "vehicle_model": "ModelB",
                    },
                },
            ]
            relations = [
                {
                    "type": "ACTIVATES",
                    "source_entity_id": "comp::ModelA::座椅按摩按键",
                    "target_entity_id": "func::ModelA::座椅按摩功能",
                    "properties": {"vehicle_model": "ModelA"},
                }
            ]
            (input_dir / "extracted_entities.json").write_text(
                json.dumps(entities, ensure_ascii=False), encoding="utf-8"
            )
            (input_dir / "extracted_relations.json").write_text(
                json.dumps(relations, ensure_ascii=False), encoding="utf-8"
            )

            ctx = make_context(
                settings=DisambiguationSettings(max_entity_reduction_pct=40.0),
                input_dir=input_dir,
                output_dir=output_dir,
            )
            decisions = [
                {
                    "entity_a_id": "comp::ModelA::座椅按摩按钮",
                    "entity_b_id": "comp::ModelA::座椅按摩按键",
                    "decision": "merge",
                    "winner_id": "comp::ModelA::座椅按摩按钮",
                    "property_merge": "union",
                }
            ]

            merged_entities, merged_relations, merge_log = execute_merge(decisions, ctx)
            results = validate(merged_entities, merged_relations, ctx)

        entity_ids = {entity["entity_id"] for entity in merged_entities}
        self.assertIn("comp::ModelA::座椅按摩按钮", entity_ids)
        self.assertNotIn("comp::ModelA::座椅按摩按键", entity_ids)
        self.assertIn("comp::ModelB::座椅按摩按钮", entity_ids)
        self.assertEqual(merged_relations[0]["source_entity_id"], "comp::ModelA::座椅按摩按钮")
        winner = next(entity for entity in merged_entities if entity["entity_id"] == "comp::ModelA::座椅按摩按钮")
        self.assertEqual(winner["properties"]["vehicle_model"], "ModelA")
        self.assertEqual(winner["properties"]["location"], "中控台")
        self.assertEqual(merge_log["entities_removed"], 1)
        self.assertTrue(results["all_pass"])


if __name__ == "__main__":
    unittest.main()
