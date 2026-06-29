from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT))


class LlmApiSettingsTest(unittest.TestCase):
    def test_loads_yaml_defaults(self) -> None:
        from car_graph_pipeline.extraction.llm_api import settings

        cfg = settings.load_settings(env={})

        self.assertEqual(cfg.model, "DeepSeek-V4-Flash")
        self.assertEqual(cfg.default_workers, 5)
        self.assertEqual(cfg.max_tokens_extract, 40000)
        self.assertEqual(cfg.chunk_target_chars, 2000)
        self.assertEqual(cfg.chunk_max_chars, 2800)

    def test_local_yaml_overrides_defaults(self) -> None:
        from car_graph_pipeline.extraction.llm_api import settings

        with TemporaryDirectory() as tmp:
            local_path = Path(tmp) / "config.local.yaml"
            local_path.write_text(
                "\n".join(
                    [
                        "llm:",
                        "  workers: 11",
                        "chunk:",
                        "  target_chars: 1500",
                    ]
                ),
                encoding="utf-8",
            )

            cfg = settings.load_settings(env={}, local_config_path=local_path)

        self.assertEqual(cfg.default_workers, 11)
        self.assertEqual(cfg.chunk_target_chars, 1500)
        self.assertEqual(cfg.chunk_max_chars, 2800)

    def test_environment_overrides_yaml(self) -> None:
        from car_graph_pipeline.extraction.llm_api import settings

        cfg = settings.load_settings(
            env={
                "CAR_GRAPH_LLM_WORKERS": "17",
                "CAR_GRAPH_CHUNK_TARGET_CHARS": "1234",
                "CAR_GRAPH_LLM_ADAPTIVE_TOKEN_STEPS": "40000,50000,70000",
            }
        )

        self.assertEqual(cfg.default_workers, 17)
        self.assertEqual(cfg.chunk_target_chars, 1234)
        self.assertEqual(cfg.adaptive_token_steps, [40000, 50000, 70000])


if __name__ == "__main__":
    unittest.main()
