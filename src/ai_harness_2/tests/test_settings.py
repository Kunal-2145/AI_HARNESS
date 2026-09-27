import os
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_harness_2.config.settings import Settings


class SettingsTests(unittest.TestCase):
    def test_evaluator_ai_api_key_is_accepted(self) -> None:
        with (
            patch("ai_harness_2.config.settings.load_dotenv"),
            patch.dict(
                os.environ,
                {"AI_API_KEY": "eval-key"},
                clear=True,
            ),
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.api_key, "eval-key")
        self.assertEqual(settings.model, "deepseek/deepseek-v4.1-flash:free")
        self.assertEqual(settings.base_url, "https://openrouter.ai/api/v1")

    def test_model_and_base_url_can_be_overridden(self) -> None:
        with (
            patch("ai_harness_2.config.settings.load_dotenv"),
            patch.dict(
                os.environ,
                {
                    "AI_API_KEY": "eval-key",
                    "LLM_MODEL": "custom/model",
                    "LLM_BASE_URL": "https://example.test/v1",
                },
                clear=True,
            ),
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.base_url, "https://example.test/v1")
        self.assertEqual(settings.model, "custom/model")

    def test_history_database_path_can_be_overridden(self) -> None:
        with (
            patch("ai_harness_2.config.settings.load_dotenv"),
            patch.dict(
                os.environ,
                {"AI_API_KEY": "eval-key", "AI_HARNESS_DB_PATH": "./history.sqlite3"},
                clear=True,
            ),
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.history_db_path, Path("history.sqlite3").resolve())

    def test_api_key_is_required(self) -> None:
        with (
            patch("ai_harness_2.config.settings.load_dotenv"),
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(RuntimeError, "AI_API_KEY is not set"),
        ):
            Settings.from_env()


if __name__ == "__main__":
    unittest.main()