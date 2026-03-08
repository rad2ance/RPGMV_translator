import csv
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import rpgmv_translator.config_manager as config_manager
from rpgmv_translator.request_controller import GPTRequestController
from rpgmv_translator.utils import contains_japanese_strict


class CSVTranslationPipelineIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.temp_dir.name)
        fixture = Path(__file__).parent / "fixtures" / "sample_translation.csv"
        self.input_csv = self.workdir / "input.csv"
        self.output_csv = self.workdir / "output.csv"
        shutil.copyfile(fixture, self.input_csv)

        self.config = config_manager.load_config()
        self.api_key = self.config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("Integration tests require openai_api_key in config.json or OPENAI_API_KEY env var")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _build_controller(self):
        return GPTRequestController(
            max_tokens=int(self.config.get("max_tokens", 2000)),
            language=self.config.get("language", "Japanese"),
            model=self.config.get("model", "gpt-4.1-mini"),
        )

    def test_csv_translation_writes_target_column(self):
        controller = self._build_controller()

        controller.process_arbitrary_csv(
            input_csv_path=str(self.input_csv),
            output_csv_path=str(self.output_csv),
            source_column="text",
            target_column="translated_text",
            id_column="id",
        )

        with open(self.output_csv, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        self.assertTrue(rows[0]["translated_text"])
        self.assertFalse(contains_japanese_strict(rows[0]["translated_text"]))
        self.assertTrue(rows[1]["translated_text"])
        self.assertEqual(rows[2]["translated_text"], "")
        self.assertFalse((Path(str(self.output_csv) + ".part")).exists())

    def test_csv_translation_resumes_from_part_file(self):
        part_path = Path(str(self.output_csv) + ".part")
        with open(part_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "translated_text"])
            writer.writeheader()
            writer.writerow({"id": "1", "translated_text": "PRETRANSLATED"})

        controller = self._build_controller()
        controller.process_arbitrary_csv(
            input_csv_path=str(self.input_csv),
            output_csv_path=str(self.output_csv),
            source_column="text",
            target_column="translated_text",
            id_column="id",
        )

        with open(self.output_csv, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["translated_text"], "PRETRANSLATED")
        self.assertTrue(rows[1]["translated_text"])
        self.assertEqual(rows[2]["translated_text"], "")


if __name__ == "__main__":
    unittest.main()
