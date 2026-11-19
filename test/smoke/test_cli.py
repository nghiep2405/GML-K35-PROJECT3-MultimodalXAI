import json
from pathlib import Path
import tempfile
import unittest

from multimodal_xai.cli import main


class CliTests(unittest.TestCase):
    def test_missing_external_artifact_report_is_blocked(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            report = root / "results/preflight.json"
            self.assertEqual(main(["preflight", "db15k", str(root), "--json-out", str(report)]), 2)
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "BLOCKED")
            self.assertEqual(len(result["missing"]), 1)

    def test_import_visual_genome_runner_without_artifacts(self):
        from multimodal_xai.visual_genome import train
        self.assertTrue(callable(train.main))
