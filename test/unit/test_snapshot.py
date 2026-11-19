from pathlib import Path
import hashlib
import json
import tempfile
import unittest

from multimodal_xai.viheritage.snapshot import verify_snapshot


class SnapshotTests(unittest.TestCase):
    def test_verify_snapshot_and_reject_modified_artifact(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            processed = root / "processed"
            processed.mkdir()
            data = processed / "entities.csv"
            data.write_bytes(b"entity_index\n0\n")
            manifest = {
                "counts": {"entities": 1},
                "files": {
                    "entities": {
                        "path": "processed/entities.csv",
                        "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                    }
                },
            }
            (processed / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(verify_snapshot(root)["status"], "PASS")
            data.write_bytes(b"modified")
            self.assertEqual(verify_snapshot(root)["issues"], [
                {"name": "entities", "error": "sha256 mismatch"},
            ])
