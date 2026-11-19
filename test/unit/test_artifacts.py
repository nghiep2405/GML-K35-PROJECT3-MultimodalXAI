import os
from pathlib import Path
import tempfile
import unittest

from multimodal_xai.artifacts.layouts import flickr_checkpoint
from multimodal_xai.artifacts.preflight import missing_inputs
from multimodal_xai.visual_genome.guard import WriteGuard


class ArtifactTests(unittest.TestCase):
    def test_find_historical_and_new_flickr_checkpoints(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            runs = root / "runs"
            reference = root / "reference"
            for name, group in [("seed_2027", "3SEED"), ("no_edge", "ABLATION")]:
                file = reference / group / name / "best_model.pt"
                file.parent.mkdir(parents=True)
                file.touch()
                self.assertEqual(flickr_checkpoint(runs, reference, name), file)
            fresh = runs / "seed_2027" / "best_model.pt"
            fresh.parent.mkdir(parents=True)
            fresh.touch()
            self.assertEqual(flickr_checkpoint(runs, reference, "seed_2027"), fresh)
            with self.assertRaises(FileNotFoundError):
                flickr_checkpoint(runs, reference, "unknown")

    def test_flickr_preflight_requires_both_modalities(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            for modality in ["images", "texts"]:
                split = root / "data/Flickr30k/processed/clip_vit_b32_openai" / modality / "test"
                split.mkdir(parents=True)
                if modality == "images":
                    (split / "shard_00000.pt").touch()
            missing = missing_inputs("flickr30k", root)
            self.assertEqual(len(missing), 1)
            self.assertIn(os.path.join("texts", "test"), missing[0])

    def test_visual_genome_guard_blocks_outside_writes(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            guard = WriteGuard(root / "dataset", root / "experiment_v2")
            path = root / "experiment_v2/runs/a.json"
            guard.save_json(path, {"ok": True})
            self.assertTrue(path.exists())
            with self.assertRaises(PermissionError):
                guard.save_json(root / "dataset/bad.json", {})
