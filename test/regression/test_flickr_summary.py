import json
from pathlib import Path
import tempfile
import unittest

from multimodal_xai.artifacts.parity import compare_flickr_summaries


BUNDLE = Path(__file__).resolve().parents[2] / "our_run_result"
REFERENCE = BUNDLE / "results/Flickr30k/final_test_summary.json"


class FlickrReplayTests(unittest.TestCase):
    @unittest.skipUnless(REFERENCE.is_file(), "requires the local Flickr30k summary")
    def test_summary_comparison_rejects_metric_drift(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            result = root / "parity.csv"
            self.assertTrue(compare_flickr_summaries(REFERENCE, REFERENCE, result))
            original = json.loads(REFERENCE.read_text(encoding="utf-8"))
            original["runs"][0]["test_clip_plus_graph"]["MRR"] -= 0.01
            changed = root / "changed.json"
            changed.write_text(json.dumps(original), encoding="utf-8")
            self.assertFalse(compare_flickr_summaries(REFERENCE, changed, result))
            self.assertIn("FAIL", result.read_text(encoding="utf-8"))
