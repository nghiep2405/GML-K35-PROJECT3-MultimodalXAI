import importlib.util
from pathlib import Path
import unittest

from multimodal_xai.viheritage.snapshot import verify_snapshot


BUNDLE = Path(__file__).resolve().parents[2] / "our_run_result"


class LocalArtifactTests(unittest.TestCase):
    @unittest.skipUnless((BUNDLE / "data/MMKG/processed/split_manifest.json").is_file(),
                         "requires the local ViHeritage snapshot")
    def test_viheritage_published_snapshot(self):
        report = verify_snapshot(BUNDLE / "data/MMKG")
        self.assertEqual(report["issues"], [])
        self.assertEqual(report["checked"], 19)
        self.assertEqual(report["counts"]["triples"], 427)

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "torch_geometric")),
                         "requires PyTorch and PyG")
    def test_flickr_model_and_self_loop_graph(self):
        import torch
        from torch_geometric.data import Batch
        from multimodal_xai.flickr30k.graph import graph_for
        from multimodal_xai.flickr30k.model import SpatialGAT
        nodes = torch.ones(2, 512)
        graph = graph_for(nodes, {}, 0, edge_mode="no_edge", seed=2026)
        self.assertEqual(graph.edge_index.tolist(), [[0, 1], [0, 1]])
        model = SpatialGAT().eval()
        with torch.inference_mode():
            result = model.encode_graph(Batch.from_data_list([graph]))
        self.assertEqual(result.shape, (1, 512))
        self.assertTrue(torch.isfinite(result).all())
