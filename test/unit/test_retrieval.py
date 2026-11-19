import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("numpy"), "requires NumPy")
class SamplerTests(unittest.TestCase):
    def test_visual_genome_sampler_is_deterministic_and_rotates(self):
        from multimodal_xai.visual_genome.sampler import EpochQuerySampler
        rows = [("image", str(i)) for i in range(6)]
        first = EpochQuerySampler(rows, seed=2027, per_image=3)
        second = EpochQuerySampler(list(reversed(rows)), seed=2027, per_image=3)
        self.assertEqual(first.sample(0), second.sample(0))
        self.assertEqual(len(set(first.sample(0) + first.sample(1))), 6)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
class RankingTests(unittest.TestCase):
    def test_flickr_rank_counts_strictly_higher_scores(self):
        import torch
        from multimodal_xai.retrieval.ranking import ranking_metrics
        result = ranking_metrics(torch.tensor([[1., 0., 1.], [0., 1., 2.]]), torch.tensor([0, 1]))
        self.assertEqual(result["Recall@1"], 0.5)
        self.assertEqual(result["MRR"], 0.75)
