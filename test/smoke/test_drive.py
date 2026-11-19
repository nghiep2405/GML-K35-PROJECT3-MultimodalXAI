import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from multimodal_xai import drive


class DriveTests(unittest.TestCase):
    def options(self, root, *args):
        parser = argparse.ArgumentParser()
        drive.add_parser(parser.add_subparsers(dest="command"))
        return parser.parse_args(["drive", *args, "--root", str(root)])

    def test_fb_train_uses_preprocessed_inputs_and_separate_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            adamf = root / "our_run_result/data/FB15K"
            for name in ("processed/fb15k237_img/retrieval_bridge/manifest.json",
                         "sources/AdaMF-MAT/mmkgc/base/Base.cpp"):
                path = adamf / name
                path.parent.mkdir(parents=True)
                path.touch()
            args = self.options(root, "fb15k237", "train", "--seed", "2028")
            with patch.object(drive.subprocess, "call", return_value=0) as call:
                self.assertEqual(drive.run(args), 0)
            self.assertEqual(call.call_args.args[0][-4:],
                             ["workflow", "04b2_adamf_fb15k237_seed2028", "--through", "19"])
            self.assertEqual(call.call_args.kwargs["env"]["ADAMF_RUN_ROOT"], str(root.resolve() / "runs/FB15K/train"))
            self.assertEqual(call.call_args.kwargs["env"]["ADAMF_RESUME"], "auto")

    def test_flickr_evaluate_requires_new_runs_before_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = self.options(root, "flickr30k", "evaluate")
            with patch.object(drive.subprocess, "call") as launch:
                with self.assertRaisesRegex(FileNotFoundError, "best_model.pt"):
                    drive.run(args)
            launch.assert_not_called()


    def test_replay_uses_existing_bundle_without_legacy_model_tree(self):
        root = Path(__file__).resolve().parents[2]
        for pipeline, dataset in (("db15k", "DB15K"), ("fb15k237", "FB15K")):
            with self.subTest(pipeline=pipeline):
                args = self.options(root, pipeline, "replay")
                with patch.object(drive.subprocess, "call", return_value=0) as call:
                    self.assertEqual(drive.run(args), 0)
                env = call.call_args.kwargs["env"]
                self.assertEqual(Path(env["XAI2_DATA_ROOT"]), root / "our_run_result/data" / dataset)
                self.assertEqual(Path(env["XAI2_CHECKPOINT_ROOT"]),
                                 root / "our_run_result/results" / dataset / "3SEED")
                self.assertEqual(call.call_args.args[0][-2:], ["--through", "5"])

    def test_local_command_finds_editable_repository_from_another_directory(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(drive.Path, "cwd", return_value=Path(folder)):
                self.assertEqual(drive.local_root(), root)
