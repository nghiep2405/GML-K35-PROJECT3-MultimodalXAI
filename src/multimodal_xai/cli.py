"""Command-line entry points for the refactored project."""
import argparse
import importlib
import json
import os
from pathlib import Path

from .artifacts.preflight import missing_inputs
from . import drive
from .workflows.runner import catalog, run_workflow


FLICKR = {"prepare": "data", "train": "train", "evaluate": "evaluate"}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="multimodal-xai")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight", help="check the minimum artifact layout")
    check.add_argument("pipeline", choices=["flickr30k", "visual_genome", "db15k", "fb15k237", "viheritage"])
    check.add_argument("root", type=Path)
    check.add_argument("--action", choices=["prepare", "train", "evaluate", "explain"], default="evaluate")
    check.add_argument("--json-out", type=Path)
    flickr = commands.add_parser("flickr30k")
    flickr.add_argument("action", choices=FLICKR)
    flickr.add_argument("--root", required=True, type=Path)
    flickr.add_argument("--output", type=Path, help="negative-cache directory, run directory or summary path")
    flickr.add_argument("--reference", type=Path, help="Flickr30k directory of historical checkpoints")
    flickr.add_argument("--published-config", action="store_true", help="use Flickr30k run_experiments.sh hyperparameters")
    vg = commands.add_parser("visual-genome")
    vg.add_argument("--root", required=True, type=Path)
    vg.add_argument("--seed", type=int, default=2026)
    vg.add_argument("--edge-mode", choices=["real_edges", "no_edge", "random_edges"], default="real_edges")
    vg.add_argument("--run-name", required=True)
    vg.add_argument("--epochs", type=int)
    graph = commands.add_parser("visual-genome-graph")
    graph.add_argument("root", type=Path)
    graph.add_argument("output", type=Path)
    compare = commands.add_parser("compare-flickr30k")
    compare.add_argument("reference", type=Path)
    compare.add_argument("actual", type=Path)
    compare.add_argument("output", type=Path)
    compare.add_argument("--tolerance", type=float, default=2e-4)
    work = commands.add_parser("workflow", help="run stages extracted from a notebook")
    work.add_argument("name", choices=catalog())
    work.add_argument("--through", type=int, help="stop after this code cell index")
    snapshot = commands.add_parser("verify-viheritage")
    snapshot.add_argument("root", type=Path)
    snapshot.add_argument("--output", type=Path)
    commands.add_parser("list-workflows")
    drive_parser = drive.add_parser(commands)
    local_parser = drive.add_parser(commands, "local")
    args = parser.parse_args(argv)

    if args.command in ("drive", "local"):
        try:
            return drive.run(args)
        except (FileNotFoundError, ValueError) as exc:
            (local_parser if args.command == "local" else drive_parser).error(str(exc))

    if args.command == "visual-genome-graph":
        from .visual_genome.visualize import render_sample
        render_sample(args.root, args.output)
        return 0
    if args.command == "compare-flickr30k":
        from .artifacts.parity import compare_flickr_summaries
        return int(not compare_flickr_summaries(args.reference, args.actual, args.output, args.tolerance))
    if args.command == "verify-viheritage":
        from .viheritage.snapshot import verify_snapshot
        result = verify_snapshot(args.root)
        rendered = json.dumps(result, indent=2, ensure_ascii=False)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return int(result["status"] != "PASS")
    if args.command == "list-workflows":
        for name, entry in catalog().items():
            print(f"{name}: {', '.join(str(s['cell']) for s in entry['stages'])}")
        return 0
    if args.command == "preflight":
        missing = missing_inputs(args.pipeline, args.root, args.action)
        result = {"status": "BLOCKED" if missing else "READY", "pipeline": args.pipeline, "missing": missing}
        rendered = json.dumps(result, indent=2, ensure_ascii=False)
        print(rendered)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(rendered + "\n", encoding="utf-8")
        return 2 if missing else 0
    if args.command == "workflow":
        run_workflow(args.name, args.through)
        return 0

    os.environ["GRAPHGAT_PROJECT_ROOT"] = str(args.root.resolve())
    if args.command == "flickr30k":
        if args.output:
            variables = {"prepare": "FLICKR_NEGATIVES_ROOT", "train": "FLICKR_RUNS_ROOT", "evaluate": "FLICKR_SUMMARY"}
            os.environ[variables[args.action]] = str(args.output.resolve())
        if args.reference:
            os.environ["FLICKR_REFERENCE_ROOT"] = str(args.reference.resolve())
        if args.published_config:
            if args.action != "train":
                parser.error("--published-config applies only to Flickr30k train")
            os.environ.update(FLICKR_LEARNING_RATE="1e-4", FLICKR_MAX_EPOCHS="50",
                              FLICKR_MIN_EPOCHS="50", FLICKR_PATIENCE="51",
                              FLICKR_MIN_DELTA="0.0001", FLICKR_QUERY_BATCH="32",
                              FLICKR_CHECKPOINT_EVERY="200")
        importlib.import_module(f".flickr30k.{FLICKR[args.action]}", __package__).main()
        return 0
    if Path(args.run_name).name != args.run_name or args.run_name in {"", ".", ".."}:
        parser.error("--run-name must be a single directory name")
    os.environ.update(
        GRAPHGAT_SEEDS=str(args.seed), GRAPHGAT_EDGE_MODE=args.edge_mode,
        GRAPHGAT_RUN_NAME=args.run_name, GRAPHGAT_QUERIES_PER_IMAGE="3",
    )
    if args.epochs is not None:
        os.environ["GRAPHGAT_EPOCHS"] = str(args.epochs)
    from .visual_genome.train import main as train
    train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
