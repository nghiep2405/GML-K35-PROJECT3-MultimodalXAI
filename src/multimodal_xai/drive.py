"""Run experiments directly from the uploaded bundle on local machines or Drive."""
import os
from pathlib import Path
import subprocess
import sys

from .artifacts.preflight import missing_inputs


DEFAULT_ROOT = Path("/content/drive/MyDrive/MultimodalXAI")
ACTIONS = {
    "flickr30k": ("train", "evaluate", "compare", "replay", "explain"),
    "db15k": ("train", "train-evaluate", "replay", "explain", "quantify"),
    "fb15k237": ("train", "train-evaluate", "replay", "explain", "quantify"),
    "visual-genome": ("train", "evaluate"),
    "viheritage": ("verify",),
}


def local_root():
    """Find the repository from the working directory or editable installation."""
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for candidate in (start, *start.parents):
            if (candidate / "our_run_result/data").is_dir():
                return candidate
    return Path.cwd()


def add_parser(commands, name="drive"):
    parser = commands.add_parser(name, help="run with uploaded our_run_result data and checkpoints")
    parser.add_argument("pipeline", choices=ACTIONS)
    parser.add_argument("action")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT if name == "drive" else local_root())
    parser.add_argument("--seed", type=int)
    parser.add_argument("--variant", default="main")
    return parser


def run(args):
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project folder missing: {root}. Select the repository or mounted Drive project.")
    pipeline, action, variant = args.pipeline, args.action, args.variant
    if action not in ACTIONS[pipeline]:
        raise ValueError(f"{pipeline} actions: {', '.join(ACTIONS[pipeline])}")

    baseline = root / "our_run_result"
    outputs = root / "runs"
    env = os.environ.copy()

    def launch(*command, **settings):
        return subprocess.call(
            [sys.executable, "-m", "multimodal_xai.cli", *map(str, command)],
            env={**env, **{key: str(value) for key, value in settings.items()}},
        )

    def require(*paths):
        missing = [str(path) for path in paths if not Path(path).exists()]
        if missing:
            raise FileNotFoundError("Missing bundle inputs:\n" + "\n".join(missing))

    def check(name, data, mode):
        missing = missing_inputs(name, data, mode)
        if missing:
            raise FileNotFoundError("Missing processed artifacts:\n" + "\n".join(missing))

    def workflow(name, through=None, **settings):
        command = ["workflow", name]
        if through is not None:
            command.extend(("--through", through))
        return launch(*command, **settings)

    if pipeline == "viheritage":
        snapshot = baseline / "data/MMKG"
        check("viheritage", snapshot, "evaluate")
        return launch("verify-viheritage", snapshot, "--output", outputs / "ViHeritage/verification.json")

    if pipeline == "flickr30k":
        out = outputs / "Flickr30k"
        summary = out / "evaluate/final_test_summary.json"
        reference = baseline / "results/Flickr30k"
        if action == "compare":
            require(reference / "final_test_summary.json", summary)
            return launch("compare-flickr30k", reference / "final_test_summary.json",
                          summary, out / "evaluate/parity.csv")
        if action == "train":
            seed = args.seed if args.seed is not None else 2026
            runs = {"main": ("real_edges", f"seed_{seed}"), "no_edge": ("no_edge", "no_edge"),
                    "random_edges": ("random_edges", "random_edges")}
            if variant not in runs or seed not in (2026, 2027, 2028) or (variant != "main" and seed != 2026):
                raise ValueError("Flickr30k uses main seeds 2026–2028 and ablation seed 2026")
            check("flickr30k", baseline, "train")
            edge_mode, name = runs[variant]
            return launch("flickr30k", "train", "--root", baseline, "--output", out / "train",
                          "--published-config", FLICKR_SEED=seed, FLICKR_EDGE_MODE=edge_mode,
                          FLICKR_RUN_NAME=name, FLICKR_NEGATIVES_ROOT=baseline / "data/Flickr30k/retrieval")
        if action == "evaluate":
            # Evaluate a complete set of new runs, keeping the group baseline independent.
            for name in ("seed_2026", "seed_2027", "seed_2028", "no_edge", "random_edges"):
                require(out / "train" / name / "best_model.pt")
            check("flickr30k", baseline, "evaluate")
            return launch("flickr30k", "evaluate", "--root", baseline, "--output", summary,
                          "--reference", out / "train", FLICKR_RUNS_ROOT=out / "train")
        check("flickr30k", baseline, "explain")
        require(reference / "final_test_summary.json",
                *(reference / f"3SEED/seed_{seed}/best_model.pt" for seed in (2026, 2027, 2028)),
                *(reference / f"ABLATION/{name}/best_model.pt" for name in ("no_edge", "random_edges")))
        image_dir = baseline / "data/Flickr30k/raw_images"
        if action == "explain":
            require(image_dir)
        return workflow("06_flickr30k_gat_xai_level1", 8 if action == "replay" else 29,
                        XAI_FINAL_ROOT=baseline, XAI_OUT=out / action, XAI_WRITE_DRIVE=0,
                        XAI_DOWNLOAD_TEST_IMAGES=0, XAI_IMAGE_DIR=str(image_dir) if image_dir.is_dir() else "")

    if pipeline == "visual-genome":
        out = outputs / "VisualGenome/experiment_v2"
        seed = args.seed if args.seed is not None else 2026
        variants = {"main": ("real_edges", "main"), "no_edge": ("no_edge", "no_edge"),
                    "random_edges": ("random_edges", "random_edges")}
        if variant not in variants or seed not in (2026, 2027, 2028) or (variant != "main" and seed != 2026):
            raise ValueError("Visual Genome uses main seeds 2026–2028 and ablation seed 2026")
        if action == "evaluate" and variant == "random_edges":
            raise ValueError("Visual Genome held-out evaluator does not reproduce randomized edges")
        edge_mode, label = variants[variant]
        name = f"vg_full_{label}_seed{seed}"
        check("visual_genome", root, action)
        settings = {"GRAPHGAT_OUTPUT_ROOT": out, "GRAPHGAT_PROJECT_ROOT": root}
        if action == "train":
            return launch("visual-genome", "--root", root, "--seed", seed, "--edge-mode", edge_mode,
                          "--run-name", name, **settings)
        require(out / "model/gat_dynamic_v2" / name / f"seed_{seed}/best_gat.pt")
        return workflow("visual_genome_evaluate_test", GRAPHGAT_ENABLE_HELDOUT_TEST=1,
                        GRAPHGAT_EVAL_SEED=seed, GRAPHGAT_EVAL_RUN_NAME=name,
                        GRAPHGAT_EVAL_EDGE_MODE=edge_mode, **settings)

    dataset = "DB15K" if pipeline == "db15k" else "FB15K"
    adamf = baseline / "data" / dataset
    out = outputs / dataset
    check(pipeline, adamf, action)
    if pipeline == "db15k":
        require(adamf / "db15k_image_retrieval/runs/db15k_clean_v1/manifest.json")
    if action in ("replay", "explain", "quantify"):
        # The frozen group model tree supplies checkpoint identities and encoder revisions.
        checkpoints = baseline / "results" / dataset / "3SEED"
        for seed in (2027, 2028, 2029):
            folder = checkpoints / f"seed_{seed}"
            require(folder / "best_adamf.pt")
        require(checkpoints / "seed_2027" / ("clip_encoder.json" if pipeline == "db15k" else "text_encoder.json"))
        canonical = ("db15k_image_retrieval/runs/db15k_v1/processed/entities.parquet"
                     if pipeline == "db15k" else
                     "fb15k237_image_retrieval/runs/fb15k237_v1/processed_stage/entities.parquet")
        require(adamf / canonical)
        settings = {"XAI2_PROJECT_ROOT": root, "XAI2_DATA_ROOT": adamf,
                    "XAI2_CHECKPOINT_ROOT": checkpoints,
                    "XAI2_DATASETS": dataset, "XAI2_FINAL_ROOT": baseline,
                    "XAI2_OUT": outputs / ({"replay": "Replay", "explain": "XAI-Level2", "quantify": "XAI-Level3"}[action]),
                    "XAI2_WRITE_DRIVE": 0, "XAI2_REUSE": 0}
        if action == "quantify":
            reference = outputs / "XAI-Level2"
            require(reference / dataset / "C6_K_star.json")
            return workflow("10a_xai_quant_db15k_fb15k", XAI2_REFERENCE_ROOT=reference, **settings)
        return workflow("07_db15k_fb15k_adamf_xai_level2", 5 if action == "replay" else 20, **settings)

    seed = args.seed if args.seed is not None else 2027
    if seed not in (2027, 2028, 2029):
        raise ValueError("AdaMF training seeds: 2027, 2028, 2029")
    if variant != "main" and (pipeline != "db15k" or seed != 2027 or variant not in (
        "no_edges", "random_graph", "shuffled_edges_zero_overlap_v4"
    )):
        raise ValueError("These ablation workflows are available only for DB15K seed 2027")
    settings = {"ADAMF_DATA_ROOT": adamf, "ADAMF_PROJECT_ROOT": root, "ADAMF_RUN_ROOT": out / "train"}
    if pipeline == "db15k":
        name = f"04b1_adamf_db15k_seed{seed}"
        if variant != "main":
            name += "_ablation_" + variant
        through = 11
    else:
        require(adamf / "sources/AdaMF-MAT/mmkgc/base/Base.cpp")
        name = f"04b2_adamf_fb15k237_seed{seed}"
        through = 19
        settings.update(ADAMF_RESUME="auto", ADAMF_REPO=out / "work/AdaMF-MAT",
                        ADAMF_LOCAL_CACHE=out / "work/cache")
    return workflow(name, through if action == "train" else None, **settings)
