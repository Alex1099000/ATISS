"""Run fast evaluation for selected ATISS checkpoints."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

VAL_LINE_RE = re.compile(r"^val epoch: (?P<epoch>\d+) - loss: (?P<loss>[0-9.]+)")
MODEL_RE = re.compile(r"^model_(?P<epoch>\d+)$")
DEFAULT_CHECKPOINT_FILES = [
    "checkpoints/bedrooms_lightning/model_00100",
    "checkpoints/bedrooms_lightning/model_00200",
    "checkpoints/bedrooms_lightning/model_00300",
    "checkpoints/bedrooms_lightning/model_00500",
    "checkpoints/bedrooms_lightning/model_00800",
]


def parse_validation_losses(stats_file):
    validation_losses = []
    with open(stats_file) as stats:
        for line in stats:
            match = VAL_LINE_RE.match(line)
            if match:
                validation_losses.append(
                    {
                        "epoch": int(match.group("epoch")),
                        "loss": float(match.group("loss")),
                    }
                )
    if not validation_losses:
        raise RuntimeError(f"No validation losses found in {stats_file}")
    return validation_losses


def available_checkpoints(checkpoint_directory):
    checkpoints = []
    for path in Path(checkpoint_directory).glob("model_*"):
        match = MODEL_RE.match(path.name)
        if match:
            checkpoints.append(
                {
                    "epoch": int(match.group("epoch")),
                    "path": path,
                }
            )
    if not checkpoints:
        raise RuntimeError(f"No model_* checkpoints found in {checkpoint_directory}")
    return sorted(checkpoints, key=lambda item: item["epoch"])


def nearest_validation_loss(checkpoint_epoch, validation_losses):
    return min(
        validation_losses,
        key=lambda item: (abs(item["epoch"] - checkpoint_epoch), item["epoch"]),
    )


def checkpoint_metadata(checkpoint_path, validation_losses):
    path = Path(checkpoint_path)
    match = MODEL_RE.match(path.name)
    checkpoint_epoch = int(match.group("epoch")) if match else None
    validation = None
    if checkpoint_epoch is not None and validation_losses:
        validation = nearest_validation_loss(checkpoint_epoch, validation_losses)
    return {
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_path": path,
        "validation_epoch": validation["epoch"] if validation else None,
        "validation_loss": validation["loss"] if validation else None,
    }


def selected_checkpoint_files(args):
    return args.checkpoint_files or DEFAULT_CHECKPOINT_FILES


def select_checkpoints(checkpoint_directory, stats_file, top_k):
    validation_losses = parse_validation_losses(stats_file)
    ranked = []
    for checkpoint in available_checkpoints(checkpoint_directory):
        validation = nearest_validation_loss(checkpoint["epoch"], validation_losses)
        ranked.append(
            {
                "checkpoint_epoch": checkpoint["epoch"],
                "checkpoint_path": checkpoint["path"],
                "validation_epoch": validation["epoch"],
                "validation_loss": validation["loss"],
            }
        )
    return sorted(
        ranked,
        key=lambda item: (item["validation_loss"], item["checkpoint_epoch"]),
    )[:top_k]


def run_eval(args, checkpoint):
    output_directory = Path(args.output_directory) / checkpoint["checkpoint_path"].name
    command = [
        sys.executable,
        "scripts/evaluate_atiss_metrics.py",
        args.config_file,
        str(output_directory),
        args.path_to_pickled_3d_future_models,
        args.path_to_floor_plan_textures,
        "--weight_file",
        str(checkpoint["checkpoint_path"]),
        "--processed_dataset_directory",
        args.processed_dataset_directory,
        "--annotation_file",
        args.annotation_file,
        "--n_synthesized_scenes",
        str(args.n_synthesized_scenes),
        "--fid_repeats",
        str(args.fid_repeats),
        "--seed",
        str(args.seed),
        "--skip_classifier",
    ]
    if args.skip_fid:
        command.append("--skip_fid")
    if args.export_meshes:
        command.append("--export_meshes")

    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)

    metrics_path = output_directory / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as metrics_file:
            metrics = json.load(metrics_file)
    return {
        "checkpoint": checkpoint["checkpoint_path"].name,
        "checkpoint_path": str(checkpoint["checkpoint_path"]),
        "checkpoint_epoch": checkpoint["checkpoint_epoch"],
        "validation_epoch": checkpoint["validation_epoch"],
        "validation_loss": checkpoint["validation_loss"],
        "output_directory": str(output_directory),
        "metrics": metrics,
    }


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Evaluate selected checkpoints with a fast metrics setup."
    )
    parser.add_argument(
        "--checkpoint_directory",
        default="checkpoints/bedrooms_lightning",
        help="Directory containing model_* checkpoints and stats.txt",
    )
    parser.add_argument(
        "--stats_file",
        default=None,
        help="Optional path to stats.txt for annotating checkpoints with val loss.",
    )
    parser.add_argument(
        "--checkpoint_files",
        nargs="+",
        default=None,
        help=(
            "Explicit checkpoint paths to evaluate. Defaults to a curated set: "
            + ", ".join(DEFAULT_CHECKPOINT_FILES)
        ),
    )
    parser.add_argument(
        "--config_file",
        default="configs/atiss/bedrooms_zakharov.yaml",
        help="ATISS config file used by evaluate_atiss_metrics.py",
    )
    parser.add_argument(
        "--output_directory",
        default="outputs/eval_best_bedrooms_checkpoints",
        help="Directory where per-checkpoint eval outputs are written",
    )
    parser.add_argument(
        "--path_to_pickled_3d_future_models",
        default="/workspace-SR006.nfs2/datasets/atiss_processed/bedrooms/threed_future_model_bedroom.pkl",
        help="Pickled 3D-FUTURE dataset path",
    )
    parser.add_argument(
        "--path_to_floor_plan_textures",
        default="/workspace-SR006.nfs2/datasets/3d-front/3D-FRONT-texture",
        help="3D-FRONT floor texture directory",
    )
    parser.add_argument(
        "--processed_dataset_directory",
        default="/workspace-SR006.nfs2/datasets/atiss_processed/bedrooms",
        help="Processed ATISS dataset directory",
    )
    parser.add_argument(
        "--annotation_file",
        default="configs/splits/bedroom_threed_front_splits.csv",
        help="Train/val/test split CSV",
    )
    parser.add_argument("--n_synthesized_scenes", type=int, default=150)
    parser.add_argument("--fid_repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--export_meshes", action="store_true")
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    stats_file = args.stats_file or os.path.join(args.checkpoint_directory, "stats.txt")
    validation_losses = []
    if os.path.exists(stats_file):
        validation_losses = parse_validation_losses(stats_file)
    selected = [
        checkpoint_metadata(checkpoint_path, validation_losses)
        for checkpoint_path in selected_checkpoint_files(args)
    ]

    print("Selected checkpoints:")
    for item in selected:
        message = item["checkpoint_path"].name
        if item["validation_epoch"] is not None:
            message += " nearest val epoch {}, val loss {:.5f}".format(
                item["validation_epoch"], item["validation_loss"]
            )
        print(message)

    Path(args.output_directory).mkdir(parents=True, exist_ok=True)
    results = []
    for checkpoint in tqdm(selected, desc="Evaluating checkpoints"):
        tqdm.write(f"Evaluating {checkpoint['checkpoint_path'].name}")
        results.append(run_eval(args, checkpoint))
    summary_path = Path(args.output_directory) / "summary.json"
    with open(summary_path, "w") as summary_file:
        json.dump(results, summary_file, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
