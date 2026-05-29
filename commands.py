"""Public project CLI.

Use Fire for command dispatch and Hydra Compose API for configuration loading.
Legacy ATISS scripts remain available internally, but reviewer-facing commands
are exposed here.
"""

from __future__ import annotations

import json

import fire

from furniture_layout_generation.config import as_container, hydra_config
from furniture_layout_generation.data import download_data
from furniture_layout_generation.export import (
    export_onnx_model,
    export_tensorrt_engine,
)
from furniture_layout_generation.inference import (
    infer_layout,
    inference_paths_from_config,
    load_walls_from_json,
)
from furniture_layout_generation.mlflow_model import log_model
from furniture_layout_generation.runner import run_command, run_python


def _config(overrides: list[str] | None = None) -> dict:
    with hydra_config("config", overrides) as config:
        return as_container(config)


def _flag(enabled: bool, name: str) -> list[str]:
    return [name] if enabled else []


class Commands:
    """Furniture layout generation commands."""

    def config(self, *overrides: str) -> None:
        """Print the resolved Hydra configuration."""

        print(json.dumps(_config(list(overrides)), indent=2))

    def download_data(self) -> None:
        """Pull data and model artifacts from DVC storage."""

        download_data()

    def preprocess(self, *overrides: str) -> None:
        """Preprocess raw 3D-FRONT/3D-FUTURE data."""

        config = _config(list(overrides))
        data = config["data"]
        preprocess = config["preprocess"]
        run_python(
            "scripts/preprocess_data.py",
            [
                data["processed_dir"],
                data["front_dir"],
                data["future_dir"],
                data["model_info"],
                data["floor_textures"],
                "--path_to_invalid_scene_ids",
                data["invalid_scene_ids"],
                "--path_to_invalid_bbox_jids",
                data["invalid_bbox_jids"],
                "--annotation_file",
                data["annotation_file"],
                "--room_side",
                preprocess["room_side"],
                "--dataset_filtering",
                data["dataset_filtering"],
                "--up_vector",
                preprocess["up_vector"],
                "--background",
                preprocess["background"],
                "--camera_target",
                preprocess["camera_target"],
                "--camera_position",
                preprocess["camera_position"],
                "--window_size",
                preprocess["window_size"],
                *_flag(preprocess["without_lamps"], "--without_lamps"),
            ],
        )

    def pickle_future(self, *overrides: str) -> None:
        """Create the pickled 3D-FUTURE dataset used during inference."""

        config = _config(list(overrides))
        data = config["data"]
        run_python(
            "scripts/pickle_threed_future_dataset.py",
            [
                data["processed_dir"],
                data["front_dir"],
                data["future_dir"],
                data["model_info"],
                "--path_to_invalid_bbox_jids",
                data["invalid_bbox_jids"],
                "--path_to_invalid_scene_ids",
                data["invalid_scene_ids"],
                "--annotation_file",
                data["annotation_file"],
                "--dataset_filtering",
                data["dataset_filtering"],
            ],
        )

    def train(self, *overrides: str) -> None:
        """Train ATISS with the PyTorch Lightning training entrypoint."""

        config = _config(list(overrides))
        if config["train"]["pull_data"]:
            download_data()
        train = config["train"]
        args = [
            train["config_file"],
            train["output_dir"],
            "--n_processes",
            train["n_processes"],
            "--seed",
            train["seed"],
            "--experiment_tag",
            train["experiment_tag"],
        ]
        if train["with_wandb_logger"]:
            args.append("--with_wandb_logger")
        if train["with_mlflow_logger"]:
            args.extend(
                [
                    "--with_mlflow_logger",
                    "--mlflow_tracking_uri",
                    train["mlflow_tracking_uri"],
                    "--mlflow_experiment_name",
                    train["mlflow_experiment_name"],
                ],
            )
        run_python("scripts/train_network_lightning.py", args)

    def generate(self, *overrides: str) -> None:
        """Generate top-down scene renderings and optional 3D assets."""

        config = _config(list(overrides))
        data = config["data"]
        model = config["model"]
        evaluation = config["eval"]
        run_python(
            "scripts/generate_scenes.py",
            [
                data["atiss_config"],
                "outputs/generated",
                data["pickled_3d_future"],
                data["floor_textures"],
                "--weight_file",
                model["checkpoint"],
                "--n_sequences",
                10,
                "--without_screen",
                "--with_orthographic_projection",
                "--window_size",
                evaluation["window_size"],
                "--room_side",
                evaluation["room_side"],
                "--camera_position",
                "0,4,0",
                "--camera_target",
                "0,0,0",
                "--up_vector",
                "0,0,-1",
            ],
        )

    def evaluate(self, *overrides: str) -> None:
        """Compute KL, FID, and optional classifier metrics."""

        config = _config(list(overrides))
        data = config["data"]
        model = config["model"]
        evaluation = config["eval"]
        run_python(
            "scripts/evaluate_atiss_metrics.py",
            [
                data["atiss_config"],
                evaluation["output_dir"],
                data["pickled_3d_future"],
                data["floor_textures"],
                "--weight_file",
                model["checkpoint"],
                "--processed_dataset_directory",
                data["processed_dir"],
                "--annotation_file",
                data["annotation_file"],
                "--n_synthesized_scenes",
                evaluation["n_synthesized_scenes"],
                "--fid_repeats",
                evaluation["fid_repeats"],
                *_flag(evaluation["skip_classifier"], "--skip_classifier"),
            ],
        )

    def evaluate_llm_baseline(self, *overrides: str) -> None:
        """Compute evaluation metrics for the OpenRouter LLM baseline."""

        config = _config(list(overrides))
        data = config["data"]
        evaluation = config["eval"]
        run_python(
            "scripts/evaluate_openrouter_llm_baseline.py",
            [
                data["atiss_config"],
                evaluation["output_dir"],
                data["pickled_3d_future"],
                data["floor_textures"],
                "--processed_dataset_directory",
                data["processed_dir"],
                "--annotation_file",
                data["annotation_file"],
                "--n_synthesized_scenes",
                evaluation["n_synthesized_scenes"],
                "--fid_repeats",
                evaluation["fid_repeats"],
                *_flag(evaluation["skip_classifier"], "--skip_classifier"),
            ],
        )

    def export_onnx(self, *overrides: str) -> None:
        """Export one ATISS autoregressive decode step to ONNX."""

        config = _config(list(overrides))
        data = config["data"]
        export = config["export"]
        download_data()
        export_onnx_model(
            checkpoint_path=export["checkpoint_path"],
            output_path=export["output_path"],
            config_file=data["atiss_config"],
            pickled_3d_future=data["pickled_3d_future"],
            floor_textures=data["floor_textures"],
            opset=export["opset"],
        )

    def export_tensorrt(self, *overrides: str) -> None:
        """Build a TensorRT engine from the configured ONNX artifact."""

        config = _config(list(overrides))
        tensorrt = config["export"]["tensorrt"]
        export_tensorrt_engine(
            onnx_path=tensorrt["onnx_path"],
            engine_path=tensorrt["engine_path"],
            fp16=tensorrt["fp16"],
        )

    def serve_fastapi(self, *overrides: str) -> None:
        """Start the project FastAPI inference service."""

        config = _config(list(overrides))
        serve = config["serve"]
        download_data()
        run_command(
            [
                "uvicorn",
                "furniture_layout_generation.server:app",
                "--host",
                serve["host"],
                "--port",
                serve["port"],
            ],
        )

    def serve_mlflow(self, *overrides: str) -> None:
        """Serve the exported model artifact with MLflow Serving."""

        config = _config(list(overrides))
        serve = config["serve"]
        download_data()
        run_command(
            [
                "mlflow",
                "models",
                "serve",
                "-m",
                serve["model_uri"],
                "--host",
                serve["host"],
                "--port",
                serve["port"],
                "--no-conda",
            ],
        )

    def log_mlflow_model(
        self,
        artifact_path: str = "furniture-layout-pyfunc",
        registered_model_name: str = "furniture-layout-generation",
        *overrides: str,
    ) -> None:
        """Log the project inference wrapper to the active MLflow run."""

        config = _config(list(overrides))
        serve = config["serve"]
        download_data()
        log_model(
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            tracking_uri=serve["tracking_uri"],
            experiment_name=serve["experiment_name"],
        )

    def infer(
        self,
        input_json: str = "configs/infer/example_room.json",
        output_dir: str = "outputs/interactive_inference",
        checkpoint: str | None = None,
        render_images: bool = True,
    ) -> None:
        """Run ATISS inference for a room JSON."""

        config = _config([])
        download_data()
        walls = load_walls_from_json(input_json)
        response = infer_layout(
            walls=walls,
            paths=inference_paths_from_config(config, checkpoint),
            output_directory=output_dir,
            render_images=render_images,
        )
        print(json.dumps(response, indent=2))


def main() -> None:
    """Run the Fire CLI."""

    fire.Fire(Commands)


if __name__ == "__main__":
    main()
