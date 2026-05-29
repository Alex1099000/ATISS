"""MLflow pyfunc wrapper for furniture layout inference."""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc

from furniture_layout_generation.config import as_container, hydra_config
from furniture_layout_generation.data import download_data
from furniture_layout_generation.inference import (
    infer_layout_with_components,
    inference_paths_from_config,
    load_inference_components,
    walls_from_points,
)


class FurnitureLayoutPyFunc(mlflow.pyfunc.PythonModel):
    """Serve the ATISS inference pipeline through MLflow pyfunc."""

    def load_context(self, context) -> None:
        download_data()
        with hydra_config("config") as config:
            self.paths = inference_paths_from_config(as_container(config))
        self.components = load_inference_components(self.paths)

    def predict(self, context, model_input, params: dict[str, Any] | None = None):
        request = _first_request(model_input)
        walls = request.get("walls")
        if walls is None and request.get("points") is not None:
            walls = walls_from_points(request["points"])
        if walls is None:
            raise ValueError("Model input must contain either 'walls' or 'points'")

        result = infer_layout_with_components(
            walls=walls,
            components=self.components,
            checkpoint=self.paths.checkpoint,
            output_directory=request.get("output_dir", "outputs/mlflow_inference"),
            seed=int(request.get("seed", 27)),
            render_images=bool(request.get("render_images", False)),
        )
        return [result]


def _first_request(model_input) -> dict[str, Any]:
    if isinstance(model_input, dict):
        return model_input
    if hasattr(model_input, "to_dict"):
        records = model_input.to_dict(orient="records")
        if records:
            return records[0]
    if isinstance(model_input, list) and model_input:
        return model_input[0]
    raise ValueError("Empty MLflow model input")


def log_model(
    artifact_path: str = "furniture-layout-pyfunc",
    registered_model_name: str | None = "furniture-layout-generation",
    tracking_uri: str = "http://127.0.0.1:8080",
    experiment_name: str = "furniture-layout-generation",
) -> None:
    """Log the project inference wrapper to the active MLflow run."""

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="furniture-layout-pyfunc"):
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=FurnitureLayoutPyFunc(),
            registered_model_name=registered_model_name,
        )
