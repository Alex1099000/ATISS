"""FastAPI inference server for interactive ATISS layout generation."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from furniture_layout_generation.config import as_container, hydra_config
from furniture_layout_generation.data import download_data
from furniture_layout_generation.inference import (
    infer_layout_with_components,
    inference_paths_from_config,
    load_inference_components,
    walls_from_points,
)


class Wall(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class PredictionRequest(BaseModel):
    room_id: str = "room"
    walls: list[Wall] | None = None
    points: list[tuple[float, float]] | None = None
    seed: int = 27
    render_images: bool = True
    output_dir: str = "outputs/server_inference"


class PredictionResponse(BaseModel):
    room_id: str
    n_objects: int
    checkpoint: str
    objects: list[dict[str, Any]]
    renders: dict[str, str] = Field(default_factory=dict)
    scene_path: str | None = None


app = FastAPI(title="Furniture Layout Generation API", version="1.0.0")


@lru_cache(maxsize=1)
def cached_components():
    download_data()
    with hydra_config("config") as config:
        paths = inference_paths_from_config(as_container(config))
    return paths, load_inference_components(paths)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    paths, components = cached_components()
    if request.walls:
        walls = [wall.model_dump() for wall in request.walls]
    elif request.points:
        walls = walls_from_points(request.points)
    else:
        raise HTTPException(
            status_code=400,
            detail="Request must contain either walls or points",
        )

    result = infer_layout_with_components(
        walls=walls,
        components=components,
        checkpoint=paths.checkpoint,
        output_directory=request.output_dir,
        seed=request.seed,
        render_images=request.render_images,
    )
    return PredictionResponse(room_id=request.room_id, **result)
