"""Root inference entrypoint.

Example:
python infer.py --input_json configs/infer/example_room.json --checkpoint checkpoints/bedrooms_lightning/model_00300
"""

from __future__ import annotations

import json

import fire

from furniture_layout_generation.config import as_container, hydra_config
from furniture_layout_generation.data import download_data
from furniture_layout_generation.inference import (
    infer_layout,
    inference_paths_from_config,
    load_walls_from_json,
)


def main(
    input_json: str = "configs/infer/example_room.json",
    output_dir: str = "outputs/interactive_inference",
    checkpoint: str | None = None,
    seed: int = 27,
    render_images: bool = True,
) -> None:
    """Run layout generation from a JSON file containing walls or points."""

    download_data()
    with hydra_config("config") as config:
        paths = inference_paths_from_config(as_container(config), checkpoint)
    response = infer_layout(
        walls=load_walls_from_json(input_json),
        paths=paths,
        output_directory=output_dir,
        seed=seed,
        render_images=render_images,
    )
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
