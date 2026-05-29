"""Evaluate an OpenRouter LLM as a layout-generation baseline.

The script mirrors ``evaluate_atiss_metrics.py`` as closely as possible:
it uses the same test rooms, converts LLM JSON layouts to ATISS scene tensors,
renders generated scenes with the same renderer, and computes KL/FID/geometry
metrics through the already implemented helpers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
from evaluate_atiss_metrics import (
    aggregate_geometry_metrics,
    compute_fid,
    compute_kl,
    recreate_directory,
    resolve_config_paths,
    scene_geometry_metrics,
)
from pyrr import Matrix44
from shapely.affinity import translate
from shapely.geometry import Point, Polygon
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.utils import render
from tqdm import tqdm
from training_utils import load_config
from utils import create_scene, export_scene, floor_plan_from_scene

from scene_synthesis.datasets import filter_function, get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.utils import get_textured_objects

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3-flash-preview"


def load_dotenv(path=".env"):
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)


def room_polygon_from_scene(scene):
    vertices, faces = scene.floor_plan
    vertices = vertices - scene.floor_plan_centroid
    polygons = []
    for face in faces:
        points = [
            (float(vertices[index, 0]), float(vertices[index, 2])) for index in face
        ]
        polygon = Polygon(points)
        if polygon.is_valid and polygon.area > 0:
            polygons.append(polygon)
    if not polygons:
        raise ValueError(f"Could not build room polygon for scene {scene.scene_id}")
    from shapely.ops import unary_union

    return unary_union(polygons).buffer(0)


def room_payload(scene, allowed_categories):
    polygon = room_polygon_from_scene(scene)
    points = [
        {"x": round(float(x_coord), 3), "y": round(float(y_coord), 3)}
        for x_coord, y_coord in list(polygon.exterior.coords)[:-1]
    ]
    return {
        "room_type": "bedroom",
        "coordinate_system": {
            "units": "meters",
            "origin": "room_center",
            "x_axis": "right",
            "y_axis": "forward",
        },
        "room_polygon": points,
        "allowed_categories": allowed_categories,
        "constraints": {
            "max_objects": 12,
            "objects_must_be_inside_polygon": True,
            "avoid_collisions": True,
            "use_realistic_bedroom_furniture": True,
        },
    }


def system_prompt():
    return (
        "You generate physically plausible bedroom furniture layouts. "
        "Return only valid JSON matching the requested schema. Do not include "
        "markdown, comments, explanations, or trailing commas. Use only allowed "
        "categories. All object centers must be inside the room polygon. "
        "Avoid object overlaps."
    )


def user_prompt(payload):
    schema = {
        "objects": [
            {
                "category": "one of allowed_categories",
                "center_x": 0.0,
                "center_y": 0.0,
                "size_x": 1.6,
                "size_y": 2.0,
                "rotation_degrees": 0.0,
            }
        ]
    }
    return (
        "Generate a furniture layout for this bedroom.\n\n"
        f"Input JSON:\n{json.dumps(payload, indent=2)}\n\n"
        "Return JSON schema:\n"
        f"{json.dumps(schema, indent=2)}"
    )


def openrouter_chat(api_key, model, prompt, temperature, max_tokens, timeout):
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com",
            "X-Title": "ATISS LLM Baseline",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body}") from exc

    if "choices" not in payload:
        raise RuntimeError(
            f"OpenRouter returned no choices: {json.dumps(payload)[:1000]}"
        )
    return payload["choices"][0]["message"]["content"]


def extract_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise
        return json.loads(match.group(0))


def cached_llm_response(
    cache_dir,
    scene_id,
    api_key,
    model,
    prompt,
    temperature,
    max_tokens,
    timeout,
    retries,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{scene_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    last_error = None
    for attempt in range(retries):
        try:
            content = openrouter_chat(
                api_key, model, prompt, temperature, max_tokens, timeout
            )
            parsed = extract_json(content)
            cache_path.write_text(json.dumps(parsed, indent=2))
            return parsed
        except (
            RuntimeError,
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"OpenRouter request failed for {scene_id}: {last_error}")


def class_size_defaults(raw_dataset, encoded_dataset):
    class_labels = list(encoded_dataset.class_labels)
    sizes_by_class = {label: [] for label in class_labels}
    for idx in range(len(raw_dataset)):
        room = raw_dataset[idx]
        classes = np.asarray(room.class_labels).argmax(axis=-1)
        sizes = np.asarray(room.sizes)
        for class_idx, size in zip(classes, sizes, strict=False):
            label = class_labels[int(class_idx)]
            if label not in {"start", "end"}:
                sizes_by_class[label].append(size)

    defaults = {}
    global_sizes = []
    for label, values in sizes_by_class.items():
        if values:
            median = np.median(np.asarray(values), axis=0)
            defaults[label] = median.astype(np.float32)
            global_sizes.append(median)
    fallback = (
        np.median(np.asarray(global_sizes), axis=0).astype(np.float32)
        if global_sizes
        else np.array([0.35, 0.35, 0.35], dtype=np.float32)
    )
    return defaults, fallback


def object_polygon(center_x, center_y, half_size_x, half_size_y, angle):
    local_corners = np.array(
        [
            [-half_size_x, -half_size_y],
            [half_size_x, -half_size_y],
            [half_size_x, half_size_y],
            [-half_size_x, half_size_y],
        ],
        dtype=np.float64,
    )
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    corners = local_corners.dot(rotation.T) + np.array([center_x, center_y])
    return Polygon(corners).buffer(0)


def move_polygon_inside(polygon, room_polygon):
    if polygon.difference(room_polygon).area <= 1e-6:
        return polygon
    source = polygon.centroid
    best = polygon
    best_outside = polygon.difference(room_polygon).area
    minx, miny, maxx, maxy = room_polygon.bounds
    candidates = [
        room_polygon.representative_point(),
        room_polygon.centroid,
        *[
            Point(x_coord, y_coord)
            for x_coord in np.linspace(minx, maxx, 9)
            for y_coord in np.linspace(miny, maxy, 9)
            if room_polygon.contains(Point(x_coord, y_coord))
        ],
    ]
    for target in candidates:
        moved = translate(best, xoff=target.x - source.x, yoff=target.y - source.y)
        outside = moved.difference(room_polygon).area
        if outside < best_outside:
            best = moved
            best_outside = outside
        if outside <= 1e-6:
            return moved
    return best


def llm_layout_to_scene(
    llm_layout, scene, encoded_dataset, size_defaults, fallback_size
):
    class_labels = list(encoded_dataset.class_labels)
    allowed = [label for label in class_labels if label not in {"start", "end"}]
    class_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    room_poly = room_polygon_from_scene(scene)

    output_classes = [
        np.eye(len(class_labels), dtype=np.float32)[class_to_idx["start"]]
    ]
    output_translations = [np.zeros(3, dtype=np.float32)]
    output_sizes = [np.zeros(3, dtype=np.float32)]
    output_angles = [np.zeros(1, dtype=np.float32)]
    placed_polygons = []

    for raw_object in llm_layout.get("objects", [])[:12]:
        category = str(raw_object.get("category", "")).strip()
        if category not in allowed:
            continue

        default_size = size_defaults.get(category, fallback_size)
        full_size_x = float(raw_object.get("size_x", default_size[0] * 2))
        full_size_y = float(raw_object.get("size_y", default_size[2] * 2))
        half_size_x = max(0.05, min(full_size_x / 2, 2.5))
        half_size_y = max(0.05, min(full_size_y / 2, 2.5))
        angle = np.deg2rad(float(raw_object.get("rotation_degrees", 0.0)))
        center_x = float(raw_object.get("center_x", 0.0))
        center_y = float(raw_object.get("center_y", 0.0))

        polygon = object_polygon(center_x, center_y, half_size_x, half_size_y, angle)
        polygon = move_polygon_inside(polygon, room_poly)
        centroid = polygon.centroid

        collides = False
        for placed in placed_polygons:
            if polygon.intersection(placed).area / max(polygon.area, 1e-6) > 0.35:
                collides = True
                break
        if collides:
            continue
        placed_polygons.append(polygon)

        output_classes.append(
            np.eye(len(class_labels), dtype=np.float32)[class_to_idx[category]]
        )
        output_translations.append(
            np.array([centroid.x, float(default_size[1]), centroid.y], dtype=np.float32)
        )
        output_sizes.append(
            np.array(
                [half_size_x, float(default_size[1]), half_size_y], dtype=np.float32
            )
        )
        output_angles.append(np.array([angle], dtype=np.float32))

    output_classes.append(
        np.eye(len(class_labels), dtype=np.float32)[class_to_idx["end"]]
    )
    output_translations.append(np.zeros(3, dtype=np.float32))
    output_sizes.append(np.zeros(3, dtype=np.float32))
    output_angles.append(np.zeros(1, dtype=np.float32))

    return {
        "class_labels": np.asarray(output_classes, dtype=np.float32),
        "translations": np.asarray(output_translations, dtype=np.float32),
        "sizes": np.asarray(output_sizes, dtype=np.float32),
        "angles": np.asarray(output_angles, dtype=np.float32),
    }


def numpy_scene_to_torch_boxes(scene_np):
    return {
        key: torch.from_numpy(value[None]).float() for key, value in scene_np.items()
    }


def render_llm_scene(
    scene_np,
    current_scene,
    objects_dataset,
    classes,
    floor_texture_directory,
    output_directory,
    scene_index,
    render_scene,
    export_meshes,
    scene_renderer,
    window_size,
    camera_position,
    camera_target,
    up_vector,
    background,
):
    if not render_scene and not export_meshes:
        return

    floor_plan, tr_floor, _ = floor_plan_from_scene(
        current_scene, floor_texture_directory
    )
    bbox_params_t = np.concatenate(
        [
            scene_np["class_labels"],
            scene_np["translations"],
            scene_np["sizes"],
            scene_np["angles"],
        ],
        axis=-1,
    )[None]
    renderables, trimesh_meshes = get_textured_objects(
        bbox_params_t, objects_dataset, classes
    )
    renderables += floor_plan
    trimesh_meshes += tr_floor

    if render_scene:
        path_to_image = os.path.join(output_directory, f"{scene_index:05d}")
        render(
            renderables,
            behaviours=[SaveFrames(path_to_image + ".png", 1)],
            size=window_size,
            camera_position=camera_position,
            camera_target=camera_target,
            up_vector=up_vector,
            background=background,
            n_frames=1,
            scene=scene_renderer,
        )

    if export_meshes:
        path_to_objs = os.path.join(output_directory, f"{scene_index:05d}_scene")
        os.makedirs(path_to_objs, exist_ok=True)
        export_scene(path_to_objs, trimesh_meshes)


def generate_llm_scenes(
    raw_dataset,
    encoded_dataset,
    objects_dataset,
    floor_texture_directory,
    output_directory,
    n_scenes,
    api_key,
    model,
    temperature,
    max_tokens,
    timeout,
    retries,
    cache_dir,
    render_scenes,
    export_meshes,
    window_size,
    room_side,
    camera_position,
    camera_target,
    up_vector,
    background,
):
    os.makedirs(output_directory, exist_ok=True)
    classes = np.array(encoded_dataset.class_labels)
    allowed_categories = [
        label for label in classes.tolist() if label not in {"start", "end"}
    ]
    size_defaults, fallback_size = class_size_defaults(raw_dataset, encoded_dataset)
    synthesized_scenes = []
    geometry_scene_metrics = []

    scene_renderer = None
    if render_scenes:
        scene_renderer = create_scene(size=window_size, background=background)
        scene_renderer.up_vector = up_vector
        scene_renderer.camera_target = camera_target
        scene_renderer.camera_position = camera_position
        scene_renderer.light = camera_position
        scene_renderer.camera_matrix = Matrix44.orthogonal_projection(
            left=-room_side,
            right=room_side,
            bottom=room_side,
            top=-room_side,
            near=0.1,
            far=6,
        )

    for scene_index in tqdm(range(n_scenes), desc="Generating LLM scenes"):
        dataset_index = np.random.choice(len(raw_dataset))
        current_scene = raw_dataset[dataset_index]
        scene_id = f"{scene_index:05d}_{current_scene.scene_id}"
        payload = room_payload(current_scene, allowed_categories)
        try:
            llm_layout = cached_llm_response(
                cache_dir,
                scene_id,
                api_key,
                model,
                user_prompt(payload),
                temperature,
                max_tokens,
                timeout,
                retries,
            )
        except RuntimeError as exc:
            print(f"Warning: {exc}. Using an empty layout for this scene.", flush=True)
            llm_layout = {"objects": []}
        scene_np = llm_layout_to_scene(
            llm_layout, current_scene, encoded_dataset, size_defaults, fallback_size
        )
        synthesized_scenes.append(scene_np)
        geometry_scene_metrics.append(
            scene_geometry_metrics(current_scene, numpy_scene_to_torch_boxes(scene_np))
        )
        render_llm_scene(
            scene_np,
            current_scene,
            objects_dataset,
            classes,
            floor_texture_directory,
            output_directory,
            scene_index,
            render_scenes,
            export_meshes,
            scene_renderer,
            window_size,
            camera_position,
            camera_target,
            up_vector,
            background,
        )

    return synthesized_scenes, aggregate_geometry_metrics(geometry_scene_metrics)


def parse_args(argv):
    parser = argparse.ArgumentParser("Evaluate an OpenRouter LLM layout baseline")
    parser.add_argument("config_file")
    parser.add_argument("output_directory")
    parser.add_argument("path_to_pickled_3d_future_models")
    parser.add_argument("path_to_floor_plan_textures")
    parser.add_argument("--processed_dataset_directory", required=True)
    parser.add_argument(
        "--annotation_file", default="configs/splits/bedroom_threed_front_splits.csv"
    )
    parser.add_argument("--n_synthesized_scenes", type=int, default=100)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_key_env", default="OPENROUTER_API_KEY")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--split", choices=["training", "validation"], default="validation"
    )
    parser.add_argument(
        "--window_size", type=lambda x: tuple(map(int, x.split(","))), default="256,256"
    )
    parser.add_argument("--room_side", type=float, default=3.1)
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,4,0",
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
    )
    parser.add_argument(
        "--up_vector", type=lambda x: tuple(map(float, x.split(","))), default="0,0,-1"
    )
    parser.add_argument(
        "--background", type=lambda x: list(map(float, x.split(","))), default="1,1,1,1"
    )
    parser.add_argument("--export_meshes", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--skip_classifier", action="store_true")
    parser.add_argument("--fid_repeats", type=int, default=10)
    parser.add_argument("--classifier_repeats", type=int, default=10)
    parser.add_argument("--classifier_epochs", type=int, default=10)
    parser.add_argument("--classifier_batch_size", type=int, default=256)
    parser.add_argument("--classifier_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=27)
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    load_dotenv(args.dotenv)
    api_key = os.environ.get(args.api_key_env)
    if not api_key and not args.skip_generation:
        raise RuntimeError(f"Set {args.api_key_env} in {args.dotenv} or environment")

    logging.getLogger("trimesh").setLevel(logging.ERROR)
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running classifier code on", device)

    os.makedirs(args.output_directory, exist_ok=True)
    generated_directory = os.path.join(args.output_directory, "generated")
    cache_dir = Path(args.output_directory) / "openrouter_cache"
    if not args.skip_generation:
        recreate_directory(generated_directory)
    metrics = {}

    config = resolve_config_paths(load_config(args.config_file), args.config_file)
    raw_dataset, encoded_dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"]),
        ),
        split=config["validation"].get("splits", ["test"]),
    )
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_future_models
    )

    if args.skip_generation:
        synthesized_scenes = None
    else:
        synthesized_scenes, geometry_metrics = generate_llm_scenes(
            raw_dataset,
            encoded_dataset,
            objects_dataset,
            args.path_to_floor_plan_textures,
            generated_directory,
            args.n_synthesized_scenes,
            api_key,
            args.model,
            args.temperature,
            args.max_tokens,
            args.timeout,
            args.retries,
            cache_dir,
            render_scenes=not args.skip_fid or not args.skip_classifier,
            export_meshes=args.export_meshes,
            window_size=args.window_size,
            room_side=args.room_side,
            camera_position=args.camera_position,
            camera_target=args.camera_target,
            up_vector=args.up_vector,
            background=args.background,
        )
        metrics["col_obj"] = geometry_metrics["col_obj"]
        metrics["r_out"] = geometry_metrics["r_out"]
        metrics["geometry"] = geometry_metrics
        metrics["kl_class_labels"] = compute_kl(
            raw_dataset,
            encoded_dataset,
            synthesized_scenes,
            os.path.join(args.output_directory, f"{args.split}_stats.npz"),
        )

    if not args.skip_fid:
        metrics["fid"] = compute_fid(
            args.processed_dataset_directory,
            args.annotation_file,
            generated_directory,
            args.output_directory,
            n_repeats=args.fid_repeats,
        )

    if not args.skip_classifier:
        from evaluate_atiss_metrics import train_synthetic_vs_real_classifier

        metrics["synthetic_vs_real_accuracy"] = train_synthetic_vs_real_classifier(
            args.processed_dataset_directory,
            args.annotation_file,
            generated_directory,
            os.path.join(args.output_directory, "classifier"),
            args.classifier_batch_size,
            args.classifier_num_workers,
            args.classifier_epochs,
            args.classifier_repeats,
            device,
        )

    metrics_path = os.path.join(args.output_directory, "metrics.json")
    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
