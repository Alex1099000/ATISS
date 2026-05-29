"""Inference, postprocessing, and rendering utilities for ATISS."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import moderngl
import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw
from pyrr import Matrix44
from shapely.affinity import scale, translate
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points, triangulate, unary_union
from simple_3dviz import Lines, Mesh, Scene
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.utils import render
from yaml import CLoader as Loader
from yaml import load

from scene_synthesis.datasets import filter_function, get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network
from scene_synthesis.utils import get_textured_objects

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_CONFIG_FILE = "configs/atiss/bedrooms_zakharov.yaml"
DEFAULT_CHECKPOINT = "checkpoints/bedrooms_lightning/model_00300"
DEFAULT_PICKLED_3D_FUTURE = (
    "/workspace-SR006.nfs2/datasets/atiss_processed/bedrooms/"
    "threed_future_model_bedroom.pkl"
)
DEFAULT_FLOOR_TEXTURES = "/workspace-SR006.nfs2/datasets/3d-front/3D-FRONT-texture"


@dataclass
class InferencePaths:
    config_file: str = DEFAULT_CONFIG_FILE
    checkpoint: str = DEFAULT_CHECKPOINT
    pickled_3d_future: str = DEFAULT_PICKLED_3D_FUTURE
    floor_textures: str = DEFAULT_FLOOR_TEXTURES


def inference_paths_from_config(config: dict, checkpoint: str | None = None):
    """Build inference paths from the resolved Hydra config."""

    data = config["data"]
    model = config["model"]
    return InferencePaths(
        config_file=data["atiss_config"],
        checkpoint=checkpoint or model["checkpoint"],
        pickled_3d_future=data["pickled_3d_future"],
        floor_textures=data["floor_textures"],
    )


class PolygonRoom:
    """Small room object matching the subset of CachedRoom used by inference."""

    def __init__(
        self, walls, room_layout_size=(64, 64), room_side=3.1, room_scale=1.35
    ):
        self.scene_id = "user_room"
        self.walls = walls
        self.room_layout_size = room_layout_size
        self.room_side = room_side
        self.room_scale = room_scale
        points = [(float(wall["x1"]), float(wall["y1"])) for wall in walls]
        polygon = Polygon(points).buffer(0)
        self.polygon = scale(
            polygon,
            xfact=room_scale,
            yfact=room_scale,
            origin=polygon.centroid,
        ).buffer(0)
        if not self.polygon.is_valid or self.polygon.area <= 0:
            raise ValueError("Input walls must form a valid closed room polygon")
        self.floor_plan_centroid = np.array(
            [self.polygon.centroid.x, 0.0, self.polygon.centroid.y]
        )
        self._floor_plan = None
        self._room_layout = None

    @property
    def floor_plan(self):
        if self._floor_plan is None:
            polygon = self.polygon
            triangles = [
                triangle
                for triangle in triangulate(polygon)
                if polygon.buffer(1e-8).contains(triangle.representative_point())
            ]
            vertex_to_index = {}
            vertices = []
            faces = []
            for triangle in triangles:
                face = []
                for x_coord, y_coord in list(triangle.exterior.coords)[:3]:
                    key = (round(float(x_coord), 8), round(float(y_coord), 8))
                    if key not in vertex_to_index:
                        vertex_to_index[key] = len(vertices)
                        vertices.append(
                            [
                                x_coord - self.floor_plan_centroid[0],
                                0.0,
                                y_coord - self.floor_plan_centroid[2],
                            ]
                        )
                    face.append(vertex_to_index[key])
                faces.append(face)
            self._floor_plan = (
                np.asarray(vertices, dtype=np.float32),
                np.asarray(faces, dtype=np.int64),
            )
        vertices, faces = self._floor_plan
        return np.copy(vertices), np.copy(faces)

    @property
    def room_mask(self):
        if self._room_layout is None:
            width, height = self.room_layout_size
            image = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(image)
            centered = [
                (x - self.floor_plan_centroid[0], y - self.floor_plan_centroid[2])
                for x, y in list(self.polygon.exterior.coords)[:-1]
            ]
            pixels = [
                (
                    int((x / (2 * self.room_side) + 0.5) * (width - 1)),
                    int((0.5 - y / (2 * self.room_side)) * (height - 1)),
                )
                for x, y in centered
            ]
            draw.polygon(pixels, fill=255)
            self._room_layout = np.asarray(image).astype(np.float32) / 255.0
        return self._room_layout[:, :, None]


def load_config(config_file):
    with Path(config_file).open() as file:
        return load(file, Loader=Loader)


def create_scene(size=(256, 256), background=(1, 1, 1, 1)):
    if os.environ.get("DISPLAY") is not None:
        return Scene(size=size, background=background)

    create_standalone_context = moderngl.create_standalone_context
    try:
        moderngl.create_standalone_context = lambda **kwargs: create_standalone_context(
            backend="egl",
            **kwargs,
        )
        return Scene(size=size, background=background)
    finally:
        moderngl.create_standalone_context = create_standalone_context


def walls_from_points(points):
    if len(points) < 3:
        raise ValueError("At least three points are required")
    walls = []
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        walls.append(
            {
                "x1": float(point[0]),
                "y1": float(point[1]),
                "x2": float(next_point[0]),
                "y2": float(next_point[1]),
            }
        )
    return walls


def resolve_config_paths(config, config_file):
    config = dict(config)
    config["data"] = dict(config["data"])
    config_directory = os.path.dirname(os.path.abspath(config_file))
    root_directory = os.path.abspath(os.path.join(config_directory, os.pardir))
    for key in (
        "annotation_file",
        "path_to_invalid_scene_ids",
        "path_to_invalid_bbox_jids",
    ):
        path = config["data"].get(key)
        if path and not os.path.isabs(path):
            candidates = [
                os.path.normpath(os.path.join(config_directory, path)),
                os.path.normpath(os.path.join(root_directory, path)),
                os.path.normpath(os.path.join(PROJECT_ROOT, path)),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    config["data"][key] = candidate
                    break
    return config


def load_inference_components(paths: InferencePaths, device=None):
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = resolve_config_paths(load_config(paths.config_file), paths.config_file)
    raw_dataset, encoded_dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"]),
        ),
        split=config["validation"].get("splits", ["test"]),
    )
    network, _, _ = build_network(
        encoded_dataset.feature_size,
        encoded_dataset.n_classes,
        config,
        paths.checkpoint,
        device=device,
    )
    network.eval()
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(paths.pickled_3d_future)
    return {
        "device": device,
        "paths": paths,
        "config": config,
        "encoded_dataset": encoded_dataset,
        "network": network,
        "objects_dataset": objects_dataset,
        "class_labels": np.array(encoded_dataset.class_labels),
        "sample_room_layout_size": tuple(raw_dataset[0].room_mask.shape[:2]),
    }


def room_mask_tensor(room, device):
    room_mask = torch.from_numpy(
        np.transpose(room.room_mask[None, :, :, 0:1], (0, 3, 1, 2))
    )
    return room_mask.to(device)


def boxes_to_numpy(boxes):
    return {key: value[0].detach().cpu().numpy() for key, value in boxes.items()}


def object_polygon(translation, size, angle):
    half_width = float(abs(size[0]))
    half_length = float(abs(size[2]))
    corners = np.array(
        [
            [-half_width, -half_length],
            [half_width, -half_length],
            [half_width, half_length],
            [-half_width, half_length],
        ]
    )
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    center = np.array([translation[0], translation[2]])
    points = corners.dot(rotation.T) + center
    return Polygon(points).buffer(0)


def room_polygon(room):
    vertices, faces = room.floor_plan
    polygons = []
    for face in faces:
        polygon = Polygon([(vertices[index, 0], vertices[index, 2]) for index in face])
        if polygon.is_valid and polygon.area > 0:
            polygons.append(polygon)
    return unary_union(polygons).buffer(0)


def move_inside_room(polygon, room_poly, max_steps=10):
    adjusted = polygon
    for _ in range(max_steps):
        outside = adjusted.difference(room_poly)
        if outside.area <= 1e-6:
            return adjusted
        source, target = nearest_points(adjusted.centroid, room_poly)
        dx = target.x - source.x
        dy = target.y - source.y
        if abs(dx) + abs(dy) < 1e-6:
            target = room_poly.representative_point()
            dx = target.x - source.x
            dy = target.y - source.y
        adjusted = translate(adjusted, xoff=dx, yoff=dy)
    if adjusted.difference(room_poly).area <= polygon.difference(room_poly).area:
        return adjusted

    best_polygon = polygon
    best_area = polygon.difference(room_poly).area
    minx, miny, maxx, maxy = room_poly.bounds
    candidate_points = [
        room_poly.representative_point(),
        room_poly.centroid,
        *[
            Point(x_coord, y_coord)
            for x_coord in np.linspace(minx, maxx, 9)
            for y_coord in np.linspace(miny, maxy, 9)
            if room_poly.contains(Point(x_coord, y_coord))
        ],
    ]
    source = polygon.centroid
    for target in candidate_points:
        candidate = translate(
            polygon,
            xoff=target.x - source.x,
            yoff=target.y - source.y,
        )
        outside_area = candidate.difference(room_poly).area
        if outside_area < best_area:
            best_polygon = candidate
            best_area = outside_area
        if outside_area <= 1e-6:
            return candidate
    return best_polygon


def polygon_center(polygon):
    centroid = polygon.centroid
    return np.array([centroid.x, centroid.y], dtype=np.float32)


def postprocess_layout(boxes_np, room, collision_threshold=0.35):
    room_poly = room_polygon(room)
    class_labels = boxes_np["class_labels"]
    translations = np.copy(boxes_np["translations"])
    sizes = np.copy(boxes_np["sizes"])
    angles = np.copy(boxes_np["angles"])
    keep_indices = []
    placed_polygons = []

    for index in range(1, class_labels.shape[0] - 1):
        polygon = object_polygon(
            translations[index], sizes[index], float(angles[index, 0])
        )
        polygon = move_inside_room(polygon, room_poly)
        center = polygon_center(polygon)
        translations[index, 0] = center[0]
        translations[index, 2] = center[1]

        has_large_collision = False
        for placed in placed_polygons:
            overlap = polygon.intersection(placed).area
            relative_overlap = overlap / max(polygon.area, 1e-6)
            if relative_overlap > collision_threshold:
                has_large_collision = True
                break

        if has_large_collision:
            continue
        placed_polygons.append(polygon)
        keep_indices.append(index)

    selected = [0, *keep_indices, class_labels.shape[0] - 1]
    return {
        "class_labels": class_labels[selected],
        "translations": translations[selected],
        "sizes": sizes[selected],
        "angles": angles[selected],
    }


def layout_to_json(layout, class_labels):
    objects = []
    for index in range(1, layout["class_labels"].shape[0] - 1):
        class_index = int(layout["class_labels"][index].argmax())
        angle = float(layout["angles"][index, 0])
        objects.append(
            {
                "class_id": class_index,
                "class_label": str(class_labels[class_index]),
                "x": float(layout["translations"][index, 0]),
                "y": float(layout["translations"][index, 2]),
                "w": float(layout["sizes"][index, 0] * 2),
                "l": float(layout["sizes"][index, 2] * 2),
                "sin_theta": float(math.sin(angle)),
                "cos_theta": float(math.cos(angle)),
                "angle": angle,
            }
        )
    return objects


def layout_to_tensor(layout):
    return np.concatenate(
        [
            layout["class_labels"],
            layout["translations"],
            layout["sizes"],
            layout["angles"],
        ],
        axis=-1,
    )[None]


def procedural_floor_texture(size=512, tile=64):
    texture = Image.new("RGBA", (size, size), (184, 168, 132, 255))
    draw = ImageDraw.Draw(texture)
    for y_coord in range(size):
        shade = 12 if (y_coord // (tile // 2)) % 2 == 0 else -6
        draw.line(
            [(0, y_coord), (size, y_coord)],
            fill=(184 + shade, 168 + shade, 132 + shade, 255),
        )
    for x_coord in range(0, size, tile):
        draw.line([(x_coord, 0), (x_coord, size)], fill=(112, 88, 54, 255), width=2)
    for y_coord in range(0, size, tile // 2):
        draw.line([(0, y_coord), (size, y_coord)], fill=(134, 110, 76, 255), width=1)
    return np.ascontiguousarray(texture)


def floor_mesh(room):
    vertices, faces = room.floor_plan
    floor = trimesh.Trimesh(vertices, faces, process=False)
    uv = np.copy(vertices[:, [0, 2]])
    uv -= uv.min(axis=0)
    uv /= 0.3
    floor.visual = trimesh.visual.TextureVisuals(
        uv=uv,
        material=trimesh.visual.material.SimpleMaterial(
            image=Image.fromarray(procedural_floor_texture()),
        ),
    )
    return floor


def floor_renderables(room, tile_size=0.35):
    minx, miny, maxx, maxy = room_polygon(room).bounds
    tiles = []
    y_coord = miny
    row = 0
    while y_coord < maxy:
        x_coord = minx
        col = 0
        while x_coord < maxx:
            tile = Polygon(
                [
                    (x_coord, y_coord),
                    (min(x_coord + tile_size, maxx), y_coord),
                    (min(x_coord + tile_size, maxx), min(y_coord + tile_size, maxy)),
                    (x_coord, min(y_coord + tile_size, maxy)),
                ]
            )
            clipped = tile.intersection(room_polygon(room))
            if clipped.area > 1e-5 and clipped.geom_type == "Polygon":
                triangles = [
                    triangle
                    for triangle in triangulate(clipped)
                    if clipped.buffer(1e-8).contains(triangle.representative_point())
                ]
                vertices = []
                faces = []
                for triangle in triangles:
                    face = []
                    for px, py in list(triangle.exterior.coords)[:3]:
                        face.append(len(vertices))
                        vertices.append([px, 0.0, py])
                    faces.append(face)
                if vertices:
                    color = (
                        (0.74, 0.64, 0.48, 1.0)
                        if (row + col) % 2 == 0
                        else (0.62, 0.52, 0.38, 1.0)
                    )
                    tiles.append(
                        Mesh.from_faces(
                            np.asarray(vertices, dtype=np.float32),
                            np.asarray(faces, dtype=np.int64),
                            colors=color,
                        )
                    )
            x_coord += tile_size
            col += 1
        y_coord += tile_size
        row += 1
    return tiles


def floor_outline_renderable(room):
    vertices, _ = room.floor_plan
    boundary_points = []
    for x_coord, y_coord in list(room.polygon.exterior.coords):
        boundary_points.append(
            [
                x_coord - room.floor_plan_centroid[0],
                0.035,
                y_coord - room.floor_plan_centroid[2],
            ]
        )
    segments = []
    for index in range(len(boundary_points) - 1):
        segments.extend([boundary_points[index], boundary_points[index + 1]])
    return Lines(segments, colors=(0.05, 0.05, 0.05, 1.0), width=0.035)


def render_layout(
    room,
    layout,
    components,
    output_directory,
    basename="layout",
    size=(512, 512),
):
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    bbox_params_t = layout_to_tensor(layout)
    renderables, trimesh_meshes = get_textured_objects(
        bbox_params_t,
        components["objects_dataset"],
        components["class_labels"],
    )
    floors = floor_renderables(room)
    floor_outline = floor_outline_renderable(room)
    renderables.extend([*floors, floor_outline])
    trimesh_meshes.append(floor_mesh(room))

    views = {
        "topdown": {
            "camera_position": (0, 4, 0),
            "camera_target": (0, 0, 0),
            "up_vector": (0, 0, -1),
            "orthographic": True,
            "render_floor_first": True,
        },
        "front": {
            "camera_position": (0, 2.5, -6),
            "camera_target": (0, 0, 0),
            "up_vector": (0, 1, 0),
            "orthographic": False,
            "render_floor_first": True,
        },
        "corner": {
            "camera_position": (4, 3, -4),
            "camera_target": (0, 0, 0),
            "up_vector": (0, 1, 0),
            "orthographic": False,
            "render_floor_first": True,
        },
    }
    image_paths = {}
    for view_name, view in views.items():
        scene = create_scene(size=size)
        scene.up_vector = view["up_vector"]
        scene.camera_target = view["camera_target"]
        scene.camera_position = view["camera_position"]
        scene.light = view["camera_position"]
        if view["orthographic"]:
            scene.camera_matrix = Matrix44.orthogonal_projection(
                left=-3.1,
                right=3.1,
                bottom=3.1,
                top=-3.1,
                near=0.1,
                far=8,
            )
        frame_path = output_path / f"{basename}_{view_name}.png"
        view_renderables = (
            [*floors, floor_outline, *renderables[: -len(floors) - 1]]
            if view["render_floor_first"]
            else renderables
        )
        render(
            view_renderables,
            behaviours=[SaveFrames(str(frame_path), 1)],
            size=size,
            camera_position=view["camera_position"],
            camera_target=view["camera_target"],
            up_vector=view["up_vector"],
            background=(1, 1, 1, 1),
            n_frames=1,
            scene=scene,
        )
        image_paths[view_name] = str(frame_path)

    scene_path = output_path / f"{basename}.obj"
    trimesh.Scene(trimesh_meshes).export(scene_path)
    return image_paths, str(scene_path)


def infer_layout(
    walls,
    paths: InferencePaths | None = None,
    output_directory="outputs/interactive_inference",
    seed=27,
    render_images=True,
    render_size=(1024, 1024),
    room_scale=1.35,
):
    paths = paths or InferencePaths()
    components = load_inference_components(paths)
    return infer_layout_with_components(
        walls=walls,
        components=components,
        checkpoint=paths.checkpoint,
        output_directory=output_directory,
        seed=seed,
        render_images=render_images,
        render_size=render_size,
        room_scale=room_scale,
    )


def infer_layout_with_components(
    walls,
    components,
    checkpoint=DEFAULT_CHECKPOINT,
    output_directory="outputs/interactive_inference",
    seed=27,
    render_images=True,
    render_size=(1024, 1024),
    room_scale=1.35,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    room = PolygonRoom(
        walls,
        room_layout_size=components["sample_room_layout_size"],
        room_side=3.1,
        room_scale=room_scale,
    )
    room_mask = room_mask_tensor(room, components["device"])
    with torch.no_grad():
        raw_boxes = components["network"].generate_boxes(
            room_mask=room_mask,
            device=components["device"],
        )
    boxes = components["encoded_dataset"].post_process(raw_boxes)
    layout = postprocess_layout(boxes_to_numpy(boxes), room)
    objects = layout_to_json(layout, components["class_labels"])
    response = {
        "objects": objects,
        "n_objects": len(objects),
        "checkpoint": checkpoint,
    }
    if render_images:
        image_paths, scene_path = render_layout(
            room,
            layout,
            components,
            output_directory,
            size=render_size,
        )
        response["renders"] = image_paths
        response["scene_path"] = scene_path
    return response


def load_walls_from_json(path):
    payload = json.loads(Path(path).read_text())
    if "walls" in payload:
        return payload["walls"]
    if "points" in payload:
        return walls_from_points(payload["points"])
    raise ValueError("Input JSON must contain either 'walls' or 'points'")
