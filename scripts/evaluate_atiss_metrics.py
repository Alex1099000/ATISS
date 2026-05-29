#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
#

"""Evaluate ATISS checkpoints with the metrics used in the paper.

This script aggregates the metrics implemented by the released repository:
category KL divergence, FID over top-down renderings, and real-vs-synthetic
classification accuracy.
"""

import argparse
import json
import logging
import os
import shutil
import sys

import numpy as np
import torch
from cleanfid import fid
from PIL import Image
from shapely.geometry import Polygon
from shapely.ops import unary_union
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.utils import render
from torchvision import models
from tqdm import tqdm
from training_utils import load_config
from utils import create_scene, export_scene, floor_plan_from_scene

from scene_synthesis.datasets import filter_function, get_dataset_raw_and_encoded
from scene_synthesis.datasets.splits_builder import CSVSplitsBuilder
from scene_synthesis.datasets.threed_front import CachedThreedFront
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network
from scene_synthesis.utils import get_textured_objects


def categorical_kl(p, q):
    return float((p * (np.log(p + 1e-6) - np.log(q + 1e-6))).sum())


def room_polygon_from_scene(scene):
    vertices, faces = scene.floor_plan
    vertices = vertices - scene.floor_plan_centroid
    polygons = []
    for face in faces:
        points = [(vertices[index, 0], vertices[index, 2]) for index in face]
        polygon = Polygon(points)
        if polygon.is_valid and polygon.area > 0:
            polygons.append(polygon)
    if not polygons:
        return None
    return unary_union(polygons).buffer(0)


def object_polygon(translation, size, angle):
    half_width = float(abs(size[0]))
    half_length = float(abs(size[2]))
    local_corners = np.array(
        [
            [-half_width, -half_length],
            [half_width, -half_length],
            [half_width, half_length],
            [-half_width, half_length],
        ],
        dtype=np.float64,
    )
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float64,
    )
    center = np.array([translation[0], translation[2]], dtype=np.float64)
    corners = local_corners.dot(rotation.T) + center
    return Polygon(corners).buffer(0)


def scene_geometry_metrics(scene, boxes, eps=1e-6):
    room_polygon = room_polygon_from_scene(scene)
    if room_polygon is None or room_polygon.is_empty:
        return {
            "n_objects": 0,
            "n_colliding_objects": 0,
            "n_out_of_room_objects": 0,
        }

    object_polygons = []
    class_labels = boxes["class_labels"][0].cpu().numpy()
    translations = boxes["translations"][0].cpu().numpy()
    sizes = boxes["sizes"][0].cpu().numpy()
    angles = boxes["angles"][0].cpu().numpy()
    for object_index in range(1, class_labels.shape[0] - 1):
        polygon = object_polygon(
            translations[object_index],
            sizes[object_index],
            float(angles[object_index, 0]),
        )
        if polygon.is_valid and polygon.area > eps:
            object_polygons.append(polygon)

    colliding_objects = set()
    for first_index in range(len(object_polygons)):
        for second_index in range(first_index + 1, len(object_polygons)):
            intersection = object_polygons[first_index].intersection(
                object_polygons[second_index]
            )
            if intersection.area > eps:
                colliding_objects.add(first_index)
                colliding_objects.add(second_index)

    out_of_room_objects = 0
    for polygon in object_polygons:
        outside_area = polygon.difference(room_polygon).area
        if outside_area > eps:
            out_of_room_objects += 1

    return {
        "n_objects": len(object_polygons),
        "n_colliding_objects": len(colliding_objects),
        "n_out_of_room_objects": out_of_room_objects,
    }


def aggregate_geometry_metrics(scene_metrics):
    n_objects = sum(m["n_objects"] for m in scene_metrics)
    n_colliding_objects = sum(m["n_colliding_objects"] for m in scene_metrics)
    n_out_of_room_objects = sum(m["n_out_of_room_objects"] for m in scene_metrics)
    if n_objects == 0:
        col_obj = 0.0
        r_out = 0.0
    else:
        col_obj = n_colliding_objects / n_objects
        r_out = n_out_of_room_objects / n_objects
    return {
        "col_obj": float(col_obj),
        "r_out": float(r_out),
        "n_objects": int(n_objects),
        "n_colliding_objects": int(n_colliding_objects),
        "n_out_of_room_objects": int(n_out_of_room_objects),
    }


def image_paths(directory):
    return sorted(
        [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".png")
        ]
    )


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, directory, train=True):
        images = image_paths(directory)
        n_images = len(images) // 2
        start = 0 if train else n_images
        self.images = images[start : start + n_images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


class ThreedFrontRenderDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx].image_path


class SyntheticVRealDataset(torch.utils.data.Dataset):
    def __init__(self, real, synthetic):
        self.n_items = min(len(real), len(synthetic))
        self.real = real
        self.synthetic = synthetic

    def __len__(self):
        return 2 * self.n_items

    def __getitem__(self, idx):
        if idx < self.n_items:
            image_path = self.real[idx]
            label = 1
        else:
            image_path = self.synthetic[idx - self.n_items]
            label = 0

        img = Image.open(image_path)
        img = np.asarray(img).astype(np.float32) / np.float32(255)
        img = np.transpose(img[:, :, :3], (2, 0, 1))

        return torch.from_numpy(img), torch.tensor([label], dtype=torch.float)


class AlexNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.fc = torch.nn.Linear(9216, 1)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = self.fc(x.view(len(x), -1))
        return torch.sigmoid(x)


class AverageMeter:
    def __init__(self):
        self.value_sum = 0
        self.count = 0

    def __iadd__(self, value):
        if torch.is_tensor(value):
            self.value_sum += value.sum().item()
            self.count += value.numel()
        else:
            self.value_sum += value
            self.count += 1
        return self

    @property
    def value(self):
        return self.value_sum / self.count


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def recreate_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


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
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    config["data"][key] = candidate
                    break
            else:
                config["data"][key] = candidates[0]
    return config


def load_eval_dataset(config, split_name, without_lamps=False):
    config = dict(config)
    config["data"] = dict(config["data"])
    if "eval" not in config["data"]["encoding_type"]:
        config["data"]["encoding_type"] += "_eval"
    split = config[split_name].get("splits", ["test"])
    return get_dataset_raw_and_encoded(
        config["data"],
        filter_function(config["data"], split=split, without_lamps=without_lamps),
        path_to_bounds=None,
        split=split,
    )


def build_model(config, encoded_dataset, weight_file, device):
    network, _, _ = build_network(
        encoded_dataset.feature_size,
        encoded_dataset.n_classes,
        config,
        weight_file,
        device=device,
    )
    network.eval()
    return network


def generate_scenes(
    config,
    raw_dataset,
    encoded_dataset,
    network,
    objects_dataset,
    floor_texture_directory,
    output_directory,
    n_scenes,
    device,
    render_scenes=True,
    export_meshes=False,
    window_size=(256, 256),
    room_side=3.1,
    camera_position=(0, 4, 0),
    camera_target=(0, 0, 0),
    up_vector=(0, 0, -1),
    background=(1, 1, 1, 1),
):
    make_dirs(output_directory)
    classes = np.array(encoded_dataset.class_labels)
    synthesized_scenes = []
    geometry_scene_metrics = []
    scene = None

    if render_scenes:
        from pyrr import Matrix44

        scene = create_scene(size=window_size, background=background)
        scene.up_vector = up_vector
        scene.camera_target = camera_target
        scene.camera_position = camera_position
        scene.light = camera_position
        scene.camera_matrix = Matrix44.orthogonal_projection(
            left=-room_side,
            right=room_side,
            bottom=room_side,
            top=-room_side,
            near=0.1,
            far=6,
        )

    for i in tqdm(range(n_scenes), desc="Generating scenes"):
        scene_idx = np.random.choice(len(raw_dataset))
        current_scene = raw_dataset[scene_idx]
        floor_plan, tr_floor, room_mask = floor_plan_from_scene(
            current_scene, floor_texture_directory
        )
        bbox_params = network.generate_boxes(
            room_mask=room_mask.to(device), device=device
        )
        boxes = encoded_dataset.post_process(bbox_params)
        synthesized_scenes.append({k: v[0].cpu().numpy() for k, v in boxes.items()})
        geometry_scene_metrics.append(scene_geometry_metrics(current_scene, boxes))

        if not render_scenes and not export_meshes:
            continue

        bbox_params_t = (
            torch.cat(
                [
                    boxes["class_labels"],
                    boxes["translations"],
                    boxes["sizes"],
                    boxes["angles"],
                ],
                dim=-1,
            )
            .cpu()
            .numpy()
        )
        renderables, trimesh_meshes = get_textured_objects(
            bbox_params_t, objects_dataset, classes
        )
        renderables += floor_plan
        trimesh_meshes += tr_floor

        if render_scenes:
            path_to_image = os.path.join(output_directory, f"{i:05d}")
            render(
                renderables,
                behaviours=[SaveFrames(path_to_image + ".png", 1)],
                size=window_size,
                camera_position=camera_position,
                camera_target=camera_target,
                up_vector=up_vector,
                background=background,
                n_frames=1,
                scene=scene,
            )

        if export_meshes:
            path_to_objs = os.path.join(output_directory, f"{i:05d}_scene")
            make_dirs(path_to_objs)
            export_scene(path_to_objs, trimesh_meshes)

    return synthesized_scenes, aggregate_geometry_metrics(geometry_scene_metrics)


def compute_kl(raw_dataset, encoded_dataset, synthesized_scenes, output_path):
    gt_class_labels = sum([d["class_labels"].sum(0) for d in encoded_dataset]) / sum(
        [d["class_labels"].shape[0] for d in encoded_dataset]
    )
    syn_class_labels = sum(
        [d["class_labels"][1:-1].sum(0) for d in synthesized_scenes]
    ) / sum([d["class_labels"].shape[0] - 2 for d in synthesized_scenes])

    classes = np.array(encoded_dataset.class_labels)
    gt_cooccurrences = np.zeros((len(classes) - 2, len(classes) - 2))
    syn_cooccurrences = np.zeros((len(classes) - 2, len(classes) - 2))
    for gt_scene, syn_scene in zip(encoded_dataset, synthesized_scenes, strict=False):
        gt_classes = gt_scene["class_labels"].argmax(axis=-1)
        syn_classes = syn_scene["class_labels"][1:-1].argmax(axis=-1)

        for ii in range(len(gt_classes)):
            row = gt_classes[ii]
            for jj in range(ii + 1, len(gt_classes)):
                col = gt_classes[jj]
                gt_cooccurrences[row, col] += 1

        for ii in range(len(syn_classes)):
            row = syn_classes[ii]
            for jj in range(ii + 1, len(syn_classes)):
                col = syn_classes[jj]
                syn_cooccurrences[row, col] += 1

    kl = categorical_kl(gt_class_labels, syn_class_labels)
    np.savez(
        output_path,
        stats={"class_labels": kl},
        classes=classes,
        gt_class_labels=gt_class_labels,
        syn_class_labels=syn_class_labels,
        gt_cooccurrences=gt_cooccurrences,
        syn_cooccurrences=syn_cooccurrences,
    )
    return kl


def create_real_render_folder(
    processed_dataset_directory, annotation_file, output_directory
):
    make_dirs(output_directory)
    config = dict(train_stats="dataset_stats.txt", room_layout_size="256,256")
    splits_builder = CSVSplitsBuilder(annotation_file)
    test_real = ThreedFrontRenderDataset(
        CachedThreedFront(
            processed_dataset_directory,
            config=config,
            scene_ids=splits_builder.get_splits(["test"]),
        )
    )
    for i, image_path in enumerate(test_real):
        shutil.copyfile(image_path, os.path.join(output_directory, f"{i:05d}.png"))
    return len(test_real)


def compute_fid(
    processed_dataset_directory,
    annotation_file,
    synthesized_directory,
    work_directory,
    n_repeats=10,
):
    real_directory = os.path.join(work_directory, "fid_real")
    fake_directory = os.path.join(work_directory, "fid_fake")
    if os.path.exists(real_directory):
        shutil.rmtree(real_directory)
    if os.path.exists(fake_directory):
        shutil.rmtree(fake_directory)
    make_dirs(real_directory)
    make_dirs(fake_directory)

    n_real = create_real_render_folder(
        processed_dataset_directory, annotation_file, real_directory
    )
    synthesized_images = image_paths(synthesized_directory)
    if len(synthesized_images) == 0:
        raise RuntimeError(f"No synthesized images found in {synthesized_directory}")
    sample_with_replacement = len(synthesized_images) < n_real
    if sample_with_replacement:
        print(
            "Using replacement for FID because only {} synthesized images are "
            "available for {} real images".format(len(synthesized_images), n_real),
            flush=True,
        )

    scores = []
    for _ in tqdm(range(n_repeats), desc="Computing FID"):
        for f in os.listdir(fake_directory):
            os.remove(os.path.join(fake_directory, f))
        selected = np.random.choice(
            synthesized_images, n_real, replace=sample_with_replacement
        )
        for i, image_path in enumerate(selected):
            shutil.copyfile(image_path, os.path.join(fake_directory, f"{i:05d}.png"))
        scores.append(
            float(
                fid.compute_fid(
                    real_directory,
                    fake_directory,
                    device=torch.device("cpu"),
                    use_dataparallel=False,
                )
            )
        )

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scores": scores,
    }


def train_synthetic_vs_real_classifier(
    processed_dataset_directory,
    annotation_file,
    synthesized_directory,
    output_directory,
    batch_size,
    num_workers,
    epochs,
    repeats,
    device,
):
    make_dirs(output_directory)
    config = dict(train_stats="dataset_stats.txt", room_layout_size="256,256")
    splits_builder = CSVSplitsBuilder(annotation_file)
    train_real = ThreedFrontRenderDataset(
        CachedThreedFront(
            processed_dataset_directory,
            config=config,
            scene_ids=splits_builder.get_splits(["train", "val"]),
        )
    )
    test_real = ThreedFrontRenderDataset(
        CachedThreedFront(
            processed_dataset_directory,
            config=config,
            scene_ids=splits_builder.get_splits(["test"]),
        )
    )
    train_synthetic = ImageFolderDataset(synthesized_directory, True)
    test_synthetic = ImageFolderDataset(synthesized_directory, False)

    train_dataset = SyntheticVRealDataset(train_real, train_synthetic)
    test_dataset = SyntheticVRealDataset(test_real, test_synthetic)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    scores = []
    for repeat in range(repeats):
        model = AlexNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        acc_meter = None
        for _ in tqdm(range(epochs), desc=f"Training classifier {repeat+1}"):
            for x, y in train_dataloader:
                model.train()
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_hat = model(x)
                loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.eval()
                acc_meter = AverageMeter()
                for x, y in test_dataloader:
                    x = x.to(device)
                    y = y.to(device)
                    y_hat = model(x)
                    acc = (torch.abs(y - y_hat) < 0.5).float().mean()
                    acc_meter += acc
        scores.append(float(acc_meter.value))

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scores": scores,
    }


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Evaluate ATISS with KL, FID, and synthetic-vs-real accuracy"
    )
    parser.add_argument("config_file")
    parser.add_argument("output_directory")
    parser.add_argument("path_to_pickled_3d_future_models")
    parser.add_argument("path_to_floor_plan_textures")
    parser.add_argument("--weight_file", required=True)
    parser.add_argument("--processed_dataset_directory", required=True)
    parser.add_argument(
        "--annotation_file", default="configs/splits/bedroom_threed_front_splits.csv"
    )
    parser.add_argument("--n_synthesized_scenes", type=int, default=1000)
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
    logging.getLogger("trimesh").setLevel(logging.ERROR)
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running code on", device)

    make_dirs(args.output_directory)
    generated_directory = os.path.join(args.output_directory, "generated")
    if not args.skip_generation:
        recreate_directory(generated_directory)
    metrics = {}

    config = resolve_config_paths(load_config(args.config_file), args.config_file)
    raw_dataset, encoded_dataset = load_eval_dataset(config, args.split)
    network = build_model(config, encoded_dataset, args.weight_file, device)
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_future_models
    )

    if args.skip_generation:
        synthesized_scenes = None
    else:
        synthesized_scenes, geometry_metrics = generate_scenes(
            config,
            raw_dataset,
            encoded_dataset,
            network,
            objects_dataset,
            args.path_to_floor_plan_textures,
            generated_directory,
            args.n_synthesized_scenes,
            device,
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
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
