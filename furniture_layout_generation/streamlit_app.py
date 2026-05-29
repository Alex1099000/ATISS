"""Streamlit demo for ATISS furniture layout generation."""

from __future__ import annotations

import math
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

from furniture_layout_generation.config import as_container, hydra_config
from furniture_layout_generation.data import DataDownloadError, download_data
from furniture_layout_generation.inference import (
    infer_layout,
    inference_paths_from_config,
)

OUTPUT_DIR = "outputs/streamlit_inference"
SEED = 27
ROOM_SIDE = 3.1
CANVAS_SIZE = 512
SNAP_DEGREES = 5.0
RENDER_SIZE = (1024, 1024)
GRID_STEP_METERS = 0.5
ROOM_RENDER_SCALE = 1.35


def canvas_points(json_data, room_side):
    if not json_data or "objects" not in json_data:
        return []
    points = []
    for item in json_data["objects"]:
        if item.get("type") != "circle":
            continue
        left = float(item["left"])
        top = float(item["top"])
        width = float(item.get("width", 0))
        height = float(item.get("height", 0))
        x_pixel = left + width / 2
        y_pixel = top + height / 2
        x_coord = (x_pixel / 512 - 0.5) * 2 * room_side
        y_coord = (0.5 - y_pixel / 512) * 2 * room_side
        points.append((x_coord, y_coord))
    return points


def metric_to_canvas(point, room_side):
    x_coord, y_coord = point
    return (
        int((x_coord / (2 * room_side) + 0.5) * (CANVAS_SIZE - 1)),
        int((0.5 - y_coord / (2 * room_side)) * (CANVAS_SIZE - 1)),
    )


def draw_grid(draw, room_side):
    grid_color = "#eeeeee"
    axis_color = "#d0d0d0"
    n_steps = int((2 * room_side) / GRID_STEP_METERS)
    for step in range(n_steps + 1):
        coord = -room_side + step * GRID_STEP_METERS
        x_pixel, _ = metric_to_canvas((coord, 0), room_side)
        _, y_pixel = metric_to_canvas((0, coord), room_side)
        draw.line([(x_pixel, 0), (x_pixel, CANVAS_SIZE)], fill=grid_color, width=1)
        draw.line([(0, y_pixel), (CANVAS_SIZE, y_pixel)], fill=grid_color, width=1)

    center_x, center_y = metric_to_canvas((0, 0), room_side)
    draw.line([(center_x, 0), (center_x, CANVAS_SIZE)], fill=axis_color, width=2)
    draw.line([(0, center_y), (CANVAS_SIZE, center_y)], fill=axis_color, width=2)


def canvas_grid_drawing(room_side):
    objects = []
    n_steps = int((2 * room_side) / GRID_STEP_METERS)
    for step in range(n_steps + 1):
        coord = -room_side + step * GRID_STEP_METERS
        x_pixel, _ = metric_to_canvas((coord, 0), room_side)
        _, y_pixel = metric_to_canvas((0, coord), room_side)
        stroke = "#d0d0d0" if abs(coord) < 1e-6 else "#eeeeee"
        width = 2 if abs(coord) < 1e-6 else 1
        objects.append(
            {
                "type": "line",
                "version": "4.4.0",
                "originX": "left",
                "originY": "top",
                "left": x_pixel,
                "top": 0,
                "width": 0,
                "height": CANVAS_SIZE,
                "fill": None,
                "stroke": stroke,
                "strokeWidth": width,
                "strokeDashArray": None,
                "strokeLineCap": "butt",
                "strokeDashOffset": 0,
                "strokeLineJoin": "miter",
                "strokeUniform": False,
                "strokeMiterLimit": 4,
                "scaleX": 1,
                "scaleY": 1,
                "angle": 0,
                "flipX": False,
                "flipY": False,
                "opacity": 1,
                "shadow": None,
                "visible": True,
                "backgroundColor": "",
                "fillRule": "nonzero",
                "paintFirst": "fill",
                "globalCompositeOperation": "source-over",
                "skewX": 0,
                "skewY": 0,
                "x1": 0,
                "x2": 0,
                "y1": -CANVAS_SIZE / 2,
                "y2": CANVAS_SIZE / 2,
                "selectable": False,
                "evented": False,
            }
        )
        objects.append(
            {
                "type": "line",
                "version": "4.4.0",
                "originX": "left",
                "originY": "top",
                "left": 0,
                "top": y_pixel,
                "width": CANVAS_SIZE,
                "height": 0,
                "fill": None,
                "stroke": stroke,
                "strokeWidth": width,
                "strokeDashArray": None,
                "strokeLineCap": "butt",
                "strokeDashOffset": 0,
                "strokeLineJoin": "miter",
                "strokeUniform": False,
                "strokeMiterLimit": 4,
                "scaleX": 1,
                "scaleY": 1,
                "angle": 0,
                "flipX": False,
                "flipY": False,
                "opacity": 1,
                "shadow": None,
                "visible": True,
                "backgroundColor": "",
                "fillRule": "nonzero",
                "paintFirst": "fill",
                "globalCompositeOperation": "source-over",
                "skewX": 0,
                "skewY": 0,
                "x1": -CANVAS_SIZE / 2,
                "x2": CANVAS_SIZE / 2,
                "y1": 0,
                "y2": 0,
                "selectable": False,
                "evented": False,
            }
        )
    return {"version": "4.4.0", "objects": objects}


def room_preview(points, room_side, with_grid=True):
    image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
    draw = ImageDraw.Draw(image)
    if with_grid:
        draw_grid(draw, room_side)
    pixel_points = [metric_to_canvas(point, room_side) for point in points]
    if len(pixel_points) >= 2:
        draw.line(pixel_points, fill="#1f77b4", width=4)
    if len(pixel_points) >= 3:
        draw.line([pixel_points[-1], pixel_points[0]], fill="#1f77b4", width=4)
        draw.polygon(pixel_points, outline="#1f77b4", fill="#eaf4ff")
    for index, pixel_point in enumerate(pixel_points, start=1):
        x_coord, y_coord = pixel_point
        draw.ellipse(
            (x_coord - 5, y_coord - 5, x_coord + 5, y_coord + 5),
            fill="#1f77b4",
            outline="white",
            width=2,
        )
        draw.text((x_coord + 8, y_coord - 8), str(index), fill="#1f77b4")
    return image


def snap_orthogonal_points(points, max_degrees=SNAP_DEGREES):
    if len(points) < 2:
        return points

    snapped = [points[0]]
    max_tangent = math.tan(math.radians(max_degrees))
    for point in points[1:]:
        prev_x, prev_y = snapped[-1]
        x_coord, y_coord = point
        dx = x_coord - prev_x
        dy = y_coord - prev_y
        if abs(dx) >= abs(dy):
            if abs(dx) > 1e-6 and abs(dy / dx) <= max_tangent:
                y_coord = prev_y
        elif abs(dy) > 1e-6 and abs(dx / dy) <= max_tangent:
            x_coord = prev_x
        snapped.append((x_coord, y_coord))

    if len(snapped) >= 3:
        first_x, first_y = snapped[0]
        last_x, last_y = snapped[-1]
        dx = first_x - last_x
        dy = first_y - last_y
        if abs(dx) >= abs(dy):
            if abs(dx) > 1e-6 and abs(dy / dx) <= max_tangent:
                snapped[0] = (first_x, last_y)
        elif abs(dy) > 1e-6 and abs(dx / dy) <= max_tangent:
            snapped[0] = (last_x, first_y)
    return snapped


def walls_from_points(points):
    walls = []
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        walls.append(
            {
                "x1": point[0],
                "y1": point[1],
                "x2": next_point[0],
                "y2": next_point[1],
            }
        )
    return walls


def default_walls(width, length):
    half_width = width / 2
    half_length = length / 2
    points = [
        (-half_width, -half_length),
        (half_width, -half_length),
        (half_width, half_length),
        (-half_width, half_length),
    ]
    return walls_from_points(points)


@st.cache_resource
def load_paths():
    with hydra_config("config") as config:
        paths = inference_paths_from_config(as_container(config))
    download_data(
        required_paths=[
            paths.config_file,
            paths.checkpoint,
            paths.pickled_3d_future,
            paths.floor_textures,
        ],
        strict=False,
    )
    return paths


def main() -> None:
    st.set_page_config(page_title="Furniture Layout Generation", layout="wide")
    st.title("Furniture Layout Generation")
    st.caption("Draw room corners or use the rectangular room preset, then run ATISS.")

    col_draw, col_result = st.columns([1, 1])
    with col_draw:
        st.subheader("Room Input")
        mode = st.radio("Input mode", ["Draw corners", "Rectangle"], horizontal=True)
        walls = None
        if mode == "Draw corners":
            st.write(
                "Use point mode: click room corners clockwise or counterclockwise."
            )
            canvas = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=3,
                stroke_color="#1f77b4",
                background_color="#ffffff",
                height=CANVAS_SIZE,
                width=CANVAS_SIZE,
                drawing_mode="point",
                initial_drawing=canvas_grid_drawing(ROOM_SIDE),
                point_display_radius=5,
                key="room_canvas",
            )
            points = canvas_points(canvas.json_data, ROOM_SIDE)
            snapped_points = snap_orthogonal_points(points)
            st.session_state["room_points"] = points
            st.write(f"Detected points: {len(points)}")
            st.image(
                room_preview(snapped_points, ROOM_SIDE),
                caption="Room contour and grid preview",
                width="stretch",
            )
            if len(points) >= 3:
                walls = walls_from_points(snapped_points)
        else:
            width = st.number_input("Room width, meters", value=5.0, step=0.1)
            length = st.number_input("Room length, meters", value=4.0, step=0.1)
            walls = default_walls(width, length)
            points = [(wall["x1"], wall["y1"]) for wall in walls]
            st.image(
                room_preview(points, ROOM_SIDE),
                caption="Room contour and grid preview",
                width="stretch",
            )

    with col_result:
        st.subheader("Generated Layout")
        if st.button("Generate Layout", type="primary", disabled=walls is None):
            with st.spinner("Running ATISS inference and rendering..."):
                try:
                    paths = load_paths()
                except DataDownloadError as exc:
                    st.error(str(exc))
                    st.stop()
                result = infer_layout(
                    walls=walls,
                    paths=paths,
                    output_directory=OUTPUT_DIR,
                    seed=SEED,
                    render_images=True,
                    render_size=RENDER_SIZE,
                    room_scale=ROOM_RENDER_SCALE,
                )
            st.success(f"Generated {result['n_objects']} objects")
            renders = result.get("renders", {})
            for name, image_path in renders.items():
                st.image(image_path, caption=name, width="stretch")
            if result.get("scene_path"):
                scene_path = Path(result["scene_path"])
                st.download_button(
                    "Download OBJ scene",
                    data=scene_path.read_bytes(),
                    file_name=scene_path.name,
                )


if __name__ == "__main__":
    main()
