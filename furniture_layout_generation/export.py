"""Model export helpers for ONNX and TensorRT artifacts."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch

from furniture_layout_generation.inference import (
    InferencePaths,
    load_inference_components,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ATISSDecodeStepWrapper(torch.nn.Module):
    """Exportable wrapper for one deterministic ATISS autoregressive step."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(
        self,
        room_mask,
        class_labels,
        translations,
        sizes,
        angles,
        next_class_labels,
        next_translations,
        next_angles,
    ):
        boxes = {
            "class_labels": class_labels,
            "translations": translations,
            "sizes": sizes,
            "angles": angles,
        }
        features = self.network._encode(boxes, room_mask)
        class_logits = self.network.hidden2output.class_layer(features)

        translation_params = self.network.hidden2output.get_translations_dmll_params(
            features,
            next_class_labels,
        )
        angle_params = self.network.hidden2output.angle_layer(
            torch.cat(
                [
                    features,
                    self.network.hidden2output.fc_class_labels(next_class_labels),
                    self.network.hidden2output.pe_trans_x(next_translations[:, :, 0:1]),
                    self.network.hidden2output.pe_trans_y(next_translations[:, :, 1:2]),
                    self.network.hidden2output.pe_trans_z(next_translations[:, :, 2:3]),
                ],
                dim=-1,
            ),
        )
        size_features = torch.cat(
            [
                features,
                self.network.hidden2output.fc_class_labels(next_class_labels),
                self.network.hidden2output.pe_trans_x(next_translations[:, :, 0:1]),
                self.network.hidden2output.pe_trans_y(next_translations[:, :, 1:2]),
                self.network.hidden2output.pe_trans_z(next_translations[:, :, 2:3]),
                self.network.hidden2output.pe_angle_z(next_angles),
            ],
            dim=-1,
        )
        size_x_params = self.network.hidden2output.size_layer_x(size_features)
        size_y_params = self.network.hidden2output.size_layer_y(size_features)
        size_z_params = self.network.hidden2output.size_layer_z(size_features)

        return (
            class_logits,
            translation_params["translations_x_probs"],
            translation_params["translations_x_means"],
            translation_params["translations_x_scales"],
            translation_params["translations_y_probs"],
            translation_params["translations_y_means"],
            translation_params["translations_y_scales"],
            translation_params["translations_z_probs"],
            translation_params["translations_z_means"],
            translation_params["translations_z_scales"],
            angle_params,
            size_x_params,
            size_y_params,
            size_z_params,
        )


@contextmanager
def patched_fast_transformer_full_mask():
    """Handle scalar tensor sequence lengths produced by ONNX tracing."""

    import fast_transformers.masking as masking
    import fast_transformers.transformers as transformers

    original_masking_full_mask = masking.FullMask
    original_transformers_full_mask = transformers.FullMask

    def full_mask_compat(mask=None, N=None, M=None, device="cpu"):
        if (
            isinstance(mask, torch.Tensor)
            and mask.dtype != torch.bool
            and mask.numel() == 1
        ):
            mask = int(mask.detach().cpu().item())
        return original_masking_full_mask(mask=mask, N=N, M=M, device=device)

    try:
        masking.FullMask = full_mask_compat
        transformers.FullMask = full_mask_compat
        yield
    finally:
        masking.FullMask = original_masking_full_mask
        transformers.FullMask = original_transformers_full_mask


def export_onnx_model(
    checkpoint_path: str | Path,
    output_path: str | Path,
    config_file: str | Path = "configs/atiss/bedrooms_zakharov.yaml",
    pickled_3d_future: str | Path = (
        "/workspace-SR006.nfs2/datasets/atiss_processed/bedrooms/"
        "threed_future_model_bedroom.pkl"
    ),
    floor_textures: str
    | Path = "/workspace-SR006.nfs2/datasets/3d-front/3D-FRONT-texture",
    opset: int = 17,
) -> None:
    """Export a single ATISS autoregressive decode step to ONNX.

    Full scene generation remains an outer loop around this decode step because
    it stops dynamically when the model predicts the end token.
    """

    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    components = load_inference_components(
        InferencePaths(
            config_file=str(config_file),
            checkpoint=str(checkpoint),
            pickled_3d_future=str(pickled_3d_future),
            floor_textures=str(floor_textures),
        ),
        device=torch.device("cpu"),
    )
    network = components["network"].cpu().eval()
    wrapper = ATISSDecodeStepWrapper(network).eval()
    n_classes = network.n_classes
    room_size = components["sample_room_layout_size"]

    class_labels = torch.zeros(1, 1, n_classes, dtype=torch.float32)
    class_labels[0, 0, -2] = 1.0
    dummy_inputs = (
        torch.zeros(1, 1, room_size[0], room_size[1], dtype=torch.float32),
        class_labels,
        torch.zeros(1, 1, 3, dtype=torch.float32),
        torch.zeros(1, 1, 3, dtype=torch.float32),
        torch.zeros(1, 1, 1, dtype=torch.float32),
        class_labels,
        torch.zeros(1, 1, 3, dtype=torch.float32),
        torch.zeros(1, 1, 1, dtype=torch.float32),
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with patched_fast_transformer_full_mask():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            output,
            export_params=True,
            opset_version=opset,
            input_names=[
                "room_mask",
                "class_labels",
                "translations",
                "sizes",
                "angles",
                "next_class_labels",
                "next_translations",
                "next_angles",
            ],
            output_names=[
                "next_class_logits",
                "translations_x_probs",
                "translations_x_means",
                "translations_x_scales",
                "translations_y_probs",
                "translations_y_means",
                "translations_y_scales",
                "translations_z_probs",
                "translations_z_means",
                "translations_z_scales",
                "angle_params",
                "size_x_params",
                "size_y_params",
                "size_z_params",
            ],
        )


def export_tensorrt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    fp16: bool = True,
) -> None:
    """Build a TensorRT engine from ONNX using the TensorRT Python API."""

    import tensorrt as trt

    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_file}")

    engine_file = Path(engine_path)
    engine_file.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    try:
        builder = trt.Builder(logger)
    except TypeError as exc:
        raise RuntimeError(
            "TensorRT is installed, but the CUDA runtime could not initialize. "
            "Check that the TensorRT wheel matches the available CUDA/driver "
            "stack and that the current process can create a CUDA context."
        ) from exc
    if builder is None:
        raise RuntimeError(
            "TensorRT failed to create an engine builder. Check CUDA and "
            "TensorRT runtime compatibility."
        )
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_file.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("Failed to parse ONNX model:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT failed to build a serialized engine")

    engine_file.write_bytes(serialized_engine)
