from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

torch = None
F = None
AutoImageProcessor = None
SegformerForSemanticSegmentation = None
_TRANSFORMERS_IMPORT_ERROR: Exception | None = None
_ML_IMPORT_ATTEMPTED = False


def _ensure_ml_imports() -> None:
    global torch
    global F
    global AutoImageProcessor
    global SegformerForSemanticSegmentation
    global _TRANSFORMERS_IMPORT_ERROR
    global _ML_IMPORT_ATTEMPTED

    if _ML_IMPORT_ATTEMPTED:
        return
    _ML_IMPORT_ATTEMPTED = True
    try:
        import torch as torch_module
        import torch.nn.functional as functional_module
        from transformers import AutoImageProcessor as auto_image_processor
        from transformers import SegformerForSemanticSegmentation as segformer_model
    except Exception as exc:  # pragma: no cover
        _TRANSFORMERS_IMPORT_ERROR = exc
        return

    torch = torch_module
    F = functional_module
    AutoImageProcessor = auto_image_processor
    SegformerForSemanticSegmentation = segformer_model


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_model_metadata(model_path: Path) -> dict[str, Any]:
    return {
        "classes": read_json_file(model_path / "classes.json"),
        "training_args": read_json_file(model_path / "training_args.json"),
        "config": read_json_file(model_path / "config.json"),
    }


def infer_model_input_wh(
    metadata: dict[str, Any],
    cli_width: int | None,
    cli_height: int | None,
    cli_square_size: int | None,
) -> tuple[int, int, str]:
    if cli_width is not None or cli_height is not None:
        if cli_width is None or cli_height is None:
            raise RuntimeError("Set both model_input_width and model_input_height, not just one.")
        return int(cli_width), int(cli_height), "manual_width_height"

    if cli_square_size is not None:
        if cli_square_size <= 0:
            return 0, 0, "original_frame_size"
        return int(cli_square_size), int(cli_square_size), "manual_square_size"

    classes = metadata.get("classes", {})
    for width_key, height_key, source in (
        ("input_width", "input_height", "classes_json"),
        ("width", "height", "classes_json"),
    ):
        if width_key in classes and height_key in classes:
            try:
                return int(classes[width_key]), int(classes[height_key]), source
            except Exception:
                pass

    training_args = metadata.get("training_args", {})
    if "width" in training_args and "height" in training_args:
        try:
            return int(training_args["width"]), int(training_args["height"]), "training_args_json"
        except Exception:
            pass

    return 1024, 576, "fallback_1024x576"


class ManualSegformerProcessor:
    def __init__(self) -> None:
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.source = "manual_imagenet_normalization"

    def __call__(self, images: np.ndarray, return_tensors: str = "pt") -> dict[str, Any]:
        _ensure_ml_imports()
        if return_tensors != "pt":
            raise ValueError("ManualSegformerProcessor only supports return_tensors='pt'.")
        if torch is None:
            raise RuntimeError("torch is required for SegFormer inference.")
        if images.dtype != np.uint8:
            images = np.clip(images, 0, 255).astype(np.uint8)
        x = images.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return {"pixel_values": torch.from_numpy(x)}


def default_device(requested: str | None = None) -> str:
    _ensure_ml_imports()
    if requested and requested != "auto":
        if requested.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
            print(
                "[WARN] CUDA was requested for map building, but CUDA is not available. "
                "Falling back to CPU for this run. To make CPU explicit, set "
                "`device`: `cpu` in extraction/config/map_builder_config.json.",
                file=sys.stderr,
            )
            return "cpu"
        return requested
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    print(
        "[WARN] CUDA was not detected for map building. Using CPU. "
        "To make CPU explicit, set `device`: `cpu` in extraction/config/map_builder_config.json.",
        file=sys.stderr,
    )
    return "cpu"


def load_segformer(model_path: Path, device: str) -> tuple[Any, Any, str]:
    _ensure_ml_imports()
    if _TRANSFORMERS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Could not import torch/transformers. Install torch, transformers, pillow, and safetensors. "
            f"Original import error: {_TRANSFORMERS_IMPORT_ERROR}"
        )
    if not model_path.exists():
        raise RuntimeError(f"SegFormer model folder does not exist: {model_path}")

    processor_source = "manual_imagenet_normalization"
    processor: Any = ManualSegformerProcessor()
    preprocessor_config = model_path / "preprocessor_config.json"
    if preprocessor_config.exists() and AutoImageProcessor is not None:
        try:
            processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True)
            processor_source = "preprocessor_config_json"
        except Exception as exc:
            print(
                f"[WARN] Could not load AutoImageProcessor from {preprocessor_config}; "
                f"using manual ImageNet normalization. Error: {exc}",
                file=sys.stderr,
            )

    model = SegformerForSemanticSegmentation.from_pretrained(str(model_path), local_files_only=True)
    model.to(device)
    model.eval()
    return processor, model, processor_source


def normalise_id2label(id2label: Any) -> dict[int, str]:
    out: dict[int, str] = {}
    if isinstance(id2label, dict):
        for key, value in id2label.items():
            try:
                out[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
    return out


def resolve_wall_class_id(model: Any, wall_class_id: int | None, wall_label: str) -> int:
    if wall_class_id is not None:
        return int(wall_class_id)

    id2label = normalise_id2label(getattr(model.config, "id2label", {}))
    target = wall_label.strip().lower()
    for idx, label in id2label.items():
        if label.strip().lower() == target:
            return idx
    for idx, label in id2label.items():
        if target in label.strip().lower():
            return idx

    num_labels = int(getattr(model.config, "num_labels", 0) or 0)
    if num_labels == 2:
        print(
            "[WARN] Could not find wall label in model config id2label; defaulting to wall_class_id=1.",
            file=sys.stderr,
        )
        return 1

    raise RuntimeError(f"Could not auto-detect wall class id. Model id2label={id2label}.")


def resize_for_model(frame_bgr: np.ndarray, model_input_width: int, model_input_height: int) -> np.ndarray:
    if model_input_width <= 0 or model_input_height <= 0:
        return frame_bgr
    return cv2.resize(frame_bgr, (int(model_input_width), int(model_input_height)), interpolation=cv2.INTER_AREA)


def infer_wall_probability(
    frame_bgr: np.ndarray,
    processor: Any,
    model: Any,
    device: str,
    wall_class_id: int,
    model_input_width: int,
    model_input_height: int,
    use_fp16: bool,
) -> np.ndarray:
    _ensure_ml_imports()
    if torch is None or F is None:
        raise RuntimeError("torch is required for wall probability inference.")

    orig_h, orig_w = frame_bgr.shape[:2]
    model_frame = resize_for_model(frame_bgr, model_input_width, model_input_height)
    rgb = cv2.cvtColor(model_frame, cv2.COLOR_BGR2RGB)

    inputs = processor(images=rgb, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        if use_fp16 and device.startswith("cuda"):
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

        logits = outputs.logits
        logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        if logits.shape[1] == 1:
            wall_prob = torch.sigmoid(logits[0, 0])
        else:
            if wall_class_id < 0 or wall_class_id >= logits.shape[1]:
                raise RuntimeError(
                    f"wall_class_id={wall_class_id} is outside model logits channels={logits.shape[1]}"
                )
            probs = torch.softmax(logits, dim=1)
            wall_prob = probs[0, wall_class_id]

    return wall_prob.detach().float().cpu().numpy().astype(np.float32)
