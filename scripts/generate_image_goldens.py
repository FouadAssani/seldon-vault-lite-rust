#!/usr/bin/env python3
"""Generate image preprocessing and vision goldens for Seldon Vault Lite.

This is a laboratory-only reference tool. It is never launched by the product
and is not a Cargo dependency. It requires the frozen model cache, PyTorch, and
Transformers; the committed fixtures are compact JSON files with fingerprints.

The official image processor depends on torchvision, which is not installed in
the reference environment. The preprocessing here therefore reproduces the
same tensor operations directly: PIL-equivalent RGB conversion (RGBA composited
over white), aspect-ratio preserving target size, torchvision's uint8
antialiased bicubic resize via torch.nn.functional.interpolate, rescaling to
[0, 1], patchification, and padded positions. The committed resize and patch
fingerprints are checked by the Rust tests.

Usage:
    .model-cache/venv313/bin/python scripts/generate_image_goldens.py
"""

import hashlib
import json
import math
import struct
import zlib
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / ".model-cache" / "f9972ac22e696a105c482202646979e33e4b5a1e"
PATCH_SIZE = 16
POOLING = 3
SUPPORTED = (70, 140, 280, 560, 1120)


def decode_png(path: Path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", path
    pos = 8
    idat = b""
    width = height = color_type = None
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
            assert depth == 8 and compression == 0 and filter_method == 0 and interlace == 0
        elif kind == b"IDAT":
            idat += chunk
        elif kind == b"IEND":
            break
    raw = zlib.decompress(idat)
    channels = {0: 1, 2: 3, 6: 4}[color_type]
    stride = width * channels
    out = bytearray(height * stride)
    previous = bytearray(stride)
    offset = 0
    for y in range(height):
        filter_type = raw[offset]
        offset += 1
        line = bytearray(raw[offset : offset + stride])
        offset += stride
        for i in range(stride):
            a = line[i - channels] if i >= channels else 0
            b = previous[i]
            c = previous[i - channels] if i >= channels else 0
            if filter_type == 1:
                line[i] = (line[i] + a) & 255
            elif filter_type == 2:
                line[i] = (line[i] + b) & 255
            elif filter_type == 3:
                line[i] = (line[i] + ((a + b) // 2)) & 255
            elif filter_type == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                predictor = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + predictor) & 255
        out[y * stride : (y + 1) * stride] = line
        previous = line
    return width, height, channels, bytes(out)


def to_rgb_tensor(width: int, height: int, channels: int, pixels: bytes):
    tensor = torch.frombuffer(pixels, dtype=torch.uint8).reshape(height, width, channels)
    tensor = tensor.permute(2, 0, 1).contiguous()
    if channels == 1:
        return tensor.repeat(3, 1, 1)
    if channels == 4:
        value = tensor.to(torch.float32) / 255.0
        value = value[:3] * value[3:] + (1.0 - value[3:])
        return (value * 255.0).round().clamp(0, 255).to(torch.uint8)
    return tensor


def target_size(height: int, width: int, max_patches: int):
    total_px = height * width
    target_px = max_patches * PATCH_SIZE**2
    factor = math.sqrt(target_px / total_px)
    side_mult = POOLING * PATCH_SIZE
    target_height = int(math.floor(factor * height / side_mult)) * side_mult
    target_width = int(math.floor(factor * width / side_mult)) * side_mult
    if target_height == 0 and target_width == 0:
        raise ValueError("zero target")
    max_side_length = (max_patches // (POOLING**2)) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)
    assert target_height * target_width <= target_px
    return target_height, target_width


def preprocess(path: Path, max_soft_tokens: int):
    width, height, channels, pixels = decode_png(path)
    rgb = to_rgb_tensor(width, height, channels, pixels)
    max_patches = max_soft_tokens * POOLING**2
    target_height, target_width = target_size(height, width, max_patches)
    if (target_height, target_width) != (height, width):
        rgb = torch.nn.functional.interpolate(
            rgb[None],
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )[0]
    assert rgb.dtype == torch.uint8
    scaled = rgb.to(torch.float32) * (1.0 / 255.0)
    patch_rows = target_height // PATCH_SIZE
    patch_cols = target_width // PATCH_SIZE
    patches = (
        scaled.reshape(3, patch_rows, PATCH_SIZE, patch_cols, PATCH_SIZE)
        .permute(1, 3, 2, 4, 0)
        .reshape(patch_rows * patch_cols, -1)
    )
    grid = torch.meshgrid(
        torch.arange(patch_cols), torch.arange(patch_rows), indexing="xy"
    )
    positions = torch.stack(grid, dim=-1).reshape(patch_rows * patch_cols, 2)
    pixel_values = torch.zeros(max_patches, PATCH_SIZE * PATCH_SIZE * 3)
    pixel_values[: patch_rows * patch_cols] = patches
    padded_positions = torch.full((max_patches, 2), -1, dtype=torch.long)
    padded_positions[: patch_rows * patch_cols] = positions
    soft_tokens = (patch_rows * patch_cols) // (POOLING**2)
    return {
        "width": width,
        "height": height,
        "channels": channels,
        "target_width": target_width,
        "target_height": target_height,
        "patch_grid_width": patch_cols,
        "patch_grid_height": patch_rows,
        "valid_patches": patch_rows * patch_cols,
        "soft_tokens": soft_tokens,
        "max_soft_tokens": max_soft_tokens,
        "resized": rgb,
        "pixel_values": pixel_values,
        "positions": padded_positions,
    }


def sha_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def first_values(tensor: torch.Tensor, count: int = 8):
    return [float(v) for v in tensor.reshape(-1)[:count].tolist()]


CASES = [
    ("rgb_2x2", "fixtures/media/rgb_2x2.png", 280),
    ("rgba_2x2", "fixtures/media/rgba_2x2.png", 70),
    ("gray_2x2", "fixtures/media/gray_2x2.png", 70),
    ("quadrants_16x16", "fixtures/media/quadrants_16x16.png", 70),
    ("gradient_16x16", "fixtures/media/gradient_16x16.png", 70),
]


def main():
    preprocess_cases = []
    prepared = []
    for name, source, budget in CASES:
        data = preprocess(ROOT / source, budget)
        prepared.append((name, source, budget, data))
        preprocess_cases.append(
            {
                "name": name,
                "source": source,
                "width": data["width"],
                "height": data["height"],
                "channels": data["channels"],
                "max_soft_tokens": budget,
                "target_width": data["target_width"],
                "target_height": data["target_height"],
                "patch_grid_width": data["patch_grid_width"],
                "patch_grid_height": data["patch_grid_height"],
                "valid_patches": data["valid_patches"],
                "soft_tokens": data["soft_tokens"],
                "resized_sha256": sha_tensor(data["resized"]),
                "pixel_values_sha256": sha_tensor(data["pixel_values"]),
                "positions_sha256": sha_tensor(data["positions"]),
                "first_patch": first_values(data["pixel_values"]),
                "last_patch": first_values(data["pixel_values"][data["valid_patches"] - 1]),
                "first_position": data["positions"][0].tolist(),
                "last_position": data["positions"][data["valid_patches"] - 1].tolist(),
            }
        )

    preprocess_document = {
        "schema_version": 1,
        "provenance": (
            "Generated outside the product with Python 3.13, PyTorch 2.15.0 CPU, "
            "float32, and torch.nn.functional.interpolate(mode='bicubic', "
            "antialias=True, align_corners=False) on uint8 tensors. RGBA is "
            "composited over white and rounded half to even. The official "
            "Gemma4ImageProcessor was not instantiated because torchvision is "
            "not installed in the reference environment."
        ),
        "cases": preprocess_cases,
    }
    (ROOT / "fixtures/media/preprocess_golden.json").write_text(
        json.dumps(preprocess_document, indent=2) + "\n"
    )

    model = AutoModel.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    del tokenizer

    layer_outputs = []

    def capture_layer(_module, _inputs, output):
        layer_outputs.append(output.detach())

    hooks = [
        layer.register_forward_hook(capture_layer)
        for layer in model.vision_tower.encoder.layers
    ]
    patch_output = []

    def capture_patch(_module, _inputs, output):
        patch_output.append(output.detach())

    patch_hook = model.vision_tower.patch_embedder.register_forward_hook(capture_patch)

    vision_cases = []
    for name, source, budget, data in prepared:
        pixel_values = data["pixel_values"][None]
        positions = data["positions"][None]
        input_ids = torch.tensor(
            [[2, 255999] + [258880] * data["soft_tokens"] + [258882, 1]]
        )
        layer_outputs.clear()
        patch_output.clear()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_position_ids=positions,
                return_dict=True,
            )
            features = outputs.image_hidden_states
            pooled = outputs.last_hidden_state.mean(dim=1)[0]
            normalized = torch.nn.functional.normalize(pooled, dim=-1)
        valid = data["valid_patches"]
        vision_cases.append(
            {
                "name": name,
                "source": source,
                "max_soft_tokens": budget,
                "soft_tokens": data["soft_tokens"],
                "input_ids": input_ids[0].tolist(),
                "vision_features_sha256": sha_tensor(features),
                "vision_features_first": first_values(features),
                "patch_output_sha256": sha_tensor(patch_output[0][0, :valid]),
                "patch_output_first": first_values(patch_output[0][0, 0]),
                "layer_sha256": [sha_tensor(layer[0, :valid]) for layer in layer_outputs],
                "layer_first": [first_values(layer[0, 0]) for layer in layer_outputs],
                "image_embedding": [float(v) for v in normalized.tolist()],
                "image_embedding_sha256": sha_tensor(normalized),
            }
        )

    patch_hook.remove()
    for hook in hooks:
        hook.remove()

    vision_document = {
        "schema_version": 1,
        "model": "google/embeddinggemma-2",
        "revision": "f9972ac22e696a105c482202646979e33e4b5a1e",
        "provenance": (
            "Generated outside the product with Transformers 5.22.2 and "
            "PyTorch 2.15.0 CPU in float32. Image-only inputs use the token "
            "layout <bos><|image><|image|>*soft_tokens<image|><eos>, mean "
            "pooling over every token, and L2 normalization."
        ),
        "cases": vision_cases,
    }
    (ROOT / "fixtures/model/image_golden.json").write_text(
        json.dumps(vision_document, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
