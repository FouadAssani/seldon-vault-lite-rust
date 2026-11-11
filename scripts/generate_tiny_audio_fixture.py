#!/usr/bin/env python3
"""Laboratory-only generator for the tiny official audio fixture.

This script is never run by the delivered product. It builds a small composite
EmbeddingGemma 2 model with:

- the committed tiny text model weights;
- a deterministic two-layer Gemma 4 audio tower with hidden 16, two heads of
  dimension 8, subsampling channels [128, 8], chunk size 2, left context 3, and
  output projection 16;
- a deterministic 0.125 s sine wave that produces 12 log-mel frames and three
  audio soft tokens.

It writes fixtures/model/tiny_audio/config.json, model.safetensors, and
golden.json. The golden records the official log-mel features, SSCP output,
both encoder layer outputs, projected audio features, and the final normalized
embedding. The product test uses this fixture to exercise the audio tower and
the text connection without the 1.5 GB frozen model.

Requirements: Python 3.13, NumPy, PyTorch, Transformers 5.22.2, and the frozen
model cache at .model-cache/f9972ac22e696a105c482202646979e33e4b5a1e.
"""

import json
import os

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers.models.embedding_gemma2.configuration_embedding_gemma2 import (
    EmbeddingGemma2Config,
)
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
    EmbeddingGemma2Model,
)
from transformers.models.gemma4.feature_extraction_gemma4 import (
    Gemma4AudioFeatureExtractor,
)

REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
CACHE = f".model-cache/{REVISION}"
OUTPUT = "fixtures/model/tiny_audio"

AUDIO_CONFIG = {
    "model_type": "gemma4_audio",
    "dtype": "float32",
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "conv_kernel_size": 3,
    "subsampling_conv_channels": [128, 8],
    "attention_chunk_size": 2,
    "attention_context_left": 3,
    "attention_context_right": 0,
    "attention_logit_cap": 50.0,
    "rms_norm_eps": 1e-6,
    "output_proj_dims": 16,
    "residual_weight": 0.5,
    "gradient_clipping": 1e10,
    "hidden_act": "silu",
    "use_clipped_linears": True,
    "attention_invalid_logits_value": -1000000000.0,
}

TOKEN_IDS = {
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
    "boa_token_id": 3,
    "audio_token_id": 4,
    "eoa_token_index": 5,
    "image_token_id": 6,
    "boi_token_id": 7,
    "eoi_token_id": 8,
    "video_token_id": 9,
}


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    torch.manual_seed(20261111)

    text_config = json.load(open("fixtures/model/tiny_text/config.json", encoding="utf-8"))
    config_dict = {
        "model_type": "embedding_gemma2",
        "text_config": text_config,
        "audio_config": AUDIO_CONFIG,
        **TOKEN_IDS,
    }
    config = EmbeddingGemma2Config.from_dict(config_dict)
    model = EmbeddingGemma2Model(config)
    model.eval()

    tiny_weights = load_file("fixtures/model/tiny_text/model.safetensors")
    language_model_state = {
        key[len("language_model.") :]: value
        for key, value in tiny_weights.items()
        if key.startswith("language_model.")
    }
    missing, unexpected = model.language_model.load_state_dict(
        language_model_state, strict=False
    )
    if missing or unexpected:
        raise RuntimeError(f"language model load mismatch: {missing} {unexpected}")

    # The frozen model uses finite clipping bounds. The constructor defaults to
    # +/-infinity, which the product loader rejects as a non-finite weight.
    for module in model.modules():
        if hasattr(module, "input_min") and hasattr(module, "output_max"):
            module.input_min.fill_(-20.0)
            module.input_max.fill_(20.0)
            module.output_min.fill_(-20.0)
            module.output_max.fill_(20.0)

    sample_rate = 16_000
    sample_count = 2_000
    time = np.arange(sample_count, dtype=np.float64) / sample_rate
    samples = (0.5 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)
    fade = 80
    samples[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    samples[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    feature_extractor = Gemma4AudioFeatureExtractor.from_pretrained(CACHE)
    features = feature_extractor([samples], sampling_rate=sample_rate, return_tensors="pt")
    input_features = features["input_features"]
    input_features_mask = features["input_features_mask"]

    mask = input_features_mask[0].numpy().copy()
    length = len(mask)
    for _ in range(2):
        output_length = (length + 2 - 3) // 2 + 1
        mask = mask[::2][:output_length]
        length = len(mask)
    audio_tokens = int(mask.sum())
    if audio_tokens != 3:
        raise RuntimeError(f"expected three audio tokens, got {audio_tokens}")

    input_ids = [
        TOKEN_IDS["bos_token_id"],
        TOKEN_IDS["boa_token_id"],
        *[TOKEN_IDS["audio_token_id"]] * audio_tokens,
        TOKEN_IDS["eoa_token_index"],
        TOKEN_IDS["eos_token_id"],
    ]

    layer_outputs = {}
    hooks = []
    for index, layer in enumerate(model.audio_tower.layers):
        hooks.append(
            layer.register_forward_hook(
                lambda module, inputs, output, index=index: layer_outputs.__setitem__(
                    index, output.detach()
                )
            )
        )
    with torch.no_grad():
        output = model(
            input_ids=torch.tensor([input_ids]),
            input_features=input_features,
            input_features_mask=input_features_mask,
        )
        subsampled = model.audio_tower.subsample_conv_projection(
            input_features, input_features_mask
        )
    for hook in hooks:
        hook.remove()

    valid_after_subsampling = subsampled[1][0].numpy().astype(bool)
    sscp_output = subsampled[0][0].numpy()[valid_after_subsampling].astype(np.float32)
    layers = [
        layer_outputs[index][0].numpy()[valid_after_subsampling].astype(np.float32)
        for index in range(len(model.audio_tower.layers))
    ]
    audio_features = output.audio_hidden_states.numpy().astype(np.float32)
    embedding = torch.nn.functional.normalize(
        output.last_hidden_state.mean(dim=1), p=2, dim=1
    )[0].numpy().astype(np.float32)

    with open(os.path.join(OUTPUT, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config_dict, handle, indent=2)
        handle.write("\n")

    state = {
        key: value.detach().to(torch.float32).contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(state, os.path.join(OUTPUT, "model.safetensors"))

    golden = {
        "schema_version": 1,
        "model": "google/embeddinggemma-2",
        "revision": REVISION,
        "provenance": (
            "Generated outside the product with Transformers 5.22.2, PyTorch "
            "2.15.0 CPU, and NumPy 2.5.4 in float32. The tiny text weights are "
            "the committed tiny_text fixture; the audio tower is deterministic "
            "random initialization with finite clipping bounds. The input is a "
            "0.125 s 440 Hz sine that produces 12 mel frames and three audio "
            "soft tokens."
        ),
        "sample_rate": sample_rate,
        "samples": [float(value) for value in samples],
        "mel_features": input_features[0].numpy().astype(np.float32).tolist(),
        "mel_mask": [bool(value) for value in input_features_mask[0].numpy()],
        "input_ids": input_ids,
        "audio_tokens": audio_tokens,
        "sscp_output": sscp_output.tolist(),
        "layer_outputs": [layer.tolist() for layer in layers],
        "audio_features": audio_features.tolist(),
        "embedding": embedding.tolist(),
    }
    with open(os.path.join(OUTPUT, "golden.json"), "w", encoding="utf-8") as handle:
        json.dump(golden, handle, indent=2)
        handle.write("\n")
    print(f"wrote {OUTPUT}/config.json, model.safetensors, and golden.json")
    print(
        f"samples={len(samples)} mel={input_features.shape[1]} "
        f"tokens={audio_tokens} embedding_dim={embedding.shape[0]}"
    )


if __name__ == "__main__":
    main()
