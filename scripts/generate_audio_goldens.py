#!/usr/bin/env python3
"""Laboratory-only generator for the M5 audio goldens.

This script is never run by the delivered product. It reads the committed WAV
fixtures, runs the official Transformers audio feature extractor and the
official EmbeddingGemma 2 model in float32, and writes:

- fixtures/media/audio_preprocess_golden.json
- fixtures/model/audio_golden.json

Requirements: Python 3.13, NumPy, PyTorch, Transformers 5.22.2, and the frozen
model cache at .model-cache/f9972ac22e696a105c482202646979e33e4b5a1e.

The WAV fixtures themselves are generated outside this script. See
fixtures/media/README.md for the documented deterministic synthesis and the
macOS say/afconvert command used for the French speech clip.
"""

import hashlib
import json
import wave

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.models.gemma4.feature_extraction_gemma4 import (
    Gemma4AudioFeatureExtractor,
)

REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
MODEL = "google/embeddinggemma-2"
CACHE = f".model-cache/{REVISION}"
CASES = [
    ("silence_1s", "fixtures/media/audio/silence_1s.wav"),
    ("short_tone_100ms", "fixtures/media/audio/short_tone_100ms.wav"),
    ("long_tone_3s", "fixtures/media/audio/long_tone_3s.wav"),
    ("french_speech", "fixtures/media/audio/french_speech.wav"),
    ("music_chords_2s", "fixtures/media/audio/music_chords_2s.wav"),
    ("rain_noise_2s", "fixtures/media/audio/rain_noise_2s.wav"),
]


def read_wav(path):
    """Read a mono or multichannel PCM WAV and return float32 mono samples."""
    with wave.open(path, "rb") as handle:
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        data = handle.readframes(frames)
    if sample_width == 2:
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 1:
        samples = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 3:
        raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
        values = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        values = (values << 8) >> 8
        samples = values.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"unsupported sample width {sample_width}")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples.astype(np.float32), sample_rate, channels, sample_width


def f32_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


def first_eight(values):
    return [float(value) for value in np.asarray(values, dtype=np.float32).ravel()[:8]]


def main():
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(CACHE)
    feature_extractor = Gemma4AudioFeatureExtractor.from_pretrained(CACHE)
    model = AutoModel.from_pretrained(CACHE, dtype=torch.float32)
    model.eval()
    tower = model.audio_tower

    preprocess = {
        "schema_version": 1,
        "model": MODEL,
        "revision": REVISION,
        "cases": [],
    }
    golden = {
        "schema_version": 1,
        "model": MODEL,
        "revision": REVISION,
        "provenance": (
            "Generated outside the product with Transformers 5.22.2 and "
            "PyTorch 2.15.0 CPU in float32. Audio-only inputs use the token "
            "layout <bos><|audio><|audio|>*soft_tokens<audio|><eos>, mean "
            "pooling over every token, and L2 normalization. Layer probes "
            "capture the output of every audio encoder layer over valid soft "
            "tokens."
        ),
        "cases": [],
    }

    for name, path in CASES:
        samples, sample_rate, channels, sample_width = read_wav(path)
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

        input_ids = (
            [tokenizer.bos_token_id, tokenizer.boa_token_id]
            + [tokenizer.audio_token_id] * audio_tokens
            + [tokenizer.eoa_token_id, tokenizer.eos_token_id]
        )

        layer_outputs = {}
        hooks = []
        for index, layer in enumerate(tower.layers):
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
            audio_output = tower(
                input_features,
                input_features_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            subsampled = tower.subsample_conv_projection(
                input_features, input_features_mask
            )
        for hook in hooks:
            hook.remove()

        embedding = torch.nn.functional.normalize(
            output.last_hidden_state.mean(dim=1), p=2, dim=1
        )[0].numpy().astype(np.float32)
        audio_features = output.audio_hidden_states.numpy().astype(np.float32)
        valid_after_subsampling = subsampled[1][0].numpy().astype(bool)
        sscp_output = subsampled[0][0].numpy()[valid_after_subsampling].astype(np.float32)
        layer_hashes = []
        layer_first = []
        for index in range(len(tower.layers)):
            layer_values = layer_outputs[index][0].numpy()[valid_after_subsampling].astype(
                np.float32
            )
            layer_hashes.append(f32_hash(layer_values))
            layer_first.append(first_eight(layer_values))

        mel_features = input_features[0].numpy().astype(np.float32)
        preprocess["cases"].append(
            {
                "name": name,
                "source": path,
                "sample_rate": sample_rate,
                "channels": channels,
                "sample_width": sample_width,
                "samples": int(len(samples)),
                "padded_samples": ((len(samples) + 127) // 128) * 128,
                "mel_frames": int(input_features.shape[1]),
                "valid_mel_frames": int(input_features_mask.sum()),
                "audio_tokens": audio_tokens,
                "mel_features_sha256": f32_hash(mel_features),
                "mel_features_first": first_eight(mel_features),
            }
        )
        golden["cases"].append(
            {
                "name": name,
                "source": path,
                "sample_rate": sample_rate,
                "channels": channels,
                "samples": int(len(samples)),
                "mel_frames": int(input_features.shape[1]),
                "valid_mel_frames": int(input_features_mask.sum()),
                "audio_tokens": audio_tokens,
                "input_ids": input_ids,
                "mel_features_sha256": f32_hash(mel_features),
                "mel_features_first": first_eight(mel_features),
                "sscp_output_sha256": f32_hash(sscp_output),
                "sscp_output_first": first_eight(sscp_output),
                "layer_sha256": layer_hashes,
                "layer_first": layer_first,
                "audio_features_sha256": f32_hash(audio_features),
                "audio_features_first": first_eight(audio_features),
                "audio_embedding": [float(value) for value in embedding],
                "audio_embedding_sha256": f32_hash(embedding),
            }
        )
        print(
            f"{name}: samples={len(samples)} mel={input_features.shape[1]} "
            f"valid={int(input_features_mask.sum())} tokens={audio_tokens}"
        )

    with open("fixtures/media/audio_preprocess_golden.json", "w", encoding="utf-8") as handle:
        json.dump(preprocess, handle, indent=2)
        handle.write("\n")
    with open("fixtures/model/audio_golden.json", "w", encoding="utf-8") as handle:
        json.dump(golden, handle, indent=2)
        handle.write("\n")
    print("wrote fixtures/media/audio_preprocess_golden.json")
    print("wrote fixtures/model/audio_golden.json")


if __name__ == "__main__":
    main()
