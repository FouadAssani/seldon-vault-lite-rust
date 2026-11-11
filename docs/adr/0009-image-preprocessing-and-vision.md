# ADR 0009 — Native image preprocessing and vision tower

- **Status:** accepted for milestone M4.
- **Date:** 2026-11-11.
- **Depends on:** ADR 0001 (std only), ADR 0003 (model freeze), ADR 0004
  (text embeddings), ADR 0007 (PNG/DEFLATE), ADR 0008 (JPEG).

## Context

Milestone M4 requires real image embeddings in the same 768-dimensional space
as text. The frozen google/embeddinggemma-2 checkpoint contains a Gemma 4
vision tower and a projection into the text hidden space. The official image
processor is a Torchvision-backend Gemma4ImageProcessor, but torchvision was
not installed in the reference environment; its tensor operations were
reproduced directly with PyTorch and committed as goldens.

The product must remain std-only, must not launch an external decoder or
processor, and must not invent an image embedding from metadata. The
preprocessing and vision math therefore had to be implemented in Rust and
validated against the official reference.

## Decisions

### 1. Preprocessing is bit-exact with the reference tensor path

The official processor performs, per image:

1. RGB conversion (do_convert_rgb): grayscale is replicated; RGBA is
   composited over white with the float tensor path and rounded half to even.
2. Aspect-ratio preserving resize to the largest size that fits the visual
   token budget, with both sides multiples of pooling_kernel_size *
   patch_size (48 pixels).
3. Antialiased bicubic resize (tvF.resize, bicubic, antialias, align_corners
   false) on uint8 tensors.
4. Rescale by 1/255.
5. Patchification into 16x16 patches in pixel-major, channel-last order.
6. Padded (x, y) positions, with (-1, -1) for padding.

The Rust implementation reproduces the torchvision uint8 path exactly:
separable horizontal then vertical passes, Keys cubic convolution with
a = -0.5, per-output normalized weights, 16-bit fixed-point quantization, and
round-half-up integer accumulation. It matches the reference resized bytes,
padded patch values, and positions bit for bit on the committed fixtures.

Rationale: a tolerance on preprocessing would hide color, resize, or patch
ordering errors that later stages cannot recover. Bit-exact preprocessing makes
the vision parity result meaningful.

### 2. Explicit preprocessing caps

The product refuses an image before allocating:

- input side at most 16,384 pixels, input at most 16,777,216 pixels;
- visual token budgets limited to the official 70, 140, 280, 560, and 1,120;
- patch budget at most 10,080 patches;
- resize taps at most 8,192 and resize weights at most 1,048,576;
- padded patch tensor at most 7,741,440 FP32 values.

The default budget is 280 soft tokens, matching the frozen processor.

### 3. Only real patches enter the vision tower

The official encoder receives the padded 2,520-patch tensor and masks padding
keys in bidirectional attention. Padding queries are discarded by the pooler.
Processing only the real patches is mathematically equivalent and avoids a
large amount of useless scalar work. The pooler still uses the official 3x3
binning over the real patch grid, scales by sqrt(768), and strips padding.

### 4. Vision tower details

The implementation follows the frozen gemma4_vision configuration:

- patch linear projection over 3*16*16 = 768 values;
- learned two-axis position embeddings, summed and zeroed for padding;
- 16 bidirectional encoder layers, 12 query heads and 12 key/value heads of
  dimension 64, attention scale 1.0;
- RMSNorm on query and key with learned weights, unscaled RMSNorm on value;
- axial 2-D RoPE with theta 100, using head_dim/4 frequencies and the
  H-H-W-W frequency layout;
- gated GELU feed-forward with tanh approximation;
- 3x3 spatial average pooling and sqrt(hidden size) scaling;
- RMSNorm without a learned scale and a 768 to 512 linear projection
  (embed_vision).

Clipped linears, feature standardization, non-axial RoPE, FP16, attention
bias, and non-zero dropout are rejected as unsupported.

### 5. Image prompt and embedding

An image-only input is encoded as:

    <bos><|image><|image|> * soft_tokens<image|><eos>

The projected vision soft tokens replace the placeholder embeddings. The
per-layer embedding (PLE) signal is computed from the final inputs_embeds
including the vision features, matching the EmbeddingGemma2 model override
(not the generic Gemma 4 path that computes PLE from padding embeddings).
Mean pooling covers every token, then the 512 to 768 projection, L2
normalization, and Matryoshka truncation with renormalization.

The special ids are read from the frozen configuration and validated against
the tokenizer: bos 2, eos 1, pad 0, begin-of-image 255,999, image 258,880,
end-of-image 258,882.

### 6. Tolerances and fixtures

- Preprocessing: exact SHA-256 equality for resized bytes, padded patch
  values, and positions.
- Vision layer probes: absolute plus relative tolerance
  (atol <= 5e-5, rtol <= 5e-4) on the first values of the patch embedder
  and each of the 16 encoder layers. The wider bound than the text-layer
  tolerance reflects 16 stacked scalar layers and compiler-dependent
  accumulation order: the measured maximum on the frozen probe is about
  3.7e-4 on a large activation, while the final embedding remains within
  1e-4 and above 0.99999 cosine.
- Final image embedding: cosine >= 0.99999 and maximum absolute difference
  <= 1e-4 after L2 normalization.

The fixtures are fixtures/media/preprocess_golden.json and
fixtures/model/image_golden.json, generated outside the product by
scripts/generate_image_goldens.py with Transformers 5.22.2 and PyTorch
2.15.0 CPU in float32. The generator and the model cache are laboratory-only;
the delivered binary never runs them.

## Consequences

- The default image parity test is expensive because the official processor
  upscales a tiny image to 576 patches at the 70-token budget and the scalar
  tower runs 16 layers. The default suite covers one full case; the five-case
  corpus is an ignored test for manual verification.
- The preprocessing fixtures cover grayscale, RGB, RGBA, gradients,
  structured quadrants, 90-degree and 180-degree rotations, and 2:1 and 1:2
  aspect ratios. Vision parity for the rotation and aspect-ratio variants,
  text-to-image ranking, and image-to-image ranking still need dedicated
  fixtures before M4 can be declared complete.
- The CLI can embed an image with embed --image. Image documents are stored
  with modality image, indexed with empty chunk text, and searched by exact
  vector search. An end-to-end test indexes one image and retrieves it with a
  text query. A two-image ranking test ranks a colorful quadrant image above a
  grayscale image for the text query "a colorful four-quadrant test pattern";
  an image query with the quadrant image ranks the identical image first and
  the grayscale image second. Broader ranking quality remains pending. No
  image modality is declared supported end to end until those paths are
  tested.
