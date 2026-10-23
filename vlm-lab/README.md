# VLM internals: seven small experiments

Start with the **architecture / “B” section** at the end of your pasted lesson.
These scripts make each step observable before you load a pretrained VLM.
They print tensor shapes, small attention matrices, and numerical checks.

Everything runs on **CPU**. No GPU, dataset, API key, or model-weight download
is needed. Only lesson 05 trains anything: a tiny linear projector on generated
color images. Lesson 07 uses the real Transformers implementation with a tiny,
randomly initialized Qwen vision configuration. It is not a pretrained model.

## Setup and first run

From the workspace root:

```bash
cd vlm-lab
python ../setup_env.py
uv run --locked python 01_patch_embedding.py
```

The setup helper prepares the verified FlashAttention wheel and runs
`uv sync --locked` at the root. Subsequent syncs can use that uv command directly.
The shared workspace uses Python 3.12 and PyTorch 2.6.0. Nano-vLLM,
quantization, and this lab use one root `.venv` and `uv.lock`; `uv` discovers
them even when commands run from this directory.
Installation needs network access for packages; the lessons themselves run offline.
`torch` may install CUDA dependencies even though these examples use CPU tensors.

Run all six core lessons:

```bash
uv run --locked python run_all.py
```

For the optional Qwen implementation lesson:

```bash
uv run --locked python 07_qwen_internals.py
# Or run every lesson:
uv run --locked python run_all.py --qwen
```

The shared environment includes the Qwen dependencies by default and pins
Transformers to 4.57.6 because lesson 07 imports internal classes whose names
can change between releases. No separate VLM environment or extra flag is needed.

## Follow this order

Read each file, predict its output shapes, run it, then try the modification.
`B` means batch size, `N` token count, `D` embedding width; images use `[B,C,H,W]`.

| Script | Question it answers | Try changing |
| --- | --- | --- |
| [01_patch_embedding.py](01_patch_embedding.py) | How do pixels become a sequence? Why can Conv2d replace Linear? | Patch size and embedding width |
| [02_attention_and_position.py](02_attention_and_position.py) | How do patches exchange information? Why add position? | The patch permutation or position vectors |
| [03_spatial_merger.py](03_spatial_merger.py) | How do four nearby patches become one LLM-width vector? | The patch grid size |
| [04_multimodal_sequence.py](04_multimodal_sequence.py) | Where do visual vectors enter text embeddings? Who can attend to whom? | Put `color?` before the image and inspect its causal mask row |
| [05_learn_alignment.py](05_learn_alignment.py) | Why does equal vector width fail to create shared meaning? | The number of training steps |
| [06_resolution_budget.py](06_resolution_budget.py) | Why does image resolution affect token count and memory? | Image sizes and KV-cache dimensions |
| [07_qwen_internals.py](07_qwen_internals.py) | What does the actual Qwen implementation do differently? | A local image and its visual-token budget |

### 01 — Patching and projection

For non-overlapping square patches of side `P`:

```text
N = (H / P) * (W / P)
pixels per RGB patch = 3 * P * P
[B,3,H,W] -> [B,N,3*P*P] -> [B,N,D]
```

The default solves your first exercise: `336 / 14 = 24`, so `N = 576`
and output shape is **`[1,576,768]`**.

```bash
uv run --locked python 01_patch_embedding.py --height 224 --width 224 --patch 16 --dim 512
```

That gives `[1,196,512]`. The script copies the same Conv2d weights into the
Linear calculation, checks numerical equivalence, and reconstructs the pixels
from the unprojected patches. Non-divisible dimensions are rejected explicitly.

### 02 — Attention and position

```text
Q = X Wq; K = X Wk; V = X Wv
A = softmax(Q K^T / sqrt(dk))
context = A V
```

Each row of `A` is one query patch's distribution over key patches. It sums to
one. Without position, permuting patches just permutes the resulting outputs.
Assigning fixed position vectors lets changing a patch's location change its
context. This is a one-head demonstration; its position vectors are not Qwen's
rotary position encoding. The random weights have no learned object semantics.

### 03 — Spatial merger

A `2x2` group means spatial neighbors. In a row-major `4x4` grid, the first
group contains patch IDs `[0,1,4,5]`, rather than the first four flat IDs.
The script makes that reorder explicit, normalizes each patch, concatenates
each group, and applies an MLP:

```text
[B,Hgrid,Wgrid,Dvision] -> [B,N/4,4*Dvision] -> [B,N/4,Dllm]
```

For a `448x448` processed image and patch size 14, there are 1024 patches and
256 merged visual tokens. A merger changes sequence length and embedding width;
it does not translate the image into words.

```bash
uv run --locked python 03_spatial_merger.py --grid-size 48
```

This gives 2304 patches and 576 visual tokens. For your second exercise,
setting the LLM width to 2048 would give **`[1,576,2048]`**. The script uses 64
to keep the computation small.

### 04 — From image to a multimodal sequence

This file connects a small patch projection, one vision Transformer layer,
a merger, placeholder replacement, and decoder self-attention. Text embeddings
stay in their slots; image placeholders receive continuous visual vectors.
The vision encoder uses bidirectional attention. The decoder uses a causal mask:
later answer positions can use earlier image slots. Earlier positions cannot
attend directly to later slots.

Changing only the visual vectors must change the later assistant representation
while leaving the representations before the image unchanged. This checks the
information path, not answer quality. The simplified decoder attention omits
Qwen's complete decoder stack, position handling, and token generation.

For exercise 3: a caption is a text bottleneck that can omit small details,
positions, and relationships. Continuous visual features can preserve details
useful for a later question, though the encoder and merger also discard some
information. Matching embedding widths allows the tensors to connect;
**training is needed to make their meanings compatible**.

### 05 — Learn alignment yourself

The generated dataset contains noisy red, green, and blue swatches. A frozen
encoder maps average RGB into width-8 features. Three fixed width-8 vectors
stand in for the color words. Only an `8x8` projector learns, using cross-entropy
on cosine similarities to those text vectors.

With the default seed, the verified run produced:

```text
Held-out accuracy before alignment:   0.0%
Held-out accuracy after alignment:  100.0%
Accuracy with mismatched pairs:       0.0%
Accuracy with image features zeroed: 33.3%
```

```bash
uv run --locked python 05_learn_alignment.py --steps 20
```

The test images are newly generated samples from the same three-color task.
These results demonstrate learning a correspondence, not general visual
reasoning. This is supervised image-to-class-text alignment; full CLIP training
uses paired image/text encoders, and VLM instruction tuning uses other objectives.

### 06 — Token and memory arithmetic

For already-resized static images with spatial patch 14 and merge size 2:

```text
visual tokens = H * W / (28 * 28)
explicit dense vision attention scores: B * heads * Npatches^2 * bytes
image-prefix KV cache: 2 * B * layers * Nvisual * KV_heads * head_dim * bytes
```

The spatial merge is why the effective spatial factor here is `14 * 2 = 28`.
Temporal patching is a separate dimension, shown in lesson 07.

```bash
uv run --locked python 06_resolution_budget.py --sizes 224 448 896 --layers 28 --kv-heads 2
```

Doubling both image dimensions gives four times as many visual tokens and
sixteen times as many dense attention-score elements. The table estimates one
explicit vision score tensor and the image contribution to the decoder KV cache.
Its decoder dimensions are illustrative, not a Qwen3B configuration or a total
VRAM prediction. Qwen combines window/full attention, and efficient kernels can
avoid storing the dense score matrix. Weights and other memory terms still matter.

### 07 — Inspect real Qwen components without downloading weights

The real Qwen2.5-VL image processor resizes the image, normalizes pixels,
duplicates a static frame for temporal patching, and arranges patches for the
spatial merger. A tiny instance of the actual vision implementation then runs
two layers, one with window attention and one with full attention.

Default observed shapes:

```text
image_grid_thw                 [1,32,32]
pixel_values                  [1024,1176]   # 1176 = RGB * 2 frames * 14 * 14
Conv3d patch embedding        [1024,32]
2x2 merger output             [256,64]
```

The temporal copies are packed into each patch, so one static image still has
`T=1` temporal grid group. The script checks both temporal copies and token
counts, traces module shapes with hooks, and prints the installed source path.

```bash
uv run --locked python 07_qwen_internals.py --max-tokens 64
uv run --locked python 07_qwen_internals.py --image /path/to/invoice.jpg --max-tokens 256
```

Inspect `Qwen2_5_VisionPatchEmbed`, `Qwen2_5_VLPatchMerger`, and
`Qwen2_5_VisionTransformerPretrainedModel.forward` in that source file.
To follow the later handoff into the LLM, look for `image_token_id`,
`get_placeholder_mask`, and `masked_scatter` in `Qwen2_5_VLModel.forward`.
The full model has pretrained weights, many more layers, a language decoder,
and multimodal position handling; this lesson only runs its tiny vision side.

## Official references

- [PyTorch Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
  and [Unfold](https://docs.pytorch.org/docs/stable/generated/torch.nn.Unfold.html).
- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html),
  including its boolean-mask convention (`True` blocks attention).
- [Qwen2.5-VL overview](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_vl).
- [Pinned Qwen image preprocessing source](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py).
- [Pinned Qwen vision/merger/model source](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py).
- [Pinned Qwen processor source](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py),
  which expands image placeholder tokens before model execution.

After these experiments, the next practical step is pretrained Qwen inference
on one document, followed by OCR/extraction and RAG. Those later topics from the
roadmap are outside this first architecture lab.
