"""Run real Qwen2.5-VL preprocessing and a tiny random vision encoder offline."""

import argparse
import inspect

import torch
from PIL import Image, ImageDraw
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Optional local image; otherwise use synthetic quadrants")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.max_tokens < 4:
        parser.error("max-tokens must be at least 4")
    torch.manual_seed(0)
    torch.set_num_threads(1)

    if args.image:
        with Image.open(args.image) as opened:
            image = opened.convert("RGB")
    else:
        image = Image.new("RGB", (448, 448), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 223, 223), fill="red")
        draw.rectangle((224, 0, 447, 223), fill="green")
        draw.rectangle((0, 224, 223, 447), fill="blue")

    # Construct the processor directly: no from_pretrained, no network request.
    processor = Qwen2VLImageProcessor(
        min_pixels=4 * 28 * 28, max_pixels=args.max_tokens * 28 * 28,
        patch_size=14, temporal_patch_size=2, merge_size=2,
    )
    inputs = processor(images=image, return_tensors="pt")
    pixels, grid = inputs["pixel_values"], inputs["image_grid_thw"]
    expected_visual_tokens = int(grid.prod(-1).sum().item()) // 4
    # A static image is repeated across the temporal patch dimension.
    temporal_patches = pixels.reshape(-1, 3, 2, 14, 14)
    torch.testing.assert_close(temporal_patches[:, :, 0], temporal_patches[:, :, 1])

    # Same implementation as Qwen, deliberately tiny and randomly initialized.
    config = Qwen2_5_VLVisionConfig(
        depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
        out_hidden_size=64, patch_size=14, temporal_patch_size=2,
        spatial_merge_size=2, window_size=112, fullatt_block_indexes=[1],
    )
    config._attn_implementation = "eager"
    vision = Qwen2_5_VisionTransformerPretrainedModel(config).eval()

    def trace(name):
        def hook(module, module_inputs, output):
            print(f"{name:24s} {tuple(module_inputs[0].shape)} -> {tuple(output.shape)}")
        return hook

    handles = [vision.patch_embed.register_forward_hook(trace("Real Conv3d patch embed")),
               vision.merger.register_forward_hook(trace("Real spatial merger"))]
    try:
        print("Original image [width,height]:", image.size)
        print("image_grid_thw [T,Hpatch,Wpatch]:", grid.tolist())
        print("Processed pixel_values:", tuple(pixels.shape))
        print("Static-image temporal copies match; 1176 = 3*2*14*14 values per patch.")
        visual = vision(pixels, grid_thw=grid)
    finally:
        for handle in handles:
            handle.remove()
    assert visual.shape == (expected_visual_tokens, 64)
    assert expected_visual_tokens <= args.max_tokens
    print("Final visual embeddings:", tuple(visual.shape))
    print("PASS: real vision output count matches T*Hpatch*Wpatch / merge_size**2.")
    print("Layer 0 uses window attention; layer 1 uses full attention.")
    print("Random tiny model: outputs have no pretrained visual semantics.")
    print("Read the installed implementation:", inspect.getsourcefile(type(vision)))


if __name__ == "__main__":
    main()
