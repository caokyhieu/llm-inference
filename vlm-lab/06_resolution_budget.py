"""Count visual tokens and estimate selected memory terms without allocating them."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[224, 336, 448, 896])
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()
    if min(args.layers, args.kv_heads, args.head_dim, *args.sizes) <= 0:
        parser.error("all sizes and decoder dimensions must be positive")
    if any(side % 28 for side in args.sizes):
        parser.error("use already-processed image sizes divisible by 14*2=28")

    mib = 1024**2
    print("Static RGB images; spatial patch=14, spatial merge=2, batch=1.")
    print("Sizes are AFTER resizing. These are analytical terms, not measured VRAM.")
    print(f"KV assumptions: {args.layers} layers, {args.kv_heads} KV heads, "
          f"head dimension {args.head_dim}, FP16, batch=1 (illustrative decoder).")
    print(f"{'Image':>10} {'Patches':>9} {'Visual':>8} "
          f"{'Dense ViT scores MiB':>21} {'Image KV MiB':>14}")
    for side in args.sizes:
        patches = (side // 14) ** 2
        visual = patches // 4
        # One explicit dense score tensor, 16 vision heads, FP16, batch=1.
        dense_scores = 16 * patches**2 * 2
        # K and V per LLM layer, using KV heads (not query heads for GQA).
        image_kv = 2 * args.layers * visual * args.kv_heads * args.head_dim * 2
        print(f"{side:4d}x{side:<4d} {patches:9d} {visual:8d} "
              f"{dense_scores / mib:21.2f} {image_kv / mib:14.2f}")
    print("\nDouble BOTH image dimensions -> 4x tokens -> 16x dense score elements.")
    print("The image-related KV term grows 4x in that example.")
    print("Qwen2.5-VL combines window and full vision attention.")
    print("Flash/SDPA kernels can avoid materializing the dense score tensor.")
    print("Total VRAM also includes weights, text/generated KV, activations, and workspaces.")
    print("These estimates do not establish whether a model fits on an 11 GB GPU.")


if __name__ == "__main__":
    main()
