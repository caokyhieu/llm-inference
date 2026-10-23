"""Pixels -> patches -> embeddings; prove Conv2d == Unfold + Linear."""

import argparse

import torch
from torch import nn
from torch.nn import functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=336)
    parser.add_argument("--width", type=int, default=336)
    parser.add_argument("--patch", type=int, default=14)
    parser.add_argument("--dim", type=int, default=768)
    args = parser.parse_args()
    if min(args.height, args.width, args.patch, args.dim) <= 0:
        parser.error("dimensions and patch size must be positive")
    if args.height % args.patch or args.width % args.patch:
        parser.error("height and width must be divisible by patch size")

    torch.manual_seed(0)
    torch.set_num_threads(1)
    image = torch.rand(1, 3, args.height, args.width)  # B, C, H, W
    patches = F.unfold(image, kernel_size=args.patch, stride=args.patch)
    patches = patches.transpose(1, 2)  # B, N, C*P*P

    projection = nn.Conv2d(3, args.dim, args.patch, stride=args.patch)
    with torch.no_grad():
        feature_map = projection(image)  # B, D, grid_h, grid_w
        embeddings = feature_map.flatten(2).transpose(1, 2)  # B, N, D
        # Use EXACTLY the same weights, flattened in C, patch_y, patch_x order.
        linear_result = F.linear(
            patches, projection.weight.flatten(1), projection.bias
        )
    torch.testing.assert_close(embeddings, linear_result, atol=2e-6, rtol=1e-5)

    # Non-overlapping patches preserve all input pixels before projection.
    reconstructed = F.fold(
        patches.transpose(1, 2), (args.height, args.width),
        kernel_size=args.patch, stride=args.patch,
    )
    torch.testing.assert_close(image, reconstructed)
    print("Image [B,C,H,W]:          ", tuple(image.shape))
    print("Patch grid [rows,cols]:   ", (args.height // args.patch, args.width // args.patch))
    print("Flat patches [B,N,C*P*P]: ", tuple(patches.shape))
    print("Conv feature map:        ", tuple(feature_map.shape))
    print("Embeddings [B,N,D]:      ", tuple(embeddings.shape))
    print("Conv vs Linear max error:", (embeddings - linear_result).abs().max().item())
    print("PASS: equivalent projections; patch extraction reconstructs the image.")
    print("These random weights do not yet encode learned visual meaning.")


if __name__ == "__main__":
    main()
