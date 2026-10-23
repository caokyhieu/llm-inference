"""Group actual 2x2 spatial neighbors, then project to the LLM width."""

import argparse

import torch
from torch import nn


def group_neighbors(grid, merge=2):
    """[B,H,W,D] -> [B,(H/m)*(W/m),m*m*D] in spatial block order."""
    batch, height, width, dim = grid.shape
    if merge <= 0 or height % merge or width % merge:
        raise ValueError("grid dimensions must be divisible by a positive merge size")
    return (
        grid.reshape(batch, height // merge, merge, width // merge, merge, dim)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, (height // merge) * (width // merge), merge * merge * dim)
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=32)
    args = parser.parse_args()
    if args.grid_size <= 0 or args.grid_size % 2:
        parser.error("grid-size must be a positive even number")
    torch.manual_seed(0)
    torch.set_num_threads(1)

    ids = torch.arange(16).reshape(1, 4, 4, 1)
    grouped_ids = group_neighbors(ids)
    print("Patch IDs in a 4x4 grid:\n", ids[0, :, :, 0])
    print("Correct 2x2 groups:\n", grouped_ids[0])
    assert grouped_ids[0, 0].tolist() == [0, 1, 4, 5]
    print("A naive row-major reshape would group [0,1,2,3], which is wrong.")

    vision_dim, llm_dim = 32, 64
    grid = torch.randn(1, args.grid_size, args.grid_size, vision_dim)
    # Normalize each patch, concatenate neighbors, then apply a learned MLP.
    norm = nn.RMSNorm(vision_dim, eps=1e-6)
    grouped = group_neighbors(norm(grid))
    projector = nn.Sequential(
        nn.Linear(4 * vision_dim, 4 * vision_dim),
        nn.GELU(),
        nn.Linear(4 * vision_dim, llm_dim),
    )
    visual_tokens = projector(grouped)
    assert visual_tokens.shape == (1, args.grid_size**2 // 4, llm_dim)
    print("\nVision grid [B,H,W,Dv]:    ", tuple(grid.shape))
    print("Concatenated [B,N/4,4*Dv]:", tuple(grouped.shape))
    print("Visual tokens [B,N/4,Dllm]:", tuple(visual_tokens.shape))
    print("PASS: four neighboring patches become one visual token.")
    print("The merger learns a projection; it does not generate a caption.")
    print("Qwen preprocessing also reorders patches; this toy starts in grid order.")


if __name__ == "__main__":
    main()
