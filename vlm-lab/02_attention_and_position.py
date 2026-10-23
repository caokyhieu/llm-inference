"""Inspect Q/K/V attention and see why a vision encoder needs position."""

import math

import torch
from torch import nn


def attention(x, qkv):
    q, k, v = qkv(x).chunk(3, dim=-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    weights = scores.softmax(dim=-1)  # Each query row sums to one.
    return weights @ v, weights


@torch.no_grad()
def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_printoptions(precision=3, sci_mode=False)
    # Four made-up patch embeddings; this is NOT a trained vision encoder.
    patches = torch.randn(1, 4, 4)
    qkv = nn.Linear(4, 12, bias=False)
    context, weights = attention(patches, qkv)
    print("Q/K/V each have shape [1,4,4].")
    print("Attention rows=query patch, columns=key patch:\n", weights[0])
    print("Context = attention @ V; shape:", tuple(context.shape))
    torch.testing.assert_close(weights.sum(-1), torch.ones(1, 4))

    permutation = torch.tensor([2, 0, 3, 1])
    shuffled_context, _ = attention(patches[:, permutation], qkv)
    no_position_error = (shuffled_context - context[:, permutation]).abs().max()
    torch.testing.assert_close(shuffled_context, context[:, permutation])
    print("\nWithout position, shuffling input only shuffles output.")
    print("Permutation equivariance error:", no_position_error.item())

    # Keep these positions fixed while moving patch CONTENT between positions.
    # Four basis vectors are an illustrative positional embedding, not Qwen RoPE.
    position = torch.eye(4).unsqueeze(0)
    positioned_context, _ = attention(patches + position, qkv)
    moved_context, _ = attention(patches[:, permutation] + position, qkv)
    position_error = (moved_context - positioned_context[:, permutation]).abs().max()
    assert position_error > 1e-3
    print("With fixed positions, moving patches changes their context.")
    print("Position-sensitive difference:", position_error.item())
    print("PASS: rows normalize; position makes spatial arrangement observable.")
    print("Attention weights alone are not a reliable explanation of an answer.")


if __name__ == "__main__":
    main()
