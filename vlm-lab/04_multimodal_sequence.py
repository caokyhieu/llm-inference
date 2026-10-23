"""Trace a tiny image into placeholder embeddings and causal text attention."""

import torch
from torch import nn

@torch.no_grad()
def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    # Tiny, randomly initialized image -> vision encoder -> merger pipeline.
    image = torch.rand(1, 3, 56, 56)
    patch_embed = nn.Conv2d(3, 16, kernel_size=14, stride=14)
    patches = patch_embed(image).flatten(2).transpose(1, 2)  # [1,16,16]
    position = torch.randn(1, 16, 16) * 0.02
    encoder = nn.TransformerEncoderLayer(
        d_model=16, nhead=2, dim_feedforward=32, dropout=0, batch_first=True
    ).eval()
    features = encoder(patches + position)
    # Same spatial grouping as lesson 03: [B,H/m,m,W/m,m,D].
    grouped = (
        features.reshape(1, 2, 2, 2, 2, 16)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(1, 4, 4 * 16)
    )
    visual = nn.Linear(4 * 16, 32)(grouped)  # [1,4,32]

    # These names/IDs are illustrative, not the Qwen tokenizer's real IDs.
    names = ["<user>", "<vision_start>", "img0", "img1", "img2", "img3",
             "<vision_end>", "color?", "<assistant>"]
    image_id = 9
    ids = torch.tensor([[0, 1, image_id, image_id, image_id, image_id, 2, 3, 4]])
    text_embed = nn.Embedding(10, 32)
    sequence = text_embed(ids)
    image_mask = ids == image_id
    assert image_mask.sum().item() == visual.shape[1]
    sequence[image_mask] = visual.reshape(-1, 32)
    torch.testing.assert_close(sequence[image_mask], visual[0])
    torch.testing.assert_close(sequence[~image_mask], text_embed(ids)[~image_mask])

    length = sequence.shape[1]
    # nn.MultiheadAttention: True means BLOCK this query/key pair.
    blocked = torch.ones(length, length, dtype=torch.bool).triu(diagonal=1)
    decoder_attention = nn.MultiheadAttention(32, 2, dropout=0, batch_first=True).eval()
    hidden, weights = decoder_attention(sequence, sequence, sequence, attn_mask=blocked)
    assert torch.count_nonzero(weights[0][blocked]).item() == 0
    print("Image -> patch embeddings:", tuple(patches.shape))
    print("Vision encoder -> features:", tuple(features.shape))
    print("Merger -> visual embeddings:", tuple(visual.shape))
    print("Replace four image placeholders -> sequence:", tuple(sequence.shape))
    print("\nQuery/key order:", " | ".join(names))
    print("Allowed causal attention (row=query, column=key):\n", (~blocked).int())
    print("Assistant-position attention mass on image slots:",
          weights[0, -1, image_mask[0]].sum().item())

    changed = sequence.clone()
    changed[:, 2:6] += 2  # Change only the image representations.
    changed_hidden, _ = decoder_attention(changed, changed, changed, attn_mask=blocked)
    torch.testing.assert_close(hidden[:, :2], changed_hidden[:, :2])
    assert not torch.allclose(hidden[:, -1], changed_hidden[:, -1])
    print("PASS: changing image slots affects the later assistant, not earlier tokens.")
    print("Random weights demonstrate information flow, not image understanding.")
    print("Qwen uses decoder self-attention here, rather than a separate cross-attention module.")


if __name__ == "__main__":
    main()
