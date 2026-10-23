"""Learn a tiny projector between frozen image features and fixed text vectors."""

import argparse

import torch
from torch import nn
from torch.nn import functional as F


def make_images(samples_per_color, generator):
    labels = torch.arange(3).repeat_interleave(samples_per_color)
    rgb = F.one_hot(labels, num_classes=3).float()[:, :, None, None]
    noise = torch.randn(len(labels), 3, 16, 16, generator=generator) * 0.15
    images = (rgb + noise).clamp(0, 1)
    return images, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    torch.manual_seed(7)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(42)
    train_images, train_labels = make_images(64, generator)
    test_images, test_labels = make_images(32, generator)

    # A deliberately simple frozen vision encoder: average RGB, then rotate
    # into 8 dimensions. It cannot recognize objects or spatial relationships.
    vision = nn.Linear(3, 8, bias=False)
    with torch.no_grad():
        vision.weight.copy_(torch.linalg.qr(torch.randn(8, 3)).Q)
    vision.requires_grad_(False)
    train_features = vision(train_images.mean(dim=(2, 3))).detach()
    test_features = vision(test_images.mean(dim=(2, 3))).detach()

    # Stand-ins for frozen text embeddings for "red", "green", and "blue".
    # The text side is a fixed lookup table, not a language model.
    text_vectors = torch.eye(8)[:3]
    projector = nn.Linear(8, 8, bias=False)
    with torch.no_grad():
        projector.weight.copy_(torch.eye(8))
    optimizer = torch.optim.Adam(projector.parameters(), lr=0.03)

    def logits(features):
        aligned = F.normalize(projector(features), dim=-1)
        return aligned @ F.normalize(text_vectors, dim=-1).T / 0.1

    @torch.no_grad()
    def accuracy(features, labels):
        return (logits(features).argmax(-1) == labels).float().mean().item()

    before = accuracy(test_features, test_labels)
    print("Both image and text vectors have width 8 before any training.")
    print(f"Held-out accuracy before alignment: {before:.1%}")
    for step in range(args.steps):
        # Supervised image-to-class-text alignment, not full CLIP pretraining.
        loss = F.cross_entropy(logits(train_features), train_labels)
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == args.steps:
            print(f"step={step + 1:3d} loss={loss.item():.4f}")

    after = accuracy(test_features, test_labels)
    wrong_pairs = accuracy(test_features, (test_labels + 1) % 3)
    no_image = accuracy(torch.zeros_like(test_features), test_labels)
    assert vision.weight.grad is None
    print(f"Held-out accuracy after alignment:  {after:.1%}")
    print(f"Accuracy with mismatched pairs:     {wrong_pairs:.1%}")
    print(f"Accuracy with image features zeroed:{no_image:7.1%}")
    print("Only the projector was trained; the vision encoder stayed frozen.")
    print("This teaches alignment on noisy color swatches, not general VLM ability.")


if __name__ == "__main__":
    main()
