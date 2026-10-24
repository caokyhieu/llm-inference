"""Learn Triton with vector addition, then trace Nano-vLLM's KV-cache writer.

Run: uv run --no-sync python kernel_lab.py
Add --kv-cache to exercise the actual cache kernel without loading a model.
"""

import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # One GPU program owns a tile of BLOCK_SIZE elements.
    program = tl.program_id(0)
    offsets = program * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # The final tile may extend beyond the tensor's allocation.
    valid = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=valid, other=0.0)
    y = tl.load(y_ptr + offsets, mask=valid, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=valid)


def run_add(size: int, block_size: int):
    # Host code allocates tensors and launches the GPU programs.
    x = torch.arange(size, device="cuda", dtype=torch.float32)
    y = torch.full_like(x, 10.0)
    output = torch.empty_like(x)
    num_programs = triton.cdiv(size, block_size)
    add_kernel[(num_programs,)](x, y, output, size, BLOCK_SIZE=block_size)

    # Compare with a separate PyTorch implementation, including the final tile.
    torch.testing.assert_close(output, x + y, rtol=0, atol=0)
    print(f"ADD: {size} elements, {num_programs} programs, {block_size} elements/program")
    for program in range(min(num_programs, 4)):
        start = program * block_size
        end = min(start + block_size, size)
        print(f"  program {program}: valid element indices {start}..{end - 1}")
    if num_programs > 4:
        print("  ...")
    print(f"  masked positions in the final program: {num_programs * block_size - size}")
    print(f"  first results: {output[:6].tolist()}")
    print("  PASS: exact match with PyTorch")


def run_kv_cache():
    from nanovllm.layers.attention import store_kvcache

    # Selecting K/V from a packed QKV allocation creates strided tensor views,
    # just as projection outputs can do in the model.
    num_tokens, num_heads, head_dim, block_size = 4, 2, 4, 256
    packed = torch.arange(
        num_tokens * 3 * num_heads * head_dim, device="cuda", dtype=torch.float32
    ).view(num_tokens, 3, num_heads, head_dim)
    keys, values = packed[:, 1], packed[:, 2]
    assert not keys.is_contiguous()

    shape = (2, block_size, num_heads, head_dim)
    k_cache = torch.full(shape, -1.0, device="cuda")
    v_cache = torch.full_like(k_cache, -1.0)
    expected_k, expected_v = k_cache.clone(), v_cache.clone()
    slot_ids = [257, 2, -1, 255]
    slots = torch.tensor(slot_ids, device="cuda", dtype=torch.int32)

    # PyTorch reference: scatter each token into its assigned physical slot.
    flat_k = expected_k.view(-1, num_heads, head_dim)
    flat_v = expected_v.view(-1, num_heads, head_dim)
    for token, slot in enumerate(slot_ids):
        if slot != -1:
            flat_k[slot] = keys[token]
            flat_v[slot] = values[token]

    # Execute the unchanged kernel from nanovllm/layers/attention.py.
    store_kvcache(keys, values, k_cache, v_cache, slots)
    torch.testing.assert_close(k_cache, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(v_cache, expected_v, rtol=0, atol=0)

    print("KV CACHE: one GPU program per input token")
    print(f"  K shape: {tuple(keys.shape)}, strides: {keys.stride()}")
    for token, slot in enumerate(slot_ids):
        if slot == -1:
            print(f"  program {token}: slot -1, skip this token")
        else:
            print(f"  program {token}: slot {slot}, block {slot // block_size}, offset {slot % block_size}")
    print("  PASS: K/V match PyTorch; untouched slots retain their sentinel values")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1003)
    parser.add_argument("--block-size", type=int, choices=[64, 128, 256, 512, 1024], default=256)
    parser.add_argument("--kv-cache", action="store_true")
    args = parser.parse_args()
    if args.size < 1:
        parser.error("--size must be positive")
    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    run_add(args.size, args.block_size)
    if args.kv_cache:
        run_kv_cache()


if __name__ == "__main__":
    main()
