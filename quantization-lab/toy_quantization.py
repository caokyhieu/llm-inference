"""See rounding error from simple symmetric, uniform weight quantization."""

weights = [-1.0, -0.3, 0.0, 0.2, 0.8]
print("Original weights:", weights)
for bits in (8, 4):
    limit = 2 ** (bits - 1) - 1
    scale = max(abs(value) for value in weights) / limit
    integers = [max(-limit, min(limit, round(value / scale))) for value in weights]
    restored = [value * scale for value in integers]
    error = max(abs(a - b) for a, b in zip(weights, restored))
    print(f"\nUniform INT{bits}: scale={scale:.6f}")
    print("  integer codes:", integers)
    print("  reconstructed:", [round(value, 6) for value in restored])
    print(f"  maximum absolute error: {error:.6f}")
print("\nThese Python lists illustrate the arithmetic; they are not packed storage.")
print("The model experiment uses NF4, a different, non-uniform 4-bit format.")
