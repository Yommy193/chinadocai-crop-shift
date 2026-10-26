"""Small, dependency-free perceptual-hash utilities for plate audit evidence."""

from __future__ import annotations

from PIL import Image


def dhash64(image: Image.Image) -> int:
    """Return the standard 64-bit grayscale difference hash of an image."""
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(grayscale.get_flattened_data())
    value = 0
    for row in range(8):
        start = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[start + column] > pixels[start + column + 1])
    return value


def hamming_distance(first: int, second: int) -> int:
    """Count differing dHash bits with explicit unsigned-64 validation."""
    if not 0 <= first < 2**64 or not 0 <= second < 2**64:
        raise ValueError("dHash values must be unsigned 64-bit integers")
    return (first ^ second).bit_count()


def chunks16(value: int) -> tuple[int, int, int, int]:
    """Split an unsigned 64-bit hash into four deterministic 16-bit chunks."""
    if not 0 <= value < 2**64:
        raise ValueError("dHash values must be unsigned 64-bit integers")
    return tuple((value >> shift) & 0xFFFF for shift in (48, 32, 16, 0))
