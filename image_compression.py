"""
Image Compression Module (Unit 4 - IP Syllabus)
=================================================
Demonstrates compression concepts:
- JPEG compression at different quality levels (lossy)
- PNG compression (lossless)
- Run-Length Encoding (RLE)
- Huffman Coding demonstration
- Compression ratio comparison visualization
"""

import cv2
import numpy as np
from collections import Counter
import heapq


def compress_jpeg(img, quality=50):
    """
    JPEG Compression (Unit 4 - Lossy Compression)
    Encodes the image as JPEG at the given quality level (1-100).
    Lower quality = higher compression = more artifacts.
    JPEG uses DCT internally for transform coding.
    Returns the decoded image and compressed size in bytes.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, encoded = cv2.imencode('.jpg', img, encode_params)
    compressed_size = len(encoded)
    # Decode back to see the quality loss
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return decoded, compressed_size


def compress_png(img):
    """
    PNG Compression (Unit 4 - Lossless Compression)
    Encodes the image as PNG (lossless).
    No quality loss, but typically larger file than lossy JPEG.
    Uses deflate algorithm internally.
    Returns compressed size in bytes.
    """
    _, encoded = cv2.imencode('.png', img)
    return len(encoded)


def run_length_encode(binary_img):
    """
    Run-Length Encoding / RLE (Unit 4)
    Compresses data by storing consecutive runs of the same value
    as (value, count) pairs. Very effective for binary images
    with large uniform regions.
    Returns the RLE data and compression ratio.
    """
    flat = binary_img.flatten()
    original_size = len(flat)

    rle = []
    current_val = flat[0]
    count = 1

    for i in range(1, len(flat)):
        if flat[i] == current_val:
            count += 1
        else:
            rle.append((current_val, count))
            current_val = flat[i]
            count = 1
    rle.append((current_val, count))

    # Each RLE entry = 2 values (value + count)
    compressed_size = len(rle) * 2
    ratio = original_size / compressed_size if compressed_size > 0 else 0

    return rle, original_size, compressed_size, ratio


def huffman_encode(img):
    """
    Huffman Coding (Unit 4)
    Variable-length prefix coding that assigns shorter codes
    to more frequent pixel values. Optimal prefix-free code.
    Returns the Huffman table, and compression statistics.
    """
    flat = img.flatten()
    original_bits = len(flat) * 8  # 8 bits per pixel

    # Count frequency of each pixel value
    freq = Counter(flat)

    # Build Huffman tree using priority queue
    heap = [[count, [value, ""]] for value, count in freq.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        heap[0][1][1] = "0"
    else:
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                if isinstance(pair, list):
                    pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                if isinstance(pair, list):
                    pair[1] = '1' + pair[1]
            merged = [lo[0] + hi[0]] + lo[1:] + hi[1:]
            heapq.heappush(heap, merged)

    # Extract codes
    huffman_tree = heap[0]
    codes = {}

    def extract_codes(node):
        if len(node) == 2 and isinstance(node[0], (int, np.integer)):
            codes[node[0]] = node[1]
        elif isinstance(node, list):
            for item in node[1:]:
                if isinstance(item, list):
                    extract_codes(item)

    extract_codes(huffman_tree)

    # Calculate compressed size
    compressed_bits = sum(len(codes.get(p, "00000000")) * c for p, c in freq.items())
    ratio = original_bits / compressed_bits if compressed_bits > 0 else 0

    # Get top-5 most common pixel values and their codes
    top_codes = []
    for val, count in freq.most_common(5):
        code = codes.get(val, "?")
        top_codes.append(f"Pixel {val}: '{code}' (freq: {count})")

    return codes, original_bits, compressed_bits, ratio, top_codes


def build_compression_comparison(img):
    """
    Compression Comparison (Unit 4 - Coding Redundancy)
    Creates images at JPEG Q=10, Q=50, Q=90 for side-by-side comparison.
    Returns dict of {label: image} and stats text.
    """
    original_size = img.size  # bytes (uncompressed)
    png_size = compress_png(img)

    results = {}
    stats = []

    stats.append(f"Original: {original_size} bytes")
    stats.append(f"PNG (lossless): {png_size} bytes")

    for q in [10, 50, 90]:
        decoded, size = compress_jpeg(img, quality=q)
        label = f"JPEG Q={q}"
        results[label] = decoded
        ratio = original_size / size if size > 0 else 0
        stats.append(f"{label}: {size} bytes (ratio: {ratio:.1f}x)")

    # RLE on thresholded version
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, rle_orig, rle_comp, rle_ratio = run_length_encode(binary)
    stats.append(f"RLE (binary): {rle_comp} vals (ratio: {rle_ratio:.1f}x)")

    # Huffman
    _, huff_orig, huff_comp, huff_ratio, _ = huffman_encode(img)
    stats.append(f"Huffman: {huff_comp} bits (ratio: {huff_ratio:.2f}x)")

    return results, stats
