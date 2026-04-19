"""
Image Enhancement Module (Unit 2 - IP Syllabus)
=================================================
Demonstrates spatial and frequency domain enhancement techniques:
- Thresholding, Image Negative, Contrast Stretching
- Histogram Equalization, Gray Level Slicing
- Image Smoothing: Gaussian (low-pass), Median filtering
- Image Sharpening: Laplacian (high-pass), Unsharp masking
- Histogram visualization
"""

import cv2
import numpy as np


def histogram_equalization(img):
    """
    Histogram Equalization (Unit 2)
    Redistributes pixel intensities to span the full range,
    improving contrast especially in poorly-lit images.
    """
    return cv2.equalizeHist(img)


def contrast_stretching(img):
    """
    Contrast Stretching / Min-Max Normalization (Unit 2)
    Stretches the pixel intensity range to occupy full 0-255 range.
    Formula: out = (pixel - min) / (max - min) * 255
    """
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val == min_val:
        return img
    stretched = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return stretched


def apply_threshold(img, thresh=127):
    """
    Binary Thresholding (Unit 2)
    Converts grayscale image to binary using Otsu's method
    for automatic threshold selection.
    """
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def image_negative(img):
    """
    Image Negative (Unit 2)
    Inverts all pixel intensities: output = 255 - input
    Useful for enhancing white/gray detail in dark regions.
    """
    return 255 - img


def gray_level_slicing(img, low=100, high=200):
    """
    Gray Level Slicing (Unit 2)
    Highlights pixels in a specific intensity range [low, high],
    suppressing all others. Useful for isolating features.
    """
    sliced = np.zeros_like(img)
    sliced[(img >= low) & (img <= high)] = 255
    return sliced


def smooth_gaussian(img, ksize=5):
    """
    Gaussian Smoothing / Low-Pass Spatial Filter (Unit 2)
    Applies a Gaussian kernel to blur the image, reducing
    high-frequency noise while preserving edges better than box filter.
    """
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def smooth_median(img, ksize=5):
    """
    Median Filtering (Unit 2)
    Replaces each pixel with the median of its neighborhood.
    Excellent for removing salt-and-pepper noise while preserving edges.
    """
    return cv2.medianBlur(img, ksize)


def sharpen_laplacian(img):
    """
    Laplacian Sharpening / High-Pass Spatial Filter (Unit 2)
    Applies the Laplacian operator (2nd derivative) to detect edges,
    then adds them back to the original image for sharpening.
    """
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    # Add edges back to original for sharpening effect
    sharpened = cv2.convertScaleAbs(img.astype(np.float64) - laplacian)
    return sharpened


def sharpen_unsharp(img, sigma=1.0, strength=1.5):
    """
    Unsharp Masking / Derivative Filter (Unit 2)
    Creates a blurred version, subtracts it from the original to get
    the detail mask, then adds the amplified detail back.
    Formula: sharpened = original + strength * (original - blurred)
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def plot_histogram(img, width=256, height=200):
    """
    Histogram Visualization (Unit 2)
    Computes and draws the grayscale histogram as an image.
    Returns a BGR image of the histogram plot.
    """
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist / hist.max() * height  # Normalize to fit display height

    # Create blank canvas (dark background)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw histogram bars
    for x in range(256):
        bar_height = int(hist[x][0])
        # Color gradient: blue for dark pixels, green for mid, red for bright
        color = (
            max(0, 255 - x),        # Blue channel
            min(x, 255 - x) * 2,    # Green channel (peaks at mid)
            x                        # Red channel
        )
        cv2.line(canvas, (x, height), (x, height - bar_height), color, 1)

    return canvas


def get_all_enhancements(face_img):
    """
    Applies all enhancement techniques and returns a dictionary
    of {name: processed_image} for grid display.
    """
    results = {
        "Original": face_img,
        "Hist. Equalized": histogram_equalization(face_img),
        "Contrast Stretch": contrast_stretching(face_img),
        "Image Negative": image_negative(face_img),
        "Gaussian Blur": smooth_gaussian(face_img),
        "Median Filter": smooth_median(face_img),
        "Laplacian Sharp": sharpen_laplacian(face_img),
        "Unsharp Mask": sharpen_unsharp(face_img),
    }
    return results
