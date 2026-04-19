"""
Image Analysis Module (Unit 3 - IP Syllabus)
=============================================
Demonstrates image segmentation and feature extraction techniques:
- Edge Detection: Canny, Sobel
- Segmentation: Watershed, Thresholding-based
- Feature Extraction: Hu Moments, Texture (GLCM-inspired), Contour-based
- Boundary Representation: Chain Code
"""

import cv2
import numpy as np


def detect_edges_canny(img, low=50, high=150):
    """
    Canny Edge Detection (Unit 3 - Edge-based Segmentation)
    Uses a multi-stage algorithm:
    1. Gaussian smoothing to reduce noise
    2. Gradient computation using Sobel
    3. Non-maximum suppression to thin edges
    4. Hysteresis thresholding with low/high thresholds
    """
    return cv2.Canny(img, low, high)


def detect_edges_sobel(img):
    """
    Sobel Edge Detection (Unit 3 - Classification of Edges)
    Computes gradient in X and Y directions separately using
    Sobel operator (1st derivative), then combines them.
    Highlights horizontal and vertical edges.
    """
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    # Combine both gradients: magnitude = sqrt(Gx^2 + Gy^2)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return magnitude


def watershed_segmentation(img):
    """
    Watershed Segmentation (Unit 3 - Watershed Transformation)
    Treats the image as a topographic surface and finds
    boundaries by flooding from markers.
    """
    # Need BGR for watershed
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to get binary
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal with morphological opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background (dilate)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground (distance transform + threshold)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(img_color, markers)

    # Create output: mark boundaries in white
    output = img.copy()
    output[markers == -1] = 255

    return output


def compute_hu_moments(img):
    """
    Hu Moments (Unit 3 - Moment-based Descriptor)
    Computes 7 Hu invariant moments which are invariant to
    translation, rotation, and scale.
    Returns both the moments array and a formatted string.
    """
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments).flatten()
    # Log transform for better readability
    hu_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    text_lines = []
    for i, val in enumerate(hu_log):
        text_lines.append(f"H{i+1}: {val:.4f}")

    return hu_moments, text_lines


def compute_texture_features(img):
    """
    Texture Features (Unit 3 - Texture-based Features)
    Computes texture descriptors inspired by GLCM properties:
    - Contrast: measures intensity variation between neighbors
    - Energy: measures textural uniformity
    - Homogeneity: measures closeness of distribution to diagonal
    Simplified computation using pixel neighborhood statistics.
    """
    img_float = img.astype(np.float64)

    # Contrast: variance of pixel intensities
    contrast = np.var(img_float)

    # Energy: sum of squared pixel values (normalized)
    normalized = img_float / 255.0
    energy = np.sum(normalized ** 2) / img.size

    # Homogeneity: mean of absolute differences between adjacent pixels
    diff_h = np.abs(np.diff(img_float, axis=1))
    diff_v = np.abs(np.diff(img_float, axis=0))
    homogeneity = 1.0 / (1.0 + np.mean(np.concatenate([diff_h.flatten(), diff_v.flatten()])))

    # Mean intensity
    mean_intensity = np.mean(img_float)

    # Smoothness
    variance = np.var(img_float / 255.0)
    smoothness = 1 - 1 / (1 + variance)

    features = {
        "Contrast": contrast,
        "Energy": energy,
        "Homogeneity": homogeneity,
        "Mean": mean_intensity,
        "Smoothness": smoothness
    }

    text_lines = [f"{k}: {v:.4f}" for k, v in features.items()]
    return features, text_lines


def boundary_chain_code(img):
    """
    Chain Code (Unit 3 - Boundary Representation)
    Extracts the largest contour and computes the Freeman chain code
    (8-directional). Returns the contour image and chain code string.
    
    Direction mapping:
    3 2 1
    4 . 0
    5 6 7
    """
    # Threshold
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Draw contours on blank canvas
    contour_img = np.zeros_like(img)

    if len(contours) > 0:
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_img, [largest], -1, 255, 1)

        # Compute chain code from contour points
        chain = []
        pts = largest.reshape(-1, 2)
        # Direction vectors: 0=right, 1=up-right, ..., 7=down-right
        dx_map = {(1,0):0, (1,-1):1, (0,-1):2, (-1,-1):3,
                  (-1,0):4, (-1,1):5, (0,1):6, (1,1):7}

        for i in range(1, min(len(pts), 50)):  # Limit to first 50 for display
            dx = int(np.sign(pts[i][0] - pts[i-1][0]))
            dy = int(np.sign(pts[i][1] - pts[i-1][1]))
            direction = dx_map.get((dx, dy), -1)
            if direction >= 0:
                chain.append(str(direction))

        chain_str = "".join(chain[:30]) + ("..." if len(chain) > 30 else "")
    else:
        chain_str = "No contour found"

    return contour_img, chain_str


def get_all_analyses(face_img):
    """
    Applies all analysis techniques and returns a dictionary
    of {name: processed_image} for grid display.
    Also returns text annotations for moments/texture.
    """
    canny = detect_edges_canny(face_img)
    sobel = detect_edges_sobel(face_img)
    watershed = watershed_segmentation(face_img)
    contour_img, chain_str = boundary_chain_code(face_img)
    _, hu_lines = compute_hu_moments(face_img)
    _, tex_lines = compute_texture_features(face_img)

    results = {
        "Canny Edges": canny,
        "Sobel Edges": sobel,
        "Watershed Seg.": watershed,
        "Chain Code": contour_img,
    }

    annotations = {
        "hu_moments": hu_lines,
        "texture": tex_lines,
        "chain_code": chain_str
    }

    return results, annotations
