"""
Image Transforms Module (Unit 6 - IP Syllabus)
================================================
Demonstrates 2D transforms and frequency domain filtering:
- Discrete Fourier Transform (DFT) and magnitude spectrum
- Discrete Cosine Transform (DCT)
- Singular Value Decomposition (SVD)
- Frequency domain filters: Ideal LP, Butterworth LP, High-pass
"""

import cv2
import numpy as np


def compute_dft(img):
    """
    2D Discrete Fourier Transform (Unit 6 - DFT)
    Converts the image from spatial domain to frequency domain.
    Returns the magnitude spectrum (log-scaled) for visualization.
    Low frequencies (smooth areas) at center, high frequencies (edges) at borders.
    """
    # Convert to float
    f = np.float32(img)

    # Compute 2D DFT
    dft = cv2.dft(f, flags=cv2.DFT_COMPLEX_OUTPUT)

    # Shift zero-frequency component to center
    dft_shift = np.fft.fftshift(dft)

    # Compute magnitude spectrum: log(1 + |F(u,v)|)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    magnitude_spectrum = np.log1p(magnitude)

    # Normalize to 0-255 for display
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude_spectrum.astype(np.uint8)


def compute_dct(img):
    """
    2D Discrete Cosine Transform (Unit 6 - Cosine Transform)
    Similar to DFT but uses only real cosines (no imaginary part).
    Widely used in JPEG compression. Energy compaction property
    concentrates most signal energy in top-left corner.
    """
    # Convert to float
    f = np.float32(img)

    # Compute 2D DCT
    dct = cv2.dct(f)

    # Log transform for visualization
    dct_vis = np.log1p(np.abs(dct))

    # Normalize
    dct_vis = cv2.normalize(dct_vis, None, 0, 255, cv2.NORM_MINMAX)
    return dct_vis.astype(np.uint8)


def compute_svd(img, k=10):
    """
    Singular Value Decomposition (Unit 6 - SVD)
    Decomposes image matrix A = U * S * V^T
    Reconstructs using only top-k singular values,
    demonstrating image approximation / compression.
    """
    f = np.float64(img)

    # Compute SVD
    U, S, Vt = np.linalg.svd(f, full_matrices=False)

    # Reconstruct with top-k singular values
    reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

    # Clip and convert
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    return reconstructed


def apply_ideal_lowpass(img, cutoff=30):
    """
    Ideal Low-Pass Filter in Frequency Domain (Unit 6)
    Passes all frequencies within a circle of radius 'cutoff'
    and blocks everything outside. Causes ringing artifacts
    due to the sharp cutoff.
    
    H(u,v) = 1 if D(u,v) <= cutoff, else 0
    """
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # DFT
    f = np.float32(img)
    dft = cv2.dft(f, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Create ideal low-pass mask
    mask = np.zeros((rows, cols, 2), np.float32)
    for u in range(rows):
        for v in range(cols):
            dist = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
            if dist <= cutoff:
                mask[u, v] = 1

    # Apply filter
    filtered = dft_shift * mask

    # Inverse DFT
    f_ishift = np.fft.ifftshift(filtered)
    result = cv2.idft(f_ishift)
    result = cv2.magnitude(result[:, :, 0], result[:, :, 1])
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)


def apply_butterworth_lowpass(img, cutoff=30, order=2):
    """
    Butterworth Low-Pass Filter (Unit 6)
    Smoother transition than ideal filter (no ringing).
    
    H(u,v) = 1 / (1 + (D(u,v)/cutoff)^(2*order))
    
    Higher order = sharper cutoff (approaches ideal filter).
    """
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # DFT
    f = np.float32(img)
    dft = cv2.dft(f, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Create Butterworth mask
    mask = np.zeros((rows, cols, 2), np.float32)
    for u in range(rows):
        for v in range(cols):
            dist = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
            if dist == 0:
                mask[u, v] = 1
            else:
                mask[u, v] = 1 / (1 + (dist / cutoff) ** (2 * order))

    # Apply filter
    filtered = dft_shift * mask

    # Inverse DFT
    f_ishift = np.fft.ifftshift(filtered)
    result = cv2.idft(f_ishift)
    result = cv2.magnitude(result[:, :, 0], result[:, :, 1])
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)


def apply_highpass(img, cutoff=30):
    """
    Ideal High-Pass Filter in Frequency Domain (Unit 6)
    Blocks low frequencies (smooth areas) and passes high
    frequencies (edges, details). Result shows edge structure.
    
    H(u,v) = 0 if D(u,v) <= cutoff, else 1
    """
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # DFT
    f = np.float32(img)
    dft = cv2.dft(f, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Create ideal high-pass mask (inverse of low-pass)
    mask = np.ones((rows, cols, 2), np.float32)
    for u in range(rows):
        for v in range(cols):
            dist = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
            if dist <= cutoff:
                mask[u, v] = 0

    # Apply filter
    filtered = dft_shift * mask

    # Inverse DFT
    f_ishift = np.fft.ifftshift(filtered)
    result = cv2.idft(f_ishift)
    result = cv2.magnitude(result[:, :, 0], result[:, :, 1])
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)


def get_all_transforms(face_img):
    """
    Applies all transform techniques and returns a dictionary
    of {name: processed_image} for grid display.
    """
    results = {
        "DFT Spectrum": compute_dft(face_img),
        "DCT": compute_dct(face_img),
        "SVD (k=10)": compute_svd(face_img, k=10),
        "Ideal LPF": apply_ideal_lowpass(face_img, cutoff=15),
        "Butterworth LPF": apply_butterworth_lowpass(face_img, cutoff=15, order=2),
        "Ideal HPF": apply_highpass(face_img, cutoff=10),
    }
    return results
