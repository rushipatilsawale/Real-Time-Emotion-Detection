"""
Real-Time Emotion Detection — Enhanced with IP Syllabus Concepts
=================================================================
This is the main application that integrates all 6 units of the
Image Processing syllabus into a real-time webcam-based emotion detector.

Controls:
    1  - Toggle Enhancement panel    (Unit 2)
    2  - Toggle Analysis panel       (Unit 3)
    3  - Toggle Compression panel    (Unit 4)
    4  - Toggle Transforms panel     (Unit 6)
    H  - Toggle Histogram window
    P  - Toggle preprocessing (histogram equalization before CNN)
    ESC - Quit

Unit 1 (Intro): Grayscale conversion, resizing, pixel normalization, image formats
Unit 5 (Object Recognition): Haar Cascade face detection, CNN emotion classification
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Import IP syllabus modules
from image_enhancement import get_all_enhancements, histogram_equalization, plot_histogram
from image_analysis import get_all_analyses
from image_transforms import get_all_transforms
from image_compression import build_compression_comparison
from utils import create_labeled_grid, create_text_panel, add_hud_overlay


# ─────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────

# Load trained CNN model (Unit 5 - Object Recognition / Pattern Classification)
model = load_model("emotion_model.h5")

# Load Haar Cascade face detector (Unit 5 - Automated Object Recognition)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Emotion labels — 7 pattern classes
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ─────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────

active_panels = set()          # Currently open panels: '1','2','3','4','H'
preprocess_on = False          # Whether to apply hist eq before CNN
last_face = None               # Cache the last detected face for panels

# Window names
WIN_MAIN = "Emotion Detector (IP Course Project)"
WIN_ENHANCE = "Unit 2: Image Enhancement"
WIN_ANALYSIS = "Unit 3: Image Analysis"
WIN_COMPRESS = "Unit 4: Image Compression"
WIN_TRANSFORM = "Unit 6: Image Transforms"
WIN_HISTOGRAM = "Histogram"
WIN_FEATURES = "Features & Stats"


def process_face_for_prediction(face_gray):
    """
    Preprocesses a grayscale face for CNN prediction.
    Unit 1 concepts: resizing (sampling), normalization (linear operation)
    """
    face = cv2.resize(face_gray, (48, 48))     # Resampling (Unit 1)
    face = face / 255.0                         # Normalization (Unit 1)
    face = face.reshape(1, 48, 48, 1)           # Reshape for CNN
    return face


def build_enhancement_panel(face_img):
    """Builds the Unit 2 Enhancement visualization panel."""
    enhancements = get_all_enhancements(face_img)
    images = list(enhancements.values())
    labels = list(enhancements.keys())
    return create_labeled_grid(images, labels, cols=2)


def build_analysis_panel(face_img):
    """Builds the Unit 3 Analysis visualization panel."""
    analyses, annotations = get_all_analyses(face_img)
    images = list(analyses.values())
    labels = list(analyses.keys())
    grid = create_labeled_grid(images, labels, cols=2)

    # Build text panel for feature values
    all_lines = []
    all_lines.append("--- Hu Moments ---")
    all_lines.extend(annotations["hu_moments"])
    all_lines.append("")
    all_lines.append("--- Texture Features ---")
    all_lines.extend(annotations["texture"])
    all_lines.append("")
    all_lines.append(f"Chain: {annotations['chain_code']}")

    text_panel = create_text_panel(all_lines, width=grid.shape[1],
                                   title="Feature Extraction (Unit 3)")

    # Stack grid and text panel vertically
    combined = np.vstack([grid, text_panel])
    return combined


def build_compression_panel(face_img):
    """Builds the Unit 4 Compression visualization panel."""
    comp_images, stats = build_compression_comparison(face_img)
    images = list(comp_images.values())
    labels = list(comp_images.keys())
    grid = create_labeled_grid(images, labels, cols=3)

    # Stats text panel
    text_panel = create_text_panel(stats, width=grid.shape[1],
                                   title="Compression Stats (Unit 4)")

    combined = np.vstack([grid, text_panel])
    return combined


def build_transform_panel(face_img):
    """Builds the Unit 6 Transforms visualization panel."""
    transforms = get_all_transforms(face_img)
    images = list(transforms.values())
    labels = list(transforms.keys())
    return create_labeled_grid(images, labels, cols=3)


# ─────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────

# Start webcam (Unit 1 - Image Acquisition)
cap = cv2.VideoCapture(0)

print("=" * 50)
print("  Real-Time Emotion Detection")
print("  IP Course Project - All 6 Units")
print("=" * 50)
print("\n  Controls:")
print("    1 - Enhancement (Unit 2)")
print("    2 - Analysis (Unit 3)")
print("    3 - Compression (Unit 4)")
print("    4 - Transforms (Unit 6)")
print("    H - Histogram")
print("    P - Toggle Preprocessing")
print("    ESC - Quit\n")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to grab frame")
        break

    # ── Unit 1: Color Space Conversion ──
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Unit 5: Face Detection (Object Recognition) ──
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]      # Extract face region
        face_48 = cv2.resize(face_roi, (48, 48))  # Resize for processing

        # Cache face for panel displays
        last_face = face_48.copy()

        # ── Preprocessing toggle (Unit 2: Histogram Equalization) ──
        if preprocess_on:
            face_for_pred = histogram_equalization(face_48)
        else:
            face_for_pred = face_48

        # ── Unit 5: CNN Pattern Classification ──
        face_input = process_face_for_prediction(face_for_pred)
        pred = model.predict(face_input, verbose=0)
        emotion = emotions[np.argmax(pred)]
        confidence = np.max(pred) * 100

        # ── Draw detection results ──
        # Blue rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 130, 0), 2)

        # Emotion label with confidence
        text = f"{emotion} ({confidence:.1f}%)"
        # Background for text
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - 30), (x + tw + 8, y - 2), (255, 130, 0), cv2.FILLED)
        cv2.putText(frame, text, (x + 4, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        # Show preprocessing indicator
        if preprocess_on:
            cv2.putText(frame, "[HIST EQ ON]", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # ── HUD Overlay ──
    add_hud_overlay(frame, active_panels, preprocess_on)

    # ── Show main window ──
    cv2.imshow(WIN_MAIN, frame)

    # ── Show active panels (only when a face is available) ──
    if last_face is not None:
        if '1' in active_panels:
            panel = build_enhancement_panel(last_face)
            cv2.imshow(WIN_ENHANCE, panel)
        else:
            cv2.destroyWindow(WIN_ENHANCE) if cv2.getWindowProperty(WIN_ENHANCE, cv2.WND_PROP_VISIBLE) >= 1 else None

        if '2' in active_panels:
            panel = build_analysis_panel(last_face)
            cv2.imshow(WIN_ANALYSIS, panel)
        else:
            cv2.destroyWindow(WIN_ANALYSIS) if cv2.getWindowProperty(WIN_ANALYSIS, cv2.WND_PROP_VISIBLE) >= 1 else None

        if '3' in active_panels:
            panel = build_compression_panel(last_face)
            cv2.imshow(WIN_COMPRESS, panel)
        else:
            cv2.destroyWindow(WIN_COMPRESS) if cv2.getWindowProperty(WIN_COMPRESS, cv2.WND_PROP_VISIBLE) >= 1 else None

        if '4' in active_panels:
            panel = build_transform_panel(last_face)
            cv2.imshow(WIN_TRANSFORM, panel)
        else:
            cv2.destroyWindow(WIN_TRANSFORM) if cv2.getWindowProperty(WIN_TRANSFORM, cv2.WND_PROP_VISIBLE) >= 1 else None

        if 'H' in active_panels:
            hist_img = plot_histogram(last_face)
            cv2.imshow(WIN_HISTOGRAM, hist_img)
        else:
            cv2.destroyWindow(WIN_HISTOGRAM) if cv2.getWindowProperty(WIN_HISTOGRAM, cv2.WND_PROP_VISIBLE) >= 1 else None

    # ── Keyboard Input ──
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord('1'):
        if '1' in active_panels:
            active_panels.discard('1')
        else:
            active_panels.add('1')
    elif key == ord('2'):
        if '2' in active_panels:
            active_panels.discard('2')
        else:
            active_panels.add('2')
    elif key == ord('3'):
        if '3' in active_panels:
            active_panels.discard('3')
        else:
            active_panels.add('3')
    elif key == ord('4'):
        if '4' in active_panels:
            active_panels.discard('4')
        else:
            active_panels.add('4')
    elif key == ord('h') or key == ord('H'):
        if 'H' in active_panels:
            active_panels.discard('H')
        else:
            active_panels.add('H')
    elif key == ord('p') or key == ord('P'):
        preprocess_on = not preprocess_on
        status = "ON" if preprocess_on else "OFF"
        print(f"  Preprocessing (Hist. Eq.): {status}")

# ── Cleanup ──
cap.release()
cv2.destroyAllWindows()
print("\n✅ Application closed.")