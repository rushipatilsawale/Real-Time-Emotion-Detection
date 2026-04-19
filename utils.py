"""
Utility Module
===============
Helper functions for creating labeled image grids and text overlays
used by the main real-time application.
"""

import cv2
import numpy as np


# Display constants
CELL_SIZE = 120          # Each cell in the grid is 120x120 pixels
LABEL_HEIGHT = 25        # Height reserved for text label above each image
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
FONT_THICKNESS = 1
FONT_COLOR = (0, 255, 0)       # Green text
BG_COLOR = (30, 30, 30)        # Dark gray background
BORDER_COLOR = (80, 80, 80)    # Subtle border


def create_labeled_grid(images, labels, cols=2, cell_size=CELL_SIZE):
    """
    Arranges images into a labeled grid for panel display.
    
    Args:
        images: list of grayscale or BGR images
        labels: list of string labels for each image
        cols: number of columns in the grid
        cell_size: size of each cell (square)
    
    Returns:
        BGR image of the complete grid
    """
    n = len(images)
    rows = (n + cols - 1) // cols  # Ceiling division

    total_w = cols * cell_size
    total_h = rows * (cell_size + LABEL_HEIGHT)

    # Create canvas
    canvas = np.full((total_h, total_w, 3), BG_COLOR, dtype=np.uint8)

    for idx in range(n):
        row = idx // cols
        col = idx % cols

        x = col * cell_size
        y = row * (cell_size + LABEL_HEIGHT)

        # Get and resize image
        img = images[idx]
        if img is None:
            continue

        # Convert grayscale to BGR if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Resize to fit cell
        img_resized = cv2.resize(img, (cell_size, cell_size))

        # Place image below the label area
        canvas[y + LABEL_HEIGHT:y + LABEL_HEIGHT + cell_size, x:x + cell_size] = img_resized

        # Draw label
        label = labels[idx] if idx < len(labels) else ""
        text_size = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        text_x = x + (cell_size - text_size[0]) // 2  # Center text
        text_y = y + LABEL_HEIGHT - 8
        cv2.putText(canvas, label, (text_x, text_y), FONT, FONT_SCALE,
                    FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        # Draw subtle border
        cv2.rectangle(canvas, (x, y + LABEL_HEIGHT),
                      (x + cell_size - 1, y + LABEL_HEIGHT + cell_size - 1),
                      BORDER_COLOR, 1)

    return canvas


def create_text_panel(lines, width=300, line_height=22, title=None):
    """
    Creates a panel image containing text lines.
    Used for displaying feature values, compression stats, etc.
    
    Args:
        lines: list of strings to display
        width: width of the panel
        line_height: pixel height per line
        title: optional title displayed at top in different color
    
    Returns:
        BGR image of the text panel
    """
    n_lines = len(lines) + (2 if title else 0)
    height = max(n_lines * line_height + 20, 60)

    canvas = np.full((height, width, 3), BG_COLOR, dtype=np.uint8)

    y = 20
    if title:
        cv2.putText(canvas, title, (10, y), FONT, 0.55,
                    (0, 200, 255), 1, cv2.LINE_AA)  # Orange title
        y += line_height + 5
        cv2.line(canvas, (10, y - 10), (width - 10, y - 10), BORDER_COLOR, 1)
        y += 5

    for line in lines:
        cv2.putText(canvas, line, (10, y), FONT, FONT_SCALE,
                    (200, 200, 200), FONT_THICKNESS, cv2.LINE_AA)
        y += line_height

    return canvas


def add_text_overlay(frame, text, position, color=(0, 255, 0), scale=0.5):
    """
    Adds styled text with a dark background rectangle onto a frame.
    """
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, scale, 1)
    x, y = position

    # Draw background rectangle
    cv2.rectangle(frame, (x - 2, y - text_h - 4),
                  (x + text_w + 4, y + baseline + 2),
                  (0, 0, 0), cv2.FILLED)

    # Draw text
    cv2.putText(frame, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)
    return frame


def add_hud_overlay(frame, active_panels, preprocess_on):
    """
    Adds a Heads-Up Display showing key bindings and active panels
    at the bottom of the main webcam frame.
    """
    h, w = frame.shape[:2]

    # Semi-transparent overlay at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 110), (w, h), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    y = h - 90
    controls = [
        "[1] Enhancement",
        "[2] Analysis",
        "[3] Compression",
        "[4] Transforms",
        "[H] Histogram",
        "[P] Preprocess: " + ("ON" if preprocess_on else "OFF"),
        "[ESC] Quit"
    ]

    # Draw controls in a row
    x = 10
    for ctrl in controls:
        # Determine if this panel is active
        key = ctrl[1]
        is_active = key in active_panels

        color = (0, 255, 0) if is_active else (150, 150, 150)
        cv2.putText(frame, ctrl, (x, y), FONT, 0.35, color, 1, cv2.LINE_AA)
        text_w = cv2.getTextSize(ctrl, FONT, 0.35, 1)[0][0]
        x += text_w + 15

        if x > w - 50:
            x = 10
            y += 18

    # Title
    cv2.putText(frame, "IP Course Project - Real-Time Emotion Detection",
                (10, h - 95), FONT, 0.4, (0, 200, 255), 1, cv2.LINE_AA)

    return frame
