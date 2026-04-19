"""Quick verification script to test all IP modules."""
import cv2
import numpy as np
from image_enhancement import get_all_enhancements
from image_analysis import get_all_analyses
from image_transforms import get_all_transforms
from image_compression import build_compression_comparison
from utils import create_labeled_grid

# Load test image
img = cv2.imread('test.png', 0)
if img is None:
    print('No test image, creating dummy 48x48')
    img = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
else:
    img = cv2.resize(img, (48, 48))

print(f"Test image shape: {img.shape}")

# Phase 1: Enhancement (Unit 2)
enh = get_all_enhancements(img)
print(f"[Unit 2] Enhancement: {len(enh)} techniques - {list(enh.keys())}")

# Phase 2: Analysis (Unit 3)
ana, ann = get_all_analyses(img)
print(f"[Unit 3] Analysis: {len(ana)} techniques - {list(ana.keys())}")
print(f"         Hu moments: {len(ann['hu_moments'])} values")
print(f"         Texture: {len(ann['texture'])} features")
print(f"         Chain code: {ann['chain_code'][:40]}")

# Phase 3: Transforms (Unit 6)
trn = get_all_transforms(img)
print(f"[Unit 6] Transforms: {len(trn)} techniques - {list(trn.keys())}")

# Phase 4: Compression (Unit 4)
comp, stats = build_compression_comparison(img)
print(f"[Unit 4] Compression: {len(comp)} comparisons")
for s in stats:
    print(f"         {s}")

# Grid test
images = list(enh.values())
labels = list(enh.keys())
grid = create_labeled_grid(images, labels, cols=2)
print(f"\nGrid panel created: {grid.shape}")

print("\nAll IP modules working correctly!")
