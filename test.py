import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("emotion_model.h5")

# Emotion labels (IMPORTANT)
emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Load test image (change filename if needed)
img = cv2.imread("test.png", 0)

# Check if image loaded properly
if img is None:
    print("❌ Error: Image not found")
    exit()

# Preprocess image
img = cv2.resize(img, (48,48))
img = img / 255.0
img = img.reshape(1,48,48,1)

# Predict emotion
pred = model.predict(img)
emotion = emotions[np.argmax(pred)]
confidence = np.max(pred) * 100

# Output result
print(f"Detected Emotion: {emotion} ({confidence:.2f}%)")