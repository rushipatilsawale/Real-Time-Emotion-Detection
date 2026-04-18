import os

mapping = {
    "0": "angry",
    "1": "disgust",
    "2": "fear",
    "3": "happy",
    "4": "sad",
    "5": "surprise",
    "6": "neutral"
}

base_dirs = ["dataset/train", "dataset/val", "dataset/test"]

for base in base_dirs:
    for old_name, new_name in mapping.items():
        old_path = os.path.join(base, old_name)
        new_path = os.path.join(base, new_name)

        if os.path.exists(old_path):
            os.rename(old_path, new_path)

print("✅ Renaming completed!")