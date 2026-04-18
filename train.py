# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from model import model

# train_dir = 'dataset/train'
# val_dir = 'dataset/val'
# test_dir = 'dataset/test'

# train_datagen = ImageDataGenerator(rescale=1./255)
# val_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_data = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(48,48),
#     color_mode='grayscale',
#     batch_size=64,
#     class_mode='categorical'
# )

# val_data = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=(48,48),
#     color_mode='grayscale',
#     batch_size=64,
#     class_mode='categorical'
# )

# test_data = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(48,48),
#     color_mode='grayscale',
#     batch_size=64,
#     class_mode='categorical'
# )

# # 🔥 TRAIN MODEL
# model.fit(
#     train_data,
#     epochs=20,
#     validation_data=val_data
# )

# # 🔥 FINAL TEST
# loss, acc = model.evaluate(test_data)
# print("Test Accuracy:", acc)

# # SAVE MODEL
# model.save("emotion_model.h5")














from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from model import model

# Dataset paths
train_dir = 'dataset/train'
val_dir = 'dataset/val'
test_dir = 'dataset/test'

# 🔥 DATA AUGMENTATION (IMPORTANT)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Validation & Test (NO augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48,48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48,48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical'
)

# 🔥 EARLY STOPPING (prevents overfitting)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 🔥 TRAIN MODEL
history = model.fit(
    train_data,
    epochs=30,                 # increased epochs
    validation_data=val_data,
    callbacks=[early_stop]
)

# 🔥 FINAL TEST
loss, acc = model.evaluate(test_data)
print("✅ Test Accuracy:", acc)

# SAVE MODEL
model.save("emotion_model.h5")
print("✅ Model saved successfully!")