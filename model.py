from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Create model
model = Sequential()

# 🧠 Convolution Block 1
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

# 🧠 Convolution Block 2
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# 🧠 Convolution Block 3
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# 🔽 Flatten
model.add(Flatten())

# 🧠 Fully Connected Layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# 🎯 OUTPUT LAYER (IMPORTANT: 7 classes)
model.add(Dense(7, activation='softmax'))

# ⚙️ Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 📊 Model Summary
model.summary()