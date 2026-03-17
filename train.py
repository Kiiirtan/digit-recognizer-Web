import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Load dataset
train_data = pd.read_csv("data/train.csv")

# Separate features and labels
X = train_data.drop("label", axis=1).values / 255.0
Y = train_data["label"].values

# Reshape for CNN
X = X.reshape(-1, 28, 28, 1)

# One-hot encoding
Y = keras.utils.to_categorical(Y, 10)

# Train-validation split
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, test_size=0.1, random_state=42
)

# Build CNN model
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    X_train,
    Y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_valid, Y_valid)
)

# Save model
model.save("model/digit_model.keras")

print("Model trained and saved successfully.")