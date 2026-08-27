# ==========================================
# Movie Review Sentiment Analysis using LSTM
# Training Script
# ==========================================

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==========================================
# Configuration
# ==========================================

VOCAB_SIZE = 10000
MAX_LENGTH = 200
EMBEDDING_DIM = 128
LSTM_UNITS = 64
EPOCHS = 5
BATCH_SIZE = 64

MODEL_DIR = "models"
IMAGE_DIR = "images"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


# ==========================================
# Load Dataset
# ==========================================

print("Loading IMDB dataset...")

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)

print("Dataset loaded successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# Padding
# ==========================================

print("\nPadding sequences...")

X_train_padded = pad_sequences(
    X_train,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_test_padded = pad_sequences(
    X_test,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

print("Training shape:", X_train_padded.shape)
print("Testing shape:", X_test_padded.shape)


# ==========================================
# Build LSTM Model
# ==========================================

print("\nBuilding LSTM model...")

model = Sequential([
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_LENGTH
    ),

    LSTM(LSTM_UNITS),

    Dense(1, activation="sigmoid")
])


# ==========================================
# Compile Model
# ==========================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==========================================
# Train Model
# ==========================================

print("\nTraining model...")

history = model.fit(
    X_train_padded,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2
)


# ==========================================
# Evaluate Model
# ==========================================

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    X_test_padded,
    y_test,
    verbose=1
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ==========================================
# Save Model
# ==========================================

model_path = os.path.join(
    MODEL_DIR,
    "movie_review_lstm_model.keras"
)

model.save(model_path)

print("\nModel saved to:", model_path)


# ==========================================
# Accuracy Graph
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("LSTM Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

accuracy_path = os.path.join(
    IMAGE_DIR,
    "accuracy_plot.png"
)

plt.savefig(accuracy_path, dpi=300, bbox_inches="tight")
plt.show()

print("Accuracy graph saved to:", accuracy_path)


# ==========================================
# Loss Graph
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("LSTM Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

loss_path = os.path.join(
    IMAGE_DIR,
    "loss_plot.png"
)

plt.savefig(loss_path, dpi=300, bbox_inches="tight")
plt.show()

print("Loss graph saved to:", loss_path)


print("\n==========================================")
print("Training Completed Successfully!")
print("==========================================")
