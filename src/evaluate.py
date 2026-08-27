# ==========================================
# Movie Review Sentiment Analysis
# Model Evaluation
# ==========================================

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==========================================
# Configuration
# ==========================================

VOCAB_SIZE = 10000
MAX_LENGTH = 200

MODEL_PATH = "models/movie_review_lstm_model.keras"


# ==========================================
# Load Dataset
# ==========================================

print("Loading IMDB test dataset...")

(_, _), (X_test, y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)


# ==========================================
# Padding
# ==========================================

X_test_padded = pad_sequences(
    X_test,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)


# ==========================================
# Load Model
# ==========================================

print("Loading trained model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# Evaluate
# ==========================================

test_loss, test_accuracy = model.evaluate(
    X_test_padded,
    y_test,
    verbose=1
)

print("\n==========================================")
print("Model Evaluation")
print("==========================================")

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ==========================================
# Generate Predictions
# ==========================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_test_padded,
    verbose=1
).flatten()

predicted_labels = (
    predictions >= 0.5
).astype(int)


# ==========================================
# Correct and Wrong Predictions
# ==========================================

correct_predictions = np.sum(
    predicted_labels == y_test
)

wrong_predictions = np.sum(
    predicted_labels != y_test
)

total_predictions = len(y_test)


print("\n==========================================")
print("Prediction Analysis")
print("==========================================")

print("Total Test Reviews:", total_predictions)
print("Correct Predictions:", correct_predictions)
print("Wrong Predictions:", wrong_predictions)


# ==========================================
# Accuracy Calculation
# ==========================================

calculated_accuracy = (
    correct_predictions / total_predictions
)

print(
    f"Calculated Accuracy: "
    f"{calculated_accuracy * 100:.2f}%"
)


# ==========================================
# Show Some Wrong Predictions
# ==========================================

wrong_indices = np.where(
    predicted_labels != y_test
)[0]

print("\n==========================================")
print("Sample Wrong Predictions")
print("==========================================")

for index in wrong_indices[:5]:

    actual = (
        "Positive"
        if y_test[index] == 1
        else "Negative"
    )

    predicted = (
        "Positive"
        if predicted_labels[index] == 1
        else "Negative"
    )

    probability = predictions[index]

    print("\nReview Index:", index)
    print("Actual:", actual)
    print("Predicted:", predicted)
    print(
        "Positive Probability:",
        f"{probability:.4f}"
    )
