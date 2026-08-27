# ==========================================
# Movie Review Sentiment Prediction
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
# Load Model
# ==========================================

print("Loading trained model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# Load Word Index
# ==========================================

word_index = imdb.get_word_index()


# ==========================================
# Encode Review
# ==========================================

def encode_review(review):

    words = review.lower().split()

    encoded = []

    for word in words:

        # Remove simple punctuation
        word = word.strip(".,!?;:\"'()[]{}")

        if word in word_index:

            # IMDB reserves indexes 0-3
            encoded.append(word_index[word] + 3)

        else:

            # Unknown word
            encoded.append(2)

    # Start-of-sequence token
    encoded = [1] + encoded

    # Padding
    padded = pad_sequences(
        [encoded],
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    return padded


# ==========================================
# Predict Sentiment
# ==========================================

def predict_sentiment(review):

    encoded_review = encode_review(review)

    probability = model.predict(
        encoded_review,
        verbose=0
    )[0][0]

    if probability >= 0.5:

        sentiment = "Positive"

        confidence = probability

    else:

        sentiment = "Negative"

        confidence = 1 - probability

    print("\nReview:")
    print(review)

    print("\nPrediction:")
    print(sentiment)

    print("\nPositive Probability:")
    print(f"{probability:.4f}")

    print("\nConfidence:")
    print(f"{confidence * 100:.2f}%")

    return sentiment


# ==========================================
# Example Reviews
# ==========================================

if __name__ == "__main__":

    review1 = (
        "This movie was absolutely amazing "
        "and I loved every minute of it."
    )

    review2 = (
        "This movie was boring, terrible "
        "and a complete waste of time."
    )

    predict_sentiment(review1)

    print("\n" + "-" * 60)

    predict_sentiment(review2)
