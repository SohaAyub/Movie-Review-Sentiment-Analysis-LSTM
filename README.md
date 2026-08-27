# Movie Review Sentiment Analysis Using LSTM

## Overview

This project implements a Movie Review Sentiment Analysis system using Natural Language Processing (NLP) and a Long Short-Term Memory (LSTM) neural network.

The model is trained on the IMDB Movie Reviews dataset to classify movie reviews into two sentiment categories: Positive and Negative.

The project demonstrates the complete workflow of an NLP-based Deep Learning application, including dataset loading, text encoding, sequence padding, word embeddings, LSTM model development, training, evaluation, custom predictions, and error analysis.

---

## Project Objectives

The main objectives of this project are:

- Understand the fundamentals of Natural Language Processing.
- Work with the IMDB movie review dataset.
- Convert text data into numerical sequences.
- Apply sequence padding for uniform input length.
- Build an LSTM-based Deep Learning model.
- Train the model for binary sentiment classification.
- Evaluate model performance using test data.
- Visualize training and validation performance.
- Test the model with custom movie reviews.
- Analyze incorrectly classified reviews.
- Save the trained model for future use.

---

## Dataset

This project uses the IMDB Movie Reviews dataset available through TensorFlow Keras.

The dataset contains 50,000 movie reviews divided into training and testing sets.

| Property | Value |
|---|---|
| Dataset | IMDB Movie Reviews |
| Training Samples | 25,000 |
| Testing Samples | 25,000 |
| Number of Classes | 2 |
| Classes | Positive, Negative |
| Vocabulary Used | 10,000 words |
| Maximum Sequence Length | 200 |

The dataset is automatically downloaded using:

```python
from tensorflow.keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=10000
)
