# Dataset

This project uses the IMDB Movie Reviews dataset for binary sentiment classification.

The dataset contains movie reviews labeled as either positive or negative.

The dataset is automatically downloaded through TensorFlow Keras:

```python
from tensorflow.keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=10000
)
