import numpy as np
import tensorflow as tf

# Load the model and inspect its weights or input shape
model = tf.keras.models.load_model('models/sudoku_digit_model.h5')
print("Model loaded.")
