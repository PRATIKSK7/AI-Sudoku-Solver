import sys
import os
import cv2
import numpy as np

# Create a dummy white digit on black background
dummy_digit = np.zeros((28, 28), dtype=np.uint8)
cv2.putText(dummy_digit, "4", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 2)

from ai.digit_classifier import DigitClassifier

classifier = DigitClassifier()
print("Model loaded?", classifier.model is not None)

digit, conf, softmax = classifier.predict_with_softmax(dummy_digit)
print(f"Prediction: {digit}, Confidence: {conf:.2f}")
print(f"Softmax: {softmax}")
