import cv2
import numpy as np
from ai.detector import process_frame
from ai.digit_classifier import DigitClassifier

# Load a dummy sudoku image to test full integration
img = np.zeros((500, 500, 3), dtype=np.uint8)
# We can't really test the full computer vision without a real Sudoku image.
# But we can test if the model loads.
classifier = DigitClassifier()
if classifier.model is None:
    print("FATAL: Model is None")
else:
    print("Model loaded perfectly.")
