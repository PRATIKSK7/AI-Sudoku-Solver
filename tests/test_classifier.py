import pytest
import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai.digit_classifier import DigitClassifier

def test_digit_classifier_empty_cell():
    # Mock model
    classifier = DigitClassifier(model_path='non_existent_model.h5')
    
    # Create a completely black 28x28 image (empty cell)
    empty_cell = np.zeros((28, 28), dtype=np.uint8)
    
    digit, conf = classifier.predict(empty_cell)
    assert digit == 0
    assert conf == 0.0 # Forced confidence for non-existent model
