import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai.detector import process_frame
from ai.digit_classifier import DigitClassifier

def test_process_frame_no_board():
    classifier = DigitClassifier(model_path='non_existent_model.h5')
    
    # Random noise image (no board)
    frame = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    
    final_frame, success, solved_board = process_frame(frame, classifier)
    
    assert success == False
    assert solved_board == None
    # Ensure it returns the original frame
    np.testing.assert_array_equal(final_frame, frame)
