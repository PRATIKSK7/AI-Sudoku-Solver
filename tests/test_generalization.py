import os
import sys
import cv2
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai.digit_classifier import DigitClassifier
from ai.detector import diagnose_frame
from ai.detector import diagnose_frame

@pytest.fixture(scope="module")
def classifier():
    return DigitClassifier()

def test_generalization_boards(classifier):
    boards_dir = "datasets/test_boards"
    
    # Check if directory exists and has images
    if not os.path.exists(boards_dir):
        pytest.skip("Test boards directory not found.")
        
    image_files = [f for f in os.listdir(boards_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        pytest.skip("No test images found.")
        
    failed_boards = []
    
    for img_file in image_files:
        img_path = os.path.join(boards_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Skipping {img_file}: Could not read image")
            failed_boards.append(f"{img_file}: Could not read image")
            continue
            
        print(f"Processing {img_file}...")
        try:
            result = diagnose_frame(img, classifier)
            print(f"Result for {img_file}: {result['stage']} / Success: {result['success']}")
        except Exception as e:
            print(f"Crash on {img_file}: {e}")
            failed_boards.append(f"{img_file}: Exception {e}")
            continue
        
        if not result["success"]:
            failed_boards.append(f"{img_file}: Failed at stage {result['stage']} with error: {result.get('error', 'Unknown')}")
            
    assert len(failed_boards) == 0, f"Failed boards:\n" + "\n".join(failed_boards)
