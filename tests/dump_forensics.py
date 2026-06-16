import cv2
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ai.detector import find_board, extract_board
from ai.segmentation import split_boxes, extract_digit
from ai.digit_classifier import DigitClassifier

os.makedirs('debug/raw_cells', exist_ok=True)
os.makedirs('debug/threshold_cells', exist_ok=True)
os.makedirs('debug/segmented_digits', exist_ok=True)
os.makedirs('debug/cnn_inputs', exist_ok=True)

img = cv2.imread('datasets/test_boards/board4.jpg')
if img is None:
    print("Could not find board4.jpg")
    sys.exit(1)

board_cnt, _, _ = find_board(img)
warped, M, max_dim = extract_board(img, board_cnt)
gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
cells = split_boxes(gray_warped)

clf = DigitClassifier()
print("Starting extraction...")

for idx, cell in enumerate(cells):
    result = extract_digit(cell, cell_idx=idx)
    if result[0] is not None:
        digit, conf, _ = clf.predict(result[0])
        print(f"Cell {idx:02d}: Extracted successfully. Pred: {digit}, Conf: {conf:.2f}")
    else:
        print(f"Cell {idx:02d}: EMPTY or failed extraction.")

print("Done. Check debug directories.")
