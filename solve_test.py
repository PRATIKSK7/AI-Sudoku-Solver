import cv2
import numpy as np
from ai.solver import solve_sudoku

# We need the full board. Is there a full board in debug/08_ocr_matrix.txt?
import os
if os.path.exists("debug/08_ocr_matrix.txt"):
    with open("debug/08_ocr_matrix.txt", "r") as f:
        board_str = f.read()
    print("Found board:", board_str)
else:
    print("No board found.")
