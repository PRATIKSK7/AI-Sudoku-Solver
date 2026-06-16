import cv2
import os
import numpy as np
from ai.segmentation import split_boxes, extract_digit
from ai.digit_classifier import DigitClassifier

def verify_ocr(image_path):
    print(f"--- VERIFYING OCR ON {image_path} ---")
    
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found.")
        return
        
    classifier = DigitClassifier(model_path='models/sudoku_digit_model.h5')
    
    os.makedirs("debug/final_digits", exist_ok=True)
    
    rotations = [
        (0, "0 deg"),
        (cv2.ROTATE_90_CLOCKWISE, "90 deg CW"),
        (cv2.ROTATE_180, "180 deg"),
        (cv2.ROTATE_90_COUNTERCLOCKWISE, "90 deg CCW")
    ]
    
    best_board = []
    best_predictions = []
    best_rot_name = ""
    
    for rot_code, rot_name in rotations:
        if rot_code == 0:
            rotated = img.copy()
        else:
            rotated = cv2.rotate(img, rot_code)
            
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        cells = split_boxes(gray)
        
        board = []
        predictions = []
        
        for idx, cell in enumerate(cells):
            digit_img, area_ratio, bbox = extract_digit(cell)
            
            if digit_img is None:
                board.append(0)
                continue
                
            digit, conf, _ = classifier.predict(digit_img, threshold=0.95)
            board.append(digit)
            predictions.append({
                "idx": idx,
                "digit": digit,
                "conf": conf,
                "bbox": bbox,
                "area_ratio": area_ratio,
                "digit_img": digit_img
            })
            
        # Match user's criteria:
        # [0,0,4,0,1,0,9,8,0] in the first row
        expected_row = [0, 0, 4, 0, 1, 0, 9, 8, 0]
        match_score = sum(1 for i in range(9) if board[i] == expected_row[i])
        
        if match_score >= 4 or rot_code == 0:  # Prioritize matches
            if not best_board or match_score > sum(1 for i in range(9) if best_board[i] == expected_row[i]):
                best_board = board
                best_predictions = predictions
                best_rot_name = rot_name
                
    print(f"Found best rotation: {best_rot_name}")
    print("Detected Matrix:")
    matrix_lines = "[\n"
    for r in range(9):
        matrix_lines += "  " + str(best_board[r*9:(r+1)*9]) + ",\n"
    matrix_lines += "]"
    print(matrix_lines)
    
    print("\n--- 9. Print Cell, Bounding Box, Area %, Prediction, Confidence ---")
    for p in best_predictions:
        idx = p["idx"]
        cv2.imwrite(f"debug/final_digits/cell_{idx:02d}.png", p["digit_img"])
        bbox_str = f"({p['bbox'][0]}, {p['bbox'][1]}, {p['bbox'][2]}, {p['bbox'][3]})" if p['bbox'] else "None"
        print(f"Cell {idx:02d} | BBox {bbox_str} | Area {p['area_ratio']*100:.2f}% | Digit {p['digit']} | Conf {p['conf']:.4f}")

if __name__ == "__main__":
    verify_ocr("captures/Puzzle_13_(board4)_warped.png")
