import cv2
import os
import numpy as np
from ai.segmentation import split_boxes, extract_digit
from ai.digit_classifier import DigitClassifier

def run_report(image_path, threshold=0.95):
    print(f"--- OCR REPORT FOR {image_path} ---")
    print(f"Using threshold: {threshold}\n")
    
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found.")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cells = split_boxes(gray)
    
    classifier = DigitClassifier(model_path='models/sudoku_digit_model.h5')
    
    os.makedirs("debug", exist_ok=True)
    
    accepted = []
    rejected = []
    
    for idx, cell in enumerate(cells):
        # Save raw cell
        cv2.imwrite(f"debug/cell_{idx:02d}.png", cell)
        
        digit_img, area_ratio = extract_digit(cell)
        
        if digit_img is None:
            # Rejection type 1: No contour found
            rejected.append((idx, 0.0, "No valid digit contour found (empty cell or noise)"))
            continue
            
        # Save processed digit
        cv2.imwrite(f"debug/cell_{idx:02d}_thresh.png", digit_img)
        
        digit, conf, top3 = classifier.predict(digit_img, threshold=threshold)
        
        if digit == 0:
            # Rejection type 2: Low confidence
            reason = f"Low confidence. Top prediction was {top3[0][0]} at {top3[0][1]:.4f}" if top3 else "No prediction"
            rejected.append((idx, conf, reason))
        else:
            accepted.append((idx, digit, conf))
            
    print("=== ACCEPTED DIGITS ===")
    for idx, digit, conf in accepted:
        print(f"Cell {idx:02d}: Predicted = {digit} | Confidence = {conf:.4f}")
        
    print("\n=== REJECTED DIGITS ===")
    for idx, conf, reason in rejected:
        print(f"Cell {idx:02d}: Confidence = {conf:.4f} | Reason = {reason}")
        
    print("\n=== EXPECTED VS ACTUAL (Manual Check Required) ===")
    print("Please review the debug/ folder to identify issues with segmentation vs classification.")

if __name__ == "__main__":
    run_report("captures/Puzzle_13_(board4)_warped.png")
