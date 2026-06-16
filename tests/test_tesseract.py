import cv2
import pytesseract
import os

for i in range(81):
    img = cv2.imread(f"debug/threshold_cells/cell_{i:02d}.png", 0)
    if img is not None:
        # Tesseract expects black text on white background
        img = cv2.bitwise_not(img)
        text = pytesseract.image_to_string(img, config='--psm 10 -c tessedit_char_whitelist=123456789')
        text = text.strip()
        if text:
            print(f"Cell {i:02d}: {text}")
