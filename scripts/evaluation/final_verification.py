import cv2
import numpy as np
import json
from ai.detector import diagnose_frame
from ai.digit_classifier import DigitClassifier
import os

def create_empty_board(base_img):
    img = base_img.copy()
    # Find board and blank out the inside
    from ai.perspective import find_board
    cnt, _, _ = find_board(img)
    if cnt is not None:
        cv2.drawContours(img, [cnt], -1, (255, 255, 255), -1)
        # re-draw a black border so find_board still works
        cv2.drawContours(img, [cnt], -1, (0, 0, 0), 10)
    return img

def create_single_digit_board(base_img):
    img = create_empty_board(base_img)
    from ai.perspective import find_board
    cnt, _, _ = find_board(img)
    if cnt is not None:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(img, "7", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
    return img

def test_pipeline(img, name, clf):
    print(f"\n==============================================")
    print(f"TEST: {name}")
    print(f"==============================================")
    
    result = diagnose_frame(img, clf)
    
    cells = result.get("cell_details", [])
    
    cnn_calls = 0
    empty_cells = 0
    accepted_predictions = 0
    confidences = []
    
    if "error" in result and result["error"] == "All 81 cells evaluated to 0.":
        # Hard fail OCR
        empty_cells = 81
        cnn_calls = 0
        accepted_predictions = 0
    elif cells:
        for c in cells:
            if c["is_empty"]:
                empty_cells += 1
            else:
                cnn_calls += 1
                if c["predicted"] != 0:
                    accepted_predictions += 1
                    confidences.append(c["confidence"])
                    
    avg_conf = sum(confidences)/len(confidences) if confidences else 0.0
    
    print(f"1. Empty Board Logic Tested: {'YES' if name == 'Empty Board' else 'N/A'}")
    print(f"2. Single Digit Logic Tested: {'YES' if name == 'Single Digit Board' else 'N/A'}")
    print(f"3. OCR Matrix:")
    board_data = result.get("board")
    if board_data is None:
        board_data = [0]*81
    matrix = np.array(board_data).reshape(9,9)
    print(matrix)
    print(f"4. Number of CNN calls: {cnn_calls}")
    print(f"5. Number of cells classified as empty (< 5% area): {empty_cells}")
    print(f"6. Number of accepted predictions (> 0.85 conf): {accepted_predictions}")
    print(f"7. Average confidence score: {avg_conf:.4f}")
    
    if result.get("success"):
        print("8. Board Successfully Solved!")
        cv2.imwrite(f"captures/{name.replace(' ', '_')}_solved.png", result["output_frame"])
    else:
        print(f"8. Solve blocked: {result.get('error', 'Validation Failed')}")
        
    # Save the warped board and cell images for debugging
    if result.get("warped") is not None:
        cv2.imwrite(f"captures/{name.replace(' ', '_')}_warped.png", result["warped"])
        
    import os
    os.makedirs(f"captures/{name.replace(' ', '_')}_cells", exist_ok=True)
    for d in result.get("cell_details", []):
        if not d["is_empty"] and d.get("digit_img") is not None:
            cv2.imwrite(f"captures/{name.replace(' ', '_')}_cells/cell_{d['index']}_pred_{d.get('predicted', 0)}.png", d["digit_img"])

if __name__ == "__main__":
    clf = DigitClassifier(model_path='models/sudoku_digit_model.h5')
    
    board4 = cv2.imread("datasets/test_boards/board4.jpg")
    if board4 is not None:
        test_pipeline(board4, "Puzzle 13 (board4)", clf)
    else:
        print("Could not load base image datasets/test_boards/board4.jpg")

    board5 = cv2.imread("datasets/test_boards/board5.jpg")
    if board5 is not None:
        test_pipeline(board5, "Another Sudoku (board5)", clf)
    else:
        print("Could not load base image datasets/test_boards/board5.jpg")
