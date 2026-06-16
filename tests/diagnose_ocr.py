"""
COMPREHENSIVE OCR DIAGNOSTIC — Phases 1-6
Runs entirely headless. No Streamlit. No webcam. No retrain.
"""
import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from skimage.segmentation import clear_border

SEP = "=" * 60

# ─────────────────────────────────────────────
# PHASE 0 — MODEL LOAD
# ─────────────────────────────────────────────
print(f"\n{SEP}")
print("PHASE 0 — MODEL LOAD")
print(SEP)

from ai.digit_classifier import DigitClassifier
classifier = DigitClassifier(model_path='models/sudoku_digit_model.h5')
if classifier.model is None:
    print("FATAL: model is None. Stopping.")
    sys.exit(1)
print(f"  input_shape  = {classifier.model.input_shape}")
print(f"  output_shape = {classifier.model.output_shape}")

# ─────────────────────────────────────────────
# PHASE 1 — SYNTHETIC DIGIT DIRECT TO MODEL
# ─────────────────────────────────────────────
print(f"\n{SEP}")
print("PHASE 1 — DIRECT MODEL TEST (synthetic white-on-black 28x28)")
print(SEP)

for digit_char in ["4", "1", "9", "7"]:
    canvas = np.zeros((28, 28), dtype=np.uint8)
    cv2.putText(canvas, digit_char, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    
    img_f = canvas.astype(np.float32) / 255.0
    tensor = img_f.reshape(1, 28, 28, 1)
    
    preds = classifier.model.predict(tensor, verbose=0)
    cls = int(np.argmax(preds, axis=1)[0])
    conf = float(preds[0][cls])
    
    nonzero = cv2.countNonZero(canvas)
    print(f"  Digit '{digit_char}': nonzero={nonzero:3d}  predicted={cls}  conf={conf:.4f}  "
          f"min={img_f.min():.2f}  max={img_f.max():.2f}")

# ─────────────────────────────────────────────
# PHASE 2 — SYNTHETIC CELL → extract_digit → MODEL
# ─────────────────────────────────────────────
print(f"\n{SEP}")
print("PHASE 2 — SYNTHETIC CELL through extract_digit()")
print(SEP)

from ai.segmentation import extract_digit

for digit_char in ["4", "1", "9", "7"]:
    # Simulate a typical Sudoku cell: white background, black digit (like printed paper)
    cell = np.ones((60, 60), dtype=np.uint8) * 240
    cv2.putText(cell, digit_char, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 20, 3)
    
    border_px = np.concatenate([cell[0,:], cell[-1,:], cell[:,0], cell[:,-1]])
    median_border = np.median(border_px)
    
    result = extract_digit(cell)
    
    if result is None:
        print(f"  Digit '{digit_char}': extract_digit → None  (border_median={median_border:.0f})")
        
        # Manual diagnostic: reproduce each step
        if median_border < 128:
            cell_work = cv2.bitwise_not(cell)
            print(f"    → Border < 128, so inverted the cell")
        else:
            cell_work = cell.copy()
            print(f"    → Border >= 128, cell kept as-is")
        
        thresh = cv2.adaptiveThreshold(cell_work, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        nz_before_clear = cv2.countNonZero(thresh)
        
        thresh_cleared = clear_border(thresh)
        nz_after_clear = cv2.countNonZero(thresh_cleared)
        
        cnts, _ = cv2.findContours(thresh_cleared.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"    → thresh nonzero BEFORE clear_border: {nz_before_clear}")
        print(f"    → thresh nonzero AFTER  clear_border: {nz_after_clear}")
        print(f"    → contours found after clear_border:  {len(cnts)}")
        
        if len(cnts) > 0:
            areas = [cv2.contourArea(c) for c in cnts]
            bboxes = [cv2.boundingRect(c) for c in cnts]
            h_thresh, w_thresh = thresh_cleared.shape
            print(f"    → thresh shape: {thresh_cleared.shape}")
            for i, (a, bb) in enumerate(zip(areas, bboxes)):
                x,y,wb,hb = bb
                passes = hb >= (h_thresh * 0.15) and wb >= 2
                print(f"    → contour {i}: area={a:.0f}  bbox={bb}  "
                      f"h_check={hb}>={h_thresh*0.15:.1f}? {passes}")
    else:
        # Got a digit image — run model
        digit, conf, softmax = classifier.predict_with_softmax(result)
        nz = cv2.countNonZero(result)
        print(f"  Digit '{digit_char}': extract_digit → shape={result.shape}  nonzero={nz}  "
              f"predicted={digit}  conf={conf:.4f}  (border_median={median_border:.0f})")

# ─────────────────────────────────────────────
# PHASE 3 — INVERSION TEST
# ─────────────────────────────────────────────
print(f"\n{SEP}")
print("PHASE 3 — INVERSION TEST (normal vs inverted to model)")
print(SEP)

for digit_char in ["4", "7"]:
    cell = np.ones((60, 60), dtype=np.uint8) * 240
    cv2.putText(cell, digit_char, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 20, 3)
    
    # Normal path (through extract_digit)
    result_normal = extract_digit(cell)
    
    # Force-invert the cell before extract_digit (pretend dark background)
    cell_inv = cv2.bitwise_not(cell)
    result_inv = extract_digit(cell_inv)
    
    def test_result(label, r):
        if r is None:
            print(f"    {label}: extract_digit → None")
            return
        digit, conf, _ = classifier.predict_with_softmax(r)
        
        # Also test with inverted image directly
        r_flip = cv2.bitwise_not(r)
        d2, c2, _ = classifier.predict_with_softmax(r_flip)
        
        print(f"    {label}: pred={digit} conf={conf:.4f}  |  inverted_pred={d2} inverted_conf={c2:.4f}")
    
    print(f"  Digit '{digit_char}':")
    test_result("normal_cell ", result_normal)
    test_result("inverted_cell", result_inv)

# ─────────────────────────────────────────────
# PHASE 4 — MODEL EXPECTATIONS: WHAT DOES THE CNN ACTUALLY WANT?
# ─────────────────────────────────────────────
print(f"\n{SEP}")
print("PHASE 4 — CNN EXPECTATIONS: testing all 10 synthetic digits")
print(SEP)

for d in range(10):
    canvas = np.zeros((28, 28), dtype=np.uint8)
    if d > 0:
        cv2.putText(canvas, str(d), (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    
    img_f = canvas.astype(np.float32) / 255.0
    tensor = img_f.reshape(1, 28, 28, 1)
    preds = classifier.model.predict(tensor, verbose=0)
    cls = int(np.argmax(preds, axis=1)[0])
    conf = float(preds[0][cls])
    top3 = sorted(enumerate(preds[0]), key=lambda x: -x[1])[:3]
    top3_str = "  ".join([f"{i}:{v:.3f}" for i, v in top3])
    print(f"  digit={d}  predicted={cls}  conf={conf:.4f}  top3=[{top3_str}]")

# ─────────────────────────────────────────────
# PHASE 5 — CLEAR_BORDER DESTRUCTION TEST
# ─────────────────────────────────────────────
print(f"\n{SEP}")
print("PHASE 5 — CLEAR_BORDER DESTRUCTION TEST")
print(SEP)
print("  Testing if clear_border destroys digits that touch the cell edge...")

for digit_char in ["1", "4", "7", "9"]:
    cell = np.ones((50, 50), dtype=np.uint8) * 230
    # Draw digit closer to edges to simulate real extraction
    cv2.putText(cell, digit_char, (5, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 10, 3)
    
    # Process like extract_digit does
    border_px = np.concatenate([cell[0,:], cell[-1,:], cell[:,0], cell[:,-1]])
    if np.median(border_px) < 128:
        cell = cv2.bitwise_not(cell)
    
    thresh = cv2.adaptiveThreshold(cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    nz_before = cv2.countNonZero(thresh)
    
    cleared = clear_border(thresh)
    nz_after = cv2.countNonZero(cleared)
    
    destroyed_pct = ((nz_before - nz_after) / max(nz_before, 1)) * 100
    
    cnts, _ = cv2.findContours(cleared.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"  '{digit_char}': before={nz_before:4d}  after={nz_after:4d}  "
          f"destroyed={destroyed_pct:.1f}%  contours_left={len(cnts)}")

# ─────────────────────────────────────────────
# PHASE 6 — MARGIN CROP TEST  
# ─────────────────────────────────────────────
print(f"\n{SEP}")
print("PHASE 6 — MARGIN CROP TEST (10% crop from split_boxes)")
print(SEP)
print("  Checking if 10% margin crop removes digit pixels...")

for digit_char in ["1", "4"]:
    # Simulate a grid cell at typical warped resolution (e.g. 450/9 = 50px)
    cell_full = np.ones((50, 50), dtype=np.uint8) * 230
    cv2.putText(cell_full, digit_char, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 10, 2)
    
    h, w = cell_full.shape
    margin_y = int(h * 0.10)
    margin_x = int(w * 0.10)
    cropped = cell_full[margin_y:h-margin_y, margin_x:w-margin_x]
    
    nz_full = cv2.countNonZero(cv2.bitwise_not(cell_full))
    nz_crop = cv2.countNonZero(cv2.bitwise_not(cropped))
    
    print(f"  '{digit_char}': full_size={cell_full.shape}  cropped_size={cropped.shape}  "
          f"dark_px_full={nz_full}  dark_px_cropped={nz_crop}")
    
    # Now run extract_digit on the cropped version
    result = extract_digit(cropped)
    if result is None:
        print(f"         → extract_digit(cropped) = None  ← DIGIT LOST!")
    else:
        d, c, _ = classifier.predict_with_softmax(result)
        print(f"         → extract_digit(cropped) = shape {result.shape}  pred={d}  conf={c:.4f}")

print(f"\n{SEP}")
print("DIAGNOSIS COMPLETE")
print(SEP)
