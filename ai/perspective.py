import cv2
import numpy as np
import imutils

# FIX: Improve corner ordering in find_board()
def order_points_robust(pts):
    """
    Robust ordering method:
      - top-left:     point with smallest (x + y)
      - top-right:    point with largest  (x - y)
      - bottom-right: point with largest  (x + y)
      - bottom-left:  point with smallest (x - y)
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff_x_minus_y = pts[:, 0] - pts[:, 1]
    
    rect[0] = pts[np.argmin(s)]             # top-left
    rect[2] = pts[np.argmax(s)]             # bottom-right
    rect[1] = pts[np.argmax(diff_x_minus_y)] # top-right
    rect[3] = pts[np.argmin(diff_x_minus_y)] # bottom-left
    return rect

def find_board(img):
    """
    Finds the Sudoku board using a multi-strategy pipeline.
    Returns (contour, thresh_img, gray_img)
    """
    img_area = img.shape[0] * img.shape[1]
    best_contour = None
    best_area = 0
    best_thresh = None
    best_gray = None
    
    def process_contours(cnts, thresh, gray, strategy_name):
        nonlocal best_contour, best_area, best_thresh, best_gray
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            area = cv2.contourArea(c)
            # Filter: area > 5% and < 95% of image area
            if area < 0.05 * img_area or area > 0.95 * img_area:
                continue
            
            peri = cv2.arcLength(c, True)
            # Try epsilon values: [0.01, 0.02, 0.03, 0.04, 0.05] * perimeter
            for eps_mult in [0.01, 0.02, 0.03, 0.04, 0.05]:
                approx = cv2.approxPolyDP(c, eps_mult * peri, True)
                if len(approx) == 4:
                    print(f"[find_board] Strategy {strategy_name} SUCCESS — contour area: {area}px")
                    if area > best_area:
                        best_area = area
                        best_contour = approx
                        best_thresh = thresh
                        best_gray = gray
                    return True
        return False

    # FIX: STRATEGY 1 — Adaptive Threshold + Dilate
    print("[find_board] Trying Strategy 1: Adaptive Threshold")
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    blur1 = cv2.GaussianBlur(gray1, (9, 9), 0)
    thresh1 = cv2.adaptiveThreshold(blur1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh1 = cv2.dilate(thresh1, kernel1, iterations=1)
    cnts1 = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = imutils.grab_contours(cnts1)
    process_contours(cnts1, thresh1, gray1, "1")

    # FIX: STRATEGY 2 — Otsu Threshold
    print("[find_board] Trying Strategy 2: Otsu Threshold")
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    blur2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    _, thresh2 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts2 = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)
    process_contours(cnts2, thresh2, gray2, "2")

    # FIX: STRATEGY 3 — Canny Edge Detection
    print("[find_board] Trying Strategy 3: Canny Edge Detection")
    gray3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    blur3 = cv2.GaussianBlur(gray3, (5, 5), 0)
    median = np.median(blur3)
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edged = cv2.Canny(blur3, lower, upper)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh3 = cv2.dilate(edged, kernel3, iterations=2)
    cnts3 = cv2.findContours(thresh3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts3 = imutils.grab_contours(cnts3)
    process_contours(cnts3, thresh3, gray3, "3")

    # FIX: STRATEGY 4 — CLAHE + Adaptive
    print("[find_board] Trying Strategy 4: CLAHE + Adaptive")
    gray4 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray4_clahe = clahe.apply(gray4)
    thresh4 = cv2.adaptiveThreshold(gray4_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    cnts4 = cv2.findContours(thresh4.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts4 = imutils.grab_contours(cnts4)
    process_contours(cnts4, thresh4, gray4, "4")

    if best_contour is None:
        print("[find_board] All strategies failed.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return None, gray, gray
        
    return best_contour, best_thresh, best_gray

def extract_board(img, contour):
    # FIX: Use the robust corner ordering internally
    pts = contour.reshape(4, 2)
    rect = order_points_robust(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    calculated_width = max(int(widthA), int(widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    calculated_height = max(int(heightA), int(heightB))
    
    # FIX: Ensure the warp output size is always square and at least 450x450 pixels
    max_dim = max(calculated_width, calculated_height, 450)
    max_dim = int(((max_dim + 8) // 9) * 9) # keep it divisible by 9 for grid cleanly
    
    dst = np.array([
        [0, 0],
        [max_dim - 1, 0],
        [max_dim - 1, max_dim - 1],
        [0, max_dim - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (max_dim, max_dim))
    
    return warped, M, max_dim

def reverse_extract(original_img, warped_overlay, M):
    M_inv = np.linalg.inv(M)
    h, w = original_img.shape[:2]
    
    reverse_warped = cv2.warpPerspective(warped_overlay, M_inv, (w, h))
    gray_reverse = cv2.cvtColor(reverse_warped, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_reverse, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(original_img, original_img, mask=mask_inv)
    final_img = cv2.add(img1_bg, reverse_warped)
    return final_img
