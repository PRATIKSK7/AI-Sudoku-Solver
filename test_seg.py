import cv2
import numpy as np
from skimage.segmentation import clear_border

def test_segmentation(img_path):
    cell_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    border_pixels = np.concatenate([cell_img[0, :], cell_img[-1, :],
                                    cell_img[:, 0], cell_img[:, -1]])
    if np.median(border_pixels) < 128:
        cell_img = cv2.bitwise_not(cell_img)

    h, w = cell_img.shape[:2]
    block_size = max(int(min(h, w) * 0.4), 11)
    if block_size % 2 == 0:
        block_size += 1

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    cell_img = clahe.apply(cell_img)

    thresh = cv2.adaptiveThreshold(
        cell_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 5
    )

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

    # NEW: Clear border to remove grid lines
    cleared = clear_border(thresh)

    cnts, _ = cv2.findContours(cleared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cnts) == 0:
        print(f"{img_path}: No contours")
        return
        
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    digit_c = cnts[0] # just take the largest for testing
    
    # NEW: Masking to remove noise inside bounding box
    mask = np.zeros_like(cleared)
    cv2.drawContours(mask, [digit_c], -1, 255, -1)
    clean_digit = cv2.bitwise_and(cleared, mask)

    x, y, w_bbox, h_bbox = cv2.boundingRect(digit_c)
    roi = clean_digit[y:y + h_bbox, x:x + w_bbox]

    print(f'\n--- {img_path} ---')
    # Pad to 28x28 roughly just to print
    roi_resized = cv2.resize(roi, (28, 28))
    for row in roi_resized:
        print(''.join(['#' if p > 128 else '.' for p in row]))

test_segmentation('debug/cell_02.png')
test_segmentation('debug/cell_04.png')
test_segmentation('debug/cell_06.png')
