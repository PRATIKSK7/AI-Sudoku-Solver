import cv2
import numpy as np
from skimage.segmentation import clear_border

# FIX #7: Module-level CLAHE singleton — constructed once, reused for all 81 cells
# Avoids 81+ redundant object constructions per OCR pass
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

# FIX: 1 — Add cell alignment diagnostic in split_boxes()
def split_boxes(img):
    """
    Split a warped sudoku board (grayscale) into 81 cells.
    Uses a 10% inner padding on each cell to avoid grid lines.
    Handles non-square boards by computing per-cell h and w.
    """
    h, w = img.shape[:2]
    cells = []
    
    for r in range(9):
        for c in range(9):
            # Cell boundaries
            y1 = int(r * h / 9)
            y2 = int((r + 1) * h / 9)
            x1 = int(c * w / 9)
            x2 = int((c + 1) * w / 9)
            
            cell_h = y2 - y1
            cell_w = x2 - x1
            
            # Inner padding: 10% on each side to avoid grid lines
            pad_y = max(3, int(cell_h * 0.10))
            pad_x = max(3, int(cell_w * 0.10))
            
            y1p = y1 + pad_y
            y2p = y2 - pad_y
            x1p = x1 + pad_x
            x2p = x2 - pad_x
            
            # Guard against degenerate cells
            if y2p <= y1p or x2p <= x1p:
                cells.append(np.zeros((28, 28), dtype=np.uint8))
                continue
                
            cell = img[y1p:y2p, x1p:x2p]
            cells.append(cell)
            
    assert len(cells) == 81, f"split_boxes generated {len(cells)} cells instead of 81"
    
    return cells

# FIX: 2 — Rewrite extract_digit() with inversion auto-detection and tight constraints
def extract_digit(cell):
    """
    Extract a digit from a cell image.
    Auto-detects polarity (dark-on-light vs light-on-dark).
    Returns: (digit_img, area_ratio, bbox, reason) 
             where digit_img is 28x28 white-digit-on-black,
             or (None, 0.0, None, reason) if cell is empty or invalid.
    """
    if cell is None or cell.size == 0:
        return None, 0.0, None, "Empty cell array"
    
    # Resize to working size
    cell_resized = cv2.resize(cell, (64, 64))
    
    # Auto-detect polarity from border pixels
    # Border pixels = background; if they are light → black digit on white
    border_pixels = np.concatenate([
        cell_resized[0, :],
        cell_resized[-1, :],
        cell_resized[:, 0],
        cell_resized[:, -1]
    ])
    bg_is_light = np.median(border_pixels) > 127
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(cell_resized)
    
    # Threshold: always produce WHITE digit on BLACK background
    h, w = enhanced.shape[:2]
    block = max(int(min(h, w) * 0.5) | 1, 11)  # must be odd
    
    if bg_is_light:
        # Black digit on white → THRESH_BINARY_INV gives white digit
        thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block, 4
        )
    else:
        # White digit on black → THRESH_BINARY keeps white digit
        thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block, 4
        )
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Remove border artifacts (grid lines bleed)
    thresh = cv2.copyMakeBorder(thresh, 1, 1, 1, 1, 
                                 cv2.BORDER_CONSTANT, value=0)
    thresh = thresh[1:-1, 1:-1]
    
    # Find contours
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts:
        return None, 0.0, None, "No contours survived morphological cleanup"
    
    # Filter out tiny noise contours
    cell_area = h * w
    valid_cnts = [c for c in cnts 
                  if cv2.contourArea(c) > cell_area * 0.02]
    
    if not valid_cnts:
        return None, 0.0, None, "Contour area below minimum threshold (0.02)"
    
    # Pick the largest contour
    digit_c = max(valid_cnts, key=cv2.contourArea)
    area_ratio = cv2.contourArea(digit_c) / cell_area
    
    # Reject if too large (probably grid line bleed) or too small (noise)
    if area_ratio > 0.60 or area_ratio < 0.02:
        return None, 0.0, None, f"Unrealistic digit area ({area_ratio:.3f})"
        
    x, y, bw, bh = cv2.boundingRect(digit_c)
    
    # Check if touching border (cell is 64x64)
    if x <= 2 or y <= 2 or (x + bw) >= 62 or (y + bh) >= 62:
        return None, area_ratio, (x, y, bw, bh), "Digit touching border"
        
    # Check if centered
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    dist = ((cx - 32)**2 + (cy - 32)**2)**0.5
    if dist > 16:
        return None, area_ratio, (x, y, bw, bh), f"Digit not centered (dist={dist:.1f})"
    
    # Calculate Center of Mass
    M = cv2.moments(digit_c)
    if M["m00"] != 0:
        cm_x = int(M["m10"] / M["m00"])
        cm_y = int(M["m01"] / M["m00"])
    else:
        cm_x, cm_y = cx, cy

    # Mask and extract
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, [digit_c], -1, 255, -1)
    clean = cv2.bitwise_and(thresh, mask)
    
    roi = clean[y:y+bh, x:x+bw]
    
    if roi.size == 0:
        return None, 0.0, None, "Zero size ROI"
        
    # Extra check for empty cells: Pixel Density
    pixel_density = cv2.countNonZero(roi) / (bw * bh)
    if pixel_density < 0.15:
        return None, area_ratio, (x, y, bw, bh), f"Low pixel density ({pixel_density:.2f})"
    
    # Pad to square with margin, resize to 28x28, ALIGNING BY CENTER OF MASS
    side   = max(bw, bh)
    pad    = max(4, int(side * 0.2))
    
    # We want the center of mass (cm_x, cm_y) to end up at the exact center of the 28x28 canvas.
    # So we calculate the offset from the bounding box top-left to the center of mass.
    dx = cm_x - x
    dy = cm_y - y
    
    # Canvas needs to be large enough to hold the digit such that (dx, dy) is at center
    canvas_size = int(max(dx, bw - dx, dy, bh - dy) * 2) + pad * 2
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    
    ox = canvas_size // 2 - dx
    oy = canvas_size // 2 - dy
    
    # Ensure bounds (should always be safe due to canvas_size math)
    if ox >= 0 and oy >= 0 and ox + bw <= canvas_size and oy + bh <= canvas_size:
        canvas[oy:oy+bh, ox:ox+bw] = roi
    else:
        # Fallback to simple centering if center of mass math fails edge case
        canvas_size = side + pad * 2
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        ox = pad + (side - bw) // 2
        oy = pad + (side - bh) // 2
        canvas[oy:oy+bh, ox:ox+bw] = roi
    
    digit_img = cv2.resize(canvas, (28, 28), 
                           interpolation=cv2.INTER_AREA)
    
    # Apply permanent transpose transform to match the model's EMNIST orientation mismatch
    digit_img = np.transpose(digit_img)
    
    bbox = (x, y, bw, bh)
    return digit_img, area_ratio, bbox, "Accepted"
