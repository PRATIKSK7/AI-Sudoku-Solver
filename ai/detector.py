import cv2
import numpy as np
import time

from ai.perspective import find_board, extract_board, reverse_extract
from ai.segmentation import split_boxes, extract_digit
from ai.solver import get_board_solution
from ai.utils.utils import display_numbers_on_board

# FIX: 3 — Add draw_solution_grid() as module-level function
def draw_solution_grid(original_board, solved_board, cell_size=64):
    """
    Renders a clean Sudoku grid image:
    - White background
    - Thin gray lines for cells, thick black lines for 3x3 boxes
    - Black digits for original clues
    - Blue digits (BGR: 200, 80, 0 → renders as blue in RGB) for solved cells
    """
    import numpy as np
    import cv2
    
    padding   = 6
    grid_px   = cell_size * 9 + padding * 2
    img       = np.ones((grid_px, grid_px, 3), dtype=np.uint8) * 255
    font      = cv2.FONT_HERSHEY_SIMPLEX
    
    for idx in range(81):
        r, c      = divmod(idx, 9)
        x         = padding + c * cell_size
        y         = padding + r * cell_size
        orig_val  = original_board[idx] if idx < len(original_board) else 0
        solved_val = solved_board[idx]  if idx < len(solved_board)  else 0
        
        if solved_val == 0:
            continue
        
        is_clue    = (orig_val != 0)
        color      = (0, 0, 0) if is_clue else (180, 50, 0)
        # (180,50,0) in BGR = strong blue in RGB display
        thickness  = 2 if is_clue else 1
        font_scale = cell_size / 78.0
        
        text               = str(solved_val)
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        tx = x + (cell_size - tw) // 2
        ty = y + (cell_size + th) // 2 - 2
        cv2.putText(img, text, (tx, ty), font, font_scale, 
                    color, thickness, cv2.LINE_AA)
    
    # Thin lines — all 10 grid lines in each direction
    for i in range(10):
        pos = padding + i * cell_size
        cv2.line(img, (pos, padding), (pos, grid_px - padding), 
                 (190, 190, 190), 1)
        cv2.line(img, (padding, pos), (grid_px - padding, pos), 
                 (190, 190, 190), 1)
    
    # Thick lines — 3x3 box borders
    for i in range(0, 10, 3):
        pos = padding + i * cell_size
        cv2.line(img, (pos, padding), (pos, grid_px - padding), 
                 (0, 0, 0), 3)
        cv2.line(img, (padding, pos), (grid_px - padding, pos), 
                 (0, 0, 0), 3)
    
    # Outer border — slightly thicker
    cv2.rectangle(img, (padding, padding), 
                  (grid_px - padding, grid_px - padding), (0, 0, 0), 4)
    
    return img


def validate_board_size(board):
    """
    Ensure the board has exactly 81 cells.
    Raises ValueError if corrupted to prevent downstream IndexError.
    """
    if len(board) != 81:
        raise ValueError(f"Corrupted board size: expected 81 cells, found {len(board)}")
    return True

def validate_board(board):
    """
    Validate a 81-length board for Sudoku rule violations.
    Returns a list of error strings and a set of conflicting cell indices.
    """
    errors = []
    conflicts = set()

    grid = [board[i*9:(i+1)*9] for i in range(9)]
    # Check rows
    for r in range(9):
        seen = {}
        for c in range(9):
            v = grid[r][c]
            if v != 0:
                if v in seen:
                    errors.append(f"Row {r+1}: duplicate {v} at col {seen[v]+1} and {c+1}")
                    conflicts.add(r * 9 + c)
                    conflicts.add(r * 9 + seen[v])
                else:
                    seen[v] = c

    # Check columns
    for c in range(9):
        seen = {}
        for r in range(9):
            v = grid[r][c]
            if v != 0:
                if v in seen:
                    errors.append(f"Col {c+1}: duplicate {v} at row {seen[v]+1} and {r+1}")
                    conflicts.add(r * 9 + c)
                    conflicts.add(seen[v] * 9 + c)
                else:
                    seen[v] = r

    # Check 3x3 boxes
    for box_r in range(3):
        for box_c in range(3):
            seen = {}
            for r in range(box_r*3, box_r*3+3):
                for c in range(box_c*3, box_c*3+3):
                    v = grid[r][c]
                    if v != 0:
                        if v in seen:
                            pr, pc = seen[v]
                            errors.append(f"Box ({box_r+1},{box_c+1}): duplicate {v}")
                            conflicts.add(r * 9 + c)
                            conflicts.add(pr * 9 + pc)
                        else:
                            seen[v] = (r, c)

    return errors, conflicts


# FIX: 1 — Rewrite resolve_conflicts() with iterative resolution
def resolve_conflicts(board, cell_details):
    """
    Iteratively resolve all OCR conflicts by zeroing the lowest-confidence
    cell in each conflicting group until the board passes validation.
    Runs up to 10 iterations to handle cascading conflicts.
    Maximum 81 - 17 = 64 cells can be zeroed (17 is minimum clues for 
    a valid sudoku). Never zero more than that.
    """
    corrected = list(board)
    
    for iteration in range(10):  # max 10 passes
        errors, conflicts = validate_board(corrected)
        
        if not errors:
            print(f"[resolve_conflicts] Clean after {iteration} iterations.")
            return corrected, [], set()
        
        print(f"[resolve_conflicts] Iteration {iteration}: "
              f"{len(conflicts)} conflicting cells, {len(errors)} errors")
        
        # Count non-zero cells — stop if we'd go below 17 clues
        nonzero = sum(1 for x in corrected if x != 0)
        if nonzero <= 17:
            print("[resolve_conflicts] Hit 17-clue floor — stopping.")
            break
        
        # For each conflict group (cells sharing a duplicate value in 
        # the same row/col/box), zero the one with lowest confidence.
        # Process ALL conflict groups in this pass before re-validating.
        
        zeroed_this_pass = set()
        
        # Build conflict groups: find every (value, scope) that has duplicates
        grid = [corrected[i*9:(i+1)*9] for i in range(9)]
        
        def get_conf(idx):
            if idx < len(cell_details) and cell_details[idx] is not None:
                return cell_details[idx].get("confidence", 0.0)
            return 0.0
        
        # Check rows
        for r in range(9):
            seen = {}
            for c in range(9):
                idx = r * 9 + c
                v = corrected[idx]
                if v == 0:
                    continue
                if v in seen:
                    # Duplicate found — zero the lower-confidence one
                    prev_idx = seen[v]
                    if get_conf(idx) >= get_conf(prev_idx):
                        victim = prev_idx
                        seen[v] = idx  # keep current as winner
                    else:
                        victim = idx
                    if victim not in zeroed_this_pass:
                        corrected[victim] = 0
                        zeroed_this_pass.add(victim)
                        print(f"  [row {r+1}] Zeroed cell {victim} "
                              f"(was {board[victim]}, conf={get_conf(victim):.3f})")
                else:
                    seen[v] = idx
        
        # Check columns
        for c in range(9):
            seen = {}
            for r in range(9):
                idx = r * 9 + c
                v = corrected[idx]
                if v == 0:
                    continue
                if v in seen:
                    prev_idx = seen[v]
                    if get_conf(idx) >= get_conf(prev_idx):
                        victim = prev_idx
                        seen[v] = idx
                    else:
                        victim = idx
                    if victim not in zeroed_this_pass:
                        corrected[victim] = 0
                        zeroed_this_pass.add(victim)
                        print(f"  [col {c+1}] Zeroed cell {victim} "
                              f"(was {board[victim]}, conf={get_conf(victim):.3f})")
                else:
                    seen[v] = idx
        
        # Check 3x3 boxes
        for box_r in range(3):
            for box_c in range(3):
                seen = {}
                for r in range(box_r*3, box_r*3+3):
                    for c in range(box_c*3, box_c*3+3):
                        idx = r * 9 + c
                        v = corrected[idx]
                        if v == 0:
                            continue
                        if v in seen:
                            prev_idx = seen[v]
                            if get_conf(idx) >= get_conf(prev_idx):
                                victim = prev_idx
                                seen[v] = idx
                            else:
                                victim = idx
                            if victim not in zeroed_this_pass:
                                corrected[victim] = 0
                                zeroed_this_pass.add(victim)
                                print(f"  [box {box_r+1},{box_c+1}] "
                                      f"Zeroed cell {victim} "
                                      f"(was {board[victim]}, "
                                      f"conf={get_conf(victim):.3f})")
                        else:
                            seen[v] = idx
        
        if not zeroed_this_pass:
            print("[resolve_conflicts] No cells zeroed this pass — stuck.")
            break
    
    # Final check
    validate_board_size(corrected)
    errors, conflicts = validate_board(corrected)
    return corrected, errors, conflicts


def process_frame(frame, classifier):
    """
    Full pipeline for webcam. Returns (output_frame, success, solved_board).
    """
    original = frame.copy()

    rot_code, warped, M, max_dim = get_best_orientation(frame, classifier)
    if rot_code is None or warped is None:
        return original, False, None

    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cells = split_boxes(gray_warped)

    board = []
    cell_details = []
    for cell in cells:
        res = extract_digit(cell)
        if len(res) >= 4:
            digit_img, area_ratio, bbox, reason = res[:4]
        else:
            digit_img, area_ratio, bbox, reason = res[0], res[1], None, "Unknown"
            
        if digit_img is None:
            board.append(0)
            cell_details.append({"confidence": 0.0})
        else:
            digit, conf, top3 = classifier.predict(digit_img, threshold=0.0)
            accepted = True
            if conf < 0.98: accepted = False
            elif len(top3) > 1 and (top3[0][1] - top3[1][1] < 0.20): accepted = False
            
            if accepted:
                board.append(digit)
                cell_details.append({"confidence": conf})
            else:
                board.append(0)
                cell_details.append({"confidence": 0.0})

    if sum(board) == 0:
        return original, False, None

    # Validate and resolve conflicts
    try:
        validate_board_size(board)
    except ValueError as e:
        return original, False, None
        
    errors, conflicts = validate_board(board)
    if errors:
        board, _, _ = resolve_conflicts(board, cell_details)

    original_board = list(board)

    try:
        status, solved_board = get_board_solution(board)
        success = (status == "SOLVED")
    except Exception:
        return original, False, None

    if not success:
        return original, False, None

    blank = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    display_numbers_on_board(blank, solved_board, original_board, new_color=(0, 255, 0))
    if rot_code == cv2.ROTATE_90_CLOCKWISE:
        blank = cv2.rotate(blank, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rot_code == cv2.ROTATE_180:
        blank = cv2.rotate(blank, cv2.ROTATE_180)
    elif rot_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
        blank = cv2.rotate(blank, cv2.ROTATE_90_CLOCKWISE)
        
    output = reverse_extract(original, blank, M)

    return output, True, solved_board


def diagnose_frame(frame, classifier):
    """
    Diagnostic pipeline for upload mode. Returns a dict with stage-by-stage info.
    """
    result = {
        "stage": "START",
        "board": None,
        "corrected_board": None,
        "solved_board": None,
        "success": False,
        "output_frame": frame.copy(),
        "cell_details": [],
        "recognized_count": 0,
        "warped": None,
        "validation_errors": [],
        "conflicts": set(),
    }

    original = frame.copy()

    # Create debug directories
    import os
    os.makedirs("debug/cells", exist_ok=True)
    
    cv2.imwrite("debug/01_original.jpg", frame)

    t_start = time.perf_counter()
    # STAGE: GRID
    board_cnt, _, _ = find_board(frame)
    t_grid = time.perf_counter()
    print(f"[Timing] Grid Detection: {t_grid - t_start:.4f}s")
    
    if board_cnt is None:
        result["stage"] = "GRID"
        result["error"] = "No 4-sided contour found in the image."
        return result

    cv2.imwrite("debug/01_original.jpg", frame)

    contour_img = frame.copy()
    cv2.drawContours(contour_img, [board_cnt], -1, (0, 0, 255), 3)
    cv2.imwrite("debug/02_detected_contour.jpg", contour_img)

    corners_img = frame.copy()
    for pt in board_cnt:
        cv2.circle(corners_img, tuple(pt[0]), 10, (255, 0, 0), -1)
    cv2.imwrite("debug/03_corner_points.jpg", corners_img)

    warped, M, max_dim = extract_board(frame, board_cnt)
    if warped is None:
        result["stage"] = "WARP"
        result["error"] = "Failed to extract board perspective."
        return result

    result["warped"] = warped.copy()
    cv2.imwrite("debug/04_warped_board.jpg", warped)

    # Auto-Orientation Engine
    original_warped_base = warped.copy()
    best_result = None
    best_recognized = -1
    
    rotations = [
        (0, "0 deg"),
        (cv2.ROTATE_90_CLOCKWISE, "90 deg CW"),
        (cv2.ROTATE_180, "180 deg"),
        (cv2.ROTATE_90_COUNTERCLOCKWISE, "90 deg CCW")
    ]
    
    for rot_code, rot_name in rotations:
        if rot_code == 0:
            current_warped = original_warped_base.copy()
        else:
            current_warped = cv2.rotate(original_warped_base, rot_code)
            
        t_rot_start = time.perf_counter()
        # STAGE: OCR
        gray_warped = cv2.cvtColor(current_warped, cv2.COLOR_BGR2GRAY)
        cells = split_boxes(gray_warped)
        
        board = []
        cell_details = []
        for idx, cell in enumerate(cells):
            res = extract_digit(cell)
            if len(res) >= 4:
                digit_img, area_ratio, bbox, reason = res[:4]
            else:
                digit_img, area_ratio, bbox, reason = res[0], res[1], None, "Unknown"
                
            cv2.imwrite(f"debug/cells/cell_{idx:02d}.png", cell)
            if digit_img is None:
                detail = {
                    "index": idx,
                    "cell_img": cell,           # No copy — read-only reference for display
                    "digit_img": None,
                    "predicted": 0,
                    "confidence": 0.0,
                    "top3": [],
                    "is_empty": True
                }
                board.append(0)
            else:
                digit, conf, top3 = classifier.predict(digit_img, threshold=0.0)
                accepted = True
                if conf < 0.98: accepted = False
                elif len(top3) > 1 and (top3[0][1] - top3[1][1] < 0.20): accepted = False
                
                if not accepted: digit = 0
                detail = {
                    "index": idx,
                    "cell_img": cell.copy(),    # Copy only for non-empty cells shown in UI
                    "digit_img": digit_img,
                    "predicted": digit,
                    "confidence": conf,
                    "top3": top3,
                    "is_empty": not accepted,
                    "area_ratio": area_ratio,
                    "bbox": bbox,
                    "digit_c": None
                }
                board.append(digit)
            cell_details.append(detail)

        recognized_count = sum(1 for d in board if d != 0)
        t_ocr = time.perf_counter()
        
        # FIX #5: Single summary print per rotation instead of 81 per-cell prints
        print(f"[{rot_name}] OCR complete: {recognized_count}/81 digits detected. [Timing] OCR: {t_ocr - t_rot_start:.4f}s")
        
        iteration_result = {
            "warped": current_warped,
            "board": list(board),
            "cell_details": cell_details,
            "recognized_count": recognized_count,
            "rot_code": rot_code
        }
        
        if recognized_count < 17:
            iteration_result["stage"] = "OCR"
            iteration_result["error"] = f"NOT ENOUGH DIGITS DETECTED. Found {recognized_count}, minimum 17."
            success = False
        else:
            errors, conflicts = validate_board(board)
            iteration_result["validation_errors"] = errors
            iteration_result["conflicts"] = conflicts
            
            if errors:
                print(f"[{rot_name}] Validation errors found. Attempting conflict resolution...")
                board, _, _ = resolve_conflicts(board, cell_details)
                iteration_result["corrected_board"] = list(board)
                
                # Re-validate after resolution
                errors2, conflicts2 = validate_board(board)
                if errors2:
                    print(f"[{rot_name}] STILL INVALID after resolution. Hard failing.")
                    iteration_result["stage"] = "VALIDATION_FAILED"
                    iteration_result["error"] = "Duplicate digits persist after conflict resolution."
                    iteration_result["validation_errors"] = errors2
                    iteration_result["conflicts"] = conflicts2
                    success = False
                else:
                    print(f"[{rot_name}] Conflicts resolved successfully.")
                    iteration_result["validation_errors"] = []
                    iteration_result["conflicts"] = set()
                    success = True  # Allow solver to proceed
            else:
                iteration_result["corrected_board"] = list(board)
                
            original_board = list(board)
            if not iteration_result.get("validation_errors", errors):
                t_solve_start = time.perf_counter()
                try:
                    print(f"[{rot_name}] OCR BOARD:")
                    print(np.array(board).reshape(9, 9))
                    v_errors, v_conflicts = validate_board(board)
                    print(f"[{rot_name}] VALIDATION ERRORS:", v_errors)
                    print(f"[{rot_name}] CONFLICT CELLS:", v_conflicts)
                    status, solved_board = get_board_solution(board)
                    print(f"[{rot_name}] SOLVER STATUS:", status)
                    success = (status == "SOLVED")
                except Exception as e:
                    success = False
                    status = "ERROR"
                    iteration_result["error"] = f"Solver crashed: {e}"
                t_solve = time.perf_counter()
                print(f"[Timing] [{rot_name}] Solver: {t_solve - t_solve_start:.4f}s")
                
            if success:
                t_render_start = time.perf_counter()
                iteration_result["solved_board"] = solved_board
                iteration_result["success"] = True
                iteration_result["stage"] = "DONE"
                
                # Project the solved grid
                blank = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
                from ai.utils.utils import display_numbers_on_board
                display_numbers_on_board(blank, solved_board, original_board, new_color=(0, 255, 0))
                
                # Un-rotate the canvas so it perfectly overlays the physical rotation
                if rot_code == cv2.ROTATE_90_CLOCKWISE:
                    blank = cv2.rotate(blank, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rot_code == cv2.ROTATE_180:
                    blank = cv2.rotate(blank, cv2.ROTATE_180)
                elif rot_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
                    blank = cv2.rotate(blank, cv2.ROTATE_90_CLOCKWISE)
                    
                iteration_result["output_frame"] = reverse_extract(original, blank, M)
                
                t_render = time.perf_counter()
                print(f"[Timing] [{rot_name}] Rendering: {t_render - t_render_start:.4f}s")
                
                result.update(iteration_result)
                print(f"✅ AUTO-ORIENTATION ENGINE: Locked onto {rot_name} orientation!")
                return result
            else:
                iteration_result["stage"] = "SOLVER"
                if "error" not in iteration_result:
                    iteration_result["error"] = f"Solver returned: {status}."
                    
        # Track best effort
        if recognized_count > best_recognized:
            best_recognized = recognized_count
            best_result = iteration_result
            
    # Exhausted all 4 rotations, return the best effort
    print("❌ AUTO-ORIENTATION ENGINE: Failed to find solvable orientation.")
    result.update(best_result)
    result["success"] = False
    result["output_frame"] = original
    return result

# FIX: 4 — Nuclear fallback in debug_frame(): if >15 conflicts, retrain-free grid-based OCR
def ocr_fallback_tesseract_free(warped, cell_size=64):
    """
    Fallback digit reader that does NOT use the CNN.
    Uses classical CV: template matching against synthetic digit templates.
    Returns an 81-element list of ints (0 = empty).
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) \
           if len(warped.shape) == 3 else warped.copy()
    
    h, w   = gray.shape[:2]
    board  = []
    
    # Build synthetic templates for digits 1-9 at multiple sizes
    templates = {}
    for digit in range(1, 10):
        tmpl = np.zeros((cell_size, cell_size), dtype=np.uint8)
        fs   = cell_size / 48.0
        cv2.putText(tmpl, str(digit), 
                    (int(cell_size*0.2), int(cell_size*0.8)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, 255, 2)
        templates[digit] = tmpl
    
    for r in range(9):
        for c in range(9):
            y1 = int(r * h / 9)
            y2 = int((r+1) * h / 9)
            x1 = int(c * w / 9)
            x2 = int((c+1) * w / 9)
            
            pad_y = max(3, (y2-y1)//8)
            pad_x = max(3, (x2-x1)//8)
            cell  = gray[y1+pad_y:y2-pad_y, x1+pad_x:x2-pad_x]
            
            if cell.size == 0:
                board.append(0)
                continue
            
            # Check if cell is empty (very uniform brightness)
            std = np.std(cell)
            if std < 12:
                board.append(0)
                continue
            
            # Resize to template size
            cell_rs = cv2.resize(cell, (cell_size, cell_size))
            
            # Binarize
            _, binary = cv2.threshold(cell_rs, 0, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Match against each digit template
            best_digit = 0
            best_score = -1
            for digit, tmpl in templates.items():
                result = cv2.matchTemplate(binary, tmpl, 
                                           cv2.TM_CCOEFF_NORMED)
                score = float(result.max())
                if score > best_score:
                    best_score = score
                    best_digit = digit
            
            # Only accept if match score is reasonable
            board.append(best_digit if best_score > 0.25 else 0)
    
    return board

def debug_frame(frame, classifier):
    import os
    import time
    os.makedirs("debug/cells", exist_ok=True)
    
    result = {
        "stage": "START",
        "board": None,
        "cell_details": [],
        "validation_errors": [],
        "conflicts": set(),
        "error": None
    }
    
    # 1. Original image
    cv2.imwrite("debug/01_original.jpg", frame)
    
    # 2. Board detection
    board_cnt, _, _ = find_board(frame)
    if board_cnt is None:
        result["stage"] = "GRID"
        result["error"] = "No 4-sided contour found."
        return result
        
    contour_img = frame.copy()
    cv2.drawContours(contour_img, [board_cnt], -1, (0, 0, 255), 3)
    cv2.imwrite("debug/02_detected_contour.jpg", contour_img)
    
    # 3. Corner detection
    corners_img = frame.copy()
    for pt in board_cnt:
        cv2.circle(corners_img, tuple(pt[0]), 10, (255, 0, 0), -1)
    cv2.imwrite("debug/03_corner_points.jpg", corners_img)
    
    # 4. Perspective warp
    warped, M, max_dim = extract_board(frame, board_cnt)
    if warped is None:
        result["stage"] = "WARP"
        result["error"] = "Perspective warp failed."
        return result
    cv2.imwrite("debug/04_warped_board.jpg", warped)
    
    # 5. Grid overlay
    grid_img = warped.copy()
    h, w = grid_img.shape[:2]
    for i in range(10):
        cv2.line(grid_img, (0, int(i*h/9)), (w, int(i*h/9)), (0, 255, 0), 2)
        cv2.line(grid_img, (int(i*w/9), 0), (int(i*w/9), h), (0, 255, 0), 2)
    cv2.imwrite("debug/05_grid_overlay.jpg", grid_img)
    
    # 6. Cell numbering
    num_img = grid_img.copy()
    for r in range(9):
        for c in range(9):
            idx = r * 9 + c
            cv2.putText(num_img, str(idx), (int(c*w/9) + 5, int(r*h/9) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imwrite("debug/06_cell_numbering.jpg", num_img)
    
    # 7. Digit contour extraction & CNN
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cells = split_boxes(gray_warped)
    
    board = []
    cell_details = []
    cont_img = warped.copy()
    
    for idx, cell in enumerate(cells):
        res = extract_digit(cell)
        if len(res) >= 4:
            digit_img, area_ratio, bbox, reason = res[:4]
        else:
            digit_img, area_ratio, bbox, reason = res[0], res[1], None, "Unknown"
            
        cv2.imwrite(f"debug/cells/cell_{idx:02d}.png", cell)
        
        if digit_img is None:
            print(f"Cell {idx}\nPrediction=N/A\nConfidence=0.000\nTop-3 Predictions=[]\nRejected\nReason={reason}\n")
            board.append(0)
            cell_details.append({
                "index": idx, "cell_img": cell, "digit_img": None,
                "predicted": 0, "confidence": 0.0, "is_empty": True
            })
        else:
            digit, conf, top3 = classifier.predict(digit_img, threshold=0.0)
            accepted = True
            reject_reason = reason
            if reason != "Accepted":
                accepted = False
            elif conf < 0.98:
                accepted = False
                reject_reason = "Low Confidence"
            elif len(top3) > 1 and (top3[0][1] - top3[1][1] < 0.20):
                accepted = False
                reject_reason = "Ambiguous Top-2"
                
            print(f"Cell {idx}\nPrediction={top3[0][0] if top3 else digit}\nConfidence={conf:.3f}\nTop-3={top3}\n{'Accepted' if accepted else 'Rejected'}")
            if not accepted:
                print(f"Reason={reject_reason}\n")
            else:
                print("\n")
                
            if not accepted:
                digit = 0
                
            board.append(digit)
            cell_details.append({
                "index": idx, "cell_img": cell.copy(), "digit_img": digit_img,
                "predicted": digit, "confidence": conf, "top3": top3, "is_empty": not accepted
            })
            
            # Draw bbox and area on 07
            c_col = idx % 9
            r_row = idx // 9
            cell_w = int(w/9)
            cell_h = int(h/9)
            x_offset = int(c_col * cell_w)
            y_offset = int(r_row * cell_h)
            
            if bbox is not None:
                x, y, bw, bh = bbox
                cv2.rectangle(
                    cont_img,
                    (x_offset + x, y_offset + y),
                    (x_offset + x + bw, y_offset + y + bh),
                    (0, 0, 255),
                    2
                )
                cv2.putText(
                    cont_img,
                    f"{area_ratio*100:.1f}%",
                    (x_offset + x, y_offset + y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1
                )
    cv2.imwrite("debug/07_digit_contours.jpg", cont_img)
    
    # Sequential Consistency Check
    if len(board) < 81:
        pad = [0] * (81 - len(board))
        board.extend(pad)
        
    final_board = list(board)
    
    print("LEN CELLS =", len(cells))
    print("LEN BOARD =", len(board))
    print("LEN FINAL BOARD =", len(final_board))

    if len(final_board) != 81:
        raise RuntimeError(
            f"Invalid final_board length {len(final_board)}"
        )

    for idx, v in enumerate(final_board):
        if idx >= len(final_board):
            continue
            
        if v == 0:
            continue
            
        r, c = divmod(idx, 9)
        conflict = False
        
        for c2 in range(9):
            pos = r * 9 + c2
            if pos >= len(final_board):
                continue
            if pos != idx and final_board[pos] == v:
                conflict = True
                
        for r2 in range(9):
            pos = r2 * 9 + c
            if pos >= len(final_board):
                continue
            if pos != idx and final_board[pos] == v:
                conflict = True
                
        br, bc = r // 3, c // 3
        for r2 in range(br*3, br*3+3):
            for c2 in range(bc*3, bc*3+3):
                pos = r2 * 9 + c2
                if pos >= len(final_board):
                    continue
                if pos != idx and final_board[pos] == v:
                    conflict = True
                
        if conflict:
            print(f"Cell {idx} (Digit {v}) rejected due to consistency check.")
            final_board[idx] = 0
            if idx < len(cell_details):
                cell_details[idx]["is_empty"] = True
                cell_details[idx]["predicted"] = 0
            
    print("Final Board Length:", len(final_board))
    board = final_board

    # 8. OCR Matrix Dump
    with open("debug/08_ocr_matrix.txt", "w") as f:
        f.write("[\n")
        for r in range(9):
            f.write("  " + str(board[r*9:(r+1)*9]) + ",\n")
        f.write("]\n")

    result["board"] = board
    result["cell_details"] = cell_details
    
    # FIX: 2 — Rewrite debug_frame() conflict handling to use Fix 1
    errors, conflicts = validate_board(board)
    original_conflicts = set(conflicts)
    original_errors    = list(errors)
    
    if errors:
        print(f"[debug_frame] OCR conflicts detected: {len(errors)} errors. "
              f"Attempting iterative resolution...")
        # Attempt iterative conflict resolution
        corrected_board, remaining_errors, remaining_conflicts = \
            resolve_conflicts(board, cell_details)
        
        print(f"[debug_frame] After resolution: "
              f"{len(remaining_errors)} errors remain.")
    else:
        corrected_board    = board
        remaining_errors   = []
        remaining_conflicts = set()
    
    # Attempt to solve with the (possibly corrected) board
    from ai.solver import get_board_solution
    from ai.utils.utils import display_numbers_on_board
    
    clue_count = sum(1 for x in corrected_board if x != 0)
    print(f"[debug_frame] Attempting solve with {clue_count} clues...")
    
    status, solved = get_board_solution(corrected_board)
    print(f"[debug_frame] Solver status: {status}")
    
    if status == "SOLVED":
        result["solved_board"]          = solved
        result["stage"]                 = "SOLVED"
        result["board"]                 = corrected_board
        result["ocr_conflicts_resolved"] = len(original_conflicts) > 0
        result["original_conflicts"]    = sorted(original_conflicts)
        result["original_errors"]       = original_errors
        result["warped"]                = warped
        result["solution_grid_img"]     = draw_solution_grid(
                                              corrected_board, solved, 
                                              cell_size=64)
        blank = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        display_numbers_on_board(blank, solved, corrected_board, new_color=(0, 255, 0))
        result["output_frame"]          = reverse_extract(frame, blank, M)
    else:
        # Last resort: zero ALL originally conflicting cells and retry
        print("[debug_frame] Normal solve failed. Trying last-resort: "
              "zero all original conflict cells...")
        last_resort = list(board)
        for idx in original_conflicts:
            last_resort[idx] = 0
        
        # Clean any remaining conflicts from last_resort
        last_resort, _, _ = resolve_conflicts(
            last_resort, 
            [{} for _ in range(81)]  # no confidence info, zero all
        )
        
        status2, solved2 = get_board_solution(last_resort)
        print(f"[debug_frame] Last-resort solver status: {status2}")
        
        if status2 == "SOLVED":
            result["solved_board"]          = solved2
            result["stage"]                 = "SOLVED"
            result["board"]                 = last_resort
            result["ocr_conflicts_resolved"] = True
            result["original_conflicts"]    = sorted(original_conflicts)
            result["original_errors"]       = original_errors
            result["warped"]                = warped
            result["solution_grid_img"]     = draw_solution_grid(
                                                  last_resort, solved2,
                                                  cell_size=64)
            blank = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
            display_numbers_on_board(blank, solved2, last_resort, new_color=(0, 255, 0))
            result["output_frame"]          = reverse_extract(frame, blank, M)
        else:
            conflict_count = len(original_conflicts)
            if conflict_count > 15:
                print(f"[debug_frame] {conflict_count} conflicts — CNN is "
                      f"systematically failing. Trying template-match fallback...")
                
                fallback_board = ocr_fallback_tesseract_free(warped, cell_size=64)
                fb_errors, fb_conflicts = validate_board(fallback_board)
                
                print(f"[debug_frame] Fallback board conflicts: {len(fb_errors)}")
                
                if len(fb_errors) < conflict_count:
                    # Fallback is better — try to solve it
                    fb_corrected, _, _ = resolve_conflicts(
                        fallback_board, 
                        [{"confidence": 0.5} for _ in range(81)]
                    )
                    status_fb, solved_fb = get_board_solution(fb_corrected)
                    
                    if status_fb == "SOLVED":
                        result["solved_board"]          = solved_fb
                        result["stage"]                 = "SOLVED"
                        result["board"]                 = fb_corrected
                        result["ocr_conflicts_resolved"] = True
                        result["original_conflicts"]    = sorted(original_conflicts)
                        result["original_errors"]       = original_errors
                        result["warped"]                = warped
                        result["solution_grid_img"]     = draw_solution_grid(
                                                              fb_corrected, solved_fb,
                                                              cell_size=64)
                        blank = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
                        display_numbers_on_board(blank, solved_fb, fb_corrected, new_color=(0, 255, 0))
                        result["output_frame"]          = reverse_extract(frame, blank, M)
                        return result

            result["stage"]              = "OCR_VALIDATION"
            result["validation_errors"]  = original_errors
            result["conflicts"]          = original_conflicts
            result["solved_board"]       = None
            result["output_frame"]       = frame
            result["solution_grid_img"]  = draw_solution_grid(corrected_board, [0]*81, cell_size=64)
            
    return result

def get_best_orientation(frame, classifier):
    board_cnt, _, _ = find_board(frame)
    if board_cnt is None: return None, None, None, None
    warped, M, max_dim = extract_board(frame, board_cnt)
    if warped is None: return None, None, None, None

    rotations = [
        (0, "0 deg"),
        (cv2.ROTATE_90_CLOCKWISE, "90 deg CW"),
        (cv2.ROTATE_180, "180 deg"),
        (cv2.ROTATE_90_COUNTERCLOCKWISE, "90 CCW")
    ]
    
    best_score = -9999
    best_rot = 0
    best_warped = warped
    
    for rot_code, rot_name in rotations:
        current_warped = warped if rot_code == 0 else cv2.rotate(warped, rot_code)
        gray_warped = cv2.cvtColor(current_warped, cv2.COLOR_BGR2GRAY)
        cells = split_boxes(gray_warped)
        
        board = []
        recognized = 0
        total_conf = 0.0
        
        for cell in cells:
            res = extract_digit(cell)
            if len(res) >= 4:
                digit_img, area_ratio, bbox, reason = res[:4]
            else:
                digit_img = res[0]
                
            if digit_img is None:
                board.append(0)
            else:
                digit, conf, top3 = classifier.predict(digit_img, threshold=0.0)
                accepted = True
                if conf < 0.98: accepted = False
                elif len(top3) > 1 and (top3[0][1] - top3[1][1] < 0.20): accepted = False
                
                if accepted:
                    board.append(digit)
                    if digit != 0:
                        recognized += 1
                        total_conf += conf
                else:
                    board.append(0)
                    
        errors, _ = validate_board(board)
        score = recognized * 10 + total_conf - len(errors) * 100
        
        if score > best_score:
            best_score = score
            best_rot = rot_code
            best_warped = current_warped
            
    return best_rot, best_warped, M, max_dim

def process_multi_frame(frames, classifier):
    from collections import Counter
    import time
    frames.sort(key=lambda x: x[0], reverse=True)
    best_img = frames[0][1]
    
    rot_code, best_warped, best_M, best_max_dim = get_best_orientation(best_img, classifier)
    if rot_code is None:
        return {"success": False, "error": "Board detection failed."}
        
    votes = [[] for _ in range(81)]
    
    for blur, img in frames:
        board_cnt, _, _ = find_board(img)
        if board_cnt is None: continue
        warped, M, max_dim = extract_board(img, board_cnt)
        if warped is None: continue
        
        current_warped = warped if rot_code == 0 else cv2.rotate(warped, rot_code)
        gray_warped = cv2.cvtColor(current_warped, cv2.COLOR_BGR2GRAY)
        cells = split_boxes(gray_warped)
        
        for idx, cell in enumerate(cells):
            res = extract_digit(cell)
            if len(res) >= 4:
                digit_img = res[0]
            else:
                digit_img = res[0]
                
            if digit_img is None:
                votes[idx].append((0, 0.0))
            else:
                digit, conf, top3 = classifier.predict(digit_img, threshold=0.0)
                accepted = True
                if conf < 0.98: accepted = False
                elif len(top3) > 1 and (top3[0][1] - top3[1][1] < 0.20): accepted = False
                
                if accepted:
                    votes[idx].append((digit, conf))
                else:
                    votes[idx].append((0, 0.0))
                
    final_board = []
    cell_details = []
    for idx, cell_votes in enumerate(votes):
        if not cell_votes:
            final_board.append(0)
            cell_details.append({"confidence": 0.0, "is_empty": True})
            continue
            
        digits = [v[0] for v in cell_votes]
        counter = Counter(digits)
        most_common_digit, count = counter.most_common(1)[0]
        
        confs = [v[1] for v in cell_votes if v[0] == most_common_digit]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        
        final_board.append(most_common_digit)
        cell_details.append({"confidence": avg_conf, "is_empty": most_common_digit == 0})
        
    errors, conflicts = validate_board(final_board)
    clue_count = sum(1 for d in final_board if d != 0)
    
    result = {
        "success": False,
        "board": final_board,
        "recognized_count": clue_count,
        "validation_errors": errors,
        "warped": best_warped,
        "output_frame": best_img,
        "cell_details": cell_details
    }
    
    if clue_count < 17:
        result["error"] = f"Insufficient clues: {clue_count} < 17"
        return result
        
    if errors:
        result["error"] = "Sudoku rule violations detected."
        return result
        
    try:
        status, solved_board = get_board_solution(final_board)
        if status == "SOLVED":
            result["success"] = True
            result["solved_board"] = solved_board
            
            blank = np.zeros((best_max_dim, best_max_dim, 3), dtype=np.uint8)
            display_numbers_on_board(blank, solved_board, final_board, new_color=(0, 255, 0))
            if rot_code == cv2.ROTATE_90_CLOCKWISE:
                blank = cv2.rotate(blank, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rot_code == cv2.ROTATE_180:
                blank = cv2.rotate(blank, cv2.ROTATE_180)
            elif rot_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
                blank = cv2.rotate(blank, cv2.ROTATE_90_CLOCKWISE)
                
            output = reverse_extract(best_img, blank, best_M)
            result["output_frame"] = output
        else:
            result["error"] = f"Solver failed: {status}"
    except Exception as e:
        result["error"] = f"Solver crashed: {str(e)}"
        
    return result
