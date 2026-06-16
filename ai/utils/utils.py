import cv2
import numpy as np
import math

def order_points(pts):
    """
    Order points in top-left, top-right, bottom-right, bottom-left order.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def display_image(title, image, wait=True):
    """
    Utility function to display an image for debugging.
    """
    cv2.imshow(title, image)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def display_numbers_on_board(img, new_numbers, original_numbers, new_color=(0, 255, 0), orig_color=None):
    """
    Draws the solved numbers onto a blank Sudoku board image.
    img: Blank canvas matching the warped board's shape.
    new_numbers: 81-length list of solved digits.
    original_numbers: 81-length list of original detected digits.
    new_color: BGR tuple for the text color of new digits.
    orig_color: BGR tuple for the text color of original digits. If None, original digits are NOT drawn.
    """
    sec_w = int(img.shape[1] / 9)
    sec_h = int(img.shape[0] / 9)
    
    font_scale = sec_w / 60.0
    thickness = max(2, int(font_scale * 2.5))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for x in range(9):
        for y in range(9):
            idx = (y * 9) + x
            
            text = ""
            color = new_color
            
            if original_numbers[idx] != 0:
                if orig_color is not None:
                    text = str(original_numbers[idx])
                    color = orig_color
            elif new_numbers[idx] != 0:
                text = str(new_numbers[idx])
                color = new_color
                
            if text != "":
                # Calculate text size for perfect centering
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Center coords
                center_x = (x * sec_w) + (sec_w // 2)
                center_y = (y * sec_h) + (sec_h // 2)
                
                # Bottom-left origin for putText
                origin_x = center_x - (text_w // 2)
                origin_y = center_y + (text_h // 2)
                
                cv2.putText(img, text, (origin_x, origin_y), font, font_scale, color, thickness, cv2.LINE_AA)
                
    return img

def draw_grid_debug(img):
    """
    Utility to draw grid lines and (Row, Col) indices on an image for validation.
    """
    sec_w = int(img.shape[1] / 9)
    sec_h = int(img.shape[0] / 9)
    font_scale = sec_w / 100.0
    
    for i in range(0, 9):
        pt1 = (0, sec_h * i)
        pt2 = (img.shape[1], sec_h * i)
        pt3 = (sec_w * i, 0)
        pt4 = (sec_w * i, img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)
        cv2.line(img, pt3, pt4, (255, 255, 0), 2)
        
    for x in range(9):
        for y in range(9):
            text = f"{y},{x}"
            cv2.putText(img, text, (x * sec_w + 5, y * sec_h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1, cv2.LINE_AA)
                        
    return img
