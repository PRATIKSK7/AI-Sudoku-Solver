import io
import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

from ai.digit_classifier import DigitClassifier
from ai.solver import get_board_solution
from ai.detector import process_frame
from ai.perspective import find_board, extract_board
from ai.segmentation import split_boxes, extract_digit

app = FastAPI(title="Real-Time AI Sudoku Solver API", version="1.0")
classifier = DigitClassifier(model_path='models/sudoku_digit_model.h5')

class BoardRequest(BaseModel):
    board: List[int]

@app.get("/")
def root():
    return {"message": "AI Sudoku Solver API is running."}

@app.post("/solve")
def solve_board(request: BoardRequest):
    """
    Takes an 81-length array, solves it, and returns the solved array.
    """
    if len(request.board) != 81:
        raise HTTPException(status_code=400, detail="Board must be an array of length 81.")
        
    board = list(request.board)
    status, solved_board = get_board_solution(board)
    
    if status != "SOLVED":
        raise HTTPException(status_code=400, detail=f"Sudoku board is {status.lower()}.")
        
    return {"success": True, "solved_board": solved_board}

@app.post("/predict")
async def predict_digits(file: UploadFile = File(...)):
    """
    Takes an image of a Sudoku board, runs the CV pipeline to extract the board and digits,
    and returns the predicted 81-length array.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
        
    board_cnt, _, _ = find_board(img)
    if board_cnt is None:
        raise HTTPException(status_code=400, detail="Could not detect Sudoku board in the image.")
        
    warped, _, _ = extract_board(img, board_cnt)
    if warped is None:
        raise HTTPException(status_code=400, detail="Perspective warp failed.")
        
    cells = split_boxes(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
    
    board = []
    confidences = []
    
    for cell in cells:
        digit_img, area_ratio = extract_digit(cell)
        if digit_img is None:
            board.append(0)
            confidences.append(1.0)
        else:
            digit, conf, _ = classifier.predict(digit_img)
            if conf > 0.5:
                board.append(digit)
                confidences.append(conf)
            else:
                board.append(0)
                confidences.append(conf)
                
    return {"success": True, "board": board, "confidences": confidences}

@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    """
    Takes an image, runs the full end-to-end pipeline (detect, extract, solve, project),
    and returns the processed image with the solution overlaid.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
        
    final_frame, success, _ = process_frame(img, classifier)
    
    if not success:
        # If it failed to solve or detect, just return the original frame with an error header
        # but for simplicity we return the original image
        final_frame = img
        
    _, encoded_img = cv2.imencode('.jpg', final_frame)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")
