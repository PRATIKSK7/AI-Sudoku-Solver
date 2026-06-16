import os

class Config:
    PROJECT_NAME = "Real-Time AI Sudoku Solver"
    MODEL_PATH = os.path.join("models", "sudoku_digit_model.h5")
    IMAGE_SIZE = 28
    BATCH_SIZE = 64
    EPOCHS = 20
    CONFIDENCE_THRESHOLD = 0.5
