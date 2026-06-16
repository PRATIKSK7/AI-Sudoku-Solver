# Real-Time AI Sudoku Solver 🧩

An industrial-grade computer vision and deep learning application that detects, recognizes, solves, and projects Sudoku solutions in real-time from a webcam feed or uploaded images.

## Features
- **Sudoku Grid Detection:** Detects the board, handles rotation, tilt, shadows, and noise.
- **Perspective Correction:** Performs bird-eye transformation to extract a perfectly aligned grid.
- **Digit Extraction:** Isolates each cell and extracts digits.
- **CNN Digit Recognition:** Custom Keras model trained to recognize Sudoku digits.
- **Solver Engine:** Highly optimized Backtracking algorithm that solves the puzzle in milliseconds.
- **Solution Projection:** Real-time projection of solved digits back onto the original frame.
- **Professional UI:** Streamlit dashboard for webcam integration and image upload.
- **API Backend:** FastAPI endpoints for integration.

## Tech Stack
- **Frontend:** Streamlit
- **Backend:** FastAPI, Python 3.11+
- **Computer Vision:** OpenCV, NumPy, Imutils
- **Deep Learning:** TensorFlow, Keras
- **Deployment:** Docker, Docker Compose

## Quick Start (Docker)

1. Make sure you have Docker and Docker Compose installed.
2. Build and run the containers:
   ```bash
   docker-compose up --build
   ```
3. Open Streamlit UI: http://localhost:8501
4. Open FastAPI Docs: http://localhost:8000/docs

## Manual Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the CNN model (if not already trained):
   ```bash
   python train_model.py
   ```
3. Run FastAPI backend:
   ```bash
   uvicorn main:app --reload
   ```
4. Run Streamlit UI:
   ```bash
   streamlit run app.py
   ```

## Directory Structure

- `app.py`: Streamlit Frontend
- `main.py`: FastAPI Backend
- `solver.py`: Sudoku Backtracking Logic
- `digit_classifier.py`: CNN model loader and predictor
- `train_model.py`: Model training pipeline
- `perspective.py`: Board detection and perspective warp
- `segmentation.py`: Cell splitting and digit extraction
- `detector.py`: Main CV pipeline orchestrator
- `utils.py`: Shared utilities (drawing, displaying)
