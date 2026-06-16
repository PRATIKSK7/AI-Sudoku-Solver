import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("PHASE 1 — MODEL VALIDATION")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

model_path = "models/sudoku_digit_model.h5"
try:
    model = load_model(model_path)
    print("model.input_shape:", model.input_shape)
    print("model.output_shape:", model.output_shape)
    model.summary()
except Exception as e:
    print("Error loading model:", e)

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("PHASE 2 — SINGLE DIGIT DEBUG")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# Create a synthetic digit exactly like extract_digit creates
# which should look like a white digit on a black background
dummy_digit = np.zeros((28, 28), dtype=np.uint8)
cv2.putText(dummy_digit, "4", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 2)

print("Original Cell (dummy_digit): Shape:", dummy_digit.shape)

# Simulating predict_with_softmax logic
from ai.digit_classifier import DigitClassifier
classifier = DigitClassifier(model_path=model_path)

_, thresh = cv2.threshold(dummy_digit, 128, 255, cv2.THRESH_BINARY)
print("Thresholded Cell: non-zero pixels:", cv2.countNonZero(thresh))

# Wait, if cv2.countNonZero(thresh) < 20, it returns 0.
if cv2.countNonZero(thresh) < 20:
    print("Digit is considered empty because countNonZero < 20")
else:
    # Let's inspect the tensor values
    img = dummy_digit.astype('float32') / 255.0
    print("Normalized Cell: min:", np.min(img), "max:", np.max(img))
    
    img_tensor = img.reshape(1, 28, 28, 1)
    print("Tensor Shape:", img_tensor.shape)
    
    prediction = model.predict(img_tensor, verbose=0)
    print("Raw Output Vector:\n", prediction)
    
    class_id = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][class_id]
    print(f"Argmax Result: {class_id}")
    print(f"Confidence: {confidence:.2f}")

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("PHASE 3 — PREPROCESSING AUDIT")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("Let's simulate the EXACT output of extract_digit to see what is fed to predict_with_softmax")

from ai.segmentation import extract_digit

# Let's create a 50x50 cell that is white with a black digit, like a piece of paper
cell_paper = np.ones((50, 50), dtype=np.uint8) * 255
# draw a black digit "4"
cv2.putText(cell_paper, "4", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0), 3)

extracted = extract_digit(cell_paper)
if extracted is not None:
    print("Extracted digit shape:", extracted.shape)
    print("Extracted digit non-zero pixels:", cv2.countNonZero(extracted))
    print("Extracted digit min/max:", np.min(extracted), np.max(extracted))
    
    print("\nNow testing predict_with_softmax on the extracted digit:")
    digit, conf, softmax = classifier.predict_with_softmax(extracted)
    print(f"Predicted Digit: {digit}, Confidence: {conf:.2f}")
    
    print("\nIf the prediction is still wrong, maybe the model expects different normalization?")
else:
    print("extract_digit returned None")
