import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/sudoku_digit_model.h5")

files = [
    "debug/cnn_input/cnn_01.png",
    "debug/cnn_input/cnn_02.png",
    "debug/cnn_input/cnn_04.png"
]

for file in files:
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))

    pred = model.predict(img, verbose=0)[0]

    print("\n", file)
    print("Predicted:", np.argmax(pred))
    print("Confidence:", np.max(pred))

