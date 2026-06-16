import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('models/sudoku_digit_model.h5')

img = cv2.imread('debug/cnn_input/cnn_02.png', cv2.IMREAD_GRAYSCALE)

# Try erosions
for er in range(1, 5):
    kernel = np.ones((er, er), np.uint8)
    eroded = cv2.erode(img, kernel, iterations=1)
    
    transforms = {
        'normal': eroded,
        'transpose': np.transpose(eroded),
        'rot90_cw': cv2.rotate(eroded, cv2.ROTATE_90_CLOCKWISE),
        'rot90_ccw': cv2.rotate(eroded, cv2.ROTATE_90_COUNTERCLOCKWISE)
    }
    
    for name, t_img in transforms.items():
        tensor = cv2.resize(t_img, (28, 28)).astype('float32') / 255.0
        tensor = tensor.reshape(1, 28, 28, 1)
        preds = model.predict(tensor, verbose=0)[0]
        if np.argmax(preds) == 1:
            print(f"Erosion {er}x{er} + {name} -> 1! Conf: {np.max(preds)*100:.2f}%")

