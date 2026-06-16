import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('models/sudoku_digit_model.h5')

def evaluate_img(path, true_label):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print(f"\nEvaluating {path} (Expected: {true_label})")
    for er in range(1, 4):
        kernel = np.ones((er, er), np.uint8)
        eroded = cv2.erode(img, kernel, iterations=1)
        
        transforms = {
            'normal': eroded,
            'transpose': np.transpose(eroded),
            'rot90_cw': cv2.rotate(eroded, cv2.ROTATE_90_CLOCKWISE),
            'rot90_ccw': cv2.rotate(eroded, cv2.ROTATE_90_COUNTERCLOCKWISE),
            'rot90_ccw+fliplr': np.flip(cv2.rotate(eroded, cv2.ROTATE_90_COUNTERCLOCKWISE), axis=1)
        }
        
        for name, t_img in transforms.items():
            tensor = cv2.resize(t_img, (28, 28)).astype('float32') / 255.0
            tensor = tensor.reshape(1, 28, 28, 1)
            preds = model.predict(tensor, verbose=0)[0]
            pred_label = np.argmax(preds)
            if pred_label == true_label:
                print(f"  Erosion {er}x{er} + {name}: Pred={pred_label} Conf={preds[pred_label]*100:.2f}%")

evaluate_img('debug/cnn_input/cnn_01.png', 6)
evaluate_img('debug/cnn_input/cnn_02.png', 1)
evaluate_img('debug/cnn_input/cnn_04.png', 5)

