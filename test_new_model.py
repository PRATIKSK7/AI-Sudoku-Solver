import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('models/sudoku_digit_model_new.h5')

def evaluate_img(path, true_label):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    transforms = {
        'normal': img,
        'transpose': np.transpose(img),
        'rot90_ccw+fliplr': np.flip(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), axis=1)
    }
    for name, t_img in transforms.items():
        tensor = cv2.resize(t_img, (28, 28)).astype('float32') / 255.0
        tensor = tensor.reshape(1, 28, 28, 1)
        preds = model.predict(tensor, verbose=0)[0]
        pred_label = np.argmax(preds)
        print(f"[{path}] {name}: Pred={pred_label} Conf={preds[pred_label]*100:.2f}% | Expected: {true_label}")

evaluate_img('debug/cnn_input/cnn_01.png', 6)
evaluate_img('debug/cnn_input/cnn_02.png', 1)
evaluate_img('debug/cnn_input/cnn_04.png', 5)

