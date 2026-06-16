import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('models/sudoku_digit_model.h5')

img = cv2.imread('debug/cnn_input/cnn_02.png', cv2.IMREAD_GRAYSCALE)

transforms = {
    'normal': img,
    'rot90_cw': cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
    'rot180': cv2.rotate(img, cv2.ROTATE_180),
    'rot90_ccw': cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
    'fliplr': np.flip(img, axis=1),
    'flipud': np.flip(img, axis=0),
    'transpose': np.transpose(img),
    'transpose+fliplr': np.flip(np.transpose(img), axis=1),
    'transpose+flipud': np.flip(np.transpose(img), axis=0),
    'rot90_cw+fliplr': np.flip(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), axis=1),
    'rot90_ccw+fliplr': np.flip(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), axis=1)
}

print(f"\nEvaluating cnn_02.png for digit 1")
for name, t_img in transforms.items():
    tensor = cv2.resize(t_img, (28, 28)).astype('float32') / 255.0
    tensor = tensor.reshape(1, 28, 28, 1)
    preds = model.predict(tensor, verbose=0)[0]
    pred_label = np.argmax(preds)
    conf_1 = preds[1]
    print(f"  {name:<20}: Pred={pred_label} (Conf {preds[pred_label]*100:.2f}%) | Conf for 1: {conf_1*100:.2f}%")

