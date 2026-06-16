import cv2
import numpy as np
import tensorflow as tf

img = cv2.imread('debug/cnn_input/cnn_01.png', cv2.IMREAD_GRAYSCALE)
model1 = tf.keras.models.load_model('models/sudoku_digit_model.h5')

def test_trans(name, trans_img):
    t = trans_img.astype('float32') / 255.0
    t = t.reshape(1, 28, 28, 1)
    preds = model1.predict(t, verbose=0)[0]
    print(f"{name}: pred={np.argmax(preds)} conf={preds[np.argmax(preds)]:.3f}")

test_trans("Original", img)
test_trans("Transposed", np.transpose(img))
test_trans("Flipped X", np.flip(img, axis=1))
test_trans("Flipped Y", np.flip(img, axis=0))
test_trans("Rot 90 CW", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
test_trans("Rot 90 CCW", cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
test_trans("Rot 180", cv2.rotate(img, cv2.ROTATE_180))

