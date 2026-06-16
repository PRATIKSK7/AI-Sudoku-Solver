import cv2
import numpy as np
import tensorflow as tf

img = cv2.imread('debug/cnn_input/cnn_01.png', cv2.IMREAD_GRAYSCALE)
tensor = img.astype('float32') / 255.0
tensor = tensor.reshape(1, 28, 28, 1)

try:
    model1 = tf.keras.models.load_model('models/sudoku_digit_model.h5')
    preds = model1.predict(tensor, verbose=0)[0]
    print(f"model1: pred={np.argmax(preds)} conf={preds[np.argmax(preds)]:.3f}")
except Exception as e:
    print("model1 failed:", e)

try:
    model2 = tf.keras.models.load_model('models/sudoku_digit_model_new.h5')
    preds = model2.predict(tensor, verbose=0)[0]
    print(f"model2: pred={np.argmax(preds)} conf={preds[np.argmax(preds)]:.3f}")
except Exception as e:
    print("model2 failed:", e)

# Test with EMNIST transpose
tensor_t = np.transpose(img, (1, 0)) # wait, transpose in 2D is (1,0)
tensor_t = np.flip(tensor_t, axis=1) # EMNIST style flip
tensor_t = tensor_t.astype('float32') / 255.0
tensor_t = tensor_t.reshape(1, 28, 28, 1)

preds = model1.predict(tensor_t, verbose=0)[0]
print(f"model1 (transposed): pred={np.argmax(preds)} conf={preds[np.argmax(preds)]:.3f}")

