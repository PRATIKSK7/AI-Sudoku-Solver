import cv2
import numpy as np
import tensorflow as tf
import os

model = tf.keras.models.load_model('models/sudoku_digit_model.h5')
input_dir = 'debug/cnn_input'
files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

print("Testing TRANSPOSE transform on all images:")
confs = []
for f in files:
    img = cv2.imread(os.path.join(input_dir, f), cv2.IMREAD_GRAYSCALE)
    t_img = np.transpose(img)
    tensor = cv2.resize(t_img, (28, 28)).astype('float32') / 255.0
    tensor = tensor.reshape(1, 28, 28, 1)
    preds = model.predict(tensor, verbose=0)[0]
    conf = np.max(preds)
    pred = np.argmax(preds)
    confs.append(conf)

print(f"Average confidence with TRANSPOSE: {np.mean(confs)*100:.2f}%")

print("\nTesting ROT90_CCW+FLIPLR transform on all images:")
confs2 = []
for f in files:
    img = cv2.imread(os.path.join(input_dir, f), cv2.IMREAD_GRAYSCALE)
    t_img = np.flip(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), axis=1)
    tensor = cv2.resize(t_img, (28, 28)).astype('float32') / 255.0
    tensor = tensor.reshape(1, 28, 28, 1)
    preds = model.predict(tensor, verbose=0)[0]
    conf = np.max(preds)
    pred = np.argmax(preds)
    confs2.append(conf)

print(f"Average confidence with ROT90_CCW+FLIPLR: {np.mean(confs2)*100:.2f}%")

