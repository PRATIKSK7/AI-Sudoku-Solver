import cv2
import numpy as np
import tensorflow as tf
import os

model = tf.keras.models.load_model('models/sudoku_digit_model.h5')

input_dir = 'debug/cnn_input'
files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

transforms = ['normal', 'rot90_cw', 'rot180', 'rot90_ccw', 'fliplr', 'flipud', 'transpose', 'transpose_flip']

def apply_t(img, name):
    if name == 'normal': return img
    if name == 'rot90_cw': return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if name == 'rot180': return cv2.rotate(img, cv2.ROTATE_180)
    if name == 'rot90_ccw': return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if name == 'fliplr': return np.flip(img, axis=1)
    if name == 'flipud': return np.flip(img, axis=0)
    if name == 'transpose': return np.transpose(img)
    if name == 'transpose_flip': return np.flip(np.transpose(img), axis=1)

results = {t: [] for t in transforms}
cnn_04_preds = {}
cnn_01_preds = {}
cnn_02_preds = {}

for f in files:
    img = cv2.imread(os.path.join(input_dir, f), cv2.IMREAD_GRAYSCALE)
    
    for t in transforms:
        t_img = apply_t(img, t)
        tensor = cv2.resize(t_img, (28, 28)).astype('float32') / 255.0
        tensor = tensor.reshape(1, 28, 28, 1)
        preds = model.predict(tensor, verbose=0)[0]
        
        conf = np.max(preds)
        pred = np.argmax(preds)
        results[t].append(conf)
        
        if f == 'cnn_04.png': cnn_04_preds[t] = pred
        if f == 'cnn_01.png': cnn_01_preds[t] = pred
        if f == 'cnn_02.png': cnn_02_preds[t] = pred

print(f"{'Transform':<20} | {'Avg Conf':<10} | {'cnn_01':<6} | {'cnn_02':<6} | {'cnn_04':<6}")
for t in transforms:
    avg = np.mean(results[t])
    print(f"{t:<20} | {avg*100:.2f}%     | {cnn_01_preds[t]:<6} | {cnn_02_preds[t]:<6} | {cnn_04_preds[t]:<6}")

