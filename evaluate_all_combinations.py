import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('models/sudoku_digit_model.h5')

img1 = cv2.imread('debug/cnn_input/cnn_01.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('debug/cnn_input/cnn_02.png', cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread('debug/cnn_input/cnn_04.png', cv2.IMREAD_GRAYSCALE)

def get_all_transforms(img):
    return {
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

t1 = get_all_transforms(img1)
t2 = get_all_transforms(img2)
t4 = get_all_transforms(img4)

for k in t1.keys():
    t_img1 = cv2.resize(t1[k], (28, 28)).astype('float32') / 255.0
    t_img2 = cv2.resize(t2[k], (28, 28)).astype('float32') / 255.0
    t_img4 = cv2.resize(t4[k], (28, 28)).astype('float32') / 255.0
    
    p1 = np.argmax(model.predict(t_img1.reshape(1, 28, 28, 1), verbose=0)[0])
    p2 = np.argmax(model.predict(t_img2.reshape(1, 28, 28, 1), verbose=0)[0])
    p4 = np.argmax(model.predict(t_img4.reshape(1, 28, 28, 1), verbose=0)[0])
    
    if p1 == 6 and p2 == 1 and p4 == 5:
        print(f"BINGO! {k} gives 6, 1, 5!")
    else:
        print(f"{k}: {p1}, {p2}, {p4}")

