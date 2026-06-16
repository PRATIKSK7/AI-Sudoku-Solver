import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('models/sudoku_digit_model.h5')

def evaluate_img(path, true_label):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return
    
    transforms = {
        'normal': img,
        'rot90_cw': cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        'rot180': cv2.rotate(img, cv2.ROTATE_180),
        'rot90_ccw': cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
        'fliplr': np.flip(img, axis=1),
        'flipud': np.flip(img, axis=0),
        'transpose': np.transpose(img),
        'transpose+fliplr': np.flip(np.transpose(img), axis=1)
    }
    
    print(f"\nEvaluating {path} (Expected: {true_label})")
    
    for name, t_img in transforms.items():
        tensor = cv2.resize(t_img, (28, 28)).astype('float32') / 255.0
        tensor = tensor.reshape(1, 28, 28, 1)
        preds = model.predict(tensor, verbose=0)[0]
        pred_label = np.argmax(preds)
        conf = preds[pred_label]
        
        match = "YES" if pred_label == true_label else "NO"
        print(f"  {name:<16}: Pred={pred_label} Conf={conf*100:.2f}% | Match: {match}")

evaluate_img('debug/cnn_input/cnn_01.png', 6)
evaluate_img('debug/cnn_input/cnn_02.png', 1)
evaluate_img('debug/cnn_input/cnn_04.png', 5)
