import tensorflow as tf
import numpy as np
import cv2
import os

class DigitClassifier:
    def __init__(self, model_path='models/sudoku_digit_model.h5'):
        self.model = None

        # Resolve relative paths from the project root (parent of ai/)
        if not os.path.isabs(model_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, model_path)

        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"WARNING: Model not found at {model_path}")

    def predict(self, cell_img, threshold=0.60):
        """
        Predict the digit in a 28x28 grayscale image.
        Returns (digit, confidence, top3).
        """
        if self.model is None:
            return 0, 0.0, []

        if cell_img.shape != (28, 28):
            cell_img = cv2.resize(cell_img, (28, 28))

        tensor = cell_img.astype(np.float32) / 255.0
        tensor = tensor.reshape(1, 28, 28, 1)

        preds = self.model.predict(tensor, verbose=0)[0]
        ranked = sorted(enumerate(preds), key=lambda x: -x[1])
        top3 = [(int(i), float(p)) for i, p in ranked[:3]]

        primary_cls, primary_conf = top3[0]

        # FIX: 3 — Add dual-polarity prediction in DigitClassifier
        inverted = 1.0 - tensor  # flip black↔white
        inv_preds = self.model.predict(inverted, verbose=0)[0]
        inv_cls  = int(np.argmax(inv_preds))
        inv_conf = float(inv_preds[inv_cls])
        
        # Use inverted prediction if significantly more confident
        if inv_conf > primary_conf + 0.15 and inv_cls != 0:
            final_cls  = inv_cls
            final_conf = inv_conf
            print(f"[predict] Polarity flip improved confidence: "
                  f"{primary_conf:.3f} → {inv_conf:.3f} (digit {inv_cls})")
        else:
            final_cls  = primary_cls
            final_conf = primary_conf

        if final_cls != 0 and final_conf >= threshold:
            return final_cls, final_conf, top3
        else:
            return 0, final_conf, top3
