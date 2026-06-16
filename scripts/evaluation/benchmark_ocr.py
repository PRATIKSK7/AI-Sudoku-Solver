import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re

def main():
    model_path = "models/sudoku_digit_model.h5"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    input_dir = "debug/cnn_input"
    if not os.path.exists(input_dir):
        print(f"No test images found in {input_dir}")
        return

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
    if not files:
        print("No png files found.")
        return

    print(f"\nBenchmarking {len(files)} images...")
    
    y_true = []
    y_pred = []
    confidences = []
    
    # Ground truth mapping based on manual inspection of earlier failing cells
    # Since we don't have true labels for all, we will use a regex if the filename indicates it
    # Or just log the outputs if no ground truth. For benchmarking, let's assume we can parse 
    # it or just output the diagnostics. The prompt says "load 100 random digit crops, predict, display accuracy".
    # I'll use a mocked ground truth generator for the purpose of the confusion matrix if needed,
    # or read from filename if labeled. Since they are just cnn_01.png, we'll just log predictions
    # and build a pseudo-CM.
    
    # Actually, to compute accuracy, we need labels. We'll prompt the user or just extract from filename if they are named like 'cnn_01_label_6.png'.
    # If no labels, we will just print them and skip accuracy/CM, or mock it for the script structure.
    
    labels_exist = any("label" in f for f in files)
    
    for f in files:
        path = os.path.join(input_dir, f)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        # Apply the exact same transpose used in ai/segmentation.py
        img = np.transpose(img)
        tensor = img.astype("float32") / 255.0
        tensor = tensor.reshape(1, 28, 28, 1)
        
        preds = model.predict(tensor, verbose=0)[0]
        
        # Get top 3 classes
        ranked = sorted(enumerate(preds), key=lambda x: -x[1])
        top3 = [(int(i), float(p)) for i, p in ranked[:3]]
        
        digit = top3[0][0]
        conf = top3[0][1]
        
        confidences.append(conf)
        y_pred.append(digit)
        
        top3_str = ", ".join([f"{cls}({p*100:.1f}%)" for cls, p in top3])
        print(f"File: {f:<20} Predicted: {digit}  Confidence: {conf*100:.2f}% | Top 3: {top3_str}")
        
        # If labeled, parse it: e.g., cnn_01_label_6.png
        if labels_exist:
            match = re.search(r'label_(\d)', f)
            if match:
                y_true.append(int(match.group(1)))
            else:
                y_true.append(-1)
                
    avg_conf = np.mean(confidences)
    print(f"\nAverage Confidence: {avg_conf*100:.2f}%")
    
    if labels_exist and all(y != -1 for y in y_true):
        acc = np.mean(np.array(y_true) == np.array(y_pred))
        print(f"OCR Accuracy: {acc*100:.2f}%")
        
        cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('OCR Confusion Matrix')
        plt.savefig('debug/confusion_matrix.png')
        print("Saved confusion matrix to debug/confusion_matrix.png")
    else:
        print("\nNote: Filenames do not contain '_label_X', skipping accuracy and confusion matrix calculation.")

if __name__ == "__main__":
    main()
