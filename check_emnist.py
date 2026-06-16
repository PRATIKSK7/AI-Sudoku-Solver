import pandas as pd
import numpy as np
import cv2

df = pd.read_csv("/Users/pratikskanoj/Downloads/sudoku_ocr_dataset/emnist-digits-train.csv", header=None, nrows=10)
y = df.iloc[:,0].values
X = df.iloc[:,1:].values.reshape(-1, 28, 28)

# The rotation in train_emnist.py
X_trans = np.transpose(X, (0, 2, 1))
X_trans = np.flip(X_trans, axis=2)

for i in range(5):
    cv2.imwrite(f"debug/emnist_raw_{i}_label_{y[i]}.png", X[i])
    cv2.imwrite(f"debug/emnist_trans_{i}_label_{y[i]}.png", X_trans[i])

