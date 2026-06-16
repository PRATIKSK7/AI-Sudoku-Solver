import cv2
import os

img_path = "datasets/custom_dataset/images/train/11_flip70.jpg"
img = cv2.imread(img_path)
print(f"Image shape: {img.shape}")
print(f"Max value: {img.max()}, Min value: {img.min()}")

