import cv2
import sys
import glob

def print_ascii(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return
    img = cv2.resize(img, (28, 28))
    print(f"--- {path} ---")
    for r in range(28):
        line = ""
        for c in range(28):
            line += "#" if img[r,c] > 128 else "."
        print(line)

for f in sorted(glob.glob("debug/emnist_raw_*.png"))[:2]:
    print_ascii(f)
for f in sorted(glob.glob("debug/emnist_trans_*.png"))[:2]:
    print_ascii(f)

