import os

for split in ["train", "val"]:
    counts = {}
    label_dir = f"datasets/custom_dataset/labels/{split}"
    if not os.path.exists(label_dir): continue
    for f in os.listdir(label_dir):
        with open(os.path.join(label_dir, f), 'r') as fp:
            for line in fp:
                cls = line.split()[0]
                counts[cls] = counts.get(cls, 0) + 1
    print(f"{split} counts: {counts}")

