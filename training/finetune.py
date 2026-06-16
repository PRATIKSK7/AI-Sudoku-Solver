import cv2
import numpy as np
import tensorflow as tf
import os

model = tf.keras.models.load_model('models/sudoku_digit_model.h5')

X_train = []
Y_train = []

labels = {
    'cnn_00.png': 7,
    'cnn_01.png': 6,
    'cnn_02.png': 1,
    'cnn_04.png': 5,
    'cnn_05.png': 2,
    'cnn_06.png': 8,
    'cnn_07.png': 4,
    'cnn_08.png': 1,
    'cnn_09.png': 1,
    'cnn_10.png': 9,
    'cnn_11.png': 2,
    'cnn_12.png': 8,
    'cnn_14.png': 7,
    'cnn_15.png': 1,
    'cnn_17.png': 1,
    'cnn_18.png': 7,
    'cnn_19.png': 1,
    'cnn_20.png': 8,
    'cnn_22.png': 3,
    'cnn_23.png': 1,
    'cnn_24.png': 2,
    'cnn_26.png': 2,
    'cnn_27.png': 2,
    'cnn_28.png': 1,
    'cnn_29.png': 2,
    'cnn_30.png': 2,
    'cnn_31.png': 1,
    'cnn_32.png': 2,
    'cnn_33.png': 1,
    'cnn_34.png': 4,
    'cnn_35.png': 8,
    'cnn_36.png': 3,
    'cnn_37.png': 3,
    'cnn_38.png': 3,
    'cnn_39.png': 2,
    'cnn_40.png': 2,
    'cnn_41.png': 3,
    'cnn_42.png': 2,
    'cnn_44.png': 2,
    'cnn_45.png': 2,
    'cnn_46.png': 3,
    'cnn_47.png': 2,
    'cnn_48.png': 2,
    'cnn_50.png': 2,
    'cnn_54.png': 1,
    'cnn_55.png': 3,
    'cnn_56.png': 1,
    'cnn_57.png': 1,
    'cnn_60.png': 1,
    'cnn_62.png': 4,
    'cnn_63.png': 1,
    'cnn_64.png': 1,
    'cnn_66.png': 4,
    'cnn_68.png': 9,
    'cnn_69.png': 6,
    'cnn_72.png': 3,
    'cnn_73.png': 2,
    'cnn_76.png': 2,
    'cnn_77.png': 2,
    'cnn_79.png': 4
}

input_dir = 'debug/cnn_input'
for f in os.listdir(input_dir):
    if f in labels:
        img = cv2.imread(os.path.join(input_dir, f), cv2.IMREAD_GRAYSCALE)
        
        # apply the permanent transformation we'll use in code
        img = np.transpose(img)
        
        tensor = cv2.resize(img, (28, 28)).astype('float32') / 255.0
        X_train.append(tensor)
        Y_train.append(labels[f])
        
        # augment slightly
        for _ in range(5):
            X_train.append(tensor)
            Y_train.append(labels[f])

X_train = np.array(X_train).reshape(-1, 28, 28, 1)
Y_train = tf.keras.utils.to_categorical(np.array(Y_train), 10)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=20, batch_size=4, verbose=1)

model.save('models/sudoku_digit_model.h5')
print("Model fine-tuned and saved.")

