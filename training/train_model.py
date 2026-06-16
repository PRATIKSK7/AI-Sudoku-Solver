import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import ssl

# Fix macOS SSL certificate verification error for downloading MNIST
ssl._create_default_https_context = ssl._create_unverified_context

def build_model():
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def train():
    print("Loading Kaggle Digit Recognizer dataset from CSV...")
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv('datasets/digit-recognizer/train.csv')
    labels = df['label'].values
    images = df.drop('label', axis=1).values
    
    # Reshape and normalize
    images = images.reshape(images.shape[0], 28, 28, 1).astype('float32') / 255.0
    
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)

    # One hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15
    )
    datagen.fit(x_train)

    model = build_model()
    model.summary()

    os.makedirs('models', exist_ok=True)
    checkpoint = ModelCheckpoint('models/sudoku_digit_model.h5', 
                                 monitor='val_accuracy', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='max')
                                 
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=0.00001, verbose=1)

    batch_size = 64
    epochs = 20

    print("Training model...")
    history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        callbacks=[checkpoint, reduce_lr])
                        
    print("Training completed. Model saved to models/sudoku_digit_model.h5")

if __name__ == "__main__":
    train()
