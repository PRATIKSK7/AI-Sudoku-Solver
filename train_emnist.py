import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

print("Loading dataset...")

df = pd.read_csv(
    "/Users/pratikskanoj/Downloads/sudoku_ocr_dataset/emnist-digits-train.csv",
    header=None
)

y = df.iloc[:,0].values
X = df.iloc[:,1:].values

print("Dataset loaded:", X.shape)

X = X.reshape(-1,28,28)

# EMNIST rotation fix
X = np.transpose(X,(0,2,1))
X = np.flip(X,axis=2)

X = X.astype("float32")/255.0
X = X.reshape(-1,28,28,1)

y = to_categorical(y,10)

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42
)

print("Building model...")

model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(28,28,1)
    )
)

model.add(BatchNormalization())

model.add(
    Conv2D(
        32,
        (3,3),
        activation="relu"
    )
)

model.add(MaxPooling2D())

model.add(Dropout(0.25))

model.add(
    Conv2D(
        64,
        (3,3),
        activation="relu"
    )
)

model.add(BatchNormalization())

model.add(MaxPooling2D())

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256,activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(10,activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint = ModelCheckpoint(
    "models/sudoku_digit_model_new.h5",
    monitor="val_accuracy",
    save_best_only=True
)

print("Training started...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test,y_test),
    epochs=10,
    batch_size=128,
    callbacks=[checkpoint]
)

loss,acc = model.evaluate(X_test,y_test)

print("FINAL ACCURACY:",acc)
