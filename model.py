# model.py
from keras.models import Sequential
from keras.layers import (
    Dense, Conv2D, Flatten,
    MaxPooling2D, Input, RandomFlip, Activation,
    RandomTranslation, RandomRotation, RandomZoom,
    GlobalAveragePooling2D, BatchNormalization)
from config import INPUT_SHAPE, NUM_CLASSES


def build_model():
    model = Sequential([
        Input(shape=INPUT_SHAPE),

        # Data Augmentation
        RandomFlip("horizontal"),

        Conv2D(32, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        
        GlobalAveragePooling2D(),

        Dense(128, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax"),
    ])
    return model
