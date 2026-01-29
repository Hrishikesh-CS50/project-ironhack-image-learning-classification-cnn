# model.py
from keras.models import Sequential
from keras.layers import (
    Dense, Conv2D, Flatten, Dropout,
    MaxPooling2D, Input, RandomFlip, Activation,
    RandomTranslation, RandomRotation, RandomZoom,
    GlobalAveragePooling2D, BatchNormalization)
from config import INPUT_SHAPE, NUM_CLASSES


def build_model():
    model = Sequential([
        Input(shape=INPUT_SHAPE),

        # Data Augmentation
        RandomFlip("horizontal"),
        RandomRotation(0.1), # Small rotation helps with orientation invariance
        RandomTranslation(0.1, 0.1), 
        
        #VGG-style architecture
        
        # Block 1 - Double Conv (32 filters)
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        # BLOCK 2: Double Conv (64 filters)
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # BLOCK 3: Double Conv (128 filters)
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        # Head
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model
