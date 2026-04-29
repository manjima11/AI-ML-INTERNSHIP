"""
CNN Model Training Script
Dataset: CIFAR-10 (10 classes of 32x32 color images)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# ── Class labels ──────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ── 1. Load & preprocess data ──────────────────────────────────────────────────
def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # Normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0
    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test  = keras.utils.to_categorical(y_test,  10)
    return (x_train, y_train), (x_test, y_test)


# ── 2. Build CNN architecture ──────────────────────────────────────────────────
def build_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Architecture:
        Input → [Conv2D → BN → ReLU] x2 → MaxPool → Dropout
               → [Conv2D → BN → ReLU] x2 → MaxPool → Dropout
               → [Conv2D → BN → ReLU] x2 → MaxPool → Dropout
               → Flatten → Dense(512) → BN → Dropout → Dense(10, softmax)
    """
    model = models.Sequential([
        # ── Block 1 ──
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Block 2 ──
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Block 3 ──
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Classifier head ──
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="CIFAR10_CNN")

    return model


# ── 3. Data augmentation ───────────────────────────────────────────────────────
def build_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name="augmentation")


# ── 4. Train ───────────────────────────────────────────────────────────────────
def train(epochs=30, batch_size=64):
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()
    aug   = build_augmentation()

    # Wrap: augment → model
    inputs  = keras.Input(shape=(32, 32, 3))
    x       = aug(inputs, training=True)
    outputs = model(x)
    full_model = keras.Model(inputs, outputs, name="AugCNN")

    full_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    full_model.summary()

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=4, verbose=1),
    ]

    history = full_model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    loss, acc = full_model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✅  Test accuracy: {acc*100:.2f}%  |  Test loss: {loss:.4f}")

    # Save
    full_model.save("cnn_model.h5")
    print("💾  Model saved → cnn_model.h5")

    # Plot
    _plot_history(history)
    return full_model, history


def _plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"],     label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy"); axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss"); axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    print("📊  Training plot saved → training_history.png")
    plt.close()


# ── 5. Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train(epochs=30, batch_size=64)