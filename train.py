"""Train a small CNN to classify fruit images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf


IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42


def build_model(num_classes: int) -> tf.keras.Model:
    """Create a compact CNN suitable for a small fruit dataset."""
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
        ],
        name="data_augmentation",
    )

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(*IMAGE_SIZE, 3)),
            data_augmentation,
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.40),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="fruit_cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_plot(history: tf.keras.callbacks.History, output_path: Path) -> None:
    """Save accuracy and loss curves for the internship report."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fruit CNN")
    parser.add_argument("--data-dir", default="data", help="Directory containing train/ and validation/")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = data_dir / "train"
    validation_dir = data_dir / "validation"
    if not train_dir.exists() or not validation_dir.exists():
        raise FileNotFoundError("Expected data/train and data/validation directories. See README.md.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, seed=SEED, shuffle=True
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        validation_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, seed=SEED, shuffle=False
    )
    class_names = train_ds.class_names
    if class_names != validation_ds.class_names:
        raise ValueError(f"Train and validation classes differ: {class_names} vs {validation_ds.class_names}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    validation_ds = validation_ds.cache().prefetch(autotune)
    model = build_model(len(class_names))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(output_dir / "fruit_cnn.keras", monitor="val_accuracy", save_best_only=True),
    ]
    history = model.fit(train_ds, validation_data=validation_ds, epochs=args.epochs, callbacks=callbacks)

    model.save(output_dir / "fruit_cnn.keras")
    (output_dir / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    save_training_plot(history, output_dir / "training_history.png")
    loss, accuracy = model.evaluate(validation_ds, verbose=0)
    print(f"Validation accuracy: {accuracy:.2%}")
    print(f"Saved model and class names to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
