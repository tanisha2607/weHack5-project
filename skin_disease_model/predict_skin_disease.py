#!/usr/bin/env python3
"""
Skin disease classifier — just run this file directly.

    python predict_skin_disease.py

It will:
  1. Auto-load the model (best_model_final.keras) and class names
     (class_names.json) from the SAME FOLDER as this script.
  2. Ask you to type/paste the path to an image.
  3. Print the prediction and confidence.
  4. Loop, so you can test multiple images without reloading the model.

Requires (one-time setup):
    pip install tensorflow pillow numpy

Make sure these three files are in the same folder:
    predict_skin_disease.py
    best_model_final.keras
    class_names.json
"""

import json
import sys
from pathlib import Path

import numpy as np

# --- Fixed config: everything lives next to this script ---
SCRIPT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = SCRIPT_DIR / "best_model_final.weights.h5"
CLASSES_PATH = SCRIPT_DIR / "class_names.json"
IMG_SIZE = (224, 224)


def load_class_names(path: Path) -> list:
    with open(path, "r") as f:
        return json.load(f)


def build_model(num_classes: int):
    """
    Rebuilds the exact architecture used at training time.
    We rebuild in code + load_weights() instead of load_model() on the
    full .keras file, because saving/loading the full model config across
    different TensorFlow/Keras versions (e.g. Colab -> local Windows) can
    fail during deserialization. Weights-only loading sidesteps that.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetB0

    base_model = EfficientNetB0(
        include_top=False,
        weights=None,  # weights come from our .h5 file, not ImageNet, at load time
        input_shape=(224, 224, 3),
        pooling="avg"
    )

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    return models.Model(inputs, outputs)


def preprocess_image(image_path: str):
    """Load an image and preprocess it exactly as done at training time."""
    import tensorflow as tf

    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)  # add batch dimension
    # Model has EfficientNet preprocessing baked in as its first layer,
    # so raw 0-255 pixel values are passed in here — no manual rescaling.
    return arr


def predict(model, class_names, image_path: str, top_k: int = 3) -> dict:
    """
    Returns:
        {"prediction": str, "confidence": float, "top_k": [{"label", "confidence"}, ...]}
    """
    arr = preprocess_image(image_path)
    probs = model.predict(arr, verbose=0)[0]

    top_idx = int(np.argmax(probs))
    top_indices = np.argsort(probs)[::-1][:top_k]

    return {
        "prediction": class_names[top_idx],
        "confidence": float(probs[top_idx]),
        "top_k": [
            {"label": class_names[i], "confidence": float(probs[i])}
            for i in top_indices
        ],
    }


def main():
    if not WEIGHTS_PATH.exists():
        print(f"Error: weights file not found at {WEIGHTS_PATH}", file=sys.stderr)
        print("Make sure best_model_final.weights.h5 is in the same folder as this script.", file=sys.stderr)
        sys.exit(1)
    if not CLASSES_PATH.exists():
        print(f"Error: class names file not found at {CLASSES_PATH}", file=sys.stderr)
        print("Make sure class_names.json is in the same folder as this script.", file=sys.stderr)
        sys.exit(1)

    print("Loading model (this happens once)...")
    class_names = load_class_names(CLASSES_PATH)
    model = build_model(num_classes=len(class_names))
    model.load_weights(str(WEIGHTS_PATH))
    print(f"Model loaded. {len(class_names)} classes available.\n")

    # Loop so the model stays loaded in memory across multiple test images
    while True:
        image_path = input("Enter image path (or 'q' to quit): ").strip().strip('"')

        if image_path.lower() in ("q", "quit", "exit"):
            print("Exiting.")
            break

        if not image_path:
            continue

        if not Path(image_path).exists():
            print(f"  File not found: {image_path}\n")
            continue

        try:
            result = predict(model, class_names, image_path, top_k=3)
        except Exception as e:
            print(f"  Error running inference: {e}\n")
            continue

        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("Top 3:")
        for item in result["top_k"]:
            print(f"  - {item['label']}: {item['confidence']:.2%}")
        print()


if __name__ == "__main__":
    main()
