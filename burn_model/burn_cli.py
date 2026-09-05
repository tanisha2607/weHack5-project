import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)

# IMPORTANT:
# Keep this order exactly the same as during training.
CLASS_NAMES = [
    "first_degree",
    "second_degree",
    "third_degree"
]

WEIGHTS_PATH = "burnModel.weights.h5"


# ============================================================
# DATA AUGMENTATION
# Same architecture used during training
# ============================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.20),
    tf.keras.layers.RandomTranslation(
        height_factor=0.10,
        width_factor=0.10
    ),
    tf.keras.layers.RandomContrast(0.20),
    tf.keras.layers.RandomBrightness(0.20)
], name="data_augmentation")


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )

    inputs = layers.Input(shape=(224, 224, 3))

    x = data_augmentation(inputs)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(
        len(CLASS_NAMES),
        activation="softmax"
    )(x)

    model = models.Model(inputs, outputs)

    return model


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading burn severity model...")

if not os.path.exists(WEIGHTS_PATH):
    print(f"ERROR: Could not find {WEIGHTS_PATH}")
    print("Make sure the weights file is in the same folder as this script.")
    exit()

model = build_model()

model.load_weights(WEIGHTS_PATH)

print("Model loaded successfully!")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_burn(image_path):

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Resize exactly as during training
    image = image.resize(IMG_SIZE)

    # Convert to NumPy
    image = np.array(image, dtype=np.float32)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Prediction
    probabilities = model.predict(
        image,
        verbose=0
    )[0]

    # Get highest probability
    predicted_index = np.argmax(probabilities)

    confidence = probabilities[predicted_index]

    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence


# ============================================================
# CLI
# ============================================================

print("\n========================================")
print("      BURN SEVERITY CLASSIFIER")
print("========================================")

while True:

    image_path = input(
        "\nEnter image path (or type 'exit' to quit):\n> "
    ).strip().strip('"')

    if image_path.lower() == "exit":
        print("\nExiting...")
        break

    if not os.path.exists(image_path):
        print("\n❌ Image file not found.")
        print("Please check the path and try again.")
        continue

    try:

        print("\nProcessing image...")

        prediction, confidence = predict_burn(image_path)

        # Make output easier to read
        severity = prediction.replace("_", " ").upper()

        print("\n----------------------------------------")
        print("              RESULT")
        print("----------------------------------------")
        print(f"Prediction : {severity}")
        print(f"Confidence : {confidence * 100:.2f}%")
        print("----------------------------------------")

    except Exception as e:

        print("\n❌ Could not process the image.")
        print("Error:", e)