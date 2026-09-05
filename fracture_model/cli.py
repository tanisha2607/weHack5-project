import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fracture_resnet18.pth"
)

IMG_SIZE = 224

# Class order used by ImageFolder during training
CLASSES = ["fractured", "not fractured"]

# Same normalization used during training
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    # Create ResNet-18 architecture
    model = models.resnet18(weights=None)

    # Replace final layer exactly as during training
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    # Load saved weights
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    # Support both:
    # 1. state_dict-only .pth files
    # 2. checkpoint files containing "model_state_dict"

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

        # Use saved class names if available
        classes = checkpoint.get("classes", CLASSES)

    else:
        model.load_state_dict(checkpoint)
        classes = CLASSES

    model = model.to(device)
    model.eval()

    return model, classes


# ============================================================
# PREDICTION
# ============================================================

def predict_image(model, classes, image_path):

    try:
        # Open image
        image = Image.open(image_path)

        # Apply the exact evaluation preprocessing
        image_tensor = transform(image)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        # Move to CPU/GPU
        image_tensor = image_tensor.to(device)

        # Inference
        with torch.no_grad():

            outputs = model(image_tensor)

            probabilities = torch.softmax(outputs, dim=1)

            predicted_index = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = (
                probabilities[0][predicted_index].item() * 100
            )

        prediction = classes[predicted_index]

        return prediction, confidence

    except Exception:
        return None, None


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    # Check model file
    if not os.path.exists(MODEL_PATH):

        print()
        print("=" * 55)
        print("ERROR: Model file not found.")
        print(f"Please place '{MODEL_PATH}' in the same folder")
        print("as this Python script.")
        print("=" * 55)
        print()

        return

    # Load model once at startup
    print()
    print("=" * 55)
    print("        BONE FRACTURE X-RAY CLASSIFIER")
    print("=" * 55)
    print()
    print("Loading model...")

    try:
        model, classes = load_model()

    except Exception as e:

        print()
        print("ERROR: Could not load the model.")
        print("Please check that the .pth file matches")
        print("the ResNet-18 architecture.")
        print()
        return

    print("Model loaded successfully.")
    print(f"Device: {device}")
    print()
    print("-" * 55)
    print("Enter the path of an X-ray image.")
    print("Type 'q' or 'quit' to exit.")
    print("-" * 55)


    # ========================================================
    # CONTINUOUS PREDICTION LOOP
    # ========================================================

    while True:

        print()
        image_path = input("X-ray path > ").strip()

        # Quit
        if image_path.lower() in ["q", "quit", "exit"]:
            print()
            print("Thank you. Exiting...")
            print()
            break

        # Remove quotes if user pastes a quoted path
        image_path = image_path.strip('"').strip("'")

        # Check whether file exists
        if not os.path.isfile(image_path):

            print()
            print("❌ File not found.")
            print("Please check the path and try again.")
            continue

        # Check whether it is a valid image
        try:
            with Image.open(image_path) as img:
                img.verify()

        except Exception:

            print()
            print("❌ Invalid image file.")
            print("Please provide a valid X-ray image.")
            continue

        # Run prediction
        prediction, confidence = predict_image(
            model,
            classes,
            image_path
        )

        if prediction is None:

            print()
            print("❌ Could not process this image.")
            print("Please try another image.")
            continue

        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        print()
        print("=" * 55)
        print("                    RESULT")
        print("=" * 55)

        if prediction.lower() == "fractured":
            display_prediction = "FRACTURED"
        else:
            display_prediction = "NOT FRACTURED"

        print()
        print(f"  Prediction : {display_prediction}")
        print(f"  Confidence : {confidence:.2f}%")
        print()
        print("=" * 55)


if __name__ == "__main__":
    main()