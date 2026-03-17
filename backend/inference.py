import numpy as np
import logging
from tensorflow.keras.models import load_model
from PIL import Image
import io
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Model" / "digit_model.keras"

logger.info(f"Loading model from {MODEL_PATH}...")
try:
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise


def preprocess_image_bytes(image_bytes):
    """
    Preprocess raw image bytes for the CNN model.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28))

    img_array = np.array(image)

    # Invert colors (MNIST style: white on black)
    img_array = 255 - img_array

    # Normalize
    img_array = img_array / 255.0

    return img_array


def predict_from_matrix(matrix):
    """
    Predict digit from a 28x28 numpy array or list.
    """
    img_array = np.array(matrix, dtype=np.float32)
    
    # Normalize if necessary
    if img_array.max() > 1:
        img_array = img_array / 255.0
        
    img_array = img_array.reshape(1, 28, 28, 1)
    
    prediction = model.predict(img_array)
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return digit, confidence


def predict_digit(image_bytes):
    """
    Main entry point for image byte prediction.
    """
    processed = preprocess_image_bytes(image_bytes)
    return predict_from_matrix(processed)