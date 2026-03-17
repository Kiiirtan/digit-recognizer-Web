import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from backend.inference import predict_digit

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Digit Recognizer API")

# Configure CORS
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(data: dict):
    try:
        if "image" not in data:
            logger.error("Request missing 'image' field")
            raise ValueError("Missing 'image' field")

        image_data = data["image"]
        
        # Basic validation of input shape
        if not isinstance(image_data, list) or len(image_data) != 28:
            logger.error(f"Invalid input shape: {len(image_data) if isinstance(image_data, list) else type(image_data)}")
            raise ValueError("Input must be a 28x28 array")

        # Convert to numpy array and normalize if necessary
        image = np.array(image_data, dtype=np.float32)
        
        if image.shape != (28, 28):
            logger.error(f"Invalid image shape after conversion: {image.shape}")
            raise ValueError("Input must be a 28x28 array")

        if image.max() > 1:
            image = image / 255.0

        # Use the inference helper
        from backend.inference import predict_from_matrix
        
        digit, confidence = predict_from_matrix(image)
        
        logger.info(f"Predicted digit: {digit} with confidence: {confidence:.4f}")

        return {
            "digit": digit,
            "confidence": round(confidence, 4)
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="Internal server error")
#venv\Scripts\activate        
#uvicorn backend.app:app --reload