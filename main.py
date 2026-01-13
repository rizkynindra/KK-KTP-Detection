from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load model
try:
    model = YOLO("model/best.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class ImageInput(BaseModel):
    image: str  # Base64 string

@app.get("/")
def read_root():
    return {"message": "KTP/KK Detection API is running"}

CONFIDENCE_THRESHOLD = 0.90

@app.post("/predict")
def predict(input_data: ImageInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Decode base64
        image_data = base64.b64decode(input_data.image)
        image = Image.open(io.BytesIO(image_data))
        results = model(image)
        
        # Process results
        predictions = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf >= CONFIDENCE_THRESHOLD:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    predictions.append({
                        "class": cls_name,
                        "confidence": conf,
                        "box": box.xyxy[0].tolist()
                    })

        if not predictions:
            return {"message": "No detection found", "predictions": []}

        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
