import requests
import base64
import numpy as np
from PIL import Image
import io
import json

def create_dummy_image():
    # Create a random image (100x100)
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def test_predict():
    url = "http://localhost:8000/predict"
    img_b64 = create_dummy_image()
    payload = {"image": img_b64}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("Response Code:", response.status_code)
        print("Response Body:", json.dumps(response.json(), indent=2))
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        if e.response:
             print(e.response.text)

if __name__ == "__main__":
    test_predict()
