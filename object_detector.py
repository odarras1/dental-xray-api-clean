from ultralytics import YOLO 
from flask import request, Flask, jsonify
from waitress import serve
from PIL import Image
import os
import requests

app = Flask(__name__)

def download_model():
    file_id = "15SpJ06hYGSpDsstc-YmVl3q5ziTCDpWp"  # Replace with your actual file ID
    destination = "best.pt"
    if not os.path.exists(destination):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        response.raise_for_status()  # Raise error if download failed
        with open(destination, "wb") as f:
            f.write(response.content)
        print("Model download complete.")

# Call this once at startup before loading the model
download_model()

# Load model once to avoid reloading on every request
model = YOLO("best.pt")

@app.route("/")
def root():
    with open("templates/index.html") as file:
        return file.read()

@app.route("/detect", methods=["POST"])
def detect():
    buf = request.files["image_file"]
    boxes = detect_objects_on_image(buf.stream)
    return jsonify(boxes)

def detect_objects_on_image(buf):
    results = model.predict(Image.open(buf))
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        prob_percentage = f"{prob * 100:.2f}%"
        output.append([x1, y1, x2, y2, result.names[class_id], prob_percentage])
    return output

serve(app, host='0.0.0.0', port=8080)
