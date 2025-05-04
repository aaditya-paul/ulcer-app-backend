from flask import Flask, request, jsonify
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import io
from PIL import Image
from supabase import create_client, Client
import uuid  # For generating unique file names
from dotenv import load_dotenv
import os
load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)

# 🔹 Supabase Configuration
SUPABASE_URL = "https://ftyccnkwwmvccjwercxl.supabase.co"  # Replace with your Supabase URL
SUPABASE_KEY = os.getenv("SUPABASE_API_KEY")  # Replace with your Supabase Key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 🔹 Load Model
checkpoint = torch.load("ulcer_classification_mobilenetv3.pth", map_location=torch.device("cpu"))
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
num_ftrs = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(num_ftrs, len(checkpoint["class_to_index"]))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

class_to_index = checkpoint["class_to_index"]
index_to_class = {v: k for k, v in class_to_index.items()}

# 🔹 Image Processing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 🔹 API Endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        print("Missing image file")
        return jsonify({"error": "No file uploaded"}), 400

    if "email" not in request.form or "name" not in request.form or "number" not in request.form:
        print("Missing form data")
        return jsonify({"error": "Missing form data"}), 400

    name = request.form.get("name")
    email = request.form.get("email")
    number = request.form.get("number")
    file = request.files["file"]

    # Generate a unique filename
    unique_filename = f"{uuid.uuid4()}.jpg"

    try:
        # 🔹 Upload image to Supabase Storage
        image_bytes = file.read()
        response = supabase.storage.from_("ulcer-images").upload(unique_filename, image_bytes)

        # 🔹 Get the public URL of the uploaded image
        image_url = f"{SUPABASE_URL}/storage/v1/object/public/ulcer-images/{unique_filename}"

        # 🔹 Process image for model prediction
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(image)
            probs = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)

        predicted_label = index_to_class[predicted.item()]
        confidence = probs[0][predicted.item()].item() * 100

        # 🔹 Insert prediction data into Supabase
        data = {
            "name": name,
            "email": email,
            "number": number,
            "predicted_class": predicted_label,
            "confidence": round(confidence, 2),
            "image_url": image_url
        }

        supabase.table("ulcer_predictions").insert(data).execute()

        return jsonify({
            "predicted_class": predicted_label,
            "confidence": f"{confidence:.2f}%",
            "image_url": image_url
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


# 🔹 Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
