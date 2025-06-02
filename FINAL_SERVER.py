from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load model and mappings
checkpoint = torch.load("ulcer_classification_mobilenetv3.pth")
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
num_ftrs = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(num_ftrs, len(checkpoint['class_to_index']))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class_to_index = checkpoint['class_to_index']
index_to_class = {v: k for k, v in class_to_index.items()}

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Helper function to get the message
def get_ulcer_message(label):
    label = label.lower()
    if label == "grade0":
        return "You have no symptoms of ulcer."
    elif label == "grade1":
        return ("You have a Grade 1 ulcer. Please visit your local physician for treatment — "
                "this is manageable at the primary care level.")
    elif label == "grade2":
        return ("You have a Grade 2 ulcer. You must consult a diabetic foot specialist. "
                "Further diagnosis is needed to determine whether it is a neuropathic ulcer, "
                "neuroischemic ulcer, or ischemic ulcer. Treatment depends on the type.")
    elif label == "grade3":
        return ("You have a Grade 3 ulcer — a deep ulcer, possibly with osteomyelitis. "
                "You should visit a specialized foot care center immediately.")
    elif label == "grade4":
        return ("You have a Grade 4 ulcer. Your foot is at serious risk, and there is a possibility of amputation. "
                "Go to a hospital and consult a diabetic foot specialist immediately.")
    elif label == "grade5":
        return ("You have a Grade 5 ulcer. This is a very high-risk condition — it can lead to limb loss or even be life-threatening. "
                "Seek emergency care from a specialist hospital immediately.")
    else:
        return "Unknown grade. Please check the image or contact support."

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image found in request"}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)

    predicted_label = index_to_class[predicted.item()]
    confidence = probs[0][predicted.item()].item() * 100
    message = get_ulcer_message(predicted_label)

    return jsonify({
        # "predicted_label": predicted_label,
        "predicted_label": message,
        "confidence": confidence,
        "message": message
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
