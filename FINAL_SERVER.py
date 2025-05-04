# from flask import Flask, request, jsonify
# import torch
# from torchvision import models, transforms
# from PIL import Image
# import io

# # Initialize Flask app and model
# app = Flask(__name__)

# # Load the model (same code you provided earlier)
# checkpoint = torch.load("ulcer_classification_mobilenetv3.pth")
# model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
# num_ftrs = model.classifier[3].in_features
# model.classifier[3] = torch.nn.Linear(num_ftrs, len(checkpoint['class_to_index']))
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

# class_to_index = checkpoint['class_to_index']
# index_to_class = {v: k for k, v in class_to_index.items()}

# # Define preprocessing transformation
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Define prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image found in request"}), 400

#     # Get the image file from the request
#     image_file = request.files['image']
#     image = Image.open(io.BytesIO(image_file.read())).convert('RGB')  # Convert to RGB

#     # Preprocess image
#     image_tensor = transform(image).unsqueeze(0)

#     # Run inference
#     with torch.no_grad():
#         output = model(image_tensor)
#         probs = torch.nn.functional.softmax(output, dim=1)
#         _, predicted = torch.max(output, 1)

#     # Get the predicted label and confidence
#     predicted_label = index_to_class[predicted.item()]
#     confidence = probs[0][predicted.item()].item() * 100

#     return jsonify({"predicted_label": predicted_label, "confidence": confidence})

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000)


from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

checkpoint = torch.load("ulcer_classification_mobilenetv3.pth")
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
num_ftrs = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(num_ftrs, len(checkpoint['class_to_index']))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class_to_index = checkpoint['class_to_index']
index_to_class = {v: k for k, v in class_to_index.items()}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

    return jsonify({"predicted_label": predicted_label, "confidence": confidence})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
