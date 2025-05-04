import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np

# Load the saved MobileNetV3-Small model and labels
checkpoint = torch.load("ulcer_classification_mobilenetv3.pth")
print(checkpoint.keys())  # Debug: Check saved keys

# Initialize MobileNetV3-Small
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
num_ftrs = model.classifier[3].in_features  # Match the classifier layer from training
model.classifier[3] = torch.nn.Linear(num_ftrs, len(checkpoint['class_to_index']))

# Load trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load class mappings
class_to_index = checkpoint['class_to_index']
index_to_class = {v: k for k, v in class_to_index.items()}
print("Loaded class mapping:", class_to_index)  # Debug: Print class-to-index mapping

# Define preprocessing (same as training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to classify a new image
def classify_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)  # Get confidence scores
        _, predicted = torch.max(output, 1) 

    # Convert confidence scores to % format
    confidence_percentages = probs.numpy()[0] * 100  
    formatted_confidences = {index_to_class[i]: f"{confidence_percentages[i]:.2f}%" for i in range(len(confidence_percentages))}

    print(f"Confidence Scores (in %): {formatted_confidences}")  # Debug: Print model confidence scores
    return index_to_class[predicted.item()]

# Example usage
image_path = "test.jpg"
predicted_label = classify_image(image_path)
#? OUTPUT
print(f"Predicted Grade: {predicted_label}")