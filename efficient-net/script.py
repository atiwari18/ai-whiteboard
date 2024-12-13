import cv2
import numpy as np
from efficientnet_pytorch import EfficientNet
import torch
from torchvision import transforms

print("test")

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3  # Update this with the number of classes in your dataset

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)

model.load_state_dict(torch.load("efficient-net/efficientnet_model.pth"))  # Load your trained weights
model.to(device)
model.eval()  # Set the model to evaluation mode

# Class names (replace with your actual class names)
class_names = ['Draw', 'Erase', 'Stop']  # Update these names accordingly

# Function to preprocess frame for EfficientNet
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(frame)

# Start video capture
cap = cv2.VideoCapture(0)  # 0 for webcam or 'path/to/video.mp4' for a video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_frame = preprocess_frame(frame).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        # Perform inference
        output = model(input_frame)
        _, predicted = torch.max(output, 1)  # Get predicted class
        predicted_class = class_names[predicted.item()]  # Get the class name

    # Display the resulting frame with the predicted class
    cv2.putText(frame, f'Prediction: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Video Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
