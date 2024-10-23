import torch
import torchvision
import cv2
import numpy as np
from torchvision.transforms import functional as F

# Load a pre-trained RetinaNet model
model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")
model.eval()  # Set the model to evaluation mode

# Move the model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Capture video from webcam or video file
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path

# Function to draw bounding boxes and labels on the frame
def draw_detections(frame, predictions, threshold=0.5):
    boxes = predictions[0]['boxes'].cpu().detach().numpy()
    labels = predictions[0]['labels'].cpu().detach().numpy()
    scores = predictions[0]['scores'].cpu().detach().numpy()

    for i, box in enumerate(boxes):
        if scores[i] > threshold:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 255, 0), thickness=2)
            label = f"Label: {labels[i]}, Score: {scores[i]:.2f}"
            cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

# Read frames from the video in a loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB and then to a tensor
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = F.to_tensor(frame_rgb).unsqueeze(0).to(device)  # Add batch dimension

    # Perform object detection
    with torch.no_grad():
        predictions = model(frame_tensor)

    # Draw detections on the frame
    frame_with_detections = draw_detections(frame, predictions)

    # Display the frame with detections
    cv2.imshow('RetinaNet Object Detection', frame_with_detections)

    # Press 'q' to quit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
