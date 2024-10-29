import cv2
import torch
from torchvision import models, transforms
from PIL import Image

# Define the same transformations as used during training
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Adjusted size for Inception v3
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Define function to load the pre-trained and saved model
def load_model(model_path="inception_v3_trained_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Inception v3 model
    model = models.inception_v3(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)  # Adjust output classes as needed (e.g., 3 classes here)
    model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, 3)  # Ensure auxiliary is also set

    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)  # Set the model to evaluation mode and move to device

    return model, device


# Function to make predictions based on video frame input
def predict_gesture(frame, model, device):
    # Convert OpenCV frame (NumPy array) to PIL image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame)
    pil_image = pil_image.resize((299, 299))  # Ensure consistent size for Inception v3

    # Transform the PIL image as per training requirements
    frame_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(frame_tensor)
        _, predicted = torch.max(output, 1)

    return predicted.item()


# Function to run gesture detection
def run_gesture_detection():
    model, device = load_model("inception_v3_trained_model.pth")  # Adjust the path if needed
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame to avoid mirror effect
        frame = cv2.flip(frame, 1)

        # Use the trained model to detect the gesture
        gesture = predict_gesture(frame, model, device)

        # Show detected gesture on the screen
        if gesture == 0:
            gesture_label = "Draw Gesture"
        elif gesture == 1:
            gesture_label = "Erase Gesture"
        else:
            gesture_label = "Stop Gesture"

        # Display the label on the frame
        cv2.putText(frame, f"Detected: {gesture_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the video feed
        cv2.imshow("Gesture Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_gesture_detection()
