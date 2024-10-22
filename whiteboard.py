import cv2
import torch
from Gesture_Model_MobileNetV2 import model, transform  # Import the trained model and transform from your existing file
from PIL import Image

# Define function to load model and prepare it for inference
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    return model, device

# Function to make predictions based on video frame input
def predict_gesture(frame, model, device):
    # Convert OpenCV frame (NumPy array) to PIL image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame)
    pil_image = pil_image.resize((224, 224))  # Ensure consistent size

    # Transform the PIL image as per training requirements
    frame_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(frame_tensor)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Function to run gesture detection
def run_gesture_detection():
    model, device = load_model()
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
            gesture_label = "Unknown Gesture"

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
