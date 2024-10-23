import os
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Load the YOLO model using OpenCV DNN
# This reads the pre-trained weights and configuration file for the custom YOLO model.
print("Loading YOLO model...")
net = cv2.dnn.readNet("./Yolo/yolov3_custom_last.weights", "./Yolo/yolov3_custom.cfg")
print("YOLO model loaded successfully.\n")

# Load class labels
# This reads the class labels from a text file ('obj.names') and stores them in a list.
print("Loading class labels from './Yolo/obj.names'...")
classes = []
with open("./Yolo/obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
print(f"Class labels loaded: {classes}\n")

# Get YOLO output layer names
# Retrieves the names of the output layers from the YOLO network.
# This is necessary for processing the network's predictions.
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print("YOLO output layers determined.\n")

# Define a function to preprocess images for YOLO
# Converts an image to a format that the YOLO model expects.
# `blobFromImage` scales the image, resizes it, and normalizes it.
def preprocess_image(image):
    print("Preprocessing image for YOLO...")
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    print("Image preprocessing complete.")
    return blob

# Function to load an image from the specified file path
# Reads an image from disk and retrieves its dimensions.
def load_image(file_path):
    print(f"Loading image: {file_path}")
    image = cv2.imread(file_path)
    height, width = image.shape[:2]
    print(f"Image loaded. Dimensions: {width}x{height}")
    return image, width, height

# Mapping of folder names to class indices
# Defines a dictionary to map folder names to their corresponding class indices.
# This is used to assign correct labels to images based on the folder they are located in.
folder_to_class = {'Draw': 0, 'Erase': 1, 'Stop': 2}
print(f"Folder-to-class mapping: {folder_to_class}\n")

# Load images and corresponding labels
# This function scans through the dataset directory, collects all image paths, and assigns labels based on the folder name.
def load_dataset(dataset_dir):
    print(f"Loading dataset from: {dataset_dir}")
    image_paths = []
    labels = []

    # Iterate over each class folder and collect image paths and their labels
    for label_name, class_id in folder_to_class.items():
        folder_path = os.path.join(dataset_dir, label_name)
        print(f"Processing folder: {folder_path}")
        for image_name in os.listdir(folder_path):
            image_paths.append(os.path.join(folder_path, image_name))
            labels.append(class_id)

    print(f"Total images loaded: {len(image_paths)}\n")
    return image_paths, labels

# Evaluation function
# This function processes each image using the YOLO model to predict the class and compares it to the true labels.
# Metrics like precision, recall, and accuracy are calculated to evaluate the model's performance.
def evaluate_yolo_model(image_paths, true_labels):
    print("Starting YOLO model evaluation...\n")
    all_predictions = []
    all_ground_truth = []

    # Loop through each image in the dataset for evaluation
    for image_path, true_label in zip(image_paths, true_labels):
        # Load and prepare the image
        print(f"\nEvaluating image: {image_path}")
        image, width, height = load_image(image_path)
        blob = preprocess_image(image)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        # Process the YOLO output
        # Loop through the detections made by YOLO, extract the class with the highest score (confidence), and save it.
        print("Processing YOLO outputs...")
        for out in outputs:
            for detection in out:
                scores = detection[5:]  # class scores start after the first 5 values
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Only consider predictions with confidence > 0.5
                if confidence > 0.5:
                    # Object detected
                    print(f"Detected class: {classes[class_id]} with confidence: {confidence:.2f}")
                    class_ids.append(class_id)

        # Determine the predicted class based on the most frequently detected class ID
        if len(class_ids) > 0:
            predicted_class = max(set(class_ids), key=class_ids.count)
        else:
            predicted_class = -1  # Handle case where no object is detected

        all_predictions.append(predicted_class)
        all_ground_truth.append(true_label)

    # Calculate and print evaluation metrics
    print("\nCalculating evaluation metrics...")
    precision = precision_score(all_ground_truth, all_predictions, average='macro', zero_division=1)
    recall = recall_score(all_ground_truth, all_predictions, average='macro', zero_division=1)
    accuracy = accuracy_score(all_ground_truth, all_predictions)

    # Display the results
    print(f"\nEvaluation Results:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

# Load dataset and evaluate
# This initiates the dataset loading and model evaluation.
dataset_dir = "../Test" #CHANGE ACCORDING TO THE FOLDER YOU WANT TO TRAIN WITH.
image_paths, true_labels = load_dataset(dataset_dir)
evaluate_yolo_model(image_paths, true_labels)
