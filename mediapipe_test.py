import cv2 
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
from mediapipe_model import get_prediction

folders = ["Draw", "Stop", "Erase"]
ground_truth = []
predicted = []
def main():
    for folder in folders:
        this_path = "Test/" + folder + "/"
        for filename in os.listdir(this_path):
            frame = cv2.imread(this_path+filename)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            predicted.append(get_prediction(rgb_frame))
            ground_truth.append(folder)

    
    # Calculate accuracy
    accuracy = accuracy_score(ground_truth, predicted)
    print("Accuracy:", accuracy)

    # Calculate precision
    precision = precision_score(ground_truth, predicted, average='macro') # You can also use 'micro' or 'weighted'
    print("Precision:", precision)

    # Calculate recall
    recall = recall_score(ground_truth, predicted, average='macro') # You can also use 'micro' or 'weighted'
    print("Recall:", recall)

if __name__ == "__main__":
    main()
