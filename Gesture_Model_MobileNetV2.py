import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder("Train/", transform=transform)
val_data = datasets.ImageFolder("Val/", transform=transform)
test_data = datasets.ImageFolder("Test/", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


#Load pre-trained model (Transfer Learning)
model = models.mobilenet_v2(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.last_channel, len(train_data.classes))
)

#Defining Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

def train_model(m, c, o, t, v, epochs=10):
    # Determine if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the appropriate device (GPU or CPU)

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
            optimizer.zero_grad()  # Reset gradients to zero

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update model parameters

            running_loss += loss.item()  # Accumulate the loss

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient computation for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Calculate loss
                val_loss += loss.item()  # Accumulate validation loss

                _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
                total += labels.size(0)  # Count total number of labels
                correct += (predicted == labels).sum().item()  # Count correct predictions

        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {running_loss/len(train_loader)}, "  # Average training loss
              f"Val Loss: {val_loss/len(val_loader)}, "  # Average validation loss
              f"Accuracy: {100 * correct / total:.2f}%")  # Validation accuracy

train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10)

# Testing function
def model_evaluation(m, t):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate accuracy
    accuracy = 100 * (torch.tensor(all_predictions) == torch.tensor(all_labels)).sum().item() / len(all_labels)
    print(f"\nTest Accuracy: {accuracy:.2f}%")

    # Calculate Precision and Recall (macro-averaged)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

# Test the model
model_evaluation(model, test_loader)