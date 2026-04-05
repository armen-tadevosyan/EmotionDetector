"""
Creates and trains a model
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import kagglehub
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import EmotionCNN 

# 1. Download/Path Handling
# Returns the path to the cached dataset on your MacBook
data_path = kagglehub.dataset_download("subhaditya/fer2013plus")

# 2. Configuration & Hyperparameters
if torch.cuda.is_available():
    device_type = "cuda"
elif torch.backends.mps.is_available():
    device_type = "mps" 
else:
    device_type = "cpu"

DEVICE = torch.device(device_type)
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 1e-4 
EPOCHS = 10
NUM_WORKERS = os.cpu_count() if os.cpu_count() else 0

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Center pixels around zero
])

def run_training():
    # 4. Dynamic Path Detection 
    train_dir = None
    test_dir = None
    
    for root, dirs, files in os.walk(data_path):
        if 'train' in dirs:
            train_dir = os.path.join(root, 'train')
        if 'test' in dirs:
            test_dir = os.path.join(root, 'test')

    if not train_dir or not test_dir:
        raise FileNotFoundError(f"Could not locate 'train' or 'test' directories in {data_path}")

    print(f"Loading data from: {train_dir}")
    train_set = datasets.ImageFolder(root=train_dir, transform=transform)
    test_set = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=NUM_WORKERS, pin_memory=True)

    # 5. Initialize Model, Optimizer, and Loss
    model = EmotionCNN(num_classes=len(train_set.classes)).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # 6. Training & Validation Loop
    print(f"Starting training on {DEVICE}...")
    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad(): 
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {accuracy:.2f}%")

    # 7. Save Model Weights
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'./models/model_{timestamp}.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Training complete! Model weights saved to {save_path}")

if __name__ == "__main__":
    run_training()