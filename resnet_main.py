import os
import time
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import kagglehub
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from collections import Counter

# Ensure logs are flushed immediately so epoch stats appear live.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


# 1. Download/Path Handling
data_path = kagglehub.dataset_download("subhaditya/fer2013plus")


# 2. Configuration & Hyperparameters
if torch.cuda.is_available():
    device_type = "cuda"
elif torch.backends.mps.is_available():
    device_type = "mps"
else:
    device_type = "cpu"

DEVICE = torch.device(device_type)
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 10
NUM_WORKERS = os.cpu_count() if os.cpu_count() else 0
MODEL_PATH = "ferplus_resnet18.pth"
EXCLUDED_CLASS = "contempt"


# Keep 3 channels to fully use pretrained ResNet weights.
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class FilteredImageFolder(Dataset):
    def __init__(self, root, excluded_class, transform):
        self.base = datasets.ImageFolder(root=root, transform=transform)
        self.excluded_class = excluded_class

        if excluded_class not in self.base.class_to_idx:
            raise ValueError(f"Excluded class '{excluded_class}' not found in dataset classes: {self.base.classes}")

        excluded_idx = self.base.class_to_idx[excluded_class]
        self.classes = [name for name in self.base.classes if name != excluded_class]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        old_to_new_idx = {}
        for class_name, old_idx in self.base.class_to_idx.items():
            if class_name == excluded_class:
                continue
            old_to_new_idx[old_idx] = self.class_to_idx[class_name]

        self.samples = []
        for path, old_idx in self.base.samples:
            if old_idx == excluded_idx:
                continue
            self.samples.append((path, old_to_new_idx[old_idx]))

        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        sample = self.base.loader(path)
        if self.base.transform is not None:
            sample = self.base.transform(sample)
        return sample, label


def build_loaders(train_dir, test_dir):
    train_set = FilteredImageFolder(root=train_dir, excluded_class=EXCLUDED_CLASS, transform=transform)
    test_set = FilteredImageFolder(root=test_dir, excluded_class=EXCLUDED_CLASS, transform=transform)

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
    }

    # persistent_workers only works when worker count is > 0.
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["multiprocessing_context"] = "forkserver"

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_set, train_loader, test_loader


def describe_dataset(train_set, test_loader):
    test_set = test_loader.dataset
    print(f"Train images: {len(train_set)}", flush=True)
    print(f"Test images: {len(test_set)}", flush=True)
    print(f"Classes ({len(train_set.classes)}): {train_set.classes}", flush=True)

    train_targets = train_set.targets
    test_targets = test_set.targets
    train_counts = Counter(train_targets)
    test_counts = Counter(test_targets)

    print("Class distribution:", flush=True)
    for class_idx, class_name in enumerate(train_set.classes):
        print(
            f"  {class_name}: train={train_counts.get(class_idx, 0)} "
            f"test={test_counts.get(class_idx, 0)}",
            flush=True,
        )

    print("Sample training files:", flush=True)
    for path, label_idx in train_set.samples[:5]:
        print(f"  {path} -> {train_set.classes[label_idx]}", flush=True)


def build_model(num_classes, pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def locate_split_dirs():
    train_dir = None
    test_dir = None
    for root, dirs, _ in os.walk(data_path):
        if "train" in dirs:
            train_dir = os.path.join(root, "train")
        if "test" in dirs:
            test_dir = os.path.join(root, "test")

    if not train_dir or not test_dir:
        raise FileNotFoundError(f"Could not locate 'train' or 'test' directories in {data_path}")
    return train_dir, test_dir


def load_data():
    train_dir, test_dir = locate_split_dirs()
    print(f"Dataset cache path: {data_path}", flush=True)
    print(f"Loading data from: {train_dir}", flush=True)
    print(f"Validation data from: {test_dir}", flush=True)
    train_set, train_loader, test_loader = build_loaders(train_dir, test_dir)
    if train_set.class_to_idx != test_loader.dataset.class_to_idx:
        raise ValueError("Train/Test class mappings do not match after filtering.")
    print(f"Excluded class: {EXCLUDED_CLASS}", flush=True)
    describe_dataset(train_set, test_loader)
    return train_set, train_loader, test_loader


def evaluate_saved_weights(test_loader, num_classes):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Saved weights not found at: {MODEL_PATH}")

    model = build_model(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
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
    print(f"Saved model accuracy on test set: {accuracy:.2f}%", flush=True)


def run_training(num_epochs=EPOCHS, save_model=True):
    train_set, train_loader, test_loader = load_data()

    model = build_model(num_classes=len(train_set.classes), pretrained=True)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting ResNet18 training on {DEVICE}...", flush=True)
    total_elapsed = 0.0
    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()

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

        epoch_duration = time.perf_counter() - epoch_start
        accuracy = 100 * correct / total
        total_elapsed += epoch_duration
        avg_epoch_time = total_elapsed / (epoch + 1)
        eta_seconds = avg_epoch_time * (num_epochs - (epoch + 1))
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {train_loss / len(train_loader):.4f} "
            f"- Val Acc: {accuracy:.2f}% - Time: {epoch_duration:.1f}s - ETA: {eta_seconds / 60:.1f}m",
            flush=True,
        )

    if save_model:
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Training complete! Model weights saved to {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FERPlus ResNet18 trainer/evaluator")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--inspect-only", action="store_true", help="Print dataset details only")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Epoch count for --train")
    args = parser.parse_args()

    if args.train:
        run_training(num_epochs=args.epochs)
    else:
        train_set, _, test_loader = load_data()
        if args.inspect_only:
            print("Inspection complete. Skipping training/evaluation.", flush=True)
        else:
            evaluate_saved_weights(test_loader, num_classes=len(train_set.classes))
