# resnet_predict.py -- loads Rahul's trained ResNet18 model and runs inference on a single image
# Used by game.py so the game can show the AI's own prediction as feedback after each trial

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# same label order as FilteredImageFolder in resnet_main.py -- alphabetical, contempt excluded
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# same transform used at training time, but without the random flip (we don't want to augment at inference)
resnet_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # FER+ images are grayscale, ResNet needs 3 channels
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # ImageNet stats
])


def load_resnet_model(model_path: str, device: torch.device) -> nn.Module:
    # builds a ResNet18 with our custom output layer (7 classes) and loads the saved weights
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(EMOTION_LABELS))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


def predict_image(image_path: str, model: nn.Module, device: torch.device) -> tuple:
    # runs the model on a single image file
    # returns: (predicted label, confidence as a float 0-1, full probability array)
    img = Image.open(image_path).convert("RGB")
    tensor = resnet_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        confidence, idx = torch.max(probs, dim=0)

    return EMOTION_LABELS[idx.item()], confidence.item(), probs.cpu().numpy()
