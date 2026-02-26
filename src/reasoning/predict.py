"""
Script to make emotion predictions
"""
import argparse
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from model import EmotionCNN

EMOTION_LABELS = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

predict_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_model(model_path: str, device: torch.device) -> EmotionCNN:
    """
    loads the model from the model path
    :Return: the EmotionCNN model
    """
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint['fc2.bias'].shape[0]
    model = EmotionCNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def predict(image_path: str, model_path: str) -> str:
    """
    Makes a prediction on an image
    :Returns: the prediction
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(model_path, device)
    image = Image.open(image_path).convert("RGB")
    tensor = predict_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
    label = EMOTION_LABELS[predicted_idx.item()]
    # Print all predictions for debugging purposes
    print(f"\nPrediction: {label} ({confidence.item()*100:.1f}% confidence)\n")
    for i, prob in enumerate(probabilities[0]):
        bar = '█' * int(prob.item() * 30)
        print(f"  {EMOTION_LABELS[i]:<12} {bar:<30} {prob.item()*100:.1f}%")
    return label

def predict_frame(face_img: np.ndarray, model: EmotionCNN, device: torch.device) -> tuple[str, float, np.ndarray]:
    """
    Makes a prediction on a cropped face
    :Returns: label, confidence, and prob array.
    """
    pil_img = Image.fromarray(face_img)
    tensor = predict_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        confidence, idx = torch.max(probs, dim=0)
    for i, prob in enumerate(probs):
        bar = '█' * int(prob.item() * 30)
        print(f"  {EMOTION_LABELS[i]:<12} {bar:<30} {prob.item()*100:.1f}%")
    print(f"---------------------------")
    return EMOTION_LABELS[idx.item()], confidence.item(), probs.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run emotion inference on a single image.")
    parser.add_argument("image", type=str, help="Path to the input image")
    parser.add_argument("--model", type=str, default="./models/ferplus_model.pth", help="Path to model checkpoint")
    args = parser.parse_args()
    predict(args.image, args.model)