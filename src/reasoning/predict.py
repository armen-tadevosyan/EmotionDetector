"""
Script to make emotion predictions
"""
import argparse

import cv2
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

def predict(image_path: str, model_path: str, focus_face: bool = True) -> str:
    """
    Makes a prediction on an image.
    :param image_path: path to image
    :param model_path: path to model
    :param focus_face: If True, detects and crops to the largest face before inference.
    :Returns: the prediction
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(model_path, device)

    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if focus_face:
        # Zoom in on a detected face
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        if len(faces) == 0:
            print("No face detected — running inference on full image.")
        else:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            print(f"Face detected at x={x}, y={y}, w={w}, h={h}")
            image_rgb = cv2.cvtColor(image_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    tensor = predict_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
    label = EMOTION_LABELS[predicted_idx.item()]
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
    parser.add_argument("--focus_face", type=bool, default=True, help="Focus on a face in an image or not")
    args = parser.parse_args()
    predict(args.image, args.model, args.focus_face)