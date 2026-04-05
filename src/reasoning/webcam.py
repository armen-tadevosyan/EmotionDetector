"""
Enables webcam with predictions on each frame for testing
"""
import cv2
import torch
from predict import load_model, predict_frame


def run_webcam(model_path: str):
    """
    Runs the webcam and applies the chosen model to each frame
    :param model_path: the model path
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on {device}")
    model = load_model(model_path, device)
    # Uses haar cascades to find face
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        # Draws a frame and label around the face
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            label, confidence, probs = predict_frame(face_rgb, model, device)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence*100:.0f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change this to whatever model to use if you want a different one
    run_webcam("./models/ferplus_model.pth")