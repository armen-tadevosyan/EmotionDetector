"""
label_faces.py
Classifies every face image in an input directory using the
emotion model from resnet_predict.py, then copies each image into:

    labeled_faces/<emotion>/<filename>

Usage:
    python label_faces.py --model resnet_model.pth --input path/to/faces
    # optional flags:
    #   --output  labeled_faces   (change the root output directory)
    #   --device  cpu | cuda | mps
"""

import argparse
import math
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from resnet_predict import load_resnet_model, predict_image, EMOTION_LABELS

MINIMUM_CONFIDENCE = 0.747  # 0.747 is ideal for minimizing relative standard deviation based on a grid search

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sort face images into emotion subdirectories using ResNet18."
    )
    parser.add_argument("--model",  required=True, help="Path to the .pth weights file.")
    parser.add_argument("--input",  required=True, help="Directory of face images to classify.")
    parser.add_argument("--output", default="src/labeled_faces", help="Root output directory.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[info] Device : {device}")
    print(f"[info] Loading model from: {args.model}")
    model = load_resnet_model(args.model, device)
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = [
        p for p in input_dir.rglob("*")
        if p.suffix.lower() == ".png"
    ]
    print(f"[info] Found {len(image_paths)} image(s).\n")

    if not image_paths:
        print("[warn] No images found — nothing to do.")
        return

    output_root = Path(args.output)
    for emotion in EMOTION_LABELS + ["uncertain"]:
        (output_root / emotion).mkdir(parents=True, exist_ok=True)

    counters: dict[str, int] = {e: 0 for e in EMOTION_LABELS + ["uncertain"]}
    errors: list[str] = []
    uncertains: list[str] = []

    for img_path in tqdm(image_paths, desc="Classifying", unit="img"):
        try:
            emotion, confidence, probs = predict_image(str(img_path), model, device)
            if confidence < MINIMUM_CONFIDENCE:
                uncertains.append(f"Uncertain about {img_path}: {emotion} {confidence} {probs}")
                emotion = "uncertain"

            dest = output_root / emotion / img_path.name
            # avoid overwriting files that share a name
            if dest.exists():
                n = 1
                while dest.exists():
                    dest = output_root / emotion / f"{img_path.stem}_{n}{img_path.suffix}"
                    n += 1
            shutil.copy2(img_path, dest)
            counters[emotion] += 1

        except Exception as exc:
            errors.append(f"{img_path}: {exc}")

    print(f"\nDone — images saved to '{output_root}/'")
    for emotion, count in sorted(counters.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"{emotion:<12}  {count:>5}  {bar}")

    num_emotions = len(EMOTION_LABELS)
    total_images = len(image_paths)
    expected_mean = (total_images - counters['uncertain']) / num_emotions
    observed_counts = [counters[emotion] for emotion in EMOTION_LABELS]
    variance = sum((count - expected_mean) ** 2 for count in observed_counts) / num_emotions
    std_dev = math.sqrt(variance)
    print("\nStatistics assuming equal true class distribution:")
    print(f"Expected mean per emotion: {expected_mean:.2f}")
    print("\nDeviation from expected mean:")
    for emotion in EMOTION_LABELS:
        diff = counters[emotion] - expected_mean
        print(f"{emotion:<12}  count={counters[emotion]:>5}  diff={diff:>8.2f}")
    print(f"\nStandard Deviation: {std_dev:.2f}")
    print(f"Relative standard deviation of emotion counts from expected mean: {(std_dev/expected_mean):.4f}")
    if errors:
        print(f"\n[warn] {len(errors)} files failed:")
        [print(e) for e in errors]
    if uncertains:
        [print(e) for e in uncertains]


if __name__ == "__main__":
    main()