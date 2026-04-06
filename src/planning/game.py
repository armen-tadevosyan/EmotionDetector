# game.py -- main game loop
# connects the planning layer, ResNet model, and reasoning explanations
# run from src/ with: python game.py

import os
import sys
import time
import random

import pygame
import torch
import kagglehub
from PIL import Image

# add planning/ and reasoning/ to path so imports work from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "planning"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "reasoning"))

from adaptive_planner import AdaptivePlanner
from config import DIFFICULTY_TIERS
from resnet.resnet_predict import load_resnet_model, predict_image, EMOTION_LABELS
from rules import get_explanation, get_tip

# use the model's own output labels as the choices -- this way they can never go out of sync
EMOTIONS = EMOTION_LABELS

WINDOW_W = 800
WINDOW_H = 600
IMAGE_SIZE = 250
TRIALS_PER_SESSION = 15
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ferplus_resnet18.pth")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (200, 0, 0)
GREEN = (0, 180, 0)
GRAY  = (200, 200, 200)


def locate_train_dir(data_path):
    for root, dirs, _ in os.walk(data_path):
        if "train" in dirs:
            return os.path.join(root, "train")
    raise FileNotFoundError("Could not find train directory")


def build_image_index(train_dir):
    index = {}
    for emotion in EMOTIONS:
        folder = os.path.join(train_dir, emotion)
        if not os.path.isdir(folder):
            continue
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if files:
            index[emotion] = files
    return index


def sample_image(image_index, difficulty, target_emotion=None):
    if target_emotion and target_emotion in image_index:
        emotion = target_emotion
    else:
        tier_emotions = [e for e in DIFFICULTY_TIERS[difficulty] if e in image_index]
        if not tier_emotions:
            tier_emotions = list(image_index.keys())
        emotion = random.choice(tier_emotions)
    return random.choice(image_index[emotion]), emotion


def draw_text(surface, text, font, color, x, y, center=False):
    rendered = font.render(text, True, color)
    rect = rendered.get_rect()
    if center:
        rect.center = (x, y)
    else:
        rect.topleft = (x, y)
    surface.blit(rendered, rect)
    return rect.bottom


def draw_wrapped_text(surface, text, font, color, x, y, max_width):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        if font.size(test)[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    for i, line in enumerate(lines):
        draw_text(surface, line, font, color, x, y + i * (font.get_height() + 2))


def run_game():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Loading model...")
    model = load_resnet_model(MODEL_PATH, device)

    print("Loading dataset...")
    data_path = kagglehub.dataset_download("subhaditya/fer2013plus")
    train_dir = locate_train_dir(data_path)
    image_index = build_image_index(train_dir)

    planner = AdaptivePlanner()

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Emotion Recognition Game")
    clock = pygame.time.Clock()
    font       = pygame.font.SysFont("Arial", 20)
    font_large = pygame.font.SysFont("Arial", 28, bold=True)
    font_small = pygame.font.SysFont("Arial", 16)

    STATE_QUESTION = "question"
    STATE_FEEDBACK = "feedback"
    STATE_DONE     = "done"

    decision = planner.decide_next()
    img_path, true_emotion = sample_image(image_index, decision["difficulty"], decision["target_emotion"])
    current_img = None

    state       = STATE_QUESTION
    trial_count = 0
    trial_start = time.time()

    fb_correct      = False
    fb_user_choice  = ""
    fb_true_emotion = ""
    fb_model_pred   = ""
    fb_confidence   = 0.0
    fb_explanation  = ""
    fb_tip          = ""

    key_map = {
        pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3,
        pygame.K_5: 4, pygame.K_6: 5, pygame.K_7: 6,
    }

    running = True
    while running:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if state == STATE_QUESTION:
                    if event.key in key_map:
                        chosen_idx = key_map[event.key]
                        if chosen_idx < len(EMOTIONS):
                            user_choice   = EMOTIONS[chosen_idx]
                            response_time = time.time() - trial_start
                            correct       = (user_choice == true_emotion)

                            model_pred, conf, _ = predict_image(img_path, model, device)
                            explanation = get_explanation(true_emotion)
                            tip         = get_tip()

                            planner.update(true_emotion, correct, response_time)

                            fb_correct      = correct
                            fb_user_choice  = user_choice
                            fb_true_emotion = true_emotion
                            fb_model_pred   = model_pred
                            fb_confidence   = conf
                            fb_explanation  = explanation
                            fb_tip          = tip

                            trial_count += 1
                            state = STATE_FEEDBACK

                elif state == STATE_FEEDBACK:
                    if event.key == pygame.K_SPACE:
                        if trial_count >= TRIALS_PER_SESSION:
                            state = STATE_DONE
                        else:
                            decision = planner.decide_next()
                            img_path, true_emotion = sample_image(
                                image_index, decision["difficulty"], decision["target_emotion"]
                            )
                            current_img = None
                            trial_start = time.time()
                            state = STATE_QUESTION

                elif state == STATE_DONE:
                    if event.key == pygame.K_ESCAPE:
                        running = False

        # drawing

        if state == STATE_QUESTION:

            # top info bar
            summary = planner.state.summary()
            draw_text(screen, f"Trial: {trial_count + 1}/{TRIALS_PER_SESSION}", font_small, BLACK, 10, 10)
            draw_text(screen, f"Difficulty: {planner.current_difficulty}/3", font_small, BLACK, 10, 28)
            draw_text(screen, f"Accuracy: {summary['rolling_accuracy']*100:.0f}%", font_small, BLACK, 10, 46)

            # face image
            if current_img is None:
                try:
                    pil_img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                    raw = pil_img.tobytes()
                    current_img = pygame.image.fromstring(raw, (IMAGE_SIZE, IMAGE_SIZE), "RGB")
                except Exception:
                    current_img = pygame.Surface((IMAGE_SIZE, IMAGE_SIZE))
                    current_img.fill(GRAY)

            img_x = (WINDOW_W - IMAGE_SIZE) // 2
            screen.blit(current_img, (img_x, 70))

            # question
            draw_text(screen, "What emotion is this?", font, BLACK, WINDOW_W // 2, 340, center=True)

            # numbered choices
            for i, emotion in enumerate(EMOTIONS):
                col = i % 4
                row = i // 4
                x = 80 + col * 170
                y = 370 + row * 28
                draw_text(screen, f"{i+1}. {emotion}", font_small, BLACK, x, y)

            draw_text(screen, "Press 1-7 to answer", font_small, GRAY, WINDOW_W // 2, WINDOW_H - 20, center=True)

        elif state == STATE_FEEDBACK:

            result_text  = "Correct!" if fb_correct else f"Wrong! It was: {fb_true_emotion}"
            result_color = GREEN if fb_correct else RED
            draw_text(screen, result_text, font_large, result_color, WINDOW_W // 2, 30, center=True)

            # show image again
            if current_img is None:
                try:
                    pil_img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                    raw = pil_img.tobytes()
                    current_img = pygame.image.fromstring(raw, (IMAGE_SIZE, IMAGE_SIZE), "RGB")
                except Exception:
                    current_img = pygame.Surface((IMAGE_SIZE, IMAGE_SIZE))
                    current_img.fill(GRAY)

            small_img = pygame.transform.scale(current_img, (150, 150))
            screen.blit(small_img, (20, 60))

            # feedback text
            draw_text(screen, f"You said: {fb_user_choice}", font, BLACK, 200, 65)
            draw_text(screen, f"AI predicted: {fb_model_pred} ({fb_confidence*100:.0f}%)", font, BLACK, 200, 95)

            # explanation
            draw_text(screen, "Why:", font, BLACK, 20, 230)
            draw_wrapped_text(screen, fb_explanation, font_small, BLACK, 20, 255, WINDOW_W - 40)
            draw_wrapped_text(screen, fb_tip, font_small, GRAY, 20, 300, WINDOW_W - 40)

            draw_text(screen, "Press SPACE to continue", font_small, GRAY, WINDOW_W // 2, WINDOW_H - 20, center=True)

        elif state == STATE_DONE:

            summary = planner.state.summary()
            draw_text(screen, "Session Complete!", font_large, BLACK, WINDOW_W // 2, 40, center=True)
            draw_text(screen, f"Trials: {trial_count}", font, BLACK, 50, 100)
            draw_text(screen, f"Final Accuracy: {summary['rolling_accuracy']*100:.0f}%", font, BLACK, 50, 130)

            draw_text(screen, "Per-emotion accuracy:", font, BLACK, 50, 175)
            stats = summary["emotion_stats"]
            for i, (emotion, acc) in enumerate(stats.items()):
                if acc is not None:
                    draw_text(screen, f"{emotion}: {acc*100:.0f}%", font_small, BLACK, 70, 205 + i * 24)

            draw_text(screen, "Press ESC to exit", font_small, GRAY, WINDOW_W // 2, WINDOW_H - 20, center=True)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    run_game()
