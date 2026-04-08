import pygame
import random
import time
from pathlib import Path

from .adaptive_planner import AdaptivePlanner
from .config import EMOTIONS

pygame.init()

# Screen setup
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Emotion Guesser")

font = pygame.font.Font(None, 36)

# Planner
planner = AdaptivePlanner()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
GREEN = (100, 200, 100)
BLUE  = (100, 100, 255)

# Root directory for labeled face images
BASE_DIR = Path(__file__).resolve().parent
LABELED_FACES_DIR = BASE_DIR / "../labeled_faces"
LABELED_FACES_DIR = LABELED_FACES_DIR.resolve()


def get_random_image_for_emotion(emotion):
    emotion_dir = LABELED_FACES_DIR / emotion
    if not emotion_dir.is_dir():
        return None
    image_paths = [p for p in emotion_dir.iterdir() if p.suffix.lower() == ".png"]
    if not image_paths:
        return None
    return random.choice(image_paths)

# load and resize image to fit game.py screen
def load_and_scale_image(image_path, max_width=500, max_height=300):
    image = pygame.image.load(str(image_path))
    original_width, original_height = image.get_size()

    scale = min(max_width / original_width, max_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    return pygame.transform.smoothscale(image, (new_width, new_height))


# Create buttons with optional y_position arg
def create_buttons(options, y_position=HEIGHT-150):
    buttons = []
    button_width = 150
    button_height = 50
    spacing = 20

    total_width = len(options) * button_width + (len(options) - 1) * spacing
    start_x = (WIDTH - total_width) // 2

    for i, option in enumerate(options):
        rect = pygame.Rect(
            start_x + i * (button_width + spacing),
            y_position, 
            button_width,
            button_height
        )
        buttons.append((rect, option))

    return buttons

# Draw buttons
def draw_buttons(buttons):
    for rect, text in buttons:
        pygame.draw.rect(screen, GRAY, rect)
        pygame.draw.rect(screen, BLACK, rect, 2)
        label = font.render(text, True, BLACK)
        label_rect = label.get_rect(center=rect.center)
        screen.blit(label, label_rect)


def draw_emotion_image_fixed(image):
    if image:
        screen.blit(image, image.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 40)))
    else:
        text = font.render("No image available", True, BLACK)
        screen.blit(text, text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50)))
        
def draw_stats_summary(summary):
    y = 150
    lines = [
        f"Rolling Accuracy: {summary['rolling_accuracy']:.2f}",
        f"Avg Response Time: {summary['avg_response_time']:.2f}s",
        f"Current Error Streak: {summary['error_streak']}"
    ]
    for line in lines:
        text = font.render(line, True, BLUE)
        screen.blit(text, text.get_rect(center=(WIDTH // 2, y)))
        y += 40
        
# Set up helper-function for preparing game for next round
def setup_next_round():
    global current_emotion, current_image, buttons, start_time

    decision = planner.decide_next()
    target_emotion = decision.get("target_emotion")

    if target_emotion:
        current_emotion = target_emotion
    else:
        current_emotion = random.choice(EMOTIONS) # fall back to random 

    image_path = get_random_image_for_emotion(current_emotion)
    current_image = load_and_scale_image(image_path) if image_path else None

    options = random.sample([e for e in EMOTIONS if e != current_emotion], 3)
    options.append(current_emotion)
    random.shuffle(options)

    buttons = create_buttons(options)
    start_time = time.time()
        

# Game loop
running = True
round_count = 0
MAX_ROUNDS = 10
session_phase = "game"  # "game" or "stats"

# Storing random image 
setup_next_round()

while running:
    screen.fill(WHITE)

    # Drawing Game phase
    if session_phase == "game":
        draw_emotion_image_fixed(current_image)
        draw_buttons(buttons)

    elif session_phase == "stats":
        summary = planner.state.summary()
        draw_stats_summary(summary)
        buttons = create_buttons(["Continue", "End"], HEIGHT - 150)
        draw_buttons(buttons)

    pygame.display.flip()

    # Event 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos

            for rect, label in buttons:
                if rect.collidepoint(mouse_pos):

                    # Stats 
                    if session_phase == "stats":
                        if label == "Continue":
                            round_count = 0
                            session_phase = "game"
                            # Prepare new round
                            setup_next_round()
                        else:  # End
                            running = False

                    # Game buttons
                    elif session_phase == "game":
                        user_answer = label
                        response_time = time.time() - start_time
                        correct = (user_answer == current_emotion)

                        # Update planner
                        planner.update(current_emotion, correct, response_time, user_answer)
                        print(f"Round {round_count + 1}: User picked {user_answer} | Correct: {correct}")

                        round_count += 1

                        # Check if max rounds have been met
                        if round_count >= MAX_ROUNDS:
                            session_phase = "stats"
                        else:
                            # Prepare next round
                            setup_next_round()

    pygame.time.delay(50)

pygame.quit()