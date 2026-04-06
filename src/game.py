# This is the main game loop. This file should be the central orchestrator of the entire system
# It connects 3 layers:
# Planning layer - AdaptivePlanner decides what emotion/difficulty to show next based on the 
# users rolling accuracy, error streak, and response time.
# Perception layer - EmotionCNN flassifies the face image and returns a predicted emotion + 
# confidence. We use this to show the model's prediction as feedback after each answer
# Reasoning layer - Will eventually return a text explanation of why a face shows a given emotion
# like "eyebrows raised and mouth open = suprised" currently replaced with placeholder strings in get_explantion()

# Flow per trial (an idea):
# 1. Planner decides difficulty level and optional target emotion
# 2. Sample a matching image from the dataset
# 3. Run the image through EmotionCNN to get a prediction
# 4. Display the image to the user via pygame
# 5. User presses a number key (1-7) to select an emotion
# 6. Score the answer, show feedback + model prediction + explanation
# 7. Feed result back into the planner (emotion, correct, response_time)
# 8. Press SPACE then repeat from step 1
import pygame
import random
import time

from planning.adaptive_planner import AdaptivePlanner
from planning.config import EMOTIONS

pygame.init()

# Screen setup
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Emotion Detector")

font = pygame.font.Font(None, 36)

# Planner
planner = AdaptivePlanner()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
GREEN = (100, 200, 100)

# Create buttons
def create_buttons(options):
    buttons = []
    button_width = 150
    button_height = 50
    spacing = 20

    start_x = (WIDTH - (len(options) * (button_width + spacing))) // 2

    for i, option in enumerate(options):
        rect = pygame.Rect(
            start_x + i * (button_width + spacing),
            HEIGHT - 150,
            button_width,
            button_height
        )
        buttons.append((rect, option))

    return buttons

# Draw buttons
def draw_buttons(buttons):
    for rect, text in buttons:
        pygame.draw.rect(screen, GRAY, rect)
        label = font.render(text, True, BLACK)
        screen.blit(label, (rect.x + 10, rect.y + 10))

# Simulated "image" -- placeholder for now
def draw_emotion(emotion):
    text = font.render(f"Emotion: {emotion}", True, BLACK)
    screen.blit(text, (WIDTH // 2 - 100, HEIGHT // 2 - 50))

# Game loop
running = True

# Initial state
current_emotion = random.choice(EMOTIONS)
start_time = time.time()

while running:
    screen.fill(WHITE)

    # Get next target from planner
    decision = planner.decide_next()
    target_emotion = decision.get("target_emotion")

    if target_emotion:
        current_emotion = target_emotion

    # Pick 4 options 
    options = random.sample(EMOTIONS, 3)
    if current_emotion not in options:
        options.append(current_emotion)
    random.shuffle(options)

    buttons = create_buttons(options)

    draw_emotion(current_emotion)
    draw_buttons(buttons)

    pygame.display.flip()

    answered = False

    while not answered:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                answered = True

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos

                for rect, label in buttons:
                    if rect.collidepoint(mouse_pos):
                        # User clicked an answer
                        user_answer = label
                        response_time = time.time() - start_time

                        correct = (user_answer == current_emotion)

                        # Update planner
                        planner.update(
                            current_emotion,
                            correct,
                            response_time,
                            user_answer
                        )

                        print(f"User picked: {user_answer} | Correct: {correct}")

                        # Reset for next round
                        start_time = time.time()
                        answered = True

    pygame.time.delay(500)

pygame.quit()