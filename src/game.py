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