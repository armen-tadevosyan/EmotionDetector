# config.py -- constant & settings 

EMOTIONS = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

DIFFICULTY_TIERS = {
    1: ["happy", "neutral"],  # easiest
    2: ["sad", "surprise", "angry"],
    3: ["fear", "disgust", "contempt"],  # hardest (...unsure on what to do with contempt sicne it seems hard)
}

ACCURACY_WINDOW = 10

# Utility weights
LEARNING_WEIGHT = 0.5  # learning gain
CONFUSION_WEIGHT = 0.3   # confusion focus
FRUSTRATION_WEIGHT = 0.2  # frustration penalty

# Threshold
FRUSTRATION_STREAK_THRESHOLD = 3
RESPONSE_TIME_THRESHOLD = 3.0

MASTER_THRESHOLD = 0.80 # if accuracy > threshold, deprioritize that emotion