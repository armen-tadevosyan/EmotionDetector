# config.py -- constant & settings 

EMOTIONS = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

DIFFICULTY_TIERS = {
    1: ["happy", "neutral"],  # easiest
    2: ["sad", "surprise", "angry"],
    3: ["fear", "disgust", "contempt"],  # hardest (...unsure on what to do with contempt sicne it seems hard)
}

ACCURACY_WINDOW = 10

# Utility weights
ALPHA = 0.5  # learning gain
BETA = 0.3   # confusion focus
GAMMA = 0.2  # frustration penalty

FRUSTRATION_STREAK_THRESHOLD = 3
RESPONSE_TIME_THRESHOLD = 3.0

MASTER_THRESHOLD = 0.80