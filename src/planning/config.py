# config.py -- constant & settings 
# Any valye that needs these values imports them from here

# Order matches the alphabetical class indices assigned by ImageFolder during training,
# so index 0 = angry, index 1 = disgust, etc. This must stay in sync with the model.
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Difficulty tiers map difficulty level (int) to the emotions shown at that level
DIFFICULTY_TIERS = {
    1: ["happy", "neutral"],  # easiest
    2: ["sad", "surprise", "angry"],
    3: ["fear", "disgust"],  # hardest (...unsure on what to do with contempt sicne it seems hard)
}

# How many recent trials to consider when computing rolling accuracy
# Only the last N trials count. Older trials are dropped automatically.
# A smaller window makes the planner react faster, a larger window makes it more stable
ACCURACY_WINDOW = 10

# Utility weights control how much each factor influences the planner's decision.
# They should sum to 1.0 
LEARNING_WEIGHT = 0.5  # learning gain
CONFUSION_WEIGHT = 0.3   # confusion focus
FRUSTRATION_WEIGHT = 0.2  # frustration penalty

# If the error streak exceeds this, the frustration signal is at full strength.
# For ex, 3 consecutive wrong answers = max frustration signal
FRUSTRATION_STREAK_THRESHOLD = 3

# If average response time exceeds this, the time signal is at full strength
# Slow reponses suggest cognitive overload or confusion
RESPONSE_TIME_THRESHOLD = 3.0

# If per-emotion accuracy exceeds this threshold, that emotion is considered mastered
# and the planner will deprioritize targeting it.
MASTER_THRESHOLD = 0.80 