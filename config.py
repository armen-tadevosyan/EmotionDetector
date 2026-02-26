# Basically having all the tunable constants in one place

# Emotions supported by the classifier
EMOTIONS = ["happy", "sad", "angry", "suprised", "fearful", "disgusted", "neutral"]

# Difficulty tiers - based on empirical CNN per-class error rates
# Tier 1 = easiest (high model confidence, most distinct), Tier 3 = hardest
DIFFICULTY_TIERS = {
    1: ["happy", "neutral"],
    2: ["sad", "surprised", "angry"],
    3: ["fearful", "disgusted"],
}

# Rolling accuracy window size (number of recent trials to consider)
ACCURACY_WINDOW = 10

# Utility function weights
ALPHA = 0.5 # Weight for learning gain
BETA = 0.3
GAMMA = 0.2

FRUSTRATION_STREAK_THRESHOLD
RESPONSE_TIME_THRESHOLD =

# Mastery threshold, if accuracy on an emotion exceeds this deprioritize it
MASTER_THRESHOLD = 0.80