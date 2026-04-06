# Constants & settings for adaptive planner

# Supported emotions (should match model)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Rolling accuracy window (number of recent trials to consider)
ACCURACY_WINDOW = 10

# Utility weights (for adaptive_planner.py)
LEARNING_WEIGHT = 0.5      # importance of learning gain (focus on weak emotions)
CONFUSION_WEIGHT = 0.3     # bonus for targeting most confused emotion
FRUSTRATION_WEIGHT = 0.2   # penalty for user frustration

# Frustration / Response thresholds
FRUSTRATION_STREAK_THRESHOLD = 3   # number of consecutive errors considered frustrating
RESPONSE_TIME_THRESHOLD = 3.0      # seconds considered “slow response”

# Mastery threshold -- marks emotion as 'learned'
LEARNED_THRESHOLD = 0.80  # accuracy above this -> emotion considered learned