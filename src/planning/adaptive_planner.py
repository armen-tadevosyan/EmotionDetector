# adaptive_planner.py -- the BRAIN of this model i hope...

from state import LearnerState
from config import (
    DIFFICULTY_TIERS,
    LEARNING_WEIGHT,
    CONFUSION_WEIGHT,
    FRUSTRATION_STREAK_THRESHOLD,
    RESPONSE_TIME_THRESHOLD,
    CONFUSION_WEIGHT,
    FRUSTRATION_WEIGHT
)

MAX_DIFFICULTY = max(DIFFICULTY_TIERS.keys())
MIN_DIFFICULTY = min(DIFFICULTY_TIERS.keys())

MIN_TRIALS_BEFORE_INCREASE = 3 # must spend at least this many trials at a level before going up

class AdaptivePlanner:
    def __init__(self):
        self.state = LearnerState()
        self.current_difficulty = MIN_DIFFICULTY 
        self.trials_at_difficulty = 0

    # update from User Interaction 
    def update(self, emotion, correct, response_time):
        self.state.update(emotion, correct, response_time)
        self.trials_at_difficulty += 1

    # Utility function *Core Idea*
    # How good is the action for the user right now?
    def compute_utility(self, action):
        accuracy = self.state.rolling_accuracy() # How well user is doing
        response_time = self.state.avg_response_time() # How hard user is thinking 
        error_streak = self.state.error_streak # Frustration level

        # Learning gain
        # Accuracy low = more to learn -> higher to gain
        # Accuracy high = less to learn
        learning_gain = LEARNING_WEIGHT * (1 - accuracy)

        if action == "increase_difficulty":
            learning_gain += LEARNING_WEIGHT * accuracy

        if action == "target_confusion":
            confused = self.state.most_confused_emotion()
            if confused is not None:
                confusion_penalty = 1 - self.state.emotion_accuracy(confused)
                learning_gain += CONFUSION_WEIGHT * confusion_penalty

        # Penalty
        # More mistakes = more frustration
        streak_signal = error_streak / max(FRUSTRATION_STREAK_THRESHOLD, 1)
        time_signal = response_time / max(RESPONSE_TIME_THRESHOLD, 1)
        frustration = FRUSTRATION_WEIGHT * (0.05 * streak_signal + 0.5 * time_signal)

        # Harder problems = more stress
        if action == "increase_difficulty":
            frustration += FRUSTRATION_WEIGHT * 0.5
        # Easier problems = less stress
        elif action == "decrease_difficulty":
            frustration -= FRUSTRATION_WEIGHT * 0.25

        # Pick action that max. learning but avoids frustration
        return learning_gain - frustration

    # Select Action
    def select_action(self):
        actions = [
            "increase_difficulty",
            "maintain_difficulty",
            "decrease_difficulty",
            "target_confusion"
        ]

        accuracy = self.state.rolling_accuracy()

        # mask impossible actions at difficulty boundaries
        if self.current_difficulty >= MAX_DIFFICULTY:
            actions.remove("increase_difficulty")
        if self.current_difficulty <= MIN_DIFFICULTY:
            actions.remove("decrease_difficulty")

        if self.trials_at_difficulty < MIN_TRIALS_BEFORE_INCREASE:
            if "increase_difficulty" in actions:
                actions.remove("increase_difficulty")
            if "decrease_difficulty" in actions:
                actions.remove("decrease_difficulty")

        if accuracy >= 0.7 and "decrease_difficulty" in actions:
            actions.remove("decrease_difficulty")
        
        utilities = {a: self.compute_utility(a) for a in actions} # scoring 
        return max(utilities, key=utilities.get) # Choose highest score

    # Final Decision
    def decide_next(self):
        action = self.select_action()

        if action == "increase_difficulty":
            self.current_difficulty = min(MAX_DIFFICULTY, self.current_difficulty + 1)
            self.trials_at_difficulty = 0
        elif action == "decrease_difficulty":
            self.current_difficulty = max(MIN_DIFFICULTY, self.current_difficulty - 1)
            self.trials_at_difficulty = 0

        result = {"difficulty": self.current_difficulty, "target_emotion": None}

        if action == "target_confusion":
            result["target_emotion"] = self.state.most_confused_emotion()

        return result
