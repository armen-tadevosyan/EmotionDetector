# adaptive_planner.py -- the BRAIN of this model i hope...

from state import LearnerState
from config import DIFFICULTY_TIERS


class AdaptivePlanner:
    # Initialize by creating new state tracker & Level 1
    def __init__(self):
        self.state = LearnerState()
        self.current_difficulty = 1 # Start at level 1 (easy level)

    # update from User Interaction 
    def update(self, emotion, correct, response_time):
        self.state.update(emotion, correct, response_time)

    # Utility function *Core Idea*
    # How good is the action for the user right now?
    def compute_utility(self, action):
        accuracy = self.state.rolling_accuracy() # How well user is doing
        response_time = self.state.avg_response_time() # How hard user is thinking 
        error_streak = self.state.error_streak # Frustration level

        # Learning gain
        # Accuracy low = more to learn -> higher to gain
        # Accuracy high = less to learn
        learning_gain = (1 - accuracy)

        if action == "target_confusion":
            learning_gain += 0.5

        # Penalty
        # More mistakes = more frustration
        frustration = 0.1 * error_streak + 0.05 * response_time

        # Harder problems = more stress
        if action == "increase_difficulty":
            frustration += 0.2
        # Easier problems = less stress
        elif action == "decrease_difficulty":
            frustration -= 0.1

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

        utilities = {a: self.compute_utility(a) for a in actions} # scoring 
        return max(utilities, key=utilities.get) # Choose highest score

    # Final Decision
    def decide_next(self):
        action = self.select_action()

        if action == "increase_difficulty":
            self.current_difficulty += 1

        elif action == "decrease_difficulty":
            self.current_difficulty = max(1, self.current_difficulty - 1)

        elif action == "target_confusion":
            emotion = self.state.most_confused_emotion() # what does the user struggle with the most?
            return {
                "difficulty": self.current_difficulty,
                "target_emotion": emotion
            }

        return {
            "difficulty": self.current_difficulty
        }