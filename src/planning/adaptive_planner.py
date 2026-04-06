# Decides which emotion to show

from .state import LearnerState
from .config import EMOTIONS, LEARNING_WEIGHT, CONFUSION_WEIGHT, FRUSTRATION_WEIGHT
import random


class AdaptivePlanner:
    def __init__(self):
        self.state = LearnerState()

    # Update from user interaction
    def update(self, emotion, correct, response_time, user_answer=None):
        self.state.update(emotion, correct, response_time, user_answer)

    # Utility function (MDP-style reward)
    def compute_utility(self, emotion):
        # How well user understands this emotion
        acc = self.state.emotion_accuracy(emotion)

        # Learning potential: lower accuracy -> higher value
        learning_gain = 1 - acc

        # Confusion bonus -- focus on weak areas 
        confusion_bonus = 0
        confused = self.state.most_confused_emotion()
        if emotion == confused:
            confusion_bonus = 1

        # Frustration penalty
        frustration = (
            0.1 * self.state.error_streak + # Keeps getting things wrong 
            0.05 * self.state.avg_response_time() # Taking a long time
        )

        # Combine (weighted utility)
        utility = (
            LEARNING_WEIGHT * learning_gain +
            CONFUSION_WEIGHT * confusion_bonus -
            FRUSTRATION_WEIGHT * frustration
        )

        return utility

    # Choose next emotion -- ACTION
    def select_next_emotion(self):
        # Give each emotion a utility score 
        utilities = {
            emotion: self.compute_utility(emotion)
            for emotion in EMOTIONS
    }

    # soft exploration (epsilon-greedy) -- pick random emotion 20% of time
        if random.random() < 0.2:
            return random.choice(EMOTIONS)

        return max(utilities, key=utilities.get)

    # Final decision
    def decide_next(self):
        next_emotion = self.select_next_emotion()

        return {
            "target_emotion": next_emotion,
            "summary": self.state.summary()
        }