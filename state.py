from config import EMOTIONS, ACCURACY_WINDOW, DIFFICULTY_TIERS
from collections import deque


# Meant to track everything about the user
class LearnerState:
    def __init(self):
        # Rolling window of emotion-correct tuples
        self.history = deque(maxlen=ACCURACY_WINDOW)


    def update():


    def rolling accuracy(self) -> float:



    def emotion_accuracy() -> float:

    def summary(self):
