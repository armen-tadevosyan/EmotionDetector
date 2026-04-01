# state.py -- tracks user performance 

from config import EMOTIONS, ACCURACY_WINDOW
from collections import deque, defaultdict
import numpy as np


class LearnerState:
    def __init__(self):
        # Store (emotion, correct, and response_time) tuples
        self.history = deque(maxlen=ACCURACY_WINDOW)

        # Track per-emotion performance
        self.emotion_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        self.error_streak = 0

    # Update State
    def update(self, emotion, correct, response_time):
        # Update history
        self.history.append((emotion, correct, response_time))

        # Update per-emotion stats
        self.emotion_stats[emotion]["total"] += 1
        if correct:
            self.emotion_stats[emotion]["correct"] += 1

        # Update error streak
        if correct:
            self.error_streak = 0
        else:
            self.error_streak += 1

    # Metrics
    def rolling_accuracy(self) -> float:
        if not self.history:
            return 0.5 # base-line before any data
        correct = sum(1 for _, c, _ in self.history if c)
        return correct / len(self.history)

    def emotion_accuracy(self, emotion) -> float:
        stats = self.emotion_stats[emotion]
        if stats["total"] == 0:
            return None 
        return stats["correct"] / stats["total"]

    def avg_response_time(self) -> float:
        # Rolling window only - not all-time average
        times = [t for _, _, t in self.history]
        if not times:
            return 0.0
        return float(np.mean(times))

    def most_confused_emotion(self):
        # Return the seen emotion with the lowest accuracy. 
        # Returns None if no emotions have been seen yet.
        seen = {
            e: self.emotion_accuracy(e)
            for e in EMOTIONS
            if self.emotion_accuracy(e) is not None
        }
        if not seen:
            return None
        return min(seen, key=seen.get)

    # Summary -- "snapshot" of the user's performance
    def summary(self) -> dict:
        return {
            "rolling_accuracy": self.rolling_accuracy(),
            "error_streak": self.error_streak,
            "avg_response_time": self.avg_response_time(),
            "emotion_stats": {
                e: self.emotion_accuracy(e)
                for e in EMOTIONS
                if self.emotion_accuracy(e) is not None
            },
        }