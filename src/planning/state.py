# state.py -- tracks user performance 

from config import EMOTIONS, ACCURACY_WINDOW
from collections import deque, defaultdict
import numpy as np


class LearnerState:
    def __init__(self):
        # Store (emotion, correct) tuples
        self.history = deque(maxlen=ACCURACY_WINDOW)

        # Track per-emotion performance
        self.emotion_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        # Extra signals
        self.response_times = []
        self.error_streak = 0

    # Update State
    def update(self, emotion, correct, response_time):
        # Update history
        self.history.append((emotion, correct))

        # Update per-emotion stats
        self.emotion_stats[emotion]["total"] += 1
        if correct:
            self.emotion_stats[emotion]["correct"] += 1

        # Update response time
        self.response_times.append(response_time)

        # Update error streak
        if correct:
            self.error_streak = 0
        else:
            self.error_streak += 1

    # Metrics
    def rolling_accuracy(self) -> float:
        if not self.history:
            return 0.5

        correct = sum(1 for _, c in self.history if c)
        return correct / len(self.history)

    def emotion_accuracy(self, emotion) -> float:
        stats = self.emotion_stats[emotion]
        if stats["total"] == 0:
            return 0.5
        return stats["correct"] / stats["total"]

    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0
        return np.mean(self.response_times)

    def most_confused_emotion(self):
        # Lowest accuracy emotion = most confused
        lowest_acc = 1.0
        target = None

        for emotion in EMOTIONS:
            acc = self.emotion_accuracy(emotion)
            if acc < lowest_acc:
                lowest_acc = acc
                target = emotion

        return target

    # Summary 
    def summary(self):
        return {
            "rolling_accuracy": self.rolling_accuracy(),
            "error_streak": self.error_streak,
            "avg_response_time": self.avg_response_time(),
        }