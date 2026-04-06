# Tracks learner performance

from collections import deque, defaultdict
import numpy as np
from config import EMOTIONS, ACCURACY_WINDOW

class LearnerState:
    def __init__(self):
        # Rolling history of (emotion, correct) tuples
        self.history = deque(maxlen=ACCURACY_WINDOW)

        # Track perfomance per emotion stats: {"happy": {"correct": 0, "total": 0}, ...}
        self.emotion_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        # Track response times
        self.response_times = deque(maxlen=ACCURACY_WINDOW)

        # Track consecutive errors (frustration)
        self.error_streak = 0

        # Track confusion pairs: (true_emotion, predicted_emotion) -> count
        self.confusion_matrix = defaultdict(int)

    # Update state after a trial
    def update(self, true_emotion, correct, response_time, user_answer=None):
        # 1️. Update rolling history
        self.history.append((true_emotion, correct))

        # 2️. Update per-emotion stats
        self.emotion_stats[true_emotion]["total"] += 1
        if correct:
            self.emotion_stats[true_emotion]["correct"] += 1

        # 3️. Update response times
        self.response_times.append(response_time)

        # 4. Update error streak
        if correct:
            self.error_streak = 0
        else:
            self.error_streak += 1

        # 5️. Update confusion matrix -- track confusion based on user's answer
        if not correct and user_answer is not None:
            self.confusion_matrix[(true_emotion, user_answer)] += 1

    # Metrics -- measuring user performance 
    
    # Recent history
    def rolling_accuracy(self) -> float:
        if not self.history:
            return 0.5
        correct_count = sum(1 for _, correct in self.history if correct)
        return correct_count / len(self.history)

    # Accuracy for ONE emotion
    def emotion_accuracy(self, emotion: str) -> float:
        stats = self.emotion_stats[emotion]
        if stats["total"] == 0:
            return 0.5  # default if never seen
        return stats["correct"] / stats["total"]

    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0
        return np.mean(self.response_times)

    # Return the emotion with lowest accuracy -- target for next trial
    def most_confused_emotion(self) -> str:
        lowest_acc = 1.0 # High baseline
        target_emotion = None
        for emotion in EMOTIONS:
            acc = self.emotion_accuracy(emotion)
            if acc < lowest_acc:
                lowest_acc = acc
                target_emotion = emotion
        return target_emotion

    # Summary for planner
    def summary(self):
        return {
            "rolling_accuracy": self.rolling_accuracy(),
            "error_streak": self.error_streak,
            "avg_response_time": self.avg_response_time()
        }