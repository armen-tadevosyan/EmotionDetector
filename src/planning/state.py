# state.py -- tracks user performance across trials
# LearnerState is a live snapshot of how the user is doing
# It is updated after every trial and queried by the planner to make decisions.

from config import EMOTIONS, ACCURACY_WINDOW
from collections import deque, defaultdict
import numpy as np


class LearnerState:
    def __init__(self):
        # A rolling window of the last ACCURACY_WINDOW trials.
        # Each entry is a tuple: (emotion, correct, response_time)
        # When its full the oldest entry is automatically dropped when a new one is added.
        # This ensures metrics only reflect recent performance, not the whole session
        self.history = deque(maxlen=ACCURACY_WINDOW)

        # Per-emotion performance tracker.
        # Used to find which specific emotion the user struggles with most.
        self.emotion_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        # Counts how many wrong answers in a row the user has given.
        # Resets to 0 on any correct answer. Used as a frustration signal
        self.error_streak = 0

    # Update State
    def update(self, emotion, correct, response_time):
        # Called after every trial with the result. Updates all internal tracking.
        
        # Add this trial to the rolling window
        self.history.append((emotion, correct, response_time))

        # Update the all-time per-emotion stats 
        self.emotion_stats[emotion]["total"] += 1
        if correct:
            self.emotion_stats[emotion]["correct"] += 1

        # Update error streak
        if correct:
            self.error_streak = 0 # reset on correct answer
        else:
            self.error_streak += 1 # increment on wrong answer

    # Metrics
    def rolling_accuracy(self) -> float:
        # Fraction of correct answers in the recent window. Returns 0.5 (baseline)
        # if no trials have happened yet, so the planner doesn't assume the 
        # user is failing or succeeding at the start.

        if not self.history:
            return 0.5 
        correct = sum(1 for _, c, _ in self.history if c)
        return correct / len(self.history)

    def emotion_accuracy(self, emotion) -> float:
        # Fraction of correct answers for a specific emotion across the full session.
        # Returns None if the emotion hasn't been seen yet this lets the planner
        # differentiate "never seen" from "always wrong"

        stats = self.emotion_stats[emotion]
        if stats["total"] == 0:
            return None 
        return stats["correct"] / stats["total"]

    def avg_response_time(self) -> float:
        # Average response time across the rolling window (not all-time)
        # This ensures that slow early trials don't permanently skew the metric
        # if the user speeds up as they get more comfortable.
        times = [t for _, _, t in self.history]
        if not times:
            return 0.0
        return float(np.mean(times))

    def most_confused_emotion(self):
        # Returns the emotion the user has gotten wrong most often (lowest accuracy).
        # Only considers emotions that have actually been shown. Skips unseen emotions 
        # to avoid cold-start false positives (ex. returning "angry" just because it 
        # hasn't been seen yet and a defaults to 0%). Returns None if not emotions have
        # been seen yet.

        seen = {
            e: self.emotion_accuracy(e)
            for e in EMOTIONS
            if self.emotion_accuracy(e) is not None # skip unseen emotions
        }
        if not seen:
            return None
        return min(seen, key=seen.get) # emotion with the lowest accuracy

    # Summary -- "snapshot" of the user's performance. Used by stimulate.py 
    # to print readable output after each trial.
    def summary(self) -> dict:
        return {
            "rolling_accuracy": self.rolling_accuracy(),
            "error_streak": self.error_streak,
            "avg_response_time": self.avg_response_time(),
            # Only include emotions that have been seen at least once
            "emotion_stats": {
                e: self.emotion_accuracy(e)
                for e in EMOTIONS
                if self.emotion_accuracy(e) is not None
            },
        }