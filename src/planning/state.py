# Meant to track everything about the user
class State:
    # Initialize the state of the learning system. 
    # Represents the "state" in the MDP. It tracks user performance over time.
    def __init(self):
        self.history = []
        self.confusion_matrix = {}
        self.difficulty = "easy"

    # Updates the state after each user interaction.
    def update(self, true_emotion, user_answer, response_time):
        correct = (true_emotion == user_answer)

        self.history.append({
            "true": true_emotion, # the correct label
            "answer": user_answer, # what the user guessed
            "correct": correct, 
            "time": response_time # how long they took
        })

        # Updates the confusion matrix only if its incorrect. 
        # Like if the user thought "sad" was "angry"
        if not correct:
            key = (true_emotion, user_answer)
            self.confusion_matrix[key] = self.confusion_matrix.get(key, 0) + 1

    # Computes rolling accuracy over the last N attempts.
    # This is used to estimate learning progress
    def get_accuracy(self, windows=5):
        recent = self.history[-window:]
        if not recent:
            return 0
        return sum(h["correct"] for h in recent) / len(recent)
    
    # Computes average response time over recent attempts. 
    def get_avg_time(self, window=5):
        recent = self.history[-window:]
        if not recent: 
            return 0
        return sum(h["time"] for h in recent) / len(recent)
    
    # Counts how many incorrect answers in a row. this is used for detecting frustration.
    def get_error_streak(self):
        streak = 0
        for h in reversed(self.history):
            if h["correct"]:
                break
            streak += 1
        return streak