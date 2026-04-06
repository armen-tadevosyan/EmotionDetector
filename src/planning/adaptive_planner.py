# adaptive_planner.py -- the BRAIN of this model i hope...
# It decides what to show the user next based on their performance.
# It treats learning as a decision-making problem with 4 possible actions:
# increase_difficulty - move the user up to harder emotions
# maintain_difficulty - keep showing emotions at the current level
# decrease_difficulty - drop back to easier emotions (user is struggling)
# target_confusion - stay at the same level but focus on the most missed emotion
# With every trial it scores each action using a utility finction and picks the best one.

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

MAX_DIFFICULTY = max(DIFFICULTY_TIERS.keys()) # highest tier (3)
MIN_DIFFICULTY = min(DIFFICULTY_TIERS.keys()) # lowest tier (1)

# The user must spend at least this many trials at a difficulty level
# before the planner is allowed to move them up or down. 
MIN_TRIALS_BEFORE_INCREASE = 3 

class AdaptivePlanner:
    def __init__(self):
        # LearnerState tracks all performance metrics (accuracy, streaks, response time)
        self.state = LearnerState()

        # Start every session at the easiest difficulty
        self.current_difficulty = MIN_DIFFICULTY 

        # Counts how many trials have been spent at the current difficulty level.
        # Resets to 0 whenever the difficulty changes
        self.trials_at_difficulty = 0

    # update from User Interaction 
    def update(self, emotion, correct, response_time):
        self.state.update(emotion, correct, response_time)
        self.trials_at_difficulty += 1

    # Scores how good a given action is for the user right now.
    def compute_utility(self, action):
        accuracy = self.state.rolling_accuracy() # recent correctness rate
        response_time = self.state.avg_response_time() # recent average response time
        error_streak = self.state.error_streak # consecutive wrong answers

    
        # Base gain: low accurarcy means more room to improve at the current level
        learning_gain = LEARNING_WEIGHT * (1 - accuracy)

        # There is a bonus for increase_difficulty. Higher accuracy means the user is ready for more
        # At 100% accuracy this adds LEARNING_WEIGHT * 1.0 = 0.5, making increase_difficulty
        # competitive against maintain_difficulty
        if action == "increase_difficulty":
            learning_gain += LEARNING_WEIGHT * accuracy
        
        # Bonus for target_confusion. If a specific emotion has low accurary, 
        # targeting it has high learning potential
        if action == "target_confusion":
            confused = self.state.most_confused_emotion()
            if confused is not None:
                confusion_penalty = 1 - self.state.emotion_accuracy(confused)
                learning_gain += CONFUSION_WEIGHT * confusion_penalty

        # Penalty
        # Normalize each signal against its threshold so values are on a comparable scale.
        streak_signal = error_streak / max(FRUSTRATION_STREAK_THRESHOLD, 1)
        time_signal = response_time / max(RESPONSE_TIME_THRESHOLD, 1)

        # Combine signals into a single frustration score
        # Streaks are weighted lightly because short streaks are normal. Time is weighted more.
        frustration = FRUSTRATION_WEIGHT * (0.05 * streak_signal + 0.5 * time_signal)

        # Harder problems = more stress
        if action == "increase_difficulty":
            frustration += FRUSTRATION_WEIGHT * 0.5
        # Easier problems gives a small relief bonus so there is less stress
        elif action == "decrease_difficulty":
            frustration -= FRUSTRATION_WEIGHT * 0.25

        return learning_gain - frustration

    # Scores all valid actions and returns the onse with the highest utility.
    # Some actions are masked (removed from consideration) before scoring 
    # based on hard constraints
    def select_action(self):
        actions = [
            "increase_difficulty",
            "maintain_difficulty",
            "decrease_difficulty",
            "target_confusion"
        ]

        # Pull accuracy here so the masks below can use it
        accuracy = self.state.rolling_accuracy()

        # remove actions that are structurally impossible
        if self.current_difficulty >= MAX_DIFFICULTY:
            actions.remove("increase_difficulty") # already at the top
        if self.current_difficulty <= MIN_DIFFICULTY:
            actions.remove("decrease_difficulty") # already at the bottom

        # Don't allow difficulty changes until the user has spent enough trials
        # at the current level. Prevents flip-flopping from a single good/bad answer
        if self.trials_at_difficulty < MIN_TRIALS_BEFORE_INCREASE:
            if "increase_difficulty" in actions:
                actions.remove("increase_difficulty")
            if "decrease_difficulty" in actions:
                actions.remove("decrease_difficulty")

        # Don't drop difficulty if the user is performing well. Prevents the planner 
        # from lowering difficulty just because the frustration relief bonus makes 
        # decrease_difficulty look attractive when no other options exist.
        if accuracy >= 0.7 and "decrease_difficulty" in actions:
            actions.remove("decrease_difficulty")
        
        # Score remaining actions and return the best one
        utilities = {a: self.compute_utility(a) for a in actions} # scoring 
        return max(utilities, key=utilities.get) # Choose highest score

    def decide_next(self):
        action = self.select_action()

        # Apply the chosen action to update difficulty state
        if action == "increase_difficulty":
            self.current_difficulty = min(MAX_DIFFICULTY, self.current_difficulty + 1)
            self.trials_at_difficulty = 0 # reset level timer on any change
        elif action == "decrease_difficulty":
            self.current_difficulty = max(MIN_DIFFICULTY, self.current_difficulty - 1)
            self.trials_at_difficulty = 0 # reset level timer on any change
        # maintain_difficulty and target_confusion leave difficulty unchanged

        result = {"difficulty": self.current_difficulty, "target_emotion": None}

        if action == "target_confusion":
            result["target_emotion"] = self.state.most_confused_emotion()

        return result
