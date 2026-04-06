# gives fake session data into the planner
# Tests the planning layer without images, model, and pygame
# Each scenario constrcuts a sequence that simulates a specific type of 
# user, then runs them through AdaptivePlanner and prints what the planner
# decides after each trial

# Run it from src with: python simulate.py
#Simulates user sessions for the updated AdaptivePlanner (Markov-style, no fixed difficulty tiers)

import sys
import os

# Add the planning/ folder to the path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "planning"))

from adaptive_planner import AdaptivePlanner

# Runs a single test scenario through AdaptivePlanner instance.
def run_scenario(name, trials):
    print(f"\n{'='*55}")
    print(f"Scenario: {name}")
    print(f"{'='*55}")

    # Fresh planner for each scenario. No state carries over between scenarios
    planner = AdaptivePlanner()

    for i, (emotion, correct, response_time) in enumerate(trials):
        # Feeds the trial result into the planner
        planner.update(emotion, correct, response_time)
        
        # Asks the planner what to show next
        decision = planner.decide_next()

        # Get summary for readable output
        state_summary = planner.state.summary()
        
        # target_emotion can be None if all emotions are learned
        # Updated decision dictionary so planner chooses target_emotion rather than difficulty tier
        target = decision.get('target_emotion', None)

        # Prints a line showing what emotion was shown and whether the user got it right,
        # rolling accuracy and error streak (frustration signal), what the planner 
        # decided: target emotion
        print(
            f"Trial {i+1:02d} | "
            f"Emotion={emotion:<10}"
            f"Correct={str(correct):<5}"
            f"rt={response_time:.1f}s | "
            f"roll_acc={state_summary['rolling_accuracy']:.2f} "
            f"streak={state_summary['error_streak']} | "
            f"target={decision['target_emotion']}"
        )

## Example Scenerios below:

# Struggling user scenario. Mostly wrong answers and slow response time
# Expected: Planner targets the emotion user struggled the most with -- possible exploration 
struggling = [
    ("happy", False, 5.0),
    ("neutral", False, 4.5),
    ("sad", False, 6.0),
    ("happy", True, 3.0),
    ("sad", False, 5.0),
]
run_scenario("Struggling user", struggling)

# A user getting better. Mostly correct answers and fast response times.
# Expected: Planner sees high accuracy -- utilities will be similar since accuracy is high
getting_better = [
    ("happy", True, 1.0),
    ("neutral", True, 1.2),
    ("sad", True, 1.5),
    ("fear", True, 1.8),
    ("happy", True, 1.1),
]
run_scenario("User improving quickly", getting_better)

# Good overall accuracy, but fails one emotion consistently (fear). 
# Expected: Planner will focus on fear until the user has 'learned' it (learned_threshold is crossed)
emotion_weakness = [
    ("happy", True, 1.5),
    ("neutral", True, 1.5),
    ("fear", False, 4.0),  # repeatedly misses fear
    ("fear", False, 4.5),
    ("fear", False, 5.0),
]
run_scenario("Emotion specific weakness (fear)", emotion_weakness)

# Only 2 trials. 
# Expected behavior: No crash, neutral behavior on trial 1.
# limited data --> rely on default accuracy (0.5) and exploration
cold_start = [
    ("happy", True, 2.0),
    ("sad", False, 3.5),
]
run_scenario("cold start (2 trials)", cold_start)