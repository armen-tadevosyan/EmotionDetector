# gives fake session data into the planner
# Tests the planning layer without images, model, and pygame
# Each scenario constrcuts a sequence that simulates a specific type of 
# user, then runs them through AdaptivePlanner and prints what the planner
# decides after each trial

# Run it from src with: python simulate.py

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

        # Grab a summary snapshot for readable output
        state_summary = planner.state.summary()

        # Prints a line showing what emotion was shown and whether the user got it right,
        # rolling accuracy and error streak (frustration signal), what the planner 
        # decided: difficulty level and target emotion (if any)
        print(
            f"Trial {i+1:02d} | "
            f"Emotion={emotion:<10}"
            f"Correct={str(correct):<5}"
            f"rt={response_time:.1f}s | "
            f"roll_acc={state_summary['rolling_accuracy']:.2f} "
            f"streak={state_summary['error_streak']} | "
            f"diff={decision['difficulty']} "
            f"target={decision['target_emotion']}"
        )

# Struggling user scenario. Mostly wrong answers and slow response time
# Expected behavior: difficulty stays at 1, planner targets the most missed emotion
struggling = (
    [("happy", False, 5.0)] * 3 +
    [("neutral", False, 4.5)] * 3 +
    [("sad", False, 6.0)] * 2 + 
    [("happy", True, 3.0)] * 2
)
run_scenario("Struggling user who is mostly wrong and slow", struggling)

# A user getting better. Mostly correct answers and fast response times.
# Expected behavior: difficulty climbs from 1 to 2 to 3 and holds at 3
getting_better = (
    [("happy", True, 1.0)] * 4 + 
    [("neutral", True, 1.2)] * 3 + 
    [("sad", True, 1.5)] * 3  
)
run_scenario("User who is getting better. They are mostly correct and fast", getting_better)

# Good overall accuracy, but fails one emotion consistently (fear). 
# Expected behavior: planner reaches diff=3 (where fear is), then 
# locks target_emotion=fear as soon as the misses start.
emotion_weakness = (
    [("happy", True, 1.5)] * 3 +
    [("neutral", True, 1.5)] * 3 +
    [("fear", False, 5)] * 4 # repeatedly misses fear
)
run_scenario("User who has an emotion specific weakness like fear.", emotion_weakness)

# Only 2 trials. 
# Expected behavior: No crash, neutral behavior on trial 1.
# target_confusion kicks in on trial 2 after the first miss.
cold_start = [
    ("happy", True, 2.0),
    ("sad", False, 3.5),
]
run_scenario("cold start (2 trials)", cold_start)