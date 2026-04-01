# gives fake session data into the planner

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "planning"))

from adaptive_planner import AdaptivePlanner

def run_scenario(name, trials):
    print(f"\n{'='*55}")
    print(f"Scenario: {name}")
    print(f"{'='*55}")

    planner = AdaptivePlanner()

    for i, (emotion, correct, response_time) in enumerate(trials):
        planner.update(emotion, correct, response_time)
        decision = planner.decide_next()
        state_summary = planner.state.summary()

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

# struggling user scenario
struggling = (
    [("happy", False, 5.0)] * 3 +
    [("neutral", False, 4.5)] * 3 +
    [("sad", False, 6.0)] * 2 + 
    [("happy", True, 3.0)] * 2
)
run_scenario("Struggling user who is mostly wrong and slow", struggling)

# good user scenario
getting_better = (
    [("happy", True, 1.0)] * 4 + 
    [("neutral", True, 1.2)] * 3 + 
    [("sad", True, 1.5)] * 3  
)
run_scenario("User who is getting better. They are mostly correct and fast", getting_better)

# emotion specific weakness
emotion_weakness = (
    [("happy", True, 1.5)] * 3 +
    [("neutral", True, 1.5)] * 3 +
    [("fear", False, 5)] * 4
)
run_scenario("User who has an emotion specific weakness like fear.", emotion_weakness)

# user with a cold start 
cold_start = [
    ("happy", True, 2.0),
    ("sad", False, 3.5),
]
run_scenario("cold start (2 trials)", cold_start)