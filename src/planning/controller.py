from .adaptive_planner import AdaptivePlanner

planner = AdaptivePlanner()

def run_step(true_emotion, user_answer, response_time):
    # 1. Check if user is correct
    correct = (user_answer == true_emotion)

    # 2. Update planner with user performance
    planner.update(true_emotion, correct, response_time, user_answer)

    # 3. Decide next action
    decision = planner.decide_next()

    return {
        "true_emotion": true_emotion, # true emotion of image -- correct answer
        "user_answer": user_answer,
        "correct": correct,
        "next_step": decision
    }