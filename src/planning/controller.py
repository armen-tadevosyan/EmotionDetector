from .adaptive_planner import AdaptivePlanner
from predict import predict_frame

planner = AdaptivePlanner()


def run_step(frame, user_answer, response_time):
    # 1. AI determines correct emotion from image
    true_emotion, confidence, probs = predict_frame(frame)

    # 2. Check if user is correct
    correct = (user_answer == true_emotion)

    # 3. Update planner with user performance
    planner.update(true_emotion, correct, response_time, user_answer)

    # 4. Decide next action
    decision = planner.decide_next()

    return {
        "true_emotion": true_emotion, # true emotion of image -- correct answer
        "user_answer": user_answer,
        "confidence": confidence,
        "correct": correct,
        "next_step": decision
    }