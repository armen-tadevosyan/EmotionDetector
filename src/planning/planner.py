from planning.state import State

# put the different actions here like where they got confused, 
# where to increase/decrease difficulty, where to target confusion
ACTIONS = []

DIFFICULTY_ORDER = ["easy", "medium", "hard"]

class Planner:
    def __init__(self):
        self.state = State()

    def update_state(self, true_emotion, user_)