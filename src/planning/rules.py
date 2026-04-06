# rules.py -- reasoning layer
# maps each emotion to a few short explanations of the facial cues that signal it
# the game picks one randomly after each trial to show the user why a face looks the way it does

import random

EMOTION_RULES = {
    "happy": [
        "The corners of the mouth are pulled upward into a smile.",
        "The cheeks are raised and the eyes may appear slightly squinted -- this is called a Duchenne smile.",
        "Look for the upward curve at both sides of the lips. That's the clearest sign of happiness.",
    ],
    "sad": [
        "The inner corners of the eyebrows are pulled upward and together, forming a slope.",
        "The corners of the mouth are turned downward, and the lower lip may jut out slightly.",
        "The eyes often look droopy or unfocused, like the person is about to cry.",
    ],
    "angry": [
        "The eyebrows are pulled down and together, forming a hard V-shape above the nose.",
        "The lips are often pressed tightly together or pulled back to show teeth.",
        "The eyes are narrowed into a sharp, focused stare.",
    ],
    "surprise": [
        "The eyebrows are raised high and arched -- much higher than normal.",
        "The eyes are very wide open, showing a lot of white above and below the iris.",
        "The mouth is open in a round 'O' shape, like the person just heard unexpected news.",
    ],
    "fear": [
        "The eyebrows are raised AND pulled together at the same time -- this separates fear from surprise.",
        "The upper eyelids are raised but the lower lids are tense and raised too.",
        "The mouth is often pulled back at the corners horizontally, like a grimace.",
    ],
    "disgust": [
        "The nose is wrinkled, as if the person is smelling something unpleasant.",
        "The upper lip is raised on one or both sides, sometimes showing the upper teeth.",
        "The eyebrows are slightly lowered and the eyes may be narrowed.",
    ],
    "neutral": [
        "The face is completely relaxed -- no muscle tension in the brows, cheeks, or mouth.",
        "The mouth is gently closed and the eyes have a soft, unfocused look.",
        "Neutral is what a face looks like when no particular emotion is being felt or shown.",
    ],
}

GENERAL_TIPS = [
    "Tip: Focus on the eyebrows first -- they give away a lot!",
    "Tip: The mouth shape is often the clearest clue.",
    "Tip: Compare the left and right sides of the face -- are they symmetrical?",
    "Tip: Eye openness is a big signal. Wide = surprise or fear. Narrow = anger or disgust.",
]


def get_explanation(emotion: str) -> str:
    explanations = EMOTION_RULES.get(emotion.lower())
    if not explanations:
        return f"The face is showing {emotion}."
    return random.choice(explanations)


def get_tip() -> str:
    return random.choice(GENERAL_TIPS)
