# gesture_config.py

# Internal Gestures used for ML classification
GESTURES = [
    "open_palm",
    "closed_fist",
    "thumbs_up",
    "thumbs_down",
    "swipe_left",
    "swipe_right"
]

# Bootstrap finger navigation mapping (0 to 5 fingers extended)
FINGER_MAPPING = {
    0: "BACK",     # Closed fist 
    1: "NEXT",     # One finger
    2: "PREV",     # Two fingers
    3: "SELECT",   # Three fingers
    4: "HELP",     # Four fingers
    5: "HOME"      # Five fingers
}
