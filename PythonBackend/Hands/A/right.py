class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def detect_right_hand(points):
    """
    Detects the letter 'A' using the right hand.
    `points` is a list of Point objects corresponding to MediaPipe landmarks.
    Returns "A" if detected, else "".
    """

    translation = ""

    # Landmarks indices for fingers:
    # Thumb: 1,2,3,4 | Index: 5,6,7,8 | Middle: 9,10,11,12 | Ring: 13,14,15,16 | Pinky: 17,18,19,20

    # Check if fingers (except thumb) are folded
    fingers_folded = all([
        points[8].y > points[6].y,   # Index finger tip below PIP joint
        points[12].y > points[10].y, # Middle finger
        points[16].y > points[14].y, # Ring finger
        points[20].y > points[18].y  # Pinky
    ])

    # Thumb is extended out to the side (not folded)
    thumb_extended = points[4].x > points[3].x  # Right hand: thumb tip x > IP joint x

    if fingers_folded and thumb_extended:
        translation = "A"

    return translation
