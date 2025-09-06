# Hands/A/left.py

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def detect_left_hand(points):
    """
    Detect letter 'A' for left hand using MediaPipe landmarks.
    `points` is a list of Point objects corresponding to landmarks.
    """
    translation = ""

    # Finger tips indices: 8=index, 12=middle, 16=ring, 20=pinky
    # Finger base indices: 5=index, 9=middle, 13=ring, 17=pinky

    fingers_folded = all([
        points[8].y > points[5].y,   # Index folded
        points[12].y > points[9].y,  # Middle folded
        points[16].y > points[13].y, # Ring folded
        points[20].y > points[17].y  # Pinky folded
    ])

    # Thumb extended to the side (for left hand)
    # Thumb tip x should be less than base x (points[4].x < points[2].x)
    thumb_extended = points[4].x < points[2].x

    if fingers_folded and thumb_extended:
        translation = "A"

    return translation
