import math
import numpy as np

class Point:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

def distance(a: Point, b: Point) -> float:
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

def vec(a: Point, b: Point) -> np.ndarray:
    return np.array([b.x - a.x, b.y - a.y, b.z - a.z], dtype=float)

def angle_deg(a: Point, b: Point, c: Point) -> float:
    """Angle at point b (a-b-c) in degrees."""
    v1 = vec(b, a)
    v2 = vec(b, c)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 180.0
    cos_a = float(np.dot(v1, v2) / (n1 * n2))
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))

def detect_left_hand(landmarks: list, debug: bool = False) -> bool:
    """
    Detect ASL letter 'D' for left hand.
    Rules:
    - Index finger extended upward
    - Middle, ring, pinky curled
    - Thumb near curled fingers
    """
    if not landmarks or len(landmarks) < 21:
        if debug: print("invalid input length")
        return False

    # Mirror x for left hand (so we can use right-hand logic)
    mirrored = [Point(1 - lm.x, lm.y, lm.z) for lm in landmarks]
    return _detect_d_internal(mirrored, debug)

def _detect_d_internal(landmarks: list, debug: bool = False) -> bool:
    """Internal D detection using the same logic as the working right hand detector."""
    wrist = landmarks[0]
    thumb_cmc, thumb_mcp, thumb_ip, thumb_tip = landmarks[1], landmarks[2], landmarks[3], landmarks[4]
    index_mcp, index_pip, index_dip, index_tip = landmarks[5], landmarks[6], landmarks[7], landmarks[8]
    middle_mcp, middle_pip, middle_dip, middle_tip = landmarks[9], landmarks[10], landmarks[11], landmarks[12]
    ring_mcp, ring_pip, ring_dip, ring_tip = landmarks[13], landmarks[14], landmarks[15], landmarks[16]
    pinky_mcp, pinky_pip, pinky_dip, pinky_tip = landmarks[17], landmarks[18], landmarks[19], landmarks[20]

    palm_width = distance(index_mcp, pinky_mcp)
    contact_threshold = max(palm_width * 0.4, 0.03)

    # --- Index finger should be extended up ---
    index_angle_pip = angle_deg(index_mcp, index_pip, index_tip)
    index_angle_dip = angle_deg(index_mcp, index_dip, index_tip)
    index_points_up = index_tip.y < index_mcp.y - 0.03
    index_extended = (index_angle_pip > 150 and index_angle_dip > 140 and index_points_up)

    # --- Other fingers curled ---
    def is_curled(mcp, pip, tip):
        angle_val = angle_deg(mcp, pip, tip)
        return angle_val < 120 or distance(tip, mcp) < distance(pip, mcp) * 0.95

    middle_curled = is_curled(middle_mcp, middle_pip, middle_tip)
    ring_curled   = is_curled(ring_mcp, ring_pip, ring_tip)
    pinky_curled  = is_curled(pinky_mcp, pinky_pip, pinky_tip)
    curled_count = sum([middle_curled, ring_curled, pinky_curled])

    # --- Thumb should touch curled fingers but not index ---
    thumb_to_middle = distance(thumb_tip, middle_tip)
    thumb_to_ring   = distance(thumb_tip, ring_tip)
    thumb_to_pinky  = distance(thumb_tip, pinky_tip)
    thumb_to_index  = distance(thumb_tip, index_tip)

    thumb_contacts = (thumb_to_middle < contact_threshold or
                      thumb_to_ring   < contact_threshold or
                      thumb_to_pinky  < contact_threshold)
    thumb_not_touch_index = thumb_to_index > max(palm_width * 0.25, 0.03)

    # --- Index should be separated from middle ---
    index_middle_sep = distance(index_tip, middle_tip)
    sep_ok = index_middle_sep > max(palm_width * 0.3, 0.05)

    ok = index_extended and curled_count >= 2 and thumb_contacts and thumb_not_touch_index and sep_ok

    if debug:
        print("=== LEFT HAND D DETECTION DEBUG ===")
        print("index_angle_pip", round(index_angle_pip,1), 
              "index_angle_dip", round(index_angle_dip,1), 
              "index_points_up", index_points_up, "index_extended", index_extended)
        print("middle_curled", middle_curled, "ring_curled", ring_curled, "pinky_curled", pinky_curled, "curled_count", curled_count)
        print("palm_width", round(palm_width,3), "contact_threshold", round(contact_threshold,3))
        print("thumb_to_mid", round(thumb_to_middle,3), 
              "thumb_to_ring", round(thumb_to_ring,3), 
              "thumb_to_pinky", round(thumb_to_pinky,3), 
              "thumb_to_index", round(thumb_to_index,3))
        print("thumb_contacts", thumb_contacts, "thumb_not_touch_index", thumb_not_touch_index)
        print("index_middle_sep", round(index_middle_sep,3), "sep_ok", sep_ok)
        print("final_ok", ok)

    return ok

def create_sample_left_d_landmarks():
    """Create sample left-hand landmarks forming a 'D' (before mirroring)"""
    lms = []
    lms.append(Point(0.5, 0.8, 0.0))  # wrist
    
    # Thumb (bent inward near curled fingers) - LEFT HAND POSITION
    lms.extend([
        Point(0.55, 0.75),  # thumb_cmc
        Point(0.57, 0.70),  # thumb_mcp  
        Point(0.58, 0.66),  # thumb_ip
        Point(0.54, 0.62)   # thumb_tip (touching curled fingers)
    ])
    
    # Index (extended up) - LEFT HAND POSITION
    lms.extend([
        Point(0.48, 0.65),  # index_mcp
        Point(0.475, 0.50), # index_pip
        Point(0.47, 0.35),  # index_dip
        Point(0.465, 0.20)  # index_tip (extended upward)
    ])
    
    # Middle (curled down) - LEFT HAND POSITION
    lms.extend([
        Point(0.50, 0.65),  # middle_mcp
        Point(0.505, 0.58), # middle_pip
        Point(0.51, 0.61),  # middle_dip
        Point(0.515, 0.63)  # middle_tip (curled)
    ])
    
    # Ring (curled down) - LEFT HAND POSITION
    lms.extend([
        Point(0.52, 0.66),  # ring_mcp
        Point(0.525, 0.59), # ring_pip
        Point(0.53, 0.62),  # ring_dip
        Point(0.535, 0.64)  # ring_tip (curled)
    ])
    
    # Pinky (curled down) - LEFT HAND POSITION
    lms.extend([
        Point(0.54, 0.67),  # pinky_mcp
        Point(0.545, 0.60), # pinky_pip
        Point(0.55, 0.63),  # pinky_dip
        Point(0.555, 0.65)  # pinky_tip (curled)
    ])
    
    return lms

# --- Example Test ---
if __name__ == "__main__":
    print("=== Testing Left Hand D Detector ===")
    
    # Test with sample landmarks
    sample = create_sample_left_d_landmarks()
    result = detect_left_hand_d(sample, debug=True)
    print(f"\nSample landmarks detected as 'D': {result}")
    
    # Test edge cases
    print(f"\nEmpty landmarks: {detect_left_hand_d([])}")
    print(f"None input: {detect_left_hand_d(None)}")
    
    # Show mirroring effect
    print("\n=== Mirroring Demonstration ===")
    print("Original left hand landmarks (first 5 points):")
    for i in range(5):
        lm = sample[i]
        print(f"  Point {i}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")
    
    print("After mirroring for processing:")
    mirrored = [Point(1 - lm.x, lm.y, lm.z) for lm in sample[:5]]
    for i, lm in enumerate(mirrored):
        print(f"  Point {i}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")

    if result:
        print("\n✓ SUCCESS: All detection criteria met for ASL 'D'")
    else:
        print("\n✗ FAILED: Some detection criteria not met")