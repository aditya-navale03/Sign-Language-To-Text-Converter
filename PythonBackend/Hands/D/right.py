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

def detect_right_hand(landmarks: list, debug: bool = False) -> bool:
    """
    Detect ASL letter 'D' for right hand.
    
    In ASL 'D':
    - Index finger is extended upward
    - Middle, ring, and pinky fingers are curled/closed
    - Thumb is extended and touches the curled fingers
    - Palm faces forward (toward camera)
    
    Args:
        landmarks: list of 21 Point objects representing hand landmarks
        debug: if True, prints detailed detection information
    
    Returns:
        bool: True if right hand is making ASL 'D' shape, False otherwise
    """
    if not landmarks or len(landmarks) < 21:
        if debug: print("Invalid input - need 21 landmarks")
        return False

    # Extract landmarks
    wrist = landmarks[0]
    thumb_cmc, thumb_mcp, thumb_ip, thumb_tip = landmarks[1], landmarks[2], landmarks[3], landmarks[4]
    index_mcp, index_pip, index_dip, index_tip = landmarks[5], landmarks[6], landmarks[7], landmarks[8]
    middle_mcp, middle_pip, middle_dip, middle_tip = landmarks[9], landmarks[10], landmarks[11], landmarks[12]
    ring_mcp, ring_pip, ring_dip, ring_tip = landmarks[13], landmarks[14], landmarks[15], landmarks[16]
    pinky_mcp, pinky_pip, pinky_dip, pinky_tip = landmarks[17], landmarks[18], landmarks[19], landmarks[20]

    # Calculate palm dimensions
    palm_width = distance(index_mcp, pinky_mcp)
    contact_threshold = max(palm_width * 0.4, 0.03)

    # --- 1. Index finger should be extended upward ---
    # Check angles at joints for full extension
    index_angle_pip = angle_deg(index_mcp, index_pip, index_tip)
    index_angle_dip = angle_deg(index_mcp, index_dip, index_tip)
    
    # Check if index points upward (y decreases from mcp to tip)
    index_points_up = index_tip.y < index_mcp.y - 0.03
    
    # Index is extended if angles are large (straight) and points up
    index_extended = (index_angle_pip > 150 and index_angle_dip > 140 and index_points_up)

    # --- 2. Other fingers (middle, ring, pinky) should be curled ---
    def is_curled(mcp, pip, tip):
        """Check if a finger is curled based on angle and tip position"""
        angle_val = angle_deg(mcp, pip, tip)
        # Finger is curled if angle is small or tip is close to mcp
        return angle_val < 120 or distance(tip, mcp) < distance(pip, mcp) * 0.95

    middle_curled = is_curled(middle_mcp, middle_pip, middle_tip)
    ring_curled = is_curled(ring_mcp, ring_pip, ring_tip)
    pinky_curled = is_curled(pinky_mcp, pinky_pip, pinky_tip)
    curled_count = sum([middle_curled, ring_curled, pinky_curled])

    # --- 3. Thumb should touch curled fingers but not index ---
    thumb_to_middle = distance(thumb_tip, middle_tip)
    thumb_to_ring = distance(thumb_tip, ring_tip)
    thumb_to_pinky = distance(thumb_tip, pinky_tip)
    thumb_to_index = distance(thumb_tip, index_tip)

    # Thumb should be close to at least one curled finger
    thumb_contacts = (thumb_to_middle < contact_threshold or
                      thumb_to_ring < contact_threshold or
                      thumb_to_pinky < contact_threshold)
    
    # Thumb should not touch the extended index finger
    thumb_not_touch_index = thumb_to_index > max(palm_width * 0.25, 0.03)

    # --- 4. Index should be separated from middle finger ---
    index_middle_sep = distance(index_tip, middle_tip)
    sep_ok = index_middle_sep > max(palm_width * 0.3, 0.05)

    # --- Final validation ---
    all_criteria_met = (index_extended and 
                       curled_count >= 2 and 
                       thumb_contacts and 
                       thumb_not_touch_index and 
                       sep_ok)

    # Debug output
    if debug:
        print("=== RIGHT HAND D DETECTION DEBUG ===")
        print(f"Palm width: {palm_width:.3f}, Contact threshold: {contact_threshold:.3f}")
        print(f"Index finger:")
        print(f"  - Angle PIP: {index_angle_pip:.1f}° (need >150°)")
        print(f"  - Angle DIP: {index_angle_dip:.1f}° (need >140°)")
        print(f"  - Points up: {index_points_up}")
        print(f"  - Extended: {index_extended}")
        print(f"Curled fingers:")
        print(f"  - Middle curled: {middle_curled}")
        print(f"  - Ring curled: {ring_curled}")
        print(f"  - Pinky curled: {pinky_curled}")
        print(f"  - Curled count: {curled_count}/3 (need ≥2)")
        print(f"Thumb distances:")
        print(f"  - To middle: {thumb_to_middle:.3f}")
        print(f"  - To ring: {thumb_to_ring:.3f}")
        print(f"  - To pinky: {thumb_to_pinky:.3f}")
        print(f"  - To index: {thumb_to_index:.3f}")
        print(f"  - Contacts curled: {thumb_contacts}")
        print(f"  - Doesn't touch index: {thumb_not_touch_index}")
        print(f"Separation:")
        print(f"  - Index-Middle: {index_middle_sep:.3f} (need >{max(palm_width * 0.3, 0.05):.3f})")
        print(f"  - Separation OK: {sep_ok}")
        print(f"FINAL RESULT: {all_criteria_met}")

    return all_criteria_met

def create_sample_right_d_landmarks():
    """Create sample right-hand landmarks forming a perfect 'D' shape."""
    landmarks = []
    
    # Wrist
    landmarks.append(Point(0.5, 0.8, 0.0))
    
    # Thumb (bent inward to touch curled fingers)
    landmarks.extend([
        Point(0.45, 0.75, 0.01),  # thumb_cmc
        Point(0.43, 0.70, 0.02),  # thumb_mcp
        Point(0.42, 0.66, 0.03),  # thumb_ip
        Point(0.46, 0.62, 0.04)   # thumb_tip (touching curled fingers)
    ])
    
    # Index finger (fully extended upward)
    landmarks.extend([
        Point(0.52, 0.65, 0.0),   # index_mcp
        Point(0.525, 0.50, 0.01), # index_pip
        Point(0.53, 0.35, 0.02),  # index_dip
        Point(0.535, 0.20, 0.03)  # index_tip (extended up)
    ])
    
    # Middle finger (curled down toward palm)
    landmarks.extend([
        Point(0.50, 0.65, 0.0),   # middle_mcp
        Point(0.495, 0.58, 0.01), # middle_pip
        Point(0.49, 0.61, 0.02),  # middle_dip
        Point(0.485, 0.63, 0.03)  # middle_tip (curled)
    ])
    
    # Ring finger (curled down toward palm)
    landmarks.extend([
        Point(0.48, 0.66, 0.0),   # ring_mcp
        Point(0.475, 0.59, 0.01), # ring_pip
        Point(0.47, 0.62, 0.02),  # ring_dip
        Point(0.465, 0.64, 0.03)  # ring_tip (curled)
    ])
    
    # Pinky finger (curled down toward palm)
    landmarks.extend([
        Point(0.46, 0.67, 0.0),   # pinky_mcp
        Point(0.455, 0.60, 0.01), # pinky_pip
        Point(0.45, 0.63, 0.02),  # pinky_dip
        Point(0.445, 0.65, 0.03)  # pinky_tip (curled)
    ])
    
    return landmarks

# Example usage and testing
if __name__ == "__main__":
    print("=== RIGHT HAND ASL 'D' DETECTOR TEST ===\n")
    
    # Test with sample landmarks
    sample_landmarks = create_sample_right_d_landmarks()
    result = detect_right_hand_d(sample_landmarks, debug=True)
    
    print(f"\n{'='*50}")
    print(f"Sample landmarks result: {'✓ DETECTED' if result else '✗ NOT DETECTED'}")
    
    # Test edge cases
    print(f"\nEdge case tests:")
    print(f"Empty list: {detect_right_hand_d([])}")
    print(f"None input: {detect_right_hand_d(None)}")
    print(f"Short list: {detect_right_hand_d([Point(0,0,0)] * 10)}")
    
    # Show landmark positions
    print(f"\nSample landmark positions:")
    landmark_names = ['wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
                     'index_mcp', 'index_pip', 'index_dip', 'index_tip',
                     'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
                     'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
                     'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip']
    
    for i, (name, lm) in enumerate(zip(landmark_names, sample_landmarks)):
        print(f"  {i:2d}. {name:12s}: ({lm.x:.3f}, {lm.y:.3f}, {lm.z:.3f})")
    
    if result:
        print(f"\n🎉 SUCCESS: Perfect ASL 'D' shape detected!")
    else:
        print(f"\n❌ FAILED: Detection criteria not met")