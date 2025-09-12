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

def detect_left_hand_f_real_camera(landmarks: list, debug: bool = False) -> bool:
    """
    Detect ASL letter 'F' for left hand - based on REAL camera ASL description.
    
    REAL ASL 'F' from camera data:
    - Tip of index finger touches tip of thumb (forming small circle)
    - Middle, ring, and little fingers are straight and slightly spread
    - Palm faces forward toward camera
    - Hand positioned at chest/shoulder level
    
    Args:
        landmarks: list of 21 Point objects from MediaPipe
        debug: if True, prints detailed detection information
    
    Returns:
        bool: True if left hand is making real ASL 'F' shape
    """
    if not landmarks or len(landmarks) < 21:
        if debug: print("❌ Invalid input - need 21 landmarks")
        return False

    # Mirror X-coordinates for left hand processing
    mirrored_landmarks = []
    for lm in landmarks:
        mirrored_landmarks.append(Point(1 - lm.x, lm.y, lm.z))
    
    return _detect_f_real_camera_internal(mirrored_landmarks, debug, hand_type="LEFT")

def _detect_f_real_camera_internal(landmarks: list, debug: bool = False, hand_type: str = "LEFT") -> bool:
    """
    Internal F detection based on real camera ASL description.
    """
    # Extract landmarks
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    
    index_tip = landmarks[8]
    index_dip = landmarks[7]
    index_pip = landmarks[6]
    index_mcp = landmarks[5]
    
    middle_tip = landmarks[12]
    middle_dip = landmarks[11]
    middle_pip = landmarks[10]
    middle_mcp = landmarks[9]
    
    ring_tip = landmarks[16]
    ring_dip = landmarks[15]
    ring_pip = landmarks[14]
    ring_mcp = landmarks[13]
    
    pinky_tip = landmarks[20]
    pinky_dip = landmarks[19]
    pinky_pip = landmarks[18]
    pinky_mcp = landmarks[17]

    # Calculate hand dimensions
    palm_width = distance(index_mcp, pinky_mcp)
    palm_length = distance(wrist, middle_mcp)

    # --- 1. THUMB-INDEX TIP CONTACT (Most Critical) ---
    # Based on real description: "Tip of index finger touches tip of thumb"
    thumb_index_distance = distance(thumb_tip, index_tip)
    tips_touching = thumb_index_distance < palm_width * 0.25  # Tighter threshold for real F
    
    # Index finger should be partially bent (not fully extended)
    index_length = distance(index_tip, index_mcp)
    index_expected_full_length = palm_length * 0.8
    index_partially_bent = index_length < index_expected_full_length * 0.85
    
    # Thumb should be in natural pinching position
    thumb_length = distance(thumb_tip, thumb_mcp)
    thumb_natural_position = thumb_length > distance(thumb_ip, thumb_mcp) * 0.9

    # --- 2. THREE FINGERS STRAIGHT AND SLIGHTLY SPREAD ---
    # Based on: "Middle, ring and little fingers are straight and slightly spread"
    extended_fingers_data = [
        (middle_tip, middle_dip, middle_pip, middle_mcp, "Middle"),
        (ring_tip, ring_dip, ring_pip, ring_mcp, "Ring"),
        (pinky_tip, pinky_dip, pinky_pip, pinky_mcp, "Pinky")
    ]
    
    def is_finger_straight_and_extended(tip, dip, pip, mcp, finger_name=""):
        """Check if finger is straight and extended (real ASL F criteria)"""
        # Finger should be extended upward
        points_up = tip.y < mcp.y - palm_width * 0.2
        
        # Finger should be relatively straight
        finger_length = distance(tip, mcp)
        expected_length = palm_length * 0.7  # Realistic length for straight finger
        is_long_enough = finger_length > expected_length
        
        # Check that finger joints are extended (not curled)
        tip_to_pip = distance(tip, pip)
        pip_to_mcp = distance(pip, mcp)
        joints_extended = tip_to_pip > pip_to_mcp * 0.6  # Finger extended beyond pip
        
        finger_good = points_up and is_long_enough and joints_extended
        
        if debug:
            print(f"  {finger_name}: up={points_up}, long={is_long_enough}, extended={joints_extended} → {finger_good}")
            print(f"    tip.y={tip.y:.3f}, mcp.y={mcp.y:.3f}, length={finger_length:.3f}")
        
        return finger_good

    # Check all three extended fingers
    straight_fingers_count = 0
    finger_results = {}
    
    for tip, dip, pip, mcp, name in extended_fingers_data:
        result = is_finger_straight_and_extended(tip, dip, pip, mcp, name)
        finger_results[name] = result
        if result:
            straight_fingers_count += 1

    # --- 3. FINGER SPREAD CHECK ---
    # "slightly spread" - fingers should have some separation but not too much
    extended_tips = [middle_tip, ring_tip, pinky_tip]
    
    # Check spacing between adjacent fingers
    middle_ring_gap = distance(middle_tip, ring_tip)
    ring_pinky_gap = distance(ring_tip, pinky_tip)
    
    # Should be spread but not too wide
    min_spread = palm_width * 0.15  # Minimum spread
    max_spread = palm_width * 0.5   # Maximum spread
    
    good_spread = (min_spread < middle_ring_gap < max_spread and 
                   min_spread < ring_pinky_gap < max_spread)

    # --- 4. INDEX FINGER POSITIONING ---
    # Index should be clearly separated from the extended fingers
    # (It's forming the circle with thumb, not part of the extended group)
    index_to_middle = distance(index_tip, middle_tip)
    index_separated_from_group = index_to_middle > palm_width * 0.3
    
    # Index should not be pointing up like the others
    index_not_pointing_up = index_tip.y > middle_tip.y - palm_width * 0.1

    # --- 5. PALM ORIENTATION ---
    # Palm should face forward (toward camera)
    v1 = vec(wrist, index_mcp)
    v2 = vec(wrist, pinky_mcp)
    normal = np.cross(v1, v2)
    palm_facing_forward = normal[2] > -0.3  # Allow some camera angle variation

    # --- FINAL DECISION ---
    contact_ok = tips_touching and index_partially_bent and thumb_natural_position
    fingers_ok = straight_fingers_count >= 2  # At least 2 of 3 fingers straight
    spread_ok = good_spread
    position_ok = index_separated_from_group and index_not_pointing_up
    orientation_ok = palm_facing_forward
    
    final_result = contact_ok and fingers_ok and spread_ok and position_ok and orientation_ok

    # Debug output
    if debug:
        print(f"=== {hand_type} HAND F DETECTION (Real Camera ASL) ===")
        print(f"📏 Palm dimensions: width={palm_width:.3f}, length={palm_length:.3f}")
        print(f"👌 Thumb-Index contact:")
        print(f"   - Distance: {thumb_index_distance:.3f} (limit: {palm_width * 0.25:.3f})")
        print(f"   - Tips touching: {tips_touching}")
        print(f"   - Index partially bent: {index_partially_bent}")
        print(f"   - Thumb natural: {thumb_natural_position}")
        print(f"👆 Straight extended fingers: {straight_fingers_count}/3 (need ≥2)")
        for name, result in finger_results.items():
            print(f"   - {name}: {result}")
        print(f"🤚 Finger spread:")
        print(f"   - Middle-Ring gap: {middle_ring_gap:.3f}")
        print(f"   - Ring-Pinky gap: {ring_pinky_gap:.3f}")
        print(f"   - Good spread: {good_spread}")
        print(f"☝️ Index positioning:")
        print(f"   - Separated from group: {index_separated_from_group}")
        print(f"   - Not pointing up: {index_not_pointing_up}")
        print(f"🤲 Palm orientation: {palm_facing_forward}")
        print(f"🎯 CRITERIA: contact_ok={contact_ok}, fingers_ok={fingers_ok}, spread_ok={spread_ok}")
        print(f"           position_ok={position_ok}, orientation_ok={orientation_ok}")
        print(f"🏆 FINAL RESULT: {'✅ REAL ASL F DETECTED' if final_result else '❌ NOT F'}")

    return final_result

def create_real_asl_left_f_landmarks():
    """Create left-hand landmarks based on REAL ASL F description (before mirroring)"""
    landmarks = []
    
    # Wrist
    landmarks.append(Point(0.5, 0.9, 0.0))
    
    # Thumb - in natural pinching position to touch index tip (LEFT HAND)
    landmarks.extend([
        Point(0.54, 0.83, 0.01),  # thumb_cmc
        Point(0.56, 0.78, 0.02),  # thumb_mcp
        Point(0.54, 0.74, 0.03),  # thumb_ip
        Point(0.52, 0.72, 0.04)   # thumb_tip (touching index tip)
    ])
    
    # Index finger - partially bent to touch thumb tip (LEFT HAND)
    landmarks.extend([
        Point(0.48, 0.75, 0.0),   # index_mcp
        Point(0.49, 0.73, 0.01),  # index_pip
        Point(0.51, 0.72, 0.02),  # index_dip (bent toward thumb)
        Point(0.52, 0.72, 0.03)   # index_tip (touching thumb tip)
    ])
    
    # Middle finger - straight and extended (LEFT HAND)
    landmarks.extend([
        Point(0.50, 0.73, 0.0),   # middle_mcp
        Point(0.51, 0.58, 0.01),  # middle_pip
        Point(0.52, 0.42, 0.02),  # middle_dip
        Point(0.53, 0.25, 0.03)   # middle_tip (straight up)
    ])
    
    # Ring finger - straight and extended (LEFT HAND)
    landmarks.extend([
        Point(0.52, 0.74, 0.0),   # ring_mcp
        Point(0.54, 0.59, 0.01),  # ring_pip
        Point(0.55, 0.43, 0.02),  # ring_dip
        Point(0.56, 0.27, 0.03)   # ring_tip (straight up, spread from middle)
    ])
    
    # Pinky finger - straight and extended (LEFT HAND)
    landmarks.extend([
        Point(0.54, 0.76, 0.0),   # pinky_mcp
        Point(0.56, 0.61, 0.01),  # pinky_pip
        Point(0.57, 0.45, 0.02),  # pinky_dip
        Point(0.58, 0.30, 0.03)   # pinky_tip (straight up, spread from ring)
    ])
    
    return landmarks

# Example usage and testing
if __name__ == "__main__":
    print("🤖 LEFT HAND ASL 'F' DETECTOR - Real Camera Description")
    print("=" * 70)
    print("Based on real ASL description:")
    print("• Tip of index finger touches tip of thumb")  
    print("• Middle, ring, and little fingers are straight and slightly spread")
    print("• Palm faces forward toward camera")
    print("• Automatic coordinate mirroring for left hand")
    print("=" * 70)
    
    # Test with real ASL F landmarks
    sample_landmarks = create_real_asl_left_f_landmarks()
    result = detect_left_hand_f_real_camera(sample_landmarks, debug=True)
    
    print(f"\n{'='*70}")
    if result:
        print("🎉 SUCCESS: Real ASL 'F' detected for left hand!")
        print("✅ This matches the actual ASL fingerspelling 'F':")
        print("   • Thumb-index tips touching (small precise circle)")
        print("   • Three fingers straight and slightly spread")
        print("   • Proper palm orientation")
        print("   • Natural hand positioning for camera")
        print("   • Left hand mirroring applied correctly")
    else:
        print("❌ FAILED: Adjust hand to match real ASL 'F'")
        print("📋 Real ASL 'F' checklist:")
        print("   1. Touch thumb TIP to index finger TIP")
        print("   2. Keep middle, ring, pinky straight and slightly apart")
        print("   3. Palm should face toward camera") 
        print("   4. Hand at chest/shoulder height")
        print("   5. Index finger partially bent (not fully extended)")
    
    # Show mirroring effect
    print(f"\n🔄 Left hand mirroring demonstration:")
    print("Original left hand coordinates (first 5 points):")
    for i in range(5):
        lm = sample_landmarks[i]
        print(f"  Point {i}: x={lm.x:.3f}, y={lm.y:.3f}")
    
    mirrored = [Point(1 - lm.x, lm.y, lm.z) for lm in sample_landmarks[:5]]
    print("After mirroring for processing:")
    for i, lm in enumerate(mirrored):
        print(f"  Point {i}: x={lm.x:.3f}, y={lm.y:.3f}")
    
    if result:
        print("\n✓ SUCCESS: All detection criteria met for ASL 'F'")
    else:
        print("\n✗ FAILED: Some detection criteria not met")
        
    # Test edge cases
    print(f"\nEdge case tests:")
    print(f"Empty list: {detect_left_hand([])}")
    print(f"None input: {detect_left_hand(None)}")
    print(f"Short list: {detect_left_hand([Point(0,0,0)] * 10)}")
    
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
        print(f"\n🎉 SUCCESS: Perfect ASL 'F' shape detected!")
    else:
        print(f"\n❌ FAILED: Detection criteria not met")
    