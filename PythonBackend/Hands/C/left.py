import math
import numpy as np

class Point:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

def distance(a: Point, b: Point) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

def detect_left_hand(landmarks: list) -> bool:
    """
    Detect ASL letter 'C' for left hand (palm facing camera).
    
    Args:
        landmarks: list of 21 Point objects representing hand landmarks
                  Following MediaPipe hand landmark convention:
                  0: WRIST
                  1-4: THUMB (CMC, MCP, IP, TIP)
                  5-8: INDEX (MCP, PIP, DIP, TIP)
                  9-12: MIDDLE (MCP, PIP, DIP, TIP)
                  13-16: RING (MCP, PIP, DIP, TIP)
                  17-20: PINKY (MCP, PIP, DIP, TIP)
    
    Returns:
        bool: True if left hand is making ASL 'C' shape, False otherwise
    """
    if not landmarks or len(landmarks) < 21:
        return False
    
    # Create mirrored landmarks for left hand to align with right-hand logic
    # Mirror X-axis: x_mirrored = 1 - x_original
    mirrored_landmarks = []
    for lm in landmarks:
        mirrored_landmarks.append(Point(1 - lm.x, lm.y, lm.z))
    
    return _detect_c_shape_internal(mirrored_landmarks)

def _detect_c_shape_internal(landmarks: list) -> bool:
    """
    Internal function to detect C shape after proper orientation alignment.
    Assumes right-hand coordinate system (used for both hands after mirroring).
    
    Args:
        landmarks: list of 21 Point objects (already mirrored if needed)
    
    Returns:
        bool: True if hand landmarks form ASL 'C' shape
    """
    # Key landmarks
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    
    index_tip = landmarks[8]
    index_pip = landmarks[7]
    index_mcp = landmarks[5]
    
    middle_tip = landmarks[12]
    middle_pip = landmarks[11]
    middle_mcp = landmarks[9]
    
    ring_tip = landmarks[16]
    ring_pip = landmarks[15]
    ring_mcp = landmarks[13]
    
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[19]
    pinky_mcp = landmarks[17]
    
    # --- 1. Palm orientation check ---
    # Calculate palm normal vector
    v1 = np.array([index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z])
    v2 = np.array([pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z])
    normal = np.cross(v1, v2)
    
    # Palm should be facing camera (positive z-direction)
    if normal[2] < 0:
        return False
    
    # --- 2. Finger curvature check ---
    # All four fingers should be curved (tips closer to wrist than expected if straight)
    fingers = [
        (index_tip, index_mcp),
        (middle_tip, middle_mcp),
        (ring_tip, ring_mcp),
        (pinky_tip, pinky_mcp)
    ]
    
    curved_count = 0
    for tip, mcp in fingers:
        # Check if finger is curved by comparing tip-wrist distance to mcp-wrist distance
        tip_to_wrist = distance(tip, wrist)
        mcp_to_wrist = distance(mcp, wrist)
        
        # For curved finger, tip should be closer to wrist than if fully extended
        # Also check that tips are not too far forward (z-direction)
        if tip_to_wrist > mcp_to_wrist * 0.9 and tip.z > mcp.z - 0.05:
            curved_count += 1
    
    if curved_count < 3:  # At least 3 fingers should be curved
        return False
    
    # --- 3. Thumb positioning ---
    # Thumb should be extended and positioned to form C shape with other fingers
    thumb_extended = distance(thumb_tip, thumb_mcp) > distance(thumb_ip, thumb_mcp)
    
    # Thumb should be separated from other fingers
    thumb_to_index_dist = distance(thumb_tip, index_tip)
    thumb_to_pinky_dist = distance(thumb_tip, pinky_tip)
    palm_width = distance(index_mcp, pinky_mcp)
    
    # Thumb should be at appropriate distance to form C opening
    thumb_separation_ok = (0.6 * palm_width < thumb_to_index_dist < 1.8 * palm_width and
                          0.8 * palm_width < thumb_to_pinky_dist < 2.0 * palm_width)
    
    if not (thumb_extended and thumb_separation_ok):
        return False
    
    # --- 4. C-shape arc validation ---
    # Check that fingertips form a smooth arc
    fingertips = [index_tip, middle_tip, ring_tip, pinky_tip]
    tip_distances_from_wrist = [distance(tip, wrist) for tip in fingertips]
    
    # Check for relatively smooth progression (no sudden jumps)
    smooth_arc = True
    for i in range(len(tip_distances_from_wrist) - 1):
        diff = abs(tip_distances_from_wrist[i] - tip_distances_from_wrist[i + 1])
        if diff > 0.08:  # Allow some variation but not too much
            smooth_arc = False
            break
    
    # --- 5. Overall hand shape validation ---
    # Check that fingers are not fully extended (would be more like a "5")
    fingers_not_extended = all(
        distance(tip, mcp) < distance(mcp, wrist) * 1.2
        for tip, mcp in fingers
    )
    
    # Check that fingers are not fully closed (would be more like a fist)
    fingers_not_closed = all(
        distance(tip, wrist) > distance(mcp, wrist) * 0.8
        for tip, mcp in fingers
    )
    
    return smooth_arc and fingers_not_extended and fingers_not_closed

# Example usage and testing
if __name__ == "__main__":
    def create_sample_left_c_landmarks():
        """Create sample landmarks that represent a 'C' shape for left hand."""
        # Note: These are left hand coordinates (before mirroring)
        landmarks = []
        
        # Wrist
        landmarks.append(Point(0.5, 0.8, 0.0))
        
        # Thumb (4 points: CMC, MCP, IP, TIP) - positioned for left hand
        landmarks.extend([
            Point(0.55, 0.75, 0.01),  # CMC
            Point(0.6, 0.7, 0.02),    # MCP
            Point(0.65, 0.65, 0.03),  # IP
            Point(0.7, 0.6, 0.04)     # TIP
        ])
        
        # Index finger (4 points: MCP, PIP, DIP, TIP)
        landmarks.extend([
            Point(0.45, 0.65, 0.0),   # MCP
            Point(0.42, 0.5, 0.01),   # PIP
            Point(0.4, 0.4, 0.02),    # DIP
            Point(0.38, 0.32, 0.03)   # TIP
        ])
        
        # Middle finger (4 points: MCP, PIP, DIP, TIP)
        landmarks.extend([
            Point(0.5, 0.62, 0.0),    # MCP
            Point(0.48, 0.45, 0.01),  # PIP
            Point(0.47, 0.35, 0.02),  # DIP
            Point(0.46, 0.25, 0.03)   # TIP
        ])
        
        # Ring finger (4 points: MCP, PIP, DIP, TIP)
        landmarks.extend([
            Point(0.55, 0.63, 0.0),   # MCP
            Point(0.54, 0.48, 0.01),  # PIP
            Point(0.53, 0.38, 0.02),  # DIP
            Point(0.52, 0.3, 0.03)    # TIP
        ])
        
        # Pinky finger (4 points: MCP, PIP, DIP, TIP)
        landmarks.extend([
            Point(0.6, 0.65, 0.0),    # MCP
            Point(0.61, 0.52, 0.01),  # PIP
            Point(0.62, 0.42, 0.02),  # DIP
            Point(0.63, 0.35, 0.03)   # TIP
        ])
        
        return landmarks
    
    # Test the left hand detector
    sample_landmarks = create_sample_left_c_landmarks()
    result = detect_left_hand_c(sample_landmarks)
    
    print("Left Hand ASL 'C' Detector Test:")
    print(f"Sample landmarks detected as 'C': {result}")
    
    # Test with invalid input
    print(f"Empty landmarks: {detect_left_hand_c([])}")
    print(f"None input: {detect_left_hand_c(None)}")
    
    # Show the mirroring effect
    print("\nMirroring demonstration:")
    print("Original landmarks (first 3 points):")
    for i in range(3):
        lm = sample_landmarks[i]
        print(f"  Point {i}: x={lm.x:.2f}, y={lm.y:.2f}, z={lm.z:.2f}")
    
    print("After mirroring:")
    for i in range(3):
        lm = sample_landmarks[i]
        print(f"  Point {i}: x={1-lm.x:.2f}, y={lm.y:.2f}, z={lm.z:.2f}")