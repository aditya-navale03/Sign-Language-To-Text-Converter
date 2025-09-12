# import math
# import numpy as np

# class Point:
#     def __init__(self, x, y, z=0.0):
#         self.x = x
#         self.y = y
#         self.z = z

# def distance(a: Point, b: Point) -> float:
#     return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

# def detect_right_hand_e(landmarks: list, debug=False) -> bool:
#     """
#     Detect ASL letter 'E' for the RIGHT hand.
#     """
#     if not landmarks or len(landmarks) < 21:
#         if debug: print("❌ Invalid input - need 21 landmarks")
#         return False

#     wrist = landmarks[0]
#     thumb_tip, thumb_ip, thumb_mcp = landmarks[4], landmarks[3], landmarks[2]
#     index_tip, index_pip, index_mcp = landmarks[8], landmarks[6], landmarks[5]
#     middle_tip, middle_pip, middle_mcp = landmarks[12], landmarks[10], landmarks[9]
#     ring_tip, ring_pip, ring_mcp = landmarks[16], landmarks[14], landmarks[13]
#     pinky_tip, pinky_pip, pinky_mcp = landmarks[20], landmarks[18], landmarks[17]

#     palm_width = distance(index_mcp, pinky_mcp)

#     # Finger curling check
#     def is_finger_curled(tip, pip, mcp):
#         tip_to_wrist = distance(tip, wrist)
#         pip_to_wrist = distance(pip, wrist)
#         mcp_to_wrist = distance(mcp, wrist)
#         curled_inward = tip_to_wrist < pip_to_wrist + palm_width * 0.15
#         tip_to_mcp = distance(tip, mcp)
#         not_extended = tip_to_mcp < mcp_to_wrist * 0.9
#         y_curled = tip.y >= pip.y - palm_width * 0.1
#         return curled_inward and not_extended and y_curled

#     fingers_curled_count = sum([
#         is_finger_curled(index_tip, index_pip, index_mcp),
#         is_finger_curled(middle_tip, middle_pip, middle_mcp),
#         is_finger_curled(ring_tip, ring_pip, ring_mcp),
#         is_finger_curled(pinky_tip, pinky_pip, pinky_mcp)
#     ])

#     # Thumb check
#     finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
#     avg_finger_x = sum(tip.x for tip in finger_tips) / 4
#     avg_finger_y = sum(tip.y for tip in finger_tips) / 4

#     thumb_to_finger_area = distance(thumb_tip, Point(avg_finger_x, avg_finger_y, 0))
#     thumb_near_fingers = thumb_to_finger_area < palm_width * 0.6
#     thumb_visible = thumb_tip.x < avg_finger_x  # RIGHT HAND: thumb on left
#     thumb_to_wrist = distance(thumb_tip, wrist)
#     thumb_mcp_to_wrist = distance(thumb_mcp, wrist)
#     thumb_not_extended = thumb_to_wrist < thumb_mcp_to_wrist * 1.3

#     # Fist compactness
#     max_spread = max(distance(finger_tips[i], finger_tips[j]) for i in range(4) for j in range(i+1,4))
#     compact_fist = max_spread < palm_width * 0.85

#     # Fist position
#     avg_tip_to_wrist = sum(distance(t, wrist) for t in finger_tips) / 4
#     avg_mcp_to_wrist = sum(distance(mcp, wrist) for mcp in [index_mcp, middle_mcp, ring_mcp, pinky_mcp]) / 4
#     fist_position = avg_tip_to_wrist < avg_mcp_to_wrist + palm_width * 0.2

#     # Final decision
#     fingers_ok = fingers_curled_count >= 3
#     thumb_ok = thumb_near_fingers and thumb_visible and thumb_not_extended
#     shape_ok = compact_fist and fist_position

#     final_result = fingers_ok and thumb_ok and shape_ok

#     if debug:
#         print("=== RIGHT HAND E DETECTION ===")
#         print(f"Fingers curled: {fingers_curled_count}/4")
#         print(f"Thumb near={thumb_near_fingers}, visible={thumb_visible}, not_extended={thumb_not_extended}")
#         print(f"Compact fist={compact_fist}, fist_position={fist_position}")
#         print(f"FINAL RESULT: {'✅ DETECTED E' if final_result else '❌ NOT E'}")

#     return final_result

# # Example realistic right-hand landmarks
# def create_realistic_right_e_landmarks():
#     l = []
#     l.append(Point(0.5, 0.9, 0.0))  # wrist
#     # Thumb (across curled fingers)
#     l.extend([Point(0.46,0.83,0.01), Point(0.44,0.78,0.02), Point(0.43,0.74,0.03), Point(0.45,0.70,0.04)])
#     # Index finger (curled)
#     l.extend([Point(0.54,0.75,0.0), Point(0.52,0.72,0.01), Point(0.50,0.74,0.02), Point(0.48,0.75,0.03)])
#     # Middle finger (curled)
#     l.extend([Point(0.52,0.73,0.0), Point(0.50,0.70,0.01), Point(0.48,0.72,0.02), Point(0.46,0.73,0.03)])
#     # Ring finger (curled)
#     l.extend([Point(0.50,0.74,0.0), Point(0.48,0.71,0.01), Point(0.46,0.73,0.02), Point(0.44,0.74,0.03)])
#     # Pinky finger (curled)
#     l.extend([Point(0.48,0.76,0.0), Point(0.46,0.73,0.01), Point(0.44,0.75,0.02), Point(0.42,0.76,0.03)])
#     return l

# if __name__ == "__main__":
#     landmarks = create_realistic_right_e_landmarks()
#     result = detect_right_hand(landmarks, debug=True)
#     print("\nRight-hand E detected:", result)
