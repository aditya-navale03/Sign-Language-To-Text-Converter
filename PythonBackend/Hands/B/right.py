import math
from typing import List, Optional

class Point:
    """Represents a 2D point with x, y coordinates."""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

class RightHandGestureDetector:
    """
    Detects gestures specifically from RIGHT hand MediaPipe landmarks.
    
    MediaPipe hand landmark indices:
    0: WRIST, 4: THUMB_TIP, 5: INDEX_MCP, 8: INDEX_TIP,
    9: MIDDLE_MCP, 12: MIDDLE_TIP, 13: RING_MCP, 16: RING_TIP,
    17: PINKY_MCP, 20: PINKY_TIP, 2: THUMB_IP
    """
    
    # Landmark indices
    WRIST = 0
    THUMB_IP = 2
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_TIP = 20
    
    def __init__(self, thumb_alignment_threshold: float = 0.1):
        """
        Initialize the right hand detector.
        
        Args:
            thumb_alignment_threshold: Maximum Y-distance for thumb alignment with index finger
        """
        self.thumb_alignment_threshold = thumb_alignment_threshold
    
    def _is_right_hand_palm_visible(self, landmarks: List[Point]) -> bool:
        """
        Check if this is a right hand with palm facing the camera.
        For right hand: when palm faces camera, the cross product should be positive.
        
        Args:
            landmarks: List of hand landmark points
            
        Returns:
            True if right hand palm is facing camera
        """
        # Vector from wrist to index base
        wrist_to_index_x = landmarks[self.INDEX_MCP].x - landmarks[self.WRIST].x
        wrist_to_index_y = landmarks[self.INDEX_MCP].y - landmarks[self.WRIST].y
        
        # Vector from wrist to pinky base  
        wrist_to_pinky_x = landmarks[self.PINKY_MCP].x - landmarks[self.WRIST].x
        wrist_to_pinky_y = landmarks[self.PINKY_MCP].y - landmarks[self.WRIST].y
        
        # Cross product for right hand palm detection
        cross_product = wrist_to_index_x * wrist_to_pinky_y - wrist_to_index_y * wrist_to_pinky_x
        
        # For right hand, positive cross product means palm is facing camera
        return cross_product > 0
    
    def _are_four_fingers_extended(self, landmarks: List[Point]) -> bool:
        """
        Check if all four fingers (index, middle, ring, pinky) are extended.
        Extended means fingertip is above (lower Y value) than the MCP joint.
        
        Args:
            landmarks: List of hand landmark points
            
        Returns:
            True if all four fingers are extended
        """
        return all([
            landmarks[self.INDEX_TIP].y < landmarks[self.INDEX_MCP].y,    # Index extended
            landmarks[self.MIDDLE_TIP].y < landmarks[self.MIDDLE_MCP].y,  # Middle extended  
            landmarks[self.RING_TIP].y < landmarks[self.RING_MCP].y,      # Ring extended
            landmarks[self.PINKY_TIP].y < landmarks[self.PINKY_MCP].y     # Pinky extended
        ])
    
    def _is_thumb_folded_against_palm(self, landmarks: List[Point]) -> bool:
        """
        Check if thumb is folded against the palm using multiple criteria.
        
        Args:
            landmarks: List of hand landmark points
            
        Returns:
            True if thumb is folded against palm
        """
        # Method 1: Check if thumb tip is closer to palm than when extended
        thumb_to_index_dist_sq = (
            (landmarks[self.THUMB_TIP].x - landmarks[self.INDEX_MCP].x) ** 2 + 
            (landmarks[self.THUMB_TIP].y - landmarks[self.INDEX_MCP].y) ** 2
        )
        
        thumb_to_base_dist_sq = (
            (landmarks[self.THUMB_TIP].x - landmarks[self.THUMB_IP].x) ** 2 + 
            (landmarks[self.THUMB_TIP].y - landmarks[self.THUMB_IP].y) ** 2
        )
        
        distance_check = thumb_to_index_dist_sq < thumb_to_base_dist_sq
        
        # Method 2: Check if thumb is positioned inward (X position relative to wrist)
        # For right hand, folded thumb should have X coordinate between wrist and index
        wrist_x = landmarks[self.WRIST].x
        index_x = landmarks[self.INDEX_MCP].x
        thumb_x = landmarks[self.THUMB_TIP].x
        
        # Thumb should be between wrist and index finger (inward position)
        if wrist_x < index_x:  # Normal orientation
            position_check = wrist_x < thumb_x < index_x
        else:  # Flipped orientation
            position_check = index_x < thumb_x < wrist_x
        
        return distance_check and position_check
    
    def _is_thumb_aligned_horizontally(self, landmarks: List[Point]) -> bool:
        """
        Check if thumb tip is horizontally aligned with index MCP joint.
        This ensures the thumb is positioned correctly for the "B" gesture.
        
        Args:
            landmarks: List of hand landmark points
            
        Returns:
            True if thumb is horizontally aligned
        """
        y_difference = abs(landmarks[self.THUMB_TIP].y - landmarks[self.INDEX_MCP].y)
        return y_difference < self.thumb_alignment_threshold
    
    def detect_right_hand_b_gesture(self, landmarks: List[Point]) -> str:
        """
        Detect "B" gesture from right hand landmarks.
        
        "B" gesture criteria:
        1. Right hand palm must be facing camera
        2. Four fingers (index, middle, ring, pinky) must be extended
        3. Thumb must be folded against palm
        4. Thumb must be horizontally aligned with index finger
        
        Args:
            landmarks: List of 21 hand landmark points
            
        Returns:
            "B" if gesture detected, empty string otherwise
        """
        if len(landmarks) != 21:
            return ""  # Invalid landmark count
        
        # Must be right hand with palm facing camera
        if not self._is_right_hand_palm_visible(landmarks):
            return ""
        
        # Check all "B" gesture conditions
        if (self._are_four_fingers_extended(landmarks) and 
            self._is_thumb_folded_against_palm(landmarks) and 
            self._is_thumb_aligned_horizontally(landmarks)):
            return "B"
        
        return ""

# Convenience function matching your original interface
def detect_right_hand(lm: List[Point]) -> str:
    """
    Detect right hand "B" gesture from MediaPipe landmarks.
    
    Args:
        lm: List of 21 landmark points from MediaPipe
        
    Returns:
        "B" if gesture detected, empty string otherwise
    """
    detector = RightHandGestureDetector()
    return detector.detect_right_hand_b_gesture(lm)

# Helper function for MediaPipe integration
def detect_from_mediapipe_right_hand(hand_landmarks) -> str:
    """
    Detect gesture from MediaPipe right hand landmarks object.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks object
        
    Returns:
        "B" if gesture detected, empty string otherwise
    """
    points = [Point(lm.x, lm.y) for lm in hand_landmarks.landmark]
    return detect_right_hand(points)