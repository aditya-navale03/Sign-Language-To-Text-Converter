import asyncio
import websockets
import json
import cv2
import numpy as np
import mediapipe as mp
import importlib

# -------------------------
# CONFIGURATION
# -------------------------
# List the letters you want to support
LETTERS = ["A", "B", "C"]

# Dynamically import left and right hand logic for each letter
# This creates a dictionary for easy access later
letter_modules = {}
for letter in LETTERS:
    left_module = importlib.import_module(f"Hands.{letter}.left")
    right_module = importlib.import_module(f"Hands.{letter}.right")
    letter_modules[letter] = {
        "left": left_module,
        "right": right_module
    }

# -------------------------
# MEDIA PIPE SETUP
# -------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Only detect one hand at a time
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# -------------------------
# HELPER FUNCTION
# -------------------------
def convert_mediapipe_to_points(mediapipe_landmarks, label, letter):
    """
    Convert MediaPipe landmarks to custom Point objects
    depending on handedness (Left/Right) and letter.
    """
    if label == "Left":
        PointClass = letter_modules[letter]["left"].Point
    else:
        PointClass = letter_modules[letter]["right"].Point
    return [PointClass(lm.x, lm.y) for lm in mediapipe_landmarks]

# -------------------------
# WEBSOCKET HANDLER
# -------------------------
async def handle_client(websocket):
    print("Client connected")
    try:
        async for message in websocket:
            try:
                # Decode incoming frame from bytes
                frame = cv2.imdecode(np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # Convert to RGB for MediaPipe processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands_detector.process(frame_rgb)

                translation = ""  # Default: no letter detected

                # If a hand is detected
                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, hand_handedness in zip(
                        results.multi_hand_landmarks, results.multi_handedness
                    ):
                        label = hand_handedness.classification[0].label  # "Left" or "Right"

                        # Draw landmarks for visualization
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style()
                        )

                        # Check each letter
                        for letter in LETTERS:
                            points = convert_mediapipe_to_points(hand_landmarks.landmark, label, letter)

                            # Call the correct detection function
                            if label == "Left":
                                detected = letter_modules[letter]["left"].detect_left_hand(points)
                            else:
                                detected = letter_modules[letter]["right"].detect_right_hand(points)

                            if detected:  # If a letter is detected
                                translation = letter
                                break  # Stop checking other letters

                        if translation:  # Stop if already detected
                            break

                # Encode frame back to bytes for sending
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()

                # Send JSON response
                response = {
                    "translation": translation,
                    "status": "success",
                    "frame": frame_bytes.hex()
                }
                await websocket.send(json.dumps(response))

            except Exception as e:
                print(f"Error processing frame: {e}")
                await websocket.send(json.dumps({"translation": "", "status": "error"}))

    except websockets.exceptions.ConnectionClosedOK:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("Connection closed")

# -------------------------
# MAIN FUNCTION
# -------------------------
async def main():
    print("Starting WebSocket server at ws://0.0.0.0:5000/ws")
    async with websockets.serve(handle_client, "0.0.0.0", 5000, max_size=2**24):
        await asyncio.Future()  # Run forever

# -------------------------
# ENTRY POINT
# -------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped gracefully")
