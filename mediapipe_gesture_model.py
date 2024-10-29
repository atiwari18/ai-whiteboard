import os
import warnings
import logging

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress protobuf warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

# Configure absl logging
import absl.logging

logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

""" -------------------------------- CODE ABOVE IS TO REMOVE WARNINGS AND LOGGING INFORMATION -------------------------------- """

import cv2
import mediapipe as mp

# Initialize mp_hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=0
)
mp_draw = mp.solutions.drawing_utils

#Function to check if a finger is extended.
def is_finger_extended(hand_landmarks, finger_tip_id, finger_pip_id):
    """Helper function to check if a finger is extended"""
    tip = hand_landmarks.landmark[finger_tip_id]
    pip = hand_landmarks.landmark[finger_pip_id]
    return tip.y < pip.y

#Function to check if the thumb is extended on each hand.
def check_thumb_extended(hand_landmarks, handedness):
    """Check if thumb is extended based on hand orientation"""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]

    # Calculate the horizontal distance between thumb tip and MCP
    thumb_distance = abs(thumb_tip.x - thumb_mcp.x)

    # Check if thumb is far enough from the palm
    return thumb_distance > 0.07  # Adjust this threshold if needed

#Function to check if the pointer finger is extended.
def is_pointer_finger_extended(hand_landmarks):
    """Check if only the index finger is extended"""
    # Check index finger
    index_extended = is_finger_extended(
        hand_landmarks,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP
    )

    # Check other fingers are not extended
    middle_extended = is_finger_extended(
        hand_landmarks,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP
    )
    ring_extended = is_finger_extended(
        hand_landmarks,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_PIP
    )
    pinky_extended = is_finger_extended(
        hand_landmarks,
        mp_hands.HandLandmark.PINKY_TIP,
        mp_hands.HandLandmark.PINKY_PIP
    )

    # Return true only if index is extended and others are not
    return index_extended and not (middle_extended or ring_extended or pinky_extended)

#Function to check if the hand is in a fist.
def is_fist(hand_landmarks):
    """Check if hand is in a fist position (all fingers closed)"""
    fingers = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]

    # Check if all fingers are closed (tips below PIPs)
    return all(not is_finger_extended(hand_landmarks, tip, pip) for tip, pip in fingers)

#Function to check if the palm is open and all fingers are extended.
def is_palm_open(hand_landmarks, handedness):
    """Check if all fingers are extended (open palm)"""
    fingers = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]

    # Check thumb extension using the new function
    thumb_extended = check_thumb_extended(hand_landmarks, handedness)

    # Check if all fingers are extended (tips above PIPs)
    fingers_extended = all(is_finger_extended(hand_landmarks, tip, pip) for tip, pip in fingers)

    return fingers_extended and thumb_extended


def main():
    cap = cv2.VideoCapture(0)

    # Add gesture state to prevent flickering
    current_gesture = None
    gesture_confidence = 0

    # Number of consecutive frames needed to change gesture
    #  - Lowering this value resulted in a quicker detection of the gesture.
    #  - Prev. value of 3/5 resulted in a delay or lack of detection in hand gesture.
    CONFIDENCE_THRESHOLD = 2

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            gesture_detected = False

            if result.multi_hand_landmarks and result.multi_handedness:
                for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get handedness (left or right)
                    hand_type = handedness.classification[0].label

                    # Detect gestures
                    if is_palm_open(hand_landmarks, hand_type):
                        if current_gesture != "stop":
                            gesture_confidence += 1
                            if gesture_confidence >= CONFIDENCE_THRESHOLD:
                                current_gesture = "stop"
                        gesture_detected = True
                    elif is_pointer_finger_extended(hand_landmarks):
                        if current_gesture != "draw":
                            gesture_confidence += 1
                            if gesture_confidence >= CONFIDENCE_THRESHOLD:
                                current_gesture = "draw"
                        gesture_detected = True
                    elif is_fist(hand_landmarks):
                        if current_gesture != "erase":
                            gesture_confidence += 1
                            if gesture_confidence >= CONFIDENCE_THRESHOLD:
                                current_gesture = "erase"
                        gesture_detected = True

                    # Display handedness and current gesture
                    cv2.putText(frame, f"Hand: {hand_type}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    if current_gesture == "draw":
                        cv2.putText(frame, "Draw Mode", (50, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    elif current_gesture == "erase":
                        cv2.putText(frame, "Erase Mode", (50, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif current_gesture == "stop":
                        cv2.putText(frame, "Stop Mode", (50, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Reset confidence if no gesture is detected
            if not gesture_detected:
                gesture_confidence = 0

            cv2.imshow("Hand Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()