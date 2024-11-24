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
import numpy as np
import math

# Initialize mp_hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.8, #Confidence Threshold for hand detection.
    min_tracking_confidence=0.8, #Confidence Threshold for hand tracking.
    model_complexity=0
)

#Utility for drawing hand landmarks on the frame.
mp_draw = mp.solutions.drawing_utils

#Function to get distance between hand landmarks
def get_3d_dist(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

#Function to check if a finger is extended.
def is_finger_extended(hand_landmarks, finger_tip_id, finger_pip_id):
    """
    Check if a finger is extended based on the relative y-coordinates of
    its tip and proximal interphalangeal (PIP) joint.
    """
    tip = hand_landmarks.landmark[finger_tip_id]
    pip = hand_landmarks.landmark[finger_pip_id]
    return tip.y < pip.y

#Function to check if the thumb is extended by measuring its distance from the middle finger DIP joint.
def check_thumb_extended(hand_landmarks):
    """Check if thumb is extended based on hand orientation"""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]

    # Calculate the horizontal distance between thumb tip and MCP
    thumb_distance = get_3d_dist(thumb_tip, middle_dip)

    # Check if thumb is far enough from the palm
    return thumb_distance > 0.05  # Adjust this threshold if needed

#Function to check if the pointer finger is extended.
def is_pointer_finger_extended(hand_landmarks):
    #Check if only the index finger is extended'

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

    #Return true if only the index finger is extended and others (including the thumb) are not.
    return index_extended and not (middle_extended or ring_extended or pinky_extended) and not check_thumb_extended(hand_landmarks)

#Function to check if the hand is in a fist.
def is_fist(hand_landmarks):
    #Check if hand is in a fist position (all fingers closed)
    fingers = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]

    # Check if all fingers are closed (tips below PIPs)
    return all(not is_finger_extended(hand_landmarks, tip, pip) for tip, pip in fingers)

#Function to check if the palm is open and all fingers are extended.
def is_palm_open(hand_landmarks):
    #Check if all fingers are extended (open palm)
    fingers = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]

    # Check thumb extension using the new function
    thumb_extended = check_thumb_extended(hand_landmarks)

    # Check if all fingers are extended (tips above PIPs)
    fingers_extended = all(is_finger_extended(hand_landmarks, tip, pip) for tip, pip in fingers)

    return fingers_extended and thumb_extended


def main():
    cap = cv2.VideoCapture(0)

    # Add gesture state to prevent flickering
    current_gesture = None
    gesture_confidence = 0

    # Number of consecutive frames needed to change gesture
    CONFIDENCE_THRESHOLD = 4

    # Canvas for drawing
    canvas = None
    drawing_color = (0, 255, 0)  # Green color for drawing
    last_position = None  # To track the last position of the pointer finger tip
    last_gesture = "None"

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            if canvas is None:
                canvas = np.zeros_like(frame)  # Initialize canvas with the same dimensions as the frame

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            gesture_detected = False

            if result.multi_hand_landmarks and result.multi_handedness:
                for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get handedness (left or right)
                    hand_type = handedness.classification[0].label



                    # Detect and handle different gestures:
                    # - Open palm = stop drawing
                    # - Pointer finger = draw mode
                    # - Fist = erase canvas
                    this_gesture = "None"
                    if is_palm_open(hand_landmarks):
                        this_gesture = "stop"
                    elif is_pointer_finger_extended(hand_landmarks):
                        this_gesture = "draw"
                    elif is_fist(hand_landmarks):
                        this_gesture = "erase"

                    #Confirm if the gesture is consistent for enough frames.
                    if this_gesture == last_gesture:
                        gesture_confidence += 1
                    else:
                        gesture_confidence = 0
                        #print("test")

                    if gesture_confidence >= CONFIDENCE_THRESHOLD:
                        current_gesture = this_gesture

                    last_gesture = this_gesture

                    # Get pointer finger tip position for drawing
                    if current_gesture == "draw":
                        pointer_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        x, y = int(pointer_tip.x * frame.shape[1]), int(pointer_tip.y * frame.shape[0])

                        if last_position is not None:
                            # Draw a line from the last position to the current position
                            cv2.line(canvas, last_position, (x, y), drawing_color, thickness=4)
                        last_position = (x, y)  # Update the last position
                    else:
                        last_position = None  # Reset last position if not in draw mode

                    if current_gesture == "erase":
                        canvas = np.zeros_like(frame)  # Clear canvas


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


            # Combine the frame and canvas without blending for display.
            mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            combined_frame = cv2.add(frame_bg, canvas)

            #Display the result, canvas and frame.
            cv2.imshow("Hand Gesture Recognition", combined_frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    main()