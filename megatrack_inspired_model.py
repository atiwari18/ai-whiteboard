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
import math

# Enum values
NONE = 0
EXTENDED = 1
GRIPPED = 2

# Initialize mp_hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=0,
)
mp_draw = mp.solutions.drawing_utils

# Helper function for calculating the distance between hand landmarks
def get_3d_dist(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

# Gets the angle between two vectors
def get_angle(a, b, c):
    vecs = [[b.x-a.x, b.y-a.y, b.z-a.z], [c.x-b.x, c.y-b.y, c.z-b.z]]
    mag1 = 0
    mag2 = 0
    dot = 0
    
    for i in range(3):
        mag1 += vecs[0][i]**2
        mag2 += vecs[1][i]**2

        dot += vecs[1][i]*vecs[0][i]

    if(mag1 != 0 and mag2 != 0):
        return math.acos(dot/(math.sqrt(mag1*mag2)))
    return math.pi

#Function to check if a finger is extended.
def get_finger_statuses(hand_landmarks):
    """Helper function to check if a finger is extended, gripped, or neither.
    finds the angle at two key points in the fingers and checks them against
    different thresholds to determine status."""
    
    extend_thresh = 20/180*math.pi
    hand = []
    base = hand_landmarks.landmark[0]
    for i in range(1, 18, 4):
        mcp = hand_landmarks.landmark[i]
        dip = hand_landmarks.landmark[i+1]
        pip = hand_landmarks.landmark[i+2]
        tip = hand_landmarks.landmark[i+3]

        crook = get_angle(mcp, dip, tip)
        if(crook < extend_thresh):
            hand.append(EXTENDED)
        elif(pip.y > mcp.y):
            hand.append(GRIPPED)
        else:
            hand.append(NONE)

    return hand

def is_thumb_gripped(hand_landmarks):

    grip_threshold = 0.12
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP+1]
    middle_knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]

    if(get_3d_dist(thumb_tip, index_knuckle) < grip_threshold or
        get_3d_dist(thumb_tip, index_base) < grip_threshold or 
        get_3d_dist(thumb_tip, middle_knuckle) < grip_threshold):
        return GRIPPED
    return NONE

def is_stop(statuses):
    return statuses[0] != EXTENDED and all(statuses[i] == EXTENDED for i in range(1,5))
def is_erase(statuses):
    return statuses[0] != EXTENDED and all(statuses[i] == GRIPPED for i in range(1,5))
def is_draw(statuses):
    return statuses[0] != EXTENDED and statuses[1] == EXTENDED and all(statuses[i] == GRIPPED for i in range(2,5))


def main():
    cap = cv2.VideoCapture(0)

    # Add gesture state to prevent flickering
    current_gesture = None
    gesture_confidence = 0

    # Number of consecutive frames needed to change gesture
    #  - Lowering this value resulted in a quicker detection of the gesture.
    #  - Prev. value of 3/5 resulted in a delay or lack of detection in hand gesture.
    CONFIDENCE_THRESHOLD = 2
    last_gesture = "None"
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            
            if result.multi_hand_landmarks and result.multi_handedness:
                for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get handedness (left or right)
                    hand_type = handedness.classification[0].label
                    finger_statuses = get_finger_statuses(hand_landmarks)
                    
                    
                    # Detect gestures
                    this_gesture = "None"                    
                    if is_stop(finger_statuses):
                        this_gesture = "stop"
                    if is_draw(finger_statuses):
                        this_gesture = "draw"
                    if is_erase(finger_statuses):
                        this_gesture = "erase"

                    if this_gesture == last_gesture:
                        gesture_confidence += 1
                    else:
                        gesture_confidence = 0
                    
                    if gesture_confidence >= CONFIDENCE_THRESHOLD:
                        current_gesture = this_gesture
                    
                    last_gesture = this_gesture
                    
                    cv2.putText(frame, str(int(finger_statuses[0])), (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, str(int(finger_statuses[1])), (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, str(int(finger_statuses[2])), (30, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, str(int(finger_statuses[3])), (40, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, str(int(finger_statuses[4])), (50, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    

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
                    else:
                        cv2.putText(frame, "No Gesture", (50, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 180), 2)

            cv2.imshow("Hand Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()