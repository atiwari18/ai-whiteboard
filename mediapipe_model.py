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


def is_stop(statuses):
    return all(statuses[i] == EXTENDED for i in range(1,5))
def is_erase(statuses):
    return all(statuses[i] == GRIPPED for i in range(1,5))
def is_draw(statuses):
    return statuses[0] != EXTENDED and statuses[1] == EXTENDED and all(statuses[i] == GRIPPED for i in range(2,5))

def get_prediction(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            finger_statuses = get_finger_statuses(hand_landmarks)
                    
            # Detect gestures
            this_gesture = "None"                    
            if is_stop(finger_statuses):
                this_gesture = "Stop"
            if is_draw(finger_statuses):
                this_gesture = "Draw"
            if is_erase(finger_statuses):
                this_gesture = "Erase"
        return this_gesture
    return "None"

