import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

def is_thumbs_up(hand_landmarks):
    # Check if the thumb position is higher than the rest of the hand, a.k.a thumbs up
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    return (thumb_tip.y < thumb_ip.y < thumb_mcp.y < thumb_cmc.y)

def is_thumbs_down(hand_landmarks):
    # Check if the thumb position is lower than the rest of the hand, a.k.a thumbs down
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    return (thumb_tip.y > thumb_ip.y > thumb_mcp.y > thumb_cmc.y)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_result = hands.process(rgb_frame)

    if hand_result.multi_hand_landmarks:
        for hand_landmarks, hand_landmarks_info in zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness):
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
            hand_confidence = hand_landmarks_info.classification[0].score
            cv2.putText(frame, f'Confidence: {hand_confidence:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            if is_thumbs_up(hand_landmarks):
                cv2.putText(frame, 'Thumbs Up', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
            if is_thumbs_down(hand_landmarks):
                cv2.putText(frame, 'Thumbs Down', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    face_result = face_detection.process(rgb_frame)

    if face_result.detections:
        for detection in face_result.detections:
            mp_drawing.draw_detection(frame, detection)

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
