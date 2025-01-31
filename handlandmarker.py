import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import subprocess
import threading

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
DISTANCE_TEXT_COLOR = (255, 0, 0)

last_volume = -1
volume_lock = threading.Lock()

def set_system_volume(volume_adjustment):
    """ Run volume adjustment in a separate thread to prevent blocking. """
    global last_volume
    with volume_lock:
        if abs(volume_adjustment - last_volume) > 5:
            subprocess.Popen(["osascript", "-e", f"set volume output volume {volume_adjustment}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            last_volume = volume_adjustment

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    if not hand_landmarks_list:
        return rgb_image

    annotated_image = rgb_image.copy()
    height, width, _ = annotated_image.shape

    distance = 0
    for idx, hand_landmarks in enumerate(hand_landmarks_list):
        handedness = handedness_list[idx]
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image, hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

        x_min = int(min(l.x for l in hand_landmarks) * width)
        y_min = int(min(l.y for l in hand_landmarks) * height) - MARGIN
        actual_handedness = "Right" if handedness[0].category_name == "Left" else "Left"
        cv2.putText(annotated_image, actual_handedness,
                    (x_min, y_min), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        thumb_tip, index_finger_tip = hand_landmarks[4], hand_landmarks[8]

        if actual_handedness == "Right":
            distance = np.sqrt((thumb_tip.x - index_finger_tip.x) ** 2 +
                            (thumb_tip.y - index_finger_tip.y) ** 2 +
                            (thumb_tip.z - index_finger_tip.z) ** 2)

            volume_adjustment = max(0, min(100, int(distance * 200)))
            threading.Thread(target=set_system_volume, args=(volume_adjustment,), daemon=True).start()

        if actual_handedness == "Left":
            distance = np.sqrt((thumb_tip.x - index_finger_tip.x) ** 2 +
                            (thumb_tip.y - index_finger_tip.y) ** 2 +
                            (thumb_tip.z - index_finger_tip.z) ** 2)
            if distance <= 0.1:
                print("Hello")
                
        distance_text = f"Distance: {distance:.2f}"
        cv2.putText(annotated_image, distance_text, (x_min, y_min + 30),
                    cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, DISTANCE_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        
        cv2.line(annotated_image,
                 (int(thumb_tip.x * width), int(thumb_tip.y * height)),
                 (int(index_finger_tip.x * width), int(index_finger_tip.y * height)),
                 DISTANCE_TEXT_COLOR, 3)

    return annotated_image

base_options = python.BaseOptions(model_asset_path='/Users/doruk/Documents/Gesture Detection/Hand Landmarker Task.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(image)

    annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)
    cv2.imshow('Hand Landmarker', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
