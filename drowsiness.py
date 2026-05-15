import cv2
import mediapipe as mp
import numpy as np
import sys

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices (MediaPipe 468 landmarks)
LEFT_EYE = [33, 133, 157, 158, 159, 160, 161, 173]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 466]

def get_eye_points(landmarks, indices):
    points = []
    for i in indices[:6]:
        points.append((landmarks[i][0], landmarks[i][1]))
    return points

def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    ear = (A + B) / (2.0 * C)
    return ear

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 30
frame_counter = 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    sys.exit(1)

print("Drowsiness Detector Running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        # Get all landmarks
        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            landmarks.append((int(lm.x * w), int(lm.y * h)))

        left_eye = get_eye_points(landmarks, LEFT_EYE)
        right_eye = get_eye_points(landmarks, RIGHT_EYE)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Draw eye dots
        for pt in left_eye:
            cv2.circle(frame, pt, 2, (0, 255, 0), -1)
        for pt in right_eye:
            cv2.circle(frame, pt, 2, (0, 255, 0), -1)

        # Drowsiness logic
        if ear < EAR_THRESHOLD:
            frame_counter += 1
            cv2.putText(frame, f"DROWSY! ({frame_counter})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if frame_counter >= CONSEC_FRAMES:
                cv2.putText(frame, "!!! ALARM !!! WAKE UP !!!", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                # Terminal beep
                sys.stdout.write('\a')
                sys.stdout.flush()
        else:
            frame_counter = max(0, frame_counter - 1)

        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        frame_counter = 0

    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()