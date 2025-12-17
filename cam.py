import cv2
import mediapipe as mp
import numpy as np
import threading
import wave
from playsound import playsound

# ==============================
# STEP 1: CREATE ALARM SOUND
# ==============================
def create_alarm_sound(filename="alarm.wav", duration=3, freq=440):
    sample_rate = 44100
    amplitude = 16000

    t = np.linspace(0, duration, int(sample_rate * duration))
    data = amplitude * np.sin(2 * np.pi * freq * t *
                              (1 + 0.5 * np.sin(2 * np.pi * 3 * t)))

    data = data.astype(np.int16)

    with wave.open(filename, 'w') as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        file.writeframes(data.tobytes())

    print(f"Alarm sound created: {filename}")


create_alarm_sound()

# ==============================
# STEP 2: MEDIAPIPE SETUP
# ==============================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# ==============================
# STEP 3: EYE RATIO FUNCTION
# ==============================
def calculate_eye_ratio(landmarks, eye_points, w, h):
    points = []
    for idx in eye_points:
        lm = landmarks[idx]
        points.append((int(lm.x * w), int(lm.y * h)))

    v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    h1 = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    return (v1 + v2) / (2.0 * h1)


# ==============================
# STEP 4: ALARM THREAD
# ==============================
def play_alarm():
    playsound("alarm.wav")


# ==============================
# STEP 5: VIDEO CAPTURE
# ==============================
cap = cv2.VideoCapture(0)

CLOSED_EYE_FRAMES = 50
EYE_RATIO_THRESHOLD = 0.23

counter = 0
alarm_on = False

# ==============================
# STEP 6: MAIN LOOP
# ==============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            left = calculate_eye_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
            right = calculate_eye_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
            avg_ratio = (left + right) / 2.0

            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                drawing_spec,
                drawing_spec
            )

            if avg_ratio < EYE_RATIO_THRESHOLD:
                counter += 1
                if counter > CLOSED_EYE_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                    if not alarm_on:
                        alarm_on = True
                        threading.Thread(target=play_alarm, daemon=True).start()
            else:
                counter = 0
                alarm_on = False

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
