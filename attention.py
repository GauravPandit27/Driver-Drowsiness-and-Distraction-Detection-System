import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO  # For phone detection
import pyttsx3  # For text-to-speech
import torch  # To check GPU availability
import time  # For sleep functionality

# Initialize MediaPipe Face Mesh with GPU support
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    min_detection_confidence=0.5, 
    refine_landmarks=True
)

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speaking rate
engine.setProperty('volume', 1)  # Set volume to maximum

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(landmarks, eye_indices):
    eye = np.array([(landmarks[i][0], landmarks[i][1]) for i in eye_indices])
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Indices for landmarks of the left and right eyes
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Thresholds
EAR_THRESHOLD = 0.25

# Load YOLO model with GPU support
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
phone_detector = YOLO("yolov8n.pt")  # Ensure the model file is available

# Flags for controlling text-to-speech and flash screen
eyes_closed_flag = False
phone_detected_flag = False
flash_screen_active = False
flash_start_time = 0
last_warning_time = 0
last_phone_warning_time = 0
warning_interval = 10  # 10 seconds interval for TTS
eyes_closed_duration = 0  # Track how long the eyes have been closed
last_eyes_open_time = time.time()  # Time when eyes were last opened
flash_duration = 3  # Flash screen duration in seconds

# Open webcam and check if available
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)

    eyes_closed = False
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]
            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            # Smooth eye closure detection (eyes need to be closed continuously for a certain time)
            if avg_ear < EAR_THRESHOLD:
                eyes_closed_duration += 1  # Increment duration if eyes are closed
            else:
                eyes_closed_duration = 0  # Reset if eyes open

            # Trigger appropriate messages based on the duration the eyes have been closed
            if eyes_closed_duration > 60:  # Eyes closed for more than 2 seconds (assuming 30 FPS)
                warning_text = "Wake Up!"
                cv2.putText(frame, warning_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Trigger TTS only if the time interval has passed
                current_time = time.time()
                if current_time - last_warning_time > warning_interval:
                    engine.say(warning_text)
                    engine.runAndWait()
                    last_warning_time = current_time
                eyes_closed_flag = True
                flash_screen_active = True  # Start flashing screen
                flash_start_time = time.time()  # Record the time the flash started

            elif eyes_closed_duration > 300:  # Eyes closed for more than 10 seconds (assuming 30 FPS)
                warning_text = "Calling Emergency Services!"
                cv2.putText(frame, warning_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Trigger TTS only if the time interval has passed
                current_time = time.time()
                if current_time - last_warning_time > warning_interval:
                    engine.say(warning_text)
                    engine.runAndWait()
                    last_warning_time = current_time
                eyes_closed_flag = True
                flash_screen_active = True  # Start flashing screen
                flash_start_time = time.time()  # Record the time the flash started

            # Reset the warning when eyes are opened again
            if eyes_closed_duration == 0 and time.time() - last_eyes_open_time > 5:
                engine.say("Eyes are back open")
                engine.runAndWait()

            # Update the time when the eyes were last opened
            if eyes_closed_duration == 0:
                last_eyes_open_time = time.time()

    # Flash screen handling (controlled with a more efficient timer)
    if flash_screen_active:
        # Toggle flash screen every 0.5 seconds for the duration of the flash
        elapsed_flash_time = time.time() - flash_start_time
        if elapsed_flash_time < flash_duration:
            if int(elapsed_flash_time * 2) % 2 == 0:  # Toggle the flash screen every 0.5 seconds
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
                alpha = 0.3
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        else:
            flash_screen_active = False  # Stop flashing screen after the defined duration

    # Phone detection using YOLO (GPU-enabled)
    phone_results = phone_detector.predict(frame, device=device, verbose=False)
    phone_detected = False
    for result in phone_results[0].boxes:
        cls_id = int(result.cls[0])
        confidence = result.conf[0]
        if cls_id == 67 and confidence > 0.5:
            phone_detected = True
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "No Phones Allowed!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Trigger TTS only if the time interval has passed
            current_time = time.time()
            if current_time - last_phone_warning_time > warning_interval:
                engine.say("No Phones Allowed!")
                engine.runAndWait()
                last_phone_warning_time = current_time
            break

    if not phone_detected:
        phone_detected_flag = False

    # Display the frame
    cv2.imshow("Attention Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

