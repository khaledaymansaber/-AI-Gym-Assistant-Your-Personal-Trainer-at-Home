import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

# ---------- Angle helper ----------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# ---------- Smoother ----------
def smooth_angle(buffer, new_value, maxlen=5):
    buffer.append(new_value)
    return sum(buffer) / len(buffer)

# Buffers for smoothing
elbow_buffer = deque(maxlen=5)
shoulder_buffer = deque(maxlen=5)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame, verbose=False)

    for r in results:
        if r.keypoints is not None:
            kpts = r.keypoints.xy.cpu().numpy()  # (num_people, 17, 2)

            for person in kpts:
                # Right side keypoints
                shoulder = person[6]
                elbow = person[8]
                wrist = person[10]
                hip = person[12]

                # Calculate angles
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                shoulder_angle = calculate_angle(hip, shoulder, wrist)

                # Smooth
                elbow_angle = smooth_angle(elbow_buffer, elbow_angle)
                shoulder_angle = smooth_angle(shoulder_buffer, shoulder_angle)

                # Check conditions
                if elbow_angle < 150:
                    feedback = "Keep arm straight!"
                    color = (0, 0, 255)
                elif shoulder_angle > 95:
                    feedback = "Don't raise your hand that much!"
                    color = (0, 0, 255)
                else:
                    feedback = "Good Form!"
                    color = (0, 255, 0)

                # Draw feedback
                cv2.putText(frame, f'Elbow: {int(elbow_angle)}', (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, f'Shoulder: {int(shoulder_angle)}', (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, feedback, (30, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Show frame
    cv2.imshow("Front Hand Raise - YOLO Pose", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()