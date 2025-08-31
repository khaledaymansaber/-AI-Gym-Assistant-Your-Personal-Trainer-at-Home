import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Define keypoint indices (COCO format)
KEYPOINTS = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14
}

def calculate_angle(a, b, c):
    """Calculate angle between three points (in degrees)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

# Rep counter variables
count = 0
stage = "down"
rep_locked = False

# Moving average buffer for smoothing
angle_buffer = deque(maxlen=5)

# Flexible thresholds (tune if needed)
DOWN_THRESHOLD = 160
UP_THRESHOLD = 172
RESET_THRESHOLD = 162   # unlock once below this

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run pose detection
    results = model(frame, verbose=False)

    for r in results:
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            continue

        kpts = r.keypoints.xy.cpu().numpy()[0]
        conf = r.keypoints.conf.cpu().numpy()[0]

        # ✅ Skip if confidence too low
        if np.any(conf[[KEYPOINTS["left_hip"], KEYPOINTS["right_hip"],
                        KEYPOINTS["left_knee"], KEYPOINTS["right_knee"]]] < 0.5):
            continue

        # ✅ Average both sides for stability
        shoulder = (kpts[KEYPOINTS["left_shoulder"]] + kpts[KEYPOINTS["right_shoulder"]]) / 2
        hip = (kpts[KEYPOINTS["left_hip"]] + kpts[KEYPOINTS["right_hip"]]) / 2
        knee = (kpts[KEYPOINTS["left_knee"]] + kpts[KEYPOINTS["right_knee"]]) / 2

        # Calculate angle
        angle = calculate_angle(shoulder, hip, knee)

        # Smooth angle
        angle_buffer.append(angle)
        smooth_angle = np.mean(angle_buffer)

        # -----------------------------
        # Rep counting with dynamic thresholds
        # -----------------------------
        if not rep_locked:
            if smooth_angle < DOWN_THRESHOLD:  # hips clearly down
                stage = "down"

            elif smooth_angle > UP_THRESHOLD and stage == "down":  # hips up
                count += 1
                stage = "up"
                rep_locked = True

        else:
            # Unlock once trainee goes below RESET_THRESHOLD
            if smooth_angle < RESET_THRESHOLD:
                rep_locked = False

        # Draw points
        for point in [shoulder, hip, knee]:
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)

        # Draw lines
        cv2.line(frame, tuple(shoulder.astype(int)), tuple(hip.astype(int)), (255, 0, 0), 3)
        cv2.line(frame, tuple(hip.astype(int)), tuple(knee.astype(int)), (255, 0, 0), 3)

        # Display info
        cv2.putText(frame, f'Angle: {int(smooth_angle)} deg', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Count: {count}', (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Glute Bridge Counter - Live", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
