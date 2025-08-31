# jumping_jacks_live.py
import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot access the camera")

# COCO keypoints for ankles and hips
KEYPOINTS = {
    "left_ankle": 15,
    "right_ankle": 16,
    "left_hip": 11,
    "right_hip": 12
}

# Function to calculate angle between ankles using hip midpoint
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return int(angle)

# thresholds
CLOSED_THRESHOLD = 30   # legs together
OPEN_THRESHOLD = 50     # legs wide apart
COOLDOWN = 0.8          # seconds to prevent double count

rep_count = 0
stage = "closed"
last_rep_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera frame not received.")
        break

    results = model(frame, verbose=False)

    for r in results:
        if r.keypoints is None:
            continue

        # get keypoints xy and confidence
        keypoints = r.keypoints.xy.cpu().numpy()
        confs = r.keypoints.conf.cpu().numpy()

        for person, conf in zip(keypoints, confs):
            # Extract required keypoints + confidences
            try:
                l_ankle = person[KEYPOINTS["left_ankle"]]
                r_ankle = person[KEYPOINTS["right_ankle"]]
                l_hip = person[KEYPOINTS["left_hip"]]
                r_hip = person[KEYPOINTS["right_hip"]]

                c_l_ankle = conf[KEYPOINTS["left_ankle"]]
                c_r_ankle = conf[KEYPOINTS["right_ankle"]]
                c_l_hip = conf[KEYPOINTS["left_hip"]]
                c_r_hip = conf[KEYPOINTS["right_hip"]]
            except:
                continue

            # ✅ Ensure all 4 keypoints visible with high confidence
            if (c_l_ankle < 0.5 or c_r_ankle < 0.5 or 
                c_l_hip < 0.5 or c_r_hip < 0.5):
                cv2.putText(frame, "Move back - full body not visible!", (50,150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                continue

            hip_mid = (l_hip + r_hip) / 2
            feet_angle = calculate_angle(l_ankle, hip_mid, r_ankle)

            # Draw keypoints
            for point in [l_ankle, r_ankle, l_hip, r_hip]:
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)

            # Draw lines
            cv2.line(frame, tuple(l_ankle.astype(int)), tuple(hip_mid.astype(int)), (0,0,255),2)
            cv2.line(frame, tuple(r_ankle.astype(int)), tuple(hip_mid.astype(int)), (0,0,255),2)

            # Rep counting with hysteresis
            now = time.time()
            if stage == "closed" and feet_angle > OPEN_THRESHOLD:
                stage = "open"
            elif stage == "open" and feet_angle < CLOSED_THRESHOLD:
                if now - last_rep_time > COOLDOWN:
                    rep_count += 1
                    last_rep_time = now
                stage = "closed"

            # Display info
            cv2.putText(frame, f"Feet Angle: {feet_angle} deg", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2)
            cv2.putText(frame, f"Reps: {rep_count}", (50,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0),3)

    cv2.imshow("Jumping Jacks Live", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
