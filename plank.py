import cv2
import numpy as np
import time
from ultralytics import YOLO
 

model = YOLO("yolov8n-pose.pt" )

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ma5tar el points deh 3al hasab el coco format
KEYPOINTS = {
    "right_shoulder": 6,
    "right_hip": 12,
    "right_ankle": 16
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return int(angle)

plank_active = False
start_time = 0
elapsed_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    for r in results:
        if r.keypoints is None:
            continue

        keypoints = r.keypoints.xy.cpu().numpy()

        for person in keypoints:
            r_shoulder = person[KEYPOINTS["right_shoulder"]]
            r_hip = person[KEYPOINTS["right_hip"]]
            r_ankle = person[KEYPOINTS["right_ankle"]]

        
            body_angle = calculate_angle(r_shoulder, r_hip, r_ankle)

            
            for point in [r_shoulder, r_hip, r_ankle]:
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)

            cv2.line(frame, tuple(r_shoulder.astype(int)), tuple(r_hip.astype(int)), (0, 0, 255), 2)
            cv2.line(frame, tuple(r_hip.astype(int)), tuple(r_ankle.astype(int)), (0, 0, 255), 2)

            # hena bashof el plank sa7 wla la2
            if 170 <= body_angle <= 190:
                if not plank_active:  # ba7seb men el zero
                    plank_active = True
                    start_time = time.time() - elapsed_time  # bakmel 3la el wa2t
                elapsed_time = time.time() - start_time
                status_text = "Plank Correct"
                color = (0, 255, 0)
            else:
                plank_active = False
                status_text = "Plank Incorrect"
                color = (0, 0, 255)

            # 3ashan a display el info
            cv2.putText(frame, f"Body Angle: {body_angle} deg", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, status_text, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"Time: {int(elapsed_time)} sec", (50, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

    cv2.imshow("Plank Timer", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

