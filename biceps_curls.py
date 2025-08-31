import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

def calculate_angle(a, b, c):
    """Calculate angle between 3 points"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab = a - b
    cb = c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

# ---- Choose side to track ----
# "left"  = track left arm only
# "right" = track right arm only
SIDE = "right"

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO pose estimation
    results = model(frame)
    keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else None

    if keypoints is not None and len(keypoints) > 0:
        person = keypoints[0]  # First detected person

        try:
            if SIDE == "left":
                # LEFT arm (shoulder=5, elbow=7, wrist=9)
                shoulder, elbow, wrist = person[5], person[7], person[9]
                angle = calculate_angle(shoulder, elbow, wrist)
                label = "L"
            else:
                # RIGHT arm (shoulder=6, elbow=8, wrist=10)
                shoulder, elbow, wrist = person[6], person[8], person[10]
                angle = calculate_angle(shoulder, elbow, wrist)
                label = "R"

            # --- Feedback logic ---
            if angle > 160:
                feedback, color = f"{label}: Raise Arm Higher", (0, 0, 255)
            elif angle < 40:
                feedback, color = f"{label}: Lower Arm Down", (0, 0, 255)
            else:
                feedback, color = f"{label}: Good Curl!", (0, 255, 0)

            # Show angle
            cv2.putText(frame, f"{label}: {int(angle)} deg",
                        (int(elbow[0]), int(elbow[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Show feedback
            cv2.putText(frame, feedback, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        except Exception as e:
            pass

    # Draw skeleton only (no boxes)
    annotated_frame = results[0].plot(boxes=False)

    # Merge skeleton + feedback
    display_frame = cv2.addWeighted(annotated_frame, 0.7, frame, 0.3, 0)

    cv2.imshow("Bicep Curls (YOLOv8)", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
