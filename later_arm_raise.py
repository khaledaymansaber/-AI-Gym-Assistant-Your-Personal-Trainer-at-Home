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

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 pose estimation
    results = model(frame)
    keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else None

    if keypoints is not None and len(keypoints) > 0:
        person = keypoints[0]  # first detected person

        try:
            # Right arm (shoulder=6, elbow=8, hip=12)
            r_shoulder = person[6]
            r_elbow = person[8]
            r_hip = person[12]

            # Left arm (shoulder=5, elbow=7, hip=11)
            l_shoulder = person[5]
            l_elbow = person[7]
            l_hip = person[11]

            # Calculate angles
            r_angle = calculate_angle(r_elbow, r_shoulder, r_hip)
            l_angle = calculate_angle(l_elbow, l_shoulder, l_hip)

            # --- Feedback for Right Arm ---
            if r_angle < 15:
                r_feedback = "Raise Higher (R)"
                r_color = (0, 0, 255)
            elif 15 <= r_angle <= 100:
                r_feedback = "Good Form (R)"
                r_color = (0, 255, 0)
            else:
                r_feedback = "Lower Arm (R)"
                r_color = (0, 0, 255)

            # --- Feedback for Left Arm ---
            if l_angle < 15:
                l_feedback = "Raise Higher (L)"
                l_color = (0, 0, 255)
            elif 15 <= l_angle <= 100:
                l_feedback = "Good Form (L)"
                l_color = (0, 255, 0)
            else:
                l_feedback = "Lower Arm (L)"
                l_color = (0, 0, 255)

            # Show angles + feedback text
            cv2.putText(frame, f"R: {int(r_angle)} deg", (int(r_shoulder[0]), int(r_shoulder[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, r_feedback, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, r_color, 3)

            cv2.putText(frame, f"L: {int(l_angle)} deg", (int(l_shoulder[0]), int(l_shoulder[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, l_feedback, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, l_color, 3)

        except Exception as e:
            pass

    # Draw skeleton
    annotated_frame = results[0].plot(boxes=False)  # 🚨 remove round boxes

    # Merge with feedback
    display_frame = cv2.addWeighted(annotated_frame, 0.7, frame, 0.3, 0)

    cv2.imshow("Lateral Arm Raise (YOLOv8)", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()