from ultralytics import YOLO
import cv2
import numpy as np
import math

class KeypointIndex:
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

def calculate_distance(point1, point2):
    if point1 is None or point2 is None:
        return 0
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

def calculate_angle( p1, p2, p3):
        """Calculate angle between three points"""
        if any(coord is None for point in [p1, p2, p3] for coord in point):
            return 0
            
        a = math.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2)
        b = math.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
        c = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        if a * c == 0:
            return 0
            
        try:
            angle_rad = math.acos((a**2 + c**2 - b**2) / (2 * a * c))
            return angle_rad * 180 / math.pi
        except ValueError:
            return 0

# Load model
model = YOLO('yolov8n-pose.pt')

# Load video
video_path = r"Copy of push up 115.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0

pushup_count = 0
state = "up"  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

   
    results = model(frame, show_boxes=False)

    for r in results:
       
        im = r.plot(boxes=False)

        keypoints = r.keypoints.xy

        if len(keypoints) > 0:
            person_kpts = keypoints[0]  
            for idx, (x, y) in enumerate(person_kpts):
                if x > 0 and y > 0:
                    # Draw keypoint index
                    cv2.putText(im, str(idx), (int(x), int(y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            dist = calculate_distance(
                person_kpts[KeypointIndex.RIGHT_SHOULDER],
                person_kpts[KeypointIndex.RIGHT_WRIST]
            )
            print(f"Frame {frame_count}: Distance (R shoulder to R wrist): {dist:.2f}")

            angle = calculate_angle(
                person_kpts[KeypointIndex.RIGHT_EAR],
                person_kpts[KeypointIndex.RIGHT_HIP],
                person_kpts[KeypointIndex.RIGHT_ANKLE]
            )

            angle2 = calculate_angle(
                person_kpts[KeypointIndex.RIGHT_SHOULDER],
                person_kpts[KeypointIndex.RIGHT_ELBOW],
                person_kpts[KeypointIndex.RIGHT_WRIST]
            )
            print(f"Frame {frame_count}: Angle (R ear, R hip, R ankle): {angle:.2f}")
            print(f"Frame {frame_count}: Angle (R shoulder, R elbow, R wrist): {angle2:.2f}")

            if angle2 < 90 and state == "up":
                state = "down"
            elif angle2 > 130 and state == "down":
                state = "up"
                pushup_count += 1  # Count when returning to up position
                print(f"Push-up completed! Total count: {pushup_count}")

            # Display pushup count and detection
            if state == "down" or angle2 < 90:
                # Display "Excellent work" on the image
                cv2.putText(im, "Excellent work!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.putText(im, f"Count: {pushup_count}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Pose Estimation with Keypoint Numbers", im)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"Final pushup count: {pushup_count}")
