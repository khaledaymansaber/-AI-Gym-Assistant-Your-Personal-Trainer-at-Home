import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Keypoints IDs (COCO format)
KEYPOINTS = {
    "right_hip": 12,
    "right_knee": 14,
    "right_ankle": 16
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return int(angle)

# Squat variables
squat_count = 0
squat_down = False
start_time = time.time()

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
            r_hip = person[KEYPOINTS["right_hip"]]
            r_knee = person[KEYPOINTS["right_knee"]]
            r_ankle = person[KEYPOINTS["right_ankle"]]

            # Calculate knee angle
            knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

            # Draw keypoints and lines
            for point in [r_hip, r_knee, r_ankle]:
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)

            cv2.line(frame, tuple(r_hip.astype(int)), tuple(r_knee.astype(int)), (0, 0, 255), 2)
            cv2.line(frame, tuple(r_knee.astype(int)), tuple(r_ankle.astype(int)), (0, 0, 255), 2)

            # Counting logic
            if knee_angle < 100:  # Down
                squat_down = True
            if knee_angle > 160 and squat_down:  # Up
                squat_count += 1
                squat_down = False

            elapsed_time = int(time.time() - start_time)

            # Display info
            cv2.putText(frame, f"Knee Angle: {knee_angle} deg", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Squats: {squat_count}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f"Time: {elapsed_time} sec", (50, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

    cv2.imshow("Squat Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import time
# from ultralytics import YOLO

# # تحميل الموديل
# model = YOLO("yolov8n-pose.pt")

# # بدل الكاميرا -> حط مسار الفيديو
# cap = cv2.VideoCapture("8836896-uhd_4096_2160_25fps.mp4")   # 👈 غير الاسم للمسار عندك

# if not cap.isOpened():
#     raise RuntimeError("Video not accessible")

# # keypoints IDs طبقاً ل COCO format
# KEYPOINTS = {
#     "right_hip": 12,
#     "right_knee": 14,
#     "right_ankle": 16
# }

# def calculate_angle(a, b, c):
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     ba = a - b
#     bc = c - b
#     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
#     angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
#     return int(angle)

# # متغيرات السكوات
# squat_count = 0
# squat_down = False
# start_time = time.time()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break  # خلص الفيديو

#     results = model(frame, verbose=False)

#     for r in results:
#         if r.keypoints is None:
#             continue

#         keypoints = r.keypoints.xy.cpu().numpy()

#         for person in keypoints:
#             r_hip = person[KEYPOINTS["right_hip"]]
#             r_knee = person[KEYPOINTS["right_knee"]]
#             r_ankle = person[KEYPOINTS["right_ankle"]]

#             # حساب زاوية الركبة
#             knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

#             # رسم النقاط والخطوط
#             for point in [r_hip, r_knee, r_ankle]:
#                 x, y = int(point[0]), int(point[1])
#                 cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)

#             cv2.line(frame, tuple(r_hip.astype(int)), tuple(r_knee.astype(int)), (0, 0, 255), 2)
#             cv2.line(frame, tuple(r_knee.astype(int)), tuple(r_ankle.astype(int)), (0, 0, 255), 2)

#             # منطق العدّ
#             if knee_angle < 100:  # نزل تحت
#                 squat_down = True
#             if knee_angle > 160 and squat_down:  # طلع فوق
#                 squat_count += 1
#                 squat_down = False

#             elapsed_time = int(time.time() - start_time)

#             # عرض المعلومات
#             cv2.putText(frame, f"Knee Angle: {knee_angle} deg", (50, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#             cv2.putText(frame, f"Squats: {squat_count}", (50, 100),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
#             cv2.putText(frame, f"Time: {elapsed_time} sec", (50, 160),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

#         # --- كروب للشخص باستخدام البوكس (اختياري) ---
#         if r.boxes is not None:
#             for box in r.boxes.xyxy.cpu().numpy():
#                 x1, y1, x2, y2 = box.astype(int)
#                 person_frame = frame[y1:y2, x1:x2]
#                 if person_frame.size > 0:
#                     person_frame = cv2.resize(person_frame, (480, 640))
#                     cv2.imshow("Person Crop", person_frame)

#     # --- Resize العرض الأساسي عشان ميكونش زووم ---
#     display_frame = cv2.resize(frame, (1280, 720))
#     cv2.imshow("Squat Counter - Video", display_frame)

#     if cv2.waitKey(30) & 0xFF == ord("q"):  # 👈 30ms delay = تقريبا 33fps
#         break

# cap.release()
# cv2.destroyAllWindows()
