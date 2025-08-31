import cv2
import numpy as np
import time
import streamlit as st
from collections import deque
from ultralytics import YOLO


# Load model
model = YOLO("yolov8n-pose.pt")

# Keypoints according to COCO format
PLANK_POINTS = {"shoulder": 6, "hip": 12, "ankle": 16}
PushUp_POINTS = {"shoulder": 6, "elbow": 8, "wrist": 10, "hip": 12, "ankle": 16,"ear": 4}
SQUAT_POINTS = {"hip": 12, "knee": 14, "ankle": 16}
JUMPING_POINTS = {"left_ankle": 15, "right_ankle": 16, "left_hip": 11, "right_hip": 12}
GLUTE_POINTS = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14
}
BICEP_POINTS = {
    "left_shoulder": 5, "left_elbow": 7, "left_wrist": 9,
    "right_shoulder": 6, "right_elbow": 8, "right_wrist": 10
}
FRONT_POINTS = {"right_shoulder": 6, "right_elbow": 8, "right_wrist": 10, "right_hip": 12}


LATERAL_POINTS = {
    "left_shoulder": 5, "left_elbow": 7, "left_hip": 11,
    "right_shoulder": 6, "right_elbow": 8, "right_hip": 12
}

def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return int(angle)

def run_exercise(source, exercise, side="right"):
    if source == "Camera":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)

    # Plank vars
    plank_active, start_time, elapsed_time = False, 0, 0
    
    # Pushup vars
    pushup_count, pushup_down, pushup_start = 0, False, time.time()

    # Squat vars
    squat_count, squat_down, squat_start = 0, False, time.time()

    # Jumping Jacks vars
    jj_stage, jj_reps, jj_last_time = "closed", 0, time.time()

    # Glute bridge vars
    gb_count, gb_stage, gb_locked = 0, "down", False
    gb_angle_buffer = deque(maxlen=5)

    # Bicep curls vars
    bc_reps, bc_stage = 0, "down"

    # Front raise vars
    elbow_buffer, shoulder_buffer = deque(maxlen=5), deque(maxlen=5)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        for r in results:
            if r.keypoints is None:
                continue

            keypoints = r.keypoints.xy.cpu().numpy()
            confs = r.keypoints.conf.cpu().numpy()

            for i, person in enumerate(keypoints):
                # ----------------- Plank -----------------
                if exercise == "Plank":
                    shoulder = person[PLANK_POINTS["shoulder"]]
                    hip = person[PLANK_POINTS["hip"]]
                    ankle = person[PLANK_POINTS["ankle"]]

                    angle = calculate_angle(shoulder, hip, ankle)

                    for p in [shoulder, hip, ankle]:
                        cv2.circle(frame, tuple(p.astype(int)), 6, (0, 255, 0), -1)
                    cv2.line(frame, tuple(shoulder.astype(int)), tuple(hip.astype(int)), (0, 0, 255), 2)
                    cv2.line(frame, tuple(hip.astype(int)), tuple(ankle.astype(int)), (0, 0, 255), 2)

                    if 170 <= angle <= 190:
                        if not plank_active:
                            plank_active = True
                            start_time = time.time() - elapsed_time
                        elapsed_time = time.time() - start_time
                        status_text, color = "Plank Correct", (0, 255, 0)
                    else:
                        plank_active = False
                        status_text, color = "Plank Incorrect", (0, 0, 255)

                    cv2.putText(frame, f"Body Angle: {angle} deg", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(frame, status_text, (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    cv2.putText(frame, f"Time: {int(elapsed_time)} sec", (50, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

                elif exercise == "PushUps":
                    shoulder = person[PushUp_POINTS["shoulder"]]
                    elbow = person[PushUp_POINTS["elbow"]]
                    wrist = person[PushUp_POINTS["wrist"]]
                    hip = person[PushUp_POINTS["hip"]]
                    ear = person[PushUp_POINTS["ear"]]
                    ankle = person[PushUp_POINTS["ankle"]]
    
                    angle = calculate_angle(ear, hip, ankle)
                    angle2 = calculate_angle(shoulder, elbow, wrist)
    
                    for p in [shoulder, elbow, wrist, hip, ear]:
                        cv2.circle(frame, tuple(p.astype(int)), 6, (0, 255, 0), -1)
                    cv2.line(frame, tuple(shoulder.astype(int)), tuple(elbow.astype(int)), (0, 0, 255), 2)
                    cv2.line(frame, tuple(elbow.astype(int)), tuple(wrist.astype(int)), (0, 0, 255), 2)

                    if angle2 < 90 and not pushup_down:
                        pushup_down = True
                    if angle2 > 130 and pushup_down:
                        pushup_count += 1
                        pushup_down = False

                    elapsed_time = int(time.time() - pushup_start)

                    cv2.putText(frame, f"Elbow Angle: {angle2} deg", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(frame, f"Push-Ups: {pushup_count}", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.putText(frame, f"Time: {elapsed_time} sec", (50, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
                
                    
                
                # ----------------- Squats -----------------
                elif exercise == "Squats":
                    hip = person[SQUAT_POINTS["hip"]]
                    knee = person[SQUAT_POINTS["knee"]]
                    ankle = person[SQUAT_POINTS["ankle"]]

                    angle = calculate_angle(hip, knee, ankle)

                    for p in [hip, knee, ankle]:
                        cv2.circle(frame, tuple(p.astype(int)), 6, (0, 255, 0), -1)
                    cv2.line(frame, tuple(hip.astype(int)), tuple(knee.astype(int)), (0, 0, 255), 2)
                    cv2.line(frame, tuple(knee.astype(int)), tuple(ankle.astype(int)), (0, 0, 255), 2)

                    if angle < 100:
                        squat_down = True
                    if angle > 160 and squat_down:
                        squat_count += 1
                        squat_down = False

                    elapsed_time = int(time.time() - squat_start)

                    cv2.putText(frame, f"Knee Angle: {angle} deg", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(frame, f"Squats: {squat_count}", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.putText(frame, f"Time: {elapsed_time} sec", (50, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

                # ----------------- Jumping Jacks -----------------
                elif exercise == "Jumping Jacks":
                    try:
                        l_ankle = person[JUMPING_POINTS["left_ankle"]]
                        r_ankle = person[JUMPING_POINTS["right_ankle"]]
                        l_hip = person[JUMPING_POINTS["left_hip"]]
                        r_hip = person[JUMPING_POINTS["right_hip"]]

                        c_l_ankle = confs[i][JUMPING_POINTS["left_ankle"]]
                        c_r_ankle = confs[i][JUMPING_POINTS["right_ankle"]]
                        c_l_hip = confs[i][JUMPING_POINTS["left_hip"]]
                        c_r_hip = confs[i][JUMPING_POINTS["right_hip"]]
                    except:
                        continue

                    if (c_l_ankle < 0.5 or c_r_ankle < 0.5 or 
                        c_l_hip < 0.5 or c_r_hip < 0.5):
                        cv2.putText(frame, "Move back - full body not visible!", (50,150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        continue

                    hip_mid = (l_hip + r_hip) / 2
                    feet_angle = calculate_angle(l_ankle, hip_mid, r_ankle)

                    for p in [l_ankle, r_ankle, l_hip, r_hip]:
                        cv2.circle(frame, tuple(p.astype(int)), 6, (0, 255, 0), -1)
                    cv2.line(frame, tuple(l_ankle.astype(int)), tuple(hip_mid.astype(int)), (0,0,255),2)
                    cv2.line(frame, tuple(r_ankle.astype(int)), tuple(hip_mid.astype(int)), (0,0,255),2)

                    now = time.time()
                    if jj_stage == "closed" and feet_angle > 50:
                        jj_stage = "open"
                    elif jj_stage == "open" and feet_angle < 30:
                        if now - jj_last_time > 0.8:
                            jj_reps += 1
                            jj_last_time = now
                        jj_stage = "closed"

                    cv2.putText(frame, f"Feet Angle: {feet_angle} deg", (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2)
                    cv2.putText(frame, f"Reps: {jj_reps}", (50,100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0),3)

                # ----------------- Glute Bridge -----------------
                elif exercise == "Glute Bridge":
                    shoulder = (person[GLUTE_POINTS["left_shoulder"]] + person[GLUTE_POINTS["right_shoulder"]]) / 2
                    hip = (person[GLUTE_POINTS["left_hip"]] + person[GLUTE_POINTS["right_hip"]]) / 2
                    knee = (person[GLUTE_POINTS["left_knee"]] + person[GLUTE_POINTS["right_knee"]]) / 2

                    angle = calculate_angle(shoulder, hip, knee)
                    gb_angle_buffer.append(angle)
                    smooth_angle = np.mean(gb_angle_buffer)

                    if not gb_locked:
                        if smooth_angle < 160:
                            gb_stage = "down"
                        elif smooth_angle > 172 and gb_stage == "down":
                            gb_count += 1
                            gb_stage = "up"
                            gb_locked = True
                    else:
                        if smooth_angle < 162:
                            gb_locked = False

                    for p in [shoulder, hip, knee]:
                        cv2.circle(frame, tuple(p.astype(int)), 6, (0,255,0), -1)
                    cv2.line(frame, tuple(shoulder.astype(int)), tuple(hip.astype(int)), (255,0,0), 3)
                    cv2.line(frame, tuple(hip.astype(int)), tuple(knee.astype(int)), (255,0,0), 3)

                    cv2.putText(frame, f"Hip Angle: {int(smooth_angle)} deg", (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2)
                    cv2.putText(frame, f"Reps: {gb_count}", (50,100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0),3)

                # ----------------- Bicep Curls -----------------
                elif exercise == "Bicep Curls":
                    if side == "left":
                        shoulder = person[BICEP_POINTS["left_shoulder"]]
                        elbow = person[BICEP_POINTS["left_elbow"]]
                        wrist = person[BICEP_POINTS["left_wrist"]]
                        label = "L"
                    else:
                        shoulder = person[BICEP_POINTS["right_shoulder"]]
                        elbow = person[BICEP_POINTS["right_elbow"]]
                        wrist = person[BICEP_POINTS["right_wrist"]]
                        label = "R"

                    angle = calculate_angle(shoulder, elbow, wrist)

                    if angle > 160:
                        bc_stage = "down"
                    elif angle < 40 and bc_stage == "down":
                        bc_reps += 1
                        bc_stage = "up"

                    for p in [shoulder, elbow, wrist]:
                        cv2.circle(frame, tuple(p.astype(int)), 6, (0, 255, 0), -1)
                    cv2.line(frame, tuple(shoulder.astype(int)), tuple(elbow.astype(int)), (255, 0, 0), 2)
                    cv2.line(frame, tuple(elbow.astype(int)), tuple(wrist.astype(int)), (255, 0, 0), 2)

                    cv2.putText(frame, f"{label} Arm Angle: {angle} deg", (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2)
                    cv2.putText(frame, f"Bicep Reps: {bc_reps}", (50,100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0),3)

                # ----------------- Front Raise -----------------
                elif exercise == "Front Raise":
                    shoulder = person[FRONT_POINTS["right_shoulder"]]
                    elbow = person[FRONT_POINTS["right_elbow"]]
                    wrist = person[FRONT_POINTS["right_wrist"]]
                    hip = person[FRONT_POINTS["right_hip"]]

                    elbow_angle = calculate_angle(shoulder, elbow, wrist)
                    shoulder_angle = calculate_angle(hip, shoulder, wrist)

                    elbow_buffer.append(elbow_angle)
                    shoulder_buffer.append(shoulder_angle)

                    elbow_angle = int(np.mean(elbow_buffer))
                    shoulder_angle = int(np.mean(shoulder_buffer))

                    if elbow_angle < 150:
                        feedback = "Keep arm straight!"
                        color = (0, 0, 255)
                    elif shoulder_angle > 95:
                        feedback = "Don't raise too high!"
                        color = (0, 0, 255)
                    else:
                        feedback = "Good Form!"
                        color = (0, 255, 0)

                    for p in [shoulder, elbow, wrist, hip]:
                        cv2.circle(frame, tuple(p.astype(int)), 6, color, -1)
                    cv2.line(frame, tuple(shoulder.astype(int)), tuple(elbow.astype(int)), color, 2)
                    cv2.line(frame, tuple(elbow.astype(int)), tuple(wrist.astype(int)), color, 2)

                    cv2.putText(frame, f"Elbow: {elbow_angle}", (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                    cv2.putText(frame, f"Shoulder: {shoulder_angle}", (50,90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                    cv2.putText(frame, feedback, (50,140),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    


                                    # ----------------- Lateral Arm Raise -----------------
                elif exercise == "Lateral Arm Raise":
                    r_shoulder = person[LATERAL_POINTS["right_shoulder"]]
                    r_elbow = person[LATERAL_POINTS["right_elbow"]]
                    r_hip = person[LATERAL_POINTS["right_hip"]]

                    l_shoulder = person[LATERAL_POINTS["left_shoulder"]]
                    l_elbow = person[LATERAL_POINTS["left_elbow"]]
                    l_hip = person[LATERAL_POINTS["left_hip"]]

                    r_angle = calculate_angle(r_elbow, r_shoulder, r_hip)
                    l_angle = calculate_angle(l_elbow, l_shoulder, l_hip)

                    # Right arm feedback
                    if r_angle < 15:
                        r_feedback, r_color = "Raise Higher (R)", (0, 0, 255)
                    elif 15 <= r_angle <= 100:
                        r_feedback, r_color = "Good Form (R)", (0, 255, 0)
                    else:
                        r_feedback, r_color = "Lower Arm (R)", (0, 0, 255)

                    # Left arm feedback
                    if l_angle < 15:
                        l_feedback, l_color = "Raise Higher (L)", (0, 0, 255)
                    elif 15 <= l_angle <= 100:
                        l_feedback, l_color = "Good Form (L)", (0, 255, 0)
                    else:
                        l_feedback, l_color = "Lower Arm (L)", (0, 0, 255)

                    # Draw skeleton points
                    for p in [r_shoulder, r_elbow, r_hip, l_shoulder, l_elbow, l_hip]:
                        cv2.circle(frame, tuple(p.astype(int)), 6, (255,255,0), -1)

                    # Show feedback
                    cv2.putText(frame, f"R: {int(r_angle)} deg", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, r_color, 2)
                    cv2.putText(frame, r_feedback, (50, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, r_color, 3)

                    cv2.putText(frame, f"L: {int(l_angle)} deg", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, l_color, 2)
                    cv2.putText(frame, l_feedback, (50, 190),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, l_color, 3)
                 

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()


# ------------------- Streamlit UI -------------------
st.title("Gym Assistant - Pose Exercises")

exercise = st.selectbox(
    "Choose Exercise:", 
    ["Plank", "Squats", "PushUps", "Jumping Jacks", "Glute Bridge", "Bicep Curls", "Front Raise", "Lateral Arm Raise"]
)


source_type = st.radio("Select Source:", ["Camera", "Upload Video"])

# Bicep curls side selector
side = "right"
if exercise == "Bicep Curls":
    side = st.radio("Choose Arm:", ["left", "right"])

video_file = None
if source_type == "Upload Video":
    video_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"])

if st.button("Start Exercise"):
    if source_type == "Camera":
        run_exercise("Camera", exercise, side)
    elif video_file is not None:
        tfile = "temp_video.mp4"
        with open(tfile, "wb") as f:
            f.write(video_file.read())
        run_exercise(tfile, exercise, side)
    else:
        st.warning("Please upload a video first!")
