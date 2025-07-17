import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

max_display_height = 800  # output window height

video_path = '../../test_videos/m_1.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video at path: {video_path}")
else:
    print("Video loaded successfully!")

hip_max_diff = 0  # max difference in hip y (r_hip_y - l_hip_y)
max_right_knee_deviation = 0
max_left_knee_deviation = 0

threshold = 0  # pixels, to ignore small jitter

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ignore_last_n = int(fps * 0.5)  # ignore last 1 second

frame_idx = 0

# 🔹 Function to compute angle using cosine rule
def calculate_angle(a, b, c):
    ab = a - b
    cb = c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # numerical stability
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        height, width = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            if frame_idx < total_frames - ignore_last_n:
                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                l_hip_y = l_hip.y * height
                r_hip_y = r_hip.y * height
                hip_diff = r_hip_y - l_hip_y

                if abs(hip_diff) > abs(hip_max_diff):
                    hip_max_diff = hip_diff

                if hip_max_diff > threshold:
                    label = f"Left hip hikes more by {hip_max_diff:.1f}px"
                    color = (0, 255, 0)
                elif hip_max_diff < -threshold:
                    label = f"Right hip hikes more by {-hip_max_diff:.1f}px"
                    color = (0, 0, 255)
                else:
                    label = "Hips balanced"
                    color = (255, 255, 255)

                cv2.putText(frame, label, (30, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

                cv2.putText(frame, f"Current hip diff: {hip_diff:.1f}px", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv2.LINE_AA)

                cv2.putText(frame, f"Left hip y: {l_hip_y:.1f}px", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Right hip y: {r_hip_y:.1f}px", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.circle(frame, (int(l_hip.x * width), int(l_hip_y)), 10, (0, 255, 0), -1)
                cv2.circle(frame, (int(r_hip.x * width), int(r_hip_y)), 10, (0, 0, 255), -1)

                # 🔹 Leg joints
                r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
                r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

                # 🔹 Convert to pixel coordinates for both sides
                r_hip_pt = np.array([r_hip.x * width, r_hip.y * height])
                r_knee_pt = np.array([r_knee.x * width, r_knee.y * height])
                r_ankle_pt = np.array([r_ankle.x * width, r_ankle.y * height])

                l_hip_pt = np.array([l_hip.x * width, l_hip.y * height])
                l_knee_pt = np.array([l_knee.x * width, l_knee.y * height])
                l_ankle_pt = np.array([l_ankle.x * width, l_ankle.y * height])

                # 🔹 Calculate angles
                right_knee_angle = calculate_angle(r_hip_pt, r_knee_pt, r_ankle_pt)
                left_knee_angle = calculate_angle(l_hip_pt, l_knee_pt, l_ankle_pt)

                right_dev = abs(180 - right_knee_angle)
                left_dev = abs(180 - left_knee_angle)

                if right_dev > max_right_knee_deviation:
                    max_right_knee_deviation = right_dev

                if left_dev > max_left_knee_deviation:
                    max_left_knee_deviation = left_dev

                # 🔹 Display angles
                cv2.putText(frame, f"Right knee angle: {right_knee_angle:.1f}°", (30, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Left knee angle: {left_knee_angle:.1f}°", (30, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

                # Optional: draw dots
                cv2.circle(frame, tuple(r_knee_pt.astype(int)), 8, (255, 255, 0), -1)
                cv2.circle(frame, tuple(l_knee_pt.astype(int)), 8, (0, 255, 255), -1)

                # 🔹 Draw pose landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
                )

        scale = max_display_height / frame.shape[0]
        new_width = int(frame.shape[1] * scale)
        new_height = int(frame.shape[0] * scale)
        resized_frame = cv2.resize(frame, (new_width, new_height))

        cv2.imshow('Pose Overlay', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("\n=== Final Hip Hike Summary ===")
print(f"Max hip y difference (r_hip_y - l_hip_y): {hip_max_diff:.1f} pixels")

if hip_max_diff > threshold:
    print("Left hip hiked more overall.")
elif hip_max_diff < -threshold:
    print("Right hip hiked more overall.")
else:
    print("Hips were balanced overall.")

print("\n=== Final Knee HKA Deviation Summary ===")
print(f"Max right knee deviation from 180°: {max_right_knee_deviation:.1f}°")
print(f"Max left knee deviation from 180°: {max_left_knee_deviation:.1f}°")

knee_threshold = 3  # degrees, small threshold to ignore noise

if max_left_knee_deviation - max_right_knee_deviation > knee_threshold:
    print("Left knee has greater HKA deviation overall.")
elif max_right_knee_deviation - max_left_knee_deviation > knee_threshold:
    print("Right knee has greater HKA deviation overall.")
else:
    print("Knees have similar HKA deviation overall.")
