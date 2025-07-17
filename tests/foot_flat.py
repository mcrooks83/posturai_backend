import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

video_path = '../../test_videos/m_1.mp4'
cap = cv2.VideoCapture(video_path)

max_display_height = 800  # output window height

if not cap.isOpened():
    print(f"Error: Could not open video at path: {video_path}")
    exit()

velocity_threshold = 2.0  # max vertical pixels movement to consider foot stationary
floor_threshold = 0.85    # normalized y (0 top, 1 bottom); ankle must be below this to consider on floor

prev_r_ankle_y = None
prev_l_ankle_y = None

detecting_right_foot = True  # start by detecting right foot contact

with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        height, width = frame.shape[:2]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

            # Current ankle y normalized (0 top, 1 bottom)
            r_ankle_y = r_ankle.y
            l_ankle_y = l_ankle.y

            if detecting_right_foot and prev_r_ankle_y is not None:
                r_vel = abs(r_ankle_y - prev_r_ankle_y)
                # Check if right ankle is near floor and velocity low (foot on ground)
                if r_vel < velocity_threshold / height and r_ankle_y > floor_threshold:
                    print(f"Right foot contact detected at frame")
                    # Resize freeze frame before showing
                    scale = max_display_height / frame.shape[0]
                    new_width = int(frame.shape[1] * scale)
                    new_height = int(frame.shape[0] * scale)
                    resized_frame = cv2.resize(frame, (new_width, new_height))
                    cv2.imshow('Foot Contact Freeze Frame', resized_frame)
                    key = cv2.waitKey(0)  # wait for key press
                    if key == ord('q'):
                        break
                    detecting_right_foot = False  # next detect left foot

            elif not detecting_right_foot and prev_l_ankle_y is not None:
                l_vel = abs(l_ankle_y - prev_l_ankle_y)
                print(l_vel)
                if l_vel < velocity_threshold / height and l_ankle_y > floor_threshold:
                    print(f"Left foot contact detected at frame")
                    scale = max_display_height / frame.shape[0]
                    new_width = int(frame.shape[1] * scale)
                    new_height = int(frame.shape[0] * scale)
                    resized_frame = cv2.resize(frame, (new_width, new_height))
                    cv2.imshow('Foot Contact Freeze Frame', resized_frame)
                    key = cv2.waitKey(0)
                    if key == ord('q'):
                        break
                    detecting_right_foot = True  # switch back to right foot

            prev_r_ankle_y = r_ankle_y
            prev_l_ankle_y = l_ankle_y

        # Show normal video feed resized
        scale = max_display_height / frame.shape[0]
        new_width = int(frame.shape[1] * scale)
        new_height = int(frame.shape[0] * scale)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        cv2.imshow('Video', resized_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
