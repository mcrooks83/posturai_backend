import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- Helper functions ---

def get_xy(landmark_id, landmarks, w, h):
    lm = landmarks[landmark_id]
    return int(lm.x * w), int(lm.y * h)

def get_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx))

def describe_tilt(label, angle):
    abs_angle = abs(angle)
    if abs_angle < 2:
        return f"{label}: Balanced"
    elif abs_angle < 5:
        side = "left" if angle > 0 else "right"
        return f"{label}: Slight drop on {side}"
    else:
        side = "left" if angle > 0 else "right"
        return f"{label}: Elevation on {side}"

def knee_deviation(hip, knee, ankle):
    hip_x, _ = hip
    knee_x, _ = knee
    ankle_x, _ = ankle
    mid_x = (hip_x + ankle_x) / 2
    deviation = knee_x - mid_x
    return deviation

def signed_angle_between(v1, v2):
    # v1 and v2 are 2D vectors: [x, y]
    v1 = np.array(v1, dtype=np.float32)
    v2 = np.array(v2, dtype=np.float32)
    
    angle = np.arctan2(
        v1[0]*v2[1] - v1[1]*v2[0],  # cross product
        v1[0]*v2[0] + v1[1]*v2[1]   # dot product
    )
    return np.degrees(angle)  # returns angle in degrees (-180, +180)

def vertical_deviation(p_top, p_bottom):
    dx = p_bottom[0] - p_top[0]
    dy = p_bottom[1] - p_top[1]
    return np.degrees(np.arctan2(dx, dy))  # horizontal deviation from vertical

# --- Load image and pose estimation ---
image_path = 'andrea_front.jpg'  # <--- change to your image file
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, _ = image.shape

with mp_pose.Pose(static_image_mode=True) as pose:
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        print("No pose detected.")
        exit()
    landmarks = results.pose_landmarks.landmark

    # Landmark IDs
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_KNEE, RIGHT_KNEE = 25, 26
    LEFT_ANKLE, RIGHT_ANKLE = 27, 28
    LEFT_EYE_INNER, RIGHT_EYE_INNER = 1, 4
    NOSE = 0

    # --- Extract points ---
    ls_x, ls_y = get_xy(LEFT_SHOULDER, landmarks, w, h)
    rs_x, rs_y = get_xy(RIGHT_SHOULDER, landmarks, w, h)
    lh_x, lh_y = get_xy(LEFT_HIP, landmarks, w, h)
    rh_x, rh_y = get_xy(RIGHT_HIP, landmarks, w, h)
    lk_x, lk_y = get_xy(LEFT_KNEE, landmarks, w, h)
    rk_x, rk_y = get_xy(RIGHT_KNEE, landmarks, w, h)
    la_x, la_y = get_xy(LEFT_ANKLE, landmarks, w, h)
    ra_x, ra_y = get_xy(RIGHT_ANKLE, landmarks, w, h)
    leye_x, leye_y = get_xy(LEFT_EYE_INNER, landmarks, w, h)
    reye_x, reye_y = get_xy(RIGHT_EYE_INNER, landmarks, w, h)
    nose_x, nose_y = get_xy(NOSE, landmarks, w, h)

    # Midpoints
    shoulder_mid = ((ls_x + rs_x) // 2, (ls_y + rs_y) // 2)
    hip_mid = ((lh_x + rh_x) // 2, (lh_y + rh_y) // 2)
    hip_center_x = (lh_x + rh_x) // 2
    hip_center_y = (lh_y + rh_y) // 2 + 150  # triangle apex Y (adjust height)

    head_mid = (nose_x, nose_y)

    # --- Calculations ---

    # Head tilt angle (eyes)
    head_angle = get_angle((leye_x, leye_y), (reye_x, reye_y))
    head_description = describe_tilt("Head tilt", head_angle)

    # Neck lateral flexion (angle between neck and spine vectors)
    vec_neck = (head_mid[0] - shoulder_mid[0], head_mid[1] - shoulder_mid[1])
    vec_spine = (hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1])
    angle_between = signed_angle_between(vec_spine, vec_neck)

    if angle_between > 5:
        neck_flexion = "Neck veers left of spine → spine  flexed right "
    elif angle_between < -5:
        neck_flexion = "Neck veers right of spine → spine  flexed left"
    else:
        neck_flexion = "Neck aligned with spine"

    # Shoulder and hip tilt angles
    shoulder_angle = get_angle((ls_x, ls_y), (rs_x, rs_y))
    hip_angle = get_angle((lh_x, lh_y), (rh_x, rh_y))
    shoulder_description = describe_tilt("Shoulders", shoulder_angle)
    hip_description = describe_tilt("Hike", hip_angle)

    # Knee deviation for internal rotation
    left_dev = knee_deviation((lh_x, lh_y), (lk_x, lk_y), (la_x, la_y))
    right_dev = knee_deviation((rh_x, rh_y), (rk_x, rk_y), (ra_x, ra_y))

    threshold = 10  # pixels
    if abs(left_dev) < threshold and abs(right_dev) < threshold:
        knee_rotation_report = "Both knees show similar alignment"
    else:
        left_internal = left_dev < 0
        right_internal = right_dev < 0
        if left_internal and right_internal:
            if abs(left_dev) > abs(right_dev):
                knee_rotation_report = "Left knee shows more internal rotation"
            elif abs(right_dev) > abs(left_dev):
                knee_rotation_report = "Right knee shows more internal rotation"
            else:
                knee_rotation_report = "Both knees show equal internal rotation"
        elif left_internal:
            knee_rotation_report = "Left knee shows internal rotation; right knee less so"
        elif right_internal:
            knee_rotation_report = "Right knee shows internal rotation; left knee less so"
        else:
            knee_rotation_report = "Neither knee shows internal rotation"

    # Pelvic rotation estimation using shoulder-hip distances
   # Improved Pelvic Rotation Estimation (based only on hips)
    hip_mid_x = (lh_x + rh_x) // 2  # midpoint between left and right hip

    left_hip_offset = lh_x - nose_x
    right_hip_offset = rh_x - nose_x
    pelvis_offset = hip_mid[0] - nose_x

    print(f"hip offsets {left_hip_offset}, {right_hip_offset}, {pelvis_offset}")

    if abs(left_hip_offset - right_hip_offset) < 20:
        pelvis_rotation = "Pelvis: Neutral"
    elif left_hip_offset > abs(right_hip_offset):
        pelvis_rotation = "Pelvis: Rotated right (left hip forward)"
    else:
        pelvis_rotation = "Pelvis: Rotated left (right hip forward)"



    # Spinal lateral flexion: torso lengths difference
    left_torso_len = abs(lh_y - ls_y)
    right_torso_len = abs(rh_y - rs_y)
    torso_diff = left_torso_len - right_torso_len
    if abs(torso_diff) < 10:
        spine_status = "Spine: Balanced (no clear lateral flexion)"
    elif torso_diff > 0:
        spine_status = "Spine: Flexed to right (shorter right side)"
    else:
        spine_status = "Spine: Flexed to left (shorter left side)"

    # Spine rotation using shoulder distances to hip midpoint
    left_shoulder_dist = abs(ls_x - hip_mid[0])
    right_shoulder_dist = abs(rs_x - hip_mid[0])
    print(f"left should dist {left_shoulder_dist}, right should dist {right_shoulder_dist}")
    diff = left_shoulder_dist - right_shoulder_dist
    if abs(diff) < 5:
        spine_rotation = "Spine: Facing forward (neutral)"
    elif diff > 0:
        spine_rotation = "Spine: Rotated toward left (right shoulder forward)"
    else:
        spine_rotation = "Spine: Rotated toward right (left shoulder forward)"

    # --- Drawing ---
    annotated_image = image.copy()

    # Pelvis triangle
    triangle_pts = np.array([
        [lh_x, lh_y],
        [rh_x, rh_y],
        [hip_center_x, hip_center_y]
    ], np.int32)
    cv2.line(annotated_image, (lh_x, lh_y), (rh_x, rh_y), (0, 255, 0), 10)  # Hips green
    cv2.polylines(annotated_image, [triangle_pts], isClosed=True, color=(0, 225, 0), thickness=3)

    # Shoulders
    cv2.line(annotated_image, (ls_x, ls_y), (rs_x, rs_y), (0, 0, 255), 10)  # Shoulders red

    # Head tilt (eyes)
    cv2.line(annotated_image, (leye_x, leye_y), (reye_x, reye_y), (255, 0, 0), 5)

    # Neck line
    cv2.line(annotated_image, shoulder_mid, head_mid, (0, 128, 255), 4)

    # Spine line
    cv2.line(annotated_image, shoulder_mid, hip_mid, (128, 0, 128), 5)

    # Left and right torso lines
    cv2.line(annotated_image, (ls_x, ls_y), (lh_x, lh_y), (200, 100, 255), 4)  # Left torso
    cv2.line(annotated_image, (rs_x, rs_y), (rh_x, rh_y), (200, 255, 100), 4)  # Right torso

    # Knees (hip-knee-ankle)
    cv2.line(annotated_image, (lh_x, lh_y), (lk_x, lk_y), (0, 255, 255), 5)  # Left hip-knee cyan
    cv2.line(annotated_image, (lk_x, lk_y), (la_x, la_y), (0, 255, 255), 5)  # Left knee-ankle cyan
    cv2.line(annotated_image, (rh_x, rh_y), (rk_x, rk_y), (255, 255, 0), 5)  # Right hip-knee yellow
    cv2.line(annotated_image, (rk_x, rk_y), (ra_x, ra_y), (255, 255, 0), 5)  # Right knee-ankle yellow

    # Draw landmarks
    mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Convert image for matplotlib
    img_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    ax1.imshow(img_rgb)
    ax1.axis('off')
    ax1.set_title("Posture Analysis", fontsize=14)

    ax2.axis('off')
    ax2.set_title("Assessment Report", fontsize=14, loc='left')

    text = f"""
HEAD
{head_description}

{neck_flexion}

{shoulder_description}

{spine_status}
{spine_rotation}

PELVIS
{hip_description}
{pelvis_rotation}

{knee_rotation_report}
"""
    ax2.text(0, 1, text, fontsize=14, verticalalignment='top', family='monospace')

    plt.tight_layout()
    plt.show()
