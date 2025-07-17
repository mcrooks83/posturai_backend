import cv2
import mediapipe as mp
import numpy as np
import os

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

# tilt is used for head, shoulders and pelvis
def describe_tilt(label, angle):
    abs_angle = abs(angle)
    description = ""
    if abs_angle < 2:
        if label == "shoulders":
            description = f"Your {label} are pretty balanced"
        else:
            description = f"Your {label} is pretty balanced"
        return description
    elif abs_angle < 5:
        side = "left" if angle > 0 else "right"
        if label == "shoulders":
            f"Your {label} are dropped on the {side}"
        else:
            f"Your {label} is dropped on the {side}"
        return description
    else:
        side = "left" if angle > 0 else "right"
        if label == "shoulders":
            description =  f"Your {label} are elevated on the {side}"
        else:
           description =  f"Your {label} is elevated on the {side}"
        return description

def knee_deviation(hip, knee, ankle):
    hip_x, _ = hip
    knee_x, _ = knee
    ankle_x, _ = ankle
    mid_x = (hip_x + ankle_x) / 2
    deviation = knee_x - mid_x
    return deviation

def signed_angle_between(v1, v2):
    v1 = np.array(v1, dtype=np.float32)
    v2 = np.array(v2, dtype=np.float32)
    angle = np.arctan2(
        v1[0]*v2[1] - v1[1]*v2[0],  # cross product
        v1[0]*v2[0] + v1[1]*v2[1]   # dot product
    )
    return np.degrees(angle)


# --- functions to assess body segements --- 

def center_axis():
    pass

# --- Main analysis & annotation function ---

def analyze_posture_and_annotate(image, landmarks):
    h, w, _ = image.shape
    lm = landmarks.landmark

    # Landmark IDs
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_KNEE, RIGHT_KNEE = 25, 26
    LEFT_ANKLE, RIGHT_ANKLE = 27, 28
    LEFT_EYE_INNER, RIGHT_EYE_INNER = 1, 4
    NOSE = 0
    LEFT_TOE, RIGHT_TOE = 31,32

    # Extract points
    ls = get_xy(LEFT_SHOULDER, lm, w, h)
    rs = get_xy(RIGHT_SHOULDER, lm, w, h)
    lh = get_xy(LEFT_HIP, lm, w, h)
    rh = get_xy(RIGHT_HIP, lm, w, h)
    lk = get_xy(LEFT_KNEE, lm, w, h)
    rk = get_xy(RIGHT_KNEE, lm, w, h)
    la = get_xy(LEFT_ANKLE, lm, w, h)
    ra = get_xy(RIGHT_ANKLE, lm, w, h)
    leye = get_xy(LEFT_EYE_INNER, lm, w, h)
    reye = get_xy(RIGHT_EYE_INNER, lm, w, h)
    nose = get_xy(NOSE, lm, w, h)
    ltoe = get_xy(LEFT_TOE, lm, w, h)
    rtoe = get_xy(RIGHT_TOE, lm, w, h)

    def get_face_side_centers(landmarks, w, h):
        # Define face landmark indices for each side
        left_indices = [1, 2, 3, 7, 9]
        right_indices = [4, 5, 6, 8, 10]

        # Helper: average pixel coordinates using get_xy
        def average_coords(indices):
            xs, ys = [], []
            for i in indices:
                x, y = get_xy(i, landmarks, w, h)
                xs.append(x)
                ys.append(y)
            return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

        left_face = average_coords(left_indices)
        right_face = average_coords(right_indices)
        #print(f"Left face center: {left_face}, Right face center: {right_face}")
        return left_face, right_face

    # mid points
    shoulder_mid = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
   
    hip_mid = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
    
    hip_center_x = (lh[0] + rh[0]) // 2
    hip_center_y = (lh[1] + rh[1]) // 2 + 150  # triangle apex Y (adjust if needed)
    head_mid = nose

    # compute a base mid point (avg of toe index and ankle)
    mid_ankle = ((la[0] + ra[0]) // 2, (la[1] + ra[1]) // 2)
    mid_toe = ((ltoe[0] + rtoe[0]) // 2, (ltoe[1] + rtoe[1]) // 2)
    base_mid = ((mid_ankle[0] + mid_toe[0]) // 2, (mid_ankle[1] + mid_toe[1]) // 2)

    # HEAD
    # average all points on the left face and right face
    left_face, right_face = get_face_side_centers(lm, w, h)
    #head_angle = get_angle(leye, reye)
    head_angle = get_angle(left_face, right_face)
    head_description = describe_tilt("head", head_angle)

    # NECK
    # neck vector is reveresed direction than the others (should make consistent)
    vec_neck = (head_mid[0] - shoulder_mid[0], head_mid[1] - shoulder_mid[1])
    vec_spine = (hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1])
    
    angle_between = signed_angle_between(vec_spine, vec_neck)
    if angle_between > 5:
        neck_flexion = "Your neck (cervical spine) is laterally flexed to the right"
    elif angle_between < -5:
        neck_flexion = "Your neck (cevical spine) is laterally flexed to the left"
    else:
        neck_flexion = "Your neck (cervical spine) is alinged with the spine"

    shoulder_angle = get_angle(ls, rs)
    hip_angle = get_angle(lh, rh)
    shoulder_description = describe_tilt("shoulders", shoulder_angle)
    pelvis_status = describe_tilt("pelvis", hip_angle)


    # --- knee or femur internal rotation --- #
    left_dev = knee_deviation(lh, lk, la)
    right_dev = knee_deviation(rh, rk, ra)
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

    # Pelvic rotation estimation
    left_hip_offset = lh[0] - base_mid[0] # left hip x position minus the base line x position
    right_hip_offset = rh[0] - base_mid[0]

    if abs(left_hip_offset - right_hip_offset) < 20:
        pelvis_status += " and the pelvis is neutral"
    elif left_hip_offset > right_hip_offset:
        pelvis_status += " and the pelvis is rotated right (left hip forward)"
    else:
        pelvis_status += " and the pelvis is rotated left (right hip forward)"

    # spine direction relative to the base_mid
    vec_center_axis = (0, 1)  # Downward vertical in image coordinates,  does assume that the image is straight
    spine_flex_angle = signed_angle_between(vec_center_axis, vec_spine)
    # print(f"spine flex angle {spine_flex_angle}", flush=True)

    # Spinal lateral flexion
    if abs(spine_flex_angle) < 1:
        spine_status = "Your spine appears vertically aligned"
    elif 1 < spine_flex_angle < 5:
        spine_status = "Your spine is slightly flexed to the left"
    elif spine_flex_angle >= 5:
        spine_status = "Your spine is flexed to the left"
    elif -5 < spine_flex_angle < -1:
        spine_status = "Your spine is slightly flexed to the right"
    elif spine_flex_angle <= -5:
        spine_status = "Your spine is flexed to the right"

    # Spinal rotation
    left_shoulder_dist = abs(ls[0] - hip_mid[0])
    right_shoulder_dist = abs(rs[0] - hip_mid[0])
    rotation_diff = left_shoulder_dist - right_shoulder_dist

    if abs(rotation_diff) < 5:
        spine_status += " and is facing forward (neutral rotation)."
    elif rotation_diff > 0:
        spine_status += " and is rotated toward the left (right shoulder forward)."
    else:
        spine_status += " and is rotated toward the right (left shoulder forward)."


    # --- Annotate image ---
    annotated_image = image.copy()

    triangle_pts = np.array([lh, rh, (hip_center_x, hip_center_y)], np.int32)
    #cv2.line(annotated_image, lh, rh, (0, 255, 0), 10)
    #cv2.polylines(annotated_image, [triangle_pts], isClosed=True, color=(0, 225, 0), thickness=6)
    #cv2.line(annotated_image, ls, rs, (0, 255, 0), 10)  # Shoulders
    #cv2.line(annotated_image, left_face, right_face, (0, 255, 0), 10)  # Head tilt
    #cv2.line(annotated_image, shoulder_mid, head_mid, (0, 255, 0), 10)  # Neck line
    #cv2.line(annotated_image, shoulder_mid, hip_mid, (0, 255, 0), 10)  # Spine line
    #cv2.line(annotated_image, base_mid, hip_mid, (0, 225, 0), 5)  # base to pelvis
    #cv2.line(annotated_image, lh, lk, (0, 255, 0), 10) 
    #cv2.line(annotated_image, rh, rk, (0, 255, 0), 10) 
    #cv2.line(annotated_image, ls, lh, (200, 100, 255), 4)  # Left torso
    #cv2.line(annotated_image, rs, rh, (200, 255, 100), 4)  # Right torso

    # --- center axis --- #
    cv2.line(
        annotated_image, 
        (base_mid[0], 0), 
        (base_mid[0], h),  # h from shape of image
        (0, 0, 0), 
        thickness=5, 
        lineType=cv2.LINE_AA
    )

    cv2.putText(
        annotated_image,
        "Center Axis",
        (base_mid[0]+10, 50),  # Adjust X and Y as needed
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,             # Font scale
        (0, 0, 0),       # Text color (black)
        2,               # Thickness
        cv2.LINE_AA
    )
    # --- end of center axis --- #

    # Knees
    #cv2.line(annotated_image, lh, lk, (0, 255, 255), 5)
    #cv2.line(annotated_image, lk, la, (0, 255, 255), 5)
    #cv2.line(annotated_image, rh, rk, (255, 255, 0), 5)
    #cv2.line(annotated_image, rk, ra, (255, 255, 0), 5)

    mp_drawing.draw_landmarks(
        annotated_image, landmarks, mp_pose.POSE_CONNECTIONS,
        connection_drawing_spec = mp_drawing.DrawingSpec(color=(237, 114, 90), thickness=5)  # thicker lines here
    
    )

    # Compose text report
    report_text = f"""
{head_description}

{neck_flexion}

{shoulder_description}

{spine_status}

{pelvis_status}
"""

    return annotated_image, report_text



def main():
    # Replace this with the path to your image file
    image_path = "../test_images/andrea_front.jpg"  # Example: "./images/person1.jpg"

    if not os.path.isfile(image_path):
        print(f"Error: File not found: {image_path}")
        return

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image.")
        return

    # Initialize MediaPipe pose detector
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        if result.pose_landmarks:
            annotated_image, report = analyze_posture_and_annotate(image, result.pose_landmarks)

            print("Posture Report:")
            print(report)

            # Display Block
             # Get screen resolution (fallback if not available)
            screen_res = (1280, 720) 
            img_h, img_w = annotated_image.shape[:2]
            scale_w = screen_res[0] / img_w
            scale_h = screen_res[1] / img_h
            scale = min(scale_w, scale_h, 1.0)  # Avoid enlarging if smaller

            new_w = int(img_w * scale)
            new_h = int(img_h * scale)

            resized_image = cv2.resize(annotated_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Show resized image
            cv2.imshow("Annotated Posture", resized_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No pose landmarks detected.")

if __name__ == "__main__":
    main()