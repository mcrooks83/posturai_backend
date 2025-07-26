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

def left_to_right_vector(left_point, right_point):
    return right_point[0] - left_point[0], right_point[1] - left_point[1]

def perpendicular_vector(vector):
    """Return a perpendicular."""
    perp = np.array([-vector[1], vector[0]])
    print(perp)
    if perp[1] > 0:
        perp = -perp
    return perp

def intersect_vectors(origin1, dir1, origin2, dir2):
    x1, y1 = origin1
    dx1, dy1 = dir1
    x2, y2 = origin2
    dx2, dy2 = dir2

    denom = dx1 * dy2 - dy1 * dx2
    if denom == 0:
        return None  # no intersection

    t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denom
    intersect_x = x1 + t * dx1
    intersect_y = y1 + t * dy1

    return intersect_x, intersect_y


def angles_between_axis_and_line(line, axis):
    """A tuple of joint angles."""
    line = np.array(line, dtype=float)
    axis = np.array(axis, dtype=float)

    line[1] = -line[1]
    axis[1] = -axis[1]

    dot_product = np.dot(line, axis)
    len_line = np.linalg.norm(line)
    len_axis = np.linalg.norm(axis)

    cos_angle = dot_product / (len_line * len_axis)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_1 = np.degrees(np.arccos(cos_angle))
    angle_2 = 180 - angle_1

    scalar = line[0] * axis[1] - line[1] * axis[0]

    if scalar > 0:
        left_angle = angle_1
        right_angle = angle_2
    else:
        left_angle = angle_2
        right_angle = angle_1
    return  left_angle, right_angle

def signed_angle_between(v1, v2):
    v1 = np.array(v1, dtype=np.float32)
    v2 = np.array(v2, dtype=np.float32)
    angle = np.arctan2(
        v1[0]*v2[1] - v1[1]*v2[0],  # cross product
        v1[0]*v2[0] + v1[1]*v2[1]   # dot product
    )
    return np.degrees(angle)

def describe_tilt(line, axis, label):
    a, b = angles_between_axis_and_line(line, axis)
    if abs(a - b) < 0.5:
        if label == "shoulders":
            return f"Your {label}s are pretty balanced"
        else:
            return f"Your {label} is pretty balanced"
    if a == max(a, b):
        if label == "shoulders":
            return f"Your right shoulder is elevated"
        else:
            return f"Your {label} is elevated on the right side"
    else:
        if label == "shoulders":
            return f"Your left shoulder is elevated"
        else:
            return f"Your {label} is elevated on the left side"


def analyze_posture_and_annotate(image, landmarks):
    h, w, _ = image.shape
    lm = landmarks.landmark

    # landmark IDs
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_KNEE, RIGHT_KNEE = 25, 26
    LEFT_ANKLE, RIGHT_ANKLE = 27, 28
    LEFT_EYE_INNER, RIGHT_EYE_INNER = 1, 4
    NOSE = 0
    LEFT_TOE, RIGHT_TOE = 31, 32

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

    # mid points
    mid_shoulder = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
    mid_hip = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
    mid_eyes = ((leye[0] + reye[0]) // 2, (leye[1] + reye[1]) // 2)

    # compute a base mid point (avg of toe index and ankle)
    mid_ankle = ((la[0] + ra[0]) // 2, (la[1] + ra[1]) // 2)
    mid_toe = ((ltoe[0] + rtoe[0]) // 2, (ltoe[1] + rtoe[1]) // 2)
    mid_base = ((mid_ankle[0] + mid_toe[0]) // 2, (mid_ankle[1] + mid_toe[1]) // 2)

    # Left foot center
    left_foot = (
        (la[0] + ltoe[0]) // 2,
        (la[1] + ltoe[1]) // 2
    )
    # Right foot center
    right_foot = (
        (ra[0] + rtoe[0]) // 2,
        (ra[1] + rtoe[1]) // 2
    )

    # body-relative axis
    spine_axis = (
        mid_shoulder[0] - mid_hip[0],
        mid_shoulder[1] - mid_hip[1]
    )

    neck_axis = (
        mid_eyes[0] - mid_shoulder[0],
        mid_eyes[1] - mid_shoulder[0]
    )

    leg_axis = (
        mid_hip[0] - mid_base[0],
        mid_hip[1] - mid_shoulder[1]
    )

    # vectors for body symmetry
    shoulder_vector = left_to_right_vector(ls, rs)
    pelvis_vector = left_to_right_vector(lh, rh)
    head_vector = left_to_right_vector(leye, reye)
    base_vector = left_to_right_vector(left_foot, right_foot)

    # ------ Perpendicular --------

    foot_perp = perpendicular_vector(base_vector)
    scale = (leye[1] - mid_base[1]) / foot_perp[1]
    if foot_perp[1] == 0:
        scale = 1
    perp_scaled = foot_perp * scale
    upright_vector = mid_base + perp_scaled

    # ------ Intersections --------
    head_intersection = intersect_vectors(leye, head_vector, mid_base, upright_vector)
    print(head_intersection)
    shoulder_intersection = intersect_vectors(ls, shoulder_vector, mid_base, upright_vector)
    print(shoulder_intersection)

    # -----------------------------

    print(signed_angle_between(upright_vector, head_vector))
    print(signed_angle_between(upright_vector, shoulder_vector))
    print(signed_angle_between(upright_vector, pelvis_vector))

    annotated_image = image.copy()

    # -----------------------------
    cv2.line(
        annotated_image,
        mid_base,
        mid_hip,
        (0, 255, 0), 4, cv2.LINE_AA
    )
    cv2.line(
        annotated_image,
        mid_hip,
        mid_shoulder,
        (0, 255, 0), 4, cv2.LINE_AA
    )
    cv2.line(
        annotated_image,
        mid_shoulder,
        nose,
        (0, 255, 0), 4, cv2.LINE_AA
    )
    cv2.line(
        annotated_image,
        left_foot,
        right_foot,
        (0, 255, 255), 4, cv2.LINE_AA
    )

    cv2.line(
        annotated_image,
        (int(mid_base[0]), int(mid_base[1])), (int(upright_vector[0]), int(upright_vector[1])),
        (255, 0, 255), 4, cv2.LINE_AA
    )

    mp_drawing.draw_landmarks(
        annotated_image, landmarks, mp_pose.POSE_CONNECTIONS,
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(237, 114, 90), thickness=5)
    )

    cv2.putText(annotated_image, f"left", (ls[0] - 30, ls[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 5)
    cv2.putText(annotated_image, f"right", (rs[0] + 10, rs[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 5)

    def draw_vector(img, origin, vector, color, length=100):
        end = (int(origin[0] + vector[0] / int(np.linalg.norm(vector)) * length),
               int(origin[1] + vector[1] / int(np.linalg.norm(vector)) * length))
        cv2.line(img, origin, end, color, 2)

    return annotated_image

def main():
    # Replace this with the path to your image file
    image_path = "../test_images/orange_relaxed.jpg"  # Example: "./images/person1.jpg"

    if not os.path.isfile(image_path):
        print(f"Error: File not found: {image_path}")
        return

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image.")
        return

    with mp_pose.Pose(static_image_mode=True) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        if result.pose_landmarks:
            annotated_image = analyze_posture_and_annotate(image, result.pose_landmarks)

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
