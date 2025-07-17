import numpy as np
import matplotlib.pyplot as plt

# Correct MediaPipe-style neutral pose (left side is higher x)
neutral_pose = {
    "nose": [150, 50],
    "left_shoulder": [200, 100],   # person's left = higher x
    "right_shoulder": [100, 100],
    "left_hip": [190, 200],
    "right_hip": [110, 200],
    "left_knee": [185, 280],
    "right_knee": [115, 280],
    "left_ankle": [180, 350],
    "right_ankle": [120, 350],
}

def rotate_point(point, center, angle_deg):
    angle_rad = np.radians(angle_deg) # convet rotation angle to radians (from degrees)

    # This matrix rotates points in 2D space by angle θ counterclockwise.
    # When multiplied by a coordinate vector, it rotates the vector around the origin.
    R = np.array([  # roation matrix
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    #The @ operator means matrix multiplication in NumPy.
    # moves the point to the origin, rotates it and then puts it back
    return list((R @ (np.array(point) - center)) + center)

def simulate_posture_variation(pose, pelvis_rotation=0, spine_flex=0, shoulder_tilt=0):
    pose = {k: list(v) for k, v in pose.items()}  # copy with lists

    spine_center = np.mean([pose["left_hip"], pose["right_hip"]], axis=0)
    shoulder_center = np.mean([pose["left_shoulder"], pose["right_shoulder"]], axis=0)

    # Rotate hips and legs
    for joint in ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]:
        pose[joint] = rotate_point(pose[joint], spine_center, pelvis_rotation)

    # Rotate shoulders + nose
    for joint in ["left_shoulder", "right_shoulder", "nose"]:
        pose[joint] = rotate_point(pose[joint], shoulder_center, shoulder_tilt)

    # Spine flexion (vertical shift)
    for joint in ["left_shoulder", "right_shoulder", "nose"]:
        pose[joint][1] += spine_flex

    return pose

def draw_skeleton(ax, pose, title="Pose"):
    connections = [
        ("nose", "left_shoulder"),
        ("nose", "right_shoulder"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"),
        ("right_knee", "right_ankle"),
    ]

    for pt1, pt2 in connections:
        x = [pose[pt1][0], pose[pt2][0]]
        y = [pose[pt1][1], pose[pt2][1]]
        ax.plot(x, y, 'k-', lw=2)

    for joint, (x, y) in pose.items():
        ax.plot(x, y, 'bo')
        ax.text(x + 3, y, joint, fontsize=8)

    # Vertical center axis spanning full plot height
    left_ankle = pose["left_ankle"]
    right_ankle = pose["right_ankle"]
    center_x = (left_ankle[0] + right_ankle[0]) / 2

    ylim = ax.get_ylim()
    ax.plot([center_x, center_x], ylim, 'r--', lw=1.5)

    ax.set_title(title)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # image y-axis convention
    # no legend

# Create synthetic pose variation
synthetic_pose = simulate_posture_variation(
    neutral_pose,
    pelvis_rotation=10,
    spine_flex=0,
    shoulder_tilt=0
)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
draw_skeleton(axes[0], neutral_pose, "Neutral Pose (MediaPipe style)")
draw_skeleton(axes[1], synthetic_pose, "Synthetic Variation")
plt.tight_layout()
plt.show()
