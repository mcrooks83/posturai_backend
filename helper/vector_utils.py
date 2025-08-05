"""Vector helper functions."""

import numpy as np


def find_midpoint(p1, p2):
    return (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2

def find_vector_between(start, end):
    return end[0] - start[0], end[1] - start[1]

def perpendicular_vector(vector):
    """A perpendicular from a vector (base)"""
    perp = np.array([-vector[1], vector[0]])
    if perp[1] > 0:
        perp = -perp
        print("perp vector: " + str(perp))
    return tuple(perp)

def intersect_vectors(origin1, dir1, origin2, dir2):
    """Find intersection of two vectors."""
    x1, y1 = origin1
    dx1, dy1 = dir1
    x2, y2 = origin2
    dx2, dy2 = dir2

    denom = dx1 * dy2 - dy1 * dx2
    if denom == 0:
        return None

    t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denom
    intersect_x = x1 + t * dx1
    intersect_y = y1 + t * dy1

    return intersect_x, intersect_y

def find_angles_between(v1, v2):
    """Angle in degrees between two vectors."""
    v1 = np.array(v1, dtype=np.float32)
    v2 = np.array(v2, dtype=np.float32)
    rad_angle = np.arctan2(
        v1[0]*v2[1] - v1[1]*v2[0],
        v1[0]*v2[0] + v1[1]*v2[1]
    )

    angle_deg = np.degrees(rad_angle)
    other_angle = 180 - angle_deg
    return angle_deg, other_angle

def get_angle_by_points(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx))

def four_angles_between(v1, v2):
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    angles = []
    angles.append(find_angles_between(v1, v2))
    angles.append(find_angles_between(v2, -v1))
    angles.append(find_angles_between(-v1, -v2))
    angles.append(find_angles_between(-v2, v1))

    return angles

def describe_tilt_two_vectors(line, axis, label):
    """Print the results."""
    a, b = find_angles_between(line, axis)
    if abs(a - b) < 1:
        if label == "shoulders":
            return f"Your {label}s are pretty balanced"
        else:
            return f"Your {label} is pretty balanced"
    if b == max(a, b):
        if label == "shoulders":
            return f"Your right shoulder is elevated"
        else:
            return f"Your {label} is elevated on the right side"
    else:
        if label == "shoulders":
            return f"Your left shoulder is elevated"
        else:
            return f"Your {label} is elevated on the left side"

def describe_tilt_with_angle(label, angle):
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
            description = f"Your {label} are elevated on the {side}"
        else:
            description = f"Your {label} is elevated on the {side}"
        return description

def find_vector_length(vector):
    return np.linalg.norm(vector)

def describe_front_functional_line_meaning(left, right):
    if abs(left - right) <= 1:
        return "Your core and thigh engagement appears symmetrical across both sides."
    elif left > right:
        return "The left front functional line is longer, which may suggest more tension on your left side."
    else:
        return "The right front functional line is longer, which may suggest more tension on your right side."

def describe_spiral_line_meaning(left, right):
    if abs(left - right) <= 1:
        return "No significant rotational pattern detected in your posture."
    elif left > right:
        return "Your left spiral line is longer, which may suggest a postural shift toward the right."
    else:
        return "Your right spiral line is longer, which may suggest a postural shift toward the left."

def knee_deviation(hip, knee, ankle):
    hip_x, _ = hip
    knee_x, _ = knee
    ankle_x, _ = ankle
    mid_x = (hip_x + ankle_x) / 2
    deviation = knee_x - mid_x
    return deviation
