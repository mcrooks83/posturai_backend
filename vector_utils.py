"""Vector helper functions."""

import numpy as np


def left_to_right_vector(left_point, right_point):
    """Calculate vector coordinates."""
    return right_point[0] - left_point[0], right_point[1] - left_point[1]

def find_midpoint(p1, p2):
    return (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2

def find_vector_between(start, end):
    return end[0] - start[0], end[1] - start[1]

def perpendicular_vector(vector):
    """A perpendicular from a vector (base)"""
    perp = np.array([-vector[1], vector[0]])
    if perp[1] > 0:
        perp = -perp
    return perp

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

def signed_angle_between(v1, v2):
    """Angle in degrees between two vectors."""
    v1 = np.array(v1, dtype=np.float32)
    v2 = np.array(v2, dtype=np.float32)
    rad_angle = np.arctan2(
        v1[0]*v2[1] - v1[1]*v2[0],
        v1[0]*v2[0] + v1[1]*v2[1]
    )

    angle_deg = np.degrees(rad_angle)

    if angle_deg > 90:
        angle_deg = 180 - angle_deg
    elif angle_deg < -90:
        angle_deg = -180 - angle_deg

    return angle_deg
