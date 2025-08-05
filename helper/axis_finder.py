"""Axis finder."""

from helper.vector_utils import find_vector_between, find_midpoint, get_angle_by_points, knee_deviation


class AxisFinder:

    def __init__(self, image, lm):
        """Get midpoints, vertical axes and base vector."""
        h, w, _ = image.shape

        # Midpoints
        self.mid_shoulder = find_midpoint(lm.ls, lm.rs)
        self.mid_hip = find_midpoint(lm.lh, lm.rh)
        self.mid_eyes = find_midpoint(lm.leye, lm.reye)
        self.mid_ankle = find_midpoint(lm.la, lm.ra)
        self.mid_toe = find_midpoint(lm.ltoe, lm.rtoe)
        self.mid_base = find_midpoint(self.mid_ankle, self.mid_toe)

        # Foot centres
        self.left_foot = find_midpoint(lm.la, lm.ltoe)
        self.right_foot = find_midpoint(lm.ra, lm.rtoe)

        # Axes
        self.spine_axis = find_vector_between(self.mid_hip, self.mid_shoulder)
        self.neck_axis = find_vector_between(self.mid_shoulder, self.mid_eyes)
        self.leg_axis = find_vector_between(self.mid_base, self.mid_hip)

        self.base_vector = find_vector_between(self.left_foot, self.right_foot)

        # Diagonals
        self.left_right_torso_diagonal = find_vector_between(lm.ls, lm.rh)
        self.right_left_torso_diagonal = find_vector_between(lm.rs, lm.lh)

        # Fascia
        self.left_thigh = find_vector_between(lm.lk, lm.lh)
        self.right_thigh = find_vector_between(lm.rk, lm.rh)
        self.left_calf = find_vector_between(lm.lk, lm.la)
        self.right_calf = find_vector_between(lm.rk, lm.ra)
        self.left_shoulder_right_ear = find_vector_between(lm.ls, lm.re)
        self.right_shoulder_left_ear = find_vector_between(lm.rs, lm.le)

        # Central axis
        self.hip_center_x = (lm.lh[0] + lm.rh[0]) // 2
        self.hip_center_y = (lm.lh[1] + lm.rh[1]) // 2 + 150
        self.head_mid = lm.nose

        self.vec_neck = (self.head_mid[0] - self.mid_shoulder[0], self.head_mid[1] - self.mid_shoulder[1])
        self.vec_spine = (self.mid_hip[0] - self.mid_shoulder[0], self.mid_hip[1] - self.mid_shoulder[1])

        self.shoulder_angle = get_angle_by_points(lm.ls, lm.rs)
        self.hip_angle = get_angle_by_points(lm.lh, lm.rh)

        self.left_dev = knee_deviation(lm.lh, lm.lk, lm.la)
        self.right_dev = knee_deviation(lm.rh, lm.rk, lm.ra)

        self.left_hip_offset = lm.lh[0] - self.mid_base[0]
        self.right_hip_offset = lm.rh[0] - self.mid_base[0]

        self.left_shoulder_dist = abs(lm.ls[0] - self.mid_hip[0])
        self.right_shoulder_dist = abs(lm.rs[0] - self.mid_hip[0])
