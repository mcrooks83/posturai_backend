from vector_utils import find_vector_between, find_midpoint


class AxisFinder:

    def __init__(self, image, lm):
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
