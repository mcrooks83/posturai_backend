"""Landmark holder."""

def _get_xy(landmark_id, landmarks, w, h):
    """Helper to get a landmark position."""
    lm = landmarks[landmark_id]
    return int(lm.x * w), int(lm.y * h)


class Landmark:
    """Stores mediapipe landmarks and their positions."""

    def __init__(self, pose, w, h):
        self.w = w
        self.h = h
        self.pose_landmark = pose
        self.lm = pose.landmark

        LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
        LEFT_HIP, RIGHT_HIP = 23, 24
        LEFT_KNEE, RIGHT_KNEE = 25, 26
        LEFT_ANKLE, RIGHT_ANKLE = 27, 28
        LEFT_EYE_INNER, RIGHT_EYE_INNER = 1, 4
        NOSE = 0
        LEFT_TOE, RIGHT_TOE = 31, 32
        LEFT_EAR, RIGHT_EAR = 7, 8

        self.ls = _get_xy(LEFT_SHOULDER, self.lm, w, h)
        self.rs = _get_xy(RIGHT_SHOULDER, self.lm, w, h)
        self.lh = _get_xy(LEFT_HIP, self.lm, w, h)
        self.rh = _get_xy(RIGHT_HIP, self.lm, w, h)
        self.lk = _get_xy(LEFT_KNEE, self.lm, w, h)
        self.rk = _get_xy(RIGHT_KNEE, self.lm, w, h)
        self.la = _get_xy(LEFT_ANKLE, self.lm, w, h)
        self.ra = _get_xy(RIGHT_ANKLE, self.lm, w, h)
        self.leye = _get_xy(LEFT_EYE_INNER, self.lm, w, h)
        self.reye = _get_xy(RIGHT_EYE_INNER, self.lm, w, h)
        self.nose = _get_xy(NOSE, self.lm, w, h)
        self.ltoe = _get_xy(LEFT_TOE, self.lm, w, h)
        self.rtoe = _get_xy(RIGHT_TOE, self.lm, w, h)
        self.le = _get_xy(LEFT_EAR, self.lm, w, h)
        self.re = _get_xy(RIGHT_EAR, self.lm, w, h)

    def get_face_side_centers(self):
        def average_coords(indices):
            xs, ys = [], []
            for i in indices:
                p = self.lm[i]
                xs.append(int(p.x * self.w))
                ys.append(int(p.y * self.h))
            return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

        left_indices = [1, 2, 3, 7, 9]
        right_indices = [4, 5, 6, 8, 10]

        return average_coords(left_indices), average_coords(right_indices)
