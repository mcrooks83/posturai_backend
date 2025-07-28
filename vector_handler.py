import cv2
import numpy as np
import imutils
from vector_utils import perpendicular_vector


class VectorHandler:

    def __init__(self, lm, af):
        self.af = af
        self.lm = lm
        self.upright_vector = None

    def find_upright_vector(self):
        foot_perp = perpendicular_vector(self.af.base_vector)

        if foot_perp[1] == 0:
            scale = 1
        else:
            scale = (self.lm.leye[1] - self.af.mid_base[1]) / foot_perp[1]

        perp_scaled = foot_perp * scale
        self.upright_vector = np.array(self.af.mid_base) + perp_scaled
        return self.upright_vector

    def rotate_picture(self, image):
        """Rotate image visually."""
        # base vector tilt angle
        x1, y1 = self.af.left_foot
        x2, y2 = self.af.right_foot

        rotation_angle_rad = np.arctan2(y2 - y1, x2 - x1)
        rotation_angle = -np.degrees(rotation_angle_rad)
        rotation_angle = (rotation_angle + 180) % 360 - 180

        angle = rotation_angle
        if rotation_angle > 90:
            angle = 180 - rotation_angle
        elif rotation_angle < -90:
            angle = -180 - rotation_angle

        print(f"Rotating image by {angle} degrees")
        rotated = imutils.rotate_bound(image, angle)
        return rotated

