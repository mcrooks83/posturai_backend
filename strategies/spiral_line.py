"""Measure spiral line."""
import cv2

from helper import vector_utils
from abstract_posture_recogniser import AbstractPostureRecogniser
from helper.axis_finder import AxisFinder
from helper.landmark import Landmark


class SpiralLine(AbstractPostureRecogniser):

    def __init__(self, image, axis_finder: AxisFinder, landmark: Landmark):
        super().__init__(image)
        self.image = image
        self.axis_finder = axis_finder
        self.landmark = landmark
        self.analysis_result = self.analyze_posture()
        self.annotated_image = self.annotate()

    def analyze_posture(self):
        # left line
        left_ear_line = [self.axis_finder.right_shoulder_left_ear,
                         self.axis_finder.right_left_torso_diagonal,
                         self.axis_finder.left_thigh,
                         self.axis_finder.left_calf]
        left_length = sum(vector_utils.find_vector_length(x) for x in left_ear_line)

        # right_line
        right_ear_line = [self.axis_finder.left_shoulder_right_ear,
                         self.axis_finder.left_right_torso_diagonal,
                         self.axis_finder.right_thigh,
                         self.axis_finder.right_calf]
        right_length = sum(vector_utils.find_vector_length(x) for x in right_ear_line)

        print("left ear line:" + str(left_length))
        print("right ear line:" + str(right_length))

        return vector_utils.describe_spiral_line_meaning(left_length, right_length)

    def annotate(self):
        annotated = self.image.copy()

        left_colour = (0, 255, 255)
        right_colour = (255, 0, 255)

        # left
        cv2.line(annotated, self.landmark.rs, self.landmark.le, left_colour, 4)
        cv2.line(annotated, self.landmark.rs, self.landmark.lh, left_colour, 4)
        cv2.line(annotated, self.landmark.lk, self.landmark.lh, left_colour, 4)
        cv2.line(annotated, self.landmark.lk, self.landmark.la, left_colour, 4)
        # right
        cv2.line(annotated, self.landmark.ls, self.landmark.re, right_colour, 4)
        cv2.line(annotated, self.landmark.ls, self.landmark.rh, right_colour, 4)
        cv2.line(annotated, self.landmark.rk, self.landmark.rh, right_colour, 4)
        cv2.line(annotated, self.landmark.rk, self.landmark.ra, right_colour, 4)

        return annotated

    def get_result(self):
        return self.analysis_result
