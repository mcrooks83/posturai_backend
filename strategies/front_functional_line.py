"""Measure front functional lines."""
import cv2

from helper import vector_utils
from abstract_posture_recogniser import AbstractPostureRecogniser
from helper.axis_finder import AxisFinder
from helper.landmark import Landmark
from helper.vector_utils import find_vector_between, find_angles_between, describe_tilt_two_vectors


class FrontFunctionalLine(AbstractPostureRecogniser):

    def __init__(self, image, axis_finder: AxisFinder, landmark: Landmark):
        super().__init__(image)
        self.image = image
        self.axis_finder = axis_finder
        self.landmark = landmark
        self.analysis_result = self.analyze_posture()
        self.annotated_image = self.annotate()

    # 12-23, 23-25
    # 11-24, 24-26
    def analyze_posture(self):
        # left line
        left_shoulder_line = [self.axis_finder.left_right_torso_diagonal,
                              self.axis_finder.left_thigh]
        left_length = sum(vector_utils.find_vector_length(x) for x in left_shoulder_line)

        # right_line
        right_shoulder_line = [self.axis_finder.right_left_torso_diagonal,
                               self.axis_finder.right_thigh]
        right_length = sum(vector_utils.find_vector_length(x) for x in right_shoulder_line)

        print("left shoulder line:" + str(left_length))
        print("right shoulder line:" + str(right_length))

        return vector_utils.describe_front_functional_line_meaning(left_length, right_length)

    def annotate(self):
        annotated = self.image.copy()

        left_colour = (0, 255, 0)
        right_colour = (0, 0, 255)

        # left
        cv2.line(annotated, self.landmark.ls, self.landmark.rh, left_colour, 4)
        cv2.line(annotated, self.landmark.rk, self.landmark.rh, left_colour, 4)
        # right
        cv2.line(annotated, self.landmark.rs, self.landmark.lh, right_colour, 4)
        cv2.line(annotated, self.landmark.lk, self.landmark.lh, right_colour, 4)

        return annotated

    def get_result(self):
        return self.analysis_result
