"""Measure spiral line."""

from helper import vector_utils
from abstract_posture_recogniser import AbstractPostureRecogniser


class SpiralLine(AbstractPostureRecogniser):

    def __init__(self, image, axis_finder=None):
        super().__init__(image)
        self.image = image
        self.analysis_result = self.analyze_posture()
        self.annotated_image = self.annotate(image)

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
