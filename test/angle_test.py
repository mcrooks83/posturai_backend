import unittest

from vector_utils import find_angles_between
import os
import cv2
import posture_recongniser


class TestAngles(unittest.TestCase):

    def setUp(self):
        image_path = "../test_images/bad_posture.jpg"
        assert os.path.isfile(image_path), f"File not found: {image_path}"

        image = cv2.imread(image_path)
        assert image is not None, "Failed to load image"

        self.recogniser = posture_recongniser.PostureRecogniser(image)

    def test_angle_with_certain_vectors(self):
        v1 = [-50, -10]
        v2 = [-10, 80]
        angle = find_angles_between(v1, v2)
        self.assertAlmostEqual(angle, -94, delta=1)

    def test_shoulder_angle(self):
        sh = self.recogniser.shoulder_vector
        axis = self.recogniser.upright_vector
        angle = find_angles_between(sh, axis)
        print(angle)
        self.assertLess(abs(angle), 10)

    def test_vectors_drawn_correctly(self):
        print("upright vector:" + str(self.recogniser.upright_vector))
        print("shoulder vector:" + str(self.recogniser.shoulder_vector))

if __name__ == '__main__':
    unittest.main()
