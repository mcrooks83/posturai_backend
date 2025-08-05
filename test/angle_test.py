import unittest

from helper.vector_utils import find_angles_between
import os
import cv2


class TestAngles(unittest.TestCase):

    def setUp(self):
        image_path = "../test_images/andrea_front.jpg"
        assert os.path.isfile(image_path), f"File not found: {image_path}"

        image = cv2.imread(image_path)
        assert image is not None, "Failed to load image"

        self.recogniser = posture_recongniser.PostureRecogniser(image)

    def test_angle_with_certain_vectors(self):
        """Calculated manually."""
        v1 = [-50, -10]
        v2 = [-10, 80]
        angle = find_angles_between(v1, v2)
        self.assertAlmostEqual(angle[0], -94, delta=1)

    def test_shoulder_angle(self):
        """Angle cannot be too big."""
        sh = self.recogniser.shoulder_vector
        axis = self.recogniser.upright_vector
        angle = find_angles_between(sh, axis)
        print(angle)
        self.assertLess(abs(angle[0]), 100)

    def test_upright_vector_is_orthogonal(self):
        """90 degrees on both sides."""
        angle = find_angles_between(self.recogniser.base_vector, self.recogniser.upright_vector)
        self.assertEqual(angle[0], 90)
        self.assertEqual(angle[1], 90)

    def test_marcus_image_have_same_results(self):
        """All three marcus images are similar."""
        # 86, 90, 88
        m1_image_path = "../test_images/marcus1.jpg"
        assert os.path.isfile(m1_image_path), f"File not found: {m1_image_path}"
        m1_image = cv2.imread(m1_image_path)
        assert m1_image is not None, "Failed to load image"
        m1_recogniser = posture_recongniser.PostureRecogniser(m1_image)

        head_angle1 = find_angles_between(m1_recogniser.upright_vector, m1_recogniser.head_vector)
        sh_angle1 = find_angles_between(m1_recogniser.upright_vector, m1_recogniser.shoulder_vector)
        pelvis_angle1 = find_angles_between(m1_recogniser.upright_vector, m1_recogniser.pelvis_vector)

        m2_image_path = "../test_images/marcus2.jpg"
        assert os.path.isfile(m2_image_path), f"File not found: {m2_image_path}"
        m2_image = cv2.imread(m2_image_path)
        assert m2_image is not None, "Failed to load image"
        m2_recogniser = posture_recongniser.PostureRecogniser(m2_image)

        head_angle2 = find_angles_between(m2_recogniser.upright_vector, m2_recogniser.head_vector)
        sh_angle2 = find_angles_between(m2_recogniser.upright_vector, m2_recogniser.shoulder_vector)
        pelvis_angle2 = find_angles_between(m2_recogniser.upright_vector, m2_recogniser.pelvis_vector)

        m3_image_path = "../test_images/marcus3.jpg"
        assert os.path.isfile(m3_image_path), f"File not found: {m3_image_path}"
        m3_image = cv2.imread(m3_image_path)
        assert m3_image is not None, "Failed to load image"
        m3_recogniser = posture_recongniser.PostureRecogniser(m3_image)

        head_angle3 = find_angles_between(m3_recogniser.upright_vector, m3_recogniser.head_vector)
        sh_angle3 = find_angles_between(m3_recogniser.upright_vector, m3_recogniser.shoulder_vector)
        pelvis_angle3 = find_angles_between(m3_recogniser.upright_vector, m3_recogniser.pelvis_vector)

        head_angles = [head_angle1, head_angle2, head_angle3]
        sh_angles = [sh_angle1, sh_angle2, sh_angle3]
        pelvis_angles = [pelvis_angle1, pelvis_angle2, pelvis_angle3]

        print("marcus angles: " + str(head_angles) + str(sh_angles) + str(pelvis_angles))

        for i in range(len(head_angles)):
            for j in range(i + 1, len(head_angles)):
                #self.assertAlmostEqual(head_angles[i][0], head_angles[j][0], delta=1)
                self.assertAlmostEqual(sh_angles[i][0], sh_angles[j][0], delta=1)
                self.assertAlmostEqual(pelvis_angles[i][0], pelvis_angles[j][0], delta=1)

    def test_base_vector_is_horizontal(self):
        """Base vector must be horizontal after rotation."""
        base_vec = self.recogniser.base_vector
        x_axis = (1, 0)
        angle = find_angles_between(base_vec, x_axis)
        self.assertTrue(
            any(abs(a) < 1 or abs(abs(a) - 180) < 1 for a in angle),
            msg=f"Base vector not horizontal enough: {angle}"
        )

if __name__ == '__main__':
    unittest.main()
