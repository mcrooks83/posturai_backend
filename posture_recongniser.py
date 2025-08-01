"""Posture recogniser."""

import cv2
import mediapipe as mp
import numpy as np

from vector_utils import find_vector_between, find_angles_between, four_angles_between, describe_tilt
import vector_handler
import axis_finder


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def _get_xy(landmark_id, landmarks, w, h):
    """Helper to get a landmark position."""
    lm = landmarks[landmark_id]
    return int(lm.x * w), int(lm.y * h)


class Landmark:
    """Stores mediapipe landmarks and their positions."""

    def __init__(self, pose, w, h):
        self.pose_landmark = pose
        lm = self.pose_landmark.landmark

        LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
        LEFT_HIP, RIGHT_HIP = 23, 24
        LEFT_KNEE, RIGHT_KNEE = 25, 26
        LEFT_ANKLE, RIGHT_ANKLE = 27, 28
        LEFT_EYE_INNER, RIGHT_EYE_INNER = 1, 4
        NOSE = 0
        LEFT_TOE, RIGHT_TOE = 31, 32

        self.ls = _get_xy(LEFT_SHOULDER, lm, w, h)
        self.rs = _get_xy(RIGHT_SHOULDER, lm, w, h)
        self.lh = _get_xy(LEFT_HIP, lm, w, h)
        self.rh = _get_xy(RIGHT_HIP, lm, w, h)
        self.lk = _get_xy(LEFT_KNEE, lm, w, h)
        self.rk = _get_xy(RIGHT_KNEE, lm, w, h)
        self.la = _get_xy(LEFT_ANKLE, lm, w, h)
        self.ra = _get_xy(RIGHT_ANKLE, lm, w, h)
        self.leye = _get_xy(LEFT_EYE_INNER, lm, w, h)
        self.reye = _get_xy(RIGHT_EYE_INNER, lm, w, h)
        self.nose = _get_xy(NOSE, lm, w, h)
        self.ltoe = _get_xy(LEFT_TOE, lm, w, h)
        self.rtoe = _get_xy(RIGHT_TOE, lm, w, h)


class PostureRecogniser:

    def __init__(self, image):
        """Process given image to get landmarks.
        Then rotate image and process it again to analyse posture and visualise."""
        if image is None:
            print("Failed to load image.")
            return

        # 1 - processing original image
        self.image = image
        self.processed_landmark = self.find_base_landmarks(self.image)

        self.axis_finder = axis_finder.AxisFinder(self.image, self.processed_landmark)
        self.vector_handler = vector_handler.VectorHandler(self.processed_landmark, self.axis_finder)

        rotated = self.vector_handler.rotate_picture(self.image)

        # 2 - processing rotated image
        self.image = rotated
        self.processed_landmark = self.find_base_landmarks(self.image)

        self.axis_finder = axis_finder.AxisFinder(self.image, self.processed_landmark)
        self.vector_handler = vector_handler.VectorHandler(self.processed_landmark, self.axis_finder)

        # initialise
        self.shoulder_vector = None
        self.pelvis_vector = None
        self.head_vector = None
        self.base_vector = None
        self.upright_vector = None

        # analyse
        self.analyze_posture()
        self.annotated_image = self.annotate(self.processed_landmark.pose_landmark)

    def find_base_landmarks(self, image):
        """Get base landmarks for image rotation."""
        with mp_pose.Pose(static_image_mode=True) as pose:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)

            if result.pose_landmarks:
                h, w, _ = self.image.shape
                processed_landmark = Landmark(result.pose_landmarks, w, h)

                return processed_landmark
        return None

    def analyze_posture(self):
        """Print analysis results."""
        self.upright_vector = self.vector_handler.find_upright_vector()

        # vectors for body symmetry
        self.shoulder_vector = find_vector_between(self.processed_landmark.ls, self.processed_landmark.rs)
        self.pelvis_vector = find_vector_between(self.processed_landmark.lh, self.processed_landmark.rh)
        self.head_vector = find_vector_between(self.processed_landmark.leye, self.processed_landmark.reye)
        self.base_vector = find_vector_between(self.axis_finder.left_foot, self.axis_finder.right_foot)

        # ------ Intersections --------
        """
        head_intersection = intersect_vectors(lm.leye, self.head_vector, self.mid_base, upright_vector)
        print(head_intersection)
        shoulder_intersection = intersect_vectors(ls, shoulder_vector, mid_base, upright_vector)
        print(shoulder_intersection)
        """
        # -----------------------------

        print(find_angles_between(self.head_vector, self.upright_vector))
        print(find_angles_between(self.shoulder_vector, self.upright_vector))
        print(find_angles_between(self.pelvis_vector, self.upright_vector))

        print(describe_tilt(self.upright_vector, self.head_vector, "head"))
        print(describe_tilt(self.upright_vector, self.shoulder_vector, "shoulders"))
        print(describe_tilt(self.upright_vector, self.pelvis_vector, "pelvis"))

        print(find_angles_between(self.base_vector, self.upright_vector))

        #print(four_angles_between(self.upright_vector, self.shoulder_vector))
#        print(sum(four_angles_between(self.upright_vector, self.shoulder_vector)))

    def annotate(self, landmarks):
        """Visualise the vectors."""
        annotated_image = self.image.copy()

        cv2.line(
            annotated_image,
            self.axis_finder.mid_base,
            self.axis_finder.mid_hip,
            (0, 255, 0), 4, cv2.LINE_AA
        )
        cv2.line(
            annotated_image,
            self.axis_finder.mid_hip,
            self.axis_finder.mid_shoulder,
            (0, 255, 0), 4, cv2.LINE_AA
        )
        cv2.line(
            annotated_image,
            self.axis_finder.mid_shoulder,
            self.processed_landmark.nose,
            (0, 255, 0), 4, cv2.LINE_AA
        )
        cv2.line(
            annotated_image,
            self.axis_finder.left_foot,
            self.axis_finder.right_foot,
            (0, 255, 255), 4, cv2.LINE_AA
        )

        cv2.line(annotated_image, self.processed_landmark.leye, self.processed_landmark.reye, (0, 200, 50), 5, cv2.LINE_AA)

        start = np.array(self.axis_finder.mid_base)
        direction = np.array(self.upright_vector)
        end = start + direction

        cv2.line(
            annotated_image,
            tuple(start.astype(int)),
            tuple(end.astype(int)),
            (255, 0, 255), 8, cv2.LINE_AA
        )

        mp_drawing.draw_landmarks(
            annotated_image, landmarks, mp_pose.POSE_CONNECTIONS,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(237, 114, 90), thickness=5)
        )

        cv2.putText(annotated_image, f"left", (self.processed_landmark.ls[0] - 30, self.processed_landmark.ls[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 5)
        cv2.putText(annotated_image, f"right", (self.processed_landmark.rs[0] + 10, self. processed_landmark.rs[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 5)

        def draw_vector(img, origin, vector, color, length=100):
            end = (int(origin[0] + vector[0] / int(np.linalg.norm(vector)) * length),
                   int(origin[1] + vector[1] / int(np.linalg.norm(vector)) * length))
            cv2.line(img, origin, end, color, 2)

        return annotated_image
