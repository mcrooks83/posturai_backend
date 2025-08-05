"""Picture annotation."""

from abc import ABC, abstractmethod
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def draw_center_axis(self, image):
    """Draws the vertical center axis line + label."""
    h = image.shape[0]
    cv2.line(
        image,
        (self.axis_finder.mid_base[0], 0),
        (self.axis_finder.mid_base[0], h),
        (0, 0, 0),
        thickness=5,
        lineType=cv2.LINE_AA
    )
    cv2.putText(
        image,
        "Center Axis",
        (self.axis_finder.mid_base[0] + 10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

def draw_pose_landmarks(self, image, landmarks):
    """Draws pose landmarks using mediapipe."""
    mp_drawing.draw_landmarks(
        image, landmarks, mp_pose.POSE_CONNECTIONS,
        connection_drawing_spec=self.drawing_spec
    )