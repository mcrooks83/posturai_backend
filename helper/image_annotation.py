"""Picture annotation."""

from abc import ABC, abstractmethod
import cv2
import mediapipe as mp

from helper.axis_finder import AxisFinder

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def draw_center_axis(image, axis_finder: AxisFinder):
    """Draws the vertical center axis line + label."""
    h = image.shape[0]
    cv2.line(
        image,
        (axis_finder.mid_base[0], 0),
        (axis_finder.mid_base[0], h),
        (0, 0, 0),
        thickness=5,
        lineType=cv2.LINE_AA
    )
    cv2.putText(
        image,
        "Center Axis",
        (axis_finder.mid_base[0] + 10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

def draw_pose_landmarks(image, landmarks):
    """Draws pose landmarks using mediapipe."""
    mp_drawing.draw_landmarks(
        image, landmarks, mp_pose.POSE_CONNECTIONS,
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(237, 114, 90), thickness=5)
    )
