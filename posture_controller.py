"""Controller class."""
import cv2
import mediapipe as mp

from helper import landmark
from helper.axis_finder import AxisFinder
from helper.landmark import Landmark
from rotated_posture_recongniser import PostureRecogniser
from strategies.central_axis import CentralAxis
from strategies.front_functional_line import FrontFunctionalLine
from strategies.spiral_line import SpiralLine


mp_pose = mp.solutions.pose

class PostureController:
    def __init__(self):
        self.image = None
        self.landmarks = None
        self.axis_finder = None
        self.strategies = {}
        self.current_strategy = None

    def process_image(self, image):
        """Call this once with the image to set everything up."""
        with mp_pose.Pose(static_image_mode=True) as pose:
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:
                raise ValueError("No landmarks detected.")

            h, w = image.shape[:2]
            self.landmarks = Landmark(results.pose_landmarks, w, h)

        self.axis_finder = AxisFinder(image, self.landmarks)

        self.strategies = {
            "spiral_line": SpiralLine(self.landmarks, self.axis_finder),
            "front_line": FrontFunctionalLine(self.landmarks, self.axis_finder),
            "central_axis": CentralAxis(self.landmarks, self.axis_finder),
        }

        for strategy in self.strategies.values():
            strategy.analyze_posture()

    def set_strategy(self, name):
        if name not in self.strategies:
            raise ValueError(f"Unknown strategy '{name}'")
        self.current_strategy = name

    def annotate(self):
        if self.current_strategy is None:
            raise RuntimeError("No strategy selected for annotation.")
        return self.strategies[self.current_strategy].annotate()

