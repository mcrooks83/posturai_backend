"""Controller class."""
import os

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
        self.landmark = None
        self.axis_finder = None
        self.strategies = {}
        self.current_strategy = None

    def process_image(self, image_path):
        """Call this once with the image to set everything up."""
        if not os.path.isfile(image_path):
            print(f"Error: File not found: {image_path}")
            return

        self.image = cv2.imread(image_path)
        if self.image is None:
            print("Failed to load image.")
            return

        with mp_pose.Pose(static_image_mode=True) as pose:
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if not results.pose_landmarks:
                print("No pose landmarks detected.")
                return

            h, w = self.image.shape[:2]
            self.landmark = Landmark(results.pose_landmarks, w, h)

        self.axis_finder = AxisFinder(self.image, self.landmark)

        # Set up strategies
        self.strategies = {
            "spiral_line": SpiralLine(self.image, self.axis_finder, self.landmark),
            "front_line": FrontFunctionalLine(self.image, self.axis_finder, self.landmark),
            "central_axis": CentralAxis(self.image, self.axis_finder, self.landmark),
        }

        # Run analysis for all strategies
        for name, strategy in self.strategies.items():
            strategy.analyze_posture()
            print(f"{name}: {strategy.get_result()}")

        # Default strategy to display (optional)
        self.set_strategy("central_axis")
        annotated_image = self.annotate()

        # Resize image to fit screen
        screen_res = (1280, 720)
        img_h, img_w = annotated_image.shape[:2]
        scale_w = screen_res[0] / img_w
        scale_h = screen_res[1] / img_h
        scale = min(scale_w, scale_h, 1.0)  # Don’t upscale if image is smaller

        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        resized_image = cv2.resize(annotated_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Show image
        cv2.imshow("Posture Annotation", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def set_strategy(self, name):
        if name not in self.strategies:
            raise ValueError(f"Unknown strategy '{name}'")
        self.current_strategy = name

    def annotate(self):
        if self.current_strategy is None:
            raise RuntimeError("No strategy selected for annotation.")
        return self.strategies[self.current_strategy].annotate()


if __name__ == "__main__":
    image_path = "test_images/andrea_front.jpg"
    controller = PostureController()
    controller.process_image(image_path)

    controller.set_strategy("front_line")
    result = controller.annotate()
