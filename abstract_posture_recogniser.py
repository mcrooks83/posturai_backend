"""Abstract base class."""

from abc import ABC, abstractmethod


class AbstractPostureRecogniser(ABC):

    def __init__(self, image, axis_finder=None):
        self.landmarks = None
        if image is None:
            print("Failed to load image.")
            return

        # 1 - processing original image
        self.image = image
        self.processed_landmark = self.find_base_landmarks(self.image)
        self.axis_finder = axis_finder

        self.vector_handler = None

        self.shoulder_vector = None
        self.pelvis_vector = None
        self.head_vector = None
        self.base_vector = None
        self.upright_vector = None

        self.annotated_image = None

    def inject_dependencies(self, image, landmarks, axis_finder):
        self.image = image
        self.landmarks = landmarks
        self.axis_finder = axis_finder

    @abstractmethod
    def analyze_posture(self):
        pass

    @abstractmethod
    def annotate(self, landmarks):
        pass
