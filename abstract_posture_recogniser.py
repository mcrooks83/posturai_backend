"""Abstract base class."""

from abc import ABC, abstractmethod

class AbstractPostureRecogniser(ABC):

    @abstractmethod
    def analyze_posture(self):
        pass

    @abstractmethod
    def annotate(self, landmarks):
        pass
