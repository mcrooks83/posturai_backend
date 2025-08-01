"""Controller class."""

from abstract_posture_recogniser import AbstractPostureRecogniser
from rotated_posture_recongniser import PostureRecogniser
from fascia_lines.front_functional_line import FrontFunctionalLine
from fascia_lines.back_functional_line import BackFunctionalLine
from fascia_lines.spiral_line import SpiralLine


class PostureController:
    def __init__(self):
        self.strategies = {
            #'default': -----
            'rotation': PostureRecogniser,
            'front_fascia': FrontFunctionalLine,
            'back_fascia': BackFunctionalLine,
            'spiral_fascia': SpiralLine
        }
        self.current_strategy = None

    def set_strategy(self, strategy_name, *args, **kwargs):
        if strategy_name in self.strategies:
            self.current_strategy = self.strategies[strategy_name](*args, **kwargs)
        else:
            raise ValueError(f"No such strategy: {strategy_name}")

    def analyze(self):
        if not self.current_strategy:
            raise RuntimeError("No strategy set")
        return self.current_strategy.analyze_posture()

    def annotate(self, landmarks):
        if not self.current_strategy:
            raise RuntimeError("No strategy set")
        return self.current_strategy.annotate(landmarks)
