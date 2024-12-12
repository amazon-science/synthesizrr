from typing import *
from synthergent.base.util import AutoEnum, auto


class ThresholdStrategy(AutoEnum):
    POSITIVE_CLASS_THRESHOLD = auto()
    GLOBAL = auto()
    TOP_PREDICTED_LABEL = auto()
    PER_LABEL = auto()


class LabelSelectionStrategy(AutoEnum):
    MAX_SCORE = auto()
    MAX_SCORE_ABOVE_THRESHOLD = auto()
