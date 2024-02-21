from typing import *
from synthesizrr.base.util import AutoEnum, auto


class Task(AutoEnum):
    """
    A Task should only relate to the outputs, not the inputs!
    E.g. "Image classification" is not a valid task type, it should just be "classification".
    Within classification, output variation can be made, especially if the predictions and metrics are different.
    E.g. binary, multi-class and multi-label classification can all be considered different tasks since they have
    significantly different metrics.
    """

    ## Classification
    BINARY_CLASSIFICATION = auto()
    MULTI_CLASS_CLASSIFICATION = auto()
    MULTI_LABEL_CLASSIFICATION = auto()

    ## Regression
    REGRESSION = auto()

    ## Embedding
    EMBEDDING = auto()

    NER = auto()

    ## Ranking & Retrieval
    RETRIEVAL_CORPUS = auto()  ## For Datasets
    RANKING = auto()
    RETRIEVAL = auto()

    ## Prompting-based techniques
    NEXT_TOKEN_PREDICTION = auto()  ## Core task
    IN_CONTEXT_LEARNING = auto()  ## Derived task

    ## Audio & Speech
    TEXT_TO_SPEECH = auto()


TaskType = Task

TaskOrStr = Union[Task, str]


class MLType(AutoEnum):
    ## "Data" MLTypes:
    BOOL = auto()
    TEXT = auto()
    CATEGORICAL = auto()
    INT = auto()
    FLOAT = auto()
    VECTOR = auto()
    SPARSE_VECTOR = auto()
    TIMESTAMP = auto()
    TENSOR = auto()
    OBJECT = auto()

    ## "Asset" MLTypes:
    DOCUMENT = auto()  ## For .txt documents, PDFs, etc
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()

    ## Schema MLTypes:
    INDEX = auto()
    GROUND_TRUTH = auto()
    PREDICTED_LABEL = auto()
    PREDICTED_PROBABILITY = auto()
    PREDICTED = auto()

    ## Ground truth label(s):
    GROUND_TRUTH_LABEL = auto()  ## TODO: Delete this.
    GROUND_TRUTH_LABEL_LIST = auto()
    GROUND_TRUTH_LABEL_COMMA_SEPARATED = auto()
    GROUND_TRUTH_LABEL_COMMA_SEPARATED_OR_LIST = auto()
    ENCODED_LABEL = auto()
    ENCODED_LABEL_LIST = auto()
    ENCODED_LABEL_COMMA_SEPARATED = auto()
    ENCODED_LABEL_COMMA_SEPARATED_OR_LIST = auto()

    ## Predicted label(s):
    PREDICTED_LABEL_COMMA_SEPARATED_OR_LIST = auto()
    ENCODED_PREDICTED_LABEL = auto()

    ## Predicted probability score(s):
    PROBABILITY_SCORE = auto()
    PROBABILITY_SCORE_COMMA_SEPERATED_OR_LIST = auto()
    PREDICTED_CORRECT = auto()
    PREDICTION_IS_CONFIDENT = auto()
    ## Each element stores a list [predicted_label, predicted_score, is_confident]:
    PREDICTED_LABEL_PREDICTED_SCORE_IS_CONFIDENT_VECTOR = auto()


DATA_ML_TYPES: Set[MLType] = {
    MLType.BOOL,
    MLType.TEXT,
    MLType.CATEGORICAL,
    MLType.INT,
    MLType.FLOAT,
    MLType.VECTOR,
    MLType.SPARSE_VECTOR,
    MLType.TIMESTAMP,
    MLType.TENSOR,
}

ASSET_ML_TYPES: Set[MLType] = {
    MLType.DOCUMENT,
    MLType.IMAGE,
    MLType.AUDIO,
    MLType.VIDEO,
}

PREDICTED_ML_TYPES: Set[MLType] = {
    MLType.PREDICTED,
    MLType.PREDICTED_LABEL,
    MLType.PREDICTED_PROBABILITY,
}

GROUND_TRUTH_ML_TYPES: Set[MLType] = {
    MLType.GROUND_TRUTH,
    MLType.GROUND_TRUTH_LABEL,
}

MLTypeSchema = Dict[str, MLType]

MLTypeOrStr = Union[MLType, str]
