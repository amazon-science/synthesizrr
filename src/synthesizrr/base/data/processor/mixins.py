from typing import *
from abc import ABC
from synthesizrr.base.data.processor import DataProcessor
from synthesizrr.base.constants import MLType, MissingColumnBehavior


class NumericInputProcessor(DataProcessor, ABC):
    """Mixin for numeric input data processors."""
    input_mltypes = [MLType.INT, MLType.FLOAT]


class CategoricalInputProcessor(DataProcessor, ABC):
    """Mixin for categorical input data processors."""
    input_mltypes = [MLType.INT, MLType.CATEGORICAL]


class CategoricalOutputProcessor(DataProcessor, ABC):
    """Mixin for categorical output data processors."""
    output_mltype = MLType.CATEGORICAL


class IntegerOutputProcessor(DataProcessor, ABC):
    """Mixin for integer output data processors."""
    output_mltype = MLType.INT


class DecimalOutputProcessor(DataProcessor, ABC):
    """Mixin for decimal output data processors."""
    output_mltype = MLType.FLOAT


class EncodedLabelOutputProcessor(DataProcessor, ABC):
    """Mixin for label output data processors."""
    output_mltype = MLType.ENCODED_LABEL


class TextInputProcessor(DataProcessor, ABC):
    """Mixin for text input data processors."""
    input_mltypes = [
        MLType.TEXT,
        MLType.CATEGORICAL,
        MLType.INT,
        MLType.FLOAT,
        MLType.BOOL,
    ]


class VectorAssemblerInputProcessor(DataProcessor, ABC):
    """Mixin for vectorAssembler input data processors."""
    input_mltypes = [
        MLType.INT,
        MLType.FLOAT,
        MLType.VECTOR,
        MLType.SPARSE_VECTOR
    ]


class LabelInputProcessor(DataProcessor, ABC):
    """Mixin for label input data processors."""
    missing_column_behavior = MissingColumnBehavior.SKIP

    input_mltypes = [
        MLType.GROUND_TRUTH_LABEL,
        MLType.ENCODED_LABEL,
        MLType.PREDICTED_LABEL,
        MLType.ENCODED_PREDICTED_LABEL,
    ]


class TextOrLabelInputProcessor(DataProcessor, ABC):
    """Mixin for text or label input data processors."""
    missing_column_behavior = MissingColumnBehavior.SKIP
    input_mltypes = LabelInputProcessor.input_mltypes + TextInputProcessor.input_mltypes


class TextOutputProcessor(DataProcessor, ABC):
    """Mixin for text output data processors."""
    output_mltype = MLType.TEXT


class BoolOutputProcessor(DataProcessor, ABC):
    """Mixin for bool output data processors."""
    output_mltype = MLType.BOOL


class VectorInputProcessor(DataProcessor, ABC):
    """Mixin for vector input data processors."""
    input_mltypes = [MLType.VECTOR]


class VectorOutputProcessor(DataProcessor, ABC):
    """Mixin for vector output data processors."""
    output_mltype = MLType.VECTOR


class SparseVectorInputProcessor(DataProcessor, ABC):
    """Mixin for sparse vector input data processors."""
    input_mltypes = [MLType.SPARSE_VECTOR]


class SparseVectorOutputProcessor(DataProcessor, ABC):
    """Mixin for sparse vector output data processors."""
    output_mltype = MLType.SPARSE_VECTOR


class NonVectorInputProcessor(DataProcessor, ABC):
    """Mixin for non-vector input data processors."""
    input_mltypes = list(set(MLType).difference({MLType.VECTOR, MLType.SPARSE_VECTOR}))


class PredictionsInputProcessor(DataProcessor, ABC):
    """Mixin for algorithm predictions data input data processors."""
    input_mltypes = [
        MLType.INDEX,
        MLType.GROUND_TRUTH_LABEL,
        MLType.ENCODED_LABEL,
        MLType.PROBABILITY_SCORE,
        MLType.PROBABILITY_SCORE_COMMA_SEPERATED_OR_LIST,
        MLType.PREDICTED_LABEL,
        MLType.PREDICTED_LABEL_COMMA_SEPARATED_OR_LIST,
        MLType.ENCODED_PREDICTED_LABEL,
        MLType.PREDICTED_CORRECT,
    ]
