from typing import *
import numpy as np
import pandas as pd
from synthesizrr.base.data.processor import SingleColumnProcessor, TextOrLabelInputProcessor, EncodedLabelOutputProcessor
from synthesizrr.base.constants import MLType
from synthesizrr.base.util import AutoEnum, auto, is_null, type_str
from synthesizrr.base.data.sdf import ScalableSeries, ScalableSeriesRawType
from pydantic import root_validator


class EncodingRange(AutoEnum):
    ONE_TO_N = auto()
    ZERO_TO_N_MINUS_ONE = auto()
    BINARY_ZERO_ONE = auto()
    BINARY_PLUS_MINUS_ONE = auto()


ENCODING_RANGE_TO_UNKNOWN_LABELS_MAP = {
    EncodingRange.ONE_TO_N: 0,
    EncodingRange.BINARY_ZERO_ONE: -1,
    EncodingRange.BINARY_PLUS_MINUS_ONE: 0,
    EncodingRange.ZERO_TO_N_MINUS_ONE: -1,
}

BINARY_POSITIVE_LABELS: Set[str] = {'1', 'Y', 'YES', 'TRUE', 'T'}
BINARY_NEGATIVE_LABELS: Set[str] = {'0', '-1', 'N', 'NO', 'FALSE', 'F'}

LabelEncoding = "LabelEncoding"


class LabelEncoding(SingleColumnProcessor, TextOrLabelInputProcessor, EncodedLabelOutputProcessor):
    """
    Fits a list of categorical or integer values and transforms each into an integer value.
    Params:
    - ENCODING_RANGE: the output range of integer values, must be long to enum EncodingRange. Values:
        - ONE_TO_N: encodes to 1, 2, 3, ... N (number of unique labels)
        - ZERO_TO_N_MINUS_ONE: encodes to 0, 1, 2, ... N-1 (number of unique labels)
        - BINARY_ZERO_ONE: encodes to  0 or 1. Throws an exception if the labels are not binary.
        - BINARY_PLUS_MINUS_ONE: encodes to -1 or +1. Throws an exception if the labels are not binary.
    - MISSING_INPUT_FILL_VALUE: the value to fill for None/NaN labels.
    - UNKNOWN_INPUT_ENCODING_VALUE: the encoding value to fill for labels which are present in the data passed to the
        transform step but not present in the data used to fit the transformer.
    """

    aliases = ['LabelEncoder']

    class Params(SingleColumnProcessor.Params):
        encoding_range: EncodingRange = EncodingRange.ONE_TO_N
        missing_input_fill_value: Optional[Any] = None
        unknown_input_encoding_value: Optional[Any] = None
        label_normalizer: Optional[Callable[[Any], str]] = None

        @root_validator(pre=False)
        def set_unknown_input_encoding_value(cls, params):
            if params.get('unknown_input_encoding_value') is None:
                params['unknown_input_encoding_value']: Any = \
                    ENCODING_RANGE_TO_UNKNOWN_LABELS_MAP[params['encoding_range']]
            return params

    label_encoding_dict: Dict[Any, int] = None  ## Stores normalized labels if label_normalizer is not None
    label_decoding_dict: Dict[int, Any] = None  ## Stores normalized labels if label_normalizer is not None

    @classmethod
    def from_labelspace(
            cls,
            labelspace: Union[Set, List, Tuple],
            label_encoding_range: EncodingRange,
            label_normalizer: Callable[[Any], str],
    ) -> LabelEncoding:
        """
        Static factory to create a LabelEncoding object from a list/set/tuple of labels.
        :param labelspace: complete set of labels for an ML training dataset.
        :param label_encoding_range: the range of values to which we should encode the labels.
        :param label_normalizer: function to normalize labels.
        :return: LabelEncoding object.
        """
        if len(labelspace) == 2:
            lb1, lb2 = tuple(labelspace)  ## Assume normalized beforehand.
            lb1: str = label_normalizer(lb1)
            lb2: str = label_normalizer(lb2)
            if lb1.upper() in BINARY_NEGATIVE_LABELS and lb2.upper() in BINARY_POSITIVE_LABELS:
                return LabelEncoding(
                    label_encoding_dict={lb1: 0, lb2: 1},
                    label_decoding_dict={0: lb1, 1: lb2},
                    params=dict(
                        encoding_range=EncodingRange.BINARY_ZERO_ONE,
                        label_normalizer=label_normalizer,
                    )
                )
            elif lb1.upper() in BINARY_POSITIVE_LABELS and lb2.upper() in BINARY_NEGATIVE_LABELS:
                return LabelEncoding(
                    label_encoding_dict={lb2: 0, lb1: 1},
                    label_decoding_dict={0: lb2, 1: lb1},
                    params=dict(
                        encoding_range=EncodingRange.BINARY_ZERO_ONE,
                        label_normalizer=label_normalizer,
                    )
                )
            label_encoding_range: EncodingRange = EncodingRange.BINARY_ZERO_ONE
        label_encoder: LabelEncoding = LabelEncoding(
            params=dict(
                encoding_range=label_encoding_range,
                label_normalizer=label_normalizer,
            )
        )
        label_encoder.fit(np.array(list(labelspace)))
        return label_encoder

    def _fit_series(self, data: ScalableSeries):
        ## Cannot use np.unique with NaNs in the data, as it replicates the nans:
        labels: np.ndarray = self._fill_missing_values(data).dropna().numpy()
        if self.params.missing_input_fill_value is not None:
            labels: np.ndarray = np.append(labels, self.params.missing_input_fill_value)
        labels: np.ndarray = np.unique(labels)  ## Makes unique.
        if self.params.label_normalizer is not None:
            ## Normalize labels before encoding:
            labels: np.ndarray = np.array([self.params.label_normalizer(lb) for lb in labels])
            labels: np.ndarray = np.unique(labels)  ## Makes unique post-normalization.
        ## The 2nd return param is an index of the unique labels, i.e. an encoding from 0 to N-1:
        labels, encoded_labels = np.unique(labels, return_inverse=True)
        num_labels, num_encodings = len(labels), len(encoded_labels)
        if num_labels == 0:
            raise ValueError(f'Input data must contain at least one non-null entry.')
        if num_labels != num_encodings:
            raise ValueError(
                f'Each label should have exactly one encoding. ' + \
                f'Found: no. unique labels={num_labels}, no. encodings={num_encodings}'
            )
        ## Adjust label encoding based on encoding range:
        if self.params.encoding_range is EncodingRange.ZERO_TO_N_MINUS_ONE:
            self.label_encoding_dict: Dict[Any, int] = dict(zip(labels, encoded_labels))
        elif self.params.encoding_range is EncodingRange.ONE_TO_N:
            ## encoded_labels goes from 0 to N-1
            self.label_encoding_dict: Dict[Any, int] = dict(zip(labels, encoded_labels + 1))
        elif self.params.encoding_range is EncodingRange.BINARY_ZERO_ONE:
            if num_labels > 2:
                raise ValueError(f'{EncodingRange.BINARY_ZERO_ONE} encoding supports <=2 labels, found {num_labels}')
            self.label_encoding_dict: Dict[Any, int] = {labels[0]: 0}
            if num_labels == 2:
                self.label_encoding_dict[labels[1]] = 1
        elif self.params.encoding_range is EncodingRange.BINARY_PLUS_MINUS_ONE:
            if num_labels > 2:
                raise ValueError(f'{EncodingRange.BINARY_PLUS_MINUS_ONE} needs <=2 labels, found {num_labels}')
            self.label_encoding_dict: Dict[Any, int] = {labels[0]: -1}
            if num_labels == 2:
                self.label_encoding_dict[labels[1]] = 1
        else:
            raise NotImplementedError(f'Unsupported encoding range: {self.params.encoding_range}')
        self.label_decoding_dict: Dict[int, Any] = {v: k for k, v in self.label_encoding_dict.items()}

    def _transform_series(self, data: ScalableSeries) -> ScalableSeries:
        if self.label_encoding_dict is None:
            raise self.FitBeforeTransformError
        data: ScalableSeries = self._fill_missing_values(data)
        if self.params.label_normalizer is not None:
            data: ScalableSeries = data.map(self.params.label_normalizer, na_action='ignore')
        return data.map(self.label_encoding_dict, na_action='ignore').fillna(self.params.unknown_input_encoding_value)

    def transform_single(self, data: Optional[Any]) -> int:
        if self.label_encoding_dict is None:
            raise self.FitBeforeTransformError
        data = self._fill_missing_value(data)
        return int(self.label_encoding_dict.get(data, self.params.unknown_input_encoding_value))

    def inverse_transform_series(
            self,
            data: Union[ScalableSeries, ScalableSeriesRawType],
    ) -> Union[ScalableSeries, ScalableSeriesRawType]:
        if self.label_decoding_dict is None:
            raise self.FitBeforeTransformError
        output: ScalableSeries = ScalableSeries.of(data).map(self.label_decoding_dict, na_action='ignore')
        if not isinstance(data, ScalableSeries):
            output: ScalableSeriesRawType = output.raw()
        return output

    def inverse_transform_single(self, data: int) -> Optional[str]:
        if self.label_decoding_dict is None:
            raise self.FitBeforeTransformError
        if not isinstance(data, int):
            raise ValueError(f'Expected input data to be an integer; found {type_str(data)} having value: {data}')
        return self.label_decoding_dict.get(data)

    def _fill_missing_value(self, data: Any):
        """TODO: replace this with a transformer or util which imputes missing values."""
        if is_null(data) and self.params.missing_input_fill_value is not None:
            return self.params.missing_input_fill_value
        return data

    def _fill_missing_values(self, data: ScalableSeries) -> ScalableSeries:
        """TODO: replace this with a transformer or util which imputes missing values."""
        if self.params.missing_input_fill_value is not None:
            return data.fillna(self.params.missing_input_fill_value)
        return data
