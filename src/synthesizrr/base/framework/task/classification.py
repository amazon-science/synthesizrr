from typing import *
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame as PandasDataFrame
from synthesizrr.base.util import as_list, as_tuple, safe_validate_arguments, is_not_null, is_list_or_set_like, flatten1d
from synthesizrr.base.util.language import _all_are_np_subtypes
from synthesizrr.base.data import ScalableSeries, ScalableSeriesRawType, ScalableDataFrame
from synthesizrr.base.constants import MLType, DataSplit, Task, MLTypeSchema, DataLayout
from synthesizrr.base.framework import Algorithm, Dataset, Predictions, Metric, Metrics
from synthesizrr.base.data import FileMetadata
from synthesizrr.base.data.processor import LabelEncoding, EncodingRange
from collections import defaultdict
from functools import partial
from pydantic import constr
from pydantic.typing import Literal


class ClassificationData(Dataset):
    tasks = (
        Task.BINARY_CLASSIFICATION,
        Task.MULTI_CLASS_CLASSIFICATION,
    )
    ground_truths_schema = {
        '{ground_truth_col}': MLType.CATEGORICAL,
    }

    @property
    def ground_truth_label_col_name(self) -> str:
        if len(self.data_schema.ground_truths_schema) == 0:
            raise ValueError(f'No ground-truth column found in {self}')
        return next(iter(self.data_schema.ground_truths_schema.keys()))

    @classmethod
    def validate_data(cls, data_schema: MLTypeSchema, data: ScalableDataFrame) -> bool:
        pass


CLASSIFICATION_PREDICTIONS_FORMAT_MSG: str = f"""
Classifier predictions returned by algorithm must be a dict in one of the following formats:
1. Top-k: 
Provides labels and probability scores for each example.
Required keys:
- 'top_k_labels' (1D/2D list or Numpy array): pre-sorted in descending order by score.
- 'top_k_scores' (1D/2D list or Numpy array): pre-sorted in descending order by score.
Optional keys:
- 'labelspace' (1D tuple): unique labels.
    E.g.
        Binary: ('COVID', 'not_COVID')
        Multi-class/Multi-label: ('A', 'B', 'C', 'D, 'E')

2. Labelwise:
Provides column of probability scores for each label.   
Required keys:
- 'labels' (1D tuple): unique labels.
    E.g.
        Binary: ('not_COVID', 'COVID')
        Multi-class/Multi-label: ('A', 'B', 'C', 'D, 'E')
- 'scores' (1D/2D list or Numpy array): 
    - Rows should be examples. 
    - If 'scores' is 1D (binary classification), we expect each element to be 
    the probability of the negative class for a different example.
    Here, the first element of 'labels' is treated as the negative class.
    Each row need not add up to 1, but each cell must be between 0 and 1 (inclusive).
    E.g.
        labels = ('not_COVID', 'COVID')
        scores = [0.90, 0.01, 0.33]
        This becomes:
            COVID
        0   0.90  
        1   0.01  
        2   0.33  

    - If 'scores' is 2D (binary/multi-class/multi-label classification), 
    each row is considered a different example. Each column is
    considered to be the scores of a particular label, across all examples,
    with index corresponding to 'labels'.
    Each row need not add up to 1, but each cell must be between 0 and 1 (inclusive) 
    E.g.
        labels = ('A', 'B', 'C', 'D, 'E')
        scores =
              0    1     2     3    4
        0  0.90  0.8  0.00  0.70  0.0
        1  0.00  0.3  0.00  0.40  0.0
        2  0.33  0.0  0.34  0.21  0.2
        This becomes:
              A    B     C     D    E
        0  0.90  0.8  0.00  0.70  0.0
        1  0.00  0.3  0.00  0.40  0.0
        2  0.33  0.0  0.34  0.21  0.2
        Here, we will consider first column to be probabilities of label 'A', 
        second to be probabilities of label 'B', etc.
""".strip()

ClassificationPredictions = "ClassificationPredictions"
TopKClassificationPredictions = "TopKClassificationPredictions"
LabelwiseClassificationPredictions = "LabelwiseClassificationPredictions"

TOP_PREDICTED_LABEL_TEMPLATE: str = 'top_{predicted_label_col_num}_predicted_label'
TOP_1_PREDICTED_LABEL: str = TOP_PREDICTED_LABEL_TEMPLATE.format(predicted_label_col_num=1)
TOP_PREDICTED_SCORE_TEMPLATE: str = 'top_{predicted_score_col_num}_predicted_score'
TOP_1_PREDICTED_SCORE: str = TOP_PREDICTED_SCORE_TEMPLATE.format(predicted_score_col_num=1)


class ClassificationPredictions(Predictions, ABC):
    tasks = (
        Task.BINARY_CLASSIFICATION,
        Task.MULTI_CLASS_CLASSIFICATION,
        Task.MULTI_LABEL_CLASSIFICATION,
    )

    _allow_multiple_subclasses: ClassVar[bool] = True

    _top_k_predictions: Optional[TopKClassificationPredictions] = None
    _labelwise_predictions: Optional[LabelwiseClassificationPredictions] = None
    labelspace: Tuple[str, ...]  ## Assume normalized
    negative_label: Optional[str] = None
    positive_label: Optional[str] = None

    ground_truths_schema = {
        '{ground_truth_col}': MLType.CATEGORICAL,
    }

    def top_1_predicted_label(self) -> ScalableSeries:
        top_k_predictions: TopKClassificationPredictions = self.to_top_k()
        return top_k_predictions.data[TOP_1_PREDICTED_LABEL]

    def top_1_predicted_score(self) -> ScalableSeries:
        top_k_predictions: TopKClassificationPredictions = self.to_top_k()
        return top_k_predictions.data[TOP_1_PREDICTED_SCORE]

    def predictions(
            self,
            *,
            multilabel: bool = False,
            score_threshold: float = 0.5,
            **kwargs,
    ) -> Union[ScalableDataFrame, ScalableSeries]:
        if multilabel:
            labelwise_predictions: LabelwiseClassificationPredictions = self.to_labelwise()
            return ScalableSeries.of(
                labelwise_predictions.predictions().apply(
                    lambda row: tuple(lb for lb in self.labelspace if row[lb] >= score_threshold),
                    axis=1
                )
            )
        return super(ClassificationPredictions, self).predictions(**kwargs)

    def predictions_multihot(self, *, score_theshold: float = 0.5, **kwargs) -> ScalableDataFrame:
        if self.task is not Task.MULTI_LABEL_CLASSIFICATION:
            raise ValueError(f'Cannot get multihot predictions for {self.class_name} with task "{self.task}"')
        labelwise_preds: LabelwiseClassificationPredictions = self.to_labelwise(**kwargs)
        return ScalableDataFrame.of(
            labelwise_preds.data[list(labelwise_preds.labelspace)].pandas() >= score_theshold
        )

    def ground_truths_multihot(self, **kwargs) -> ScalableDataFrame:
        if self.task is not Task.MULTI_LABEL_CLASSIFICATION:
            raise ValueError(f'Cannot get multihot ground-truths for {self.class_name} with task "{self.task}"')
        labels: ScalableSeries = self.ground_truths().apply(as_tuple)
        multi_hot_df: pd.DataFrame = pd.DataFrame(
            np.zeros((len(self), self.num_labels), dtype=bool),
            columns=self.labelspace
        )
        for i, lb_list in enumerate(labels):
            multi_hot_df.loc[i, lb_list]: bool = True
        return ScalableDataFrame.of(multi_hot_df)

    @property
    def ground_truth_label_col_name(self) -> str:
        if len(self.data_schema.ground_truths_schema) == 0:
            raise ValueError(f'No ground-truth column found in {self}')
        return next(iter(self.data_schema.ground_truths_schema.keys()))

    @abstractmethod
    def to_top_k(self, **kwargs) -> TopKClassificationPredictions:
        pass

    @abstractmethod
    def to_labelwise(self, **kwargs) -> LabelwiseClassificationPredictions:
        pass

    @property
    def num_labels(self) -> int:
        return len(self.labelspace)

    @property
    def is_binary(self) -> bool:
        return self.num_labels == 2

    @classmethod
    def _decode_and_normalize_ground_truth_labels(
            cls,
            predictions: ClassificationPredictions,
            label_encoder: Optional[LabelEncoding],
            predicted_label_cols: Union[Tuple, Set, List],
    ) -> ClassificationPredictions:
        if label_encoder is not None and predictions.has_ground_truths(raise_error=False):
            ## If we set encoding_range in Classifier, then we will create a label_encoder, and additionally we will
            ## encode & normalize ClassificationData's ground-truths before it is passed to Classifier.predict_step
            ## (this is done in Classifier._task_preprocess). These decoded ground-truths are passed forward to
            ## ClassificationPredictions, and thus, we will need to decode it here.
            ## However, if we do not set encoding_range in Classifier, then we only normalize ClassificationData's
            ## ground-truths (in Classifier._task_preprocess), before it is passed to Classifier.predict_step. In
            ## that case, we do not need to take any action here, as ClassificationPredictions already contains
            ## normalized ground-truths which had never been encoded.
            if predictions.task is Task.MULTI_LABEL_CLASSIFICATION:
                predictions.data[predictions.ground_truth_label_col_name]: ScalableSeries = \
                    predictions.data[predictions.ground_truth_label_col_name].apply(
                        decode_multilabel,
                        label_encoder=label_encoder,
                    )
            else:
                predictions.data[predictions.ground_truth_label_col_name]: ScalableSeries = \
                    label_encoder.inverse_transform_series(predictions.data[predictions.ground_truth_label_col_name])
        return predictions


class TopKClassificationPredictions(ClassificationPredictions):
    predictions_schema = {
        TOP_PREDICTED_LABEL_TEMPLATE: MLType.CATEGORICAL,
        TOP_PREDICTED_SCORE_TEMPLATE: MLType.FLOAT,
    }

    def to_top_k(self, **kwargs) -> TopKClassificationPredictions:
        if self._top_k_predictions is None:
            self._top_k_predictions = self
        return self._top_k_predictions

    def to_labelwise(self, **kwargs) -> LabelwiseClassificationPredictions:
        if self._labelwise_predictions is None:
            self._labelwise_predictions = self._top_k_to_labelwise(self, **kwargs)
        return self._labelwise_predictions

    @classmethod
    def _top_k_to_labelwise(cls, top_k: TopKClassificationPredictions, **kwargs) -> LabelwiseClassificationPredictions:
        raise NotImplementedError()

    @classmethod
    @safe_validate_arguments
    def from_top_k(
            cls,
            data: ClassificationData,
            labels: Union[np.ndarray, List, Tuple],
            scores: Union[np.ndarray, List, Tuple],
            labelspace: Tuple[str, ...],
            label_encoder: Optional[LabelEncoding],
            ascending: bool = False,
            **kwargs
    ) -> ClassificationPredictions:
        if isinstance(labels, (list, tuple)):
            labels: np.ndarray = np.array(labels)
        if isinstance(scores, (list, tuple)):
            scores: np.ndarray = np.array(scores)
        if not _all_are_np_subtypes(scores.dtype, {np.bool_, np.integer, np.floating}):
            raise ValueError(f'Expected scores array to have dtype as bool, int or float; found: {scores.dtype}')
        if labels.ndim == 1:
            labels: np.ndarray = labels[..., np.newaxis]  ## stackoverflow.com/a/25755697/4900327
        if scores.ndim == 1:
            scores: np.ndarray = scores[..., np.newaxis]  ## stackoverflow.com/a/25755697/4900327
        if labels.shape != scores.shape or labels.ndim != 2 or scores.ndim != 2:
            raise ValueError(
                f'Predicted labels and scores must both be 2-dimensional numpy arrays of the same shape; '
                f'predicted labels has shape {labels.shape}, '
                f'but Predicted scores has length {scores.shape}'
            )

        ncols: int = labels.shape[1]
        predicted_label_cols: List[str] = [
            TOP_PREDICTED_LABEL_TEMPLATE.format(predicted_label_col_num=col_num)
            for col_num in range(1, ncols + 1)
        ]
        predicted_score_cols: List[str] = [
            TOP_PREDICTED_SCORE_TEMPLATE.format(predicted_score_col_num=col_num)
            for col_num in range(1, ncols + 1)
        ]

        ## Typically from top-k, we expect descending-sorted columns, but we might have ascending order:
        if ascending:
            predicted_label_cols: List[str] = predicted_label_cols[::-1]
            predicted_score_cols: List[str] = predicted_score_cols[::-1]

        predictions: Dict[str, np.ndarray] = {}  ## TODO: see if this slows down RECORD and DATUM performance
        for i, predicted_label_col in enumerate(predicted_label_cols):
            predictions[predicted_label_col] = labels[:, i]
        for i, predicted_score_col in enumerate(predicted_score_cols):
            predictions[predicted_score_col] = scores[:, i]

        predictions: ClassificationPredictions = cls.from_task_data(
            data=data,
            predictions=predictions,
            labelspace=labelspace,
            **kwargs
        )
        predictions: ClassificationPredictions = cls._decode_and_normalize_ground_truth_labels(
            predictions,
            label_encoder=label_encoder,
            predicted_label_cols=predicted_label_cols,
        )
        if label_encoder is not None:
            ## If we set encoding_range in Classifier, then we will create a label_encoder, and additionally we will
            ## encode & normalize ClassificationData's ground-truths before it is passed to Classifier.predict_step
            ## (this is done in Classifier._task_preprocess). Thus, the model will learn to predict encoded values, and
            ## we must decode & normalize the predicted labels here.
            for predicted_label_col in predicted_label_cols:
                predictions.data[predicted_label_col]: ScalableSeries = \
                    label_encoder.inverse_transform_series(predictions.data[predicted_label_col])
        return predictions


class LabelwiseClassificationPredictions(ClassificationPredictions):
    ## Columns names are labels.
    predictions_schema = {
        '{label_name}': MLType.FLOAT,
    }

    def to_top_k(self, **kwargs) -> TopKClassificationPredictions:
        if self._top_k_predictions is None:
            self._top_k_predictions: TopKClassificationPredictions = self._labelwise_to_top_k(self, **kwargs)
        return self._top_k_predictions

    def to_labelwise(self, **kwargs) -> LabelwiseClassificationPredictions:
        if self._labelwise_predictions is None:
            self._labelwise_predictions: LabelwiseClassificationPredictions = self
        return self._labelwise_predictions

    @classmethod
    def _labelwise_to_top_k(
            cls,
            labelwise: LabelwiseClassificationPredictions,
            **kwargs
    ) -> ClassificationPredictions:
        ## TODO: figure out how to perform the following steps efficiently for all SDF implementations.
        if labelwise.data.layout is DataLayout.DASK:
            raise NotImplementedError()
        else:
            labelspace: List[str] = list(labelwise.labelspace)
            ## Select only the score columns:
            labelwise_predicted_scores: ScalableDataFrame = labelwise.data[labelspace]
            labelwise_predicted_scores: np.ndarray = labelwise_predicted_scores.pandas().values
            ## Typically from top-k, we want descending-sorted columns, but sorting by ascending is faster.
            labelspace: np.ndarray = np.array(labelspace)
            predicted_labels_ascending: np.ndarray = labelspace[labelwise_predicted_scores.argsort(axis=1)]
            predicted_scores_ascending: np.ndarray = np.sort(labelwise_predicted_scores, axis=1)
            return TopKClassificationPredictions.from_top_k(
                data=labelwise.as_task_data(deep=False),
                labels=predicted_labels_ascending,
                scores=predicted_scores_ascending,
                labelspace=labelwise.labelspace,
                label_encoder=None,  ## Since we start from a Predictions object, the labels are already decoded.
                ascending=True,
            )

    @classmethod
    @safe_validate_arguments
    def from_scores(
            cls,
            data: ClassificationData,
            labels: Union[np.ndarray, List, Tuple],
            scores: Union[np.ndarray, List, Tuple],
            labelspace: Tuple[str, ...],
            label_encoder: Optional[LabelEncoding],
            **kwargs
    ) -> ClassificationPredictions:
        labels: List = as_list(labels)
        if label_encoder is not None:
            ## Decode labels:
            labels: np.ndarray = np.array(labels).astype(int)
            labels: List[str] = list(label_encoder.inverse_transform_series(labels))
        if set(labels) != set(labelspace):
            raise ValueError(
                f'When creating predictions from scores, `labels` should be the set of labels which, if decoded and '
                f'normalized, exactly matches the `labelspace`.'
                f'\nFound `labels` (length={len(labels)}): {labels}'
                f'\nExpected `labelspace` (length={len(labelspace)}): {labelspace}'
            )
        if isinstance(scores, (list, tuple)):
            scores: np.ndarray = np.array(scores)
        if scores.ndim == 1:
            scores: np.ndarray = scores[..., np.newaxis]  ## stackoverflow.com/a/25755697/4900327
        ncols: int = scores.shape[1]
        negative_label: Optional[str] = None
        positive_label: Optional[str] = None
        if ncols == 1:
            ## Binary classification:
            if len(labels) != 2:
                raise ValueError(
                    f'When passing "scores" as a single column, we consider it a binary classification problem, '
                    f'with "scores" as the positive class scores. '
                    f'It is this expected that "labels" will have exactly two elements, with syntax: '
                    f'(negative_label, positive_label); however, found following `labels`: {labels}'
                )
            label_cols: List[str] = sorted(list(cls.schema_template.populate(
                allow_unfilled=True,
                features=False,
                ground_truths=False,
                label_name=labels
            ).predictions_schema.keys()))
            predictions: Dict[str, np.ndarray] = {}
            ## As per contract, first label should be negative and second should be positive.
            negative_label: str = label_cols[0]
            positive_label: str = label_cols[1]
            predictions[positive_label] = scores[:, 0]
            ## Expand the binary label into two columns. Note this can be numerically unstable e.g. 1.0 - 0.9:
            predictions[negative_label] = 1.0 - predictions[positive_label]
            for i, label_col in enumerate(label_cols):
                predictions[label_col] = scores[:, i]
        else:
            ## Binary/multi-class/multi-label classification:
            if len(labels) != ncols:
                raise ValueError(
                    f'Different number of labels and scores: found {len(labels)} labels, '
                    f'but scores has shape {scores.shape}'
                )
            label_cols: List[str] = sorted(list(cls.schema_template.populate(
                allow_unfilled=True,
                features=False,
                ground_truths=False,
                label_name=labels
            ).predictions_schema.keys()))
            if len(label_cols) == 2:
                ## As per contract, first label should be negative and second should be positive.
                negative_label: str = label_cols[0]
                positive_label: str = label_cols[1]

            predictions: Dict[str, np.ndarray] = {}  ## TODO: see if this slows down RECORD and DATUM performance
            for i, label_col in enumerate(label_cols):
                predictions[label_col] = scores[:, i]

        predictions: ClassificationPredictions = cls.from_task_data(
            data=data,
            predictions=predictions,
            labelspace=labelspace,
            negative_label=negative_label,
            positive_label=positive_label,
            **kwargs
        )
        ## This should only affect ground-truths column:
        predictions: ClassificationPredictions = cls._decode_and_normalize_ground_truth_labels(
            predictions,
            label_encoder=label_encoder,
            predicted_label_cols=label_cols,
        )

        return predictions


class Labelspace(Metric):
    aliases = ['labelspace']
    labelspace: Set = set()

    def update(self, data: Union[Dataset, Predictions]):
        data.check_in_memory()
        if not isinstance(data, (ClassificationData, ClassificationPredictions)):
            raise ValueError(
                f'Can only calculate number of labels for {ClassificationData} and {ClassificationPredictions}; '
                f'found input data of type `{type(data)}`'
            )
        ground_truths: ScalableSeries = data.ground_truths()
        if ground_truths.hasnans:
            nan_indexes: List = list(data.index()[ground_truths.isna()])
            raise ValueError(f'Found Nan or None labels in rows with following indexes:\n{nan_indexes}')
        if isinstance(data, MultiLabelClassificationData):
            ground_truths: List[Tuple] = ground_truths.apply(
                split_normalize_encode_multilabel,
                label_sep=data.label_sep,
                label_normalizer=None,
                label_encoder=None,
            )
            for gt_tuple in ground_truths:
                self.labelspace.update(set(gt_tuple))
        else:
            self.labelspace.update(ground_truths.as_set())

    def compute(self) -> Any:
        return self.labelspace


def _normalize_label(x: Any, minimal: bool = False) -> str:
    ## Fast removal of crud from labels. Useful when labels are input by users, who often add crud.
    ## Does the following:
    ## 1. Converts to string
    ## 2. Strips leading and trailing whitespace
    ## 3. Removes ".0" if it exists (e.g. if the number was "123.0", it becomes "123")
    ## 4. Replaces hyphens with underscores.
    ## 5. Replaces space (in the middle of the string), with underscores.
    ## 6. De-duplicates underscores ("__" becomes "_"). We do this twice, so {'____', '___', '__'} all map to '_'.
    ## 7. Uppercase label.
    ## When "minimal=True", we only do steps 1, 2 and 3.
    ##
    ##  ==>> Examples:
    ##  1) {'NORTH_AMERICA', 'North America', 'North-America', ' North_America'} map to 'NORTH_AMERICA'
    ##  2) 'NORTH AMER ICA' map to 'NORTH_AMER_ICA' (we do not tamper with spaces in the middle of a string)
    ##  3) {'1.0', '1', 1, 1.0} map to '1' (same for other numbers).
    ##  4) {.0, '.0'} map to '0'
    ##  5) 1.00000000000007 maps to '1.00000000000007'
    ##
    ## ==>> Performance:
    ##  This method is fast, ~370ns per item (tested on M1 Macbook Pro 16-inch 2021, Monterey v12.6.1):
    ##  >>> x = [
    ##      'NORTH_AMERICA',
    ##      'North America',
    ##      'North-America',
    ##      'North_America  ',
    ##      '1.0',
    ##      1,
    ##      1.0,
    ##      1.00000000000007,
    ##      .0,
    ##      'NORTH AMER ICA'
    ##  ]
    ##  >>> [_normalize_label(y) for y in x]
    ##  [
    ##      'NORTH_AMERICA',
    ##      'NORTH_AMERICA',
    ##      'NORTH_AMERICA',
    ##      'NORTH_AMERICA',
    ##      '1',
    ##      '1',
    ##      '1',
    ##      '1.00000000000007',
    ##      '0',
    ##      'NORTH_AMER_ICA'
    ##  ]
    ##  >>> set([_normalize_label(y) for y in x])
    ##  {'0', '1', '1.00000000000007', 'NORTH_AMER_ICA', 'NORTH_AMERICA'}
    ##  >>> len(x)
    ##  10
    ##  >>> %timeit [_normalize_label(y) for y in x]
    ##  3.71 µs ± 48.6 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    x: str = str(x).strip()
    if x == '.0':
        x = '0'
    if x.endswith('.0'):
        x = x[:-2]
    if minimal:
        return x
    return x.replace(' ', '_') \
        .replace('-', '_') \
        .replace('__', '_').replace('__', '_') \
        .upper()


class Classifier(Algorithm, ABC):
    tasks = (
        Task.BINARY_CLASSIFICATION,
        Task.MULTI_CLASS_CLASSIFICATION,
    )
    inputs = ClassificationData
    outputs = ClassificationPredictions
    dataset_statistics = ('labelspace',)

    label_encoding_range: ClassVar[Optional[EncodingRange]] = None  ## Only used to create `label_encoder`
    label_normalizer: Optional[Callable[[Any], str]]
    labelspace: Optional[Tuple[str, ...]]
    label_encoder: Optional[LabelEncoding]

    def __init__(
            self,
            *,
            stats: Optional[Metrics] = None,
            normalize_labels: bool = True,
            label_normalizer: Optional[Callable[[Any], str]] = None,
            **kwargs
    ):
        super(Classifier, self).__init__(stats=stats, **kwargs)
        if label_normalizer is not None:
            self.label_normalizer: Callable[[Any], str] = label_normalizer
        elif normalize_labels is True:
            self.label_normalizer: Callable[[Any], str] = _normalize_label
        else:
            ## Do a minimal label normalization (convert to appropriate string):
            self.label_normalizer: Callable[[Any], str] = partial(_normalize_label, minimal=True)

        if self.labelspace is None:
            self.labelspace: Tuple[str, ...] = self._create_labelspace(
                stats=stats,
                label_normalizer=self.label_normalizer,
                **kwargs,
            )
        if self.label_encoder is None:
            self.label_encoder: Optional[LabelEncoding] = self._create_label_encoder(
                labelspace=self.labelspace,
                label_normalizer=self.label_normalizer,
                **kwargs
            )

    @property
    def num_labels(self) -> int:
        return len(self.labelspace)

    @property
    def is_binary(self) -> bool:
        return len(self.labelspace) == 2

    @property
    def is_multiclass(self) -> bool:
        return len(self.labelspace) > 2

    @property
    def encoded_labelspace(self) -> Tuple[int, ...]:
        if self.label_encoder is None:
            raise ValueError(
                f'Label encoding is not done; please specify `label_encoding_range` as a class member variable in the '
                f'definition of "{self.class_name}"'
            )
        return tuple(sorted(self.label_encoder.label_encoding_dict.values()))

    @classmethod
    def _create_labelspace(
            cls,
            stats: Metrics,
            label_normalizer: Callable[[Any], str],
            **kwargs
    ) -> Tuple[str, ...]:
        if stats is None:
            raise ValueError(f'Required to either pass a custom labelspace, or calculate stats.')
        labelspace: Labelspace = stats.find(data_split=DataSplit.TRAIN, select='labelspace')
        labelspace: Tuple[str, ...] = tuple(sorted({label_normalizer(lb) for lb in labelspace.value}))
        if len(labelspace) < 2:
            raise ValueError(
                f'Post normalization, there are {len(labelspace)} labels: {labelspace}; at least 2 unique labels are '
                f'needed. Please update your data, or pass a custom `label_normalizer` method while invoking '
                f'"{Classifier.class_name}".'
            )
        return labelspace

    @classmethod
    def _create_label_encoder(
            cls,
            labelspace: Tuple[str, ...],
            label_normalizer: Callable[[Any], str],
            **kwargs,
    ) -> Optional[LabelEncoding]:
        if cls.label_encoding_range is None:
            return None
        return LabelEncoding.from_labelspace(
            labelspace=labelspace,
            label_encoding_range=cls.label_encoding_range,
            label_normalizer=label_normalizer,
        )

    def _task_preprocess(self, batch: ClassificationData, **kwargs) -> ClassificationData:
        ## Only normalize  ground-truth labels:
        if batch.has_ground_truths(raise_error=False):
            if self.label_encoder is not None:  ## i.e. if label_encoding_range was specified.
                ## Normalize and encode ground-truth labels:
                if batch.has_ground_truths(raise_error=False):
                    batch: ClassificationData = self._encode_batch_labels(batch)
            else:  ## i.e. if label_encoding_range was not specified (i.e. the user does not want to encode labels).
                batch: ClassificationData = self._normalize_batch_labels(batch)
        return batch

    def _normalize_batch_labels(self, batch: ClassificationData) -> ClassificationData:
        batch.data[batch.ground_truth_label_col_name]: ScalableSeries = \
            batch.data[batch.ground_truth_label_col_name].map(self.label_normalizer)
        return batch

    def _encode_batch_labels(self, batch: ClassificationData) -> ClassificationData:
        batch.data[batch.ground_truth_label_col_name]: ScalableSeries = self.label_encoder.transform(
            batch.data[batch.ground_truth_label_col_name]
        )
        return batch

    def _create_predictions(
            self,
            batch: ClassificationData,
            predictions: Dict,
            top_k: Optional[bool] = None,
            **kwargs
    ) -> ClassificationPredictions:
        if not isinstance(predictions, dict):
            raise ValueError(CLASSIFICATION_PREDICTIONS_FORMAT_MSG)
        if 'top_k_labels' in predictions and 'top_k_scores' in predictions:
            predictions: ClassificationPredictions = TopKClassificationPredictions.from_top_k(
                data=batch,
                labels=predictions['top_k_labels'],
                scores=predictions['top_k_scores'],
                labelspace=self.labelspace,
                label_encoder=self.label_encoder,
                **kwargs
            )
        elif 'scores' in predictions and 'labels' in predictions:
            predictions: ClassificationPredictions = LabelwiseClassificationPredictions.from_scores(
                data=batch,
                labels=predictions['labels'],
                scores=predictions['scores'],
                labelspace=self.labelspace,
                label_encoder=self.label_encoder,
                **kwargs
            )
        else:
            raise ValueError(CLASSIFICATION_PREDICTIONS_FORMAT_MSG)
        if top_k is True:
            return predictions.to_top_k()
        elif top_k is False:
            return predictions.to_labelwise()
        else:
            return predictions


class MultiLabelClassificationData(ClassificationData):
    tasks = Task.MULTI_LABEL_CLASSIFICATION
    ground_truths_schema = {
        '{ground_truth_col}': MLType.VECTOR,
    }

    label_sep: constr(min_length=1, max_length=3) = ','

    @classmethod
    def validate_data(cls, data_schema: MLTypeSchema, data: ScalableDataFrame) -> bool:
        pass


class MultiLabelClassifier(Classifier, ABC):
    tasks = Task.MULTI_LABEL_CLASSIFICATION
    inputs = MultiLabelClassificationData

    def _normalize_batch_labels(self, batch: MultiLabelClassificationData) -> MultiLabelClassificationData:
        batch.data[batch.ground_truth_label_col_name]: ScalableSeries = \
            batch.data[batch.ground_truth_label_col_name].apply(
                split_normalize_encode_multilabel,
                label_sep=batch.label_sep,
                label_normalizer=self.label_normalizer,
                label_encoder=None,
            )
        return batch

    def _encode_batch_labels(self, batch: MultiLabelClassificationData) -> MultiLabelClassificationData:
        batch.data[batch.ground_truth_label_col_name]: ScalableSeries = \
            batch.data[batch.ground_truth_label_col_name].apply(
                split_normalize_encode_multilabel,
                label_sep=batch.label_sep,
                label_normalizer=self.label_normalizer,
                label_encoder=self.label_encoder,
            )
        return batch


def split_normalize_encode_multilabel(
        labels_list: Union[List, Tuple, Set, np.ndarray, str],
        label_sep: str,
        label_normalizer: Optional[Callable],
        label_encoder: Optional[LabelEncoding],
) -> Union[Tuple[int, ...], Tuple[str, ...]]:
    lb_list: Union[List, Tuple, Set, np.ndarray, str] = labels_list
    if isinstance(lb_list, int):
        lb_list: str = str(lb_list)
    if isinstance(lb_list, str):
        # print(f'Labels is a string with value: {repr(lb_list)}')
        lb_list: str = lb_list.strip()
        if lb_list.startswith('[') and lb_list.endswith(']'):
            lb_list: str = lb_list.removeprefix('[').removesuffix(']')
        elif lb_list.startswith('(') and lb_list.endswith(')'):
            lb_list: str = lb_list.removeprefix('(').removesuffix(')')
        elif lb_list.startswith('{') and lb_list.endswith('}'):
            lb_list: str = lb_list.removeprefix('{').removesuffix('}')
        lb_list: str = lb_list.replace('"', '').replace("'", '')  ## Remove quotes
        lb_list: List[str] = [lb.strip() for lb in lb_list.split(label_sep)]
    if not is_list_or_set_like(lb_list):
        raise ValueError(
            f'Expected each label to be a list, tuple, set, etc, or string equivalent; '
            f'found: {type(labels_list)} with value: {labels_list}'
        )
    lb_list: List = as_list(lb_list)
    if label_encoder is not None:
        split_normalized_encoded_labels: Tuple[int, ...] = tuple(
            label_encoder.transform_single(label_encoder.params.label_normalizer(lb))
            for lb in lb_list
        )
        return split_normalized_encoded_labels
    elif label_normalizer is not None:
        split_normalized_labels: Tuple[str, ...] = tuple(label_normalizer(lb) for lb in lb_list)
        return split_normalized_labels
    else:
        split_labels: Tuple = tuple(lb_list)
        return split_labels


def decode_multilabel(
        encoded_labels_list: Union[List, Tuple, Set, np.ndarray, str],
        label_encoder: LabelEncoding,
) -> Tuple[str, ...]:
    enc_lb_list: List = as_list(encoded_labels_list)
    if not is_list_or_set_like(encoded_labels_list):
        raise ValueError(
            f'Expected each label to be a list, tuple, set, etc; '
            f'found: {type(encoded_labels_list)} with value: {encoded_labels_list}'
        )
    return tuple(
        label_encoder.inverse_transform_single(enc_lb)
        for enc_lb in enc_lb_list
    )
