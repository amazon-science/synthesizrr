from typing import *
from abc import ABC, abstractmethod
import pandas as pd, numpy as np
from collections import defaultdict
from synthergent.base.constants import Task, AggregationStrategy, DataSplit
from synthergent.base.util import as_tuple, type_str, check_isinstance, argmax, dict_key_with_best_value
from synthergent.base.framework import Predictions, PercentageMetric, AggregatedPercentageMetric, TabularMetric
from synthergent.base.framework import ClassificationPredictions, TopKClassificationPredictions, LabelwiseClassificationPredictions, \
    TOP_1_PREDICTED_LABEL
from synthergent.base.framework.trainer.RayTuneTrainer import _RAY_TRIAL_ID, _RAY_TRAINING_ITERATION, _ray_metric_str
from functools import singledispatchmethod
from pydantic import confloat, root_validator
from pydantic.typing import Literal


def _check_clf_preds(data: Predictions):
    if not isinstance(data, ClassificationPredictions):
        raise ValueError(
            f'Expected input data to be subclass of {ClassificationPredictions}; '
            f'found data of type {type(data)}'
        )
    if data.task not in {Task.BINARY_CLASSIFICATION, Task.MULTI_CLASS_CLASSIFICATION, Task.MULTI_LABEL_CLASSIFICATION}:
        raise ValueError(
            f'Data expected to be predictions of a binary, multi-class or multi-label classification task; '
            f'found task to be {data.task}.'
        )
    data.has_ground_truths()
    data.has_predictions()


def _check_binary_clf_preds(data: ClassificationPredictions):
    _check_clf_preds(data)
    if not data.is_binary:
        raise ValueError(f'Data expected to be binary; found {data.num_labels} classes.')
    if data.task is not Task.BINARY_CLASSIFICATION:
        raise ValueError(
            f'Data expected to be predictions of a binary classification task; '
            f'found task to be {data.task}.'
        )


def _check_multiclass_or_multilabel_clf_preds(data: ClassificationPredictions):
    _check_clf_preds(data)
    if data.task not in {Task.MULTI_CLASS_CLASSIFICATION, Task.MULTI_LABEL_CLASSIFICATION}:
        raise ValueError(
            f'Data expected to be predictions of a multi-class or multi-label classification task; '
            f'found task to be {data.task}.'
        )


def _check_multiclass_clf_preds(data: ClassificationPredictions):
    _check_clf_preds(data)
    if data.task is not Task.MULTI_CLASS_CLASSIFICATION:
        raise ValueError(
            f'Data expected to be predictions of a multi-class classification task; '
            f'found task to be {data.task}.'
        )


def _check_binary_or_multiclass_clf_preds(data: ClassificationPredictions):
    _check_clf_preds(data)
    if data.task not in {Task.BINARY_CLASSIFICATION, Task.MULTI_CLASS_CLASSIFICATION}:
        raise ValueError(
            f'Data expected to be predictions of a binary or multi-class classification task; '
            f'found task to be {data.task}.'
        )


def _check_multilabel_clf_preds(data: ClassificationPredictions):
    _check_clf_preds(data)
    if data.task is not Task.MULTI_LABEL_CLASSIFICATION:
        raise ValueError(
            f'Data expected to be predictions of a multi-label classification task; '
            f'found task to be {data.task}.'
        )


def _check_labelspace(labelspace: Optional[Set[str]], data: ClassificationPredictions) -> Set[str]:
    _check_clf_preds(data)
    if labelspace is not None and labelspace != set(data.labelspace):
        raise ValueError(
            f'Cannot compute classification error counts on data with different labels; '
            f'we have so far been calculating on labels: {labelspace}, but input data has '
            f'labels: {data.labelspace}'
        )
    return set(data.labelspace)


"""
  ___  _                          __  __       _         _         
 | _ )(_) _ _   __ _  _ _  _  _  |  \/  | ___ | |_  _ _ (_) __  ___
 | _ \| || ' \ / _` || '_|| || | | |\/| |/ -_)|  _|| '_|| |/ _|(_-<
 |___/|_||_||_|\__,_||_|   \_, | |_|  |_|\___| \__||_|  |_|\__|/__/
                           |__/                                                                  
"""


class BinaryClassificationErrorCount(TabularMetric):
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    total: int = 0

    labelspace: Optional[Set[str]] = None

    def update(self, data: ClassificationPredictions):
        _check_binary_clf_preds(data)
        self.labelspace: Set[str] = _check_labelspace(self.labelspace, data)
        top_k_predictions: TopKClassificationPredictions = data.to_top_k()
        positive_lb: Any = data.positive_label
        negative_lb: Any = data.negative_label
        counts = top_k_predictions.data.pandas().groupby(
            [data.ground_truth_label_col_name, TOP_1_PREDICTED_LABEL]
        ).size()
        for (gt_lb, pred_lb), count in counts.to_dict().items():
            self.total += count
            if gt_lb == pred_lb == positive_lb:
                self.tp += count
            elif gt_lb == pred_lb == negative_lb:
                self.tn += count
            elif gt_lb == positive_lb and pred_lb == negative_lb:
                self.fn += count
            elif gt_lb == negative_lb and pred_lb == positive_lb:
                self.fp += count
            else:
                raise ValueError(
                    f'Unrecognized labels in predictions: actual={gt_lb}, predicted={pred_lb}; '
                    f'expected one of the following (binary) labels: positive={positive_lb}, negative={negative_lb}'
                )

    def compute(self) -> Any:
        return {
            'tp': self.tp,
            'fp': self.fp,
            'tn': self.tn,
            'fn': self.fn,
        }


class Prevalence(PercentageMetric):
    ## What is the true rate of the positive class?
    _error_count: BinaryClassificationErrorCount = BinaryClassificationErrorCount()

    def update(self, data: ClassificationPredictions):
        _check_binary_clf_preds(data)
        self._error_count.update(data)

    def compute(self) -> Any:
        return (self._error_count.tp + self._error_count.fn) / self._error_count.total


class Precision(PercentageMetric):
    ## From the examples when we predict positive, how often are we correct?
    aliases = ['Positive Predictive Value', 'PPV']

    _error_count: BinaryClassificationErrorCount = BinaryClassificationErrorCount()

    def update(self, data: ClassificationPredictions):
        _check_binary_clf_preds(data)
        self._error_count.update(data)

    def compute(self) -> Any:
        return self._error_count.tp / (self._error_count.tp + self._error_count.fp)


class FalseDiscoveryRate(PercentageMetric):
    ## From the examples when we predict positive, how often are we incorrect (i.e. false positive)?
    ## Equal to 1 - Precision.
    aliases = ['FDR']

    _error_count: BinaryClassificationErrorCount = BinaryClassificationErrorCount()

    def update(self, data: ClassificationPredictions):
        _check_binary_clf_preds(data)
        self._error_count.update(data)

    def compute(self) -> Any:
        return self._error_count.fp / (self._error_count.tp + self._error_count.fp)


class NegativePredictiveValue(PercentageMetric):
    ## From the examples when we predict negative, how often are we correct?
    aliases = ['Negative Predictive Value', 'NPV']

    _error_count: BinaryClassificationErrorCount = BinaryClassificationErrorCount()

    def update(self, data: ClassificationPredictions):
        _check_binary_clf_preds(data)
        self._error_count.update(data)

    def compute(self) -> Any:
        return self._error_count.tn / (self._error_count.tn + self._error_count.fn)


class FalseOmissionRate(PercentageMetric):
    ## From the examples when we predict negative, how often are we incorrect (i.e. false negative)?
    ## Equal to 1-NPV.
    aliases = ['FOR']

    _error_count: BinaryClassificationErrorCount = BinaryClassificationErrorCount()

    def update(self, data: ClassificationPredictions) -> Any:
        _check_binary_clf_preds(data)
        self._error_count.update(data)

    def compute(self) -> Any:
        return self._error_count.fn / (self._error_count.fn + self._error_count.tn)


class Recall(PercentageMetric):
    ## From the examples when the result was actually positive, what fraction did we catch?
    aliases = ['Sensitivity', 'True Positive Rate', 'TPR']

    _error_count: BinaryClassificationErrorCount = BinaryClassificationErrorCount()

    def update(self, data: ClassificationPredictions) -> Any:
        _check_binary_clf_preds(data)
        self._error_count.update(data)

    def compute(self) -> Any:
        return self._error_count.tp / (self._error_count.tp + self._error_count.fn)


class FalseNegativeRate(PercentageMetric):
    ## From the examples when the result was actually positive, what fraction did we miss (i.e. false negative)?
    ## Equal to 1-Recall.
    aliases = ['FNR']

    _error_count: BinaryClassificationErrorCount = BinaryClassificationErrorCount()

    def update(self, data: ClassificationPredictions) -> Any:
        _check_binary_clf_preds(data)
        self._error_count.update(data)

    def compute(self) -> Any:
        return self._error_count.fn / (self._error_count.tp + self._error_count.fn)


class Specificity(PercentageMetric):
    ## From the examples when the result was actually negative, what fraction did we catch?
    aliases = ['True Negative Rate', 'TNR']

    _error_count: BinaryClassificationErrorCount = BinaryClassificationErrorCount()

    def update(self, data: ClassificationPredictions) -> Any:
        _check_binary_clf_preds(data)
        self._error_count.update(data)

    def compute(self) -> Any:
        return self._error_count.tn / (self._error_count.tn + self._error_count.fp)


class FalsePositiveRate(PercentageMetric):
    ## From the examples when the result was actually negative, what fraction did we miss (i.e. false positive)?
    ## Equal to 1-TNR i.e. 1-Specificity
    aliases = ['FPR']

    _error_count: BinaryClassificationErrorCount = BinaryClassificationErrorCount()

    def update(self, data: ClassificationPredictions) -> Any:
        _check_binary_clf_preds(data)
        self._error_count.update(data)

    def compute(self) -> Any:
        return self._error_count.fp / (self._error_count.tn + self._error_count.fp)


class Informedness(PercentageMetric):
    ## Sensitivity + Specificity - 1, i.e. Recall + Specificity - 1, i.e. TPR + TNR - 1
    aliases = ["Youden's J statistic", "Youden's J", "Youden J statistic", "Youden's statistic", "Youden statistic"]

    _error_count: BinaryClassificationErrorCount = BinaryClassificationErrorCount()

    def update(self, data: ClassificationPredictions) -> Any:
        _check_binary_clf_preds(data)
        self._error_count.update(data)

    def compute(self) -> Any:
        return (self._error_count.tp / (self._error_count.tp + self._error_count.fn)) + \
               (self._error_count.tn / (self._error_count.tn + self._error_count.fp)) - 1


class FBetaScore(PercentageMetric):
    aliases = ['FBeta', 'FScore']

    class Params(PercentageMetric.Params):
        beta: confloat(ge=0)  ## Recall is considered "β" times more important than Precision.

    _error_count: BinaryClassificationErrorCount = BinaryClassificationErrorCount()

    def update(self, data: ClassificationPredictions) -> Any:
        _check_binary_clf_preds(data)
        self._error_count.update(data)

    def compute(self) -> Any:
        ## FBeta = (1+β^2) * (P*R)/(β^2*P + R) = (1+β^2)*TP/((1+β^2)*TP + β^2*FN + FP)
        beta_square = self.params.beta ** 2
        beta_square_plus_1 = beta_square + 1
        return (beta_square_plus_1 * self._error_count.tp) / (
                beta_square_plus_1 * self._error_count.tp +
                beta_square * self._error_count.fn +
                self._error_count.fp
        )


class F1Score(FBetaScore):
    aliases = ['F1']

    class Params(FBetaScore.Params):
        beta: Literal[1] = 1


"""
  __  __        _  _    _           _                 __  __       _         _         
 |  \/  | _  _ | || |_ (_) ___  __ | | __ _  ___ ___ |  \/  | ___ | |_  _ _ (_) __  ___
 | |\/| || || || ||  _|| ||___|/ _|| |/ _` |(_-<(_-< | |\/| |/ -_)|  _|| '_|| |/ _|(_-<
 |_|  |_| \_,_||_| \__||_|     \__||_|\__,_|/__//__/ |_|  |_|\___| \__||_|  |_|\__|/__/
"""


class ConfusionMatrix(TabularMetric):
    """Confusion matrix for binary, multi-class tasks."""

    aliases = ['confusion', 'confusion_matrix']
    ## Sparse confusion matrix. Tuple is (gt, predicted) label
    confusion_matrix: Dict[Tuple[Any, Any], int] = defaultdict(lambda: 0)
    total: int = 0

    labelspace: Optional[Set[str]] = None

    def update(self, data: ClassificationPredictions) -> Any:
        ## TODO: support multilabel.
        ## Ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html
        _check_binary_or_multiclass_clf_preds(data)
        self.labelspace: Set[str] = _check_labelspace(self.labelspace, data)
        if data.task in {Task.BINARY_CLASSIFICATION, Task.MULTI_CLASS_CLASSIFICATION}:
            top_k_predictions: TopKClassificationPredictions = data.to_top_k()
            counts: Dict[Tuple[str, str], int] = top_k_predictions.data.pandas().groupby(
                [data.ground_truth_label_col_name, TOP_1_PREDICTED_LABEL]
            ).size().to_dict()
        elif data.task is Task.MULTI_LABEL_CLASSIFICATION:
            labelwise_predictions: LabelwiseClassificationPredictions = data.to_labelwise()
            multi_label_preds: pd.Series = labelwise_predictions.predictions(multilabel=True).pandas()
            multi_labels_df: pd.DataFrame = pd.DataFrame({
                'multi_label_predictions': multi_label_preds,
                data.ground_truth_label_col_name: data.data[data.ground_truth_label_col_name].pandas(),
            })
            counts: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], int] = multi_labels_df.groupby(
                [data.ground_truth_label_col_name, 'multi_label_predictions']
            ).size().to_dict()
        else:
            raise NotImplementedError(f'Cannot create confusion matrix from {type_str(data)} with task "{data.task}".')
        for (gt_lbs, pred_lbs), count in counts.items():
            for gt_lb in as_tuple(gt_lbs):  ## It is a list during multi-label
                for pred_lb in as_tuple(pred_lbs):  ## It is a list during multi-label
                    self.confusion_matrix[(gt_lb, pred_lb)] = self.confusion_matrix.get((gt_lb, pred_lb), 0) + count
                    self.total += count

    def compute(self) -> Any:
        labels: List[str] = sorted(self.labelspace)
        cf_mat: pd.DataFrame = pd.DataFrame(np.zeros((len(labels), len(labels)))).astype(int)
        cf_mat.columns = labels
        cf_mat.index = labels
        cf_mat.T.index.name = 'Predicted Label→'  ## Columns
        cf_mat.index.name = 'Ground-Truth Label↓'  ## Rows
        for i, gt_lb in enumerate(labels):
            for j, pred_lb in enumerate(labels):
                cf_mat.iloc[i, j] = self.confusion_matrix.get((gt_lb, pred_lb), 0)
        return cf_mat

    def predicted_label_counts(self) -> Dict[str, int]:
        pred_lb_counts: Dict[str, int] = {lb: 0 for lb in self.labelspace}
        for (gt_lb, pred_lb), count in self.confusion_matrix.items():
            pred_lb_counts[pred_lb] += count
        return pred_lb_counts

    def ground_truth_label_counts(self) -> Dict[str, int]:
        gt_lb_counts: Dict[str, int] = {lb: 0 for lb in self.labelspace}
        for (gt_lb, pred_lb), count in self.confusion_matrix.items():
            gt_lb_counts[gt_lb] += count
        return gt_lb_counts

    def true_positive_counts(self) -> Dict[str, int]:
        tp_counts: Dict[str, int] = {lb: 0 for lb in self.labelspace}
        for (gt_lb, pred_lb), count in self.confusion_matrix.items():
            if gt_lb == pred_lb:
                tp_counts[pred_lb] += count
        return tp_counts

    def false_positive_counts(self) -> Dict[str, int]:
        fp_counts: Dict[str, int] = {lb: 0 for lb in self.labelspace}
        for (gt_lb, pred_lb), count in self.confusion_matrix.items():
            if gt_lb != pred_lb:
                ## When something is not on the diagonal, it is a "False positive" for the predicted label.
                fp_counts[pred_lb] += count
        return fp_counts

    def false_negative_counts(self) -> Dict[str, int]:
        fn_counts: Dict[str, int] = {lb: 0 for lb in self.labelspace}
        for (gt_lb, pred_lb), count in self.confusion_matrix.items():
            if gt_lb != pred_lb:
                ## When something is not on the diagonal, it is a "False negative" for the ground-truth label.
                fn_counts[gt_lb] += count
        return fn_counts


class _AggregatedClassificationMetric(AggregatedPercentageMetric, ABC):
    class Params(PercentageMetric.Params):
        aggregation: AggregationStrategy = AggregationStrategy.AVERAGE

    _cf_mat: ConfusionMatrix = ConfusionMatrix()

    def update(self, data: ClassificationPredictions) -> Any:
        self._cf_mat.update(data)

    def compute(self) -> Any:
        labelwise_metric_values: Dict[str, float] = self._from_cf_mat(self._cf_mat)
        if self.params.aggregation is AggregationStrategy.NONE:
            return labelwise_metric_values
        return self._aggregate(list(labelwise_metric_values.values()))

    @abstractmethod
    def _from_cf_mat(self, cf_mat: ConfusionMatrix) -> Dict[str, float]:
        pass


class MacroPrecision(_AggregatedClassificationMetric):
    ## Macro-precision: take the precision of each class, and combine it via:
    ## (a) averaging i.e. Macro-averaged Precision
    ## (b) median i.e. Macro-median Precision
    ## etc.
    ## Remember that "precision" is: from examples predicted to be class X, what fraction are actually class X?
    def _from_cf_mat(self, cf_mat: ConfusionMatrix) -> Dict[str, float]:
        tp_counts: Dict[str, int] = cf_mat.true_positive_counts()
        pred_lb_counts: Dict[str, int] = cf_mat.predicted_label_counts()
        labelwise_macro_precision: Dict[str, float] = {
            lb: tp_counts[lb] / pred_lb_counts[lb]
            for lb in tp_counts.keys()
            if pred_lb_counts[lb] > 0
        }
        return labelwise_macro_precision


class MicroPrecision(_AggregatedClassificationMetric):
    ## Micro-precision is just accuracy i.e. (sum of TP) / ((TP_i+FP_i) for i in classes)
    ## This is because for a particular class "i", (TP_i+FP_i) is just the number of examples predicted for that class.
    ## So, ((TP_i+FP_i) for i in classes) becomes the total number of examples.
    ## Here, only average makes sense.
    ## Note: Micro-averaged precision, micro-averaged recall and micro-averaged F1 are all equal to the Accuracy when
    ## each data point is assigned to exactly one class (i.e. binary or multi-class classification).

    class Params(PercentageMetric.Params):
        aggregation: Literal[AggregationStrategy.AVERAGE] = AggregationStrategy.AVERAGE

    def _from_cf_mat(self, cf_mat: ConfusionMatrix) -> Dict[str, float]:
        numer: int = sum(cf_mat.true_positive_counts().values())
        denom: int = numer + sum(cf_mat.false_positive_counts().values())
        return {'all_labels': float(numer / denom)}


class MacroRecall(_AggregatedClassificationMetric):
    ## Macro-recall: take the Recall of each class, and combine it via:
    ## (a) averaging i.e. Macro-averaged Recall
    ## (b) median i.e. Macro-median Recall
    ## etc.
    ## Remember that "recall" is: from examples which are actuall class X, what fraction did we catch (i.e. correctly
    ## identify as class X)?
    def _from_cf_mat(self, cf_mat: ConfusionMatrix) -> Dict[str, float]:
        tp_counts: Dict[str, int] = cf_mat.true_positive_counts()
        gt_lb_counts: Dict[str, int] = cf_mat.ground_truth_label_counts()
        labelwise_macro_recall: Dict[str, float] = {
            lb: tp_counts[lb] / gt_lb_counts[lb]
            for lb in tp_counts.keys()
            if gt_lb_counts[lb] > 0
        }
        return labelwise_macro_recall


class MicroRecall(_AggregatedClassificationMetric):
    ## Micro-recall is (sum of TP) / ((TP_i+FN_i) for i in classes)
    ## This is because for a particular class "i", (TP_i+FN_i) is just the number of examples which actually belong
    ## to that class.
    ## Here, only average makes sense.
    ## Note: Micro-averaged precision, micro-averaged recall and micro-averaged F1 are all equal to the Accuracy when
    ## each data point is assigned to exactly one class (i.e. binary or multi-class classification).

    class Params(PercentageMetric.Params):
        aggregation: Literal[AggregationStrategy.AVERAGE] = AggregationStrategy.AVERAGE

    def _from_cf_mat(self, cf_mat: ConfusionMatrix) -> Dict[str, float]:
        numer: int = sum(cf_mat.true_positive_counts().values())
        denom: int = numer + sum(cf_mat.false_negative_counts().values())
        return {'all_labels': float(numer / denom)}


class MacroFBeta(_AggregatedClassificationMetric):
    aliases = ['FBetaMacro']

    ## Macro-F_β: take the F_β of each class, and combine it via:
    ## (a) averaging i.e. Macro-averaged F_β
    ## (b) median i.e. Macro-median F_β
    ## etc.
    ## Remember that "F_β" is: (1+β^2)*TP_i/((1+β^2)*TP_i + β^2*FN_i + FP_i)
    class Params(_AggregatedClassificationMetric.Params):
        beta: confloat(ge=0)  ## Recall is considered "β" times more important than Precision.

    def _from_cf_mat(self, cf_mat: ConfusionMatrix) -> Dict[str, float]:
        tp_counts: Dict[str, int] = cf_mat.true_positive_counts()
        fp_counts: Dict[str, int] = cf_mat.false_positive_counts()
        fn_counts: Dict[str, int] = cf_mat.false_negative_counts()
        beta_square = self.params.beta ** 2
        beta_square_plus_1 = beta_square + 1

        labelwise_fbeta: Dict[str, float] = {
            lb: (beta_square_plus_1 * tp_counts[lb]) / (
                    beta_square_plus_1 * tp_counts[lb] +
                    beta_square * fn_counts[lb] +
                    fp_counts[lb]
            )
            for lb in tp_counts.keys()
            if fp_counts[lb] > 0 or fn_counts[lb] > 0 or tp_counts[lb] > 0
        }
        return labelwise_fbeta


class MacroF1(MacroFBeta):
    aliases = ['F1Macro']

    ## F1 is 2*TP_i/(2*TP_i + FN_i + FP_i)
    class Params(MacroFBeta.Params):
        beta: Literal[1] = 1


class MicroFBeta(_AggregatedClassificationMetric):
    ## Micro-FBeta is (1+β^2)*TP_i/((1+β^2)*TP_i + β^2*FN_i + FP_i)
    ## Here, only averaging makes sense.
    ## Note: Micro-averaged precision, micro-averaged recall and micro-averaged F1 are all equal to the Accuracy when
    ## each data point is assigned to exactly one class (i.e. binary or multi-class classification).

    class Params(_AggregatedClassificationMetric.Params):
        aggregation: Literal[AggregationStrategy.AVERAGE] = AggregationStrategy.AVERAGE
        beta: confloat(ge=0)  ## Recall is considered "β" times more important than Precision.

    def _from_cf_mat(self, cf_mat: ConfusionMatrix) -> Dict[str, float]:
        tp: int = sum(cf_mat.true_positive_counts().values())
        fp: int = sum(cf_mat.false_positive_counts().values())
        fn: int = sum(cf_mat.false_negative_counts().values())
        beta_square = self.params.beta ** 2
        beta_square_plus_1 = beta_square + 1
        fbeta: float = (beta_square_plus_1 * tp) / (
                beta_square_plus_1 * tp +
                beta_square * fn +
                fp
        )
        return {'all_labels': fbeta}


class MicroF1(MicroFBeta):
    ## F1 is 2*TP_i/(2*TP_i + FN_i + FP_i)
    class Params(MicroFBeta.Params):
        beta: Literal[1] = 1


class Accuracy(PercentageMetric):
    _num_correct_examples: int = 0
    _num_total_examples: int = 0

    def update(self, data: ClassificationPredictions):
        data.has_ground_truths()
        data.has_predictions()
        data: ClassificationPredictions = data.to_top_k()
        correct: pd.Series = (data.top_1_predicted_label() == data.ground_truths()).pandas().value_counts()
        self._num_total_examples += correct.sum()
        self._num_correct_examples += correct.get(True, default=0)

    def compute(self) -> float:
        return float(self._num_correct_examples / self._num_total_examples)

    def merge(self, other: 'Accuracy') -> 'Accuracy':
        merged_metric: Accuracy = self.clear()
        merged_metric._num_correct_examples = self._num_correct_examples + other._num_correct_examples
        merged_metric._num_total_examples = self._num_total_examples + other._num_total_examples
        merged_metric.value = merged_metric.calculate()
        return merged_metric


class LabelwiseAccuracy(TabularMetric):
    aliases = ['classwise_accuracy']

    _labelspace: Optional[Tuple[str, ...]] = None
    _num_correct_examples: Dict[str, int] = dict()
    _num_total_examples: Dict[str, int] = dict()

    def update(self, data: ClassificationPredictions):
        _check_clf_preds(data)
        if self._labelspace is None:
            self._labelspace = data.labelspace
        assert self._labelspace == data.labelspace
        for lb in self._labelspace:
            self._num_total_examples.setdefault(lb, 0)
            self._num_correct_examples.setdefault(lb, 0)

        data: ClassificationPredictions = data.to_top_k()
        gt = data.ground_truths().pandas()
        for lb, lb_count in gt.value_counts().to_dict().items():
            self._num_total_examples[lb] += lb_count
        correct: pd.Series = (data.top_1_predicted_label() == data.ground_truths()).pandas()
        gt_correct = gt[correct]
        for lb, correct_lb_count in gt_correct.value_counts().to_dict().items():
            self._num_correct_examples[lb] += correct_lb_count

    def compute(self) -> Dict[str, confloat(ge=0.0, le=1.0)]:
        return {
            lb: self._num_correct_examples[lb] / self._num_total_examples[lb]
            for lb in self._labelspace
            if self._num_total_examples[lb] > 0
        }


class DatasetCartography(TabularMetric):
    aliases = ['DataMap']

    """
    Implementation of paper "Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics"
    by Swayamdipta et. al. (2020): https://arxiv.org/abs/2009.10795
    """
    IDX: ClassVar[str] = 'idx'
    GOLD_LABEL: ClassVar[str] = 'gold_label'
    GOLD_LABEL_PROB: ClassVar[str] = 'gold_label_prob'
    ALL_LABEL_PROBS: ClassVar[str] = 'all_label_probs'
    PREDICTED_LABEL: ClassVar[str] = 'predicted_label'
    PREDICTED_LABEL_MATCHES_GOLD: ClassVar[str] = 'predicted_label_matches_gold'
    CONFIDENCE: ClassVar[str] = 'confidence'
    VARIABILITY: ClassVar[str] = 'variability'
    CORRECTNESS: ClassVar[str] = 'correctness'
    CORRECTNESS_BUCKET: ClassVar[str] = 'correctness_bucket'
    CORRECTNESS_MARKER: ClassVar[str] = 'correctness_marker'
    CORRECTNESS_COLOR: ClassVar[str] = 'correctness_color'

    _labelspace: Optional[Tuple[str, ...]] = None
    _idx_col_name: Optional[str] = None
    _gt_col_name: Optional[str] = None
    _metric_data: Dict[Any, Dict] = {}

    def update(self, data: ClassificationPredictions):
        _check_clf_preds(data)
        data: ClassificationPredictions = data.to_labelwise()

        if self._labelspace is None:
            self._labelspace = data.labelspace
        assert self._labelspace == data.labelspace

        if self._idx_col_name is None:
            self._idx_col_name = data.data_schema.index_col
        assert self._idx_col_name == data.data_schema.index_col

        if self._gt_col_name is None:
            self._gt_col_name = data.ground_truth_label_col_name
        assert self._gt_col_name == data.ground_truth_label_col_name

        for row in data.data.to_list_of_dict():
            idx: str = row[self._idx_col_name]
            if idx in self._metric_data:
                raise ValueError(f'Found duplicate index in data: {self._idx_col_name}={idx}')
            self._metric_data[idx] = {}

            gold_lb: str = row[self._gt_col_name]
            check_isinstance(gold_lb, str)
            gold_lb_prob: float = float(row[gold_lb])
            self._metric_data[idx][self.GOLD_LABEL] = gold_lb
            self._metric_data[idx][self.GOLD_LABEL_PROB] = gold_lb_prob

            self._metric_data[idx][self.ALL_LABEL_PROBS]: Dict[str, float] = {
                lb: float(row[lb])
                for lb in self._labelspace
            }
            pred_lb: str = argmax(self._metric_data[idx][self.ALL_LABEL_PROBS])  ## Gets label with max prob
            check_isinstance(pred_lb, str)
            self._metric_data[idx][self.PREDICTED_LABEL] = pred_lb
            self._metric_data[idx][self.PREDICTED_LABEL_MATCHES_GOLD] = bool(pred_lb == gold_lb)

    def compute(self) -> pd.DataFrame:
        data_map_single_epoch: pd.DataFrame = pd.DataFrame([
            {
                self.IDX: idx,
                **d,
            }
            for idx, d in self._metric_data.items()
        ])
        return data_map_single_epoch

    @classmethod
    def calc_data_map(cls, detailed_final_model_metrics: pd.DataFrame, *, index_col: str) -> pd.DataFrame:
        correctness_marker_map = {
            0.0: 'circle',
            0.1: 'circle',
            0.2: 'diamond',
            0.3: 'x',
            0.4: 'x',
            0.5: 'square',
            0.6: 'cross',
            0.7: 'cross',
            0.8: 'triangle-down',
            0.9: 'triangle-up',
            1.0: 'triangle-up',
        }
        correctness_color_map = {
            0.0: '#64b5f6',
            0.1: '#64b5f6',
            0.2: '#1976d2',
            0.3: '#053061',
            0.4: '#525252',
            0.5: '#525252',
            0.6: '#525252',
            0.7: '#662506',
            0.8: '#c62828',
            0.9: '#ef5350',
            1.0: '#ef5350',
        }
        ## Take the longest:
        longest_trial_id: str = dict_key_with_best_value({
            trial_id: int(trial_df[_RAY_TRAINING_ITERATION].nunique())
            for trial_id, trial_df in detailed_final_model_metrics.groupby(_RAY_TRIAL_ID)
        }, how='max')

        trial_cart_metrics: List[Dict] = []
        for trial_id, trial_df in detailed_final_model_metrics.groupby(_RAY_TRIAL_ID):
            if trial_id != longest_trial_id:
                continue
            trial_df: pd.DataFrame = trial_df.sort_values(_RAY_TRAINING_ITERATION, ascending=True).reset_index(
                drop=True)
            trial_cart_df: List[pd.DataFrame] = []
            for training_iteration, iter_data_map_df in trial_df.set_index(_RAY_TRAINING_ITERATION)[
                f'{DataSplit.TRAIN.capitalize()}/{DatasetCartography.class_name}'
            ].to_dict().items():
                iter_data_map_df[_RAY_TRAINING_ITERATION] = training_iteration
                trial_cart_df.append(iter_data_map_df)
            trial_cart_df: pd.DataFrame = pd.concat(trial_cart_df).sort_values(
                [index_col, _RAY_TRAINING_ITERATION],
                ascending=True,
            ).reset_index(drop=True)
            for idx, idx_df in trial_cart_df.groupby(index_col):
                trial_cart_metrics.append({
                    index_col: idx,
                    cls.CONFIDENCE: float(idx_df[cls.GOLD_LABEL_PROB].mean()),
                    cls.VARIABILITY: float(idx_df[cls.GOLD_LABEL_PROB].std(ddof=0)),  ## Biased std for some reason?
                    cls.CORRECTNESS: float(idx_df[cls.PREDICTED_LABEL_MATCHES_GOLD].mean()),
                })
            break  ## Only take first one
        trial_cart_metrics: pd.DataFrame = pd.DataFrame(trial_cart_metrics)
        trial_cart_metrics[cls.CORRECTNESS_BUCKET] = trial_cart_metrics[cls.CORRECTNESS].apply(lambda x: round(x, 1))
        # print(json.dumps(trial_cart_metrics.query('confidence <= 0.4')['idx'].tolist(), indent=2), end='\n\n\n')
        trial_cart_metrics[cls.CORRECTNESS_MARKER] = trial_cart_metrics[cls.CORRECTNESS_BUCKET].map(
            correctness_marker_map
        )
        trial_cart_metrics[cls.CORRECTNESS_COLOR] = trial_cart_metrics[cls.CORRECTNESS_BUCKET].map(
            correctness_color_map
        )
        return trial_cart_metrics

    @classmethod
    def plot_data_map(cls, detailed_final_model_metrics: pd.DataFrame, *, index_col: str) -> Any:
        data_map_df: pd.DataFrame = cls.calc_data_map(detailed_final_model_metrics, index_col=index_col)
        return data_map_df.hvplot.scatter(
            x=cls.VARIABILITY, y=cls.CONFIDENCE,
            c=cls.CORRECTNESS_COLOR,
            marker=cls.CORRECTNESS_MARKER,
            size=30,
        ).opts(
            width=300,
            height=300,
            # xlim=(0, 1),
            ylim=(0, 1),
        )
