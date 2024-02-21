from typing import *
from synthesizrr.base.framework import Visualization, ClassificationPredictions
from pydantic import confloat

#
# class PrecisionRecallCurve(Visualization):
#     aliases = ['PRCurve', 'precision_recall_curve', 'pr_curve']
#
#     def plot(self, data: ClassificationPredictions) -> confloat(ge=0.0, le=1.0):
#         predictions.has_ground_truths()
#         predictions.has_predictions()
#         correct = (predictions.get_predicted_label_cols(top_k=1) == predictions.get_ground_truth_label_cols(top_k=1))
#         self._num_total_examples += len(correct)
#         self._num_correct_examples += correct.sum()
#         metric_val: float = float(round(self._num_correct_examples / self._num_total_examples, self.params.display_decimals))
#         return metric_val
