from pydantic import confloat
from typing import *
import math
from decimal import Decimal
from fractions import Fraction
from synthesizrr.base.framework import Metric, RegressionPredictions
from functools import singledispatchmethod
import numpy as np


class MeanSquaredError(Metric):
    aliases = ['MEAN_SQUARED_ERROR', 'MEAN_SQUARE_ERROR', 'MSE']
    _sum_of_sq_diff: Decimal = 0.0
    _num_total_examples: int = 0

    def update(self, data: RegressionPredictions):
        data.has_ground_truths()
        data.has_predictions()
        self._num_total_examples += len(data.ground_truth_scores)
        ## Ref: https://stackoverflow.com/a/59128677
        self._sum_of_sq_diff: Decimal = Decimal(math.fsum([
            self._sum_of_sq_diff,
            Decimal(((data.ground_truth_scores - data.predicted_scores) ** 2).sum())
        ]))

    def compute(self) -> confloat(ge=0.0, le=1.0):
        ## Ref: https://www.laac.dev/blog/float-vs-decimal-python/
        metric_val: float = float(
            Fraction(self._sum_of_sq_diff) / Fraction(self._num_total_examples)
        )
        return metric_val


class MeanAbsoluteError(Metric):
    aliases = ['MEAN_ABSOLUTE_ERROR', 'MAE']
    _sum_of_abs_diff: Decimal = 0.0
    _num_total_examples: int = 0

    def update(self, data: RegressionPredictions):
        data.has_ground_truths()
        data.has_predictions()
        self._num_total_examples += len(data.ground_truth_scores)
        ## Ref: https://stackoverflow.com/a/59128677
        self._sum_of_abs_diff: Decimal = Decimal(math.fsum([
            self._sum_of_abs_diff,
            Decimal(
                (data.ground_truth_scores - data.predicted_scores).abs().sum()
            )
        ]))

    def compute(self) -> confloat(ge=0.0, le=1.0):
        ## Ref: https://www.laac.dev/blog/float-vs-decimal-python/
        metric_val: float = float(
            Fraction(self._sum_of_abs_diff) / Fraction(self._num_total_examples)
        )
        return metric_val
