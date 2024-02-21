from typing import *
from abc import ABC, abstractmethod
import numpy as np
from synthesizrr.base.util import is_list_like
from synthesizrr.base.data import ScalableDataFrame, ScalableSeries, ScalableSeriesRawType, ScalableDataFrameRawType
from synthesizrr.base.framework import Algorithm, Dataset, Predictions
from synthesizrr.base.constants import Task, MLType, MLTypeSchema, DataLayout


class RegressionData(Dataset):
    tasks = Task.REGRESSION

    ground_truths_schema = {
        '{ground_truth_score_col_name}': MLType.FLOAT,
    }


REGRESSION_PREDICTIONS_FORMAT_MSG: str = f"""
Regression predictions returned by algorithm must be a column of scores.
This can be a list, tuple, Numpy array, Pandas Series, etc.
""".strip()


class RegressionPredictions(Predictions):
    tasks = Task.REGRESSION

    ground_truths_schema = {
        '{ground_truth_score_col_name}': MLType.FLOAT,
    }

    predictions_schema = {
        'predicted_score': MLType.FLOAT,
    }

    @property
    def predicted_scores(self) -> ScalableSeries:
        predicted_scores_col: str = next(iter(self.data_schema.predictions_schema.keys()))
        return self.data[predicted_scores_col]

    @property
    def ground_truth_scores(self) -> ScalableSeries:
        assert self.has_ground_truths
        ground_truth_scores_col: str = next(iter(self.data_schema.ground_truths_schema.keys()))
        return self.data[ground_truth_scores_col]


class Regressor(Algorithm, ABC):
    tasks = Task.REGRESSION
    inputs = RegressionData
    outputs = RegressionPredictions

    def _create_predictions(
            self,
            batch: Dataset,
            predictions: Union[List, np.ndarray],
            **kwargs
    ) -> RegressionPredictions:
        if not is_list_like(predictions):
            raise ValueError(REGRESSION_PREDICTIONS_FORMAT_MSG)
        predictions: Dict = {'predicted_score': predictions}
        return RegressionPredictions.from_task_data(
            data=batch,
            predictions=predictions,
            **kwargs
        )
