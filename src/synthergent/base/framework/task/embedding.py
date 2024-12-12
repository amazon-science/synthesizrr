from typing import *
from abc import ABC, abstractmethod
import numpy as np
from synthergent.base.util import is_list_like
from synthergent.base.data import ScalableDataFrame, ScalableSeries, ScalableSeriesRawType, ScalableDataFrameRawType
from synthergent.base.framework import Algorithm, Dataset, Predictions
from synthergent.base.constants import Task, MLType, MLTypeSchema, DataLayout


class EmbeddingData(Dataset):
    tasks = Task.EMBEDDING
    ground_truths_schema = {}  ## No ground-truths


GENERATED_EMBEDDINGS_FORMAT_MSG: str = f"""
 Embeddings returned by algorithm must be a column of vectors.
 """.strip()

EMBEDDINGS_COL: str = 'embeddings'


class Embeddings(Predictions):
    tasks = Task.EMBEDDING
    ground_truths_schema = {}  ## No ground-truths
    predictions_schema = {
        EMBEDDINGS_COL: MLType.VECTOR,
    }

    @property
    def embeddings(self) -> ScalableSeries:
        predicted_emebeddings_col: str = next(iter(self.data_schema.predictions_schema.keys()))
        return self.data[predicted_emebeddings_col]


class Embedder(Algorithm, ABC):
    tasks = Task.EMBEDDING
    inputs = EmbeddingData
    outputs = Embeddings

    def _create_predictions(
            self,
            batch: Dataset,
            predictions: Union[List, np.ndarray],
            **kwargs
    ) -> Embeddings:
        if not is_list_like(predictions):
            raise ValueError(GENERATED_EMBEDDINGS_FORMAT_MSG)
        embeddings: Dict = {EMBEDDINGS_COL: predictions}
        return Embeddings.from_task_data(
            data=batch,
            predictions=embeddings,
            **kwargs
        )
