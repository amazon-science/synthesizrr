from typing import *
from synthesizrr.base.data.processor import Nto1ColumnProcessor, VectorAssemblerInputProcessor, VectorOutputProcessor
import pandas as pd
import numpy as np
from synthesizrr.base.data.sdf import ScalableDataFrame, ScalableSeries
from synthesizrr.base.util import AutoEnum, auto, as_list, is_list_like, is_null
from synthesizrr.base.constants import MLType
from scipy.sparse import csr_matrix as SparseCSRMatrix


class InvalidBehavior(AutoEnum):
    ERROR = auto()
    KEEP = auto()


class VectorAssembler(Nto1ColumnProcessor, VectorAssemblerInputProcessor, VectorOutputProcessor):
    """
    Concatenates multiple columns into a single vector column

    Params:
    - HANDLE_INVALID: how to handle NaN values in columns
        - ERROR: Throws an error if invalid data/NaN value is present
        - KEEP: Keeps all the rows, ignores NaNs and Nones.
    """

    class Params(Nto1ColumnProcessor.Params):
        handle_invalid: InvalidBehavior = InvalidBehavior.KEEP

    def _transform_df(self, data: ScalableDataFrame) -> ScalableSeries:
        output_series: Optional[ScalableSeries] = None
        for col in sorted(list(data.columns)):
            if output_series is None:
                output_series: ScalableSeries = self._transform_series(data[col], col)
            else:
                output_series += self._transform_series(data[col], col)
        return output_series

    def _transform_series(self, data: ScalableSeries, col: str) -> ScalableSeries:
        feature_type: MLType = self.data_schema[col]
        if feature_type in {MLType.INT, MLType.FLOAT, MLType.VECTOR}:
            return data.apply(self._convert_to_list, col=col)
        elif feature_type is MLType.SPARSE_VECTOR:
            return data.apply(self._convert_sparse_vector_to_dense_vector, col=col)
        else:
            raise TypeError(f'{col} Column must be of type {self.input_mltypes}; found {feature_type}')

    def _convert_sparse_vector_to_dense_vector(self, vector: SparseCSRMatrix, col: str):
        if isinstance(vector, SparseCSRMatrix):
            dense_vector: np.ndarray = vector.toarray()[0]
        else:
            if self.params.handle_invalid is InvalidBehavior.ERROR:
                raise ValueError(f'Expected only SparseCSRMatrix in column "{col}", got a value of type {type(vector)}')
            dense_vector: Optional[np.ndarray] = None
        return self._convert_to_list(dense_vector, col)

    def _convert_to_list(self, val: Optional[Union[np.ndarray, List, Set, Tuple, Any]], col: str):
        ## Assumes the length of vectors are same throughout the column.
        if is_null(val) and self.params.handle_invalid is InvalidBehavior.ERROR:
            raise ValueError(f'Got empty value ({val}) in column: "{col}"')
        return as_list(val)