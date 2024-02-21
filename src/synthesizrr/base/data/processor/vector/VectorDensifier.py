from typing import *
from synthesizrr.base.util import is_null
from synthesizrr.base.data.processor import SingleColumnProcessor, SparseVectorInputProcessor, VectorOutputProcessor
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix as SparseCSRMatrix


class VectorDensifier(SingleColumnProcessor, SparseVectorInputProcessor, VectorOutputProcessor):
    """Converts a sparse vector column into a dense vector column. Each dense vector is a 1d numpy array."""

    class Params(SingleColumnProcessor.Params):
        output_list: bool = False

    def transform_single(self, data: SparseCSRMatrix) -> Optional[np.ndarray]:
        if is_null(data):
            return None
        if not isinstance(data, SparseCSRMatrix):
            raise ValueError(f'{str(self.__class__)} can only densify SparseCSRMatrix objects')
        data: np.ndarray = data.toarray()
        if len(data.shape) != 2:
            raise ValueError(f'Each SparseCSRMatrix to densify must have two dimensions. Found: {len(data.shape)} dims')
        if data.shape[0] != 1:
            raise ValueError(f'Each SparseCSRMatrix to densify must have exactly 1 row. Found: {data.shape[0]} rows')
        data: np.ndarray = data[0]
        if self.params.output_list:
            data: List = list(data)
        return data
