from typing import *
import re
import numpy as np
from scipy.sparse import csr_matrix as SparseCSRMatrix
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from synthergent.base.constants import MLType
from synthergent.base.util import if_else
from synthergent.base.data.sdf import ScalableSeries, ScalableSeriesRawType
from synthergent.base.data.processor import SingleColumnProcessor, TextInputProcessor
from synthergent.base.data.processor.vector.VectorDensifier import VectorDensifier
from pydantic import root_validator, validator


class TFIDFVectorization(SingleColumnProcessor, TextInputProcessor):
    """
    Performs TF-IDF Vectorization of a text column using sklearn's TFIDFVectorizer.
    Ref: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    Params:
    - OUTPUT_SPARSE: whether to output each row as a sparse row matrix (1 x N). If False, will output a 1d numpy array.
    - SKLEARN_PARAMS: dictionary of sklearn params to be unpacked as keyword arguments to the constructor
        sklearn.feature_extraction.text.TfidfVectorizer. Thus, keys are case-sensitive.
    """

    class Params(SingleColumnProcessor.Params):
        sklearn_params: Dict = {}
        output_sparse: bool = False

        @validator('sklearn_params', pre=True)
        def process_sklearn_tfidf_params(cls, sklearn_tfidf_params: Dict):
            token_pattern: Optional = sklearn_tfidf_params.get('token_pattern')
            if token_pattern is not None:
                sklearn_tfidf_params['token_pattern'] = str(sklearn_tfidf_params.get('token_pattern'))
            ngram_range: Optional = sklearn_tfidf_params.get('ngram_range')
            if ngram_range is not None:
                if isinstance(ngram_range, str):
                    ngram_range = literal_eval(ngram_range)
                if isinstance(ngram_range, list):
                    ngram_range = tuple(ngram_range)
                assert isinstance(ngram_range, tuple)
                sklearn_tfidf_params['ngram_range'] = ngram_range
            return sklearn_tfidf_params

    output_mltype = MLType.VECTOR
    vectorizer: TfidfVectorizer = None
    vector_densifier: VectorDensifier = None

    @root_validator(pre=False)
    def set_vectorizer(cls, params: Dict):
        params['vectorizer']: TfidfVectorizer = TfidfVectorizer(**params['params'].sklearn_params)
        params['vector_densifier']: VectorDensifier = VectorDensifier()
        params['output_mltype']: MLType = if_else(
            params['params'].output_sparse,
            MLType.SPARSE_VECTOR,
            MLType.VECTOR
        )
        return params

    def _fit_series(self, data: ScalableSeries):
        self.vectorizer.fit(data.pandas())  ## TODO: Super slow, replace with Dask TFIDF

    def transform_single(self, data: str) -> Union[SparseCSRMatrix, np.ndarray]:
        ## Will output a sparse matrix with only a single row.
        tfidf_vec: SparseCSRMatrix = self.vectorizer.transform([data])
        if not self.params.output_sparse:
            tfidf_vec: np.ndarray = self.vector_densifier.transform_single(tfidf_vec)
        return tfidf_vec
