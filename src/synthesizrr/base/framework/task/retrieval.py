from typing import *
from abc import ABC, abstractmethod
import time, glob, os, sys, boto3, numpy as np, pandas as pd, json, requests, gc, math
from synthesizrr.base.util import is_list_like, Parameters, MappedParameters, optional_dependency, Registry, append_to_keys, \
    MutableParameters, Schema, only_item, set_param_from_alias, random_sample, as_tuple, as_list, str_normalize, \
    INDEX_COL_DEFAULT_NAME, AutoEnum, auto, safe_validate_arguments, type_str, Timer, StringUtil, get_default
from synthesizrr.base.data import ScalableDataFrame, ScalableSeries, ScalableSeriesRawType, ScalableDataFrameRawType, FileMetadata
from synthesizrr.base.framework import Dataset, Predictions, Algorithm, load_predictions
from synthesizrr.base.framework.task.embedding import Embedder, Embeddings, EmbeddingData, EMBEDDINGS_COL
from synthesizrr.base.constants import Task, MLType, MLTypeSchema, DataLayout, DataSplit, Alias, Parallelize
from pydantic import Extra, validator, root_validator, conint, constr

RelevanceAnnotation = "RelevanceAnnotation"
RankedResult = "RankedResult"
RetrievalIndex = "RetrievalIndex"

QUERY_COL: str = 'query'


class DistanceMetric(AutoEnum):
    ## https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#additional-metrics
    ## https://github.com/facebookresearch/faiss/blob/main/faiss/MetricType.h#L44
    L2 = auto()
    L1 = auto()
    Linf = auto()
    INNER_PRODUCT = auto()
    COSINE_SIMILARITY = auto()


class RelevanceAnnotation(Parameters):
    document_id: constr(min_length=1)
    relevance_grade: Union[bool, conint(ge=0)]

    @classmethod
    def of(cls, data: Union[Parameters, Dict]) -> RelevanceAnnotation:
        if isinstance(data, RelevanceAnnotation):
            return data
        elif isinstance(data, dict):
            return cls(**data)
        raise NotImplementedError(f'Unsupported type for {RelevanceAnnotation}: {type(data)} with value: {data}')

    @root_validator(pre=True)
    def set_relevance_annotation_params(cls, params: Dict) -> Dict:
        set_param_from_alias(params, param='document_id', alias=['id', 'doc_id'])
        set_param_from_alias(params, param='relevance_grade', alias=['grade', 'relevance_level', 'level'])
        return params


class RetrievalCorpus(Dataset):
    tasks = Task.RETRIEVAL_CORPUS

    features_schema = {}
    ground_truths_schema = {}


class Queries(Dataset):
    _allow_empty_features_schema: ClassVar[bool] = True

    tasks = Task.RETRIEVAL

    features_schema = {
        '{query}': MLType.TEXT,
    }
    ground_truths_schema = {
        '{relevant_document_ids}': MLType.OBJECT,
    }

    def to_embedding_data(self) -> EmbeddingData:
        data_schema: Schema = self.data_schema.drop_ground_truths()
        return EmbeddingData.of(**{
            **self.dict(exclude={'data', 'data_schema'}),
            **dict(
                task=Task.EMBEDDING,
                data=self.data,
                data_schema=data_schema,
            ),
        })


RETRIEVAL_FORMAT_MSG: str = f"""
 Retrieval results returned by algorithm must contains a top-K document ids, 
 ordered from most-relevant to least-relevant.
 """.strip()


class RankedResult(Parameters):
    document_id: constr(min_length=1)
    document: Optional[Any] = None
    rank: conint(ge=1)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f'{self.class_name} with items:\n{StringUtil.pretty(self.dict())}'

    class Config(Parameters.Config):
        ## Allow extra keyword parameters to be stored in RankedResult.
        extra = Extra.allow

    @classmethod
    def of(cls, data: Union[Parameters, Dict]) -> RankedResult:
        if isinstance(data, RankedResult):
            return data
        elif isinstance(data, dict):
            return cls(**data)
        raise NotImplementedError(f'Unsupported type for {RankedResult}: {type(data)} with value: {data}')

    @root_validator(pre=True)
    def set_ranked_result_params(cls, params: Dict) -> Dict:
        set_param_from_alias(params, param='document_id', alias=['id', 'doc_id'])
        set_param_from_alias(params, param='document', alias=['doc', 'asset', 'passage'])
        set_param_from_alias(params, param='relevance_grade', alias=['grade', 'relevance_level', 'level'])
        return params


RETRIEVAL_RANKED_RESULTS_COL: str = 'ranked_results'
RETRIEVAL_RANKED_RESULTS_TOP_K_COL: str = 'ranked_results_top_k'

RankedResults = "RankedResults"


class RankedResults(Predictions):
    tasks = (Task.RANKING, Task.RETRIEVAL)
    ground_truths_schema = {
        '{relevant_document_ids}': MLType.OBJECT,
    }
    predictions_schema = {
        RETRIEVAL_RANKED_RESULTS_COL: MLType.OBJECT,  ## List[RankedResult]
    }

    @safe_validate_arguments
    def flatten(self, **kwargs) -> RankedResults:
        ## Converting to List of Dict speeds up a lot:
        orig_layout: DataLayout = self.data.layout
        updated_data: ScalableDataFrame = self.to_layout(DataLayout.LIST_OF_DICT) \
            ._flatten_ranked_results_data(**kwargs) \
            .to_layout(orig_layout)
        updated_data_cols: Set[str] = updated_data.columns_set
        new_cols: Set[str] = updated_data_cols - self.data.columns_set
        top_k: int = updated_data[RETRIEVAL_RANKED_RESULTS_TOP_K_COL].max()
        new_cols_data_schema: MLTypeSchema = {
            RETRIEVAL_RANKED_RESULTS_TOP_K_COL: MLType.INT,
        }
        for rank in range(top_k):
            for new_col in new_cols:
                top_k_prefix: str = f'top_{rank + 1}_'
                if not new_col.startswith(top_k_prefix):
                    continue
                new_col_without_prefix: str = StringUtil.remove_prefix(new_col, prefix=top_k_prefix)
                if new_col_without_prefix in {'document_id', 'distance_metric'}:
                    new_col_mltype: MLType = MLType.CATEGORICAL
                elif new_col_without_prefix in {'distance', }:
                    new_col_mltype: MLType = MLType.FLOAT
                else:
                    new_col_mltype: MLType = MLType.TEXT
                new_cols_data_schema[new_col] = new_col_mltype

        updated_data_schema: Schema = self.data_schema.set_features(
            {**self.data_schema.features_schema, **new_cols_data_schema},
            override=True,
        )
        return self.update_params(
            data=updated_data,
            data_schema=updated_data_schema,
        )

    @safe_validate_arguments
    def _flatten_ranked_results_data(
            self,
            *,
            top_k: Optional[conint(ge=1)] = None,
            drop: bool = False,
    ) -> ScalableDataFrame:
        data: ScalableDataFrame = self.data.apply(self._flatten_row_ranked_results, top_k=top_k, axis=1)
        if drop:
            data: ScalableDataFrame = data[[col for col in data.columns if col != RETRIEVAL_RANKED_RESULTS_COL]]
        return data

    @classmethod
    def _flatten_row_ranked_results(cls, row, top_k: Optional[conint(ge=1)] = None):
        row[RETRIEVAL_RANKED_RESULTS_COL]: List[RankedResult] = [
            RankedResult.of(rr) for rr in row[RETRIEVAL_RANKED_RESULTS_COL]
        ]
        for rank, ranked_result in enumerate(row[RETRIEVAL_RANKED_RESULTS_COL]):
            if top_k is not None and (rank + 1) > top_k:  ## If top_k=4, rank is 4, meaning rank+1 is 5, then break
                break
            if ranked_result.document is not None:
                for doc_key, doc_val in ranked_result.document.items():
                    row[f'top_{rank + 1}_{doc_key}'] = doc_val
            for other_key in {'document_id', 'distance', 'distance_metric'}:
                if hasattr(ranked_result, other_key):
                    row[f'top_{rank + 1}_{other_key}'] = getattr(ranked_result, other_key)
            row[RETRIEVAL_RANKED_RESULTS_TOP_K_COL] = rank + 1
        return row


class Retriever(Algorithm, ABC):
    tasks = Task.RETRIEVAL
    inputs = Queries
    outputs = RankedResults


class RetrievalIndex(MutableParameters, Registry, ABC):
    index: Optional[Any] = None

    class Config(MutableParameters.Config):
        extra = Extra.ignore

    @classmethod
    def of(
            cls,
            index: Optional[str] = None,
            *,
            data: Optional[Union[Dataset, Predictions, ScalableSeries, ScalableSeriesRawType, FileMetadata]] = None,
            reset_index: bool = False,
            **kwargs,
    ) -> RetrievalIndex:
        if index is not None:
            RetrievalIndexClass: Type[RetrievalIndex] = RetrievalIndex.get_subclass(index)
        elif 'name' in kwargs:
            RetrievalIndexClass: Type[RetrievalIndex] = RetrievalIndex.get_subclass(kwargs.pop('name'))
        else:
            RetrievalIndexClass: Type[RetrievalIndex] = cls
        if RetrievalIndexClass == RetrievalIndex:
            subclasses: List[str] = random_sample(
                as_list(RetrievalIndex.subclasses),
                n=3,
                replacement=False
            )
            raise ValueError(
                f'"{RetrievalIndex.class_name}" is an abstract class. '
                f'To create an instance, please either pass `index`, '
                f'or call .of(...) on a subclass of {RetrievalIndex.class_name}, e.g. {", ".join(subclasses)}'
            )

        index: RetrievalIndex = RetrievalIndexClass(**kwargs)
        if index.index is None or reset_index:
            index.initialize(**kwargs)
        if data is not None:
            index.update_index(data, **kwargs)
        return index

    @abstractmethod
    def initialize(self, **kwargs) -> Any:
        pass

    @property
    @abstractmethod
    def index_size(self) -> int:
        pass

    @abstractmethod
    def update_index(
            self,
            data: Union[ScalableSeries, ScalableSeriesRawType, FileMetadata],
            **kwargs,
    ) -> Any:
        pass

    @abstractmethod
    def retrieve(
            self,
            queries: Union[Dataset, Predictions, ScalableSeries, ScalableSeriesRawType],
            *,
            top_k: int,
            retrieve_documents: bool,
            **kwargs
    ) -> List[List[RankedResult]]:
        pass
