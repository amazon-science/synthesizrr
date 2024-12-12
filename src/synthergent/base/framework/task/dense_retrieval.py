from typing import *
from abc import ABC, abstractmethod
import time, glob, os, sys, boto3, numpy as np, pandas as pd, json, requests, gc, math, multiprocessing as mp
from synthergent.base.util import is_list_like, Parameters, MappedParameters, optional_dependency, Registry, append_to_keys, \
    MutableParameters, Schema, only_item, set_param_from_alias, random_sample, as_tuple, as_list, str_normalize, \
    INDEX_COL_DEFAULT_NAME, AutoEnum, auto, safe_validate_arguments, type_str, Timer, StringUtil, get_default
from synthergent.base.data import ScalableDataFrame, ScalableSeries, ScalableSeriesRawType, ScalableDataFrameRawType, \
    FileMetadata
from synthergent.base.framework import Dataset, Predictions, Algorithm, load_predictions
from synthergent.base.framework.task.embedding import Embedder, Embeddings, EmbeddingData, EMBEDDINGS_COL
from synthergent.base.constants import Task, MLType, MLTypeSchema, DataLayout, DataSplit, Alias, FailureAction, Parallelize
from pydantic import Extra, validator, root_validator, conint, constr
from synthergent.base.framework.task.retrieval import Retriever, RetrievalIndex, RetrievalCorpus, Queries, RankedResult, \
    RankedResults, RelevanceAnnotation, QUERY_COL, RETRIEVAL_FORMAT_MSG, RETRIEVAL_RANKED_RESULTS_COL, DistanceMetric

DenseRetrievalIndex = "DenseRetrievalIndex"


def _normalize_l2(embeddings: np.ndarray) -> np.ndarray:
    """Normalizes a matrix of vectors of shape (B, dim), by dividing each of the B vectors by its L2 norm."""

    if embeddings.ndim == 1:
        embeddings: np.ndarray = embeddings[np.newaxis, :]  ## Converts from shape (128,) to shape (128, 1)
    if embeddings.ndim != 2:
        raise ValueError(f'Can only normalize embeddings which are 1d or 2d; found {embeddings.ndim} dims')
    ## Faiss way of normalizing changes the embeddings inplace: https://github.com/facebookresearch/faiss/issues/95
    ## Equivalent numpy method is below, Ref: https://stackoverflow.com/a/8904762
    return embeddings / np.linalg.norm(embeddings, 2, axis=1, keepdims=True)


class DenseRetrievalIndex(RetrievalIndex):
    @property
    @abstractmethod
    def vector_ndim(self) -> int:
        pass

    @abstractmethod
    def retrieve(
            self,
            queries: Union[Embeddings, ScalableSeries, ScalableSeriesRawType],
            *,
            top_k: int,
            retrieve_documents: bool,
            **kwargs
    ) -> List[List[RankedResult]]:
        pass


with optional_dependency('faiss'):
    ## Excellent tutorials on Faiss: https://www.pinecone.io/learn/faiss/
    ## For MacOSX, use `pip install faiss-cpu` (Ref: https://github.com/kyamagu/faiss-wheels)
    import faiss


    class FaissIndexParams(MappedParameters):
        dict_exclude = ('distance_metric',)

        _INDEX_TO_DISTANCE_METRICS: ClassVar[Dict[str, DistanceMetric]] = append_to_keys(
            prefix='faiss.',
            d={
                'IndexFlatL2': DistanceMetric.L2,
                'IndexFlatIP': DistanceMetric.INNER_PRODUCT,
            }
        )

        _DISTANCE_METRICS_TO_FAISS: ClassVar[Dict[DistanceMetric, int]] = {
            DistanceMetric.L1: faiss.METRIC_L1,
            DistanceMetric.Linf: faiss.METRIC_Linf,
            DistanceMetric.L2: faiss.METRIC_L2,
            DistanceMetric.INNER_PRODUCT: faiss.METRIC_INNER_PRODUCT,
        }

        _mapping = append_to_keys(
            prefix='faiss.',
            d={
                'IndexFlatL2': faiss.IndexFlatL2,  ## L2 = Euclidean distance
                'IndexFlatIP': faiss.IndexFlatIP,  ## IP = Inner Product
                'IndexFlat': faiss.IndexFlat,  ## Supports multiple indexes
                'IndexLSH': faiss.IndexLSH,
                'IndexFlat1D': faiss.IndexFlat1D,
                'IndexPQ': faiss.IndexPQ,
                'IndexIVFFlat': faiss.IndexIVFFlat,
                'IndexIVFPQ': faiss.IndexIVFPQ,
            }
        )
        distance_metric: Optional[DistanceMetric] = None

        @root_validator(pre=True)
        def set_faiss_index_params(cls, params: Dict) -> Dict:
            set_param_from_alias(params, param='vector_ndim', alias=[
                'vector_dim', 'vector_size', 'ndim', 'dim', 'd', 'embedding_size', 'embedding_dim', 'embedding_ndim',
            ])

            index_name: str = str_normalize(params['name'])
            index_to_distance_metrics: Dict[str, DistanceMetric] = {
                str_normalize(k): v for k, v in cls._INDEX_TO_DISTANCE_METRICS.items()
            }
            distance_metric: Optional[DistanceMetric] = params.get('distance_metric')
            if distance_metric is None:
                if index_name in index_to_distance_metrics:
                    distance_metric: DistanceMetric = index_to_distance_metrics[index_name]
            else:
                if DistanceMetric.does_not_match_any(distance_metric):
                    raise ValueError(f'Unsupported distance_metric: "{distance_metric}"')
                distance_metric: DistanceMetric = DistanceMetric.from_str(distance_metric)
                if distance_metric is DistanceMetric.COSINE_SIMILARITY and \
                        cls._mapping[index_name] != faiss.IndexFlatIP:
                    raise ValueError(
                        f'When using distance_metric={distance_metric}, '
                        f'the index must be "{faiss.IndexFlatIP.__name__}"; found "{params["name"]}".'
                    )
                ## Ref: https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#additional-metrics
                if distance_metric in {DistanceMetric.L1, DistanceMetric.Linf} and \
                        cls._mapping[index_name] != faiss.IndexFlat:
                    raise ValueError(
                        f'When using distance_metric={distance_metric}, '
                        f'the index must be "{faiss.IndexFlat.__name__}"; found "{params["name"]}".'
                    )
            if distance_metric is not None:
                params['distance_metric']: DistanceMetric = distance_metric

            vector_ndim: int = params.pop('vector_ndim')
            args: List = [vector_ndim, ]

            if cls._mapping[index_name] == faiss.IndexFlat and distance_metric in cls._DISTANCE_METRICS_TO_FAISS:
                args.append(cls._DISTANCE_METRICS_TO_FAISS[distance_metric])

            if params.get('args') is not None:
                args: List = as_list(args) + as_list(params['args'])
            params['args']: Tuple = as_tuple(args)
            return params


    class FaissRetrievalIndex(DenseRetrievalIndex):
        aliases = ['faiss']
        index: Optional[faiss.Index] = None
        params: Optional[Union[FaissIndexParams, Dict, str]] = None
        faiss_idx2doc_id: Dict[int, str] = {}
        doc_id2faiss_idx: Dict[str, int] = {}
        docs: Dict[str, Dict] = {}

        @root_validator(pre=False)
        def set_faiss_params(cls, params: Dict) -> Dict:
            params['params'] = FaissIndexParams.of(params['params'])
            return params

        def initialize(self, **kwargs):
            index: faiss.Index = self.params.initialize()
            if isinstance(index, faiss.IndexFlat):
                index: faiss.Index = faiss.IndexIDMap(index)  ## Ref: https://stackoverflow.com/a/71927179
            self.index = index

        @property
        def index_size(self) -> int:
            return self.index.ntotal

        @property
        def vector_ndim(self) -> int:
            return self.index.d

        @safe_validate_arguments
        def update_index(
                self,
                data: Union[Embeddings, ScalableSeries, ScalableSeriesRawType, FileMetadata],
                *,
                store_documents: bool = True,
                **kwargs,
        ):
            if self.index is None:
                raise ValueError('Index has not been created.')
            set_param_from_alias(kwargs, param='indexing_batch_size', alias=[
                'indexing_num_rows', 'indexing_nrows',
            ], default=1e5)
            set_param_from_alias(kwargs, param='batch_size', alias=['num_rows', 'nrows'], default=None)
            ## Override batch size with indexing batch size:
            kwargs['batch_size']: int = kwargs.pop('indexing_batch_size')
            kwargs['progress_bar'] = False
            if isinstance(data, FileMetadata):
                preds: Predictions = load_predictions(data, **kwargs)
                if not isinstance(preds, Embeddings):
                    raise ValueError(
                        f'Expected data in "{data.path}" to contain serialized {Embeddings.class_name}; '
                        f'after reading, found object of type: {type_str(preds)}.'
                    )
                data: Embeddings = preds

            if not isinstance(data, Embeddings):
                ## Assume these are only the embeddings:
                store_documents: bool = False
                if isinstance(data, np.ndarray):
                    if not data.ndim == 2:
                        raise ValueError(f'Expected input numpy array to have exactly 2 dimensions; found: {data.ndim}')
                    data: List[np.ndarray] = list(data)
                data: np.ndarray = ScalableSeries.of(data, layout=DataLayout.NUMPY).numpy()
                data: Embeddings = Embeddings.of(
                    data_split=DataSplit.PREDICT,
                    data=ScalableDataFrame.of({
                        EMBEDDINGS_COL: data,
                        INDEX_COL_DEFAULT_NAME: np.arange(
                            self.index_size,
                            self.index_size + len(data),
                        )
                    }),
                    data_schema=Schema(
                        index_col=INDEX_COL_DEFAULT_NAME,
                        predictions_schema={
                            EMBEDDINGS_COL: MLType.VECTOR
                        }
                    )
                )
            ## It's faster to store all embeddings in a big array and then add it:
            for batch in data.iter(**kwargs):
                assert isinstance(batch, Embeddings)
                faiss_idxs: np.ndarray = np.arange(
                    self.index_size,
                    self.index_size + len(batch),
                ).astype(np.int64)
                if store_documents:
                    batch_docs: List[Optional[Dict]] = [
                        d
                        for d in batch.features(return_series=False).to_list_of_dict()
                    ]
                else:
                    batch_docs: List[Optional[Dict]] = [
                        None
                        for _ in range(len(batch))
                    ]
                for faiss_idx, doc_id, doc in zip(faiss_idxs, batch.index().numpy(), batch_docs):
                    doc_id: str = str(doc_id)
                    if faiss_idx in self.faiss_idx2doc_id:
                        raise ValueError(f'Faiss index {faiss_idx} already exists in index.')
                    self.faiss_idx2doc_id[faiss_idx] = doc_id
                    if doc_id in self.doc_id2faiss_idx:
                        raise ValueError(f'Document id "{faiss_idx}" already exists in index.')
                    self.doc_id2faiss_idx[doc_id] = faiss_idx
                    if doc is not None:
                        self.docs[doc_id] = doc
                embeddings: np.ndarray = batch.embeddings.numpy(stack=True)
                if self.params.distance_metric is DistanceMetric.COSINE_SIMILARITY:
                    ## Ref: https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-index-vectors-for-cosine-similarity
                    embeddings: np.ndarray = _normalize_l2(embeddings)
                self.index.add_with_ids(embeddings, faiss_idxs)

        def retrieve(
                self,
                queries: Union[Embeddings, ScalableSeries, ScalableSeriesRawType],
                *,
                top_k: int,
                retrieve_documents: bool,
                index_retrieval_batch_size: int = 16,
                **kwargs
        ) -> List[List[RankedResult]]:
            if isinstance(queries, Embeddings):
                queries: ScalableSeries = queries.embeddings
            if isinstance(queries, np.ndarray):
                queries: List[np.ndarray] = list(queries)
            if not isinstance(queries, ScalableSeries):
                queries: ScalableSeries = ScalableSeries.of(
                    queries,
                    layout=DataLayout.NUMPY
                )
            ranked_results: List[List[RankedResult]] = []
            for queries_batch in queries.stream(batch_size=index_retrieval_batch_size):
                assert isinstance(queries_batch, ScalableSeries)
                queries_batch_np: np.ndarray = queries_batch.numpy(stack=True)
                if self.params.distance_metric is DistanceMetric.COSINE_SIMILARITY:
                    queries_batch_np: np.ndarray = _normalize_l2(queries_batch_np)
                top_k_distances, top_k_faiss_idxs = self.index.search(
                    queries_batch_np,
                    k=top_k,
                )
                assert top_k_distances.shape[0] == top_k_faiss_idxs.shape[0] == len(queries_batch)
                for i in range(len(queries_batch)):
                    ranked_results.append([])
                    for k, (top_k_dist, top_k_faiss_idx) in enumerate(zip(
                            top_k_distances[i], top_k_faiss_idxs[i],
                    )):
                        k: int = k + 1
                        doc_id: str = self.faiss_idx2doc_id[top_k_faiss_idx]
                        doc: Optional[Any] = None
                        if retrieve_documents:
                            doc: Any = self.docs.get(doc_id)
                        ranked_results[-1].append(
                            RankedResult.of(dict(
                                rank=k,
                                document_id=doc_id,
                                document=doc,
                                distance=float(top_k_dist),
                                distance_metric=self.params.distance_metric,
                            ))
                        )
            return ranked_results


class DenseRetriever(Retriever):
    embedder: Optional[Union[Embedder, Any]] = None
    index: Optional[DenseRetrievalIndex] = None

    class Hyperparameters(Algorithm.Hyperparameters):
        embedder: Optional[Dict] = None  ## Params for embedder
        index: Optional[Dict] = None  ## Params for index

    def initialize(self, model_dir: Optional[FileMetadata] = None):
        if self.embedder is None and self.hyperparams.embedder is None:
            raise ValueError(
                f'To initialize {self.class_name}, you must either pass an embedder explicitly or set the `embedder` '
                f'hyperparam with a dict of parameters that can be used to initialize an embedder.'
            )
        elif self.embedder is None:
            self.embedder: Algorithm = Algorithm.of(**{
                **dict(task=Task.EMBEDDING),
                **self.hyperparams.embedder,
            })

        if self.index is None and self.hyperparams.index is None:
            raise ValueError(
                f'To initialize {self.class_name}, you must either pass an index explicitly or set the `index` '
                f'hyperparam with a dict of parameters that can be used to initialize an index.'
            )
        elif self.index is None:
            with Timer(task='Creating Index'):
                self.index = DenseRetrievalIndex.of(**self.hyperparams.index)

    def _task_preprocess(self, batch: Queries, **kwargs) -> Queries:
        if batch.has_ground_truths(raise_error=False):
            gt_col: str = only_item(set(batch.data_schema.ground_truths().keys()))
            relevance_annotations: ScalableSeries = batch.ground_truths(return_series=True)
            batch.data[gt_col] = ScalableSeries.of(
                [RelevanceAnnotation.of(ra) for ra in relevance_annotations],
                layout=relevance_annotations.layout,
            )
        return batch

    def predict_step(
            self,
            batch: Queries,
            retrieve_documents: bool = True,
            **kwargs,
    ) -> Dict:
        Alias.set_top_k(kwargs, default=1)
        top_k: int = kwargs.pop('top_k')
        kwargs['progress_bar'] = None
        queries: Embeddings = self._embedder_predict(batch.to_embedding_data(), top_k=top_k, **kwargs)
        kwargs.pop('progress_bar')
        ranked_results: List[List[RankedResult]] = self.index.retrieve(
            queries,
            top_k=top_k,
            retrieve_documents=retrieve_documents,
            **kwargs
        )
        return {
            'ranked_results': ranked_results
        }

    def _embedder_predict(self, query_embedding_data: EmbeddingData, **kwargs) -> Embeddings:
        from synthergent.base.framework.evaluator import Evaluator
        # print('Queries:')
        # with pd_extended_display() as disp:
        #     disp(queries.data.pandas())
        if isinstance(self.embedder, Evaluator):
            embedder_batch_size: int = get_default(
                self.embedder._create_hyperparams().batch_size,
                self.hyperparams.batch_size
            )
            query_embeddings: Embeddings = self.embedder.evaluate(
                query_embedding_data,
                **{
                    **dict(
                        batch_size=embedder_batch_size,
                        submission_batch_size=embedder_batch_size,
                        progress_bar=None,
                        return_predictions=True,
                        failure_action=FailureAction.ERROR_DELAYED,
                    ),
                    **kwargs,
                }
            )
        elif isinstance(self.embedder, Embedder):
            embedder_batch_size: int = get_default(
                self.embedder.hyperparams.batch_size,
                self.hyperparams.batch_size,
            )
            query_embeddings: Embeddings = self.embedder.predict(
                query_embedding_data,
                **{
                    **dict(
                        batch_size=embedder_batch_size,
                        progress_bar=None,
                    ),
                    **kwargs,
                }
            )
        else:
            raise ValueError(f'Expected `embedder` to be either an instance of {Embedder} or {Evaluator}')
        if not isinstance(query_embeddings, Embeddings):
            raise ValueError(f'Expected embedder output to be {Embeddings}; found: {type_str(query_embeddings)}')
        return query_embeddings

    def _create_predictions(
            self,
            batch: Queries,
            predictions: Dict,
            retrieve_documents: bool = True,
            top_k: int = 1,
            **kwargs,
    ) -> RankedResults:
        if 'ranked_results' not in predictions:
            raise ValueError(RETRIEVAL_FORMAT_MSG)
        if len(predictions['ranked_results']) != len(batch):
            raise ValueError(
                f'We expected a (possibly empty) list of ranked results for each of the input queries; '
                f'found {len(batch)} input queries but returned {len(predictions["ranked_results"])} result-lists.'
            )
        ranked_results: List[List[RankedResult]] = predictions['ranked_results']
        predictions: Dict[str, List[List[RankedResult]]] = {
            RETRIEVAL_RANKED_RESULTS_COL: ranked_results
        }
        return RankedResults.from_task_data(
            data=batch,
            predictions=predictions,
            **kwargs
        )
