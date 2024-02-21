from typing import *
from abc import ABC, abstractmethod

import ray
import time, glob, os, sys, boto3, numpy as np, pandas as pd, json, requests, gc, math, multiprocessing as mp
from synthesizrr.base.util import is_list_like, Parameters, MappedParameters, optional_dependency, Registry, append_to_keys, \
    MutableParameters, Schema, only_item, set_param_from_alias, wait, as_tuple, as_list, str_normalize, \
    INDEX_COL_DEFAULT_NAME, AutoEnum, auto, safe_validate_arguments, type_str, Timer, StringUtil, get_default, \
    dispatch, dispatch_executor, accumulate, accumulate_iter, ProgressBar, best_k, keep_keys, format_exception_msg
from synthesizrr.base.framework.ray_base import ActorComposite
from synthesizrr.base.data import ScalableDataFrame, ScalableSeries, ScalableSeriesRawType, ScalableDataFrameRawType, FileMetadata
from synthesizrr.base.framework import Dataset, load_dataset, Algorithm
from synthesizrr.base.constants import Task, MLType, MLTypeSchema, DataLayout, DataSplit, Alias, Parallelize
from pydantic import Extra, validator, root_validator, conint, constr, confloat
from synthesizrr.base.framework.task.retrieval import RetrievalIndex, RetrievalCorpus, Retriever, Queries, RelevanceAnnotation, \
    RankedResult, RankedResults, QUERY_COL, RETRIEVAL_FORMAT_MSG, RETRIEVAL_RANKED_RESULTS_COL
from collections import Counter

SparseRetrievalIndex = "SparseRetrievalIndex"


class SparseRetrievalIndex(RetrievalIndex):
    @abstractmethod
    def retrieve(
            self,
            queries: Union[Queries, ScalableSeries, ScalableSeriesRawType],
            *,
            top_k: int,
            retrieve_documents: bool,
            **kwargs
    ) -> List[List[RankedResult]]:
        pass


"""
Code modified from https://github.com/dorianbrown/rank_bm25 
"""


class BM25IndexStoreDoc(Parameters):
    doc_id: str
    doc_len: int  ## number of tokens in doc
    doc: Optional[Dict]

    @staticmethod
    def clean_doc_id(doc_id: Union[int, float, str]) -> str:
        if not (isinstance(doc_id, (int, str)) or np.issubdtype(type(doc_id), np.integer)):
            if isinstance(doc_id, float) and int(doc_id) == doc_id:
                doc_id: int = int(doc_id)
            else:
                raise ValueError(f'Expected document id to be an int or string; found: {type_str(doc_id)}')
        doc_id: str = str(doc_id)
        assert len(doc_id) > 0
        return doc_id


class BM25DistanceMetric(AutoEnum):
    BM25Okapi = auto()
    BM25L = auto()
    BM25Plus = auto()
    BM25Adpt = auto()
    BM25T = auto()


BM25IndexStore = "BM25IndexStore"


class BM25IndexStore(MutableParameters, ABC):
    class Config(MutableParameters.Config):
        extra = Extra.ignore

    distance_metric: ClassVar[BM25DistanceMetric]

    store_documents: bool = True
    tokenizer: Any
    index_size: conint(ge=0) = 0
    corpus_num_tokens: conint(ge=0) = 0
    docs: Dict[str, BM25IndexStoreDoc] = {}  ## doc_id -> document
    doc_token_freqs: Dict[str, Counter] = {}  ## doc_id -> count of each token in document.
    token_doc_freqs: Counter = Counter()  ## token -> number of docs which have this token.
    token_idfs: Optional[Dict[str, float]] = None  ## token -> token idf
    _doc_ids_np: np.ndarray = np.array([])
    _doc_lens_np: np.ndarray = np.array([])

    @property
    def avgdl(self) -> float:
        return self.corpus_num_tokens / self.index_size  ## Average document length

    @property
    def doc_ids(self) -> np.ndarray:
        if self._doc_ids_np.shape[0] != len(self.docs):
            self._doc_ids_np = np.array(list(self.docs.keys()))
        return self._doc_ids_np

    @property
    def doc_lens(self) -> np.ndarray:
        if self._doc_lens_np.shape[0] != len(self.docs):
            self._doc_lens_np = np.array([doc.doc_len for doc_id, doc in self.docs.items()], dtype=np.uint32)
        return self._doc_lens_np

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f'BM25 ({self.distance_metric.value}) Index Store with {len(self.docs)} entries'

    def concat_index(
            self,
            index2: BM25IndexStore,
            *,
            recalc_token_idfs: bool = True,
    ) -> BM25IndexStore:
        assert self.class_name == index2.class_name
        dict_keys_to_exclude: List[str] = [
            'index_size',
            'tokenizer',
            'corpus_num_tokens',
            'store_documents',
            'docs',
            'doc_token_freqs',
            'token_doc_freqs',
            'token_idfs',
            '_doc_ids_np',
            '_doc_lens_np',
        ]
        if self.dict(exclude=dict_keys_to_exclude) != index2.dict(exclude=dict_keys_to_exclude):
            raise ValueError(
                f'Params of incoming index must be the same as existing index:\n'
                f'Existing index: {self.self.dict(exclude=dict_keys_to_exclude)}\n'
                f'Incoming index: index2.dict(exclude=dict_keys_to_exclude)'
            )
        for doc_id in index2.docs:
            if doc_id in self.docs:
                ## Throw error:
                common_doc_ids: Set[str] = set(self.docs.keys()).intersection(set(index2.docs.keys()))
                raise ValueError(
                    f'Cannot combine {BM25IndexStore.class_name} objects with common document ids; '
                    f'found {len(common_doc_ids)} documents with the same ids.'
                )
        self.index_size += index2.index_size
        self.corpus_num_tokens += index2.corpus_num_tokens
        self.docs.update(index2.docs)  ## Can be sharded
        self.doc_token_freqs.update(index2.doc_token_freqs)  ## Can be sharded
        self.token_doc_freqs += index2.token_doc_freqs
        if recalc_token_idfs:
            self.recalculate_token_idfs()

    def recalculate_token_idfs(self):
        ## Recalculate Token IDF values:
        self.token_idfs: Dict[str, float] = self.calculate_token_idfs()

    def add_documents(
            self,
            documents: Union[ScalableDataFrame, ScalableDataFrameRawType],
            *,
            id_col: str,
            text_col: str,
            **kwargs,
    ):
        documents: ScalableDataFrame = ScalableDataFrame.of(documents, layout=DataLayout.LIST_OF_DICT)
        num_docs: int = len(documents)
        progress_bar: Optional[Dict] = Alias.get_progress_bar(kwargs)
        update_pbar_ever: int = 100
        pbar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=num_docs,
            unit='documents',
        )
        for i, doc_data in enumerate(documents.to_list_of_dict()):
            self._add_document(doc_data=doc_data, id_col=id_col, text_col=text_col)
            if (i + 1) % update_pbar_ever == 0:
                pbar.update(update_pbar_ever)
        pbar.update(num_docs % update_pbar_ever)
        pbar.success()
        self.recalculate_token_idfs()

    def _add_document(
            self,
            doc_data: Dict,
            *,
            id_col: str,
            text_col: str,
    ):
        doc_id: Union[int, float, str] = doc_data[id_col]
        doc_id: str = BM25IndexStoreDoc.clean_doc_id(doc_id)
        doc_text: str = doc_data[text_col]
        doc_tokens: List[str] = self._tokenize_text(doc_text)
        doc_token_freq: Counter = Counter(doc_tokens)
        doc_len: int = len(doc_tokens)

        self.corpus_num_tokens += doc_len
        self.docs[doc_id] = BM25IndexStoreDoc(
            doc_id=doc_id,
            doc_len=doc_len,
            doc=doc_data if self.store_documents else None,
        )
        self.doc_token_freqs[doc_id] = doc_token_freq
        for token, freq in doc_token_freq.items():
            self.token_doc_freqs[token] += 1
        self.index_size += 1

    def _tokenize_text(self, text: Union[str, List[str]]) -> List[str]:
        if isinstance(text, str) and self.tokenizer is not None:
            text: List[str] = self.tokenizer(text)
        if not is_list_like(text):
            raise ValueError(f'Expected text to be tokenized string, found: {type_str(text)}')
        return text

    @abstractmethod
    def calculate_token_idfs(self) -> Dict[str, float]:
        pass

    def get_top_k(self, query: str, k: int) -> List[Tuple[BM25IndexStoreDoc, float]]:
        doc_query_scores: np.ndarray = self.get_doc_scores(query)
        top_k_doc_query_score_idxs, top_k_doc_query_scores = best_k(
            doc_query_scores,
            k=k,
            how='max',
            sort='descending',
        )
        top_k_doc_ids: np.ndarray = self.doc_ids[top_k_doc_query_score_idxs]
        top_k_docs: List[BM25IndexStoreDoc] = [
            self.docs[top_k_doc_id]
            for top_k_doc_id in top_k_doc_ids
        ]
        return list(zip(top_k_docs, top_k_doc_query_scores))

    @abstractmethod
    def get_doc_scores(self, query: str) -> np.ndarray:
        pass


class BM25Okapi(BM25IndexStore):
    distance_metric = BM25DistanceMetric.BM25Okapi

    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25

    def calculate_token_idfs(self) -> Dict[str, float]:
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        token_idfs: Dict[str, float] = {}
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum: float = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if token is contained in more than half of documents
        negative_idfs = []
        for token, freq in self.token_doc_freqs.items():
            idf: float = math.log(self.index_size - freq + 0.5) - math.log(freq + 0.5)
            token_idfs[token] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(token)
        average_idf: float = idf_sum / len(token_idfs)
        eps: float = self.epsilon * average_idf
        for token in negative_idfs:
            token_idfs[token] = eps
        return token_idfs

    def get_doc_scores(self, query: str) -> np.ndarray:
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query: a string which is used to retrieve.
        :return: np array of BM25 scores (one for each document).
        """
        query_tokens: List[str] = self._tokenize_text(query)
        doc_lens: np.ndarray = self.doc_lens
        doc_query_scores: np.ndarray = np.zeros(len(self.docs))
        for token in query_tokens:
            token_freq: np.ndarray = np.array([
                self.doc_token_freqs[doc_id].get(token, 0)
                for doc_id in self.docs  ## NOTE! We purposely iterate through self.docs to ensure doc_id order. 
            ], dtype=np.uint32)
            token_query_score: np.ndarray = self.token_idfs.get(token, 0) * (
                    token_freq * (self.k1 + 1) / (token_freq + self.k1 * (1 - self.b + self.b * doc_lens / self.avgdl))
            )
            doc_query_scores += token_query_score
        return doc_query_scores


class BM25IndexStoreParams(MappedParameters):
    store_documents: bool

    _mapping = append_to_keys(
        prefix='BM25',
        d={
            'Okapi': BM25Okapi,
        }
    )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f'{BM25IndexStoreParams.class_name} with params: {StringUtil.jsonify(self.dict())}'

    @property
    def index_class_name(self) -> str:
        BM25IndexStoreClass: Type[BM25IndexStore] = self.mapped_callable()
        return BM25IndexStoreClass.class_name

    @property
    def distance_metric(self) -> BM25DistanceMetric:
        BM25IndexStoreClass: Type[BM25IndexStore] = self.mapped_callable()
        return BM25IndexStoreClass.distance_metric


def _create_index_store(
        params: BM25IndexStoreParams,
        documents: Optional[RetrievalCorpus],
        **kwargs
) -> BM25IndexStore:
    index_store: BM25IndexStore = params.initialize()
    if not isinstance(index_store, BM25IndexStore):
        raise ValueError(
            f'Expected subclass of {BM25IndexStore.class_name} to be instantiated; '
            f'found object of type: {type_str(index_store)}'
        )
    if documents is not None:
        assert isinstance(documents, RetrievalCorpus)
        kwargs['progress_bar'] = False
        if documents.in_memory():
            should_delete: bool = False
            documents_data: ScalableDataFrame = documents.data
        else:
            should_delete: bool = True
            documents_data: ScalableDataFrame = documents.read(**kwargs).data
        index_store.add_documents(documents=documents_data, **kwargs)
        if should_delete:
            del documents_data
            import gc
            gc.collect()
    return index_store


class BM25RetrievalIndexBase(SparseRetrievalIndex, ABC):
    params: Optional[Union[BM25IndexStoreParams, Dict, str]] = None

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f'{self.params.index_class_name} index having {self.index_size} entries.'

    @safe_validate_arguments
    def update_index(
            self,
            data: Union[RetrievalCorpus, ScalableSeries, ScalableSeriesRawType],
            *,
            indexing_parallelize: Parallelize = Parallelize.processes,
            indexing_max_workers: int = min(mp.cpu_count() - 1, 63),
            indexing_iter_files: bool = False,
            indexing_progress_bar: Union[Dict, bool] = False,
            **kwargs,
    ):
        if self.index is None:
            raise ValueError('Index has not been created.')
        set_param_from_alias(kwargs, param='indexing_batch_size', alias=[
            'indexing_num_rows', 'indexing_nrows',
        ], default=int(50e3))
        set_param_from_alias(kwargs, param='batch_size', alias=['num_rows', 'nrows'], default=None)
        ## Override batch size with indexing batch size:
        kwargs['batch_size']: int = kwargs.pop('indexing_batch_size')
        kwargs['progress_bar'] = False
        if isinstance(data, FileMetadata):
            dataset: Dataset = load_dataset(data, **kwargs)
            if not isinstance(dataset, RetrievalCorpus):
                raise ValueError(
                    f'Expected data in "{data.path}" to contain serialized {RetrievalCorpus.class_name}; '
                    f'after reading, found object of type: {type_str(dataset)}.'
                )
            data: RetrievalCorpus = dataset

        if not isinstance(data, RetrievalCorpus):
            ## Assume this is only a text column:
            if isinstance(data, np.ndarray):
                if not data.ndim == 1:
                    raise ValueError(f'Expected input numpy array to have exactly 1 dimensions; found: {data.ndim}')
            data: np.ndarray = ScalableSeries.of(data, layout=DataLayout.NUMPY).numpy()
            text_col: str = 'text'
            data: RetrievalCorpus = RetrievalCorpus.of(
                data_split=DataSplit.UNSUPERVISED,
                data=ScalableDataFrame.of({
                    text_col: data,
                    INDEX_COL_DEFAULT_NAME: np.arange(
                        self.index_size,
                        self.index_size + len(data),
                    )
                }),
                data_schema=Schema(
                    index_col=INDEX_COL_DEFAULT_NAME,
                    features_schema={
                        text_col: MLType.TEXT
                    }
                )
            )
        assert isinstance(data, RetrievalCorpus)
        id_col: str = data.data_schema.index_col
        text_col: str = Schema.filter_single_column(
            data.data_schema.features_schema,
            mltype=MLType.TEXT,
            allow_missing=False,
        )
        batch_size: int = kwargs['batch_size']
        pbar_total: Optional[int] = None
        if data.in_memory():
            if len(data) < batch_size * 2:  ## If we have less than 2 tasks, just do it synchronously.
                indexing_parallelize: Parallelize = Parallelize.sync
            ## Ensure each worker has at least 2 tasks to work on.
            indexing_max_workers: int = min(indexing_max_workers, math.floor(len(data) / (batch_size * 2)))
            pbar_total: int = math.ceil(len(data) / batch_size)
        if self.params.store_documents is False:
            kwargs['schema_columns_only']: bool = True
            data: RetrievalCorpus = data.update_params(
                data_schema=data.data_schema.keep_columns([id_col, text_col])
            )
        if isinstance(self, RayBM25RetrievalIndex):
            indexing_parallelize: Parallelize = Parallelize.ray

        # print(f'Corpus: {data}')
        # print(f'schema_columns_only: {kwargs.get("schema_columns_only")}')
        if indexing_parallelize in {Parallelize.threads, Parallelize.processes}:
            pbar_desc: str = f'Indexing ({indexing_parallelize} with {indexing_max_workers} workers)'
        else:
            pbar_desc: str = f'Indexing ({indexing_parallelize})'
        if not data.in_memory() and indexing_iter_files:
            corpus_gen: List[RetrievalCorpus] = list(data.iter_files(**kwargs))
            pbar_desc += f', {len(corpus_gen)} files'
        else:
            corpus_gen: Generator[RetrievalCorpus, None, None] = data.iter(**kwargs)
            pbar_desc += f', {batch_size} rows'
        pbar: ProgressBar = ProgressBar.of(
            indexing_progress_bar,
            total=pbar_total,
            desc=pbar_desc,
            unit='partition',
        )
        self._update_index_from_corpus_gen(
            corpus_gen=corpus_gen,
            pbar=pbar,
            indexing_parallelize=indexing_parallelize,
            indexing_max_workers=indexing_max_workers,
            id_col=id_col,
            text_col=text_col,
            **kwargs
        )

    @abstractmethod
    def _update_index_from_corpus_gen(
            self,
            corpus_gen: Union[List[RetrievalCorpus], Generator[RetrievalCorpus, None, None]],
            *,
            pbar: ProgressBar,
            indexing_parallelize: Parallelize,
            indexing_max_workers: int,
            id_col: str,
            text_col: str,
            **kwargs
    ) -> NoReturn:
        pass

    def retrieve(
            self,
            queries: Union[Queries, ScalableSeries, ScalableSeriesRawType],
            *,
            top_k: int,
            retrieve_documents: bool,
            **kwargs
    ) -> List[List[RankedResult]]:
        index_retrieval_batch_size: int = get_default(kwargs.pop('batch_size'), 16)
        if isinstance(queries, Queries):
            queries: ScalableSeries = queries.features(MLType.TEXT, return_series=True)
        if not isinstance(queries, ScalableSeries):
            queries: ScalableSeries = ScalableSeries.of(
                queries,
                layout=DataLayout.NUMPY,
            )
        ranked_results: List[List[RankedResult]] = []
        for queries_batch in queries.stream(batch_size=index_retrieval_batch_size):
            assert isinstance(queries_batch, ScalableSeries)
            ranked_results.extend(self._retrieve_batch(
                queries_batch=queries_batch,
                top_k=top_k,
                retrieve_documents=retrieve_documents,
            ))
        return ranked_results

    @abstractmethod
    def _retrieve_batch(
            self,
            queries_batch: ScalableSeries,
            *,
            top_k: int,
            retrieve_documents: bool,
    ) -> List[List[RankedResult]]:
        pass


class BM25RetrievalIndex(BM25RetrievalIndexBase):
    aliases = ['BM25']
    index: Optional[BM25IndexStore] = None

    @root_validator(pre=False)
    def set_bm25_params(cls, params: Dict) -> Dict:
        params['params'] = BM25IndexStoreParams.of(params['params'])
        return params

    @property
    def index_size(self) -> int:
        return self.index.index_size

    def initialize(self, **kwargs):
        self.index: BM25IndexStore = _create_index_store(self.params, documents=None)

    def _update_index_from_corpus_gen(
            self,
            corpus_gen: Union[List[RetrievalCorpus], Generator[RetrievalCorpus, None, None]],
            *,
            pbar: ProgressBar,
            indexing_parallelize: Parallelize,
            indexing_max_workers: int,
            id_col: str,
            text_col: str,
            **kwargs
    ) -> NoReturn:
        executor: Optional[Any] = dispatch_executor(
            parallelize=indexing_parallelize,
            max_workers=indexing_max_workers,
        )
        try:
            index_futs: List[Any] = []
            for documents_batch in corpus_gen:
                assert isinstance(documents_batch, RetrievalCorpus)
                index_futs.append(
                    dispatch(
                        _create_index_store,
                        **{
                            **kwargs,
                            **dict(
                                params=self.params,
                                documents=documents_batch,
                                id_col=id_col,
                                text_col=text_col,
                                parallelize=indexing_parallelize,
                                executor=executor,
                            ),
                        },
                    )
                )
            for index_store in accumulate_iter(index_futs, progress_bar=pbar):
                self.index.concat_index(index_store, recalc_token_idfs=False)
            self.index.recalculate_token_idfs()
        finally:
            del executor

    def _retrieve_batch(
            self,
            queries_batch: ScalableSeries,
            top_k: int,
            retrieve_documents: bool,
    ) -> List[List[RankedResult]]:
        batch_ranked_results: List[List[RankedResult]] = []
        for query in queries_batch:
            top_k_results: List[Tuple[BM25IndexStoreDoc, float]] = self.index.get_top_k(
                query,
                k=top_k,
            )
            batch_ranked_results.append([])
            for k, (top_k_doc, top_k_score) in enumerate(top_k_results):
                k: int = k + 1
                doc: Optional[Dict] = None
                if retrieve_documents:
                    doc: Any = top_k_doc.doc
                batch_ranked_results[-1].append(
                    RankedResult.of(dict(
                        rank=k,
                        document_id=top_k_doc.doc_id,
                        document=doc,
                        distance=float(top_k_score),
                        distance_metric=self.params.distance_metric,
                    ))
                )
        return batch_ranked_results


class RayBM25IndexStoreParams(BM25IndexStoreParams):
    num_shards: int
    shard_num_cpus: int = 6


@ray.remote
class BM25IndexStoreActor:
    def __init__(
            self,
            *,
            actor_id: str,
            params: BM25IndexStoreParams,
    ):
        self.actor_id: str = actor_id
        self.params: BM25IndexStoreParams = params
        self.index_shard: BM25IndexStore = _create_index_store(self.params, documents=None)

    def get_actor_id(self) -> str:
        return self.actor_id

    def get_index_size(self) -> int:
        return self.index_shard.index_size

    def set_index_size(self, index_size: int):
        self.index_shard.index_size = index_size

    def get_corpus_num_tokens(self) -> int:
        return self.index_shard.corpus_num_tokens

    def set_corpus_num_tokens(self, corpus_num_tokens: int):
        self.index_shard.corpus_num_tokens = corpus_num_tokens

    def get_token_doc_freqs(self) -> Counter:
        return self.index_shard.token_doc_freqs

    def set_token_doc_freqs(self, token_doc_freqs: Counter):
        self.index_shard.token_doc_freqs = token_doc_freqs

    def recalculate_token_idfs(self):
        self.index_shard.recalculate_token_idfs()

    def update_index_shard(
            self,
            documents_data: Any,
            *,
            documents_params: Dict,
            params: RayBM25IndexStoreParams,
            **kwargs,
    ):
        try:
            documents: Dataset = Dataset.of(
                **documents_params,
                data=documents_data,
            )
            assert isinstance(documents, RetrievalCorpus)
            index_store_update: BM25IndexStore = _create_index_store(
                params=params,
                documents=documents,
                **kwargs,
            )
            self.index_shard.concat_index(index_store_update, recalc_token_idfs=False)
            return True
        except Exception as e:
            print(format_exception_msg(e))
            return False

    def get_top_k_batch(
            self,
            queries_batch: ScalableSeries,
            top_k: int,
    ) -> List[List[Tuple[BM25IndexStoreDoc, float]]]:
        return [
            self.index_shard.get_top_k(
                query,
                k=top_k,
            )
            for query in queries_batch
        ]


class RayBM25RetrievalIndex(BM25RetrievalIndexBase):
    aliases = ['BM25-Ray']
    index: Optional[List[ActorComposite]] = None
    params: Optional[Union[RayBM25IndexStoreParams, Dict, str]] = None

    @root_validator(pre=False)
    def set_bm25_params(cls, params: Dict) -> Dict:
        params['params'] = RayBM25IndexStoreParams.of(params['params'])
        return params

    def initialize(self, **kwargs):
        def actor_factory(*, actor_id: str, **kwargs):
            return BM25IndexStoreActor.options(
                num_cpus=self.params.shard_num_cpus,
            ).remote(
                actor_id=actor_id,
                params=self.params,
            )

        self.index: List[ActorComposite] = ActorComposite.create_actors(
            actor_factory,
            num_actors=self.params.num_shards,
        )

    @property
    def index_size(self) -> int:
        return sum(accumulate([
            actor_composite.actor.get_index_size.remote()
            for actor_composite in self.index
        ]))

    def _update_index_from_corpus_gen(
            self,
            corpus_gen: Union[List[RetrievalCorpus], Generator[RetrievalCorpus, None, None]],
            *,
            pbar: ProgressBar,
            indexing_parallelize: Parallelize,
            indexing_max_workers: int,
            id_col: str,
            text_col: str,
            **kwargs
    ) -> NoReturn:
        num_actors: int = len(self.index)
        rnd_idx: List[int] = list(np.random.permutation(range(num_actors)))
        index_futs: List[Any] = []
        for documents_batch_i, documents_batch in enumerate(corpus_gen):
            assert isinstance(documents_batch, RetrievalCorpus)
            ## Select in randomized round-robin order:
            index_actor_composite: ActorComposite = self.index[rnd_idx[documents_batch_i % num_actors]]
            index_futs.append(
                index_actor_composite.actor.update_index_shard.remote(**{
                    **kwargs,
                    **dict(
                        documents_data=documents_batch.data,
                        documents_params={
                            **documents_batch.dict(exclude={'data'}),
                            **dict(data_idx=documents_batch_i),
                        },
                        params=self.params,
                        id_col=id_col,
                        text_col=text_col,
                        parallelize=indexing_parallelize,
                    ),
                })
            )
        assert bool(all(accumulate(index_futs, progress_bar=pbar)))
        ## Next steps:
        ## (1) docs and doc_token_freqs remain sharded.
        ## (2) index_size, corpus_num_tokens and token_doc_freqs must be merged & synced between actors
        ## (3) token_idfs must be recalculated by each actor after this sync.
        all_index_size: List[int] = []
        all_corpus_num_tokens: List[int] = []
        all_token_doc_freqs: List[Counter] = []
        for index_actor_composite in self.index:
            index_actor: ray.actor.ActorHandle = index_actor_composite.actor
            all_index_size.append(index_actor.get_index_size.remote())
            all_corpus_num_tokens.append(index_actor.get_corpus_num_tokens.remote())
            all_token_doc_freqs.append(index_actor.get_token_doc_freqs.remote())
        combined_index_size: int = 0
        combined_corpus_num_tokens: int = 0
        combined_token_doc_freqs: Counter = Counter()
        for shard_index_size, shard_corpus_num_tokens, shard_token_doc_freqs in zip(
                accumulate(all_index_size),
                accumulate(all_corpus_num_tokens),
                accumulate(all_token_doc_freqs),
        ):
            combined_index_size += shard_index_size
            combined_corpus_num_tokens += shard_corpus_num_tokens
            combined_token_doc_freqs += shard_token_doc_freqs
        set_index_size_futs: List = []
        set_corpus_num_tokens_futs: List = []
        set_token_doc_freqs_futs: List = []
        for index_actor_composite in self.index:
            index_actor: ray.actor.ActorHandle = index_actor_composite.actor
            set_index_size_futs.append(index_actor.set_index_size.remote(combined_index_size))
            set_corpus_num_tokens_futs.append(index_actor.set_corpus_num_tokens.remote(combined_corpus_num_tokens))
            set_token_doc_freqs_futs.append(index_actor.set_token_doc_freqs.remote(combined_token_doc_freqs))
        wait(set_index_size_futs)
        wait(set_corpus_num_tokens_futs)
        wait(set_token_doc_freqs_futs)
        recalculate_token_idfs_futs: List = []
        for index_actor_composite in self.index:
            index_actor: ray.actor.ActorHandle = index_actor_composite.actor
            recalculate_token_idfs_futs.append(index_actor.recalculate_token_idfs.remote())
        wait(recalculate_token_idfs_futs)

    def _retrieve_batch(
            self,
            queries_batch: ScalableSeries,
            top_k: int,
            retrieve_documents: bool,
    ) -> List[List[RankedResult]]:
        actors_top_k_results: Dict[str, List[List[Tuple[BM25IndexStoreDoc, float]]]] = {}
        for index_actor_composite in self.index:
            index_actor: ray.actor.ActorHandle = index_actor_composite.actor
            actors_top_k_results[index_actor_composite.actor_id] = index_actor.get_top_k_batch.remote(
                queries_batch,
                top_k=top_k,
            )
        actors_top_k_results: Dict[str, List[List[Tuple[BM25IndexStoreDoc, float]]]] = accumulate(actors_top_k_results)
        batch_ranked_results: List[List[RankedResult]] = []
        for all_actor_query_top_k_results in zip(*list(actors_top_k_results.values())):
            batch_ranked_results.append(
                self._merge_retrieved_results(
                    all_actor_query_top_k_results=all_actor_query_top_k_results,
                    top_k=top_k,
                    retrieve_documents=retrieve_documents,
                    how='max',
                )
            )
        assert len(batch_ranked_results) == len(queries_batch)
        return batch_ranked_results

    def _merge_retrieved_results(
            self,
            all_actor_query_top_k_results: Tuple[List[Tuple[BM25IndexStoreDoc, float]], ...],
            top_k: int,
            retrieve_documents: bool,
            how: Literal['max', 'min'],
    ) -> List[RankedResult]:
        ## For a particular query, sort the top-k results across actors.
        assert how in {'max', 'min'}
        query_top_k_results: List[Tuple[BM25IndexStoreDoc, float]] = []
        for actor_query_top_k_results in all_actor_query_top_k_results:
            query_top_k_results.extend(actor_query_top_k_results)
        query_top_k_results: List[Tuple[BM25IndexStoreDoc, float]] = sorted(
            query_top_k_results, key=lambda x: x[1], reverse=(True if how == 'max' else False)
        )
        query_ranked_results: List[RankedResult] = []
        for k, (top_k_doc, top_k_score) in enumerate(query_top_k_results):
            k: int = k + 1
            doc: Optional[Dict] = None
            if retrieve_documents:
                doc: Any = top_k_doc.doc
            query_ranked_results.append(
                RankedResult.of(dict(
                    rank=k,
                    document_id=top_k_doc.doc_id,
                    document=doc,
                    distance=float(top_k_score),
                    distance_metric=self.params.distance_metric,
                ))
            )
        return query_ranked_results[:top_k]


class SparseRetriever(Retriever):
    index: Optional[SparseRetrievalIndex] = None

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f'{self.class_name} {f"using {str(self.index)}" if self.index is not None else "(no Index)"}'

    class Hyperparameters(Algorithm.Hyperparameters):
        index: Optional[Dict] = None  ## Params for index

    def initialize(self, model_dir: Optional[FileMetadata] = None):
        if self.index is None and self.hyperparams.index is None:
            raise ValueError(
                f'To initialize {self.class_name}, you must either pass an index explicitly or set the `index` '
                f'hyperparam with a dict of parameters that can be used to initialize an index.'
            )
        elif self.index is None:
            with Timer(task='Creating Index'):
                self.index = SparseRetrievalIndex.of(**self.hyperparams.index)

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
        kwargs.pop('progress_bar', None)
        kwargs['batch_size']: Optional[int] = get_default(kwargs.pop('batch_size', None), self.hyperparams.batch_size)
        ranked_results: List[List[RankedResult]] = self.index.retrieve(
            batch,
            top_k=top_k,
            retrieve_documents=retrieve_documents,
            **kwargs
        )
        return {
            'ranked_results': ranked_results
        }

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


class RandomRetriever(Retriever):
    corpus: Optional[RetrievalCorpus] = None
    current_retrieval_seed: Optional[int] = None

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f'{self.class_name} {f"using {str(self.corpus)}" if self.corpus is not None else "(no Corpus)"}'

    class Hyperparameters(Algorithm.Hyperparameters):
        corpus: Optional[RetrievalCorpus] = None  ## Params for index
        corpus_read_as: DataLayout = DataLayout.DASK
        corpus_schema_columns_only: bool = False
        default_distance: confloat(ge=0) = 0.0
        default_distance_metric: Any = 'random_distance_metric'
        retrieval_seed: Optional[int] = None

    def initialize(self, model_dir: Optional[FileMetadata] = None):
        if self.corpus is None and self.hyperparams.corpus is None:
            raise ValueError(
                f'To initialize {self.class_name}, you must either pass an corpus explicitly or '
                f'set the `corpus` hyperparam with a dict of parameters that can be used to '
                f'initialize the corpus.'
            )
        elif self.corpus is None:
            with Timer(task='Reading Retrieval Corpus'):
                corpus: RetrievalCorpus = self.hyperparams.corpus
                self.corpus: RetrievalCorpus = corpus.read(
                    read_as=self.hyperparams.corpus_read_as,
                    persist=True,
                    schema_columns_only=self.hyperparams.corpus_schema_columns_only,
                )
        if self.hyperparams.retrieval_seed is not None:
            self.current_retrieval_seed: int = self.hyperparams.retrieval_seed

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
        kwargs.pop('progress_bar', None)
        ranked_results: List[List[RankedResult]] = self._random_retrieve(
            batch,
            top_k=top_k,
            retrieve_documents=retrieve_documents,
            **kwargs
        )
        return {
            'ranked_results': ranked_results
        }

    def _random_retrieve(
            self,
            queries: Queries,
            *,
            top_k: int,
            retrieve_documents: bool,
            **kwargs
    ) -> List[List[RankedResult]]:
        if self.current_retrieval_seed is not None:
            self.current_retrieval_seed += 1  ## Get different results each time.
        total_num_results: int = len(queries) * top_k
        corpus_random_sample: RetrievalCorpus = self.corpus.sample(
            n=total_num_results,
            seed=self.current_retrieval_seed,
            fetch_partitions=0,
            stream_as=DataLayout.PANDAS,
        )
        assert len(corpus_random_sample) == total_num_results
        index_col: str = self.corpus.data_schema.index_col
        random_ranked_results: List[List[RankedResult]] = []
        for top_k_random_results in corpus_random_sample.iter(batch_size=top_k):
            random_ranked_results.append([])
            for k, top_k_random_result in enumerate(
                    top_k_random_results.iter(
                        batch_size=1,
                        stream_as=DataLayout.LIST_OF_DICT,
                        shuffle=False,
                    )
            ):
                k: int = k + 1
                doc: Dict = top_k_random_result.data.to_record()
                random_ranked_results[-1].append(
                    RankedResult.of(dict(
                        rank=k,
                        document_id=str(doc[index_col]),
                        document=doc if retrieve_documents else None,
                        distance=self.hyperparams.default_distance,
                        distance_metric=self.hyperparams.default_distance_metric,
                    ))
                )
        return random_ranked_results

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

#
#
# class BM25Okapi(BM25):
#     k1: float = 1.5
#     b: float = 0.75
#     epsilon: float = 0.25
#
#     def _calc_idf(self, nd):
#         """
#         Calculates frequencies of terms in documents and in corpus.
#         This algorithm sets a floor on the idf values to eps * average_idf
#         """
#         # collect idf sum to calculate an average idf for epsilon value
#         idf_sum = 0
#         # collect words with negative idf to set them a special epsilon value.
#         # idf can be negative if word is contained in more than half of documents
#         negative_idfs = []
#         for word, freq in nd.items():
#             idf = math.log(self.index_size - freq + 0.5) - math.log(freq + 0.5)
#             self.idf[word] = idf
#             idf_sum += idf
#             if idf < 0:
#                 negative_idfs.append(word)
#         self.average_idf = idf_sum / len(self.idf)
#
#         eps = self.epsilon * self.average_idf
#         for word in negative_idfs:
#             self.idf[word] = eps
#
#     def get_scores(self, query):
#         """
#         The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
#         this algorithm also adds a floor to the idf value of epsilon.
#         See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
#         :param query:
#         :return:
#         """
#         score = np.zeros(self.index_size)
#         doc_len = np.array(self.doc_len)
#         for q in query:
#             q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
#             score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
#                                                (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
#         return score
#
#     def get_batch_scores(self, query, doc_ids):
#         """
#         Calculate bm25 scores between query and subset of all docs
#         """
#         assert all(di < len(self.doc_freqs) for di in doc_ids)
#         score = np.zeros(len(doc_ids))
#         doc_len = np.array(self.doc_len)[doc_ids]
#         for q in query:
#             q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
#             score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
#                                                (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
#         return score.tolist()
#
#
# class BM25L(BM25):
#     def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=0.5):
#         # Algorithm specific parameters
#         self.k1 = k1
#         self.b = b
#         self.delta = delta
#         super().__init__(corpus, tokenizer)
#
#     def _calc_idf(self, nd):
#         for word, freq in nd.items():
#             idf = math.log(self.index_size + 1) - math.log(freq + 0.5)
#             self.idf[word] = idf
#
#     def get_scores(self, query):
#         score = np.zeros(self.index_size)
#         doc_len = np.array(self.doc_len)
#         for q in query:
#             q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
#             ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
#             score += (self.idf.get(q) or 0) * q_freq * (self.k1 + 1) * (ctd + self.delta) / \
#                      (self.k1 + ctd + self.delta)
#         return score
#
#     def get_batch_scores(self, query, doc_ids):
#         """
#         Calculate bm25 scores between query and subset of all docs
#         """
#         assert all(di < len(self.doc_freqs) for di in doc_ids)
#         score = np.zeros(len(doc_ids))
#         doc_len = np.array(self.doc_len)[doc_ids]
#         for q in query:
#             q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
#             ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
#             score += (self.idf.get(q) or 0) * q_freq * (self.k1 + 1) * (ctd + self.delta) / \
#                      (self.k1 + ctd + self.delta)
#         return score.tolist()
#
#
# class BM25Plus(BM25):
#     def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=1):
#         # Algorithm specific parameters
#         self.k1 = k1
#         self.b = b
#         self.delta = delta
#         super().__init__(corpus, tokenizer)
#
#     def _calc_idf(self, nd):
#         for word, freq in nd.items():
#             idf = math.log((self.index_size + 1) / freq)
#             self.idf[word] = idf
#
#     def get_scores(self, query):
#         score = np.zeros(self.index_size)
#         doc_len = np.array(self.doc_len)
#         for q in query:
#             q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
#             score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
#                                                (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
#         return score
#
#     def get_batch_scores(self, query, doc_ids):
#         """
#         Calculate bm25 scores between query and subset of all docs
#         """
#         assert all(di < len(self.doc_freqs) for di in doc_ids)
#         score = np.zeros(len(doc_ids))
#         doc_len = np.array(self.doc_len)[doc_ids]
#         for q in query:
#             q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
#             score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
#                                                (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
#         return score.tolist()
#
#
# # BM25Adpt and BM25T are a bit more complicated than the previous algorithms here. Here a term-specific k1
# # parameter is calculated before scoring is done
#
# # class BM25Adpt(BM25):
# #     def __init__(self, corpus, k1=1.5, b=0.75, delta=1):
# #         # Algorithm specific parameters
# #         self.k1 = k1
# #         self.b = b
# #         self.delta = delta
# #         super().__init__(corpus)
# #
# #     def _calc_idf(self, nd):
# #         for word, freq in nd.items():
# #             idf = math.log((self.index_size + 1) / freq)
# #             self.idf[word] = idf
# #
# #     def get_scores(self, query):
# #         score = np.zeros(self.index_size)
# #         doc_len = np.array(self.doc_len)
# #         for q in query:
# #             q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
# #             score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
# #                                                (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
# #         return score
# #
# #
# # class BM25T(BM25):
# #     def __init__(self, corpus, k1=1.5, b=0.75, delta=1):
# #         # Algorithm specific parameters
# #         self.k1 = k1
# #         self.b = b
# #         self.delta = delta
# #         super().__init__(corpus)
# #
# #     def _calc_idf(self, nd):
# #         for word, freq in nd.items():
# #             idf = math.log((self.index_size + 1) / freq)
# #             self.idf[word] = idf
# #
# #     def get_scores(self, query):
# #         score = np.zeros(self.index_size)
# #         doc_len = np.array(self.doc_len)
# #         for q in query:
# #             q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
# #             score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
# #                                                (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
# #         return score
#
