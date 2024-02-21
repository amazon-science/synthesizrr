from typing import *
from abc import ABC, abstractmethod
import types, time, traceback, pickle, gc, os, json, numpy as np, math
import pandas as pd
from synthesizrr.base.constants import FileFormat, DataLayout, MLType, DataSplit, Task, Storage, Parallelize
from synthesizrr.base.util import Registry, MutableParameters, Parameters, set_param_from_alias, is_list_like, as_list, \
    random_sample, safe_validate_arguments, get_default, StringUtil, MLTypeSchema, AutoEnum, auto, alias, Timer, \
    not_impl, check_isinstance, keep_keys, keep_values, FileSystemUtil, type_str, parameterized_flatten, \
    punct_normalize, accumulate_iter
from synthesizrr.base.util.concurrency import run_asyncio, run_concurrent, run_parallel, run_parallel_ray, wait, wait_if_future, \
    accumulate, get_result, ThreadPoolExecutor, stop_executor, Future, dispatch_executor, dispatch
from synthesizrr.base.data import FileMetadata, ScalableDataFrame, ScalableSeries, Reader, ScalableDataFrameOrRaw, to_sdf
from synthesizrr.base.framework.task_data import Dataset, Datasets, save_dataset, load_dataset
from synthesizrr.base.framework.predictions import Predictions, save_predictions, load_predictions
from synthesizrr.base.framework.task import Prompts, NextTokens
from synthesizrr.base.util.aws import S3Util
from synthesizrr.base.framework import Algorithm
from synthesizrr.base.framework.task.classification import ClassificationData
from synthesizrr.base.framework.task.embedding import EmbeddingData, Embeddings
from synthesizrr.base.framework.task.retrieval import Queries, RankedResults, RetrievalCorpus
from synthesizrr.base.framework.task.text_generation import TextInputs, NextTokens, TextGenerations, FewShotTextGenerator, \
    TextGenerationsPredictionsBase
from synthesizrr.base.framework.evaluator import Evaluator, LoadBalancingStrategy
from synthesizrr.base.framework.chain.Chain import Step, Chain
from functools import partial
from pydantic import root_validator, Extra, conint, confloat, constr
from pydantic.typing import Literal

from synthesizrr.expt.data import SynthesizRRDataset
from synthesizrr.expt.common import CachedResultsStep, IDX_COL, LABEL_TEXT_COL, LABEL_VERBALIZATION_COL, EXAMPLE_TEXT_COL, \
    QUERY_TEXT_COL, RETRIEVED_TOP_K_COL, RETRIEVED_CONTEXT_COL, DISTANCE_COL, DISTANCE_METRIC_COL, \
    EFS_HUGGINGFACE_CACHE_DIR, DEFAULT_SEED_SET_DATA_SPLIT, DEFAULT_SEED_SET_STRATIFY_ON_GROUND_TRUTH, \
    DEFAULT_SEED, Experiment, expand_num_samples_per_label, DatasetName, Corpus, ModelName, Retriever, \
    count_num_tokens, shorten, get_templates_and_hashes, calc_label_dist


class EmbedCorpus(CachedResultsStep):

    @safe_validate_arguments
    def run(
            self,
            *,
            results_dir: FileMetadata,
            corpus: Corpus,  ## E.g. AmazonProducts
            corpus_raw_text_dir: FileMetadata,
            retriever: Retriever,  ## E.g. Contriever
            embedding_num_models: conint(ge=1) = 80,
            embedding_batch_size: conint(ge=1) = 256,
            embedding_batches_per_save: conint(ge=1) = 30,
            **kwargs,
    ) -> Dict:
        if corpus_raw_text_dir.data_schema is None:
            raise ValueError(f'Expected `corpus_raw_text_dir` to have schema, found None.')
        if len(keep_values(corpus_raw_text_dir.data_schema, MLType.TEXT)) != 1:
            raise ValueError(f'Expected `corpus_raw_text_dir` to have exactly one text column; found ')
        if corpus_raw_text_dir.format is None:
            raise ValueError(f'Expected `corpus_raw_text_dir` to have file format, found None.')

        # input_schema = {
        #     'uid': 'index',
        #
        #     'article_headline': 'object',
        #     'article_synopsis': 'object',
        #     'article_text': 'text',
        #
        #     'article_url': 'object',
        #     'article_first_published_timestamp': 'object',
        #     'article_last_modified_timestamp': 'object',
        #     'article_topic': 'object',
        #     'article_topic_depth=1': 'object',
        #     'article_topic_depth=2': 'object',
        #     'article_topic_depth=3': 'object',
        #     'article_topic_depth=4': 'object',
        #
        #     'archive_date': 'object',
        #     'archive_date_timestamp': 'object',
        #     'archive_starttime': 'object',
        #     'archive_url': 'object',
        # }

        corpus_embeddings_dir: Optional[FileMetadata] = None
        embedder: Optional[Evaluator] = self._create_embedder(
            embedder=retriever,
            num_models=embedding_num_models,
            batch_size=embedding_batch_size,
            verbosity=self.verbosity,
        )
        if embedder is not None:
            corpus_embeddings_dir: FileMetadata = self.save_to(
                results_dir=results_dir,
                corpus=corpus,
                retriever=retriever,
            )
            if len(corpus_embeddings_dir.list()) == 0:
                self.info(f'Embeddings dir "{corpus_embeddings_dir.path}" is empty, running the embedder...')
                try:
                    with Timer(
                            f'Reading embedding input data from "{corpus_raw_text_dir.path}"',
                            logger=self.info
                    ):
                        corpus_df: ScalableDataFrame = Reader.of(corpus_raw_text_dir.format).read_metadata(
                            corpus_raw_text_dir,
                            read_as=DataLayout.DASK,
                        ).persist(wait=True)
                    emb_input_data: EmbeddingData = Dataset.of(
                        task=Task.EMBEDDING,
                        split=DataSplit.UNSUPERVISED,
                        data=corpus_df,
                        data_schema=corpus_raw_text_dir.data_schema,
                    )
                    embedder.evaluate(
                        emb_input_data,
                        sharding_strategy='coarse',
                        data_loading_strategy='local',
                        save_to=corpus_embeddings_dir,
                        cache_dir=EFS_HUGGINGFACE_CACHE_DIR,
                        device='cuda',
                        batches_per_save=embedding_batches_per_save,
                    )
                    self.info(f'...done embedding {corpus} using {retriever}; saved to: "{corpus_embeddings_dir.path}"')
                finally:
                    del corpus_df
                    embedder.stop()
                    del embedder

            # efs_corpus_embeddings_dir: FileMetadata = corpus_embeddings_dir.update_params(
            #     path=corpus_embeddings_dir.path.replace(results_dir.path, efs_results_dir.path)
            # )
            # efs_corpus_embeddings_fpaths: List[str] = get_result(run_parallel_ray(
            #     embed_corpus_list_efs,
            #     efs_corpus_embeddings_dir=efs_corpus_embeddings_dir,
            # ))
            # if len(efs_corpus_embeddings_fpaths) > 0:
            #     self.info(
            #         f'Found {len(efs_corpus_embeddings_fpaths)} files at "{efs_corpus_embeddings_dir.path}":'
            #         f'\n{efs_corpus_embeddings_fpaths}'
            #     )
            # else:
            #     self.info(f'Did not find any files at "{efs_corpus_embeddings_dir.path}".')
            #     if corpus_embeddings_dir.storage is Storage.S3:
            #         ## Copy to EFS:
            #         self.info(
            #             f'Copying embeddings from: "{corpus_embeddings_dir.path}" '
            #             f'to EFS path: "{efs_corpus_embeddings_dir.path}"'
            #         )
            #         get_result(run_parallel_ray(
            #             embed_corpus_copy_to_efs,
            #             corpus_embeddings_dir=corpus_embeddings_dir,
            #             efs_corpus_embeddings_dir=efs_corpus_embeddings_dir,
            #         ))
            #         self.info(
            #             f'Done copying embeddings from: "{corpus_embeddings_dir.path}" '
            #             f'to EFS path: "{efs_corpus_embeddings_dir.path}"'
            #         )
        return dict(
            corpus_embeddings_dir=corpus_embeddings_dir,
        )

    def save_to(
            self,
            *,
            results_dir: FileMetadata,  ## E.g. RESULTS_DIR/,
            corpus: Corpus,  ## E.g. AmazonReviews
            retriever: Retriever,  ## E.g. Contriever
            **kwargs,
    ) -> FileMetadata:
        ## RESULTS_DIR/retrieval-corpus/toi_without_datasets/contriever-embeddings/
        return results_dir.subdir_in_dir('retrieval-corpus', return_metadata=True) \
            .subdir_in_dir(corpus.canonical(), return_metadata=True) \
            .subdir_in_dir(f'{retriever.canonical()}-embeddings', return_metadata=True) \
            .update_params(file_format=FileFormat.PARQUET, file_glob='*.parquet')

    @classmethod
    def _create_embedder(
            cls,
            embedder: Retriever,
            num_models: int,
            batch_size: int,
            verbosity: int,
    ) -> Optional[Evaluator]:
        if embedder is Retriever.Contriever:
            return Evaluator.of(
                'ray',
                task=Task.EMBEDDING,
                num_models=num_models,
                resources_per_model=dict(cpu=2, gpu=1),
                algorithm='huggingface-TextEmbedder',
                hyperparams=dict(
                    model_name='facebook/contriever',
                    tokenizer_encode=dict(max_length=512),
                    batch_size=batch_size,
                ),
                verbosity=verbosity,
                cache_dir=EFS_HUGGINGFACE_CACHE_DIR,
            )
        elif embedder is Retriever.MiniLM:
            return Evaluator.of(
                'ray',
                task=Task.EMBEDDING,
                num_models=num_models,
                resources_per_model=dict(cpu=2, gpu=0.5),
                algorithm='huggingface-TextEmbedder',
                hyperparams=dict(
                    model_name='sentence-transformers/all-MiniLM-L12-v2',
                    tokenizer_encode=dict(max_length=512),
                    batch_size=batch_size,
                ),
                verbosity=verbosity,
                cache_dir=EFS_HUGGINGFACE_CACHE_DIR,
            )
        elif embedder is Retriever.BM25Okapi:
            return None  ## BM25 loads the corpus in memory on initializing the retriever, no need to embed beforehand.
        elif embedder is Retriever.Random:
            return None  ## Random retriever selects from the corpus randomly, no need to embed beforehand.
        else:
            raise not_impl('embedder', embedder)

    @staticmethod
    def embed_corpus_list_efs(efs_corpus_embeddings_dir: FileMetadata) -> List[str]:
        efs_corpus_embeddings_dir.mkdir()
        return efs_corpus_embeddings_dir.list()

    @staticmethod
    def embed_corpus_copy_to_efs(
            corpus_embeddings_dir: FileMetadata,
            efs_corpus_embeddings_dir: FileMetadata,
    ):
        if S3Util.copy_s3_dir_to_local(
                source_s3_dir=corpus_embeddings_dir.path,
                destination_local_dir=efs_corpus_embeddings_dir.path,
                force_download=True,
                wait_timeout=3 * 60 * 60,
        ) is False:
            raise ValueError(
                f'Could not copy embeddings from: '
                f'"{corpus_embeddings_dir.path}" '
                f'to EFS path: "{efs_corpus_embeddings_dir.path}"'
            )


class CreateSeedSet(CachedResultsStep):
    @safe_validate_arguments
    def run(
            self,
            *,
            results_dir: FileMetadata,
            dataset_name: DatasetName,
            seed_type: Literal['generated', 'train_set'],
            seed_size: int,
            seed_set_data_split: DataSplit = DEFAULT_SEED_SET_DATA_SPLIT,
            seed_set_stratify_on_ground_truth: bool = DEFAULT_SEED_SET_STRATIFY_ON_GROUND_TRUTH,
            seed: int = DEFAULT_SEED,
            label_col: Optional[str] = None,
            label_verbalizer: Dict[str, str],
            **kwargs,
    ) -> Dict:
        label_col: str = get_default(label_col, dataset_name.label_col())

        seed_set_file: FileMetadata = self.save_to(
            results_dir=results_dir,
            dataset_name=dataset_name,
            seed_type=seed_type,
            seed_size=seed_size,
            seed_set_data_split=seed_set_data_split,
            seed_set_stratify_on_ground_truth=seed_set_stratify_on_ground_truth,
            seed=seed,
        )
        if not seed_set_file.exists():
            self.info(f'Seed set does not exist at "{seed_set_file.path}", creating it...')
            if seed_type == 'train_set':
                seed_set: Dataset = dataset_name.create_seed_set(
                    seed_size=seed_size,
                    data_split=seed_set_data_split,
                    seed=seed,
                    stratify_on_ground_truth=seed_set_stratify_on_ground_truth,
                    label_col=label_col,
                    label_verbalizer=label_verbalizer,
                )
            elif seed_type == 'generated':
                ## TODO implement seed set generation.
                raise not_impl('seed_type', seed_type)
            else:
                raise not_impl('seed_type', seed_type)
            save_dataset(
                dataset=seed_set,
                dataset_destination=seed_set_file,
                overwrite=True,
            )
            self.info(f'...done creating seed set at "{seed_set_file.path}".')
        seed_set: Dataset = load_dataset(seed_set_file, retry=10, retry_wait=10)
        self.info(f'Seed set details:')
        self.info(seed_set)
        self.info(f'Seed set label distribution:')
        self.info(calc_label_dist(seed_set.data.pandas(), label_col=label_col))

        return dict(
            seed_set=seed_set,
        )

    def save_to(
            self,
            *,
            results_dir: FileMetadata,
            dataset_name: DatasetName,
            seed_type: Literal['generated', 'train_set'],
            seed_size: int,
            seed_set_data_split: DataSplit,
            seed_set_stratify_on_ground_truth: bool,
            seed: int,
            **kwargs,
    ) -> FileMetadata:
        ## RESULTS_DIR/seed-set/ag_news/train_set/train_set_seed_set-dataset=ag_news-seed_size=500-stratified=gt_stratified-seed=42.jsonlines
        if seed_type == 'generated':
            seed_type_str: str = f'generated_seed_set'
        elif seed_type == 'train_set':
            seed_type_str: str = f'{seed_set_data_split.lower()}_set_seed_set'
        else:
            raise not_impl('seed_type', seed_type)
        stratified_str: str = 'gt_stratified' if seed_set_stratify_on_ground_truth else 'not_gt_stratified'

        return results_dir.subdir_in_dir('seed-set', return_metadata=True) \
            .subdir_in_dir(dataset_name.canonical(), return_metadata=True) \
            .subdir_in_dir(seed_type, return_metadata=True) \
            .file_in_dir(
            f"{seed_type_str}"
            f"-dataset={dataset_name.canonical()}"
            f"-seed_size={seed_size}"
            f"-stratified={stratified_str}"
            f"-seed={seed}.parquet",
            return_metadata=True,
        ).update_params(file_format=FileFormat.PARQUET)


class RetrieveFromSeedSet(CachedResultsStep):

    @safe_validate_arguments
    def run(
            self,
            *,
            results_dir: FileMetadata,
            corpus: Corpus,
            retriever: Retriever,
            corpus_raw_text_dir: FileMetadata,
            corpus_embeddings_dir: Optional[FileMetadata],
            dataset_name: DatasetName,
            seed_set: ClassificationData,
            seed_type: Literal['generated', 'train_set'],
            seed_size: int,
            seed_set_stratify_on_ground_truth: bool,
            retrieval_top_k: conint(ge=1),
            query_col: Optional[str] = None,
            label_col: Optional[str] = None,
            seed_set_data_split: DataSplit = DEFAULT_SEED_SET_DATA_SPLIT,
            retriever_num_models: conint(ge=1) = 1,
            retriever_num_shards: conint(ge=1) = 64,
            retriever_batch_size: conint(ge=1) = 16,
            seed: int = DEFAULT_SEED,
            **kwargs,
    ) -> Dict:
        query_col: str = get_default(query_col, dataset_name.query_col())
        label_col: str = get_default(label_col, dataset_name.label_col())
        retr_input: Queries = Dataset.of(
            task=Task.RETRIEVAL,
            split=DataSplit.UNSUPERVISED,
            data=seed_set.data,
            data_schema={
                seed_set.data_schema.index_col: MLType.INDEX,
                query_col: MLType.TEXT,
            },
        ).read()
        if retriever is Retriever.Random:
            retriever_batch_size: int = len(retr_input)
        retr_output_path: FileMetadata = self.save_to(
            results_dir=results_dir,
            corpus=corpus,
            retriever=retriever,
            dataset_name=dataset_name,
            seed_type=seed_type,
            seed_size=seed_size,
            seed_set_stratify_on_ground_truth=seed_set_stratify_on_ground_truth,
            seed_set_data_split=seed_set_data_split,
            retrieval_top_k=retrieval_top_k,
        )
        if not retr_output_path.exists():
            ## Run the retriever.
            self.info(f'Retrieved results do not exist at "{retr_output_path.path}", running retriever...')

            retr_evaluator: Evaluator = self._create_retriever(
                corpus_raw_text_dir=corpus_raw_text_dir,
                corpus_embeddings_dir=corpus_embeddings_dir,
                retriever=retriever,
                num_models=retriever_num_models,
                num_shards=retriever_num_shards,
                batch_size=retriever_batch_size,
                verbosity=self.verbosity,
                seed=seed,
            )
            self.info(f'Retrieval evaluator:\n{retr_evaluator}')

            ## Run the retriever:
            try:
                retr_output: Predictions = retr_evaluator.evaluate(
                    retr_input,
                    top_k=retrieval_top_k,
                    batches_per_save=1,
                    batch_size=retriever_batch_size,
                    submission_batch_size=retriever_batch_size,
                    preds=True,
                    retrieve_documents=True,
                    load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN,
                )
                self.info(f'Saving {len(retr_output)} retrieved results to: "{retr_output_path.path}"...')
                save_predictions(
                    retr_output,
                    predictions_destination=retr_output_path,
                )
                self.info(f'...saved {len(retr_output)} retrieved results to: "{retr_output_path.path}".')
            finally:
                retr_evaluator.stop()
                del retr_evaluator

        ## Even if it was saved immediately before, re-load it:
        self.info(f'Loading top_k={retrieval_top_k} retrieved results from: "{retr_output_path.path}"')
        retr_output: RankedResults = load_predictions(retr_output_path, retry=10, retry_wait=10)
        retr_output: RankedResults = retr_output.flatten()

        self.info(f'\nRetrieved results (top_k={retrieval_top_k}):')
        self.info(retr_output.to_string(data_schema=False))
        # self.info(retr_output.data.pandas())  ## TOO BIG, DON'T LOG!

        return dict(
            seed_set_retr_input=retr_input,
            seed_set_retr_output=retr_output,
        )

    def save_to(
            self,
            *,
            results_dir: FileMetadata,  ## E.g. RESULTS_DIR/,
            corpus: Corpus,
            retriever: Retriever,
            dataset_name: DatasetName,  ## E.g. ag_news
            seed_type: Literal['generated', 'train_set'],
            seed_size: int,
            seed_set_stratify_on_ground_truth: bool,
            seed_set_data_split: DataSplit,
            retrieval_top_k: int,
            **kwargs,
    ) -> FileMetadata:
        ## RESULTS_DIR/retrieval-augmented-dataset-generation/ag_news/retrieval-data/toi_without_datasets/ag_news_seed_retr_output_toi_without_datasets.jsonlines
        if seed_type == 'generated':
            seed_type_str: str = f'generated_seed_set'
        elif seed_type == 'train_set':
            seed_type_str: str = f'{seed_set_data_split.lower()}_set_seed_set'
        else:
            raise not_impl('seed_type', seed_type)

        seed_size_str = ''
        if seed_size != dataset_name.seed_size():
            seed_size_str: str = f"-seed_size={seed_size}"
        stratified_str = ''
        if seed_set_stratify_on_ground_truth != DEFAULT_SEED_SET_STRATIFY_ON_GROUND_TRUTH:
            stratified_str: str = 'gt_stratified' if seed_set_stratify_on_ground_truth else 'not_gt_stratified'
            stratified_str: str = f"-stratified={stratified_str}"

        return results_dir.subdir_in_dir('retrieval-augmented-dataset-generation', return_metadata=True) \
            .subdir_in_dir(dataset_name.canonical(), return_metadata=True) \
            .subdir_in_dir('retrieval-data', return_metadata=True) \
            .subdir_in_dir(corpus.canonical(), return_metadata=True) \
            .subdir_in_dir(retriever.canonical(), return_metadata=True) \
            .subdir_in_dir(f'retrieval_top_k={retrieval_top_k}', return_metadata=True) \
            .file_in_dir(
            f"{seed_type_str}{seed_size_str}{stratified_str}-retr_output"
            f"-dataset={dataset_name.canonical()}"
            f"-corpus={corpus.canonical()}"
            f"-retriever={retriever.canonical()}"
            f"-retrieval_top_k={retrieval_top_k}"
            f".jsonl",  ## Saving this as Parquet does not work due to the nested JSON
            return_metadata=True,
        ).update_params(file_format=FileFormat.JSONLINES)

    def _create_retriever(
            self,
            *,
            corpus_raw_text_dir: FileMetadata,
            corpus_embeddings_dir: Optional[FileMetadata],
            retriever: Retriever,
            verbosity: int,
            num_models: int,
            num_shards: int,
            batch_size: int,
            seed: int,
    ) -> Evaluator:
        embedder_params: Optional[Dict] = None
        embedder_ndim: Optional[int] = None
        if retriever is Retriever.Contriever:
            embedder_params: Dict = dict(
                algorithm='huggingface-TextEmbedder',
                hyperparams=dict(
                    model_name='facebook/contriever',
                    tokenizer_encode=dict(max_length=512),
                    batch_size=128,
                )
            )
            embedder_ndim: int = 768
            resources_per_model: Dict[str, int] = dict(cpu=50)
        elif retriever is Retriever.MiniLM:
            embedder_params: Dict = dict(
                algorithm='huggingface-TextEmbedder',
                hyperparams=dict(
                    model_name='sentence-transformers/all-MiniLM-L6-v2',
                    tokenizer_encode=dict(max_length=512),
                    batch_size=512,
                )
            )
            embedder_ndim: int = 384
            resources_per_model: Dict[str, int] = dict(cpu=50)
        elif retriever is Retriever.BM25Okapi:
            resources_per_model: Dict[str, int] = dict(cpu=6)
        elif retriever is Retriever.Random:
            resources_per_model: Dict[str, int] = dict(cpu=6)
        else:
            raise not_impl('retriever', retriever)
        check_isinstance(resources_per_model, dict)
        if retriever.is_dense():
            check_isinstance(corpus_embeddings_dir, FileMetadata)
            assert embedder_params is not None
            assert embedder_ndim is not None
            ## Dense retriever:
            retrieval_index_params: Dict = dict(
                name='faiss',
                params=dict(
                    name='IndexFlatIP',
                    ndim=embedder_ndim,
                    distance_metric='cosine_similarity',
                ),
                data=corpus_embeddings_dir,
                read_as=DataLayout.PANDAS,
                parallelize=Parallelize.threads,
                retry=10,
                retry_wait=10,  ## Seconds
            )
            retriever: Evaluator = Evaluator.of(
                'ray',
                task=Task.RETRIEVAL,
                algorithm='DenseRetriever',
                num_models=num_models,
                resources_per_model=resources_per_model,
                cache_dir=EFS_HUGGINGFACE_CACHE_DIR,
                hyperparams=dict(
                    batch_size=batch_size,
                    embedder=embedder_params,
                    index=retrieval_index_params
                ),
                verbosity=verbosity,
            )
        elif retriever.is_sparse():
            if retriever is not Retriever.BM25Okapi:
                raise not_impl('retriever', retriever, Retriever.BM25Okapi)
            from nltk import word_tokenize
            retrieval_corpus: RetrievalCorpus = Dataset.of(
                task=Task.RETRIEVAL_CORPUS,
                split=DataSplit.UNSUPERVISED,
                data=corpus_raw_text_dir,
                data_schema=corpus_raw_text_dir.data_schema,
            )
            retrieval_index_params: Dict = dict(
                name='BM25-Ray',
                params=dict(
                    name='Okapi',
                    tokenizer=word_tokenize,
                    store_documents=True,
                    num_shards=num_shards,
                    shard_num_cpus=6,
                ),
                data=retrieval_corpus,
                # read_as='pandas',
                # parallelize='threads',
                # persist=True,
                # retry=10,

                schema_columns_only=True,
                # indexing_parallelize='ray',
                indexing_iter_files=True,
                # indexing_batch_size=30_000,
                # indexing_max_workers=6,
                indexing_progress_bar=(verbosity >= 2),
            )

            retriever: Evaluator = Evaluator.of(
                'ray',
                task=Task.RETRIEVAL,
                algorithm='SparseRetriever',
                num_models=num_models,
                resources_per_model=resources_per_model,
                hyperparams=dict(
                    batch_size=batch_size,
                    index=retrieval_index_params,
                ),
                verbosity=verbosity,
            )
        else:
            if retriever is not Retriever.Random:
                raise not_impl('retriever', retriever, Retriever.Random)
            retrieval_corpus: RetrievalCorpus = Dataset.of(
                task=Task.RETRIEVAL_CORPUS,
                split=DataSplit.UNSUPERVISED,
                data=corpus_raw_text_dir,
                data_schema=corpus_raw_text_dir.data_schema,
            )

            retriever: Evaluator = Evaluator.of(
                'ray',
                task=Task.RETRIEVAL,
                algorithm='RandomRetriever',
                num_models=num_models,
                resources_per_model=resources_per_model,
                hyperparams=dict(
                    batch_size=batch_size,
                    corpus=retrieval_corpus,
                    corpus_read_as=DataLayout.DASK,
                    corpus_schema_columns_only=False,
                    retrieval_seed=seed,
                ),
                verbosity=verbosity,
            )
        return retriever


class CreateInputDatasets(CachedResultsStep, ABC):
    @classmethod
    def _save_labelwise_seed_icl_and_text_inputs(
            cls,
            seed_set: ClassificationData,
            *,
            seed_set_data_split: DataSplit,
            label_verbalizer: Dict[str, str],
            label_col: str,
            text_col: str,
            model_name: ModelName,
            label_text: str,
            max_tokens: int,
            num_samples_per_label: conint(ge=1),
            labelwise_icl_dataset_file: FileMetadata,
            labelwise_fewgen_dataset_file: FileMetadata,
            seed: int,
            verbosity: int,
    ) -> Tuple[ClassificationData, TextInputs]:
        num_labels: int = len(label_verbalizer)
        label_seed_set: ClassificationData = seed_set.filter(
            lambda row: row[label_col] == label_text
        )
        labelwise_icl_dataset: List[Dict] = []
        tokenizer: Any = model_name.tokenizer()
        for idx, ex in enumerate(label_seed_set.data.to_list_of_dict()):
            d: Dict = dict()
            d[EXAMPLE_TEXT_COL] = shorten(
                ex[text_col],
                max_tokens=max_tokens,
                tokenizer=tokenizer,
            )
            d[LABEL_TEXT_COL] = label_text
            d[LABEL_VERBALIZATION_COL] = label_verbalizer[label_text]
            d[IDX_COL] = f"ex={idx}" \
                         f"-model_name={model_name.canonical()}" \
                         f"-label={d[LABEL_TEXT_COL]}"
            labelwise_icl_dataset.append(d)

        labelwise_icl_dataset: ClassificationData = Dataset.of(
            task=Task.MULTI_CLASS_CLASSIFICATION if num_labels >= 3 else Task.BINARY_CLASSIFICATION,
            split=seed_set_data_split,
            data=to_sdf(labelwise_icl_dataset).pandas(),
            data_schema={
                IDX_COL: MLType.INDEX,
                EXAMPLE_TEXT_COL: MLType.TEXT,
                LABEL_TEXT_COL: MLType.GROUND_TRUTH,
                LABEL_VERBALIZATION_COL: MLType.TEXT,
            },
        )
        if len(labelwise_icl_dataset) < 32:
            raise ValueError(
                f'Insufficient number of ICL examples within label={label_text}; '
                f'found only {len(labelwise_icl_dataset)} ICL examples.'
            )

        labelwise_fewgen_dataset: List[Dict] = []
        for idx in range(num_samples_per_label):
            d: Dict = dict()
            d[LABEL_TEXT_COL] = label_text
            d[LABEL_VERBALIZATION_COL] = label_verbalizer[label_text]
            d[IDX_COL] = f"ex={idx}" \
                         f"-model_name={model_name.canonical()}" \
                         f"-label={d[LABEL_TEXT_COL]}"
            labelwise_fewgen_dataset.append(d)

        labelwise_fewgen_dataset: TextInputs = Dataset.of(
            task=Task.IN_CONTEXT_LEARNING,
            split=DataSplit.UNSUPERVISED,
            data=to_sdf(labelwise_fewgen_dataset).pandas(),
            data_schema={
                IDX_COL: MLType.INDEX,
                LABEL_TEXT_COL: MLType.CATEGORICAL,  ## Make it categorical for FewGen.
                LABEL_VERBALIZATION_COL: MLType.TEXT
            }
        )
        save_dataset(
            dataset=labelwise_icl_dataset,
            dataset_destination=labelwise_icl_dataset_file,
            overwrite=True,
        )
        # self.info(
        #     f'>> Saved {len(labelwise_icl_dataset)} ICL examples '
        #     f'to "{labelwise_icl_dataset_file.path}"'
        # )
        save_dataset(
            dataset=labelwise_fewgen_dataset,
            dataset_destination=labelwise_fewgen_dataset_file,
            overwrite=True,
        )
        # self.info(
        #     f'>> Saved {len(labelwise_fewgen_dataset)} FewGen examples '
        #     f'to "{labelwise_fewgen_dataset_file.path}"'
        # )
        return labelwise_icl_dataset, labelwise_fewgen_dataset

    @classmethod
    def _save_labelwise_retrieved_icl_and_text_inputs(
            cls,
            seed_set_retr_output: RankedResults,
            *,
            seed_set_data_split: DataSplit,
            label_verbalizer: Dict[str, str],
            corpus: Corpus,
            retriever: Retriever,
            query_col: str,
            context_col: str,
            label_col: str,
            model_name: ModelName,
            label_text: str,
            icl_type: Literal['retrieved', 'curated', 'seed'],
            retr_icl_top_ks: List[int],
            retr_icl_distance_range: Tuple[float, float],
            retr_icl_token_range: Tuple[int, int],
            synthesizrr_top_k_range: range,
            synthesizrr_distance_range: Tuple[float, float],  ## (0.4, 0.9)
            synthesizrr_max_tokens: int,
            num_samples_per_label: Optional[conint(ge=1)],
            seed: int,
            labelwise_icl_dataset_file: FileMetadata,
            labelwise_synthesizrr_dataset_file: FileMetadata,
            verbosity: int,
    ) -> Tuple[ClassificationData, TextInputs]:
        assert len(retr_icl_distance_range) == 2
        assert len(synthesizrr_distance_range) == 2
        assert len(retr_icl_token_range) == 2

        if retriever is Retriever.Random:
            assert icl_type in {'curated', 'seed'}

        num_labels: int = len(label_verbalizer)
        label_seed_set_retr_output: RankedResults = seed_set_retr_output.filter(
            lambda row: row[label_col] == label_text
        )

        labelwise_icl_dataset: List[pd.DataFrame] = []
        labelwise_synthesizrr_dataset: List[pd.DataFrame] = []
        executor: Optional[Any] = dispatch_executor(
            max_workers=min(20, max(1, len(label_seed_set_retr_output) // 10)),  ## At least 10 examples per process
            parallelize=Parallelize.processes,
        )
        try:
            futs: List[Any] = [
                dispatch(
                    cls._save_labelwise_retrieved_icl_and_text_inputs_get_ex_datasets,
                    ex=ex,
                    idx=idx,
                    corpus=corpus,
                    retriever=retriever,
                    query_col=query_col,
                    context_col=context_col,
                    label_col=label_col,
                    model_name=model_name,
                    label_text=label_text,
                    icl_type=icl_type,
                    retr_icl_top_ks=retr_icl_top_ks,
                    retr_icl_distance_range=retr_icl_distance_range,
                    retr_icl_token_range=retr_icl_token_range,
                    synthesizrr_top_k_range=synthesizrr_top_k_range,
                    synthesizrr_distance_range=synthesizrr_distance_range,
                    synthesizrr_max_tokens=synthesizrr_max_tokens,
                    parallelize=Parallelize.processes,
                    executor=executor,
                )
                for idx, ex in enumerate(label_seed_set_retr_output.data.to_list_of_dict())
            ]
            for ex_labelwise_icl_dataset_df, ex_labelwise_synthesizrr_dataset_df in accumulate_iter(
                    futs,
                    progress=dict(
                        desc='SynthesizRR inputs',
                        unit='example',
                        total=len(label_seed_set_retr_output)
                    ) if verbosity >= 2 else False
            ):
                labelwise_icl_dataset.append(ex_labelwise_icl_dataset_df)
                labelwise_synthesizrr_dataset.append(ex_labelwise_synthesizrr_dataset_df)
        finally:
            stop_executor(executor)

        labelwise_icl_dataset: ClassificationData = Dataset.of(
            task=Task.MULTI_CLASS_CLASSIFICATION if num_labels >= 3 else Task.BINARY_CLASSIFICATION,
            split=seed_set_data_split,
            data=ScalableDataFrame.concat(labelwise_icl_dataset).pandas(),
            data_schema={
                IDX_COL: MLType.INDEX,
                QUERY_TEXT_COL: MLType.TEXT,
                EXAMPLE_TEXT_COL: MLType.TEXT,
                RETRIEVED_TOP_K_COL: MLType.INT,
                RETRIEVED_CONTEXT_COL: MLType.TEXT,
                LABEL_TEXT_COL: MLType.GROUND_TRUTH,  ## Label is ground-truth for the ICL dataset.
                LABEL_VERBALIZATION_COL: MLType.TEXT
            },
        )
        if len(labelwise_icl_dataset) < 3:
            raise ValueError(
                f'Insufficient number of ICL examples within label={label_text}; '
                f'found only {len(labelwise_icl_dataset)} ICL examples.'
            )

        labelwise_synthesizrr_dataset: TextInputs = Dataset.of(
            task=Task.IN_CONTEXT_LEARNING,
            split=DataSplit.UNSUPERVISED,
            data=ScalableDataFrame.concat(labelwise_synthesizrr_dataset).pandas(),
            data_schema={
                IDX_COL: MLType.INDEX,
                QUERY_TEXT_COL: MLType.TEXT,
                EXAMPLE_TEXT_COL: MLType.TEXT,
                RETRIEVED_TOP_K_COL: MLType.INT,
                RETRIEVED_CONTEXT_COL: MLType.TEXT,
                LABEL_TEXT_COL: MLType.CATEGORICAL,  ## Make it categorical for the SynthesizRR.
                LABEL_VERBALIZATION_COL: MLType.TEXT
            }
        )
        if num_samples_per_label is not None:
            ## Only subsample the rewriting dataset, not the ICL dataset.
            labelwise_synthesizrr_dataset: TextInputs = labelwise_synthesizrr_dataset.sample(
                n=num_samples_per_label,
                seed=seed,
            )
        save_dataset(
            dataset=labelwise_icl_dataset,
            dataset_destination=labelwise_icl_dataset_file,
            overwrite=True,
        )
        # self.info(
        #     f'>> Saved {len(labelwise_icl_dataset)} ICL examples '
        #     f'to "{labelwise_icl_dataset_file.path}"'
        # )
        save_dataset(
            dataset=labelwise_synthesizrr_dataset,
            dataset_destination=labelwise_synthesizrr_dataset_file,
            overwrite=True,
        )
        # self.info(
        #     f'>> Saved {len(labelwise_synthesizrr_dataset)} SynthesizRR examples '
        #     f'to "{labelwise_synthesizrr_dataset_file.path}"'
        # )
        return labelwise_icl_dataset, labelwise_synthesizrr_dataset

    @classmethod
    def _save_labelwise_retrieved_icl_and_text_inputs_get_ex_datasets(
            cls,
            ex: Dict,
            *,
            idx: int,
            corpus: Corpus,
            retriever: Retriever,
            query_col: str,
            context_col: str,
            label_col: str,
            model_name: ModelName,
            label_text: str,
            icl_type: Literal['retrieved', 'curated', 'seed'],
            retr_icl_top_ks: List[int],
            retr_icl_distance_range: Tuple[float, float],
            retr_icl_token_range: Tuple[int, int],
            synthesizrr_top_k_range: range,
            synthesizrr_distance_range: Tuple[float, float],  ## (0.4, 0.9)
            synthesizrr_max_tokens: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        retr_icl_min_tokens, retr_icl_max_tokens = retr_icl_token_range
        retr_icl_top_ks: List[int] = sorted(retr_icl_top_ks)
        top_ks: List[int] = sorted(set(
            retr_icl_top_ks + [max(retr_icl_top_ks) + 1] + list(synthesizrr_top_k_range)
        ))

        context_top_k_col: Callable = lambda top_k: f'top_{top_k}_{context_col}'
        distance_top_k_col: Callable = lambda top_k: f'top_{top_k}_{DISTANCE_COL}'
        distance_metric_top_k_col: Callable = lambda top_k: f'top_{top_k}_{DISTANCE_METRIC_COL}'

        tokenizer: Any = model_name.tokenizer()
        assert ex[label_col] == label_text
        added_to_icl_dataset: bool = False
        # print(f'idx={idx}', flush=True)
        ex_labelwise_icl_dataset: List[Dict] = []
        ex_labelwise_synthesizrr_dataset: List[Dict] = []
        for top_k in top_ks:
            d: Dict = dict()
            d[RETRIEVED_TOP_K_COL] = top_k
            d[QUERY_TEXT_COL] = shorten(
                ex[query_col],
                max_tokens=synthesizrr_max_tokens,
                tokenizer=tokenizer,
            )
            d[EXAMPLE_TEXT_COL] = d[QUERY_TEXT_COL]
            retrieved_context: Optional[str] = shorten(
                ex[context_top_k_col(top_k=top_k)],
                min_tokens=retr_icl_min_tokens,
                max_tokens=retr_icl_max_tokens,
                tokenizer=tokenizer,
            )
            if retrieved_context is None:
                continue  ## Ignore this retrieved result
            d[RETRIEVED_CONTEXT_COL] = retrieved_context
            d[LABEL_TEXT_COL] = ex[label_col]
            d[LABEL_VERBALIZATION_COL] = ex[LABEL_VERBALIZATION_COL]
            d[IDX_COL] = f"ex={idx}" \
                         f"-top_k={top_k}" \
                         f"-corpus={corpus.canonical()}" \
                         f"-retriever={retriever.canonical()}" \
                         f"-model_name={model_name.canonical()}" \
                         f"-label={d[LABEL_TEXT_COL]}" \
                         f"-retr_icl_top_ks={retr_icl_top_ks}" \
                         f"-retr_icl_distance_range={retr_icl_distance_range}" \
                         f"-retr_icl_token_range={retr_icl_token_range}" \
                         f"-synthesizrr_top_k_range=range({synthesizrr_top_k_range.start}, {synthesizrr_top_k_range.stop}, {synthesizrr_top_k_range.step})" \
                         f"-synthesizrr_distance_range={synthesizrr_distance_range}" \
                         f"-synthesizrr_max_tokens={synthesizrr_max_tokens}"

            assert ex.get(distance_metric_top_k_col(top_k=top_k)) is not None
            assert ex.get(distance_top_k_col(top_k=top_k)) is not None
            if icl_type == 'retrieved':
                ## Here, we check the top_k=1,2 retrieved examples and if it is close, we add them to the ICL data.
                ## Else we check if it is close enough to add to the synthesizrr dataset.
                ## If it still isn't, we discard.
                if top_k in retr_icl_top_ks and \
                        retr_icl_distance_range[0] <= ex[distance_top_k_col(top_k=top_k)] <= \
                        retr_icl_distance_range[1]:
                    ## Add to ICL dataset
                    ex_labelwise_icl_dataset.append(d)
                elif top_k not in retr_icl_top_ks and \
                        synthesizrr_distance_range[0] <= ex[distance_top_k_col(top_k=top_k)] <= \
                        synthesizrr_distance_range[1]:
                    ## Filter by the distance range as follows:
                    ## - above 0.9: Probably an exact match.
                    ## - below 0.6: Probably about the exact same topic.
                    ## - below 0.4: Probably something completely different.
                    ex_labelwise_synthesizrr_dataset.append(d)
                else:
                    pass  ## Ignore this retrieved result
            elif icl_type == 'seed':
                ## Here, we add it to the ICL data exactly once.
                ## We also check if it is close enough to add to the synthesizrr dataset.
                ## If it still isn't, we discard.
                if added_to_icl_dataset is False:
                    ## Add to ICL dataset
                    ex_labelwise_icl_dataset.append(d)
                    added_to_icl_dataset: bool = True
                if retriever is Retriever.Random or \
                        synthesizrr_distance_range[0] <= ex[distance_top_k_col(top_k=top_k)] <= \
                        synthesizrr_distance_range[1]:
                    ## For random retrieved, add all results.
                    ## Otherwise, filter by the distance range as follows:
                    ## - above 0.9: Probably an exact match.
                    ## - below 0.6: Probably about the exact same topic.
                    ## - below 0.4: Probably something completely different.
                    ex_labelwise_synthesizrr_dataset.append(d)
                else:
                    pass  ## Ignore this retrieved result
            else:
                raise not_impl('icl_type', icl_type)

        # print('')
        # print(f'='*80)
        # print(f"Original Article:\n{ex[f'text']}")
        # print(f"\nRetrieved Article [Top-K={top_k}]:\n{shorten(ex[f'top_{top_k}_article_text'], max_tokens=1800)}")
        # if top_k > 20:  ## Keep the top-100 retrieved articles only.
        #     continue
        return pd.DataFrame(ex_labelwise_icl_dataset), pd.DataFrame(ex_labelwise_synthesizrr_dataset)


class CreateFewGenDatasets(CreateInputDatasets):
    @safe_validate_arguments
    def run(
            self,
            *,
            results_dir: FileMetadata,
            dataset_name: DatasetName,
            model_name: ModelName,
            label_verbalizer: Dict[str, str],
            seed_set: ClassificationData,
            seed_type: Literal['generated', 'train_set'],
            seed_size: int,
            seed_set_stratify_on_ground_truth: bool,
            fewgen_max_tokens: conint(ge=1),
            seed_set_data_split: DataSplit = DEFAULT_SEED_SET_DATA_SPLIT,
            num_samples_per_label: Union[Dict[str, conint(ge=1)], conint(ge=1)],
            seed: int = DEFAULT_SEED,
            text_col: Optional[str] = None,
            label_col: Optional[str] = None,
            **kwargs,
    ) -> Dict:
        text_col: str = get_default(text_col, dataset_name.text_col())
        label_col: str = get_default(label_col, dataset_name.label_col())
        num_samples_per_label: Dict[str, int] = expand_num_samples_per_label(
            num_samples_per_label=num_samples_per_label,
            label_verbalizer=label_verbalizer,
        )

        self.info(f'\nCreating ICL dataset and FewGen dataset for "{dataset_name.canonical()}"...')
        executor = dispatch_executor(
            parallelize=Parallelize.sync,
            max_workers=len(label_verbalizer)
        )
        try:
            icl_dataset_files: Dict[str, FileMetadata] = {}
            fewgen_dataset_files: Dict[str, FileMetadata] = {}
            futs: Dict[str, Future] = {}
            for label_text, label_verbalization in label_verbalizer.items():
                self.info(f'\nRunning for label="{label_text}"')
                labelwise_icl_dataset_file, labelwise_fewgen_dataset_file = self.save_to(
                    results_dir=results_dir,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    seed_type=seed_type,
                    seed_size=seed_size,
                    seed_set_stratify_on_ground_truth=seed_set_stratify_on_ground_truth,
                    label_text=label_text,
                    num_samples_per_label=num_samples_per_label[label_text],
                    seed_set_data_split=seed_set_data_split,
                    fewgen_max_tokens=fewgen_max_tokens,
                    seed=seed,
                )
                icl_dataset_files[label_text]: FileMetadata = labelwise_icl_dataset_file
                fewgen_dataset_files[label_text]: FileMetadata = labelwise_fewgen_dataset_file
                if (not labelwise_icl_dataset_file.exists()) or (not labelwise_fewgen_dataset_file.exists()):
                    futs[label_text] = dispatch(
                        self._save_labelwise_seed_icl_and_text_inputs,
                        seed_set=seed_set,
                        seed_set_data_split=seed_set_data_split,
                        label_verbalizer=label_verbalizer,
                        label_col=label_col,
                        text_col=text_col,
                        model_name=model_name,
                        label_text=label_text,
                        num_samples_per_label=num_samples_per_label[label_text],
                        max_tokens=fewgen_max_tokens,
                        seed=seed,
                        labelwise_icl_dataset_file=labelwise_icl_dataset_file,
                        labelwise_fewgen_dataset_file=labelwise_fewgen_dataset_file,
                        verbosity=self.verbosity,
                        executor=executor,
                        parallelize=Parallelize.sync,
                    )
            accumulate(
                futs,
                progress=dict(
                    desc=f'FewGen inputs: dataset_name={dataset_name.canonical()}, model_name={model_name.canonical()}'
                ) if self.verbosity >= 2 else False
            )
            icl_dataset: List[ClassificationData] = []
            fewgen_dataset: List[TextInputs] = []
            for label_text, label_verbalization in label_verbalizer.items():
                icl_dataset.append(
                    load_dataset(icl_dataset_files[label_text], retry=10, retry_wait=10)
                )
                self.info(
                    f'>> Loaded {len(icl_dataset[-1])} ICL examples '
                    f'from "{icl_dataset_files[label_text].path}"'
                )
                fewgen_dataset.append(
                    load_dataset(fewgen_dataset_files[label_text], retry=10, retry_wait=10)
                )
                self.info(
                    f'>> Loaded {len(fewgen_dataset[-1])} FewGen examples '
                    f'from "{fewgen_dataset_files[label_text].path}"'
                )
            self.info(f'...done creating ICL dataset and FewGen dataset for "{dataset_name.canonical()}".')
            icl_dataset: ClassificationData = Dataset.concat(icl_dataset).to_layout(DataLayout.PANDAS)
            icl_dataset.data = icl_dataset.data.reset_index(drop=True)
            fewgen_dataset: TextInputs = Dataset.concat(fewgen_dataset).to_layout(DataLayout.PANDAS)
            fewgen_dataset.data = fewgen_dataset.data.reset_index(drop=True)
            return dict(
                icl_dataset=icl_dataset,
                text_input_dataset=fewgen_dataset,
            )
        finally:
            stop_executor(executor)

    def save_to(
            self,
            *,
            results_dir: FileMetadata,  ## E.g. RESULTS_DIR/,
            dataset_name: DatasetName,  ## E.g. ag_news
            model_name: ModelName,
            seed_set_data_split: DataSplit,
            seed_type: Literal['generated', 'train_set'],
            seed_size: int,
            seed_set_stratify_on_ground_truth: bool,
            label_text: str,
            num_samples_per_label: conint(ge=1),
            seed: int,
            fewgen_max_tokens: conint(ge=1),
            **kwargs,
    ) -> Tuple[FileMetadata, FileMetadata]:
        ## RESULTS_DIR/retrieval-augmented-dataset-generation/ag_news/retrieval-data/toi_without_datasets/ag_news_seed_retr_output_toi_without_datasets.jsonlines
        if seed_type == 'generated':
            seed_type_str: str = f'generated_seed_set'
        elif seed_type == 'train_set':
            seed_type_str: str = f'{seed_set_data_split.lower()}_set_seed_set'
        else:
            raise not_impl('seed_type', seed_type)

        seed_size_str = ''
        if seed_size != dataset_name.seed_size():
            seed_size_str: str = f"-seed_size={seed_size}"
        stratified_str = ''
        if seed_set_stratify_on_ground_truth != DEFAULT_SEED_SET_STRATIFY_ON_GROUND_TRUTH:
            stratified_str: str = 'gt_stratified' if seed_set_stratify_on_ground_truth else 'not_gt_stratified'
            stratified_str: str = f"-stratified={stratified_str}"

        num_samples_per_label_str: str = get_default(num_samples_per_label, 'all')

        def get_save_to_file(key: str) -> FileMetadata:
            return results_dir.subdir_in_dir('few-shot-generation', return_metadata=True) \
                .subdir_in_dir(dataset_name.canonical(), return_metadata=True) \
                .subdir_in_dir(key, return_metadata=True) \
                .subdir_in_dir(model_name.canonical(), return_metadata=True) \
                .subdir_in_dir(f'num_samples_per_label={num_samples_per_label_str}', return_metadata=True) \
                .subdir_in_dir(f'label={label_text}', return_metadata=True) \
                .file_in_dir(
                f"{seed_type_str}{seed_size_str}{stratified_str}"
                f"-dataset={dataset_name.canonical()}"
                f"-model_name={model_name.canonical()}"
                f"-num_samples_per_label={num_samples_per_label_str}"
                f"-seed={seed}"
                f"-label={label_text}"
                f"-fewgen_max_tokens={fewgen_max_tokens}"
                f".parquet",
                return_metadata=True,
            ).update_params(file_format=FileFormat.PARQUET)

        icl_dataset_file: FileMetadata = get_save_to_file('icl-examples')
        fewgen_dataset_file: FileMetadata = get_save_to_file('fewgen-inputs')
        return icl_dataset_file, fewgen_dataset_file


class CreateSynthesizRRDatasets(CreateInputDatasets):
    @safe_validate_arguments
    def run(
            self,
            *,
            results_dir: FileMetadata,
            dataset_name: DatasetName,
            corpus: Corpus,
            retriever: Retriever,
            model_name: ModelName,
            label_verbalizer: Dict[str, str],
            seed_set: ClassificationData,
            seed_type: Literal['generated', 'train_set'],
            seed_size: int,
            seed_set_stratify_on_ground_truth: bool,
            seed_set_retr_input: Queries,
            seed_set_retr_output: RankedResults,
            icl_type: Literal['retrieved', 'curated', 'seed'],
            retrieval_top_k: conint(ge=1),
            retr_icl_top_ks: List[conint(ge=1)],
            retr_icl_distance_range: Tuple[float, float],
            retr_icl_token_range: Tuple[conint(ge=1), conint(ge=1)],
            synthesizrr_top_k_range: range,
            synthesizrr_distance_range: Tuple[float, float],  ## (0.4, 0.9)
            synthesizrr_max_tokens: conint(ge=1),
            seed_set_data_split: DataSplit = DEFAULT_SEED_SET_DATA_SPLIT,
            num_samples_per_label: Optional[Union[Dict[str, conint(ge=1)], conint(ge=1)]] = None,
            seed: int = DEFAULT_SEED,
            text_col: Optional[str] = None,
            query_col: Optional[str] = None,
            context_col: Optional[str] = None,
            label_col: Optional[str] = None,
            **kwargs,
    ) -> Dict:
        if icl_type == 'curated':
            raise not_impl('icl_type', icl_type)
        text_col: str = get_default(text_col, dataset_name.text_col())
        query_col: str = get_default(query_col, dataset_name.query_col())
        context_col: str = get_default(context_col, corpus.context_col())
        label_col: str = get_default(label_col, dataset_name.label_col())
        num_samples_per_label: Optional[Dict[str, int]] = expand_num_samples_per_label(
            num_samples_per_label=num_samples_per_label,
            label_verbalizer=label_verbalizer,
        )

        seed_set_retr_input: Queries = seed_set_retr_input.to_layout(DataLayout.PANDAS)
        seed_set_retr_output: RankedResults = seed_set_retr_output.to_layout(DataLayout.PANDAS)
        seed_set_retr_output.data[LABEL_VERBALIZATION_COL] = seed_set_retr_output.data[label_col].map(label_verbalizer)

        self.info(f'\nCreating ICL dataset and SynthesizRR dataset for "{dataset_name.canonical()}"...')
        executor = dispatch_executor(
            parallelize=Parallelize.sync,
            max_workers=len(label_verbalizer)
        )
        try:
            icl_dataset_files: Dict[str, FileMetadata] = {}
            synthesizrr_dataset_files: Dict[str, FileMetadata] = {}
            futs: Dict[str, Any] = {}
            for label_text, label_verbalization in label_verbalizer.items():
                self.info(f'\nRunning for label="{label_text}"')
                labelwise_icl_dataset_file, labelwise_synthesizrr_dataset_file = self.save_to(
                    results_dir=results_dir,
                    dataset_name=dataset_name,
                    corpus=corpus,
                    retriever=retriever,
                    model_name=model_name,
                    seed_type=seed_type,
                    seed_size=seed_size,
                    seed_set_stratify_on_ground_truth=seed_set_stratify_on_ground_truth,
                    icl_type=icl_type,
                    label_text=label_text,
                    retrieval_top_k=retrieval_top_k,
                    retr_icl_top_ks=retr_icl_top_ks,
                    retr_icl_distance_range=retr_icl_distance_range,
                    retr_icl_token_range=retr_icl_token_range,
                    synthesizrr_top_k_range=synthesizrr_top_k_range,
                    synthesizrr_distance_range=synthesizrr_distance_range,
                    synthesizrr_max_tokens=synthesizrr_max_tokens,
                    num_samples_per_label=get_default(num_samples_per_label, {}).get(label_text),
                    seed=seed,
                    seed_set_data_split=seed_set_data_split,
                )
                icl_dataset_files[label_text]: FileMetadata = labelwise_icl_dataset_file
                synthesizrr_dataset_files[label_text]: FileMetadata = labelwise_synthesizrr_dataset_file
                if (not labelwise_icl_dataset_file.exists()) or (not labelwise_synthesizrr_dataset_file.exists()):
                    self.info(f'\nCreating input datasets for label="{label_text}"')
                    futs[label_text] = self._save_labelwise_retrieved_icl_and_text_inputs(
                        seed_set_retr_output=seed_set_retr_output,
                        seed_set_data_split=seed_set_data_split,
                        label_verbalizer=label_verbalizer,
                        corpus=corpus,
                        retriever=retriever,
                        query_col=query_col,
                        context_col=context_col,
                        label_col=label_col,
                        model_name=model_name,
                        label_text=label_text,
                        icl_type=icl_type,
                        retr_icl_top_ks=retr_icl_top_ks,
                        retr_icl_distance_range=retr_icl_distance_range,
                        retr_icl_token_range=retr_icl_token_range,
                        synthesizrr_top_k_range=synthesizrr_top_k_range,
                        synthesizrr_distance_range=synthesizrr_distance_range,
                        synthesizrr_max_tokens=synthesizrr_max_tokens,
                        num_samples_per_label=get_default(num_samples_per_label, {}).get(label_text),
                        seed=seed,
                        labelwise_icl_dataset_file=labelwise_icl_dataset_file,
                        labelwise_synthesizrr_dataset_file=labelwise_synthesizrr_dataset_file,
                        verbosity=self.verbosity,
                        # parallelize=Parallelize.sync,
                        # executor=executor,
                    )
            accumulate(
                futs,
                progress=dict(
                    desc=f'SynthesizRR inputs: dataset_name={dataset_name.canonical()}, model_name={model_name.canonical()}'
                ) if self.verbosity >= 2 else False
            )
            icl_dataset: List[ClassificationData] = []
            synthesizrr_dataset: List[TextInputs] = []
            for label_text, label_verbalization in label_verbalizer.items():
                icl_dataset.append(
                    load_dataset(icl_dataset_files[label_text], retry=10, retry_wait=10)
                )
                self.info(
                    f'>> Loaded {len(icl_dataset[-1])} ICL examples '
                    f'from "{icl_dataset_files[label_text].path}"'
                )
                synthesizrr_dataset.append(
                    load_dataset(synthesizrr_dataset_files[label_text], retry=10, retry_wait=10)
                )
                self.info(
                    f'>> Loaded {len(synthesizrr_dataset[-1])} SynthesizRR examples '
                    f'from "{synthesizrr_dataset_files[label_text].path}"'
                )
            self.info(f'...done creating ICL dataset and SynthesizRR dataset for "{dataset_name.canonical()}".')
            icl_dataset: ClassificationData = Dataset.concat(icl_dataset).to_layout(DataLayout.PANDAS)
            icl_dataset.data = icl_dataset.data.reset_index(drop=True)
            synthesizrr_dataset: TextInputs = Dataset.concat(synthesizrr_dataset).to_layout(DataLayout.PANDAS)
            synthesizrr_dataset.data = synthesizrr_dataset.data.reset_index(drop=True)
            return dict(
                icl_dataset=icl_dataset,
                text_input_dataset=synthesizrr_dataset,
            )
        finally:
            stop_executor(executor)

    def save_to(
            self,
            *,
            results_dir: FileMetadata,  ## E.g. RESULTS_DIR/,
            dataset_name: DatasetName,  ## E.g. ag_news
            corpus: Corpus,
            retriever: Retriever,
            retrieval_top_k: int,
            model_name: ModelName,
            seed_set_data_split: DataSplit,
            seed_type: Literal['generated', 'train_set'],
            seed_size: int,
            seed_set_stratify_on_ground_truth: bool,
            icl_type: Literal['retrieved', 'curated', 'seed'],
            label_text: str,
            retr_icl_top_ks: List[conint(ge=1)],
            retr_icl_distance_range: Tuple[float, float],
            retr_icl_token_range: Tuple[conint(ge=1), conint(ge=1)],
            synthesizrr_top_k_range: range,
            synthesizrr_distance_range: Tuple[float, float],  ## (0.4, 0.9)
            synthesizrr_max_tokens: conint(ge=1),
            num_samples_per_label: Optional[conint(ge=1)],
            seed: int,
            **kwargs,
    ) -> Tuple[FileMetadata, FileMetadata]:
        ## RESULTS_DIR/retrieval-augmented-dataset-generation/ag_news/retrieval-data/toi_without_datasets/ag_news_seed_retr_output_toi_without_datasets.jsonlines
        if seed_type == 'generated':
            seed_type_str: str = f'generated_seed_set'
        elif seed_type == 'train_set':
            seed_type_str: str = f'{seed_set_data_split.lower()}_set_seed_set'
        else:
            raise not_impl('seed_type', seed_type)

        seed_size_str = ''
        if seed_size != dataset_name.seed_size():
            seed_size_str: str = f"-seed_size={seed_size}"
        stratified_str = ''
        if seed_set_stratify_on_ground_truth != DEFAULT_SEED_SET_STRATIFY_ON_GROUND_TRUTH:
            stratified_str: str = 'gt_stratified' if seed_set_stratify_on_ground_truth else 'not_gt_stratified'
            stratified_str: str = f"-stratified={stratified_str}"

        if icl_type == 'retrieved':
            icl_type_str: str = f'retrieved_icl_dataset'
        elif icl_type == 'curated':
            icl_type_str: str = f'curated_icl_dataset'
        elif icl_type == 'seed':
            icl_type_str: str = f'{seed_type}_seed_icl_dataset'
        else:
            raise not_impl('icl_type', icl_type)

        num_samples_per_label_str: str = get_default(num_samples_per_label, 'all')

        def get_save_to_file(key: str) -> FileMetadata:
            return results_dir.subdir_in_dir('retrieval-augmented-dataset-generation', return_metadata=True) \
                .subdir_in_dir(dataset_name.canonical(), return_metadata=True) \
                .subdir_in_dir(key, return_metadata=True) \
                .subdir_in_dir(corpus.canonical(), return_metadata=True) \
                .subdir_in_dir(retriever.canonical(), return_metadata=True) \
                .subdir_in_dir(f'retrieval_top_k={retrieval_top_k}', return_metadata=True) \
                .subdir_in_dir(model_name.canonical(), return_metadata=True) \
                .subdir_in_dir(f'num_samples_per_label={num_samples_per_label_str}', return_metadata=True) \
                .subdir_in_dir(f'label={label_text}', return_metadata=True) \
                .file_in_dir(
                f"{icl_type_str}"
                f"-{seed_type_str}{seed_size_str}{stratified_str}-retr_output"
                f"-dataset={dataset_name.canonical()}"
                f"-corpus={corpus.canonical()}"
                f"-retriever={retriever.canonical()}"
                f"-model_name={model_name.canonical()}"
                f"-num_samples_per_label={num_samples_per_label_str}"
                f"-seed={seed}"
                f"-label={label_text}"
                f"-retr_icl_top_ks={retr_icl_top_ks}"
                f"-retr_icl_distance_range={retr_icl_distance_range}"
                f"-retr_icl_token_range={retr_icl_token_range}"
                f"-synthesizrr_top_k_range=range({synthesizrr_top_k_range.start}, {synthesizrr_top_k_range.stop}, {synthesizrr_top_k_range.step})"
                f"-synthesizrr_distance_range={synthesizrr_distance_range}"
                f"-synthesizrr_max_tokens={synthesizrr_max_tokens}"
                f".parquet",
                return_metadata=True,
            ).update_params(file_format=FileFormat.PARQUET)

        icl_dataset_file: FileMetadata = get_save_to_file('icl-examples')
        synthesizrr_dataset_file: FileMetadata = get_save_to_file('synthesizrr-inputs')
        return icl_dataset_file, synthesizrr_dataset_file


class LLMEvaluatorStep(CachedResultsStep, ABC):
    @safe_validate_arguments
    def _run_llm_evaluators(
            self,
            *,
            dataset_name: DatasetName,
            model_name: ModelName,
            text_gens_files: Dict[int, Dict[str, FileMetadata]],
            label_verbalizer: Dict[str, str],
            max_new_tokens: conint(ge=1),
            llm_batch_size: conint(ge=1),
            llm_submission_batch_size: Optional[conint(ge=1)],
            llm_tracking_batch_size: Optional[conint(ge=1)],
            icl_dataset: ClassificationData,
            text_input_dataset: TextInputs,
            icl_template: constr(min_length=1),
            prompt_template: constr(min_length=1),
            top_p: confloat(ge=0.0, le=1.0),
            temperature: confloat(ge=0.0, le=100.0),
            llm_resources_per_model: Dict[str, conint(ge=1)],
            llm_num_concurrent_preds: conint(ge=1),
            llm_num_models: Optional[conint(ge=1)],
            label_col: str,
            llm_load_balancing_strategy: LoadBalancingStrategy,
    ) -> Dict[int, Dict[str, TextGenerationsPredictionsBase]]:
        executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=llm_num_concurrent_preds)
        llm_generations: Dict[int, Dict[str, TextGenerationsPredictionsBase]] = {}
        num_shots_list: List[int] = sorted(list(text_gens_files.keys()))
        try:
            self.info(f'Creating LLM evaluators for "{model_name.canonical()}"...')
            llm_evaluators: Dict[int, Dict[str, FewShotTextGenerator]] = self._create_llm_evaluators(
                model_name=model_name,
                num_shots_list=num_shots_list,
                label_verbalizer=label_verbalizer,
                max_new_tokens=max_new_tokens,
                llm_batch_size=llm_batch_size,
                llm_tracking_batch_size=llm_tracking_batch_size,
                llm_submission_batch_size=llm_submission_batch_size,
                icl_dataset=icl_dataset,
                text_input_dataset=text_input_dataset,
                icl_template=icl_template,
                prompt_template=prompt_template,
                top_p=top_p,
                temperature=temperature,
                llm_resources_per_model=llm_resources_per_model,
                llm_num_models=llm_num_models,
                label_col=label_col,
            )
            self.info(f'...done creating LLM evaluators for "{model_name.canonical()}"...')
            self.info(f'Running LLM evaluators for "{model_name.canonical()}"...')
            for num_shots in text_gens_files.keys():
                llm_generations.setdefault(num_shots, {})
                for label_text, text_gens_file in text_gens_files[num_shots].items():
                    labelwise_text_input_dataset: TextInputs = text_input_dataset.filter(
                        lambda row: row[label_col] == label_text
                    )
                    labelwise_text_input_dataset: TextInputs = labelwise_text_input_dataset.to_layout(DataLayout.PANDAS)
                    labelwise_text_input_dataset.data = labelwise_text_input_dataset.data.reset_index(drop=True)
                    llm_generations[num_shots][label_text] = run_concurrent(
                        self._run_single_llm_evaluator,
                        llm_evaluator=llm_evaluators[num_shots][label_text],
                        labelwise_text_input_dataset=labelwise_text_input_dataset,
                        progress_bar=dict(
                            desc=f'{model_name.canonical()}, num_shots={num_shots}, label={label_text}',
                        ) if self.verbosity >= 2 else False,
                        submission_batch_size=get_default(llm_submission_batch_size, llm_batch_size * 8),
                        text_gens_file=text_gens_file,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        num_shots=num_shots,
                        label_text=label_text,
                        llm_load_balancing_strategy=llm_load_balancing_strategy,
                        executor=executor,
                    )
                    time.sleep(10)
            self.info(f'...submitted all inputs to LLM evaluators...')
            for num_shots in text_gens_files.keys():
                ## Collect for all labels in a single num_shots:
                llm_generations[num_shots]: Dict[str, TextGenerationsPredictionsBase] = accumulate(
                    llm_generations[num_shots],
                    progress_bar=dict(
                        desc=f'Collect predictions for {len(text_gens_files[num_shots])} labels, num_shots={num_shots}'
                    ) if self.verbosity >= 2 else False,
                )
            self.info(f'...done running LLM evaluators for "{model_name.canonical()}".')
        finally:
            stop_executor(executor)
            if 'llm_evaluators' in locals():
                self._stop_llm_evaluators(llm_evaluators)
            gc.collect()
        return llm_generations

    def _run_single_llm_evaluator(
            self,
            llm_evaluator: FewShotTextGenerator,
            labelwise_text_input_dataset: TextInputs,
            progress_bar: Union[Dict, bool],
            submission_batch_size: int,
            text_gens_file: FileMetadata,
            dataset_name: DatasetName,
            model_name: ModelName,
            num_shots: int,
            label_text: str,
            llm_load_balancing_strategy: LoadBalancingStrategy,
    ) -> TextGenerationsPredictionsBase:
        self.info(
            f'>> Generating responses for {len(labelwise_text_input_dataset)} inputs for ('
            f'dataset_name={dataset_name.canonical()}, '
            f'model_name={model_name.canonical()}, '
            f'num_shots={num_shots}, '
            f'label_text={label_text}'
            f') to "{text_gens_file.path}"...'
        )
        text_gens: Predictions = llm_evaluator.predict(
            labelwise_text_input_dataset,
            progress_bar=progress_bar,
            tracker=False,
            submission_batch_size=submission_batch_size,
            load_balancing_strategy=llm_load_balancing_strategy,
        )
        assert isinstance(text_gens, TextGenerationsPredictionsBase)
        self.info(
            f'>> Saving generations ('
            f'dataset_name={dataset_name.canonical()}, '
            f'model_name={model_name.canonical()}, '
            f'num_shots={num_shots}, '
            f'label_text={label_text}'
            f') to "{text_gens_file.path}"...'
        )
        save_predictions(
            predictions=text_gens,
            predictions_destination=text_gens_file,
            overwrite=True,
        )
        return text_gens

    def _create_llm_evaluators(
            self,
            *,
            model_name: ModelName,
            num_shots_list: List[conint(ge=0)],
            label_verbalizer: Dict[str, str],
            max_new_tokens: conint(ge=1),
            llm_batch_size: conint(ge=1),
            llm_tracking_batch_size: Optional[conint(ge=1)],
            llm_submission_batch_size: Optional[conint(ge=1)],
            icl_dataset: ClassificationData,
            text_input_dataset: TextInputs,
            icl_template: constr(min_length=1),
            prompt_template: constr(min_length=1),
            top_p: confloat(ge=0.0, le=1.0),
            temperature: confloat(ge=0.0, le=100.0),
            llm_resources_per_model: Dict[str, conint(ge=1)],
            llm_num_models: Optional[conint(ge=1)],
            label_col: str,
    ) -> Dict[int, Dict[str, FewShotTextGenerator]]:
        ## Create main driver Evaluator.
        if model_name in {ModelName.LLaMa_2_13B, ModelName.LLaMa_2_13B_Chat}:
            llm_evaluator: Evaluator = self._create_hf_evaluator(
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                llm_batch_size=llm_batch_size,
                llm_num_models=llm_num_models,
                llm_resources_per_model=llm_resources_per_model,
                top_p=top_p,
                temperature=temperature,
            )
        elif model_name in {ModelName.ChatGPT}:
            llm_evaluator: Evaluator = self._create_chatgpt_evaluator(
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
            )
        elif model_name.is_claude():
            llm_evaluator: Evaluator = self._create_claude_evaluator(
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
            )
        else:
            raise not_impl('model_name', model_name)

        llm_evaluators: Dict[int, Dict[str, FewShotTextGenerator]] = {}
        for num_shots in num_shots_list:
            llm_evaluators.setdefault(num_shots, {})
            for label_text, label_verbalization in label_verbalizer.items():
                labelwise_icl_dataset: ClassificationData = icl_dataset.filter(
                    lambda row: row[label_col] == label_text
                ).to_layout(DataLayout.PANDAS)
                labelwise_icl_dataset.data = labelwise_icl_dataset.data.reset_index(drop=True)
                labelwise_text_input_dataset: TextInputs = text_input_dataset.filter(
                    lambda row: row[label_col] == label_text
                ).to_layout(DataLayout.PANDAS)
                llm_tracking_batch_size: int = get_default(
                    llm_tracking_batch_size,
                    math.ceil(len(labelwise_text_input_dataset) / 10)  ## Default: 10 increments
                )
                llm_tracking_batch_size: int = max(llm_tracking_batch_size, llm_submission_batch_size)
                llm_evaluators[num_shots][label_text]: FewShotTextGenerator = FewShotTextGenerator.of(
                    lm=llm_evaluator,
                    icl_dataset=labelwise_icl_dataset,
                    hyperparams=dict(
                        num_shots=num_shots,
                        batch_size=llm_tracking_batch_size,
                        icl_template=icl_template.format(**dict(
                            label_verbalization=label_verbalization,
                        )),
                        prompt_template=prompt_template.format(**dict(
                            label_verbalization=label_verbalization,
                        )),
                    ),
                )
        return llm_evaluators

    def _create_chatgpt_evaluator(
            self,
            *,
            model_name: ModelName,
            max_new_tokens: int,
            top_p: float,
            temperature: float,
    ) -> Evaluator:
        chatgpt: Evaluator = Evaluator.of(
            'local',
            algorithm=model_name.algorithm_name(),
            task=Task.NEXT_TOKEN_PREDICTION,
            hyperparams=dict(
                batch_size=1,
                llm=dict(
                    name='ChatOpenAI',
                    model_name=model_name.model_name(),
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                ),
                api_key=model_name.api_key(),
                retries=3,
            ),
            verbosity=1 if self.verbosity >= 3 else 0,
        )
        return chatgpt

    def _create_claude_evaluator(
            self,
            *,
            model_name: ModelName,
            max_new_tokens: int,
            top_p: float,
            temperature: float,
    ) -> Evaluator:
        claude: Evaluator = Evaluator.of(
            'local',
            algorithm=model_name.algorithm_name(),
            task=Task.NEXT_TOKEN_PREDICTION,
            hyperparams=dict(
                model_name=model_name.model_name(),
                retries=3,
                generation_params=dict(
                    name='top-p',
                    top_p=top_p,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
            ),
            verbosity=1 if self.verbosity >= 3 else 0,
        )
        return claude

    def _create_hf_evaluator(
            self,
            *,
            model_name: ModelName,
            max_new_tokens: int,
            llm_batch_size: int,
            llm_num_models: Optional[int],
            llm_resources_per_model: Dict[str, conint(ge=1)],
            top_p: float,
            temperature: float,
    ) -> Evaluator:
        hf_llm_evaluator = Evaluator.of(
            evaluator='ray',
            algorithm=model_name.algorithm_name(),
            task=Task.NEXT_TOKEN_PREDICTION,
            nested_evaluator_name='accelerate',

            cache_dir=EFS_HUGGINGFACE_CACHE_DIR,

            num_models=llm_num_models,  ## Create N copies of the model.
            resources_per_model=llm_resources_per_model,
            model_weights_dtype=model_name.model_weights_dtype(),

            hyperparams=dict(
                batch_size=llm_batch_size,
                model_name=model_name.model_name(),
                tokenizer_config=dict(
                    pad_token=model_name.pad_token(),
                    padding_side='left',
                    truncation_side='left',
                ),
                tokenizer_encode=dict(
                    max_length=model_name.max_input_length(),
                ),
                generation_params=dict(
                    name='top-p',
                    top_p=top_p,
                    temperature=temperature,
                    output_scores=False,
                    output_scores_tolerance=1e-2,
                    max_new_tokens=max_new_tokens,
                ),
            ),
            ## To work with the model interactively, set this to a number (seconds)
            ## denoting how long the models stay in memory (recommended is 15 mins).
            cache_timeout=12 * 60 * 60,
            verbosity=1 if self.verbosity >= 3 else 0,
            init_model=True,
        )
        time.sleep(30)
        return hf_llm_evaluator

    def _stop_llm_evaluators(self, llm_evaluators: Dict[int, Dict[str, FewShotTextGenerator]]):
        for num_shots in llm_evaluators.keys():
            for label_text in llm_evaluators[num_shots].keys():
                llm_evaluator: FewShotTextGenerator = llm_evaluators[num_shots][label_text]
                if isinstance(llm_evaluator.lm, Evaluator):
                    llm_evaluator.lm.stop()

    def _load_text_gens(
            self,
            *,
            text_gens_files: Dict[int, Dict[str, FileMetadata]],
            dataset_name: DatasetName,
            model_name: ModelName,
            label_col: str,
    ) -> Dict[int, TextGenerationsPredictionsBase]:
        text_gens: Dict[int, TextGenerationsPredictionsBase] = {}
        for num_shots in text_gens_files.keys():
            num_shots_text_gens_list: List[TextGenerationsPredictionsBase] = []
            self.info(f'\nLoading text generations for num_shots={num_shots}...')
            for label_text, text_gens_file in text_gens_files[num_shots].items():
                labelwise_num_shots_text_gens: TextGenerationsPredictionsBase = load_predictions(
                    text_gens_file, retry=10, retry_wait=10
                )
                num_shots_text_gens_list.append(labelwise_num_shots_text_gens)

                self.info(
                    f'>> Loaded {len(labelwise_num_shots_text_gens)} generations ('
                    f'dataset={dataset_name.canonical()}, '
                    f'model_name={model_name.canonical()}, '
                    f'num_shots={num_shots}, '
                    f'label_text={label_text}'
                    f') from "{text_gens_files[num_shots][label_text].path}".'
                )
            text_gens[num_shots]: TextGenerations = \
                Predictions.concat(num_shots_text_gens_list).to_layout(DataLayout.PANDAS)
            self.info(f'\n...loaded text generations for num_shots={num_shots}.')

            self.info(
                f'Label distribution of {len(text_gens[num_shots])} generations for ('
                f'dataset={dataset_name.canonical()}, '
                f'model_name={model_name.canonical()}, '
                f'num_shots={num_shots}'
                f'):'
            )
            self.info(calc_label_dist(text_gens[num_shots].data.pandas(), label_col=label_col))
        return text_gens


class FewGen(LLMEvaluatorStep):
    @safe_validate_arguments
    def run(
            self,
            *,
            results_dir: FileMetadata,
            dataset_name: DatasetName,
            icl_dataset: ClassificationData,
            text_input_dataset: TextInputs,
            model_name: ModelName,
            num_shots_list: List[conint(ge=0)],
            seed_type: Literal['generated', 'train_set'],
            seed_size: int,
            seed_set_stratify_on_ground_truth: bool,
            fewgen_max_tokens: conint(ge=1),
            num_samples_per_label: Union[Dict[str, conint(ge=1)], conint(ge=1)],
            label_verbalizer: Dict[str, str],
            icl_template: Optional[constr(min_length=1)] = None,
            prompt_template: Optional[constr(min_length=1)] = None,
            top_p: confloat(ge=0.0, le=1.0),
            temperature: confloat(ge=0.0, le=100.0),
            llm_resources_per_model: Dict[str, conint(ge=1)],
            llm_batch_size: conint(ge=1),
            llm_submission_batch_size: Optional[conint(ge=1)] = None,
            llm_tracking_batch_size: Optional[conint(ge=1)] = None,
            llm_num_concurrent_preds: conint(ge=1) = 6,
            llm_num_models: Optional[conint(ge=1)] = None,
            seed: int = DEFAULT_SEED,
            seed_set_data_split: DataSplit = DEFAULT_SEED_SET_DATA_SPLIT,
            text_col: Optional[str] = None,
            label_col: Optional[str] = None,
            llm_load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_USED,
            **kwargs,
    ) -> Dict:
        text_col: str = get_default(text_col, dataset_name.text_col())
        label_col: str = get_default(label_col, dataset_name.label_col())
        num_samples_per_label: Dict[str, int] = expand_num_samples_per_label(
            num_samples_per_label=num_samples_per_label,
            label_verbalizer=label_verbalizer,
        )
        icl_template, icl_template_hash, prompt_template, prompt_template_hash = get_templates_and_hashes(
            expt=Experiment.FewGen,
            dataset_name=dataset_name,
            model_name=model_name,
            icl_template=icl_template,
            prompt_template=prompt_template,
        )

        text_gens_files: Dict[int, Dict[str, FileMetadata]] = {}
        missing_text_gens_files: Dict[int, Dict[str, FileMetadata]] = {}
        for num_shots in num_shots_list:
            text_gens_files.setdefault(num_shots, {})
            for label_text, label_verbalization in label_verbalizer.items():
                text_gens_files[num_shots][label_text]: FileMetadata = self.save_to(
                    results_dir=results_dir,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    num_shots=num_shots,
                    label_text=label_text,
                    label_verbalization=label_verbalization,
                    seed_set_data_split=seed_set_data_split,
                    seed_type=seed_type,
                    seed_size=seed_size,
                    seed_set_stratify_on_ground_truth=seed_set_stratify_on_ground_truth,
                    num_samples_per_label=num_samples_per_label[label_text],
                    seed=seed,
                    fewgen_max_tokens=fewgen_max_tokens,
                    icl_template_hash=icl_template_hash,
                    prompt_template_hash=prompt_template_hash,
                )
                if not text_gens_files[num_shots][label_text].exists():
                    missing_text_gens_files.setdefault(num_shots, {})
                    missing_text_gens_files[num_shots][label_text] = text_gens_files[num_shots][label_text]
                else:
                    self.info(
                        f'>> Generations already exists for ('
                        f'dataset_name={dataset_name.canonical()}, '
                        f'model_name={model_name.canonical()}, '
                        f'num_shots={num_shots}, '
                        f'label_text={label_text}'
                        f') at "{text_gens_files[num_shots][label_text].path}"...'
                    )

        if len(missing_text_gens_files) > 0:
            llm_generations: Dict[int, Dict[str, TextGenerationsPredictionsBase]] = self._run_llm_evaluators(
                dataset_name=dataset_name,
                model_name=model_name,
                text_gens_files=missing_text_gens_files,
                label_verbalizer=label_verbalizer,
                max_new_tokens=fewgen_max_tokens,
                llm_batch_size=llm_batch_size,
                llm_submission_batch_size=llm_submission_batch_size,
                llm_tracking_batch_size=llm_tracking_batch_size,
                icl_dataset=icl_dataset,
                text_input_dataset=text_input_dataset,
                icl_template=icl_template,
                prompt_template=prompt_template,
                top_p=top_p,
                temperature=temperature,
                llm_resources_per_model=llm_resources_per_model,
                llm_num_concurrent_preds=llm_num_concurrent_preds,
                llm_num_models=llm_num_models,
                label_col=label_col,
                llm_load_balancing_strategy=llm_load_balancing_strategy,
            )
        text_gens: Dict[int, TextGenerationsPredictionsBase] = self._load_text_gens(
            text_gens_files=text_gens_files,
            dataset_name=dataset_name,
            model_name=model_name,
            label_col=label_col,
        )
        return dict(
            text_gens=text_gens,
        )

    def save_to(
            self,
            *,
            results_dir: FileMetadata,  ## E.g. RESULTS_DIR/,
            dataset_name: DatasetName,  ## E.g. ag_news
            model_name: ModelName,
            num_shots: int,
            label_text: str,
            label_verbalization: str,
            seed_set_data_split: DataSplit,
            seed_type: Literal['generated', 'train_set'],
            seed_size: int,
            seed_set_stratify_on_ground_truth: bool,
            fewgen_max_tokens: conint(ge=1),
            icl_template_hash: Optional[constr(min_length=6)],
            prompt_template_hash: Optional[constr(min_length=6)],
            num_samples_per_label: conint(ge=1),
            seed: int,
            **kwargs,
    ) -> FileMetadata:
        ## RESULTS_DIR/retrieval-augmented-dataset-generation/ag_news/retrieval-data/toi_without_datasets/ag_news_seed_retr_output_toi_without_datasets.jsonlines
        if seed_type == 'generated':
            seed_type_str: str = f'generated_seed_set'
        elif seed_type == 'train_set':
            seed_type_str: str = f'{seed_set_data_split.lower()}_set_seed_set'
        else:
            raise not_impl('seed_type', seed_type)

        seed_size_str = ''
        if seed_size != dataset_name.seed_size():
            seed_size_str: str = f"-seed_size={seed_size}"
        stratified_str = ''
        if seed_set_stratify_on_ground_truth != DEFAULT_SEED_SET_STRATIFY_ON_GROUND_TRUTH:
            stratified_str: str = 'gt_stratified' if seed_set_stratify_on_ground_truth else 'not_gt_stratified'
            stratified_str: str = f"-stratified={stratified_str}"

        num_samples_per_label_str: str = get_default(num_samples_per_label, 'all')

        icl_template_hash_str: str = '' if icl_template_hash is None \
            else f'-icl_template_hash={icl_template_hash}'
        prompt_template_hash_str: str = '' if prompt_template_hash is None \
            else f'-prompt_template_hash={prompt_template_hash}'

        label_verbalization_str: str = ''
        if label_verbalization != dataset_name.label_verbalizer()[label_text]:
            label_verbalization_str: str = f'-vb={StringUtil.hash(label_verbalization, max_len=4)}'

        return results_dir.subdir_in_dir('few-shot-generation', return_metadata=True) \
            .subdir_in_dir(dataset_name.canonical(), return_metadata=True) \
            .subdir_in_dir('fewgen-generations', return_metadata=True) \
            .subdir_in_dir(model_name.canonical(), return_metadata=True) \
            .subdir_in_dir(f'num_samples_per_label={num_samples_per_label_str}', return_metadata=True) \
            .subdir_in_dir(f'num_shots={num_shots}', return_metadata=True) \
            .subdir_in_dir(f'label_text={label_text}', return_metadata=True) \
            .file_in_dir(
            f"{seed_type_str}{seed_size_str}{stratified_str}"
            f"-dataset={dataset_name.canonical()}"
            f"-model_name={model_name.canonical()}"
            f"-num_samples_per_label={num_samples_per_label_str}"
            f"-num_shots={num_shots}"
            f"-label_text={label_text}"
            f"{label_verbalization_str}"
            f"-seed={seed}"
            f"-fewgen_max_tokens={fewgen_max_tokens}"
            f"{icl_template_hash_str}"
            f"{prompt_template_hash_str}"
            f".parquet",
            return_metadata=True,
        ).update_params(file_format=FileFormat.PARQUET)


class SynthesizRR(LLMEvaluatorStep):
    @safe_validate_arguments
    def run(
            self,
            *,
            results_dir: FileMetadata,
            dataset_name: DatasetName,
            icl_dataset: ClassificationData,
            text_input_dataset: TextInputs,
            corpus: Corpus,
            retriever: Retriever,
            model_name: ModelName,
            num_shots_list: List[conint(ge=0)],
            seed_type: Literal['generated', 'train_set'],
            seed_size: int,
            seed_set_stratify_on_ground_truth: bool,
            icl_type: Literal['retrieved', 'curated', 'seed'],
            retr_icl_top_ks: List[conint(ge=1)],
            retr_icl_distance_range: Tuple[float, float],
            retr_icl_token_range: Tuple[conint(ge=1), conint(ge=1)],
            synthesizrr_top_k_range: range,
            synthesizrr_distance_range: Tuple[float, float],  ## (0.4, 0.9)
            synthesizrr_max_tokens: conint(ge=1),
            num_samples_per_label: Optional[Union[Dict[str, conint(ge=1)], conint(ge=1)]],
            label_verbalizer: Dict[str, str],
            icl_template: Optional[constr(min_length=1)] = None,
            prompt_template: Optional[constr(min_length=1)] = None,
            top_p: confloat(ge=0.0, le=1.0),
            temperature: confloat(ge=0.0, le=100.0),
            llm_resources_per_model: Dict[str, conint(ge=1)],
            llm_batch_size: conint(ge=1),
            llm_submission_batch_size: Optional[conint(ge=1)] = None,
            llm_tracking_batch_size: Optional[conint(ge=1)] = None,
            llm_num_concurrent_preds: conint(ge=1) = 6,
            llm_num_models: Optional[conint(ge=1)] = None,
            seed: int = DEFAULT_SEED,
            seed_set_data_split: DataSplit = DEFAULT_SEED_SET_DATA_SPLIT,
            query_col: Optional[str] = None,
            context_col: Optional[str] = None,
            label_col: Optional[str] = None,
            llm_load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_USED,
            **kwargs,
    ) -> Dict:
        query_col: str = get_default(query_col, dataset_name.query_col())
        context_col: str = get_default(context_col, corpus.context_col())
        label_col: str = get_default(label_col, dataset_name.label_col())
        num_samples_per_label: Optional[Dict[str, int]] = expand_num_samples_per_label(
            num_samples_per_label=num_samples_per_label,
            label_verbalizer=label_verbalizer,
        )
        icl_template, icl_template_hash, prompt_template, prompt_template_hash = get_templates_and_hashes(
            expt=Experiment.SynthesizRR,
            dataset_name=dataset_name,
            model_name=model_name,
            icl_template=icl_template,
            prompt_template=prompt_template,
        )

        text_gens_files: Dict[int, Dict[str, FileMetadata]] = {}
        missing_text_gens_files: Dict[int, Dict[str, FileMetadata]] = {}
        for num_shots in num_shots_list:
            text_gens_files.setdefault(num_shots, {})
            for label_text, label_verbalization in label_verbalizer.items():
                text_gens_files[num_shots][label_text]: FileMetadata = self.save_to(
                    results_dir=results_dir,
                    dataset_name=dataset_name,
                    corpus=corpus,
                    retriever=retriever,
                    model_name=model_name,
                    num_shots=num_shots,
                    label_text=label_text,
                    label_verbalization=label_verbalization,
                    seed_set_data_split=seed_set_data_split,
                    seed_type=seed_type,
                    seed_size=seed_size,
                    seed_set_stratify_on_ground_truth=seed_set_stratify_on_ground_truth,
                    icl_type=icl_type,
                    retr_icl_top_ks=retr_icl_top_ks,
                    retr_icl_distance_range=retr_icl_distance_range,
                    retr_icl_token_range=retr_icl_token_range,
                    synthesizrr_top_k_range=synthesizrr_top_k_range,
                    synthesizrr_distance_range=synthesizrr_distance_range,
                    synthesizrr_max_tokens=synthesizrr_max_tokens,
                    icl_template_hash=icl_template_hash,
                    prompt_template_hash=prompt_template_hash,
                    num_samples_per_label=get_default(num_samples_per_label, {}).get(label_text),
                    seed=seed,
                )
                if not text_gens_files[num_shots][label_text].exists():
                    missing_text_gens_files.setdefault(num_shots, {})
                    missing_text_gens_files[num_shots][label_text] = text_gens_files[num_shots][label_text]
                else:
                    self.info(
                        f'>> Generations already exists for ('
                        f'dataset_name={dataset_name.canonical()}, '
                        f'model_name={model_name.canonical()}, '
                        f'num_shots={num_shots}, '
                        f'label_text={label_text}'
                        f') at "{text_gens_files[num_shots][label_text].path}"...'
                    )

        if len(missing_text_gens_files) > 0:
            llm_generations: Dict[int, Dict[str, TextGenerationsPredictionsBase]] = self._run_llm_evaluators(
                dataset_name=dataset_name,
                model_name=model_name,
                text_gens_files=missing_text_gens_files,
                label_verbalizer=label_verbalizer,
                max_new_tokens=synthesizrr_max_tokens,
                llm_batch_size=llm_batch_size,
                llm_submission_batch_size=llm_submission_batch_size,
                llm_tracking_batch_size=llm_tracking_batch_size,
                icl_dataset=icl_dataset,
                text_input_dataset=text_input_dataset,
                icl_template=icl_template,
                prompt_template=prompt_template,
                top_p=top_p,
                temperature=temperature,
                llm_resources_per_model=llm_resources_per_model,
                llm_num_concurrent_preds=llm_num_concurrent_preds,
                llm_num_models=llm_num_models,
                label_col=label_col,
                llm_load_balancing_strategy=llm_load_balancing_strategy,
            )
        text_gens: Dict[int, TextGenerationsPredictionsBase] = self._load_text_gens(
            text_gens_files=text_gens_files,
            dataset_name=dataset_name,
            model_name=model_name,
            label_col=label_col,
        )
        return dict(
            text_gens=text_gens,
        )

    def save_to(
            self,
            *,
            results_dir: FileMetadata,  ## E.g. RESULTS_DIR/,
            dataset_name: DatasetName,  ## E.g. ag_news
            corpus: Corpus,
            retriever: Retriever,
            model_name: ModelName,
            num_shots: int,
            label_text: str,
            label_verbalization: str,
            seed_set_data_split: DataSplit,
            seed_type: Literal['generated', 'train_set'],
            seed_size: int,
            seed_set_stratify_on_ground_truth: bool,
            icl_type: Literal['retrieved', 'curated', 'seed'],
            retr_icl_top_ks: List[conint(ge=1)],
            retr_icl_distance_range: Tuple[float, float],
            retr_icl_token_range: Tuple[conint(ge=1), conint(ge=1)],
            synthesizrr_top_k_range: range,
            synthesizrr_distance_range: Tuple[float, float],  ## (0.4, 0.9)
            synthesizrr_max_tokens: conint(ge=1),
            icl_template_hash: Optional[constr(min_length=6)],
            prompt_template_hash: Optional[constr(min_length=6)],
            num_samples_per_label: Optional[conint(ge=1)],
            seed: int,
            **kwargs,
    ) -> FileMetadata:
        ## RESULTS_DIR/retrieval-augmented-dataset-generation/ag_news/retrieval-data/toi_without_datasets/ag_news_seed_retr_output_toi_without_datasets.jsonlines
        if seed_type == 'generated':
            seed_type_str: str = f'generated_seed_set'
        elif seed_type == 'train_set':
            seed_type_str: str = f'{seed_set_data_split.lower()}_set_seed_set'
        else:
            raise not_impl('seed_type', seed_type)

        seed_size_str = ''
        if seed_size != dataset_name.seed_size():
            seed_size_str: str = f"-seed_size={seed_size}"
        stratified_str = ''
        if seed_set_stratify_on_ground_truth != DEFAULT_SEED_SET_STRATIFY_ON_GROUND_TRUTH:
            stratified_str: str = 'gt_stratified' if seed_set_stratify_on_ground_truth else 'not_gt_stratified'
            stratified_str: str = f"-stratified={stratified_str}"

        if icl_type == 'retrieved':
            icl_type_str: str = f'retrieved_icl_dataset'
        elif icl_type == 'curated':
            icl_type_str: str = f'curated_icl_dataset'
        elif icl_type == 'seed':
            icl_type_str: str = f'{seed_type}_seed_icl_dataset'
        else:
            raise not_impl('icl_type', icl_type)

        num_samples_per_label_str: str = get_default(num_samples_per_label, 'all')

        icl_template_hash_str: str = '' if icl_template_hash is None \
            else f'-icl_template_hash={icl_template_hash}'
        prompt_template_hash_str: str = '' if prompt_template_hash is None \
            else f'-prompt_template_hash={prompt_template_hash}'

        label_verbalization_str: str = ''
        if label_verbalization != dataset_name.label_verbalizer()[label_text]:
            label_verbalization_str: str = f'-vb={StringUtil.hash(label_verbalization, max_len=4)}'

        return results_dir.subdir_in_dir('retrieval-augmented-dataset-generation', return_metadata=True) \
            .subdir_in_dir(dataset_name.canonical(), return_metadata=True) \
            .subdir_in_dir('synthesizrr-generations', return_metadata=True) \
            .subdir_in_dir(corpus.canonical(), return_metadata=True) \
            .subdir_in_dir(retriever.canonical(), return_metadata=True) \
            .subdir_in_dir(model_name.canonical(), return_metadata=True) \
            .subdir_in_dir(f'num_samples_per_label={num_samples_per_label_str}', return_metadata=True) \
            .subdir_in_dir(f'num_shots={num_shots}', return_metadata=True) \
            .subdir_in_dir(f'label_text={label_text}', return_metadata=True) \
            .file_in_dir(
            f"{icl_type_str}"
            f"-{seed_type_str}{seed_size_str}{stratified_str}-retr_output"
            f"-dataset={dataset_name.canonical()}"
            f"-corpus={corpus.canonical()}"
            f"-retriever={retriever.canonical()}"
            f"-model_name={model_name.canonical()}"
            f"-num_samples_per_label={num_samples_per_label_str}"
            f"-num_shots={num_shots}"
            f"-label_text={label_text}"
            f"{label_verbalization_str}"
            f"-seed={seed}"
            f"-retr_icl_top_ks={retr_icl_top_ks}"
            f"-retr_icl_distance_range={retr_icl_distance_range}"
            f"-retr_icl_token_range={retr_icl_token_range}"
            f"-synthesizrr_top_k_range=range({synthesizrr_top_k_range.start}, {synthesizrr_top_k_range.stop}, {synthesizrr_top_k_range.step})"
            f"-synthesizrr_distance_range={synthesizrr_distance_range}"
            f"-synthesizrr_max_tokens={synthesizrr_max_tokens}"
            f"{icl_template_hash_str}"
            f"{prompt_template_hash_str}"
            f".parquet",
            return_metadata=True,
        ).update_params(file_format=FileFormat.PARQUET)
