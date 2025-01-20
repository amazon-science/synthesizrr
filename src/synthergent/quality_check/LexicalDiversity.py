from typing import *
import math, ray, re, multiprocessing as mp
from abc import ABC, abstractmethod
import pandas as pd
from nltk import word_tokenize
from pydantic import root_validator, conint, confloat, constr, Extra
from pydantic.typing import Literal
from sklearn.model_selection import train_test_split
from synthergent.base.constants import FileFormat, DataLayout, DataSplit, Parallelize
from synthergent.base.data import FileMetadata, Reader, Writer
from synthergent.base.data.reader import DataFrameReader
from synthergent.base.framework.ray_base import ActorComposite
from synthergent.base.framework.chain.Chain import Step
from synthergent.base.data.sdf import ScalableDataFrame, ScalableDataFrameRawType, DaskDataFrame
from synthergent.base.framework.metric import Metric, Metrics
from synthergent.base.framework.task.classification import ClassificationData
from synthergent.base.framework.task.text_generation import TextGenerationsPredictionsBase, GENERATED_TEXTS_COL
from synthergent.base.framework.task_data import Dataset
from synthergent.base.util import Parameters, as_list, as_set, \
    safe_validate_arguments, get_default, StringUtil, AutoEnum, auto, alias, not_impl, \
    binary_search, ProgressBar, shuffle_items, only_item, irange, punct_normalize, get_result, format_exception_msg, \
    dispatch, Timer, Executor, iter_batches, optional_dependency, accumulate, ignore_warnings_and_stdout, accumulate_iter
from synthergent.config import ScalingConfig
from synthergent.quality_check.QualityCheck import QualityCheck

with optional_dependency('nltk', 'spacy'):
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    import spacy
    from spacy.language import Language
    from spacy.tokens.doc import Doc


    class LexicalDiversity(QualityCheck):
        ## Self-BLEU
        class Params(QualityCheck.Params):
            """
            BaseModel for parameters. Expected to be overridden by subclasses.
            """
            col: str
            spacy_tokenization_model: str = 'en_core_web_lg'
            ngrams: Tuple[int, ...] = (1, 2, 3, 4, 5)
            num_cpus: int = 1
            batch_size: int = 40
            display_exclude: Tuple[str, ...] = ('num_cpus', 'num_gpus', 'batch_size')

        def evaluate(
                self,
                data: pd.DataFrame,
                scaling: ScalingConfig,
                executor: Optional[Executor],
                **kwargs,
        ) -> pd.DataFrame:
            scores: Dict[int, float] = LexicalDiversity.calc_self_bleu(
                docs=data[self.params.col].tolist(),
                scaling=scaling,
                executor=executor,
                verbosity=self.verbosity,
                **self.params.dict(exclude=['col']),
            )
            return pd.Series(scores, name='Self-BLEU').reset_index().rename(columns={'index': 'ngram'})

        @staticmethod
        def calc_self_bleu(
                docs: List[str],
                *,
                scaling: ScalingConfig,
                executor: Optional[Executor],
                spacy_tokenization_model: str,
                ngrams: Tuple[int, ...],
                batch_size: int,
                num_cpus: int,
                verbosity: int,
                **kwargs
        ) -> Dict[int, float]:
            ## Ensure at least 1 batch per process.
            num_docs: int = len(docs)
            with Timer('spacy_tokenize_docs', silent=verbosity <= 1):
                if scaling.parallelize in {Parallelize.ray}:
                    tokenized_docs: List[List[str]] = dispatch(
                        LexicalDiversity.spacy_tokenize_docs,
                        docs,
                        spacy_tokenization_model=spacy_tokenization_model,
                        num_cpus=min(num_cpus, 20),  ## Sent to Ray
                        spacy_tokenization_max_workers=min(num_cpus, 20),
                        batch_size=batch_size,
                        parallelize=scaling.parallelize,
                        executor=executor,
                    )
                else:
                    tokenized_docs: List[List[str]] = LexicalDiversity.spacy_tokenize_docs(
                        docs,
                        spacy_tokenization_model=spacy_tokenization_model,
                        spacy_tokenization_max_workers=min(num_cpus, 20),
                        batch_size=batch_size,
                    )
            ngram_self_bleu_scores: Dict[int, float] = {}
            for ngram in ngrams:
                if ngram == 1:
                    weights = (1.0, 0, 0, 0)
                elif ngram == 2:
                    weights = (0.5, 0.5, 0, 0)
                elif ngram == 3:
                    weights = (1.0 / 3, 1.0 / 3, 1.0 / 3, 0)
                elif ngram == 4:
                    weights = (0.25, 0.25, 0.25, 0.25)
                elif ngram == 5:
                    weights = (0.2, 0.2, 0.2, 0.2, 0.2)
                else:
                    raise ValueError
                with Timer(f'self_bleu_ngram={ngram}', silent=verbosity <= 1):
                    ngram_self_bleu_scores[ngram]: float = LexicalDiversity.self_bleu_ngram(
                        ngram=ngram,
                        weights=weights,
                        tokenized_docs=tokenized_docs,
                        num_docs=num_docs,
                        batch_size=batch_size,
                        executor=executor,
                        verbosity=verbosity,
                        parallelize=scaling.parallelize,
                        **kwargs
                    )
            return ngram_self_bleu_scores

        @staticmethod
        def spacy_tokenize_docs(
                docs: List[str],
                *,
                spacy_tokenization_model: str,
                spacy_tokenization_max_workers: int,
                batch_size: int,
                **kwargs,
        ) -> List[List[str]]:
            try:
                with ignore_warnings_and_stdout():
                    nlp: Language = spacy.load(spacy_tokenization_model, disable=['parser', 'tagger', 'ner'])
                    tokenized_docs: List[List[str]] = []
                    for sent_doc in nlp.pipe(docs, n_process=spacy_tokenization_max_workers, batch_size=batch_size):
                        toks: List[str] = []
                        for tok in sent_doc:
                            toks.append(tok.text)
                        tokenized_docs.append(toks)
                    return tokenized_docs
            except Exception as e:
                print(f'Error in "spacy_tokenize_docs":\n{format_exception_msg(e)}')
                raise e

        @staticmethod
        def self_bleu_ngram(
                *,
                weights: Tuple[float, ...],
                tokenized_docs: Union[List[List[str]], ray.ObjectRef],
                num_docs: int,
                batch_size: int,
                executor: Optional[Executor],
                verbosity: int,
                parallelize: Parallelize,
                **kwargs,
        ) -> float:
            futures: List = []
            for idx_batch in iter_batches(num_docs, batch_size):
                futures.append(dispatch(
                    LexicalDiversity.bleu_i_batch,
                    weights=weights,
                    tokenized_docs=tokenized_docs,
                    idx_batch=idx_batch,
                    executor=executor,
                    parallelize=parallelize,
                    delay=10e-3,
                    **kwargs,
                ))
            ngram_self_bleu_scores: List = []
            pbar: Optional[Dict] = None
            if verbosity >= 2:
                pbar: Dict = dict(
                    desc=LexicalDiversity.class_name,
                )
            try:
                for ngram_self_bleu_scores_batch in accumulate_iter(
                        futures,
                        progress_bar=pbar
                ):
                    ngram_self_bleu_scores.extend(ngram_self_bleu_scores_batch)
            except Exception as e:
                print(f'Error in "self_bleu_ngram": {format_exception_msg(e)}')
                raise e
            return sum(ngram_self_bleu_scores) / num_docs

        @staticmethod
        def bleu_i_batch(
                weights: Tuple[float, ...],
                tokenized_docs: Any,
                idx_batch: List[int],
                **kwargs
        ) -> List[float]:
            smoothing_function = SmoothingFunction().method1
            tokenized_docs: List[List[str]] = accumulate(tokenized_docs)
            return [
                LexicalDiversity.bleu_i(
                    weights=weights,
                    tokenized_docs=tokenized_docs,
                    smoothing_function=smoothing_function,
                    i=i,
                )
                for i in idx_batch
            ]

        @staticmethod
        def bleu_i(
                weights: Tuple[float, ...],
                tokenized_docs: List[List[str]],
                smoothing_function: Any,
                i: int,
        ) -> float:
            return sentence_bleu(
                references=tokenized_docs[:i] + tokenized_docs[i + 1:],
                hypothesis=tokenized_docs[i],
                weights=weights,
                smoothing_function=smoothing_function,
            )

    # Metric.of('Self-BLEU', params=dict(
    #     batch_size=50,
    #     num_cpus=self_bleu_num_cpus,
    #     spacy_ner_model='en_core_web_lg',
    #     max_retries=1,
    # ))
    #
    #
    # class SelfBLEU(TabularMetric):
    #     aliases = ['Self-BLEU']
    #
    #     class Params(TabularMetric.Params):
    #         class Config(TabularMetric.Params.Config):
    #             extra = Extra.allow
    #
    #         num_cpus: int = 8
    #         num_gpus: int = 0
    #         batch_size: int = 50
    #         settings: Dict = dict(
    #             spacy_tokenization_model='en_core_web_lg',
    #             ngrams=(1, 2, 3, 4, 5),
    #         )
    #         generations_col: str = GENERATED_TEXTS_COL
    #
    #     def compute_only(self, data: TextGenerationsPredictionsBase) -> Dict[int, float]:
    #         if not isinstance(data, TextGenerationsPredictionsBase):
    #             raise ValueError(
    #                 f'Expected data to be a {NextTokens} or {TextGenerations} instance; '
    #                 f'found: {type_str(data)}'
    #             )
    #         scores: Dict[int, float] = self.calc_self_bleu(
    #             docs=data.data[self.params.generations_col].tolist(),
    #             **self.params.settings,
    #             **self.params.dict(exclude={'settings'}),
    #         )
    #         return scores
