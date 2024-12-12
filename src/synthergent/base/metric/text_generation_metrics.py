from typing import *
from abc import ABC, abstractmethod
import pandas as pd, numpy as np, ray, math, random, gc, re
from collections import defaultdict
from synthergent.base.constants import Parallelize, Task, DataSplit, TaskOrStr, MLType, Alias, Status, DataLayout
from synthergent.base.util import as_tuple, type_str, optional_dependency, ignore_stdout_and_stderr, dispatch, accumulate, \
    as_list, flatten1d, iter_batches, accumulate_iter, get_default, parameterized_flatten, remove_nulls, \
    only_key, dispatch_executor, Timer, EnvUtil, set_param_from_alias, all_are_none, all_are_not_none, \
    ignore_warnings_and_stdout, best_k, whitespace_normalize, str_normalize, punct_normalize, remove_keys, StringUtil, \
    multiple_are_not_none, format_exception_msg, entropy, plotsum
from synthergent.base.framework import Dataset, Predictions, TabularMetric, Metric, CountingMetric, PercentageMetric, Evaluator, \
    Metrics, Datasets, Trainer, FileMetadata, SaveDatasetOrPredictions, load_predictions, Chain, ChainExecution, \
    RayTuneTrainer, RayTuneTrainerFinalModelsError, RayTuneTrainerTuneError
from synthergent.base.framework import TextGenerations, NextTokens, TextGenerationsPredictionsBase, GENERATED_TEXTS_COL, \
    ClassificationData, Prompts, TEXT_PROMPT_COL, PROMPT_TEMPLATE_INDEX_COL_PREFIX
from synthergent.base.framework.trainer.RayTuneTrainer import _ray_agg_final_model_metric_stats
from synthergent.base.framework.evaluator.RayEvaluator import LoadBalancingStrategy
from synthergent.base.framework.dl.torch import clear_device_cache
from synthergent.base.constants import AggregationStrategy
from ray import tune
from pydantic import conint, confloat, root_validator, Extra
from pydantic.typing import Literal


class TextLength(TabularMetric):
    aliases = [
        ## Creates every combination as an alias, e.g. num_parallel_trials, max_parallel_jobs, etc.
        '-'.join(remove_nulls([text_key, len_key, stats_key]))
        for text_key, len_key, stats_key in
        parameterized_flatten(
            [None, 'text'],
            ['len', 'length'],
            ['stats', 'statistics', None],
        )
    ]

    class Params(TabularMetric.Params):
        tokenizer: Optional[Callable] = None
        data_col: Optional[str] = None

    _row_char_count: Counter = Counter()
    _row_token_count: Optional[Counter] = None
    _num_texts: int = 0

    def update(self, data: Union[Dataset, Predictions]) -> Any:
        tokenizer: Optional[Callable] = self.params.tokenizer
        if tokenizer is not None and self._row_token_count is None:
            self._row_token_count: Counter = Counter()
        data_col: Optional[str] = self.params.data_col
        if data_col is None:
            if isinstance(data, TextGenerationsPredictionsBase):
                data_col: str = GENERATED_TEXTS_COL
            else:
                data_col: Dict[str, MLType] = {
                    col: mltype
                    for col, mltype in data.data_schema.features_schema.items()
                    if mltype is MLType.TEXT
                }
                if len(data_col) == 1:
                    data_col: str = only_key(data_col)
                else:
                    data_col: Optional[str] = None
        if data_col is None:
            raise ValueError(
                f'Could not infer {MLType.TEXT.capitalize()} column from {type_str(data)} '
                f'with schema: {data.data_schema}; please pass it explicitly to metric "{self.display_name}".'
            )
        generations: List[str] = data.data[data_col].tolist()
        for gen in generations:
            gen: str = get_default(gen, '')
            self._row_char_count[len(gen)] += 1
            if tokenizer is not None:
                tokenized_gen: List[str] = tokenizer(gen)
                self._row_token_count[len(tokenized_gen)] += 1
            self._num_texts += 1

    @property
    def _total_num_chars(self) -> int:
        return sum([num_chars * row_count for num_chars, row_count in self._row_char_count.items()])

    @property
    def _total_num_tokens(self) -> int:
        return sum([num_tokens * row_count for num_tokens, row_count in self._row_token_count.items()])

    def compute(self) -> Any:
        out: Dict = {
            'num_texts': self._num_texts,
            'total_num_chars': self._total_num_chars,
            'num_chars_avg': self._total_num_chars / self._num_texts,
            'num_chars_std': np.std(flatten1d([
                [num_chars] * row_count
                for num_chars, row_count in self._row_char_count.items()
            ]), ddof=1)
        }
        if self.params.tokenizer is not None:
            out.update({
                'total_num_tokens': self._total_num_tokens,
                'num_tokens_avg': self._total_num_tokens / self._num_texts,
                'num_tokens_std': np.std(flatten1d([
                    [num_tokens] * row_count
                    for num_tokens, row_count in self._row_token_count.items()
                ]), ddof=1)
            })
        return pd.DataFrame([out])


class CudaVisibleDevices(Metric):
    aliases = ['cuda']

    def compute_only(self, data: Any) -> Any:
        return {
            'num_devices': EnvUtil.num_gpus(),
            'devices_list': EnvUtil.cuda_visible_devices(),
        }


class RagasMetricBase(TabularMetric, ABC):
    CONTEXT: ClassVar[str] = 'context'
    QUESTION_TEXT: ClassVar[str] = 'question_text'
    ANSWER_TEXT: ClassVar[str] = 'answer_text'

    class Params(PercentageMetric.Params):
        llm_evaluator: Optional[Evaluator] = None
        algorithm: str
        hyperparams: Dict
        resources_per_model: Dict[Literal['cpu', 'gpu'], Union[confloat(ge=0.0, lt=1.0), conint(ge=0)]]
        num_models: Optional[conint(ge=1)]
        max_parallel_models: int = 0  ## 0 = no limit
        submission_batch_size: conint(ge=1) = 12
        verbosity: conint(ge=0) = 1

        question_col: str
        answer_col: str
        context_col: str

        num_cpus: int = 1  ## Set resources_per_model instead
        num_gpus: int = 0
        max_retries: int = 2

    def _preprocess_rag_gens(self, data: TextGenerationsPredictionsBase) -> TextGenerationsPredictionsBase:
        if not isinstance(data, TextGenerationsPredictionsBase):
            raise ValueError(
                f'Expected data to be a {NextTokens} or {TextGenerations} instance; '
                f'found: {type_str(data)}'
            )
        rag_gens: TextGenerationsPredictionsBase = data.copy().to_layout(DataLayout.PANDAS)
        rag_gens.data[self.CONTEXT] = rag_gens.data[self.params.context_col]
        rag_gens.data[self.QUESTION_TEXT] = rag_gens.data[self.params.question_col]
        rag_gens.data[self.ANSWER_TEXT] = rag_gens.data[self.params.answer_col]

        rag_gens.data_schema.features_schema[self.CONTEXT] = MLType.TEXT
        rag_gens.data_schema.features_schema[self.QUESTION_TEXT] = MLType.TEXT
        rag_gens.data_schema.features_schema[self.ANSWER_TEXT] = MLType.TEXT
        return rag_gens

    def _create_llm_evaluator(self) -> Evaluator:
        evaluator_class: str = 'ray'
        if str_normalize(self.params.algorithm) in {str_normalize('langchain'), str_normalize('bedrock')}:
            evaluator_class: str = 'local'

        llm_evaluator: Evaluator = Evaluator.of(
            evaluator_class,
            task=Task.NEXT_TOKEN_PREDICTION,
            algorithm=self.params.algorithm,
            hyperparams=self.params.hyperparams,
            resources_per_model=self.params.resources_per_model,
            num_models=self.params.num_models,
            max_parallel_models=self.params.max_parallel_models,
            verbosity=self.params.verbosity,
            **self.params.dict(exclude=(
                'task',
                'algorithm',
                'hyperparams',
                'resources_per_model',
                'num_models',
                'max_parallel_models',
                'verbosity',
                'question_col',
                'answer_col',
                'context_col',
                'num_cpus',
                'num_gpus',
                'max_retries',
            ))
        )
        return llm_evaluator

    def _stop_llm_evaluator(self, llm_evaluator: Optional[Evaluator]):
        if isinstance(llm_evaluator, Evaluator):
            llm_evaluator.stop()
            del llm_evaluator
            gc.collect()


class RagasFaithfulness(RagasMetricBase):
    """
    From the paper: https://arxiv.org/abs/2309.15217
    "Faithfulness refers to the idea that the answer should be grounded in the given context.
    This is important to avoid hallucinations, and to ensure that the retrieved context can act as a justification
    for the generated answer. Indeed, RAG systems are often used in applications where the factual consistency of
    the generated text w.r.t. the grounded sources is highly important, e.g. in domains such as law, where
    information is constantly evolving."

    We say that the answer a_s(q) is faithful to the context c(q) if the claims that are made in the answer can be
    inferred from the context.

    [Step 1a] To estimate faithfulness, we first use an LLM to extract a set of statements, S(a_s(q)). The aim of this
    step is to decompose longer sentences into shorter and more focused assertions.
    We use the following prompt for this step
    "
    Given a question and answer, create one or more statements from each sentence in the given answer.
    question: [question]
    answer: [answer]
    "

    [Step 1b] For each statement s_i in S, the LLM determines if s_i can be inferred from c(q) using a
    verification function v(s_i, c(q)). This verification step is carried out using the following prompt:
    "
    Consider the given context and following statements, then determine whether they are supported by the information present in the context. Provide a brief explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order at the end in the given format. Do not deviate from the specified format.
    statement: [statement 1]
    ...
    statement: [statement n]
    "

    [Step 2] The final faithfulness score, F, is then computed as F = |V|/|S|,
    where |V| is the number of statements that were supported according to the LLM
    and |S| is the total number of statements.
    """
    aliases = ['faithfulness']

    class Params(RagasMetricBase.Params):
        class Config(PercentageMetric.Params.Config):
            extra = Extra.allow

        statement_extraction_prompt: str = """
Given a Question and Answer, create one or more Statements from each sentence in the given Answer. Each statement should be separated by a newline. Only output the Statements.
Question: {question_text}
Answer: {answer_text}
Statements: """.strip() + ' '
        statement_verification_prompt: str = """
Context: {context}

Consider the above Context and following Statement. This Statement is part of a complete Paragraph which is there for reference. Determine whether the Statement is supported by the information present in the Context. Provide a brief explanation of your thinking in <thinking></thinking> tags before arriving at the Verdict (Yes/No). Provide a final Verdict for the Statement (Yes/No) at the end in a <verdict></verdict> tag. Do not deviate from this format. Only check whether the Statement is supported by the Paragraph.
Paragraph: {answer_text}
Statement: {statement}
Supported: """.strip() + ' '

        claude_replacements: List[Tuple[str, str]] = [
            ## statement_extraction_prompt:
            ('Given a Question and Answer',
             'Human: Given a Question and Answer'),
            ('Statements:', 'Assistant:'),
            ## statement_verification_prompt:
            ('Context:',
             'Human:\nContext:'),
            ('Supported:', 'Assistant:'),
        ]

        statement_extraction_ignore_prefixes: List[str] = [
            'Here is',
            'Here are',
            'As per my knowledge',
        ]

        @root_validator(pre=False)
        def _set_faithfulness_params(cls, params: Dict) -> Dict:
            if punct_normalize(params['algorithm']) in {punct_normalize('bedrock')} \
                    and punct_normalize('anthropic.claude') in punct_normalize(
                params['hyperparams'].get('model_name', '')):
                for repl in as_list(params['claude_replacements']):
                    params['statement_extraction_prompt']: str = params['statement_extraction_prompt'].replace(
                        repl[0],
                        repl[1],
                    )
                    params['statement_verification_prompt']: str = params['statement_verification_prompt'].replace(
                        repl[0],
                        repl[1],
                    )
            return params

    STATEMENTS: ClassVar[str] = 'statements'
    STATEMENT_EXTRACTION_PROMPTS: ClassVar[str] = 'statement_extraction_prompts'
    STATEMENTS_PARSED: ClassVar[str] = 'statements_parsed'
    SINGLE_STATEMENT_PARSED: ClassVar[str] = 'statement'
    VERIFICATION_PROMPTS: ClassVar[str] = 'verification_prompts'
    VERIFICATION: ClassVar[str] = 'verification'
    VERIFICATION_THINKING: ClassVar[str] = 'verification_thinking'
    VERIFICATION_VERDICT: ClassVar[str] = 'verification_verdict'
    FAITHFUL: ClassVar[str] = 'faithful'

    def compute_only(self, data: TextGenerationsPredictionsBase) -> pd.DataFrame:
        rag_gens: TextGenerationsPredictionsBase = self._preprocess_rag_gens(data)
        if self.params.llm_evaluator is None:
            llm_evaluator: Evaluator = self._create_llm_evaluator()
        else:
            llm_evaluator: Evaluator = self.params.llm_evaluator

        try:
            rag_gens_with_statements_flattened: TextGenerationsPredictionsBase = self._statement_extraction(
                rag_gens=rag_gens,
                llm_evaluator=llm_evaluator,
            )
            rag_gens_verification: TextGenerationsPredictionsBase = self._statement_verification_function(
                rag_gens_with_statements_flattened=rag_gens_with_statements_flattened,
                llm_evaluator=llm_evaluator,
            )
            return self._calc_faithfulness_metrics(rag_gens_verification)
        finally:
            if self.params.llm_evaluator is None:  ## We created the LLM evaluator
                self._stop_llm_evaluator(llm_evaluator)

    def _statement_extraction(
            self,
            *,
            rag_gens: TextGenerationsPredictionsBase,
            llm_evaluator: Evaluator,
    ) -> TextGenerationsPredictionsBase:
        index_col: str = rag_gens.data_schema.index_col
        prompts: Prompts = Prompts.of(
            split=DataSplit.UNSUPERVISED,
            task=Task.NEXT_TOKEN_PREDICTION,
            prompt_template=self.params.statement_extraction_prompt,
            prompt_template_apply='expand',
            ## Use pandas to prevent in-place modification errors:
            data=rag_gens.data.pandas().drop([
                TEXT_PROMPT_COL,
                GENERATED_TEXTS_COL,
            ], errors='ignore', axis=1),

            data_schema={
                index_col: MLType.INDEX,
                **remove_keys(
                    rag_gens.data_schema.features_schema,
                    [
                        TEXT_PROMPT_COL,
                        GENERATED_TEXTS_COL,
                    ]
                ),
            }
        ).read().apply_template()
        statement_extraction_gens: Predictions = llm_evaluator.evaluate(
            prompts,
            return_predictions=True,
            metrics=None,
            submission_batch_size=self.params.submission_batch_size,
        )
        assert isinstance(statement_extraction_gens, NextTokens)
        assert len(statement_extraction_gens) == len(rag_gens)
        statement_extraction_gens.set_layout(DataLayout.PANDAS)

        # with pd_display() as disp:
        #     disp('statement_extraction_gens.data:')
        #     disp(statement_extraction_gens.data)

        rag_gens_with_statements_df: pd.DataFrame = statement_extraction_gens.data.pandas().rename(columns={
            TEXT_PROMPT_COL: self.STATEMENT_EXTRACTION_PROMPTS,
            GENERATED_TEXTS_COL: self.STATEMENTS,
            f'{PROMPT_TEMPLATE_INDEX_COL_PREFIX}{index_col}': f'statement_extraction_{PROMPT_TEMPLATE_INDEX_COL_PREFIX}{index_col}',
        })[[
            f'statement_extraction_{PROMPT_TEMPLATE_INDEX_COL_PREFIX}{index_col}',
            self.STATEMENT_EXTRACTION_PROMPTS,
            self.STATEMENTS,
        ]].merge(
            left_on=f'statement_extraction_{PROMPT_TEMPLATE_INDEX_COL_PREFIX}{index_col}',
            right=rag_gens.data.pandas(),
            right_on=index_col,
        )
        rag_gens_with_statements_df[self.STATEMENTS_PARSED] = rag_gens_with_statements_df[self.STATEMENTS].apply(
            self._statement_parser,
            ignore_prefixes=self.params.statement_extraction_ignore_prefixes,
        )
        # with pd_display() as disp:
        #     disp('rag_gens_with_statements_df:')
        #     disp(rag_gens_with_statements_df)

        rag_gens_with_statements_flattened_df: List[Dict] = []
        for d in rag_gens_with_statements_df.to_dict(orient='records'):
            # print(d)

            for s_i, single_statement_parsed in enumerate(as_list(d[self.STATEMENTS_PARSED])):
                rag_gens_with_statements_flattened_df.append({
                    **d,
                    index_col: f'{d[index_col]}-{self.SINGLE_STATEMENT_PARSED}={StringUtil.pad_zeros(s_i, 1000)}',
                    self.SINGLE_STATEMENT_PARSED: get_default(
                        single_statement_parsed,  ## Can be None.
                        '',
                    ),
                })
        rag_gens_with_statements_flattened_df: pd.DataFrame = pd.DataFrame(rag_gens_with_statements_flattened_df)
        rag_gens_with_statements_flattened: TextGenerationsPredictionsBase = rag_gens.update_params(
            data=rag_gens_with_statements_flattened_df
        )
        rag_gens_with_statements_flattened.data_schema.features_schema[self.STATEMENT_EXTRACTION_PROMPTS] = MLType.TEXT
        rag_gens_with_statements_flattened.data_schema.features_schema[self.STATEMENTS] = MLType.TEXT
        rag_gens_with_statements_flattened.data_schema.features_schema[self.STATEMENTS_PARSED] = MLType.OBJECT
        rag_gens_with_statements_flattened.data_schema.features_schema[self.SINGLE_STATEMENT_PARSED] = MLType.TEXT

        # with pd_display() as disp:
        #     disp('rag_gens_with_statements_flattened.data:')
        #     disp(rag_gens_with_statements_flattened.data)

        return rag_gens_with_statements_flattened

    def stats(self, *, log: bool = True):
        if self.value is None:
            raise ValueError(f'You must first evaluate the metric.')
        faithfulness_df: pd.DataFrame = self.value.drop(['rag_gens_verification'], axis=1)
        faithfulness_details_df: pd.DataFrame = pd.DataFrame(
            self.value.value['rag_gens_verification'].iloc[0]
        )

    @staticmethod
    def _statement_parser(statements: str, *, ignore_prefixes: List[str]) -> Optional[List[str]]:
        statements: str = whitespace_normalize(statements)
        statements_list: List[str] = []
        for stmt in statements.split('\n'):
            include_stmt: bool = True
            if len(punct_normalize(stmt)) == 0:
                include_stmt: bool = False
            for prefix in ignore_prefixes:
                if punct_normalize(stmt).startswith(punct_normalize(prefix)):
                    include_stmt: bool = False
            if include_stmt is True:
                statements_list.append(stmt)
        if len(statements_list) == 0:
            return None
        return statements_list

    def _statement_verification_function(
            self,
            *,
            rag_gens_with_statements_flattened: TextGenerationsPredictionsBase,
            llm_evaluator: Evaluator,
    ) -> TextGenerationsPredictionsBase:
        index_col: str = rag_gens_with_statements_flattened.data_schema.index_col
        prompts: Prompts = Prompts.of(
            split=DataSplit.UNSUPERVISED,
            task=Task.NEXT_TOKEN_PREDICTION,
            prompt_template=self.params.statement_verification_prompt,
            prompt_template_apply='expand',
            ## Use pandas to prevent in-place modification errors:
            data=rag_gens_with_statements_flattened.data.pandas().drop([
                TEXT_PROMPT_COL,
                GENERATED_TEXTS_COL,
            ], errors='ignore', axis=1),
            data_schema={
                index_col: MLType.INDEX,
                **remove_keys(
                    rag_gens_with_statements_flattened.data_schema.features_schema,
                    [
                        TEXT_PROMPT_COL,
                        GENERATED_TEXTS_COL,
                    ]
                ),
            }
        ).read().apply_template()
        # with pd_display() as disp:
        #     disp('prompts:')
        #     disp(prompts.data.pandas())

        statement_verification_gens: Predictions = llm_evaluator.evaluate(
            prompts,
            return_predictions=True,
            metrics=None,
            submission_batch_size=self.params.submission_batch_size,
        )
        assert isinstance(statement_verification_gens, NextTokens)
        assert len(statement_verification_gens) == len(rag_gens_with_statements_flattened)
        statement_verification_gens.set_layout(DataLayout.PANDAS)
        # with pd_display() as disp:
        #     disp('statement_verification_gens.data')
        #     disp(statement_verification_gens.data.pandas())

        rag_gens_verification_df: pd.DataFrame = statement_verification_gens.data.pandas().rename(columns={
            TEXT_PROMPT_COL: self.VERIFICATION_PROMPTS,
            GENERATED_TEXTS_COL: self.VERIFICATION,
            f'{PROMPT_TEMPLATE_INDEX_COL_PREFIX}{index_col}': f'verification_{PROMPT_TEMPLATE_INDEX_COL_PREFIX}{index_col}',
        })[[
            f'verification_{PROMPT_TEMPLATE_INDEX_COL_PREFIX}{index_col}',
            self.VERIFICATION_PROMPTS,
            self.VERIFICATION,
        ]].merge(
            left_on=f'verification_{PROMPT_TEMPLATE_INDEX_COL_PREFIX}{index_col}',
            right=rag_gens_with_statements_flattened.data.pandas(),
            right_on=index_col,
        )
        rag_gens_verification_df[self.VERIFICATION_THINKING] = rag_gens_verification_df[self.VERIFICATION].apply(
            self._thinking_parser,
        )
        rag_gens_verification_df[self.VERIFICATION_VERDICT] = rag_gens_verification_df[self.VERIFICATION].apply(
            self._verdict_parser,
        )
        rag_gens_verification: TextGenerationsPredictionsBase = rag_gens_with_statements_flattened.update_params(
            data=rag_gens_verification_df
        )
        rag_gens_verification.data_schema.features_schema[self.VERIFICATION_PROMPTS] = MLType.TEXT
        rag_gens_verification.data_schema.features_schema[self.VERIFICATION] = MLType.TEXT
        rag_gens_verification.data_schema.features_schema[self.VERIFICATION_THINKING] = MLType.TEXT
        rag_gens_verification.data_schema.features_schema[self.VERIFICATION_VERDICT] = MLType.TEXT
        return rag_gens_verification

    @staticmethod
    def _thinking_parser(verification_text: str) -> Optional[str]:
        from bs4 import BeautifulSoup as BS
        thinking: Optional[Any] = BS(verification_text).find('thinking')
        if thinking is not None:
            thinking: str = thinking.text.strip()
            if len(thinking) == 0:
                return None
        return thinking

    @staticmethod
    def _verdict_parser(verification_text: str, verdict_prefix: str = 'verdict:') -> Optional[str]:
        from bs4 import BeautifulSoup as BS
        verdict: Optional[Any] = BS(verification_text).find('verdict')
        if verdict is not None:
            verdict: str = punct_normalize(verdict.text).strip().capitalize()
            if len(verdict) == 0:
                return None
        else:
            verdict_idx: int = verification_text.lower().find(verdict_prefix)
            if verdict_idx == -1:
                return None
            verdict: str = punct_normalize(verification_text[verdict_idx + len(verdict_prefix):]).strip().capitalize()
        if verdict.startswith('Yes'):
            return 'Yes'
        if verdict.startswith('No'):
            return 'No'
        return verdict

    def _calc_faithfulness_metrics(self, rag_gens_verification: TextGenerationsPredictionsBase) -> pd.DataFrame:
        """
        The final faithfulness score, F, is then computed as F = |V|/|S|,
        where |V| is the number of statements that were supported according to the LLM
        and |S| is the total number of statements.
        """

        # with pd_display() as disp:
        #     disp('rag_gens_verification:')
        #     disp(rag_gens_verification.data.pandas())

        def parsed_statement_is_valid(single_statement_parsed: Optional[str]):
            return isinstance(single_statement_parsed, str) and len(single_statement_parsed) > 0

        def parsed_verdict_is_valid(verdict: Optional[str]):
            return punct_normalize(verdict) in {punct_normalize('Yes'), punct_normalize('No')}

        def parsed_thinking_is_valid(thinking: Optional[str]):
            return isinstance(thinking, str) and len(thinking) > 0

        def is_faithful(verdict: Optional[str]):
            if verdict is None:
                return None
            return punct_normalize(verdict) in {punct_normalize('Yes')}

        faithfulness_df: pd.DataFrame = rag_gens_verification.data.pandas()
        faithfulness_df[self.FAITHFUL] = faithfulness_df[
            self.VERIFICATION_VERDICT
        ].apply(is_faithful)

        faithfulness_metrics: Dict = {}
        faithfulness_metrics['total_num_statements']: int = len(rag_gens_verification)
        faithfulness_metrics['valid_num_statements']: int = faithfulness_df[
            self.SINGLE_STATEMENT_PARSED
        ].apply(parsed_statement_is_valid).value_counts().get(True, 0)

        faithfulness_metrics['valid_num_verdicts']: int = faithfulness_df[
            self.VERIFICATION_VERDICT
        ].apply(parsed_verdict_is_valid).value_counts().get(True, 0)

        faithfulness_metrics['num_yes_verdicts']: int = faithfulness_df[
            self.VERIFICATION_VERDICT
        ].value_counts().get(punct_normalize('Yes').capitalize(), 0)

        # faithfulness_metrics['faithful_avg']: float = faithfulness_df[
        #     self.FAITHFUL
        # ].fillna(False).mean()
        faithfulness_metrics['faithful_avg_valid_only']: float = faithfulness_df[
            self.FAITHFUL
        ].dropna().mean()

        # print('faithfulness_metrics:')
        # print(faithfulness_metrics)

        faithfulness_metrics['faithfulness_score_total_num_statements'] = \
            faithfulness_metrics['num_yes_verdicts'] / faithfulness_metrics['total_num_statements'] if \
                faithfulness_metrics['total_num_statements'] > 0 else 0.0
        faithfulness_metrics['faithfulness_score_valid_num_statements'] = \
            faithfulness_metrics['num_yes_verdicts'] / faithfulness_metrics['valid_num_statements'] if \
                faithfulness_metrics['valid_num_statements'] > 0 else 0.0
        faithfulness_metrics['faithfulness_score_valid_num_verdicts'] = \
            faithfulness_metrics['num_yes_verdicts'] / faithfulness_metrics['valid_num_verdicts'] if \
                faithfulness_metrics['valid_num_verdicts'] > 0 else 0.0

        faithfulness_metrics['rag_gens_verification'] = faithfulness_df.to_dict(orient='records')
        faithfulness_metrics: pd.DataFrame = pd.DataFrame([faithfulness_metrics])

        return faithfulness_metrics


class RagasContextRelevance(RagasMetricBase):
    """
    From the paper: https://arxiv.org/abs/2309.15217
    "The context c(q) is considered relevant to the extent that it exclusively contains information that is
    needed to answer the question. In particular, this metric aims to penalise the inclusion of redundant
    information.

    [Step 1] To estimate context relevance, given a question q and its context c(q), the LLM extracts a
    subset of sentences, S_ext, from c(q) that are crucial to answer q, using the following prompt:
    "
    Please extract relevant sentences from the provided context that can potentially help answer the following
    question. If no relevant sentences are found, or if you believe the question cannot be answered from the
    given context, return the phrase "Insufficient Information". While extracting candidate sentences you’re
    not allowed to make any changes to sentences from given context.
    "

    [Step 2] The context relevance score (for each question) is then computed as:
    CR = (number of extracted sentences) / (total number of sentences in c(q))
    """
    aliases = ['context_relevance']

    class Params(RagasMetricBase.Params):
        class Config(PercentageMetric.Params.Config):
            extra = Extra.allow

        relevant_context_extraction_prompt: str = """
Context: {context}

Please extract relevant sentences from the above context that can potentially help answer the following question. The extracted sentences should be separated by newlines. If no relevant sentences are found, or if you believe the question cannot be answered from the given context, return the phrase "Insufficient Information". While extracting candidate sentences you’re not allowed to make any changes to sentences from given context.
question: {question_text}
Relevant Sentences: """.strip() + ' '

        claude_replacements: List[Tuple[str, str]] = [
            ## relevant_context_extraction_prompt:
            ('Context:',
             'Human:\nContext:'),
            ('Relevant Sentences:', 'Assistant:'),
        ]

        @root_validator(pre=False)
        def _set_context_relevance_params(cls, params: Dict) -> Dict:
            if punct_normalize(params['algorithm']) in {punct_normalize('bedrock')} \
                    and punct_normalize('anthropic.claude') in punct_normalize(
                params['hyperparams'].get('model_name', '')):
                for repl in as_list(params['claude_replacements']):
                    params['relevant_context_extraction_prompt']: str = params[
                        'relevant_context_extraction_prompt'].replace(
                        repl[0],
                        repl[1],
                    )
            return params

    CONTEXT_SENTENCES_PARSED: ClassVar[str] = 'context_sentences_parsed'
    CONTEXT_SENTENCES_PARSED_IS_VALID: ClassVar[str] = 'context_sentences_parsed_is_valid'
    NUM_CONTEXT_SENTENCES_PARSED: ClassVar[str] = 'num_context_sentences_parsed'
    NUM_VALID_CONTEXT_SENTENCES_PARSED: ClassVar[str] = 'num_valid_context_sentences_parsed'
    RELEVANT_CONTEXT_SENTENCES: ClassVar[str] = 'relevant_context_sentences'
    RELEVANT_CONTEXT_SENTENCES_PARSED: ClassVar[str] = 'relevant_context_sentences_parsed'
    NUM_RELEVANT_CONTEXT_SENTENCES_PARSED: ClassVar[str] = 'num_relevant_context_sentences_parsed'
    NUM_VALID_RELEVANT_CONTEXT_SENTENCES_PARSED: ClassVar[str] = 'num_valid_relevant_context_sentences_parsed'
    RELEVANT_CONTEXT_SENTENCES_PARSED_IS_VALID: ClassVar[str] = 'relevant_context_sentences_parsed_is_valid'
    RELEVANT_CONTEXT_SENTENCE_EXTRACTION_PROMPTS: ClassVar[str] = 'relevant_context_sentence_extraction_prompts'
    INSUFFICIENT_INFORMATION: ClassVar[str] = 'Insufficient Information'
    ANSWERABLE: ClassVar[str] = 'answerable'
    ANSWERABLE_IS_VALID: ClassVar[str] = 'answerable_is_valid'
    CONTEXT_RELEVANCE_SCORE: ClassVar[str] = 'context_relevance_score'
    CONTEXT_RELEVANCE_SCORE_VALID_ONLY: ClassVar[str] = 'context_relevance_score_valid_only'

    def compute_only(self, data: TextGenerationsPredictionsBase) -> pd.DataFrame:
        rag_gens: TextGenerationsPredictionsBase = self._preprocess_rag_gens(data)
        if self.params.llm_evaluator is None:
            llm_evaluator: Evaluator = self._create_llm_evaluator()
        else:
            llm_evaluator: Evaluator = self.params.llm_evaluator

        try:
            rag_gens_with_relevant_contexts: TextGenerationsPredictionsBase = self._relevant_context_extraction(
                rag_gens=rag_gens,
                llm_evaluator=llm_evaluator,
            )
            return self._calc_context_relevance_metrics(rag_gens_with_relevant_contexts)
        finally:
            if self.params.llm_evaluator is None:  ## We created the LLM evaluator
                self._stop_llm_evaluator(llm_evaluator)

    def stats(self, *, log: bool = True):
        if self.value is None:
            raise ValueError(f'You must first evaluate the metric.')
        context_relevance_df: pd.DataFrame = self.value.drop(['rag_gens_with_relevant_context'], axis=1)
        context_relevance_details_df: pd.DataFrame = pd.DataFrame(
            self.value['rag_gens_with_relevant_context'].iloc[0]
        )

    def _relevant_context_extraction(
            self,
            *,
            rag_gens: TextGenerationsPredictionsBase,
            llm_evaluator: Evaluator,
    ) -> TextGenerationsPredictionsBase:
        index_col: str = rag_gens.data_schema.index_col
        prompts: Prompts = Prompts.of(
            split=DataSplit.UNSUPERVISED,
            task=Task.NEXT_TOKEN_PREDICTION,
            prompt_template=self.params.relevant_context_extraction_prompt,
            prompt_template_apply='expand',
            ## Use pandas to prevent in-place modification errors:
            data=rag_gens.data.pandas().drop([
                TEXT_PROMPT_COL,
                GENERATED_TEXTS_COL,
            ], errors='ignore', axis=1),

            data_schema={
                index_col: MLType.INDEX,
                **remove_keys(
                    rag_gens.data_schema.features_schema,
                    [
                        TEXT_PROMPT_COL,
                        GENERATED_TEXTS_COL,
                    ]
                ),
            }
        ).read().apply_template()
        relevant_context_extraction: Predictions = llm_evaluator.evaluate(
            prompts,
            return_predictions=True,
            metrics=None,
            submission_batch_size=self.params.submission_batch_size,
        )
        assert isinstance(relevant_context_extraction, NextTokens)
        assert len(relevant_context_extraction) == len(rag_gens)
        relevant_context_extraction.set_layout(DataLayout.PANDAS)

        # with pd_display() as disp:
        #     disp('relevant_context_extraction.data:')
        #     disp(relevant_context_extraction.data)

        rag_gens_with_relevant_context_df: pd.DataFrame = relevant_context_extraction.data.pandas().rename(columns={
            f'{PROMPT_TEMPLATE_INDEX_COL_PREFIX}{index_col}': f'relevant_context_statement_extraction_{PROMPT_TEMPLATE_INDEX_COL_PREFIX}{index_col}',
            TEXT_PROMPT_COL: self.RELEVANT_CONTEXT_SENTENCE_EXTRACTION_PROMPTS,
            GENERATED_TEXTS_COL: self.RELEVANT_CONTEXT_SENTENCES,
        })[[
            f'relevant_context_statement_extraction_{PROMPT_TEMPLATE_INDEX_COL_PREFIX}{index_col}',
            self.RELEVANT_CONTEXT_SENTENCE_EXTRACTION_PROMPTS,
            self.RELEVANT_CONTEXT_SENTENCES,
        ]].merge(
            left_on=f'relevant_context_statement_extraction_{PROMPT_TEMPLATE_INDEX_COL_PREFIX}{index_col}',
            right=rag_gens.data.pandas(),
            right_on=index_col,
        )
        rag_gens_with_relevant_context_df[self.CONTEXT_SENTENCES_PARSED] = \
            rag_gens_with_relevant_context_df[self.params.context_col].apply(self._context_parser)
        rag_gens_with_relevant_context_df[self.RELEVANT_CONTEXT_SENTENCES_PARSED] = \
            rag_gens_with_relevant_context_df[self.RELEVANT_CONTEXT_SENTENCES].apply(self._relevant_context_parser)

        rag_gens_with_relevant_context: TextGenerationsPredictionsBase = rag_gens.update_params(
            data=rag_gens_with_relevant_context_df
        )
        rag_gens_with_relevant_context.data_schema.features_schema[
            self.RELEVANT_CONTEXT_SENTENCE_EXTRACTION_PROMPTS
        ] = MLType.TEXT
        rag_gens_with_relevant_context.data_schema.features_schema[
            self.RELEVANT_CONTEXT_SENTENCES
        ] = MLType.TEXT
        rag_gens_with_relevant_context.data_schema.features_schema[
            self.CONTEXT_SENTENCES_PARSED
        ] = MLType.OBJECT
        rag_gens_with_relevant_context.data_schema.features_schema[
            self.RELEVANT_CONTEXT_SENTENCES_PARSED
        ] = MLType.OBJECT

        # with pd_display() as disp:
        #     disp('rag_gens_with_relevant_context.data:')
        #     disp(rag_gens_with_relevant_context.data)

        return rag_gens_with_relevant_context

    def _calc_context_relevance_metrics(
            self,
            rag_gens_with_relevant_context: TextGenerationsPredictionsBase,
    ) -> pd.DataFrame:
        """
        The context relevance score (for each question) is then computed as:
        CR = (number of extracted sentences) / (total number of sentences in c(q))
        """

        # with pd_display() as disp:
        #     disp('rag_gens_with_relevant_context:')
        #     disp(rag_gens_with_relevant_context.data.pandas())
        def parsed_relevant_context_sentence_is_valid(relevant_context_sentence: Optional[str]):
            return isinstance(relevant_context_sentence, str) and len(relevant_context_sentence) > 0

        def parsed_context_sentence_is_valid(context_sentence: Optional[str]):
            return isinstance(context_sentence, str) and len(context_sentence) > 0

        def row_context_relevance_metrics(d: Dict) -> Dict:
            ## Numerator:
            if d[self.RELEVANT_CONTEXT_SENTENCES_PARSED] is not None:
                d[self.RELEVANT_CONTEXT_SENTENCES_PARSED_IS_VALID] = True
                if d[self.RELEVANT_CONTEXT_SENTENCES_PARSED] == self.INSUFFICIENT_INFORMATION:
                    d[self.ANSWERABLE] = False
                    d[self.ANSWERABLE_IS_VALID] = True
                    d[self.NUM_RELEVANT_CONTEXT_SENTENCES_PARSED] = 0
                    d[self.NUM_VALID_RELEVANT_CONTEXT_SENTENCES_PARSED] = 0
                else:
                    d[self.ANSWERABLE] = True
                    d[self.ANSWERABLE_IS_VALID] = True
                    d[self.NUM_RELEVANT_CONTEXT_SENTENCES_PARSED] = len(d[self.RELEVANT_CONTEXT_SENTENCES_PARSED])
                    d[self.NUM_VALID_RELEVANT_CONTEXT_SENTENCES_PARSED] = len([
                        x for x in d[self.RELEVANT_CONTEXT_SENTENCES_PARSED]
                        if parsed_relevant_context_sentence_is_valid(x)
                    ])
            else:
                d[self.ANSWERABLE] = None
                d[self.ANSWERABLE_IS_VALID] = False
                d[self.RELEVANT_CONTEXT_SENTENCES_PARSED_IS_VALID] = False
                d[self.NUM_RELEVANT_CONTEXT_SENTENCES_PARSED] = None
                d[self.NUM_VALID_RELEVANT_CONTEXT_SENTENCES_PARSED] = None

            ## Denominator:
            if d[self.CONTEXT_SENTENCES_PARSED] is not None:
                d[self.CONTEXT_SENTENCES_PARSED_IS_VALID] = True
                d[self.NUM_CONTEXT_SENTENCES_PARSED] = len(d[self.CONTEXT_SENTENCES_PARSED])
                d[self.NUM_VALID_CONTEXT_SENTENCES_PARSED] = len([
                    x for x in d[self.CONTEXT_SENTENCES_PARSED]
                    if parsed_context_sentence_is_valid(x)
                ])
            else:
                d[self.CONTEXT_SENTENCES_PARSED_IS_VALID] = False
                d[self.NUM_CONTEXT_SENTENCES_PARSED] = None
                d[self.NUM_VALID_CONTEXT_SENTENCES_PARSED] = None

            ## All:
            if all_are_not_none(
                    d[self.NUM_RELEVANT_CONTEXT_SENTENCES_PARSED],
                    d[self.NUM_CONTEXT_SENTENCES_PARSED],
            ):
                d[self.CONTEXT_RELEVANCE_SCORE]: float = \
                    d[self.NUM_RELEVANT_CONTEXT_SENTENCES_PARSED] / d[self.NUM_CONTEXT_SENTENCES_PARSED]
            else:
                d[self.CONTEXT_RELEVANCE_SCORE] = None
            ## Valid only:
            if all_are_not_none(
                    d[self.NUM_VALID_RELEVANT_CONTEXT_SENTENCES_PARSED],
                    d[self.NUM_VALID_CONTEXT_SENTENCES_PARSED],
            ):
                d[self.CONTEXT_RELEVANCE_SCORE_VALID_ONLY]: float = \
                    d[self.NUM_VALID_RELEVANT_CONTEXT_SENTENCES_PARSED] / d[self.NUM_VALID_CONTEXT_SENTENCES_PARSED]
            else:
                d[self.CONTEXT_RELEVANCE_SCORE_VALID_ONLY] = None
            return d

        context_relevance_metrics_df: List[Dict] = []
        for row in rag_gens_with_relevant_context.data.to_list_of_dict():
            context_relevance_metrics_df.append(row_context_relevance_metrics(row))
        context_relevance_metrics_df: pd.DataFrame = pd.DataFrame(context_relevance_metrics_df)

        context_relevance_metrics: Dict = {}

        context_relevance_metrics['num_relevant_contexts'] = \
            context_relevance_metrics_df[self.CONTEXT_RELEVANCE_SCORE].dropna().shape[0]
        context_relevance_metrics['context_relevance_score_avg'] = \
            context_relevance_metrics_df[self.CONTEXT_RELEVANCE_SCORE].dropna().mean()

        context_relevance_metrics['num_relevant_contexts_valid_only'] = \
            context_relevance_metrics_df[self.CONTEXT_RELEVANCE_SCORE_VALID_ONLY].dropna().shape[0]
        context_relevance_metrics['context_relevance_score_avg_valid_only'] = \
            context_relevance_metrics_df[self.CONTEXT_RELEVANCE_SCORE_VALID_ONLY].dropna().mean()

        # context_relevance_metrics['answerable_avg']: float = \
        #     context_relevance_metrics_df[self.ANSWERABLE].fillna(False).mean()
        context_relevance_metrics['answerable_avg_valid_only']: float = \
            context_relevance_metrics_df[self.ANSWERABLE].dropna().mean()

        context_relevance_metrics['rag_gens_with_relevant_context']: List[Dict] = \
            context_relevance_metrics_df.to_dict(orient='records')
        context_relevance_metrics: pd.DataFrame = pd.DataFrame([context_relevance_metrics])

        return context_relevance_metrics

    @classmethod
    def _context_parser(cls, context: str) -> Optional[List[str]]:
        if not isinstance(context, str):
            return None
        context: str = whitespace_normalize(context).strip()
        if len(context) == 0:
            return None
        context_sentences_list: List[str] = context.split('\n')
        if len(context_sentences_list) == 0:
            return None
        return context_sentences_list

    @classmethod
    def _relevant_context_parser(cls, relevant_context: str) -> Optional[Union[List[str], str]]:
        if not isinstance(relevant_context, str):
            return None
        relevant_context: str = whitespace_normalize(relevant_context).strip()
        if len(relevant_context) == 0:
            return None
        if punct_normalize(relevant_context) == punct_normalize(cls.INSUFFICIENT_INFORMATION):
            return cls.INSUFFICIENT_INFORMATION
        relevant_context_sentences_list: List[str] = relevant_context.split('\n')
        if len(relevant_context_sentences_list) == 0:
            return None
        return relevant_context_sentences_list


with optional_dependency('mauve-text'):
    class Mauve(PercentageMetric):
        class Params(PercentageMetric.Params):
            class Config(PercentageMetric.Params.Config):
                extra = Extra.allow

            references_col: str
            generations_col: str = GENERATED_TEXTS_COL
            num_cpus: int = 8
            num_gpus: int = 1
            max_retries: int = 2
            settings: Dict = dict(
                device_id=0,
                max_text_length=512,
                verbose=False,
                batch_size=1,
                num_buckets=30,
                featurize_model_name='gpt2-xl',
                mauve_scaling_factor=1,
            )

        def compute_only(self, data: TextGenerationsPredictionsBase) -> float:
            if not isinstance(data, TextGenerationsPredictionsBase):
                raise ValueError(
                    f'Expected data to be a {NextTokens} or {TextGenerations} instance; '
                    f'found: {type_str(data)}'
                )
            score: float = self.calc_mauve(
                ref_texts=data.data[self.params.references_col].tolist(),
                gen_texts=data.data[self.params.generations_col].tolist(),
                settings=self.params.settings,
                **self.params.dict(exclude={'references_col', 'settings'}),
            )
            return score

        @classmethod
        def calc_mauve(
                cls,
                ref_texts: List[str],
                gen_texts: List[str],
                settings: Dict,
                **kwargs,
        ) -> float:
            import mauve as _mauve
            with ignore_warnings_and_stdout():  ## Suppress tqdm & other outputs in MAUVE calculation
                if settings.get('device_id') is not None:  ## Use a randomly-assigned GPU.
                    settings['device_id'] = random.choice(EnvUtil.cuda_visible_devices())
                clear_device_cache()
                try:
                    computed_mauve = _mauve.compute_mauve(
                        p_text=ref_texts,
                        q_text=gen_texts,
                        **settings,
                    )
                    return float(computed_mauve.mauve)
                finally:
                    global MODEL
                    try:
                        MODEL.to('cpu')
                        del MODEL
                    except NameError as e:
                        pass
                    clear_device_cache()

with optional_dependency('nltk', 'spacy'):
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    import spacy
    from spacy.language import Language
    from spacy.tokens.doc import Doc


    class EntityCount(TabularMetric):
        class Params(TabularMetric.Params):
            class Config(TabularMetric.Params.Config):
                extra = Extra.allow

            num_cpus: int = 8
            num_gpus: int = 0
            spacy_ner_model: str = 'en_core_web_lg'
            batch_size: int = 50
            generations_col: str = GENERATED_TEXTS_COL

        ## Overvall counts of entities:
        _entity_counts: Counter = Counter()
        ## Overvall counts of entity-labels:
        _entity_label_counts: Counter = Counter()
        ## "Apple" can be the company or the fruit; this counts the spread of labels for each identified entity:
        _entitywise_label_counts: Dict[str, Counter] = {}
        ## Number of entites per row
        _row_num_entities: Counter = Counter()
        _num_rows: int = 0
        _entity_count_df: Optional[pd.DataFrame] = None

        def update(self, data: TextGenerationsPredictionsBase):
            with ignore_warnings_and_stdout():
                docs: List[str] = data.data[self.params.generations_col].tolist()
                num_docs: int = len(docs)
                batch_size: int = self.params.batch_size
                nlp: Language = spacy.load(self.params.spacy_ner_model)
                max_workers: int = max(1, self.params.num_cpus, math.floor(num_docs / (batch_size * 1)))
                for text_doc in nlp.pipe(docs, n_process=max_workers, batch_size=batch_size):
                    for ent in text_doc.ents:
                        entity: str = ent.text
                        entity_label: str = ent.label_
                        self._entity_counts[entity] += 1
                        self._entity_label_counts[entity_label] += 1
                        if entity not in self._entitywise_label_counts:
                            self._entitywise_label_counts[entity] = Counter()
                        self._entitywise_label_counts[entity][entity_label] += 1
                    self._row_num_entities[len(text_doc.ents)] += 1
                    self._num_rows += 1
                entity_count_df: List[Dict] = []
                entity_count_df_index: List[str] = []
                for entity, entity_label_count in self._entitywise_label_counts.items():
                    entity_count_df_index.append(entity)
                    entity_count_df.append(dict(entity_label_count))
                self._entity_count_df: pd.DataFrame = pd.DataFrame(
                    entity_count_df,
                    index=entity_count_df_index
                ).fillna(0).astype(int)

        def compute(self) -> pd.DataFrame:
            return self._entity_count_df

        def entity_entropy(self) -> Dict[str, float]:
            entity_counts_df: pd.DataFrame = self.value
            return {
                entity: entropy(entity_counts_df[entity] / entity_counts_df[entity].sum())
                for entity in entity_counts_df.columns
            }

        def entity_counts(self) -> Dict[str, Dict]:
            entity_counts_df: pd.DataFrame = self.value
            entity_counts_dict: Dict = {}
            for entity in entity_counts_df.columns:
                entity_count_vals: pd.Series = entity_counts_df[entity]
                entity_count_vals: pd.Series = entity_count_vals[entity_count_vals > 0]
                entity_counts_dict[entity] = {
                    'pct': (100 * entity_count_vals / entity_count_vals.sum()).sort_values(ascending=False),
                    'count': entity_count_vals.sort_values(ascending=False),
                }
            return entity_counts_dict

        def top_k_entities(
                self,
                k: int,
                *,
                entity_labels: Optional[List[str]] = None,
        ) -> Tuple[pd.DataFrame, List[str]]:
            entity_count_df: pd.DataFrame = self._entity_count_df
            if entity_labels is None:
                entity_labels: List[str] = list(entity_count_df.columns)
            entity_labels: List[str] = as_list(entity_labels)
            top_k_entities_df: Dict[str, pd.Series] = {}
            top_k_entities_per_label: Dict[str, pd.Series] = {}
            for entity_label in entity_labels:
                top_k_entities_per_label[entity_label]: pd.Series = entity_count_df[entity_label].iloc[
                    best_k(entity_count_df[entity_label], k=k, how='max', indexes_only=True)
                ]
                for how in ['pct', 'frac', 'count']:
                    if how == 'pct':
                        _col_pair = (
                                (100 * top_k_entities_per_label[entity_label]) / entity_count_df[entity_label].sum()
                        ).reset_index().rename(columns={
                            'index': f'{entity_label}',
                            entity_label: f'pct_{entity_label}'
                        })
                    elif how == 'frac':
                        _col_pair = (
                                (top_k_entities_per_label[entity_label]) / entity_count_df[entity_label].sum()
                        ).reset_index().rename(columns={
                            'index': f'{entity_label}',
                            entity_label: f'frac_{entity_label}'
                        })
                    elif how == 'count':
                        _col_pair = top_k_entities_per_label[entity_label].reset_index().rename(columns={
                            'index': f'{entity_label}',
                            entity_label: f'count_{entity_label}',
                        })
                    else:
                        raise NotImplementedError(f'Unsupported: how="{how}"')
                    for col in _col_pair.columns:
                        top_k_entities_df[col] = _col_pair[col]
            top_k_entities_df: pd.DataFrame = pd.DataFrame(top_k_entities_df).reset_index().rename(
                columns=dict(index='top_k'))
            top_k_entities_df['top_k'] = top_k_entities_df['top_k'] + 1
            return top_k_entities_df, entity_labels


    ## Modified from https://github.com/HKUNLP/ProGen/blob/43b7d25437cd2e9945a2b2bfd056026ef0b0e9af/scripts/self_bleu.py
    class SelfBLEU(TabularMetric):
        aliases = ['Self-BLEU']

        class Params(TabularMetric.Params):
            class Config(TabularMetric.Params.Config):
                extra = Extra.allow

            num_cpus: int = 8
            num_gpus: int = 0
            batch_size: int = 50
            settings: Dict = dict(
                spacy_tokenization_model='en_core_web_lg',
                ngrams=(1, 2, 3, 4, 5),
            )
            generations_col: str = GENERATED_TEXTS_COL

        def compute_only(self, data: TextGenerationsPredictionsBase) -> Dict[int, float]:
            if not isinstance(data, TextGenerationsPredictionsBase):
                raise ValueError(
                    f'Expected data to be a {NextTokens} or {TextGenerations} instance; '
                    f'found: {type_str(data)}'
                )
            scores: Dict[int, float] = self.calc_self_bleu(
                docs=data.data[self.params.generations_col].tolist(),
                **self.params.settings,
                **self.params.dict(exclude={'settings'}),
            )
            return scores

        def calc_self_bleu(
                self,
                docs: List[str],
                *,
                spacy_tokenization_model: str,
                ngrams: Tuple[int, ...],
                batch_size: int,
                **kwargs
        ) -> Dict[int, float]:
            ## Ensure at least 1 batch per process.
            num_docs: int = len(docs)
            max_workers: int = max(1, min(self.params.num_cpus, math.floor(num_docs / (batch_size * 1))))
            tokenized_docs: List[List[str]] = self.spacy_tokenize_docs(
                docs,
                spacy_tokenization_model=spacy_tokenization_model,
                max_workers=max_workers,
                batch_size=batch_size,
            )
            ## Create a process-pool to calculate Self-BLEU values:
            executor = dispatch_executor(
                parallelize=Parallelize.processes,
                max_workers=max_workers,
            )
            kwargs['parallelize'] = Parallelize.processes
            kwargs['executor'] = executor
            ngram_self_bleu_scores: Dict[int, float] = {}
            for n_gram in ngrams:
                if n_gram == 1:
                    weights = (1.0, 0, 0, 0)
                elif n_gram == 2:
                    weights = (0.5, 0.5, 0, 0)
                elif n_gram == 3:
                    weights = (1.0 / 3, 1.0 / 3, 1.0 / 3, 0)
                elif n_gram == 4:
                    weights = (0.25, 0.25, 0.25, 0.25)
                elif n_gram == 5:
                    weights = (0.2, 0.2, 0.2, 0.2, 0.2)
                else:
                    raise ValueError
                ngram_self_bleu_scores[n_gram]: float = self.self_bleu_n_gram(
                    n_gram=n_gram,
                    weights=weights,
                    tokenized_docs=tokenized_docs,
                    num_docs=num_docs,
                    batch_size=batch_size,
                    **kwargs
                )
            return ngram_self_bleu_scores

        @classmethod
        def spacy_tokenize_docs(
                cls,
                docs: List[str],
                *,
                spacy_tokenization_model: str,
                max_workers: int,
                batch_size: int,
        ) -> List[List[str]]:
            with ignore_warnings_and_stdout():
                nlp: Language = spacy.load(spacy_tokenization_model, disable=['parser', 'tagger', 'ner'])
                tokenized_docs: List[List[str]] = []
                for sent_doc in nlp.pipe(docs, n_process=max_workers, batch_size=batch_size):
                    toks: List[str] = []
                    for tok in sent_doc:
                        toks.append(tok.text)
                    tokenized_docs.append(toks)
                return tokenized_docs

        @classmethod
        def self_bleu_n_gram(
                cls,
                *,
                weights: Tuple[float, ...],
                tokenized_docs: Union[List[List[str]], ray.ObjectRef],
                num_docs: int,
                batch_size: int,
                **kwargs,
        ) -> float:
            futures: List = []
            for idx_batch in iter_batches(num_docs, batch_size):
                futures.append(dispatch(
                    cls.bleu_i_batch,
                    weights=weights,
                    tokenized_docs=tokenized_docs,
                    idx_batch=idx_batch,
                    **kwargs,
                ))
            # for i in range(0, len(docs), batch_size):
            #     futures.append(run_parallel(
            #         cls.bleu_i_batch,
            #         weights=weights,
            #         tokenized_docs=tokenized_docs,
            #         all_i=list(range(i, min(i + batch_size, len(docs)))),
            #     ))
            n_gram_self_bleu_scores: List = []
            for n_gram_self_bleu_scores_batch in accumulate_iter(futures):
                n_gram_self_bleu_scores.extend(n_gram_self_bleu_scores_batch)
            return sum(n_gram_self_bleu_scores) / num_docs

        @classmethod
        def bleu_i_batch(
                cls,
                weights: Tuple[float, ...],
                tokenized_docs: Any,
                idx_batch: List[int],
                **kwargs
        ) -> List[float]:
            smoothing_function = SmoothingFunction().method1
            tokenized_docs: List[List[str]] = accumulate(tokenized_docs)
            return [
                cls.bleu_i(
                    weights=weights,
                    tokenized_docs=tokenized_docs,
                    smoothing_function=smoothing_function,
                    i=i,
                )
                for i in idx_batch
            ]

        @classmethod
        def bleu_i(
                cls,
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


def _text_gens_to_clf_dataset(
        data: TextGenerationsPredictionsBase,
        *,
        data_split: DataSplit,
        task: Task,
        label_col: str,
        text_col: str,
) -> ClassificationData:
    if not isinstance(data, TextGenerationsPredictionsBase):
        raise ValueError(
            f'Expected data to be a {NextTokens} or {TextGenerations} instance; '
            f'found: {type_str(data)}'
        )
    index_col: str = data.data_schema.index_col
    clf_dataset: ClassificationData = Dataset.of(
        task=task,
        data_split=data_split,
        data=data.data[[index_col, text_col, label_col]],
        data_schema={
            index_col: MLType.INDEX,
            text_col: MLType.TEXT,
            label_col: MLType.GROUND_TRUTH_LABEL,
        },
    )
    return clf_dataset


def _clf_dataset_to_next_tokens(
        data: ClassificationData,
        *,
        data_split: DataSplit,
        text_col: str,
        label_col: Optional[str] = None,
) -> NextTokens:
    if not isinstance(data, ClassificationData):
        raise ValueError(
            f'Expected data to be a {ClassificationData} instance; '
            f'found: {type_str(data)}'
        )
    index_col: str = data.data_schema.index_col
    label_col: str = get_default(label_col, data.ground_truth_label_col_name)

    text_gens: NextTokens = Predictions.of(
        task=Task.NEXT_TOKEN_PREDICTION,
        data_split=data_split,
        data=data.data.pandas().reset_index(drop=True),
        data_schema={
            index_col: MLType.INDEX,
            text_col: MLType.TEXT,
            label_col: MLType.CATEGORICAL,
        },
    )
    return text_gens


def _clf_dataset_to_text_gens(
        data: ClassificationData,
        *,
        data_split: DataSplit,
        text_col: str,
        label_col: Optional[str] = None,
) -> TextGenerations:
    if not isinstance(data, ClassificationData):
        raise ValueError(
            f'Expected data to be a {ClassificationData} instance; '
            f'found: {type_str(data)}'
        )
    index_col: str = data.data_schema.index_col
    label_col: str = get_default(label_col, data.ground_truth_label_col_name)
    data_df: pd.DataFrame = data.data.pandas().reset_index(drop=True)
    data_df[GENERATED_TEXTS_COL] = data_df[text_col]

    text_gens: TextGenerations = Predictions.of(
        task=Task.IN_CONTEXT_LEARNING,
        data_split=data_split,
        data=data_df,
        data_schema={
            index_col: MLType.INDEX,
            GENERATED_TEXTS_COL: MLType.TEXT,
            label_col: MLType.CATEGORICAL,
        },
    )
    return text_gens


class LabelPreservation(Metric):
    aliases = ['classification-label-preservation', 'label-preservation-metrics']

    class Params(Metric.Params):
        evaluator_params: Dict
        metrics: Optional[List[Metric]] = None
        batch_size: Optional[int] = None
        submission_batch_size: Optional[int] = None
        label_col: str
        text_col: str
        max_retries: int = 1
        verbosity: int = 0

        @root_validator(pre=True)
        def _set_metric_params(cls, params: Dict) -> Dict:
            Alias.set_metrics(params)
            if params.get('metrics') is not None:
                params['metrics'] = [
                    Metric.of(metric).clear() for metric in as_list(params['metrics'])
                ]
            return params

    def compute_only(self, data: Union[TextGenerationsPredictionsBase, ClassificationData]) -> Dict:
        if not isinstance(data, (TextGenerationsPredictionsBase, ClassificationData)):
            raise ValueError(
                f'Expected data to be a {NextTokens}, {TextGenerations} or {ClassificationData} instance; '
                f'found: {type_str(data)}'
            )
        if isinstance(data, TextGenerationsPredictionsBase):
            task: TaskOrStr = self.params.evaluator_params.get('task', None)
        else:
            task: TaskOrStr = get_default(data.task, self.params.evaluator_params.get('task', None))
        if task is None:
            raise ValueError('Must pass task in `data`, or Evaluator params.')
        task: Task = Task.from_str(task)
        if task not in {
            Task.BINARY_CLASSIFICATION,
            Task.MULTI_CLASS_CLASSIFICATION,
            Task.MULTI_LABEL_CLASSIFICATION,
        }:
            raise ValueError(f'Task must be a classification task; found: {task}')
        evaluator: Evaluator = Evaluator.of(
            **{
                **self.params.evaluator_params,
                **dict(verbosity=self.params.verbosity, task=task)
            }
        )
        try:
            if isinstance(data, TextGenerationsPredictionsBase):
                clf_dataset: ClassificationData = _text_gens_to_clf_dataset(
                    data,
                    data_split=data.data_split,
                    task=task,
                    label_col=self.params.label_col,
                    text_col=self.params.text_col,
                )
            else:
                clf_dataset: ClassificationData = data
            assert isinstance(clf_dataset, ClassificationData)
            clf_dataset: ClassificationData = clf_dataset.read()
            if self.params.metrics is None:
                clf_preds = evaluator.evaluate(
                    clf_dataset,
                    batch_size=self.params.batch_size,
                    submission_batch_size=self.params.submission_batch_size,
                    preds=True,
                    tracker=False,
                    load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN,
                )
                if len(clf_preds) != len(clf_dataset):
                    raise ValueError(
                        f'Number of predictions does not match number of inputs: '
                        f'Inputs={len(clf_dataset)}, predictions={len(clf_preds)}.'
                    )
                return {
                    'predictions': clf_preds,
                }
            else:
                clf_preds, clf_metrics = evaluator.evaluate(
                    clf_dataset,
                    batch_size=self.params.batch_size,
                    submission_batch_size=self.params.submission_batch_size,
                    metrics=self.params.metrics,
                    preds=True,
                    tracker=False,
                    load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN,
                )
                if len(clf_preds) != len(clf_dataset):
                    raise ValueError(
                        f'Number of predictions does not match number of inputs: '
                        f'Inputs={len(clf_dataset)}, predictions={len(clf_preds)}.'
                    )
                return {
                    'predictions': clf_preds,
                    'metrics': clf_metrics,
                }
        finally:
            evaluator.stop()
            clear_device_cache()


class TextGenerationStudent(Metric):
    aliases = ['text-gen-student', 'text-generation-student-metrics', 'text-gen-student-metrics']

    class Params(Metric.Params):
        test_dataset: Dataset
        train_text_col: str = GENERATED_TEXTS_COL
        hpo: bool = False
        metrics: Metrics
        algorithm: str
        hyperparams: Dict
        search_algorithm: Optional[Literal['random', 'grid']] = None
        search_space: Optional[Dict] = None
        k_fold: Optional[conint(ge=2)] = None
        validation_dataset: Optional[Dataset] = None
        val_frac: Optional[confloat(gt=0.0, lt=1.0)] = None
        split_seed: int = 42
        objective_metric: Optional[Union[Metric, Dict, str]] = None
        objective_type: Literal['maximize', 'minimize'] = 'maximize'
        resources_per_model: Dict[Literal['cpu', 'gpu'], Union[confloat(ge=0.0, lt=1.0), conint(ge=0)]]
        tune_num_models: Optional[conint(ge=1)] = None
        test_num_models: conint(ge=1)
        max_parallel_models: int = 0  ## 0 = no limit
        eval_steps: Optional[int] = None
        eval_batch_size: int = 16
        verbosity: int = 0
        save_to: Optional[FileMetadata] = None

        @root_validator(pre=True)
        def _set_metric_params(cls, params: Dict) -> Dict:
            Alias.set_metrics(params)
            params['metrics']: Metrics = Metrics.of(params['metrics'])
            params['test_dataset'] = Dataset.of(params['test_dataset'])

            if params['hpo'] is True:
                if params.get('search_algorithm') is None:
                    raise ValueError(f'Expected `search_algorithm` to be non-None when hpo is enabled.')
                if params.get('search_space') is None:
                    raise ValueError(f'Expected `search_space` to be non-None when hpo is enabled.')
                if params.get('objective_metric') is None:
                    raise ValueError(f'Expected `objective_metric` to be non-None when hpo is enabled.')
                if params.get('tune_num_models') is None:
                    raise ValueError(f'Expected `tune_num_models` to be non-None when hpo is enabled.')
                params['objective_metric']: Metric = Metric.of(params['objective_metric'])

                set_param_from_alias(params, param='k_fold', alias=['kfold', 'num_folds'])
                set_param_from_alias(params, param='val_frac', alias=['validation_frac', 'val_split'])
                set_param_from_alias(params, param='validation_dataset', alias=['val_dataset', 'eval_dataset'])
                if all_are_none(
                        params.get('k_fold'),
                        params.get('val_frac'),
                        params.get('validation_dataset'),
                ):
                    raise ValueError(
                        f'Exactly one of `k_fold`, `val_frac` or `validation_dataset` must be not-None; '
                        f'all are None.'
                    )
                if multiple_are_not_none(
                        params.get('k_fold'),
                        params.get('val_frac'),
                        params.get('validation_dataset'),
                ):
                    raise ValueError(
                        f'Exactly one of `k_fold`, `val_frac` or `validation_dataset` must be not-None; '
                        f'more than one are not-None '
                        f'('
                        f'k_fold={params.get("k_fold")}, '
                        f'val_frac={params.get("val_frac")}, '
                        f'validation_dataset={params.get("validation_dataset")}'
                        f')'
                    )

                if params.get('validation_dataset') is not None:
                    params['validation_dataset'] = Dataset.of(params['validation_dataset'])
            return params

    def compute_only(self, data: Union[TextGenerationsPredictionsBase, ClassificationData]) -> Any:
        test_dataset: Dataset = self.params.test_dataset
        task: TaskOrStr = test_dataset.task
        if isinstance(data, ClassificationData):
            train_dataset: ClassificationData = data.update_params(
                data_split=DataSplit.TRAIN,
            )
            if train_dataset.task != test_dataset.task:
                raise ValueError(
                    f'Is passing train dataset, expected task to be same as test dataset; '
                    f'found: test_dataset.task={test_dataset.task}, train_dataset.task={train_dataset.task}'
                )
        else:
            if not isinstance(data, TextGenerationsPredictionsBase):
                raise ValueError(
                    f'Expected data to be a {NextTokens} or {TextGenerations} instance; '
                    f'found: {type_str(data)}'
                )
            if task in {
                Task.BINARY_CLASSIFICATION,
                Task.MULTI_CLASS_CLASSIFICATION,
                Task.MULTI_LABEL_CLASSIFICATION,
            }:
                train_dataset: ClassificationData = _text_gens_to_clf_dataset(
                    data,
                    data_split=DataSplit.TRAIN,
                    task=task,
                    label_col=test_dataset.ground_truth_label_col_name,
                    text_col=self.params.train_text_col,
                )
            else:
                raise NotImplementedError(f'Not sure how to convert {type_str(data)} into a {Dataset.class_name}.')

        validation_dataset: Optional[Dataset] = self.params.validation_dataset
        if validation_dataset is not None:
            if validation_dataset.task != task:
                raise ValueError(
                    f'Test dataset task must match that of trainer params; found: '
                    f'validation data task={validation_dataset.task}, trainer task: {task}'
                )

        if self.params.hpo:
            ray_trainer_params: Dict = dict(
                trainer='ray',
                task=task,
                algorithm=self.params.algorithm,
                hyperparams=self.params.hyperparams,
                search_algorithm=self.params.search_algorithm,
                search_space=self.params.search_space,
                resources_per_model=self.params.resources_per_model,
                num_models=self.params.tune_num_models,
                eval_steps=self.params.eval_steps,
                objective_metric=self.params.objective_metric,
                objective_type=self.params.objective_type,
                objective_dataset=DataSplit.VALIDATION,
                eval_batch_size=self.params.eval_batch_size,
                max_parallel_models=self.params.max_parallel_models,
                retrain_final_model=True,
                num_final_models=self.params.test_num_models,

                model_failure_retries=1,
                final_model_failure_behavior='error',
                tune_failure_retries=1,
                tune_failure_retry_wait=60 * 10,

                verbosity=self.params.verbosity,
            )
            if self.params.k_fold is not None:
                ray_trainer_params: Dict = {
                    **ray_trainer_params,
                    **dict(
                        k_fold=self.params.k_fold,
                    ),
                }
                validation_dataset: Optional[Dataset] = None
            elif validation_dataset is None and self.params.val_frac is not None:
                train_dataset = train_dataset.read(read_as=DataLayout.PANDAS).to_layout(DataLayout.PANDAS)
                index_col: str = train_dataset.data_schema.index_col
                validation_dataset: Dataset = train_dataset.update_params(
                    data=train_dataset.data.sample(
                        frac=self.params.val_frac,
                        random_state=self.params.split_seed,
                    ).reset_index(drop=True)
                )
                validation_idxs: List[str] = validation_dataset.data[index_col].tolist()
                train_dataset: Dataset = train_dataset.update_params(
                    data=train_dataset.data.query(f'{index_col} not in {validation_idxs}').reset_index(drop=True)
                )
        else:
            ## Only train final models
            ray_trainer_params: Dict = dict(
                trainer='ray',
                task=task,
                algorithm=self.params.algorithm,
                hyperparams=self.params.hyperparams,
                resources_per_model=self.params.resources_per_model,
                num_models=self.params.test_num_models,
                eval_steps=self.params.eval_steps,
                eval_batch_size=self.params.eval_batch_size,
                max_parallel_models=self.params.max_parallel_models,
                model_failure_retries=1,
                final_model_failure_behavior='error',
                verbosity=self.params.verbosity,
            )
        tune_trainer: RayTuneTrainer = Trainer.of(**ray_trainer_params)
        final_model_results, tune_results = tune_trainer.train(
            datasets=Datasets.of(
                train=train_dataset,
                validation=validation_dataset,
                test=test_dataset,
            ),
            metrics=self.params.metrics,
            save_model=self.params.save_to,
        )
        assert isinstance(final_model_results, tune.ResultGrid)
        if final_model_results.num_errors > 0:
            msg = f'\n{final_model_results.num_errors} model failures were encountered during training final models:'
            msg += '\n\n'.join([
                f'Error#{err_i + 1}:\n{format_exception_msg(err)}'
                for err_i, err in enumerate(final_model_results.errors)
            ]) + '\n'
            raise RayTuneTrainerFinalModelsError(msg)
        trialwise_final_model_metrics: Dict[str, Metrics] = tune_trainer.get_trialwise_final_model_metrics(
            final_model_results=final_model_results,
            metrics=self.params.metrics,
        )
        detailed_final_model_metrics: pd.DataFrame = tune_trainer.get_detailed_metrics(final_model_results)
        if self.params.hpo:
            assert isinstance(tune_results, tune.ResultGrid)
            tune_metrics: pd.DataFrame = tune_trainer.get_detailed_metrics(tune_results)
            return trialwise_final_model_metrics, detailed_final_model_metrics, tune_metrics
        return trialwise_final_model_metrics, detailed_final_model_metrics

    def test_stats(self) -> Dict[str, Dict[str, Union[int, float]]]:
        return self.metric_stats(DataSplit.TEST)

    def train_stats(self) -> Dict[str, Dict[str, Union[int, float]]]:
        return self.metric_stats(DataSplit.TRAIN)

    def validation_stats(self) -> Dict[str, Dict[str, Union[int, float]]]:
        return self.metric_stats(DataSplit.VALIDATION)

    def metric_stats(self, data_split: DataSplit) -> Dict[str, Dict[str, Union[int, float]]]:
        data_split: DataSplit = DataSplit(data_split)
        if self.params.hpo:
            trialwise_final_model_metrics, detailed_final_model_metrics, tune_metrics = self.value
        else:
            trialwise_final_model_metrics, detailed_final_model_metrics = self.value
        return _ray_agg_final_model_metric_stats(trialwise_final_model_metrics, data_split=data_split)


with optional_dependency('sentence_transformers'):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from synthergent.base.framework.task.dense_retrieval import _normalize_l2


    class LabelwiseCosineSimilarity(TabularMetric):

        class Params(TabularMetric.Params):
            label_col: str
            num_cpus: int = 8
            num_gpus: int = 1
            generations_col: str = GENERATED_TEXTS_COL
            embeddings_col: str = 'embeddings'
            hf_embedding_model_name: str = 'all-mpnet-base-v2'

        def compute_only(self, data: TextGenerationsPredictionsBase) -> pd.DataFrame:
            if not isinstance(data, TextGenerationsPredictionsBase):
                raise ValueError(
                    f'Expected data to be a {NextTokens} or {TextGenerations} instance; '
                    f'found: {type_str(data)}'
                )
            labelwise_cosine_sims: pd.DataFrame = self.calc_labelwise_cosine_sims(
                data.data.pandas()[[self.params.generations_col, self.params.label_col]],
                generations_col=self.params.generations_col,
                label_col=self.params.label_col,
                embeddings_col=self.params.embeddings_col,
                hf_embedding_model_name=self.params.hf_embedding_model_name,
            )
            return labelwise_cosine_sims

        @classmethod
        def calc_labelwise_cosine_sims(
                cls,
                df: pd.DataFrame,
                *,
                generations_col: str,
                label_col: str,
                embeddings_col: str,
                hf_embedding_model_name: str,
        ) -> pd.DataFrame:
            encoder = SentenceTransformer(hf_embedding_model_name)
            try:
                df[embeddings_col] = list(
                    encoder.encode(df[generations_col].apply(lambda x: str(x) if x is not None else '').to_list())
                )
                # print(df[embeddings_col])
                labelspace: List[str] = sorted(list(df[label_col].unique()))
                labelwise_cosine_sims: List[Dict] = []
                for row_idx, lb_row in enumerate(labelspace):  ## Each row
                    labelwise_cosine_sims.append({})
                    for col_idx, lb_col in enumerate(labelspace):  ## Each cell (column) in each row
                        labelwise_cosine_sims[-1][lb_col] = cls.get_cosine_sim_for_label_i_and_j(
                            embeddings_lb_i=df.query(f'{label_col} == "{lb_row}"')[embeddings_col].to_list(),
                            embeddings_lb_j=df.query(f'{label_col} == "{lb_col}"')[embeddings_col].to_list(),
                        )
                labelwise_cosine_sims: pd.DataFrame = pd.DataFrame(
                    labelwise_cosine_sims,
                    index=labelspace,
                )[labelspace]
                return labelwise_cosine_sims
            finally:
                del encoder
                clear_device_cache()

        @classmethod
        def get_cosine_sim_for_label_i_and_j(
                cls,
                embeddings_lb_i: np.ndarray,
                embeddings_lb_j: np.ndarray,
        ) -> np.float64:
            # print(len(embeddings_lb_i))
            # print(embeddings_lb_i[:3])
            return cosine_similarity(
                embeddings_lb_i,
                embeddings_lb_j,
            ).sum() / (len(embeddings_lb_i) * len(embeddings_lb_j))


    class PairwiseCosineSimilarity(TabularMetric):

        class Params(TabularMetric.Params):
            num_cpus: int = 8
            num_gpus: int = 1
            generations_col: str = GENERATED_TEXTS_COL
            embeddings_col: str = 'embeddings'
            hf_embedding_model_name: str = 'all-mpnet-base-v2'

        def compute_only(self, data: TextGenerationsPredictionsBase) -> pd.DataFrame:
            if not isinstance(data, TextGenerationsPredictionsBase):
                raise ValueError(
                    f'Expected data to be a {NextTokens} or {TextGenerations} instance; '
                    f'found: {type_str(data)}'
                )

            pairwise_cosine_sims = self.calc_pairwise_cosine_sims(
                data.data.pandas(),
                index_col=data.data_schema.index_col,
                generations_col=self.params.generations_col,
                embeddings_col=self.params.embeddings_col,
                hf_embedding_model_name=self.params.hf_embedding_model_name,
            )
            return pairwise_cosine_sims

        @classmethod
        def calc_pairwise_cosine_sims(
                cls,
                df: pd.DataFrame,
                *,
                index_col: str,
                generations_col: str,
                embeddings_col: str,
                hf_embedding_model_name: str,
        ) -> pd.DataFrame:
            encoder = SentenceTransformer(hf_embedding_model_name)
            try:
                num_rows: int = len(df)
                embeddings: np.ndarray = encoder.encode(
                    df[generations_col].apply(lambda x: str(x) if x is not None else '').to_list()
                )
                embeddings_norm: np.ndarray = _normalize_l2(embeddings)
                pairwise_cosine_sims_np: np.ndarray = embeddings_norm.dot(embeddings_norm.T)
                assert pairwise_cosine_sims_np.shape == (num_rows, num_rows)

                pairwise_cosine_sims: List[Dict] = []
                for i, idx_i in enumerate(df[index_col]):
                    pairwise_cosine_sims.append({
                        index_col: idx_i,
                        'cosine_sims': {},
                    })
                    for j, (idx_j, cosine_sim) in enumerate(zip(df[index_col], pairwise_cosine_sims_np[i, :])):
                        pairwise_cosine_sims[-1]['cosine_sims'][idx_j] = float(cosine_sim)
                pairwise_cosine_sims: pd.DataFrame = pd.DataFrame(
                    pairwise_cosine_sims,
                )
                return pairwise_cosine_sims
            finally:
                del encoder
                clear_device_cache()

        def to_labelwise_cosine_sims(
                self,
                pairwise_cosine_sims: pd.DataFrame,
                *,
                agg: Optional[AggregationStrategy],
                get_lb: Callable,
                index_col: str,
        ) -> pd.DataFrame:
            # get_lb = lambda idx: re.search(r'label=([^#-]+)', idx).group(1)
            if agg is not None:
                agg: AggregationStrategy = AggregationStrategy(agg)
            labelspace: List[str] = sorted(list(pairwise_cosine_sims[index_col].apply(get_lb).unique()))
            assert len(labelspace) > 1
            labelsiwise_cosine_sims = {}
            for idx_i, cosine_sims in zip(pairwise_cosine_sims[index_col], pairwise_cosine_sims['cosine_sims']):
                labelsiwise_cosine_sims.setdefault(get_lb(idx_i), {})
                for idx_j, cosine_sim in cosine_sims.items():
                    labelsiwise_cosine_sims[get_lb(idx_i)].setdefault(get_lb(idx_j), [])
                    labelsiwise_cosine_sims[get_lb(idx_i)][get_lb(idx_j)].append(round(float(cosine_sim), 6))
            labelsiwise_cosine_sims_agg: Dict[str, Dict[str, Union[float, List[float]]]] = {}
            for lb_i, d in labelsiwise_cosine_sims.items():
                assert isinstance(lb_i, str)
                labelsiwise_cosine_sims_agg.setdefault(lb_i, {})
                for lb_j, cosine_sims_list in labelsiwise_cosine_sims[lb_i].items():
                    assert isinstance(lb_j, str)
                    labelsiwise_cosine_sims_agg[lb_i][lb_j] = self._aggregate(cosine_sims_list, agg=agg)
            return pd.DataFrame(labelsiwise_cosine_sims_agg)[labelspace]

        @classmethod
        def _aggregate(
                cls,
                vals: List[float],
                *,
                agg: Optional[AggregationStrategy],
        ) -> Union[float, List[float]]:
            if agg is None:
                return vals
            elif agg is AggregationStrategy.AVERAGE:
                return float(np.mean(vals))
            elif agg is AggregationStrategy.MIN:
                return float(np.min(vals))
            elif agg is AggregationStrategy.MAX:
                return float(np.max(vals))
            elif agg is AggregationStrategy.MEDIAN:
                return float(np.median(vals))
            raise NotImplementedError(f'Cannot aggregate metrics using {agg}')

        @classmethod
        def plot_labelwise_cosine_sims_kde(
                cls,
                labelwise_cosine_sims_df: pd.DataFrame,
                *,
                return_plots: bool = False,
        ):
            labelspace: List[str] = sorted(list(labelwise_cosine_sims_df.columns))
            plots = []
            for i, lb_i in enumerate(labelspace):
                for j, lb_j in enumerate(labelspace):
                    plots.append(
                        pd.Series(labelwise_cosine_sims_df.iloc[i, j]).hvplot.kde().opts(
                            width=150,
                            height=150,
                            title=f'cosine_sim({lb_i}, {lb_j})',
                            fontsize={'title': 8}
                        ))
            if return_plots:
                return plots
            return plotsum(plots, how='grid').cols(len(labelspace))
