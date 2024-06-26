from typing import *
from abc import ABC, abstractmethod
import time, glob, os, sys, boto3, numpy as np, pandas as pd, json, requests, gc, warnings, random
from pandas.core.groupby import DataFrameGroupBy as PandasDataFrameGroupBy
from math import inf, exp, log
from copy import deepcopy
from synthesizrr.base.util import is_list_like, set_param_from_alias, MappedParameters, Parameters, get_default, as_list, \
    is_dict_like, as_set, is_even, type_str, ignore_warnings, str_format_args, StringUtil, format_exception_msg
from synthesizrr.base.constants import Task, MLType, MLTypeSchema, DataLayout, DataSplit, Alias, FailureAction
from synthesizrr.base.data import ScalableDataFrame, ScalableSeries, ScalableSeriesRawType, ScalableDataFrameRawType, FileMetadata
from synthesizrr.base.framework import Algorithm, Dataset, Predictions
from synthesizrr.base.framework.task.classification import Classifier, ClassificationData, ClassificationPredictions, \
    MultiLabelClassifier, MultiLabelClassificationData
from synthesizrr.base.framework.task.retrieval import Retriever, Queries, RankedResults
from pydantic import root_validator, conint, confloat, constr, Extra
from pydantic.typing import Literal

PROMPT: str = 'prompt'
PROMPT_TEMPLATE: str = 'prompt_template'
ICL_TEMPLATE: str = 'icl_template'
ICL_EXAMPLES_TEMPLATE_KEY: str = 'icl_examples'
TEXT_PROMPT_COL: str = 'prompts'
PROMPT_TEMPLATE_ID_COL: str = 'prompt_template_id'
PROMPT_TEMPLATE_EXPANDER_SEP: str = '-prompt_template='
PROMPT_TEMPLATE_INDEX_COL_PREFIX: str = 'original_'
GENERATED_TEXTS_COL: str = 'generations'
GENERATED_TOKEN_SCORES_COL_TEMPLATE: str = 'generated_token_{token_timestep_i}_scores'

TokenCombinationStrategy = Literal['sum', 'max', 'min', 'median']
TimestepCombinationStrategy = Literal['max', 'min', 'median']
ZeroShotStrategy = Literal['entailment']
ICLLabelSamplingStrategy = Literal['representative', 'balanced']
ICLEntailmentStrategy = Literal[
    'alternate',
    'positives_at_end', 'negatives_at_end',
    'positives_only', 'negatives_only',
]
ENTAILMENT_YES: str = 'Yes'
ENTAILMENT_NO: str = 'No'

ENTAILMENT_LOGIT_COL: str = 'entailment_logit'
CONTRADICTION_LOGIT_COL: str = 'contradiction_logit'
ENTAILMENT_PROBABILITY_COL: str = 'entailment_prob'
CONTRADICTION_PROBABILITY_COL: str = 'contradiction_prob'
ENTAILMENT_PREDICTION_COL: str = 'entailment_prediction'
ICL_EXAMPLE_TEMPLATE_KEY: str = 'icl'
ICL_EXAMPLE_DATASET_INDEX_TEMPLATE_KEY: str = 'icl_example_dataset_index'


def stable_softmax(*x):  ## Ref: https://stackoverflow.com/a/49212689
    if is_list_like(x) and len(x) == 1:
        x: Union[List, Tuple, np.ndarray] = x[0]
    x: np.ndarray = np.array(x)
    z: np.ndarray = x - np.max(x)
    numerator: np.ndarray = np.exp(z)
    denominator: np.float_ = np.sum(numerator)
    softmax: np.ndarray = numerator / denominator
    return softmax


class ICLSampler(Parameters):
    icl_dataset: Dataset
    prompt_template: str
    icl_template: str
    icl_filter_col: Optional[constr(min_length=1)]
    num_shots: conint(ge=0) = 0
    shots_sep: constr(min_length=0) = '\n\n'

    class Config(Parameters.Config):
        extra = Extra.ignore

    @root_validator(pre=True)
    def set_icl_sampler_params(cls, params: Dict) -> Dict:
        set_param_from_alias(params, param='icl_dataset', alias=[
            'dataset', 'examples', 'icl_examples', 'demonstrations',
        ])
        set_param_from_alias(params, param='num_shots', alias=['icl_num_shots', 'n_shots', 'shots', 'icl_shots'])
        set_param_from_alias(params, param='shots_sep', alias=['icl_shots_sep', 'shot_sep'])
        icl_dataset: Dataset = params['icl_dataset'].copy()
        icl_dataset.data = icl_dataset.data.as_layout(DataLayout.PANDAS)
        params['icl_dataset']: Dataset = icl_dataset
        return params

    @staticmethod
    def shot_idxs(num_shots: int) -> List[int]:
        return list(range(1, num_shots + 1))  ## Count as 1, 2, 3, ....

    @staticmethod
    def _icl_example_col_name(shot_i: int) -> str:
        return f'{ICL_EXAMPLE_TEMPLATE_KEY}_{shot_i}'

    @staticmethod
    def _icl_example_dataset_index_col_name(shot_i: int) -> str:
        return f'{ICL_EXAMPLE_DATASET_INDEX_TEMPLATE_KEY}_{shot_i}'

    @property
    def icl_expanded_prompt_template(self) -> str:
        """The prompt template, prepended with the ICL template N times (N=num_shots)."""
        if self.num_shots == 0:
            num_shots_icl_template: str = ''
        else:
            num_shots_icl_template: str = self.shots_sep.removesuffix('\n')
        for shot_idx in self.shot_idxs(self.num_shots):
            ## E.g. icl[item_name] becomes icl_1[item_name]
            num_shots_icl_template += self.icl_template.replace(
                f'{ICL_EXAMPLE_TEMPLATE_KEY}[',
                f'{ICL_EXAMPLE_TEMPLATE_KEY}_{shot_idx}['
            )
            if shot_idx != self.num_shots:
                num_shots_icl_template += self.shots_sep
            else:
                num_shots_icl_template += self.shots_sep.removesuffix('\n')
        # print(f'num_shots_icl_template: """\n{num_shots_icl_template}"""\n')
        prompt_template: str = self.prompt_template
        # print(f'Original prompt template: """\n{prompt_template}"""\n')
        prompt_template: str = to_second_level_prompt_template(prompt_template, exclude=ICL_EXAMPLES_TEMPLATE_KEY)
        # print(f'Second-level prompt template: """\n{prompt_template}"""\n')
        prompt_template: str = prompt_template.format(**{ICL_EXAMPLES_TEMPLATE_KEY: num_shots_icl_template})
        # print(f'ICL-Expanded prompt template: """\n{prompt_template}"""\n')
        return prompt_template

    def append_icl_examples(self, batch: Dataset) -> Dataset:
        """
        In-Context Learning examples are added as new columns to the batch of data, with column-names having syntax
        "icl_1", "icl_2", etc.
        In Python, string formatting can work with nested dicts, so each element of the column "icl_2"
        will be a dict, with keys as the columns of icl_dataset, e.g. "asin", "item_name", "brand", etc.
        See https://stackoverflow.com/a/49070458/4900327 for an example on how str.format() works with nested dicts.
        Thus, in the ICL template, a parameterized string like `{icl[item_name]}` will get expanded to
        `{icl_1[item_name]}`, `{icl_2[item_name]}`, etc. which will be populated with actual values.
        """
        # print('Appending ICL examples...')
        if self.icl_dataset is None:
            return batch

        ## We want to avoid accidentally "leaking" the answer by selecting an ICL example with the same idx as the
        ## current example in the batch.
        batch_icl_idxs_in_use: List[Set[str]] = [as_set(batch_idx) for batch_idx in batch.index()]
        assert len(batch_icl_idxs_in_use) == len(batch)
        # print(f'Columns before: {batch.data.columns}')
        batch_filter_col_vals: Optional[List] = None
        if self.icl_filter_col is not None:
            batch_filter_col_vals: List = [val for val in batch.data[self.icl_filter_col]]
        for shot_i in self.shot_idxs(self.num_shots):  ## 1, 2, 3, ...
            # print(f'shot_i: {shot_i}')
            ## E.g. batch.data['icl_1'] = [{...}, ..., {...}]
            batch.data[self._icl_example_col_name(shot_i)]: List[Dict] = self._select_shot_i_icl_examples(
                icl_dataset=self.icl_dataset,
                batch_icl_idxs_in_use=batch_icl_idxs_in_use,
                icl_filter_col=self.icl_filter_col,
                batch_filter_col_vals=batch_filter_col_vals,
            )
            batch.data_schema.features_schema[self._icl_example_col_name(shot_i)] = MLType.OBJECT
        # print(f'Columns after: {batch.data.columns}')
        return batch

    @classmethod
    def _select_shot_i_icl_examples(
            cls,
            icl_dataset: Dataset,
            batch_icl_idxs_in_use: List[Set],  ## ICL Idxs already used, for each row in the batch.
            balance_on: Optional[str] = None,
            icl_filter_col: Optional[str] = None,
            batch_filter_col_vals: Optional[List] = None,
    ) -> List[Dict]:
        icl_df: pd.DataFrame = icl_dataset.data.pandas()
        shot_i_icl_examples: List[Dict] = []
        for batch_i in range(len(batch_icl_idxs_in_use)):
            ## Remove ICL examples which already used in previous shots for this example in the batch:
            icl_df_filtered: pd.DataFrame = icl_df[
                ~icl_df[icl_dataset.data_schema.index_col].isin(batch_icl_idxs_in_use[batch_i])
            ]
            if icl_filter_col is not None:
                assert batch_filter_col_vals is not None
                batch_filter_col_val: Any = batch_filter_col_vals[batch_i]
                icl_df_filtered: pd.DataFrame = icl_df_filtered[icl_df_filtered[icl_filter_col] == batch_filter_col_val]

            try:
                if balance_on is not None:
                    ## Select a random row for each unique value of the column, then selects one of those random rows.
                    icl_example: Dict = icl_df_filtered \
                        .sample(frac=1).drop_duplicates(subset=[balance_on]) \
                        .sample(frac=1).iloc[0].to_dict()
                else:
                    ## Select random row from filtered ICL dataset, convert it to dict:
                    icl_example: Dict = icl_df_filtered.sample(frac=1).iloc[0].to_dict()
                shot_i_icl_examples.append(icl_example)
                ## Add the selected ICL example's index to the "in-use" set for the current example in the batch:
                batch_icl_idxs_in_use[batch_i].add(icl_example[icl_dataset.data_schema.index_col])
            except Exception as e:
                raise ValueError(
                    f'Error while selecting from filtered ICL set '
                    f'of {len(icl_df_filtered)} rows and columns: {icl_df_filtered.columns}:\n'
                    f'{format_exception_msg(e)}'
                )
        return shot_i_icl_examples


class ClassificationICLSampler(ICLSampler):
    strategy: ZeroShotStrategy
    label_sampling_strategy: ICLLabelSamplingStrategy = 'balanced'
    entailment_strategy: ICLEntailmentStrategy = 'alternate'
    entailment_positive_label: constr(min_length=1) = 'Yes'
    entailment_negative_label: constr(min_length=1) = 'No'
    icl_entailment_pos_dataset: Optional[ClassificationData] = None
    icl_entailment_neg_dataset: Optional[ClassificationData] = None
    label_normalizer: Callable[[Any], str]
    label_verbalizer: Dict[str, str]

    @classmethod
    def _icl_label_col_binarized_entailment(cls, icl_dataset: ClassificationData) -> str:
        return f'{icl_dataset.ground_truth_label_col_name}_binarized'

    @root_validator(pre=False)
    def set_classification_icl_sampler_params(cls, params: Dict) -> Dict:
        params['label_verbalizer']: Dict[str, str] = {
            params['label_normalizer'](lb): lb_description
            for lb, lb_description in params['label_verbalizer'].items()
        }

        icl_dataset: Dataset = params['icl_dataset']
        if not isinstance(icl_dataset, ClassificationData):
            raise ValueError(f'Only {ClassificationData} can be used as the In-Context Learning dataset.')

        ## Deepcopy to avoid the issue of overwriting the ground-truth column with the verbalized one.
        icl_dataset.data = ScalableDataFrame.of(icl_dataset.data.pandas().copy(deep=True))
        icl_gt_lb_col: str = icl_dataset.ground_truth_label_col_name
        icl_dataset.data[icl_gt_lb_col] = icl_dataset.data[icl_gt_lb_col] \
            .apply(params['label_normalizer']) \
            .map(params['label_verbalizer'])
        params['icl_dataset']: ClassificationData = icl_dataset

        def set_col_val(df: pd.DataFrame, col: str, val: Any) -> pd.DataFrame:
            df.loc[:, col] = val
            return df

        if params['strategy'] == 'entailment':
            icl_entailment_pos_dfs: Dict[str, pd.DataFrame] = {}
            icl_entailment_neg_dfs: Dict[str, pd.DataFrame] = {}
            for normalized_lb in icl_dataset.data.pandas()[icl_gt_lb_col].unique():
                icl_entailment_pos_dfs[normalized_lb] = set_col_val(
                    icl_dataset.data.pandas().query(f'`{icl_gt_lb_col}` == "{normalized_lb}"').copy(deep=True),
                    col=cls._icl_label_col_binarized_entailment(icl_dataset),
                    val=params['entailment_positive_label'],
                )
                if len(icl_entailment_pos_dfs[normalized_lb]) == 0:
                    icl_entailment_pos_dfs.pop(normalized_lb)
                icl_entailment_neg_dfs[normalized_lb] = set_col_val(
                    icl_dataset.data.pandas().query(f'`{icl_gt_lb_col}` != "{normalized_lb}"').copy(deep=True),
                    col=cls._icl_label_col_binarized_entailment(icl_dataset),
                    val=params['entailment_negative_label'],
                )
                if len(icl_entailment_neg_dfs[normalized_lb]) == 0:
                    icl_entailment_neg_dfs.pop(normalized_lb)
            icl_entailment_pos: pd.DataFrame = pd.concat(list(icl_entailment_pos_dfs.values())).reset_index(drop=True)
            params['icl_entailment_pos_dataset']: ClassificationData = icl_dataset.update_params(
                data=icl_entailment_pos
            )
            icl_entailment_neg: pd.DataFrame = pd.concat(list(icl_entailment_neg_dfs.values())).reset_index(drop=True)
            params['icl_entailment_neg_dataset']: ClassificationData = icl_dataset.update_params(
                data=icl_entailment_neg
            )
        else:
            raise NotImplementedError(f'Unsupported classification strategy: {params["strategy"]}')
        return params

    def append_icl_examples(self, batch: ClassificationData) -> ClassificationData:
        if self.icl_dataset is None:
            return batch
        if not isinstance(self.icl_dataset, ClassificationData):
            raise ValueError(
                f'In-Context Learning must be an instance of {ClassificationData}; '
                f'found {type_str(self.icl_dataset)}'
            )
        if self.strategy == 'entailment':
            return self._append_icl_examples_entailment(batch=batch)
        raise NotImplementedError(
            f'Unsupported classification strategy for sampling In-Context Learning examples: "{self.strategy}"'
        )

    def _append_icl_examples_entailment(self, batch: ClassificationData) -> ClassificationData:
        """
        When using entailment, we want the ICL examples to be selected based on `entailment_strategy`.
        """
        assert isinstance(self.icl_dataset, ClassificationData)
        assert isinstance(self.icl_entailment_pos_dataset, ClassificationData)
        assert isinstance(self.icl_entailment_neg_dataset, ClassificationData)
        ## We want to avoid accidentally "leaking" the answer by selecting an ICL example with the same idx as the
        ## current example in the batch.
        batch_icl_idxs_in_use: List[Set] = [as_set(batch_idx) for batch_idx in batch.index()]
        assert len(batch_icl_idxs_in_use) == len(batch)
        for shot_i in self.shot_idxs(self.num_shots):  ## 1, 2, 3, ...
            if self.entailment_strategy == 'alternate':
                ## If num_shots is even (e.g. num_shots=4), add examples like (1)Neg (2)Pos (3)Neg (4)Pos
                ##  If num_shots is odd (e.g. num_shots=5), add examples like (1)Pos (2)Neg (3)Pos (4)Neg (5)Pos
                if is_even(shot_i + self.num_shots):
                    icl_dataset_entailment: ClassificationData = self.icl_entailment_pos_dataset
                else:
                    icl_dataset_entailment: ClassificationData = self.icl_entailment_neg_dataset
            elif self.entailment_strategy == 'positives_at_end':
                ## If num_shots is even (e.g. num_shots=4), add examples like (1)Neg (2)Neg (3)Pos (4)Pos
                ## If num_shots is even (e.g. num_shots=5), add examples like (1)Neg (2)Neg (3)Pos (4)Pos (5)Pos
                if shot_i > self.num_shots // 2:
                    icl_dataset_entailment: ClassificationData = self.icl_entailment_pos_dataset
                else:
                    icl_dataset_entailment: ClassificationData = self.icl_entailment_neg_dataset
            elif self.entailment_strategy == 'negatives_at_end':
                ## If num_shots is even (e.g. num_shots=4), add examples like (1)Pos (2)Pos (3)Neg (4)Neg
                ## If num_shots is even (e.g. num_shots=5), add examples like (1)Pos (2)Pos (3)Neg (4)Neg (5)Neg
                if shot_i > self.num_shots // 2:
                    icl_dataset_entailment: ClassificationData = self.icl_entailment_neg_dataset
                else:
                    icl_dataset_entailment: ClassificationData = self.icl_entailment_pos_dataset
            elif self.entailment_strategy == 'positives_only':
                icl_dataset_entailment: ClassificationData = self.icl_entailment_pos_dataset
            elif self.entailment_strategy == 'negatives_only':
                icl_dataset_entailment: ClassificationData = self.icl_entailment_neg_dataset
            else:
                raise NotImplementedError(f'Unsupported `entailment_strategy`: "{self.entailment_strategy}"')

            if self.label_sampling_strategy == 'representative':
                ## Already representative of the true label distribution
                shot_i_icl_examples: List[Dict] = self._select_shot_i_icl_examples(
                    icl_dataset=icl_dataset_entailment,
                    batch_icl_idxs_in_use=batch_icl_idxs_in_use,
                )
            elif self.label_sampling_strategy == 'balanced':
                ## Here we must balance between examples of various classes in the ICL data.
                shot_i_icl_examples: List[Dict] = self._select_shot_i_icl_examples(
                    icl_dataset=icl_dataset_entailment,
                    batch_icl_idxs_in_use=batch_icl_idxs_in_use,
                    balance_on=self.icl_dataset.ground_truth_label_col_name,
                )
            else:
                raise NotImplementedError(f'Unsupported `label_sampling_strategy`: "{self.label_sampling_strategy}"')
            batch.data[self._icl_example_col_name(shot_i)] = shot_i_icl_examples
            batch.data_schema.features_schema[self._icl_example_col_name(shot_i)] = MLType.OBJECT
        return batch


def _create_prompt(prompt_template: str, prompt_prefix: str, **data) -> str:
    try:
        prompt: str = prompt_template.format(**data)
        if not prompt.startswith(prompt_prefix):
            prompt: str = prompt_prefix + prompt
        return prompt
    except Exception as e:
        raise ValueError(
            f'Could not populate prompt with data:\n{data}\nprompt_template:"""\n{prompt_template}\n"""'
            f'\nError obtained:\n{format_exception_msg(e)}'
        )


def apply_prompt_template(batch: 'Prompts', prompt_prefix: str) -> 'Prompts':
    if is_dict_like(batch.prompt_template):
        if batch.prompt_template_apply == 'expand':
            error_msg: str = f'When passing prompt template as a dict and `prompt_template_apply`="expand", ' \
                             f'it is expected that each key will be added as a column `{PROMPT_TEMPLATE_ID_COL}`, ' \
                             f'and the corresponding template will be applied to each row. '
            prompt_templates: Dict[str, str] = batch.prompt_template
            new_batch_data: List[Dict] = []
            for data_d in batch.data.to_list_of_dict():
                for prompt_template_id, prompt_template in prompt_templates.items():
                    d: Dict = deepcopy(data_d)
                    d[TEXT_PROMPT_COL]: str = _create_prompt(prompt_template, prompt_prefix=prompt_prefix, **d)
                    d[PROMPT_TEMPLATE_ID_COL] = prompt_template_id
                    d[f'{PROMPT_TEMPLATE_INDEX_COL_PREFIX}{batch.data_schema.index_col}'] = d[
                        batch.data_schema.index_col]
                    d[batch.data_schema.index_col]: str = \
                        f'{d[batch.data_schema.index_col]}{PROMPT_TEMPLATE_EXPANDER_SEP}{prompt_template_id}'
                    new_batch_data.append(d)
        elif batch.prompt_template_apply == 'map':
            error_msg: str = f'When passing prompt template as a dict and `prompt_template_apply`="map", ' \
                             f'it is expected that each key maps different values taken by ' \
                             f'`{batch.prompt_template_map_col}` column, to different prompts. '
            prompt_templates: Dict[str, str] = batch.prompt_template
            new_batch_data: List[Dict] = []
            for d in batch.data.to_list_of_dict():
                prompt_template_map_val: Any = d.get(batch.prompt_template_map_col)
                if prompt_template_map_val is None:
                    raise ValueError(
                        error_msg + f'Found `None` value in column "{batch.prompt_template_map_col}", which cannot be '
                                    f'mapped to any prompt template.'
                    )
                prompt_template: Optional[str] = prompt_templates.get(prompt_template_map_val)
                if prompt_template is None:
                    raise ValueError(
                        error_msg + f'Found no corresponding template for value {d[batch.prompt_template_map_col]} in '
                                    f'column "{batch.prompt_template_map_col}".'
                    )
                d[TEXT_PROMPT_COL]: str = _create_prompt(prompt_template, prompt_prefix=prompt_prefix, **d)
                d[PROMPT_TEMPLATE_ID_COL] = prompt_template_map_val
                d[f'{PROMPT_TEMPLATE_INDEX_COL_PREFIX}{batch.data_schema.index_col}'] = d[batch.data_schema.index_col]
                d[batch.data_schema.index_col]: str = \
                    f'{d[batch.data_schema.index_col]}{PROMPT_TEMPLATE_EXPANDER_SEP}{prompt_template_map_val}'
                new_batch_data.append(d)
        else:
            raise NotImplementedError(
                f'Unsupported value for `prompt_template_apply` ("{batch.prompt_template_apply}") '
                f'when `prompt_template` is a dict.'
            )

    else:
        if batch.prompt_template_apply != 'expand':
            raise ValueError(
                f'When `prompt_template` is not a dict, `prompt_template_apply`="expand" must be passed; we will '
                f'either apply the prompt template to each row (if it is a string) or expand each row along the prompt '
                f'template (if it is list-like).'
            )
        prompt_templates: List[str] = as_list(batch.prompt_template)
        new_batch_data: List[Dict] = []
        for data_d in batch.data.to_list_of_dict():
            for prompt_template_i, prompt_template in enumerate(prompt_templates):
                d: Dict = deepcopy(data_d)
                prompt_template_i: int = prompt_template_i + 1
                d[TEXT_PROMPT_COL]: str = _create_prompt(prompt_template, prompt_prefix=prompt_prefix, **d)
                d[PROMPT_TEMPLATE_ID_COL]: int = prompt_template_i
                d[f'{PROMPT_TEMPLATE_INDEX_COL_PREFIX}{batch.data_schema.index_col}'] = d[batch.data_schema.index_col]
                if len(prompt_templates) > 1:
                    d[batch.data_schema.index_col]: str = \
                        f'{d[batch.data_schema.index_col]}{PROMPT_TEMPLATE_EXPANDER_SEP}{prompt_template_i}'
                new_batch_data.append(d)
    batch.data = ScalableDataFrame.of(new_batch_data)
    ## Remove all TEXT columns:
    for col, mltype in list(batch.data_schema.features_schema.items()):
        if mltype == MLType.TEXT:
            batch.data_schema.features_schema.pop(col)
    batch.data_schema.features_schema[TEXT_PROMPT_COL] = MLType.TEXT
    batch.data_schema.features_schema[PROMPT_TEMPLATE_ID_COL] = MLType.CATEGORICAL
    batch.data_schema.features_schema[
        f'{PROMPT_TEMPLATE_INDEX_COL_PREFIX}{batch.data_schema.index_col}'
    ] = MLType.CATEGORICAL
    return batch


def to_second_level_prompt_template(prompt_template: str, *, exclude: Optional[Union[Set[str], str]] = None) -> str:
    """Convert '{name} {age}' to '{{name}} {{age}}'. """
    exclude: Set[str] = as_set(get_default(exclude, []))
    template_args: Set[str] = set(str_format_args(prompt_template)) - exclude
    for template_arg in template_args:
        ## If prompt_template is like: '''Is "{query}" a search-query about {label}?''', convert it to:
        ## '''Is "{{query}}" a search-query about {label}?'''
        template_arg_str: str = '{' + str(template_arg) + '}'
        template_arg_str_extra: str = '{{' + str(template_arg) + '}}'
        if template_arg_str_extra not in prompt_template:
            prompt_template: str = prompt_template.replace(template_arg_str, template_arg_str_extra)
    return prompt_template


class Prompts(Dataset):
    _allow_empty_features_schema: ClassVar[bool] = True

    tasks = Task.NEXT_TOKEN_PREDICTION
    features_schema = {
        TEXT_PROMPT_COL: MLType.TEXT,
    }
    ground_truths_schema = {}  ## No ground-truths needed

    prompt_template: Union[List[str], Tuple[str, ...], Dict[str, str], str]
    prompt_template_map_col: Optional[str] = None
    prompt_template_apply: Optional[Literal['expand', 'map']] = None

    @root_validator(pre=True)
    def set_prompt_data_params(cls, params: Dict) -> Dict:
        set_param_from_alias(params, param='prompt_template', alias=['template', 'prompt', 'prompt_templates'])
        if params.get('prompt_template_apply') is None:
            if is_dict_like(params['prompt_template']):
                params['prompt_template_apply']: str = 'map'
            else:
                params['prompt_template_apply']: str = 'expand'
        if is_dict_like(params['prompt_template']) \
                and params['prompt_template_apply'] == 'map' \
                and not isinstance(params.get('prompt_template_map_col'), str):
            raise ValueError(
                f'When passing `prompt_template` as a dict and `prompt_template_apply`="map" (default), '
                f'you must also pass `prompt_template_map_col`, which should be the column you want to map '
                f'based on the prompts dict.'
            )
        return params

    def apply_template(self) -> 'Prompts':
        prompts: Prompts = apply_prompt_template(self.copy(), prompt_prefix='')
        prompts.prompt_template = '{' + TEXT_PROMPT_COL + '}'
        return prompts

    def prompts(self) -> ScalableSeries:
        return self.data[TEXT_PROMPT_COL]


class TextGenerationsPredictionsBase(Predictions, ABC):
    ground_truths_schema = {}  ## No ground-truths needed
    predictions_schema = {
        GENERATED_TEXTS_COL: MLType.TEXT,
        GENERATED_TOKEN_SCORES_COL_TEMPLATE: MLType.FLOAT,
    }

    def generations(self) -> ScalableSeries:
        return self.data[GENERATED_TEXTS_COL]


class NextTokens(TextGenerationsPredictionsBase):
    tasks = Task.NEXT_TOKEN_PREDICTION


PROMPTING_FORMAT_MSG: str = f"""
Prompting results returned by algorithm must be a column of text.
""".strip()

## Similar to transformers.generation.configuration_utils.GenerationConfig
GenerationOutputScoresFormat = Literal['probabilities', 'log-probabilities', 'logits']


class TextGenerationParams(Parameters, ABC):  ## Becomes an anonymous class later
    class Config(Parameters.Config):
        ## Allow extra keyword parameters to be used when initializing the reader.
        ## These will be forwarded to the respective reader method like .read_csv, .read_json, etc.
        extra = Extra.allow

    strategy: ClassVar[str]

    output_scores: bool = False
    output_scores_format: GenerationOutputScoresFormat = 'probabilities'
    min_possible_score: float = 0.0
    renormalize_logits: bool = True
    max_new_tokens: conint(ge=1)
    min_new_tokens: conint(ge=1) = 1

    ## Extra params on top of HF:
    stop_sequences: Optional[List[str]] = None
    top_k_output_scores: Optional[conint(ge=1)] = None
    tokens_to_keep: Optional[List[str]] = None
    output_scores_tolerance: Optional[Union[
        confloat(gt=0, lt=1),
        confloat(le=0),
    ]] = 1e-4  ## Tokens with scores below this tolerance are ignored. Set to None to not filter any tokens.
    force_vocab_size: bool = True

    @root_validator(pre=False)
    def set_gen_params(cls, params: Dict) -> Dict:
        if params['output_scores_format'] == 'probabilities':
            params['renormalize_logits']: bool = True
            params['min_possible_score']: float = 0.0
            assert 0.0 < params['output_scores_tolerance'] < 1.0
        elif params['output_scores_format'] == 'log-probabilities':
            params['renormalize_logits']: bool = True
            params['min_possible_score']: float = -inf
            assert 0.0 < params['output_scores_tolerance'] < 1.0
            params['output_scores_tolerance']: float = log(params['output_scores_tolerance'])
            assert params['output_scores_tolerance'] < 0.0
        elif params['output_scores_format'] == 'logits':
            params['renormalize_logits']: bool = False
            params['min_possible_score']: float = -inf
            params['output_scores_tolerance']: Optional[float] = None  ## Do not filter out any tokens.
        else:
            raise NotImplementedError(f'Unsupported `output_scores_format`: "{params["output_scores_format"]}"')
        return params

    def hf_dict(self) -> Dict:
        return self.dict(exclude={
            'name',
            'stop_sequences',
            'top_k_output_scores',
            'output_scores_format',
            'min_possible_score',
            'tokens_to_keep',
            'force_vocab_size',
            'output_scores_tolerance',
        })


class GreedyDecodingParams(TextGenerationParams):
    strategy = 'GreedyDecoding'
    do_sample: Literal[False] = False  ## When doing greedy decoding, we do not sample.


class BeamSearchParams(TextGenerationParams):
    strategy = 'BeamSearch'
    do_sample: Literal[False] = False  ## When doing beam search, we do not sample.
    num_beams: conint(ge=1)


class TopKSamplingParams(TextGenerationParams):
    strategy = 'TopKSampling'
    temperature: confloat(gt=0.0, le=100.0) = 1.0
    do_sample: Literal[True] = True  ## When not doing greedy decoding, we should sample.


class NucleusSamplingParams(TextGenerationParams):
    strategy = 'NucleusSampling'
    do_sample: Literal[True] = True  ## When not doing greedy decoding, we should sample.
    temperature: confloat(gt=0.0, le=100.0) = 1.0


class LogitsProcessorListParams(TextGenerationParams):
    strategy = 'LogitsProcessorList'
    do_sample: Literal[True] = True  ## When not doing greedy decoding, we should sample.
    logits_processor: List[Any]


class TextGenerationParamsMapper(MappedParameters):
    _mapping = {
        ('GreedyDecoding', 'greedy'): GreedyDecodingParams,
        ('BeamSearch', 'beam'): BeamSearchParams,
        ('TopKSampling', 'top_k'): TopKSamplingParams,
        ('NucleusSampling', 'top_p', 'nucleus'): NucleusSamplingParams,
        ('LogitsProcessorList', 'logits_processor'): LogitsProcessorListParams,
    }


class GenerativeLM(Algorithm, ABC):
    tasks = Task.NEXT_TOKEN_PREDICTION
    inputs = Prompts
    outputs = NextTokens

    def _task_preprocess(self, batch: Prompts, *, prompt_prefix: str = '', **kwargs) -> Prompts:
        return apply_prompt_template(batch, prompt_prefix=prompt_prefix)

    @property
    @abstractmethod
    def max_num_generated_tokens(self) -> int:
        pass

    def _create_predictions(
            self,
            batch: Prompts,
            predictions: Dict,
            **kwargs
    ) -> NextTokens:
        if not isinstance(predictions, dict):
            raise ValueError(PROMPTING_FORMAT_MSG)
        if GENERATED_TEXTS_COL not in predictions:
            raise ValueError(f'Expected predictions to be a dict containing key `{GENERATED_TEXTS_COL}`')
        if 'generated_token_ids' in predictions \
                and 'generated_tokens' in predictions \
                and 'generated_token_scores' in predictions:
            generated_token_ids: np.ndarray = predictions.pop('generated_token_ids')
            generated_tokens: np.ndarray = predictions.pop('generated_tokens')
            generated_token_scores: np.ndarray = predictions.pop('generated_token_scores')
            generation_params: TextGenerationParams = self.hyperparams.generation_params
            assert isinstance(generation_params, TextGenerationParams)
            assert generated_token_ids.shape == generated_tokens.shape == generated_token_scores.shape
            batch_size: int = generated_token_ids.shape[0]
            num_generated_tokens: int = generated_token_ids.shape[1]
            for token_timestep_i in range(0, num_generated_tokens):
                generated_token_scores_col: str = GENERATED_TOKEN_SCORES_COL_TEMPLATE.format(
                    token_timestep_i=token_timestep_i + 1
                )
                predictions[generated_token_scores_col] = [
                    {
                        tok: tok_score
                        for tok, tok_score in zip(ex_timestep_tokens, ex_timestep_token_scores)
                        ## After applying output_scores_tolerance and tokens_to_keep, certain tokens should 
                        ## be 0.0 or -inf in the numpy array. We want to remove such tokens:
                        if tok_score > generation_params.min_possible_score
                    }
                    for ex_timestep_tokens, ex_timestep_token_scores in zip(
                        generated_tokens[:, token_timestep_i, :],
                        generated_token_scores[:, token_timestep_i, :],
                    )
                ]
            ## Pad the remaining cols with nulls:
            for token_timestep_i in range(num_generated_tokens, self.max_num_generated_tokens):
                generated_token_scores_col: str = GENERATED_TOKEN_SCORES_COL_TEMPLATE.format(
                    token_timestep_i=token_timestep_i + 1
                )
                predictions[generated_token_scores_col] = [None for _ in range(batch_size)]
        return NextTokens.from_task_data(
            data=batch,
            predictions=predictions,
            **kwargs
        )


class LanguageModelTaskMixin(Algorithm, ABC):
    lm: Optional[Union[GenerativeLM, Any]] = None
    icl_dataset: Optional[Dataset] = None
    icl_sampler: Optional[ICLSampler] = None  ## Will be not-None when icl_dataset is not-None.

    class Hyperparameters(Algorithm.Hyperparameters):
        lm: Optional[Dict]  ## Params for llm
        batch_size: Optional[conint(ge=1)] = 1  ## By default, predict 1 row at a time.
        prompt_template: constr(min_length=1)
        icl_template: Optional[constr(min_length=1)] = None

        ## Extra params will be passed on to ICLSampler.
        ## Filters by this column in ICL dataset to this column in the batch of data.
        icl_filter_col: Optional[constr(min_length=1)] = None

        @root_validator(pre=True)
        def set_lm_task_params(cls, params: Dict) -> Dict:
            set_param_from_alias(params, param='lm', alias=['llm', 'prompter', 'base_llm', 'base_model', 'base'])
            set_param_from_alias(params, param='prompt_template', alias=['template'])
            set_param_from_alias(params, param='icl_template', alias=['icl_prompt_template'])
            return params

    def initialize(self, model_dir: Optional[FileMetadata] = None):
        if self.icl_dataset is not None:
            if self.hyperparams.icl_template is None:
                raise ValueError(
                    f'When passing an In-Context Learning dataset, you must also pass hyperparameter '
                    f'`icl_template`'
                )
            self.icl_sampler: ICLSampler = self._create_icl_sampler(self.icl_dataset)
        if self.lm is None and self.hyperparams.lm is None:
            raise ValueError(
                f'To initialize {self.class_name}, you must either pass an `lm` explicitly or set the `lm` '
                f'hyperparam with a dict of parameters that can be used to initialize a language model.'
            )
        elif self.lm is None:
            self.lm: GenerativeLM = GenerativeLM.of(**{
                **dict(
                    model_dir=model_dir,
                    task=Task.NEXT_TOKEN_PREDICTION
                ),
                **self.hyperparams.lm,
                **dict(
                    cache_dir=getattr(self, 'cache_dir', None),
                    post_init=False,  ## Don't copy the model to device.
                )
            })

    def _create_icl_sampler(self, icl_dataset: Dataset) -> ICLSampler:
        return ICLSampler(
            icl_dataset=icl_dataset,
            **self.hyperparams.dict(),
        )

    def post_initialize(self):
        if isinstance(self.lm, Algorithm):
            self.lm.post_initialize()

    def _task_preprocess(self, batch: Dataset, **kwargs) -> Dataset:
        super(LanguageModelTaskMixin, self)._task_preprocess(batch=batch, **kwargs)
        if self.icl_sampler is None:
            return batch
        return self.icl_sampler.append_icl_examples(batch)

    @property
    def expanded_prompt_template(self) -> str:
        """The prompt template, prepended with the ICL template N times (N=num_shots)."""
        if self.icl_sampler is None:
            prompt_template: str = self.hyperparams.prompt_template
        else:
            prompt_template: str = self.icl_sampler.icl_expanded_prompt_template
        return prompt_template

    def _lm_predict(self, prompts: Prompts, **kwargs) -> NextTokens:
        from synthesizrr.base.framework.evaluator import Evaluator
        # print('Prompts:')
        # with pd_extended_display() as disp:
        #     disp(prompts.data.pandas())
        ## Calls GenerativeLM._task_preprocess & apply_prompt_template
        if isinstance(self.lm, Evaluator):
            lm_batch_size: int = get_default(
                self.lm._create_hyperparams().batch_size,
                self.hyperparams.batch_size,
            )
            next_tokens: Predictions = self.lm.evaluate(
                prompts,
                **{
                    **dict(
                        batch_size=lm_batch_size,
                        submission_batch_size=lm_batch_size,
                        progress_bar=None,
                        return_predictions=True,
                        failure_action=FailureAction.ERROR_DELAYED,
                    ),
                    **kwargs,
                },
            )
        elif isinstance(self.lm, GenerativeLM):
            lm_batch_size: int = get_default(
                self.lm.hyperparams.batch_size,
                self.hyperparams.batch_size,
            )
            next_tokens: Predictions = self.lm.predict(
                prompts,
                batch_size=lm_batch_size,
            )
        else:
            raise ValueError(f'Expected `lm` to be either an instance of {GenerativeLM} or {Evaluator}')
        if not isinstance(next_tokens, NextTokens):
            raise ValueError(f'Expected language model output to be {NextTokens}; found: {type_str(next_tokens)}')
        return next_tokens


class TextInputs(Dataset):
    tasks = Task.IN_CONTEXT_LEARNING

    features_schema = {}  ## No required features
    ground_truths_schema = {}  ## No ground-truths needed


class TextGenerations(TextGenerationsPredictionsBase):
    tasks = Task.IN_CONTEXT_LEARNING


class FewShotTextGenerator(LanguageModelTaskMixin):
    aliases = [
        'few-shot-text-generation',
        'text-generation',
        'text-generator',
    ]

    tasks = Task.IN_CONTEXT_LEARNING
    inputs = TextInputs
    outputs = TextGenerations

    class Hyperparameters(LanguageModelTaskMixin.Hyperparameters):
        pass

    def predict_step(self, batch: TextInputs, **kwargs) -> Any:
        pd.options.mode.chained_assignment = None
        with ignore_warnings():
            prompts: Prompts = Prompts.of(
                split=DataSplit.UNSUPERVISED,
                task=Task.NEXT_TOKEN_PREDICTION,
                prompt_template=self.expanded_prompt_template,
                prompt_template_apply='expand',
                ## Use pandas to prevent in-place modification errors:
                data=batch.data.as_layout(DataLayout.PANDAS),
                data_schema={
                    batch.data_schema.index_col: MLType.INDEX,
                    **batch.data_schema.features_schema,
                }
            ).read().apply_template()
            next_tokens: NextTokens = self._lm_predict(prompts, **kwargs)
        text_generations: TextGenerations = TextGenerations.of(
            task=self.task,
            **next_tokens.dict(exclude=('task')),
        )
        return text_generations

    def _create_predictions(self, batch: Dataset, predictions: Any, **kwargs) -> TextGenerations:
        assert isinstance(predictions, TextGenerations)
        return predictions


class FewShotRetrievalAugmentedTextGenerator(LanguageModelTaskMixin):
    aliases = [
        'few-shot-retrieval-augmented-text-generation',
        'few-shot-retrieval-augmented-language-modeling',
        'retrieval-augmented-text-generator',
        'retrieval-augmented-text-generation',
        'retrieval-augmented-language-modeling',
        'ralm',
    ]

    tasks = Task.IN_CONTEXT_LEARNING
    inputs = TextInputs
    outputs = TextGenerations

    retriever: Optional[Union[Retriever, Any]] = None

    class Hyperparameters(LanguageModelTaskMixin.Hyperparameters):
        query_template: Optional[str] = None
        retriever: Optional[Dict] = None  ## Params for retriever

    def initialize(self, model_dir: Optional[FileMetadata] = None):
        super(FewShotRetrievalAugmentedTextGenerator, self).initialize(model_dir)
        if self.retriever is None and self.hyperparams.retriever is None:
            raise ValueError(
                f'To initialize {self.class_name}, you must either pass an `retriever` explicitly or set the '
                f'`retriever` hyperparam with a dict of parameters that can be used to initialize a retriever.'
            )
        elif self.retriever is None:
            self.retriever: Algorithm = Algorithm.of(**{
                **dict(
                    task=Task.RETRIEVAL,
                ),
                **self.hyperparams.retriever,
            })
            assert isinstance(self.retriever, Retriever)

    # @property
    # def expanded_prompt_template(self) -> str:
    #     """The prompt template, prepended with the ICL template N times (N=num_shots)."""
    #     if self.icl_sampler is None:
    #         prompt_template: str = self.hyperparams.prompt_template
    #     else:
    #         prompt_template: str = self.icl_sampler.icl_expanded_prompt_template
    #     prompt_template: str = to_second_level_prompt_template(prompt_template, exclude=RANKED_RESULTS_TEMPLATE_KEY)
    #     prompt_template: str = prompt_template.format(**{RANKED_RESULTS_TEMPLATE_KEY: icl_template})
    #     return prompt

    def predict_step(self, batch: TextInputs, query_template: Optional[str] = None, **kwargs) -> Any:
        Alias.set_top_k(kwargs, default=1)
        kwargs['progress_bar'] = None
        query_template: Optional[str] = get_default(query_template, self.hyperparams.query_template)
        if query_template is None:
            raise ValueError(
                f'`query_template` must be passed either as a hyperparam or as a keyword argument during prediction.'
            )
        query_prompts: Prompts = Prompts.of(
            task=Task.NEXT_TOKEN_PREDICTION,
            split=DataSplit.UNSUPERVISED,
            prompt_template=query_template,
            prompt_template_apply='expand',
            ## Use pandas to prevent in-place modification errors:
            data=batch.data.as_layout(DataLayout.PANDAS),
            data_schema={
                col: MLType.INDEX if mltype is MLType.INDEX else MLType.TEXT
                for col, mltype in batch.data_schema.flatten().items()
            },
        ).read().apply_template()

        queries: Queries = Queries.of(
            split=DataSplit.UNSUPERVISED,
            task=Task.RETRIEVAL,
            data=query_prompts.data,
            data_schema={
                batch.data_schema.index_col: MLType.INDEX,
                TEXT_PROMPT_COL: MLType.TEXT,
            },
        ).read()
        ranked_results: RankedResults = self._retriever_predict(queries, **kwargs)
        ranked_results: RankedResults = ranked_results.flatten()
        prompts: Prompts = Prompts.of(
            split=DataSplit.UNSUPERVISED,
            task=Task.NEXT_TOKEN_PREDICTION,
            prompt_template=self.expanded_prompt_template,
            prompt_template_apply='expand',
            ## Use pandas to prevent in-place modification errors:
            data=ranked_results.data.as_layout(DataLayout.PANDAS),
            data_schema={
                batch.data_schema.index_col: MLType.INDEX,
                **batch.data_schema.features_schema,
                **ranked_results.data_schema.features_schema,
            }
        ).read().apply_template()

        next_tokens: NextTokens = self._lm_predict(prompts, **kwargs)
        text_generations: TextGenerations = TextGenerations.of(
            task=self.task,
            **next_tokens.dict(exclude='task'),
        )
        return text_generations

    def _retriever_predict(self, queries: Queries, **kwargs) -> RankedResults:
        from synthesizrr.base.framework.evaluator import Evaluator
        # print('Queries:')
        # with pd_extended_display() as disp:
        #     disp(queries.data.pandas())
        if isinstance(self.retriever, Evaluator):
            retriever_batch_size: int = get_default(
                self.retriever._create_hyperparams().batch_size,
                self.hyperparams.batch_size
            )
            ranked_results: Predictions = self.retriever.evaluate(
                queries,
                **{
                    **dict(
                        batch_size=retriever_batch_size,
                        submission_batch_size=retriever_batch_size,
                        progress_bar=None,
                        return_predictions=True,
                        failure_action=FailureAction.ERROR_DELAYED,
                    ),
                    **kwargs,
                }
            )
        elif isinstance(self.retriever, Retriever):
            retriever_batch_size: int = get_default(
                self.retriever.hyperparams.batch_size,
                self.hyperparams.batch_size,
            )
            ranked_results: Predictions = self.retriever.predict(
                queries,
                **{
                    **dict(
                        batch_size=retriever_batch_size,
                        progress_bar=None,
                    ),
                    **kwargs,
                }
            )
        else:
            raise ValueError(f'Expected `retriever` to be either an instance of {Retriever} or {Evaluator}')
        if not isinstance(ranked_results, RankedResults):
            raise ValueError(f'Expected retriever output to be {RankedResults}; found: {type_str(ranked_results)}')
        return ranked_results

    def _create_predictions(self, batch: Dataset, predictions: Any, **kwargs) -> TextGenerations:
        assert isinstance(predictions, TextGenerations)
        return predictions


class FewShotTextClassifier(LanguageModelTaskMixin, Classifier):
    aliases = [
        'ZeroShotTextClassifier',
        'zeroshot-text-multiclass',
        'zeroshot-text-classification',
    ]

    class Hyperparameters(LanguageModelTaskMixin.Hyperparameters):
        strategy: ZeroShotStrategy = 'entailment'
        entailment_yes_tokens: Tuple[str, ...] = ('▁Yes', '▁yes', 'Yes', 'yes')
        entailment_no_tokens: Tuple[str, ...] = ('▁No', '▁no', 'No', 'no')
        token_combination: TokenCombinationStrategy = 'max'
        timestep_combination: TimestepCombinationStrategy = 'max'
        label_verbalizer: Dict[str, str]
        prompt_template_label_key: constr(min_length=1)

        @root_validator(pre=True)
        def set_zeroshot_params(cls, params: Dict) -> Dict:
            set_param_from_alias(params, param='entailment_yes_tokens', alias=['yes_tokens'])
            set_param_from_alias(params, param='entailment_no_tokens', alias=['no_tokens'])
            set_param_from_alias(params, param='label_verbalizer', alias=['verbalizer'])
            set_param_from_alias(params, param='prompt_template_label_key', alias=['label_key'])
            if not isinstance(params['prompt_template'], str):
                raise ValueError(
                    f"Expected prompt template to be a string: "
                    f"found {type(params['prompt_template'])}"
                )
            lm_hyperparams: Dict = params['lm']['hyperparams']
            if isinstance(lm_hyperparams, Algorithm.Hyperparameters):
                lm_hyperparams: Dict = lm_hyperparams.dict()
            if not is_dict_like(lm_hyperparams):
                raise ValueError(
                    f'Expected `lm` to have dict-like `hyperparams`; '
                    f'found {type(lm_hyperparams)}'
                )
            if lm_hyperparams.get('generation_params') is None:
                lm_hyperparams['generation_params']: Dict = dict(
                    name='greedy',
                    max_new_tokens=3,
                    output_scores=True,
                )
            lm_hyperparams['generation_params']['output_scores']: bool = True
            lm_hyperparams['generation_params']['output_scores_format']: GenerationOutputScoresFormat = 'logits'
            params['lm']['hyperparams']: Dict = lm_hyperparams
            return params

    def _create_icl_sampler(self, icl_dataset: Dataset) -> ICLSampler:
        if not icl_dataset.has_ground_truths():
            raise ValueError(
                f'Expected dataset of In-Context Learning to have ground-truths; '
                f'found following data schema: {icl_dataset.data_schema} '
                f'and data columns: {icl_dataset.data.columns}'
            )
        return ClassificationICLSampler(
            icl_dataset=icl_dataset,
            **{
                **self.hyperparams.dict(),
                **dict(
                    strategy=self.hyperparams.strategy,
                    label_normalizer=self.label_normalizer,
                    label_verbalizer=self.hyperparams.label_verbalizer,
                ),
            }
        )

    def predict_step(self, batch: ClassificationData, **kwargs) -> Any:
        ## Stops Pandas SettingWithCopyWarning in output. Ref: https://stackoverflow.com/a/20627316
        pd.options.mode.chained_assignment = None
        with ignore_warnings():
            clf_prompts: Prompts = self.convert_input_clf_to_prompts(
                batch,
                prompt_template=self.expanded_prompt_template,
                prompt_template_label_key=self.hyperparams.prompt_template_label_key,
                labelspace=self.labelspace,
                label_normalizer=self.label_normalizer,
                label_verbalizer=self.hyperparams.label_verbalizer,
            )
            next_tokens: NextTokens = self._lm_predict(clf_prompts, **kwargs)
            predictions: Dict = self.convert_output_prompts_to_clf(batch, next_tokens)
        return predictions

    def convert_input_clf_to_prompts(self, *args, **kwargs) -> Prompts:
        if self.hyperparams.strategy == 'entailment':
            return self.convert_input_clf_to_prompts_entailment(*args, **kwargs)
        raise NotImplementedError(f'Unsupported strategy: "{self.hyperparams.strategy}"')

    @classmethod
    def convert_input_clf_to_prompts_entailment(
            cls,
            batch: ClassificationData,
            *,
            prompt_template: str,
            prompt_template_label_key: str,
            labelspace: Tuple[str, ...],
            label_normalizer: Callable,
            label_verbalizer: Dict[str, str],
    ) -> Prompts:
        label_col: str = prompt_template_label_key
        label_template_str: str = '{' + str(prompt_template_label_key) + '}'
        if label_template_str not in prompt_template:
            raise ValueError(
                f'Expected prompt-template to contain "{label_template_str}" '
                f'i.e. placeholder for the label column; '
                f'however, following prompt-template was passed:\n"{prompt_template}"'
            )
        prompt_template: str = to_second_level_prompt_template(prompt_template, exclude=label_col)
        label_verbalizer: Dict[str, str] = {
            ## We must normalize here because the user might not provide normalized labels in their verbalizer.
            label_normalizer(lb): lb_description
            for lb, lb_description in label_verbalizer.items()
        }
        label_templates: Dict[str, str] = {
            label_normalizer(lb): prompt_template.format(**{label_col: label_verbalizer[lb]})
            for lb in labelspace
        }
        clf_prompts: Prompts = Prompts.of(
            split=DataSplit.UNSUPERVISED,
            task=Task.NEXT_TOKEN_PREDICTION,
            prompt_template=label_templates,
            prompt_template_apply='expand',
            ## Use pandas to prevent in-place modification errors:
            data=batch.data.as_layout(DataLayout.PANDAS),
            data_schema={
                batch.data_schema.index_col: MLType.INDEX,
                **batch.data_schema.features_schema,
                label_col: MLType.CATEGORICAL,
            }
        ).read().apply_template()
        prompt_template_id_col_unique: Set[str] = set(clf_prompts.data[PROMPT_TEMPLATE_ID_COL].unique())
        if len(set(labelspace) - prompt_template_id_col_unique) > 0:
            raise ValueError(
                f'When expanding labels for zero-shot classification using entailment strategy, '
                f'we expect each label to be a new row, with the prompted label available in the column '
                f'"{PROMPT_TEMPLATE_ID_COL}"; however, "{PROMPT_TEMPLATE_ID_COL}" contains only the '
                f'following labels: {prompt_template_id_col_unique} and is missing '
                f'labels: {set(labelspace) - prompt_template_id_col_unique}'
            )
        return clf_prompts

    def convert_output_prompts_to_clf(self, batch: ClassificationData, prompt_preds: NextTokens) -> Dict:
        if self.hyperparams.strategy == 'entailment':
            return self.convert_output_prompts_to_clf_entailment(batch, prompt_preds)
        raise NotImplementedError(f'Unsupported strategy: "{self.hyperparams.strategy}"')

    def convert_output_prompts_to_clf_entailment(
            self,
            batch: ClassificationData,
            prompt_preds: NextTokens,
    ) -> Dict:
        entailment_df: pd.DataFrame = prompt_preds.data.pandas().apply(
            self.zeroshot_entailment,
            yes_tokens=self.hyperparams.entailment_yes_tokens,
            no_tokens=self.hyperparams.entailment_no_tokens,
            token_combination=self.hyperparams.token_combination,
            timestep_combination=self.hyperparams.timestep_combination,
            max_new_tokens=self.lm.hyperparams.generation_params.max_new_tokens,
            axis=1,
        ).reset_index(drop=True)
        predictions: Dict = self._entailment_df_to_predictions(
            batch,
            entailment_df,
            labelspace=self.labelspace,
            label_normalizer=self.label_normalizer,
            prompt_template_label_key=self.hyperparams.prompt_template_label_key,
        )
        return predictions

    @classmethod
    def zeroshot_entailment(
            cls,
            row: Union[pd.Series, Dict],
            *,
            yes_tokens: List[str],
            no_tokens: List[str],
            token_combination: TokenCombinationStrategy,
            timestep_combination: TokenCombinationStrategy,
            max_new_tokens: int,
    ) -> Union[pd.Series, Dict]:
        token_timesteps: List[int] = list(range(1, max_new_tokens + 1))
        yes_score: float = cls.calculate_token_score(
            row,
            tokens=yes_tokens,
            token_timesteps=token_timesteps,
            token_combination=token_combination,
            timestep_combination=timestep_combination,
        )
        no_score: float = cls.calculate_token_score(
            row,
            tokens=no_tokens,
            token_timesteps=token_timesteps,
            token_combination=token_combination,
            timestep_combination=timestep_combination,
        )
        row[ENTAILMENT_LOGIT_COL]: float = yes_score
        row[CONTRADICTION_LOGIT_COL]: float = no_score
        row[ENTAILMENT_PROBABILITY_COL], row[CONTRADICTION_PROBABILITY_COL] = stable_softmax(yes_score, no_score)
        return row

    @classmethod
    def calculate_token_score(
            cls,
            row: Union[pd.Series, Dict],
            *,
            tokens: List[str],
            token_timesteps: List[int],
            token_combination: TokenCombinationStrategy,
            timestep_combination: TimestepCombinationStrategy,
    ) -> float:
        tokens: List[str] = as_list(tokens)
        token_timesteps: List[int] = as_list(token_timesteps)
        token_score: List[float] = []
        for token_timestep_i in token_timesteps:
            token_score_col: str = GENERATED_TOKEN_SCORES_COL_TEMPLATE.format(token_timestep_i=token_timestep_i)
            if row.get(token_score_col) is None:
                continue
            token_score.append(
                cls.combine_token_scores(
                    [
                        get_default(row[token_score_col].get(token), 0.0)
                        for token in tokens
                    ],
                    combination=token_combination
                )
            )
        token_score: float = cls.combine_token_scores(token_score, combination=timestep_combination)
        return token_score

    @classmethod
    def combine_token_scores(
            cls,
            token_scores: List[float],
            combination: TokenCombinationStrategy,
    ) -> float:
        if combination == 'sum':
            return float(np.sum(token_scores))
        elif combination == 'max':
            return float(np.max(token_scores))
        elif combination == 'min':
            return float(np.min(token_scores))
        elif combination == 'median':
            return float(np.median(token_scores))
        else:
            raise NotImplementedError(f'Unsupported `combination`: "{combination}"')

    @classmethod
    def _entailment_df_to_predictions(
            cls,
            batch: ClassificationData,
            entailment_df: pd.DataFrame,
            *,
            labelspace: Tuple[str, ...],
            label_normalizer: Callable,
            prompt_template_label_key: str,
    ) -> Dict:
        predictions: Dict[str, List] = {
            'scores': [],
            'labels': labelspace,
        }
        labelwise_prompts: Dict[str, List[str]] = {}
        entailment_df.loc[:, PROMPT_TEMPLATE_ID_COL] = entailment_df[batch.data_schema.index_col].apply(
            lambda row_idx: str(row_idx).split(PROMPT_TEMPLATE_EXPANDER_SEP)[1]
        )
        entailment_df.loc[:, batch.data_schema.index_col] = entailment_df[batch.data_schema.index_col].apply(
            lambda row_idx: str(row_idx).split(PROMPT_TEMPLATE_EXPANDER_SEP)[0]
        )
        entailment_df_row_gb: PandasDataFrameGroupBy = entailment_df.groupby(batch.data_schema.index_col)
        for row_idx in batch.index():
            row_entailment_df: pd.DataFrame = entailment_df.loc[entailment_df_row_gb.groups[str(row_idx)]]
            if len(row_entailment_df) != len(labelspace):
                raise ValueError(
                    f'For row with index "{row_idx}", expected output from prompting (using entailment) '
                    f'to have {len(labelspace)} rows; found {len(row_entailment_df)} rows.'
                )
            predictions['scores'].append(
                cls._scores_from_row_entailment_df(row_entailment_df, labelspace, label_normalizer)
            )
            ## Add the exact prompts used as new columns in the predictions:
            row_labelwise_prompts: Dict = {
                f'{prompt_template_label_key}={prompt_template_id}-{PROMPT}': prompt
                for prompt_template_id, prompt in zip(
                    row_entailment_df[PROMPT_TEMPLATE_ID_COL], row_entailment_df[TEXT_PROMPT_COL]
                )
            }
            for row_label_prompt_col, row_label_prompt in row_labelwise_prompts.items():
                if row_label_prompt_col not in labelwise_prompts:
                    labelwise_prompts[row_label_prompt_col] = []
                labelwise_prompts[row_label_prompt_col].append(row_label_prompt)
        ## Add the labelwise prompts to the input batch:
        for label_prompt_col, label_prompts in labelwise_prompts.items():
            batch.data[label_prompt_col] = label_prompts
            batch.data_schema.features_schema[label_prompt_col] = MLType.TEXT
        return predictions

    @classmethod
    def _scores_from_row_entailment_df(
            cls,
            row_entailment_df: pd.DataFrame,
            labelspace: Tuple[str, ...],
            label_normalizer: Callable,
    ) -> np.ndarray:
        ## For multiclass, use the entailment logits of each row as the input to a softmax:
        entailment_logits: Dict[str, float] = dict(zip(
            row_entailment_df[PROMPT_TEMPLATE_ID_COL],
            row_entailment_df[ENTAILMENT_LOGIT_COL],
        ))
        entailment_logits: np.ndarray = np.array([entailment_logits[label_normalizer(lb)] for lb in labelspace])
        entailment_probs: np.ndarray = stable_softmax(entailment_logits)
        return entailment_probs


class FewShotTextMultiLabelClassifier(MultiLabelClassifier, FewShotTextClassifier):
    aliases = [
        'ZeroShotTextMultiLabelClassifier',
        'zeroshot-text-multilabel',
        'zeroshot-multilabel-text-classification',
        'zeroshot-text-classification-multilabel',
        'zeroshot-text-multilabel-classification',
    ]

    @classmethod
    def _scores_from_row_entailment_df(
            cls,
            row_entailment_df: pd.DataFrame,
            labelspace: Tuple[str, ...],
            label_normalizer: Callable,
    ) -> np.ndarray:
        ## For multilabel, use the entailment probability of each label of each row as the score:
        entailment_probs: Dict[str, float] = dict(zip(
            row_entailment_df[PROMPT_TEMPLATE_ID_COL],
            row_entailment_df[ENTAILMENT_PROBABILITY_COL],
        ))
        entailment_probs: np.ndarray = np.array([entailment_probs[label_normalizer(lb)] for lb in labelspace])
        return entailment_probs
