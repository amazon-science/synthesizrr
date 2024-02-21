from typing import *
import math, copy, io, pickle
from abc import ABC, abstractmethod
import numpy as np, pandas as pd, ray
from scipy import stats as sps
from synthesizrr.base.constants import DataSplit, MLType, AggregationStrategy, Parallelize, Alias
from synthesizrr.base.util import Parameters, MutableParameters, Registry, StringUtil, classproperty, as_list, \
    safe_validate_arguments, get_default, set_param_from_alias, str_normalize, dispatch, dispatch_executor, \
    accumulate, accumulate_iter, partial_sort, is_function, fn_str, as_set, format_exception_msg, is_null, Log, \
    ProgressBar
from synthesizrr.base.data.sdf import ScalableDataFrame
from pydantic import constr, conint, confloat, root_validator
from functools import singledispatchmethod

Metric = "Metric"
Metrics = "Metrics"


class MetricEvaluationError(Exception):
    pass


class Metric(MutableParameters, Registry):
    """
    A base class for all metrics, i.e. numeric values (rather than plots, etc).
    Also used as a concrete class to store metrics without evaluating them.
    """

    _allow_multiple_subclasses = False
    _allow_subclass_override = True
    required_assets: ClassVar[Tuple[MLType, ...]] = ()

    class Config(MutableParameters.Config):
        keep_untouched = (singledispatchmethod,)
        ## Ref of validating set calls: https://docs.pydantic.dev/1.10/usage/model_config/
        validate_assignment = True

    class Params(Parameters):
        """
        BaseModel for parameters. Expected to be overridden by subclasses.
        E.g.
            class RecallAtPrecision(Metric):
                class Params(Metric.Params):
                    precision: confloat(ge=0.0, le=1.0)
                ...
        """
        display_ignore: ClassVar[Tuple[str, ...]] = (
            'display_decimals',
            'num_cpus',
            'num_gpus',
            'max_retries',
            'max_workers',
            'parallelize',
            'batch_size',
        )
        display_decimals: int = 5
        num_cpus: int = 1
        num_gpus: int = 0
        max_workers: Optional[int] = None
        max_retries: int = 1  ## Do not retry

        @root_validator(pre=True)
        def _set_alias(cls, params: Dict) -> Dict:
            set_param_from_alias(params, param='display_decimals', alias=['decimals', 'rounding', 'round'])
            return params

    name: constr(min_length=1)
    params: Params = dict()
    value: Optional[Any] = None

    @classmethod
    def _pre_registration_hook(cls):
        if cls.update == Metric.update and cls.compute == Metric.compute \
                and cls.compute_only == Metric.compute_only:
            raise ValueError(
                f'As `{cls.class_name}` is a subclass of `{Metric.class_name}`, you must implement either:'
                f'\n(1) `update` and `compute` functions, OR:'
                f'\n(2) `compute_only` function'
                f'\nAt present, neither are implemented.'
            )
        elif cls.update != Metric.update and cls.compute == Metric.compute:
            raise ValueError(
                f'As `{cls.class_name}` is a subclass of `{Metric.class_name}`, you must implement both '
                f'`update` and `compute` functions together; at present, only `update` is implemented.'
            )
        elif cls.update == Metric.update and cls.compute != Metric.compute:
            raise ValueError(
                f'As `{cls.class_name}` is a subclass of `{Metric.class_name}`, you must implement both '
                f'`update` and `compute` functions together; at present, only `compute` is implemented.'
            )

    @root_validator(pre=True)
    def convert_params(cls, params: Dict):
        params['params'] = super(Metric, cls)._convert_params(cls.Params, params.get('params'))
        params['name'] = cls.class_name
        if 'value' in params:
            if is_null(params['value']):
                params.pop('value', None)
        return params

    @classmethod
    def of(
            cls,
            name: Optional[Union[Metric, Dict, str]] = None,
            **kwargs,
    ) -> Metric:
        if isinstance(name, Metric):
            return name
        if isinstance(name, dict):
            return cls.of(**name)
        if isinstance(name, str):
            if name is not None:
                MetricClass: Type[Metric] = Metric.get_subclass(name)
            else:
                MetricClass: Type[Metric] = cls
            if MetricClass == Metric:
                raise ValueError(
                    f'"{Metric.class_name}" is an abstract class. '
                    f'To create an instance, please either pass `name`, '
                    f'or call .of(...) on a subclass of "{Metric.class_name}".'
                )
            try:
                return MetricClass(**kwargs)
            except Exception as e:
                raise ValueError(f'Cannot create metric with kwargs:\n{kwargs}\nError: {format_exception_msg(e)}')
        raise NotImplementedError(f'Unsupported value for `name`; found {type(name)} with following value:\n{name}')

    def clear(self) -> Metric:
        return self.of(
            name=self.name,
            **self.dict(include={'params'}),
        )

    def __repr__(self) -> str:
        return str(self)

    def _repr_html_(self) -> str:
        out_str: str = f'<b>{self.display_name}</b>'
        if self.value is not None:
            out_str += f': {self.display_value}'
        return out_str

    def __str__(self) -> str:
        out_str: str = self.display_name
        if self.value is not None:
            out_str += f': {self.display_value}'
        return out_str

    @property
    def display_name(self) -> str:
        """Provides a single-line displayable string from the metric name and its parameters (if any)."""
        display_name: str = self.name
        param_name_values: List[Tuple[str, Any]] = sorted(
            list(self.params.dict(exclude=as_set(self.params.display_ignore)).items()),
            key=lambda x: x[0]
        )
        if len(param_name_values) != 0:
            display_name += StringUtil.LEFT_PAREN
            params_strs: List[str] = []
            for param_name, param_value in param_name_values:
                if is_function(param_value):
                    param_value: str = fn_str(param_value)
                elif isinstance(param_value, str):
                    param_value: str = f'"{param_value}"'
                else:
                    param_value: str = str(param_value)
                params_strs.append(f'{param_name}={param_value}')
            display_name += ','.join(params_strs)
            display_name += StringUtil.RIGHT_PAREN
        return display_name

    @property
    def display_value(self) -> str:
        val = self.value
        if isinstance(val, (float, int)):
            ## Keeps 'display_decimals' digits after decimal-point. E.g. f'{0.0000012345678:.{3}e}' becomes '1.235e-06'
            return f'{val:.{self.params.display_decimals}e}'
        else:
            return str(val)

    @property
    def aiw_format(self):
        return {
            'METRIC_NAME': self.name,
            'METRIC_PARAMS': self.params,
            'METRIC_VALUE': self.value
        }

    @classproperty
    def is_rolling(cls) -> bool:
        ## Ref: https://stackoverflow.com/a/59762827
        return cls.update != Metric.update and cls.compute != Metric.compute

    @classproperty
    def is_compute_only(cls) -> bool:
        return cls.compute_only != Metric.compute_only

    @safe_validate_arguments
    def evaluate(
            self,
            data: Any,
            *,
            rolling: Optional[bool] = None,
            inplace: Optional[bool] = None,
    ) -> Metric:
        if rolling is None:
            if self.is_rolling is True and self.is_compute_only is False:
                rolling: bool = True
            elif self.is_rolling is False and self.is_compute_only is True:
                rolling: bool = False
            elif self.is_rolling is True and self.is_compute_only is True:
                raise ValueError(
                    f'It is unclear if we want to call metric {self.class_name} in a rolling fashion or not; '
                    f'please pass parameter `rolling=True/False` explicitly to .evaluate(...)'
                )
            else:
                raise ValueError(
                    f'Metric is neither rolling nor compute_only; please implement either: '
                    f'\n(1) `update` and `compute` functions, or,'
                    f'\n(2) `compute_only` function.'
                )

        if rolling:
            if self.is_rolling is False:
                raise ValueError(
                    f'Cannot run .evaluate() with `rolling=True`; '
                    f'Metric subclass "{self.class_name}" has not implemented either .update(...) or .compute(), '
                    f'or both. These are needed to update the metric in a rolling fashion.'
                )
            self.update(data)
            metric_value: Any = self.compute()
            inplace: bool = get_default(inplace, True)
        else:
            if self.is_compute_only is False:
                raise ValueError(
                    f'Cannot run .evaluate() with `rolling=False`; '
                    f'Metric subclass "{self.class_name}" has not implemented .compute_only(). '
                    f'This is needed to compute the metric value in a non-rolling fashion.'
                )
            metric_value: Any = self.compute_only(data)
            inplace: bool = get_default(inplace, False)
        if isinstance(self, TabularMetric):
            metric_value: pd.DataFrame = ScalableDataFrame.of(metric_value).pandas()
        if inplace:
            ## Update .value of the current metric itself.
            metric: Metric = self
        else:
            ## Create a new metric instance with the same params but fresh internal state.
            metric: Metric = self.clear()
        metric.value = metric_value
        return metric

    def update(self, data: Any):
        """Users can override update+compute (for rolling implementation) or compute_only (for non-rolling)."""
        raise NotImplementedError(f'Cannot calculate rolling metric value using Metric subclass {self.class_name}')

    def compute(self) -> Any:
        """Users can override update+compute (for rolling implementation) or compute_only (for non-rolling)."""
        raise NotImplementedError(f'Cannot calculate rolling metric value using Metric subclass {self.class_name}')

    def compute_only(self, data: Any) -> Any:
        """Users can override update+compute (for rolling implementation) or compute_only (for non-rolling)."""
        raise NotImplementedError(f'Cannot calculate metric value using Metric subclass {self.class_name}')

    def merge(self, other: Metric) -> Metric:
        raise NotImplementedError(f'Cannot merge metrics with Metric subclass {self.class_name}')

    @classproperty
    def is_mergeable(cls) -> bool:
        return cls.merge != Metric.merge  ## Ref: https://stackoverflow.com/a/59762827


class PercentageMetric(Metric, ABC):
    """Class for metrics which can be meaningfully expressed as a percentage."""
    value: Optional[float] = None

    class Params(Metric.Params):
        display_decimals: int = 2

    @property
    def display_value(self) -> str:
        return f'{100 * self.value:.{self.params.display_decimals}f}%'


class AggregatedPercentageMetric(PercentageMetric, ABC):
    class Params(PercentageMetric.Params):
        aggregation: AggregationStrategy

        @root_validator(pre=True)
        def set_aliases(cls, params: Dict) -> Dict:
            set_param_from_alias(params, param='aggregation', alias=[
                'strategy', 'combine', 'agg_strategy', 'combination', 'agg', 'combination_strategy',
            ])
            if params.get('aggregation') is not None:
                if str_normalize(params['aggregation']) in {str_normalize('mean'), str_normalize('avg')}:
                    params['aggregation']: AggregationStrategy = AggregationStrategy.AVERAGE
            return params

    def _aggregate(self, vals: List[float]) -> float:
        if self.params.aggregation is AggregationStrategy.AVERAGE:
            return float(np.mean(vals))
        elif self.params.aggregation is AggregationStrategy.MIN:
            return float(np.min(vals))
        elif self.params.aggregation is AggregationStrategy.MAX:
            return float(np.max(vals))
        elif self.params.aggregation is AggregationStrategy.MEDIAN:
            return float(np.median(vals))
        raise NotImplementedError(f'Cannot aggregate metrics using {self.params.aggregation}')


class CountingMetric(Metric, ABC):
    """Class for metrics which are counts."""
    value: Optional[conint(ge=0)] = None

    class Params(Metric.Params):
        display_decimals: int = 2

    @property
    def display_value(self) -> str:
        return f'{self.value} ' \
               f'({StringUtil.readable_number(self.value, decimals=self.params.display_decimals, short=True)})'


class TabularMetric(Metric, ABC):
    value: Optional[pd.DataFrame] = None

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.value is not None:
            table_markdown: str = self.display_value
            table_width: int = max([len(x) for x in table_markdown.split('\n')])
            table_width_minus_2: int = max(table_width - 2, 2)
            out_str: str = f'{self.display_name}:' \
                           f'\n {"=" * table_width_minus_2} ' \
                           f'\n{table_markdown}' \
                           f'\n {"-" * table_width_minus_2} '
        else:
            out_str: str = self.display_name
        return out_str

    @property
    def display_value(self) -> str:
        return str(self.value.to_markdown())

    def _repr_html_(self) -> str:
        if self.value is not None:
            out_str: str = f'<b>{self.display_name}:</b>' \
                           f'<br>{self.value.to_html()}'
        else:
            out_str: str = f'<b>{self.display_name}</b>'
        return out_str


class Metrics(MutableParameters):
    name: Optional[str] = None
    metrics: Dict[DataSplit, List[Metric]]

    @classmethod
    def of(cls, metrics: Optional[Union[Metrics, Dict]] = None, **metrics_kwargs) -> 'Metrics':
        metrics_kwargs: Dict[DataSplit, List[Metric]] = DataSplit.convert_keys(metrics_kwargs)
        if metrics is not None:
            if isinstance(metrics, Metrics):
                metrics: Dict = metrics.metrics
            if isinstance(metrics, dict):
                if 'metrics' in metrics:
                    metrics: Dict = metrics['metrics']
                metrics_kwargs = {
                    **DataSplit.convert_keys(metrics),
                    **metrics_kwargs,
                }
        try:
            return cls(metrics=metrics_kwargs)
        except Exception as e:
            raise ValueError(f'Failed to create Metrics object.\nnError: {format_exception_msg(e)}')

    @root_validator(pre=True)
    def set_metrics(cls, params: Dict):
        metrics_dict: Dict[DataSplit, List[Metric]] = {}
        for data_split, metrics in params['metrics'].items():
            if metrics is None:
                continue
            metrics_dict[data_split]: List[Metric] = []
            for metric in as_list(metrics):
                if isinstance(metric, str):
                    metric: Metric = Metric.of(name=metric)
                elif isinstance(metric, dict):
                    metric: Metric = Metric.of(**metric)
                if not isinstance(metric, Metric):
                    raise ValueError(
                        f'Unsupported value for metric: '
                        f'{type(metric)} with following value:\n{metric}'
                    )
                metrics_dict[data_split].append(metric)
            if len(metrics_dict[data_split]) == 0:
                metrics_dict.pop(data_split, None)
        params['metrics'] = metrics_dict
        return params

    @safe_validate_arguments
    def find(
            self,
            select: Union[Union[Metric, Dict, str], List[Union[Metric, Dict, str]]],
            *,
            data_split: DataSplit,
            return_single: bool = True,
    ) -> Union[List[Metric], Metric]:
        data_split: DataSplit = DataSplit.from_str(data_split)
        select: List[Union[Metric, Dict, str]] = [Metric.of(m) for m in as_list(select)]
        metrics: List[Metric] = self[data_split]
        selected: Dict[str, Metric] = {
            metric.display_name: metric
            for select_metric in select
            for metric in metrics
            if select_metric.display_name == metric.display_name
        }
        selected: List[Metric] = list(selected.values())
        if return_single and (len(select) == len(selected) == 1):
            return selected[0]
        return selected

    def __getattr__(self, attr_name: Union[str, DataSplit]) -> Optional[List[Metric]]:
        if DataSplit.matches_any(attr_name):
            return self.metrics.get(DataSplit.from_str(attr_name))
        raise AttributeError(
            f'`{attr_name}` is neither an attribute of {self.class_name} '
            f'nor a valid value of {DataSplit}.'
        )

    def __getitem__(self, attr_name: Union[str, DataSplit]) -> Optional[List[Metric]]:
        if DataSplit.matches_any(attr_name):
            return self.metrics.get(DataSplit.from_str(attr_name))
        raise KeyError(f'Unknown value for {DataSplit}: {attr_name}')

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        out_str: str = f'''Metrics{f' for "{self.name}"' if self.name is not None else ''}'''
        long_divider: str = '=' * 80
        short_divider: str = '=' * 30
        out_str: str = long_divider + f'\n{out_str}\n' + long_divider
        data_splits: List[DataSplit] = partial_sort(
            list(self.metrics.keys()),
            [
                DataSplit.TRAIN_VAL,
                DataSplit.TRAIN,
                DataSplit.VALIDATION,
                DataSplit.TEST,
                DataSplit.UNSUPERVISED,
            ]
        )
        for data_split in data_splits:
            metrics_list: List[Metric] = self.metrics[data_split]
            out_str += f'\n{data_split.capitalize()} metrics:\n' + short_divider
            for metric in metrics_list:
                out_str += f'\n{str(metric)}'
            out_str += '\n\n'
        return out_str.strip()

    def _repr_html_(self) -> str:
        out_str: str = f'''<h3>Metrics{f' "{self.name}"' if self.name is not None else ''}</h3>'''
        data_splits: List[DataSplit] = partial_sort(
            list(self.metrics.keys()),
            [
                DataSplit.TRAIN_VAL,
                DataSplit.TRAIN,
                DataSplit.VALIDATION,
                DataSplit.TEST,
                DataSplit.UNSUPERVISED,
            ]
        )
        for data_split in data_splits:
            metrics_list: List[Metric] = self.metrics[data_split]
            out_str += f'<hr><h4>{data_split.capitalize()} metrics:</h4>'
            for metric in metrics_list:
                out_str += f'{metric._repr_html_()}<br>'
        return out_str

    @safe_validate_arguments
    def evaluate(
            self,
            data: Any,
            *,
            data_split: Optional[DataSplit] = None,
            inplace: bool = False,
            parallelize: Parallelize = Parallelize.sync,
            allow_partial_metrics: bool = True,
            return_metrics_list: bool = False,
            iter: bool = False,
            **kwargs,
    ) -> Union[Metrics, List[Metric], Generator]:
        data_split: Optional[DataSplit] = get_default(data_split, data.data_split)
        if data_split is None:
            raise ValueError(
                f'You must provide a data-split for the input data when calling {self.class_name}.compute(data)'
            )
        data_split: DataSplit = DataSplit.from_str(data_split)
        if data_split not in self.metrics:
            raise ValueError(
                f'There are no {data_split} metrics stored; available metrics are:'
                f'\n{StringUtil.pretty(self.metrics)}'
            )
        metrics_list: List[Metric] = self.metrics[data_split]

        ## Remove the progress bar from the kwargs before passing it to `dispatch_executor`
        progress_bar: Optional[Dict] = Alias.get_progress_bar(
            kwargs,
            ## By default, show progressbar when using threads/processes/ray/etc:
            default_progress_bar=parallelize != Parallelize.sync,
        )
        pbar: ProgressBar = ProgressBar.of(
            progress_bar,
            desc='Calculating metrics',
            total=len(metrics_list),
            prefer_kwargs=False,
        )
        if parallelize is Parallelize.ray:
            data_ref = ray.put(data)
            data = data_ref
        metric_futs: List[Tuple[Metric, Any]] = []
        for metric_i, compute_only_metric in [
            (metric_i, metric)
            for metric_i, metric in enumerate(metrics_list) if metric.is_compute_only
        ]:
            metric_futs.append(
                (
                    compute_only_metric,
                    dispatch(
                        self._evaluate_compute_only_metric,
                        metric_i=metric_i,
                        compute_only_metric=compute_only_metric,
                        data=data,
                        inplace=inplace,
                        **self._get_metric_exn_params(compute_only_metric, parallelize=parallelize),
                    ),
                )
            )
        for metric_i, rolling_metric in [
            (metric_i, metric)
            for metric_i, metric in enumerate(metrics_list) if metric.is_rolling
        ]:
            metric_futs.append(
                (
                    rolling_metric,
                    dispatch(
                        self._evaluate_rolling_metric,
                        metric_i=metric_i,
                        rolling_metric=rolling_metric,
                        data=data,
                        inplace=inplace,
                        **self._get_metric_exn_params(rolling_metric, parallelize=parallelize),
                    ),
                )
            )
        if iter:
            metrics_gen: Generator = accumulate_iter([_fut for _met, _fut in metric_futs], progress_bar=pbar)
            return metrics_gen
        evaluated_metrics_list: List[Tuple[int, Metric]] = []
        for metric, metric_fut in metric_futs:
            try:
                evaluated_metrics_list.append(accumulate(metric_fut))
            except Exception as e:
                error_msg: str = f'Error while calculating metric {metric.display_name}:\n{format_exception_msg(e)}'
                if allow_partial_metrics:
                    print(error_msg)
                else:
                    raise ValueError(error_msg)
            finally:
                pbar.update(1)
        if len(evaluated_metrics_list) == len(metric_futs):
            pbar.success(f'Calculated all {len(evaluated_metrics_list)} metrics')
        else:
            pbar.failed(f'Calculated {len(evaluated_metrics_list)} of {len(metric_futs)} metrics')

        if parallelize is Parallelize.ray:
            del data_ref
        evaluated_metrics_list: List[Metric] = [
            evaluated_metric
            for _, evaluated_metric in sorted(evaluated_metrics_list, key=lambda x: x[0])
        ]
        if return_metrics_list:
            return evaluated_metrics_list
        if inplace:
            self.metrics[data_split]: List[Metric] = evaluated_metrics_list
            return self
        else:
            return self.update_params(
                metrics={
                    **{
                        split: [metric.copy() for metric in metrics_list]
                        for split, metrics_list in self.metrics.items()
                        if split is not data_split
                    },
                    data_split: evaluated_metrics_list
                }
            )

    @staticmethod
    def _get_metric_exn_params(metric: Metric, *, parallelize: Parallelize) -> Dict:
        exn_params: Dict = dict(
            parallelize=parallelize,
        )
        if parallelize in {Parallelize.threads, Parallelize.processes}:
            exn_params['executor'] = dispatch_executor(
                parallelize=parallelize,
                max_workers=metric.params.max_workers,
            )
        if parallelize is Parallelize.ray:
            exn_params['num_cpus'] = metric.params.num_cpus
            exn_params['num_gpus'] = metric.params.num_gpus
            exn_params['max_retries'] = metric.params.max_retries
        # print(f'Running {metric.display_name} using {exn_params}')
        return exn_params

    @staticmethod
    def _evaluate_compute_only_metric(
            metric_i: int,
            compute_only_metric: Metric,
            data: Any,
            inplace: bool,
            **kwargs,
    ) -> Tuple[int, Metric]:
        try:
            return metric_i, compute_only_metric.evaluate(accumulate(data), rolling=False, inplace=inplace)
        except Exception as e:
            error_msg: str = f'Failed to evaluate compute-only metric "{compute_only_metric.name}" ' \
                             f'with params {compute_only_metric.params}:' \
                             f'\n{format_exception_msg(e, short=False)}'
            raise MetricEvaluationError(error_msg)

    @staticmethod
    def _evaluate_rolling_metric(
            metric_i: int,
            rolling_metric: Metric,
            data: Any,
            inplace: bool,
            **kwargs,
    ) -> Tuple[int, Metric]:
        try:
            return metric_i, rolling_metric.evaluate(accumulate(data), rolling=True, inplace=inplace)
        except Exception as e:
            error_msg: str = f'Failed to evaluate rolling metric "{rolling_metric.name}" ' \
                             f'with params {rolling_metric.params}:' \
                             f'\n{format_exception_msg(e, short=False)}'
            raise MetricEvaluationError(error_msg)


MetricsCollection = Metrics
Metrics.update_forward_refs()


def metric_stats_str(
        metric_display_name: str,
        metric_stats: Dict[str, Union[int, float, Dict]],
        *,
        range: bool = True,
        display_decimals: Optional[int] = None,
        **kwargs,
) -> str:
    assert isinstance(range, bool)

    if metric_stats.get('metric_dict') is not None:
        def metric_value_str(value, display_decimals: int) -> Optional[str]:
            if value is None:
                return None
            metric_dict: Dict = copy.deepcopy(metric_stats['metric_dict'])
            metric_dict['params']['display_decimals'] = display_decimals
            metric_dict['value'] = value
            metric: Metric = Metric.of(**metric_dict)
            if metric.value is None:
                return None
            return metric.display_value  ## Use the display value

        display_decimals: int = get_default(display_decimals, metric_stats['metric_dict']['params']['display_decimals'])
    else:
        def metric_value_str(value, display_decimals: int) -> str:
            return f'{value:.{display_decimals}e}' if value is not None else None
    display_decimals: int = get_default(display_decimals, 5)
    std_decimals: int = math.ceil(display_decimals / 2)
    num_samples_str: str = str(metric_stats["num_samples"])
    mean_str: str = metric_value_str(metric_stats["mean"], display_decimals=display_decimals)
    std_str: Optional[str] = metric_value_str(metric_stats["std"], display_decimals=std_decimals)
    min_str: Optional[str] = metric_value_str(metric_stats["min"], display_decimals=display_decimals)
    max_str: Optional[str] = metric_value_str(metric_stats["max"], display_decimals=display_decimals)

    if metric_stats.get('num_samples', 1) == 1:
        out: str = f'{metric_display_name}: {mean_str}'
    elif range is False or min_str is None or max_str is None:
        out: str = f'{metric_display_name}: {mean_str} ± {std_str} ' \
                   f'(mean ± std. across {num_samples_str} run(s))'
    else:
        out: str = f'{metric_display_name}: {mean_str} ± {std_str} ∈ [{min_str}, {max_str}] ' \
                   f'(mean ± std. with range [min, max] across {num_samples_str} run(s))'
    return out
