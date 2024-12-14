import io
from typing import *
import time, os, json, math, numpy as np, pandas as pd
from copy import deepcopy
from abc import abstractmethod, ABC
from synthergent.base.data import FileMetadata, ScalableDataFrame
from synthergent.base.constants import MLType, DataSplit, Storage, Task, Alias
from synthergent.base.framework.algorithm import Algorithm, TaskOrStr
from synthergent.base.framework.tracker import Tracker, DEFAULT_TRACKER_PARAMS
from synthergent.base.framework.metric import Metric, Metrics
from synthergent.base.framework.task_data import Dataset
from synthergent.base.framework.predictions import Predictions, load_predictions, save_predictions
from synthergent.base.util import Parameters, MutableParameters, Registry, FractionalBool, safe_validate_arguments, StringUtil, \
    Timeout, Timeout24Hr, TimeoutNever, all_are_false, accumulate, any_are_none, is_function, as_list, get_default, \
    random_sample, Log, format_exception_msg, start_daemon, stop_daemon, run_concurrent, NeverFailJsonEncoder
from synthergent.base.util.aws import S3Util
from synthergent.base.constants import _LIBRARY_NAME
from pydantic import root_validator, Extra, conint, confloat
from pydantic.typing import Literal

Evaluator = "Evaluator"


class Evaluator(MutableParameters, Registry, ABC):
    _allow_multiple_subclasses: ClassVar[bool] = False  ## Rejects multiple subclasses registered to the same name.
    _allow_subclass_override: ClassVar[bool] = True  ## Allows replacement of subclass with same name.

    class Config(Parameters.Config):
        extra = Extra.ignore

    class RunConfig(Parameters):
        class Config(Parameters.Config):
            extra = Extra.allow

    task: Optional[TaskOrStr] = None
    AlgorithmClass: Optional[Union[Type[Algorithm], str]] = None
    hyperparams: Optional[Dict] = None
    model: Optional[Algorithm] = None
    model_dir: Optional[FileMetadata] = None
    cache_dir: Optional[Union[FileMetadata, Dict, str]] = FileMetadata.of(f'~/.cache/{_LIBRARY_NAME}/models/')
    run_config: RunConfig = dict()

    validate_inputs: Optional[FractionalBool] = None
    validate_outputs: Optional[FractionalBool] = None
    ## How long to cache the model. By default, do not cache the model (i.e. cache_timeout=None):
    cache_timeout: Optional[Union[Timeout, confloat(gt=0)]] = None
    _cache_timeout_daemon_id: Optional[str] = None
    _cache_timeout_daemons: Dict[str, List[bool]] = {}
    _evaluator_is_running: bool = False

    custom_definitions: Tuple[Any, ...] = ()  ## NOTE! Pydantic complains is you make the typing Callable here.
    ## Logging verbosity. 0 = zero logging, 1 = Basic logging, 2 = verbose logging, 3 = super verbose logging.
    verbosity: conint(ge=0) = 1

    @root_validator(pre=True)
    def evaluator_params(cls, params: Dict):
        Alias.set_AlgorithmClass(params)
        Alias.set_model_dir(params)
        Alias.set_cache_dir(params)
        Alias.set_cache_timeout(params)
        Alias.set_custom_definitions(params)
        Alias.set_verbosity(params)

        if isinstance(params.get('cache_timeout'), (int, float)):
            if math.isinf(params['cache_timeout']):
                params['cache_timeout']: Timeout = TimeoutNever()
            else:
                params['cache_timeout']: Timeout = Timeout24Hr(timeout=params['cache_timeout'])

        ## Ensure user-defined classes are registered on the driver (i.e. Jupyter/IPython).
        custom_definitions: List[Any] = as_list(params.get('custom_definitions', ()))
        for custom_fn in custom_definitions:
            if not is_function(custom_fn):
                raise ValueError(
                    f'`custom_definitions` must be a function in which you include the entire source code of any '
                    f'custom classes (including imports!). '
                    f'Found object of type: {type(params["custom_definitions"])}'
                )
            custom_fn()
        params['custom_definitions'] = tuple(custom_definitions)

        if params.get('model_dir') is not None:
            params['model_dir']: FileMetadata = FileMetadata.of(params['model_dir'])

        if params.get('cache_dir') is not None:
            params['cache_dir']: FileMetadata = FileMetadata.of(params['cache_dir'])

        ## Set task, AlgorithmClass, hyperparams:
        algo_params: Dict = cls._get_algo_params(
            model=params.get('model'),
            model_dir=params.get('model_dir'),
            task=params.get('task'),
            AlgorithmClass=params.get('AlgorithmClass'),
            hyperparams=params.get('hyperparams'),
        )
        params.update(algo_params)

        ## This allows us to create a new Evaluator instance without specifying `run_config`.
        ## If it is specified, we will pick cls.RunConfig, which can be overridden by the subclass.
        params['run_config'] = cls._convert_params(cls.RunConfig, params.get('run_config', {}))
        if not isinstance(params['run_config'], Evaluator.RunConfig):
            raise ValueError(
                f'Custom run_config class does not inherit from the base class version. '
                f'Please ensure your custom class for run_config is called "{cls.RunConfig.class_name}" '
                f'and inherits from "{cls.RunConfig.class_name}" defined in the base class.'
            )
        return params

    @classmethod
    def _get_algo_params(
            cls,
            model_dir: Optional[Union[FileMetadata, str]],
            model: Optional[Algorithm],
            task: Optional[TaskOrStr],
            AlgorithmClass: Optional[Union[Algorithm, Type[Algorithm], str]],
            hyperparams: Optional[Union[Algorithm.Hyperparameters, Dict]],
    ) -> Dict:
        def hyperparams_to_dict(hyperparams):
            if isinstance(hyperparams, Algorithm.Hyperparameters):
                return hyperparams.dict()
            if not isinstance(hyperparams, dict):
                raise ValueError(
                    f'Expected `hyperparams` to be either a dict or {Algorithm.Hyperparameters}; '
                    f'found {type(hyperparams)}'
                )
            return hyperparams

        if model is not None and isinstance(model, Algorithm):
            return dict(
                model_dir=model_dir,
                task=model.task,
                AlgorithmClass=model.__class__,
                hyperparams=model.hyperparams.dict(),
            )
        if model_dir is not None:
            model_dir: FileMetadata = FileMetadata.of(model_dir)
            model_params: Optional[Dict] = Algorithm.load_params(model_dir, raise_error=False)
            if model_params is not None:
                ## The model params file was found:
                AlgorithmClass: str = get_default(AlgorithmClass, model_params['algorithm'])
                task: TaskOrStr = get_default(task, model_params['task'])
                hyperparams: Union[Algorithm.Hyperparameters, Dict] = get_default(
                    hyperparams, model_params['hyperparams'],
                )
        if AlgorithmClass is not None:
            if isinstance(AlgorithmClass, Algorithm):
                model: Algorithm = AlgorithmClass
                return dict(
                    model_dir=model_dir,
                    task=model.task,
                    AlgorithmClass=model.__class__,
                    hyperparams=model.hyperparams.dict(),
                )
            if isinstance(AlgorithmClass, str):
                if task is None:
                    raise ValueError(
                        f'If passing `algorithm` as a string (i.e. class name or alias), '
                        f'you must also pass `task`=...'
                    )
                if hyperparams is None:
                    raise ValueError(
                        f'When passing `algorithm` as a string (i.e. class name or alias), '
                        f'you must also pass `hyperparams`=...'
                    )
                AlgorithmClass: Type[Algorithm] = Algorithm.get_subclass((task, AlgorithmClass))
            if not (isinstance(AlgorithmClass, type) and issubclass(AlgorithmClass, Algorithm)):
                raise ValueError(
                    f'When passing `algorithm` as a string (i.e. class name or alias), '
                    f'this must be used with task to lookup a subclass of {Algorithm}; found: {AlgorithmClass}'
                )
            return dict(
                model_dir=model_dir,
                task=task,
                AlgorithmClass=AlgorithmClass,
                hyperparams=hyperparams_to_dict(hyperparams),
            )
        raise ValueError(
            f'Must specify `model_dir` or (`algorithm`, `task`, `hyperparams`); '
            f'found both to be empty.'
        )

    def __init__(self, **kwargs):
        super(Evaluator, self).__init__(**kwargs)

    def initialize(self, **kwargs):
        """
        Should initialize a new evaluator.
        Should not load models into memory.
        """
        pass

    def init_model(self, **kwargs) -> bool:
        """
        Load the model(s) into memory, and save them in the `model` variable.
        """
        if self.model is not None:
            ## If the model is already loaded, do not load it again.
            return False
        self.model: Any = self._load_model(**kwargs)
        if self.cache_timeout is not None:
            def cleanup_model_on_expiry():
                # print('Checking whether timeout has expired...')
                if self.cache_timeout.has_expired and self._evaluator_is_running is False:
                    print(
                        f'{self.class_name} has not been used in {self.cache_timeout.timeout} seconds, '
                        f'cleaning up model(s).',
                        flush=True
                    )
                    self.cleanup_model()
                    stop_daemon(self._cache_timeout_daemon_id, daemons=self._cache_timeout_daemons)
                    self._cache_timeout_daemon_id: Optional[str] = None

            self._cache_timeout_daemon_id: str = start_daemon(
                cleanup_model_on_expiry,
                wait=15,
                daemons=self._cache_timeout_daemons,
            )
            self.cache_timeout.reset_timeout()
        return True

    @abstractmethod
    def _load_model(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def cleanup_model(self):
        """
        Remove the model(s) from memory.
        """
        pass

    def stop(self, **kwargs):
        """
        Should stop a running evaluator, and remove models from memory, and unset the `model` variable.
        """
        self.cleanup_model()

    def __del__(self):
        self.stop()

    @classmethod
    def of(
            cls,
            evaluator: Optional[str] = None,
            *,
            init: bool = True,
            init_model: bool = False,
            model_dir: Optional[Union[FileMetadata, Dict, str]] = None,
            **kwargs,
    ) -> Evaluator:
        if model_dir is not None:
            model_dir: FileMetadata = FileMetadata.of(model_dir)
        if evaluator is not None:
            EvaluatorClass: Type[Evaluator] = Evaluator.get_subclass(evaluator)
        elif 'name' in kwargs:
            EvaluatorClass: Type[Evaluator] = Evaluator.get_subclass(kwargs.pop('name'))
        else:
            EvaluatorClass: Type[Evaluator] = cls
        if EvaluatorClass == Evaluator:
            subclasses: List[str] = random_sample(
                as_list(Evaluator.subclasses),
                n=3,
                replacement=False
            )
            raise ValueError(
                f'"{Evaluator.class_name}" is an abstract class. '
                f'To create an instance, please either pass `evaluator`, '
                f'or call .of(...) on a subclass of {Evaluator.class_name}, e.g. {", ".join(subclasses)}'
            )
        evaluator: Evaluator = EvaluatorClass(
            model_dir=model_dir,
            **kwargs,
        )
        if init:
            evaluator.initialize(**kwargs)
        if init_model:
            evaluator.init_model(**kwargs)
        return evaluator

    def __str__(self) -> str:
        params_dict: Dict = {}
        excuded: Set[str] = set()
        if issubclass(self.AlgorithmClass, Algorithm):
            params_dict['AlgorithmClass'] = self.algorithm_display_name
            excuded.add('AlgorithmClass')
        params_dict: Dict = {**params_dict, **self.dict(exclude=excuded)}
        params_str: str = json.dumps(params_dict, indent=4, cls=NeverFailJsonEncoder)
        out: str = f'{self.class_name} with params:\n{params_str}'
        return out

    def _create_hyperparams(self) -> Optional[Algorithm.Hyperparameters]:
        return self.AlgorithmClass.create_hyperparams(self.hyperparams)

    @property
    def algorithm_display_name(self, hyperparams: bool = True) -> str:
        if isinstance(self.AlgorithmClass, Algorithm):
            return str(self.AlgorithmClass)
        elif issubclass(self.AlgorithmClass, Algorithm):
            out: str = f'{self.AlgorithmClass.class_name}'
            if hyperparams:
                hyperparams_str: str = self._create_hyperparams().json(indent=4)
                out += f'-hyperparams:\n{hyperparams_str}'
            return out
        return str(self.AlgorithmClass)

    @property
    def task_display_name(self) -> str:
        return self.task.display_name() if isinstance(self.task, Task) else self.task.capitalize()

    @safe_validate_arguments
    def evaluate(
            self,
            dataset: Any,
            **kwargs
    ) -> Optional[Union[
        Predictions,
        List[Metric],
        Tuple[Predictions, List[Metric]]
    ]]:
        """
        Run the evaluator, either returning predictions, metrics, or both.
        At least one of `return_predictions` and/or `metrics` must be passed.
        - When passing only `return_predictions`, this function will return a single Predictions object.
        - When passing only `metrics`, this function will return the list of (evaluated) Metric objects.
        - When passing both `return_predictions` and `metrics`, we will return a tuple of (predictions, metrics).

        :param return_predictions: whether to return predictions or not.
            Aliases: 'return_preds', 'preds', 'predictions'
        :param metrics: list of metrics to evaluate. Each metric may be either a string, dict or Metric object.
            Aliases: 'metric', 'metrics_list', 'metric_list'
        :param predictions_destination: where to save intermediate predictions. For large prediction jobs, it might be
            useful to save results to S3 in case of model failure.
             Aliases: 'preds_destination', 'save_predictions', 'save_preds', 'save_preds_to', 'save_to'
        :param: tracker: experiment tracker to use to log evaluation runs. Defaults to no tracker.
            Aliases: 'experiment_tracker', 'experiment', 'experiment_name', 'trial', 'trial_name'
        :param kwargs: extra keyword arguments passed to Evaluator subclass.
        """
        ## Subclasses should override _run_evaluation to actually implement
        Alias.set_metrics(kwargs)
        metrics: Optional[Union[Union[Metric, Dict, str], List[Union[Metric, Dict, str]]]] = kwargs.pop('metrics', None)

        Alias.set_return_predictions(kwargs)
        return_predictions: bool = kwargs.pop('return_predictions', False)

        Alias.set_predictions_destination(kwargs)
        predictions_destination: Optional[Union[io.IOBase, FileMetadata, Dict, str]] = \
            kwargs.pop('predictions_destination', None)

        Alias.set_tracker(kwargs)

        progress_bar: Optional[Dict] = Alias.get_progress_bar(kwargs)

        if metrics is not None:
            metrics: List[Union[Metric, Dict, str]] = as_list(metrics)
            if any_are_none(metrics):
                raise ValueError(f'Empty metric found for {dataset.data_split.name.capitalize()} dataset.')
            metrics: List[Metric] = [Metric.of(metric) for metric in metrics]

        if metrics is None and predictions_destination is None and return_predictions is False:
            raise ValueError(
                f'When calling {self.class_name}.evaluate(...), in addition to the input dataset, you can pass: '
                f'(1) `metrics` (which will be calculated), and/or'
                f'(2) `predictions_destination` (location or iostream where the predictions should be saved), and/or '
                f'(3) `return_predictions=True` (returns the predictions from the function call). '
                f'At the moment, none of these have been passed, and thus model predictions and/or metrics will '
                f'be computed but not returned. Please pass one or more of the above parameters to '
                f'{self.class_name}.evaluate(...)'
            )

        if predictions_destination is not None:
            predictions_destination: FileMetadata = FileMetadata.of(predictions_destination)
            if predictions_destination.format is None:
                raise ValueError(
                    f'When passing `predictions_destination`, you must also pass the file format. '
                    f'This can be done by passing `predictions_destination` as a dict, e.g. '
                    '{"path": predictions_destination, "format": "parquet"}, or a FileMetadata object.'
                )

        if kwargs.get('tracker') is None:  ## Do not track
            kwargs['tracker']: Tracker = Tracker.noop_tracker()
        elif kwargs['tracker'] is False:  ## Do not track
            kwargs['tracker']: Tracker = Tracker.noop_tracker()
        elif isinstance(kwargs['tracker'], str):
            tracker_params: Dict = deepcopy(DEFAULT_TRACKER_PARAMS)
            if Tracker.get_subclass(kwargs['tracker'], raise_error=False) is None:
                ## Usually passed as experiment="..."
                tracker_params['experiment']: str = kwargs['tracker']
            kwargs['tracker']: Dict = tracker_params
        kwargs['tracker']: Tracker = Tracker.of(kwargs['tracker'])

        try:
            self._evaluator_is_running: bool = True  ## Ensures we do not accidentally delete the models while running.
            evaluated_predictions, evaluated_metrics = self._run_evaluation(
                dataset,
                metrics=metrics,
                return_predictions=return_predictions,
                predictions_destination=predictions_destination,
                progress_bar=progress_bar,
                **kwargs
            )
        finally:
            if self.cache_timeout is not None:  ## Rests the timeout
                self.cache_timeout.reset_timeout()
            self._evaluator_is_running: bool = False

        if return_predictions and metrics is not None:
            return evaluated_predictions, evaluated_metrics
        elif metrics is not None:
            return evaluated_metrics
        elif return_predictions:
            return evaluated_predictions
        return None  ## Both are None.

    @abstractmethod
    @safe_validate_arguments
    def _run_evaluation(
            self,
            dataset: Any,
            *,
            tracker: Tracker,
            metrics: Optional[List[Metric]],
            return_predictions: bool,
            predictions_destination: Optional[FileMetadata],
            progress_bar: Optional[Dict],
            **kwargs
    ) -> Tuple[Optional[Predictions], Optional[List[Metric]]]:
        pass

    def download_remote_model_to_cache_dir(
            self,
            cache_dir: Optional[Union[FileMetadata, Dict, str]] = None,
            *,
            force_download: bool = False,
            **kwargs,
    ) -> Optional[FileMetadata]:
        model_dir: Optional[FileMetadata] = self.model_dir
        if model_dir is None:
            return None
        if model_dir.is_path_valid_dir():
            model_dir.mkdir()
        if not model_dir.is_remote_storage():
            return model_dir
        if model_dir.storage is not Storage.S3:
            raise NotImplementedError('Can only download from S3 remote storage at the moment.')
        if not model_dir.is_path_valid_dir():
            raise ValueError(
                f'Expected `model_dir` to be a valid directory; found: "{model_dir.path}"...'
                f'did you forget a "/" at the end?'
            )
        cache_dir: FileMetadata = FileMetadata.of(get_default(cache_dir, self.cache_dir)).mkdir(return_metadata=True)
        ## Use the same path locally as the S3 bucket + key.
        ## E.g. if s3 model dir is "s3://my-bucket/path/to/model/folder/", copy the model into the following
        ## local folder: "<cache_dir>/my-bucket/path/to/model/folder/"
        cached_local_model_dir: FileMetadata = cache_dir.subdir_in_dir(
            path=StringUtil.remove_prefix(model_dir.path, StringUtil.S3_PREFIX),
            mkdir=True,
            return_metadata=True,
        )
        S3Util.copy_s3_dir_to_local(
            source_s3_dir=model_dir.path,
            destination_local_dir=cached_local_model_dir.path,
            force_download=force_download,
            log=True,
        )
        # print(f'cached_local_model_dir: {cached_local_model_dir}')
        return cached_local_model_dir

    @safe_validate_arguments
    def _evaluate_single_model(
            self,
            dataset: Any,
            model: Algorithm,
            metrics: Optional[List[Metric]],
            return_predictions: bool,
            predictions_destination: Optional[FileMetadata],
            **kwargs,
    ) -> Tuple[Optional[Predictions], Optional[List[Metric]], int]:
        """
        Here we both evaluate predictions and metrics, and possibly save predictions to a certain destination.
        """
        Alias.set_predict_batch_size(kwargs, param='batch_size', default=model.hyperparams.batch_size)

        if metrics is not None:
            ## Metrics must be calculated and returned:
            evaluated_predictions, evaluated_metrics, evaluated_num_rows = self._evaluate_metrics(
                dataset=dataset,
                model=model,
                metrics=metrics,
                return_predictions=predictions_destination is not None or return_predictions,
                **kwargs
            )
        else:
            ## We must only calculate predictions:
            evaluated_metrics: Optional[List[Metric]] = None
            evaluated_predictions: Predictions = model.predict(
                dataset,
                yield_partial=False,
                **kwargs
            )
            evaluated_num_rows: int = len(evaluated_predictions)
        save_predictions(
            predictions=evaluated_predictions,
            predictions_destination=predictions_destination,
            **kwargs,
        )
        return evaluated_predictions, evaluated_metrics, evaluated_num_rows

    def _evaluate_metrics(
            self,
            dataset: Any,
            model: Algorithm,
            metrics: Optional[List[Metric]],
            return_predictions: bool,
            **kwargs
    ) -> Tuple[Optional[Predictions], List[Metric], int]:
        all_metrics_are_rolling: bool = np.all([metric.is_rolling for metric in metrics])
        evaluated_num_rows: int = 0
        if all_metrics_are_rolling:
            evaluated_metrics: List[Metric] = [metric.clear() for metric in metrics]
            ## If all metrics support rolling calculation, then call calculate_rolling_metric...this saves memory
            ## as we only need one predictions-batch to be in memory at a time.
            predictions: List[Predictions] = []
            for partial_predictions in model.predict(
                    dataset,
                    yield_partial=True,
                    **kwargs
            ):
                evaluated_num_rows += len(partial_predictions)
                if return_predictions:
                    predictions.append(partial_predictions)
                evaluated_metrics: List[Metric] = [
                    metric.evaluate(partial_predictions, rolling=True)
                    for metric in evaluated_metrics
                ]
            if return_predictions:
                predictions: Predictions = Predictions.concat(predictions)
        else:
            predictions: Predictions = model.predict(
                dataset,
                yield_partial=False,
                **kwargs
            )
            evaluated_num_rows += len(predictions)
            evaluated_metrics: List[Metric] = [
                self._evaluate_metric(
                    predictions=predictions,
                    metric=metric,
                )
                for metric in metrics
            ]
        if return_predictions:
            return predictions, evaluated_metrics, evaluated_num_rows
        return None, evaluated_metrics, evaluated_num_rows

    @staticmethod
    def _evaluate_metric(
            predictions: Predictions,
            metric: Metric,
    ):
        try:
            return metric.evaluate(predictions, inplace=False)
        except Exception as e:
            Log.error(format_exception_msg(e))
            Log.warning(
                f'\nError, please see stack trace above. Could not calculate metric for "{metric.display_name}"'
                f'\nThis metric will not be included in the output. Calculating other metrics...'
            )

    @abstractmethod
    def _evaluate_start_msg(self, **kwargs) -> str:
        pass

    @abstractmethod
    def _evaluate_end_msg(self, **kwargs) -> str:
        pass

    @classmethod
    def _cur_batch_str(cls, batch_i: int, batches: int, prefix: str = 'step') -> str:
        batch_num_str, batches_str = cls._pad_batch_strs(batch_i=batch_i, batches=batches)
        return f'{prefix.strip()} [{batch_num_str}/{batches_str}]'

    @classmethod
    def _pad_batch_strs(cls, batch_i: int, batches: int) -> Tuple[str, str]:
        batch_num_str: str = StringUtil.pad_zeros(batch_i + 1, batches)  ## batch_i is in [0, num_batches], inclusive.
        batches_str: str = StringUtil.pad_zeros(batches, batches)
        return batch_num_str, batches_str
