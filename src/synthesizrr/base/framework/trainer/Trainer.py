from typing import *
from types import SimpleNamespace
import time, json, math
from copy import deepcopy
from abc import abstractmethod, ABC
from synthesizrr.base.data import FileMetadata
from synthesizrr.base.constants import MLType, DataSplit, Task, Alias
from synthesizrr.base.framework.algorithm import Algorithm, TaskOrStr
from synthesizrr.base.framework.metric import CountingMetric, Metric, Metrics
from synthesizrr.base.framework.task_data import Dataset, Datasets, save_dataset
from synthesizrr.base.framework.predictions import Predictions, save_predictions
from synthesizrr.base.framework.tracker import Tracker, DEFAULT_TRACKER_PARAMS
from synthesizrr.base.util import Parameters, Registry, FractionalBool, random_sample, safe_validate_arguments, StringUtil, \
    any_are_not_none, any_are_none, is_function, as_list, get_default, set_param_from_alias, classproperty, \
    optional_dependency, all_are_not_none, all_are_none, Timer, type_str, format_exception_msg
from pydantic import root_validator, Extra, conint
from pydantic.typing import Literal

Trainer = "Trainer"
TrainerSubclass = TypeVar("TrainerSubclass", bound="Trainer")


## Trainer metrics:
class RowCount(CountingMetric):
    aliases = ['num_rows', 'row_count']
    _num_rows: int = 0

    def update(self, data: Union[Dataset, Predictions]):
        data.check_in_memory()
        self._num_rows += len(data)

    def compute(self) -> int:
        return self._num_rows


class SaveDatasetOrPredictions(Metric):
    aliases = ['save-preds', 'save-predictions', 'save-dataset', 'save-data', 'save']

    class Params(Metric.Params):
        destination: Optional[Union[FileMetadata, Dict, str]]
        overwrite: bool = True

    def compute_only(self, data: Union[Dataset, Predictions]) -> Any:
        try:
            if self.params.destination is not None:
                if isinstance(data, Dataset):
                    save_dataset(
                        dataset=data,
                        dataset_destination=self.params.destination,
                        overwrite=self.params.overwrite,
                    )
                elif isinstance(data, Predictions):
                    save_predictions(
                        predictions=data,
                        predictions_destination=self.params.destination,
                        overwrite=self.params.overwrite,
                    )
                else:
                    raise NotImplementedError(f'Cannot save object of type: {type_str(data)}')
            return data
        except Exception as e:
            return format_exception_msg(e, short=True, prefix='Failed')


class CompressPredictions(Metric):  ## Hack to get predictions out of Ray Tune.
    aliases = ['compressed_predictions', 'compress']

    def compute_only(self, data: Any) -> Any:
        if not isinstance(data, Predictions):
            raise NotImplementedError(f'Can only compress {Predictions}')
        data_params: Dict = data.dict(exclude={'data'})
        data_params['data']: Dict = data.data.compress(base64_encoding=True).dict()
        return StringUtil.jsonify(data_params)


class KFold(Parameters):
    num_folds: conint(ge=1, le=26)
    prefix: str = 'fold_'
    names: Literal['alphabet', 'numbers'] = 'alphabet'
    seed: Optional[int] = None

    @classproperty
    def NO_TUNING_K_FOLD(cls):
        return KFold(num_folds=1)

    def fold_names(self) -> List[str]:
        fold_names = []
        for i in range(0, self.num_folds):
            if self.names == 'alphabet':
                fold_names.append(f'{self.prefix}{chr(i + 65)}')
            elif self.names == 'numbers':
                fold_names.append(f'{self.prefix}{i + 1}')
            else:
                raise ValueError(f'Unsupported value for `names`: {self.names}')
        return fold_names


def _remaining_num_steps(*, steps: int, step_num: int) -> int:
    remaining_num_steps: int = steps - step_num + 1  ## step_num is in [1, steps] inclusive
    remaining_num_steps: int = max(remaining_num_steps, 0)
    return remaining_num_steps


def _steps_this_iter(*, steps: int, step_num: int, steps_increment: int) -> int:
    remaining_num_steps: int = _remaining_num_steps(steps=steps, step_num=step_num)
    return min(steps_increment, remaining_num_steps)


def _step_num_after_iter(*, steps: int, step_num: int, steps_increment: int) -> int:
    return min(step_num + steps_increment - 1, steps)


class Trainer(Parameters, Registry, ABC):
    _allow_multiple_subclasses: ClassVar[bool] = False  ## Rejects multiple subclasses registered to the same name.
    _allow_subclass_override: ClassVar[bool] = True  ## Allows replacement of subclass with same name.
    trainer_metrics: ClassVar[Tuple[str, ...]] = ('row_count',)

    class Config(Parameters.Config):
        extra = Extra.ignore

    class RunConfig(Parameters):
        class Config(Parameters.Config):
            extra = Extra.allow

    AlgorithmClass: Union[Type[Algorithm], Algorithm, str]
    task: TaskOrStr = None
    hyperparams: Dict = {}
    run_config: RunConfig = {}
    seed: Optional[int] = None
    k_fold: Optional[KFold] = None
    stats_batch_size: conint(ge=1) = int(1e4)
    eval_batch_size: Optional[conint(ge=1)] = None
    shuffle: Optional[bool] = True
    validate_inputs: Optional[FractionalBool] = None
    validate_outputs: Optional[FractionalBool] = None
    metric_eval_frequency: Optional[conint(ge=1)] = 1  ## None = Disabled
    checkpoint_frequency: conint(ge=0) = 0  ## 0 = Disabled
    evaluate_test_metrics: Literal['at_end', 'each_iter'] = 'at_end'
    custom_definitions: Tuple[Any, ...] = tuple()  ## NOTE! Pydantic complains is you make the typing Callable here.
    ## Logging verbosity. 0 = zero logging, 1 = Basic logging, 2 = verbose logging.
    verbosity: conint(ge=0) = 1

    @root_validator(pre=True)
    def _set_trainer_params(cls, params: Dict):
        set_param_from_alias(params, param='AlgorithmClass', alias=['algorithm'])
        set_param_from_alias(params, param='eval_batch_size', alias=['predict_batch_size'])
        set_param_from_alias(params, param='seed', alias=['random_state', 'random_seed', 'random_state_seed'])
        set_param_from_alias(params, param='k_fold', alias=['kfold', 'num_folds'])
        set_param_from_alias(params, param='custom_definitions', alias=['udfs', 'custom', 'custom_classes'])
        set_param_from_alias(params, param='metric_eval_frequency', alias=[
            'eval_freq', 'eval_frequency', 'metric_eval_freq', 'eval_epochs',
            'steps_eval_frequency', 'steps_eval_freq', 'eval_steps', 'steps_eval'
        ])
        set_param_from_alias(params, param='evaluate_test_metrics', alias=[
            'eval_test_metrics', 'evaluate_test', 'evaluate_test_metrics', 'test_eval',
        ])
        set_param_from_alias(params, param='verbosity', alias=['verbose'])

        ## Ensure user-defined classes are registered on the driver (i.e. Jupyter/IPython).
        custom_definitions: List[Any] = as_list(get_default(
            params.get('custom_definitions'),
            tuple(),
        ))
        for custom_fn in custom_definitions:
            if not is_function(custom_fn):
                raise ValueError(
                    f'`custom_definitions` must be a function in which you include the entire source code of any '
                    f'custom classes (including imports!). '
                    f'Found object of type: {type(params["custom_definitions"])}'
                )
            custom_fn()
        params['custom_definitions'] = tuple(custom_definitions)

        if 'batch_size' in params or 'train_batch_size' in params:
            raise ValueError(
                f'To set the training batch size, please set `batch_size` in the `hyperparams` dict, '
                f'instead of passing it to {cls.class_name} directly.'
            )

        ## Set Algorithm class
        AlgorithmClass = params['AlgorithmClass']
        if isinstance(AlgorithmClass, Algorithm):
            ## If it is a model, extract the AlgorithmClass, task and hyperparams:
            params['AlgorithmClass']: Type[Algorithm] = AlgorithmClass.__class__
            params['hyperparams']: Dict = AlgorithmClass.hyperparams.dict()
            params['task']: TaskOrStr = AlgorithmClass.task
        elif isinstance(AlgorithmClass, str):
            if params.get('task') is None:
                raise ValueError(
                    f'If passing `algorithm` as a string (i.e. class name or alias), '
                    f'you must also pass `task`=...'
                )
            params['AlgorithmClass']: Type[Algorithm] = Algorithm.get_subclass((params['task'], AlgorithmClass))
        assert issubclass(params['AlgorithmClass'], Algorithm)

        ## This allows us to create a new Trainer instance without specifying `run_config`.
        ## If it is specified, we will pick cls.RunConfig, which can be overridden by the subclass.
        params['run_config'] = cls._convert_params(cls.RunConfig, params.get('run_config', {}))
        if not isinstance(params['run_config'], Trainer.RunConfig):
            raise ValueError(
                f'Custom run_config class does not inherit from the base class version. '
                f'Please ensure your custom class for run_config is called "{cls.RunConfig.class_name}" '
                f'and inherits from "{cls.RunConfig.class_name}" defined in the base class.'
            )

        if params.get('device') is not None:
            raise ValueError(
                f'Do not pass "device" to {Trainer.class_name}.of(), '
                f'instead pass it as: {Trainer.class_name}.train(device=...)'
            )

        ## Set KFold:
        k_fold = params.get('k_fold', None)
        if k_fold is None:
            params['k_fold'] = KFold.NO_TUNING_K_FOLD
        elif isinstance(k_fold, int):
            params['k_fold'] = KFold(num_folds=k_fold)
        elif isinstance(k_fold, dict):
            params['k_fold'] = KFold(**k_fold)

        return params

    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)
        self.initialize(**kwargs)

    @abstractmethod
    def initialize(self, **kwargs):
        """
        Should initialize a new trainer.
        """
        pass

    @classmethod
    def of(
            cls,
            trainer: Optional[str] = None,
            **kwargs,
    ) -> TrainerSubclass:
        if trainer is not None:
            TrainerClass: Type[Trainer] = Trainer.get_subclass(trainer)
        elif 'name' in kwargs:
            TrainerClass: Type[Trainer] = Trainer.get_subclass(kwargs.pop('name'))
        else:
            TrainerClass: Type[Trainer] = cls
        if TrainerClass == Trainer:
            subclasses: List[str] = random_sample(
                as_list(Trainer.subclasses),
                n=3,
                replacement=False
            )
            raise ValueError(
                f'"{Trainer.class_name}" is an abstract class. '
                f'To create an instance, please either pass `trainer`, '
                f'or call .of(...) on a subclass of {Trainer.class_name}, e.g. {", ".join(subclasses)}'
            )
        return TrainerClass(**kwargs)

    def __str__(self) -> str:
        params_dict: Dict = {}
        excuded: Set[str] = set()
        if issubclass(self.AlgorithmClass, Algorithm):
            params_dict['AlgorithmClass'] = self.algorithm_display_name
            excuded.add('AlgorithmClass')
        params_dict: Dict = {**params_dict, **self.dict(exclude=excuded)}
        params_str: str = json.dumps(params_dict, indent=4)
        out: str = f'{self.class_name} with params:\n{params_str}'
        return out

    def _create_hyperparams(self, hyperparams: Optional[Dict[str, Any]] = None) -> Algorithm.Hyperparameters:
        if hyperparams is None:
            hyperparams: Dict[str, Any] = self.hyperparams
        if self.seed is not None:  ## Override hyperparams seed with global seed.
            hyperparams['seed'] = self.seed
        return self.AlgorithmClass.create_hyperparams(hyperparams)

    @property
    def algorithm_display_name(self) -> str:
        if isinstance(self.AlgorithmClass, Algorithm):
            return str(self.AlgorithmClass)
        elif issubclass(self.AlgorithmClass, Algorithm):
            hyperparams_str: str = self._create_hyperparams().json(indent=4)
            return f'{self.AlgorithmClass.class_name} with hyperparams:\n{hyperparams_str}'
        return str(self.AlgorithmClass)

    @property
    def task_display_name(self) -> str:
        return self.task.display_name() if isinstance(self.task, Task) else self.task.capitalize()

    @safe_validate_arguments
    def train(
            self,
            datasets: Datasets,
            metrics: Optional[Metrics] = None,
            **kwargs
    ):
        kwargs.setdefault('task', self.task)

        progress_bar: Optional[Dict] = Alias.get_progress_bar(kwargs)

        Alias.set_load_model(kwargs)
        load_model: Optional[Union[FileMetadata, str]] = kwargs.get('load_model')
        if load_model is not None:
            load_model: FileMetadata = FileMetadata.of(load_model)
            assert isinstance(load_model, FileMetadata)
            if load_model.is_path_valid_dir() is False:
                raise ValueError(
                    f'`load_model` must be a valid directory path; '
                    f'found following {load_model.storage} path: "{load_model.path}"'
                )
            kwargs['load_model'] = load_model

        Alias.set_save_model(kwargs)
        save_model: Optional[Union[FileMetadata, str]] = kwargs.get('save_model')
        if save_model is not None:
            save_model: FileMetadata = FileMetadata.of(save_model)
            assert isinstance(save_model, FileMetadata)
            if save_model.is_path_valid_dir() is False:
                raise ValueError(
                    f'`save_model` must be a valid directory path; '
                    f'found following {save_model.storage} path: "{save_model.path}"'
                )
            kwargs['save_model'] = save_model

        Alias.set_tracker(kwargs)
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

        metrics: Optional[Metrics] = self._create_trainer_metrics(metrics)
        out = self._run_training(
            datasets=datasets,
            metrics=metrics,
            progress_bar=progress_bar,
            **kwargs
        )
        return out

    @abstractmethod
    @safe_validate_arguments
    def _run_training(
            self,
            datasets: Datasets,
            *,
            tracker: Tracker,
            progress_bar: Optional[Dict],
            metrics: Optional[Metrics] = None,
            load_model: Optional[FileMetadata] = None,
            save_model: Optional[FileMetadata] = None,
            **kwargs
    ):
        pass

    def _create_trainer_metrics(self, metrics: Optional[Metrics]) -> Optional[Metrics]:
        if metrics is None:
            return None
        trainer_metrics: Dict[DataSplit, List[Metric]] = {}
        for data_split, dataset_metrics in metrics.metrics.items():
            dataset_metrics_unique: Dict[str, Metric] = {}
            for dataset_metric in dataset_metrics:
                assert isinstance(dataset_metric, Metric)
                dataset_metrics_unique[dataset_metric.display_name] = dataset_metric
            for trainer_metric in self.trainer_metrics:
                trainer_metric: Metric = Metric.of(trainer_metric)
                if trainer_metric.display_name not in dataset_metrics_unique:
                    dataset_metrics_unique[trainer_metric.display_name] = trainer_metric.clear()
            trainer_metrics[data_split] = list(dataset_metrics_unique.values())
        return Metrics.of(**trainer_metrics)

    @safe_validate_arguments
    def _extract_datasets_and_metrics(
            self,
            datasets: Datasets,
            metrics: Optional[Metrics],
    ) -> Tuple[
        Dataset, Optional[List[Metric]],
        Optional[Dataset], Optional[List[Metric]],
        Optional[Dataset], Optional[List[Metric]],
    ]:
        train_dataset: Dataset = datasets[DataSplit.TRAIN]
        validation_dataset: Optional[Dataset] = datasets[DataSplit.VALIDATION]
        test_dataset: Optional[Dataset] = datasets[DataSplit.TEST]
        train_metrics: Optional[List[Metric]] = None
        validation_metrics: Optional[List[Metric]] = None
        test_metrics: Optional[List[Metric]] = None
        if metrics is not None:
            train_metrics: Optional[List[Metric]] = metrics[DataSplit.TRAIN]
            validation_metrics: Optional[List[Metric]] = metrics[DataSplit.VALIDATION]
            test_metrics: Optional[List[Metric]] = metrics[DataSplit.TEST]
        return train_dataset, train_metrics, \
               validation_dataset, validation_metrics, \
               test_dataset, test_metrics

    @safe_validate_arguments
    def _train_single_model(
            self,
            model: Algorithm,
            epochs: Optional[conint(ge=1)],
            steps: Optional[conint(ge=1)],
            train_dataset: Any,
            train_dataset_length: int,
            logger: Optional[Callable],
            progress_bar: Optional[Dict],
            **kwargs
    ):
        if all_are_not_none(epochs, steps):
            raise ValueError(f'Must pass at most one of `epochs` and `steps`; both were passed.')
        elif all_are_none(epochs, steps):
            raise ValueError(f'Cannot train model without passing `epochs` or `steps` in hyperparams.')
        if epochs is not None:
            for epoch_num in range(1, epochs + 1):
                self._train_epoch(
                    model=model,
                    train_dataset=train_dataset,
                    train_dataset_length=train_dataset_length,
                    epochs=epochs,
                    epoch_num=epoch_num,
                    logger=logger,
                    progress_bar=progress_bar,
                    **kwargs
                )
        elif steps is not None:
            if self.metric_eval_frequency == 1:
                raise ValueError(
                    f'Cannot use eval_steps=1 when using `steps`; this will result in calculating metrics '
                    f'every single step. Please set `eval_steps` to a higher value.'
                )
            for step_num in range(1, steps + 1, self.metric_eval_frequency):
                self._train_steps_iter(
                    model=model,
                    train_dataset=train_dataset,
                    train_dataset_length=train_dataset_length,
                    steps=steps,
                    step_num=step_num,
                    logger=logger,
                    progress_bar=progress_bar,
                    **kwargs
                )
        model.post_train_cleanup()

    def _train_epoch(
            self,
            model: Algorithm,
            train_dataset: Any,
            logger: Optional[Callable],
            epochs: int,
            epoch_num: int,
            progress_bar: Optional[Dict] = None,
            **kwargs
    ):
        epoch_timer: Timer = Timer(silent=True)
        epoch_timer.start()

        batch_size: Optional[int] = model.hyperparams.batch_size
        if batch_size is None:
            raise ValueError(f'Cannot train model without passing `batch_size` in hyperparams.')

        cur_epoch_str: str = self._cur_epoch_str(epoch_num=epoch_num, epochs=epochs)

        ## Run training phase:
        training_timer: Timer = Timer(silent=True)
        training_timer.start()
        if isinstance(progress_bar, dict):
            progress_bar['desc'] = f'Training {cur_epoch_str}'
        if self.verbosity >= 1 and progress_bar is None:
            logger(f'>> Running {cur_epoch_str}...')
        self._train_loop(
            model=model,
            batch_size=batch_size,
            train_dataset=train_dataset,
            epoch_num=epoch_num,
            epochs=epochs,
            step_num=None,
            steps=None,
            progress_bar=progress_bar,
            **kwargs
        )
        training_timer.stop()
        if self.verbosity >= 2 and progress_bar is None:
            logger(f'...training phase of {cur_epoch_str} took {training_timer.time_taken_str}.')

        ## Run metric-evaluation phase:
        metric_eval_timer: Timer = Timer(silent=True)
        metric_eval_timer.start()
        train_metrics_evaluated, validation_metrics_evaluated, test_metrics_evaluated = self._metric_eval_phase(
            model=model,
            train_dataset=train_dataset,
            logger=logger,
            epoch_num=epoch_num,
            epochs=epochs,
            step_num=None,
            steps=None,
            progress_bar=progress_bar,
            **kwargs
        )
        metric_eval_timer.stop()
        if any_are_not_none(train_metrics_evaluated, validation_metrics_evaluated, test_metrics_evaluated):
            if self.verbosity >= 2 and progress_bar is None:
                logger(f'...evaluation phase of {cur_epoch_str} took {metric_eval_timer.time_taken_str}.')

        epoch_timer.stop()
        if self.verbosity >= 1:
            logger(f'...{cur_epoch_str} took {epoch_timer.time_taken_str}.')
            logger('')

    def _train_steps_iter(
            self,
            model: Algorithm,
            train_dataset: Any,
            logger: Optional[Callable],
            steps: int,
            step_num: int,
            train_dataset_length: int,
            progress_bar: Optional[Dict] = None,
            **kwargs
    ):
        steps_iter_timer: Timer = Timer(silent=True)
        steps_iter_timer.start()

        step_num: int = min(step_num, steps)
        batch_size: Optional[int] = model.hyperparams.batch_size
        if batch_size is None:
            raise ValueError(f'Cannot train model without passing `batch_size` in hyperparams.')

        cur_step_str_with_increment: str = self._cur_step_str_with_increment(
            step_num=step_num,
            steps=steps,
            steps_increment=self.metric_eval_frequency,
        )
        cur_step_incremented_str: str = self._cur_step_str(
            step_num=step_num,
            steps=steps,
            steps_increment=self.metric_eval_frequency,
        )

        ## Run training phase:
        training_timer: Timer = Timer(silent=True)
        training_timer.start()
        if isinstance(progress_bar, dict):
            progress_bar['desc'] = f'Training {cur_step_str_with_increment}'
            progress_bar['total']: int = Dataset._steps_iter_adjust_progress_bar_num_rows(
                batch_size=batch_size,
                steps=_steps_this_iter(
                    steps=steps,
                    step_num=step_num,
                    steps_increment=self.metric_eval_frequency,
                ),
                dataset_length=train_dataset_length,
            )
        if self.verbosity >= 1 and progress_bar is None:
            logger(f'>> Running {cur_step_str_with_increment}...')
        self._train_loop(
            model=model,
            batch_size=batch_size,
            train_dataset=train_dataset,
            train_dataset_length=train_dataset_length,
            step_num=step_num,
            steps=steps,
            epoch_num=None,
            epochs=None,
            progress_bar=progress_bar,
            **kwargs
        )
        training_timer.stop()
        if self.verbosity >= 2 and progress_bar is None:
            logger(f'...training phase of {cur_step_str_with_increment} took {training_timer.time_taken_str}.')

        ## Run metric-evaluation phase:
        metric_eval_timer: Timer = Timer(silent=True)
        metric_eval_timer.start()
        train_metrics_evaluated, validation_metrics_evaluated, test_metrics_evaluated = self._metric_eval_phase(
            model=model,
            train_dataset=train_dataset,
            train_dataset_length=train_dataset_length,
            logger=logger,
            epoch_num=None,
            epochs=None,
            step_num=step_num,
            steps=steps,
            progress_bar=progress_bar,
            **kwargs
        )
        metric_eval_timer.stop()
        if any_are_not_none(train_metrics_evaluated, validation_metrics_evaluated, test_metrics_evaluated):
            if self.verbosity >= 2 and progress_bar is None:
                logger(f'...evaluation phase of {cur_step_incremented_str} took {metric_eval_timer.time_taken_str}.')

        steps_iter_timer.stop()
        if self.verbosity >= 1:
            logger(f'...{cur_step_str_with_increment} took {steps_iter_timer.time_taken_str}.')
            logger('')

    @classmethod
    def is_final_iter(
            cls,
            *,
            epoch_num: Optional[int],
            epochs: Optional[int],
            step_num: Optional[int],
            steps: Optional[int],
            steps_increment: Optional[int],
    ) -> bool:
        if epochs is not None:
            return epochs == epoch_num
        elif steps is not None:
            return steps == _step_num_after_iter(steps=steps, step_num=step_num, steps_increment=steps_increment)
        raise ValueError(f'Exactly one of `epochs` and `steps` must be non-None.')

    def _metric_eval_phase(
            self,
            model: Algorithm,
            train_dataset: Optional[Any],
            train_metrics: Optional[List[Metric]],
            validation_dataset: Optional[Any],
            validation_metrics: Optional[List[Metric]],
            test_dataset: Optional[Any],
            test_metrics: Optional[List[Metric]],
            logger: Optional[Callable],
            epoch_num: Optional[int],
            epochs: Optional[int],
            step_num: Optional[int],
            steps: Optional[int],
            progress_bar: Optional[Dict],
            **kwargs
    ) -> Tuple[Optional[List[Metric]], Optional[List[Metric]], Optional[List[Metric]]]:
        if epochs is not None:
            cur_epoch_str: str = self._cur_epoch_str(epoch_num=epoch_num, epochs=epochs)
            if isinstance(progress_bar, dict):
                progress_bar_desc: Dict = {
                    DataSplit.TRAIN: f'Evaluating {DataSplit.TRAIN.capitalize()} '
                                     f'dataset after {cur_epoch_str}',
                    DataSplit.VALIDATION: f'Evaluating {DataSplit.VALIDATION.capitalize()} '
                                          f'dataset after {cur_epoch_str}',
                    DataSplit.TEST: f'Evaluating {DataSplit.TEST.capitalize()} dataset after {epochs} epochs',
                }
            log_metrics_prefix: Dict = {
                DataSplit.TRAIN: f'Metric value after {cur_epoch_str}: '
                                 f'{DataSplit.TRAIN.name.capitalize()} ',
                DataSplit.VALIDATION: f'Metric value after {cur_epoch_str}: '
                                      f'{DataSplit.VALIDATION.name.capitalize()} ',
                DataSplit.TEST: f'Metric value after training for {epochs} epochs: '
                                f'{DataSplit.TEST.name.capitalize()} ',
            }
        elif steps is not None:
            cur_step_incremented_str: str = self._cur_step_str(
                step_num=step_num,
                steps=steps,
                steps_increment=self.metric_eval_frequency,
            )
            if isinstance(progress_bar, dict):
                progress_bar_desc: Dict = {
                    DataSplit.TRAIN: f'Evaluating {DataSplit.TRAIN.capitalize()} '
                                     f'dataset after {cur_step_incremented_str}',
                    DataSplit.VALIDATION: f'Evaluating {DataSplit.VALIDATION.capitalize()} '
                                          f'dataset after {cur_step_incremented_str}',
                    DataSplit.TEST: f'Evaluating {DataSplit.TEST.capitalize()} dataset after {steps} steps',

                }
            log_metrics_prefix: Dict = {
                DataSplit.TRAIN: f'Metric value after {cur_step_incremented_str}: '
                                 f'{DataSplit.TRAIN.name.capitalize()} ',
                DataSplit.VALIDATION: f'Metric value after {cur_step_incremented_str}: '
                                      f'{DataSplit.VALIDATION.name.capitalize()} ',
                DataSplit.TEST: f'Metric value after training for {steps} steps: '
                                f'{DataSplit.TEST.name.capitalize()} ',
            }
        else:
            raise ValueError(f'Exactly one of `epochs` or `steps` must be non-None.')

        if isinstance(progress_bar, dict):
            progress_bar['desc']: str = progress_bar_desc[DataSplit.TRAIN]
            progress_bar['total']: int = 1
        train_metrics_evaluated: Optional[List[Metric]] = self._evaluate_metrics(
            model=model,
            dataset=train_dataset,
            metrics=train_metrics,
            epoch_num=epoch_num,
            step_num=step_num,
            progress_bar=progress_bar,
            **kwargs
        )
        if self.verbosity >= 1:
            self._log_metrics(
                prefix=log_metrics_prefix[DataSplit.TRAIN],
                metrics=train_metrics_evaluated,
                logger=logger,
                **kwargs
            )

        if isinstance(progress_bar, dict):
            progress_bar['desc']: str = progress_bar_desc[DataSplit.VALIDATION]
            progress_bar['total']: int = 1
        validation_metrics_evaluated: Optional[List[Metric]] = self._evaluate_metrics(
            model=model,
            dataset=validation_dataset,
            metrics=validation_metrics,
            epoch_num=epoch_num,
            step_num=step_num,
            progress_bar=progress_bar,
            **kwargs
        )
        if self.verbosity >= 1:
            self._log_metrics(
                prefix=log_metrics_prefix[DataSplit.VALIDATION],
                metrics=validation_metrics_evaluated,
                logger=logger,
                **kwargs
            )

        test_metrics_evaluated: Optional[List[Metric]] = None
        if self.is_final_iter(
                epoch_num=epoch_num,
                epochs=epochs,
                step_num=step_num,
                steps=steps,
                steps_increment=self.metric_eval_frequency,
        ) or self.evaluate_test_metrics == 'each_iter':
            if isinstance(progress_bar, dict):
                progress_bar['desc']: str = progress_bar_desc[DataSplit.TEST]
                progress_bar['total'] = 1
            test_metrics_evaluated: Optional[List[Metric]] = self._evaluate_metrics(
                model=model,
                dataset=test_dataset,
                metrics=test_metrics,
                epoch_num=epochs,
                step_num=steps,
                force=True,
                progress_bar=progress_bar,
                **kwargs
            )
            if self.verbosity >= 1:
                self._log_metrics(
                    prefix=log_metrics_prefix[DataSplit.TEST],
                    metrics=test_metrics_evaluated,
                    logger=logger,
                    **kwargs
                )
        return train_metrics_evaluated, validation_metrics_evaluated, test_metrics_evaluated

    def _train_loop(
            self,
            model: Algorithm,
            train_dataset: Any,
            train_dataset_length: int,  ## Number of rows in training dataset.
            epochs: Optional[int],
            epoch_num: Optional[int],
            steps: Optional[int],
            step_num: Optional[int],
            batch_size: int,
            batch_logger: Optional[Callable] = None,
            **kwargs
    ):
        start: float = time.perf_counter()
        ## Priority of seed: .train() > global trainer seed > model seed
        kwargs['seed']: Optional[int] = get_default(kwargs.get('seed'), self.seed, model.hyperparams.seed)
        if all_are_not_none(steps, step_num):  ## step_num is in [1, steps] inclusive
            remaining_num_steps: int = _remaining_num_steps(steps=steps, step_num=step_num)
            if remaining_num_steps == 0:
                return
            kwargs['steps'] = _steps_this_iter(
                steps=steps,
                step_num=step_num,
                steps_increment=self.metric_eval_frequency,
            )
        else:
            kwargs['steps'] = None
        for batch_i, train_step_metrics in enumerate(model.train_iter(
                train_dataset,
                batch_size=batch_size,
                dataset_length=train_dataset_length,
                validate_inputs=self.validate_inputs,
                shuffle=self.shuffle,
                **kwargs,
        )):
            end: float = time.perf_counter()
            trained_batch_size: Optional[int] = train_step_metrics.pop('batch_size', None)
            trained_time_s: float = end - start
            batches: int = math.ceil(train_dataset_length / batch_size)  ## Total number of training set batches
            if train_step_metrics is not None and batch_logger is not None:
                out: str = self._train_step_message(
                    epoch_num=epoch_num,
                    epochs=epochs,
                    step_num=step_num,
                    steps=steps,
                    batch_i=batch_i,
                    batches=batches,
                    batch_size=batch_size,
                    train_dataset_length=train_dataset_length,
                    trained_batch_size=trained_batch_size,
                    start=start,
                    end=end,
                    train_step_metrics=train_step_metrics,
                )
                batch_logger(out)
            start: float = time.perf_counter()

    def _evaluate_metrics(
            self,
            model: Algorithm,
            dataset: Optional[Any],
            metrics: Optional[List[Metric]],
            epoch_num: Optional[int],
            step_num: Optional[int],
            eval_batch_size: Optional[int] = None,
            force: bool = False,
            **kwargs
    ) -> Optional[List[Metric]]:
        if self.metric_eval_frequency is None:
            return None
        if not force:
            if all_are_not_none(epoch_num, step_num):
                raise ValueError(f'Must pass at most one of `epoch_num` and `step_num`; both were passed.')
            elif all_are_none(epoch_num, step_num):
                raise ValueError(f'Cannot evaluate model metrics without passing either `epoch_num` or `step_num`.')
            if epoch_num is not None and ((epoch_num - 1) % self.metric_eval_frequency) != 0:
                return None
            elif step_num is not None and ((step_num - 1) % self.metric_eval_frequency) != 0:
                return None
        if any_are_none(dataset, metrics):
            return None
        kwargs.pop('batch_size', None)  ## Remove the training batch size.
        kwargs.pop('epochs', None)  ## Remove epochs
        kwargs.pop('steps', None)  ## Remove steps
        eval_batch_size: Optional[int] = get_default(
            eval_batch_size,
            self.eval_batch_size,
            model.hyperparams.batch_size,
        )
        evaluated_metrics: List[Metric] = model.evaluate(
            dataset=dataset,
            metrics=metrics,
            batch_size=eval_batch_size,
            validate_inputs=self.validate_inputs,
            validate_outputs=self.validate_outputs,
            **kwargs
        )
        return evaluated_metrics

    def _log_metrics(
            self,
            prefix: str,
            metrics: Optional[List[Optional[Metric]]],
            logger: Callable,
            **kwargs
    ):
        if metrics is None:
            return
        for metric in metrics:
            if metric is None:
                continue
            out_str: str = prefix + str(metric)
            logger(out_str)

    def _train_step_message(
            self,
            epoch_num: Optional[int],
            epochs: Optional[int],
            step_num: Optional[int],
            steps: Optional[int],
            batch_i: int,
            batches: int,
            batch_size: int,
            train_dataset_length: int,
            trained_batch_size: int,
            start: float,
            end: float,
            train_step_metrics: Dict,
            **kwargs,
    ) -> str:
        if all_are_not_none(epoch_num, epochs):
            cur_epoch_str: str = self._cur_epoch_str(epochs=epochs, epoch_num=epoch_num)
            cur_batch_str: str = self._cur_batch_str(batch_i=batch_i, batches=batches)

            out = f'{DataSplit.TRAIN.name.capitalize()} ' \
                  f'{cur_epoch_str}, ' \
                  f'{cur_batch_str} ' \
                  f'took {StringUtil.readable_seconds(end - start)}.'
        elif all_are_not_none(step_num, steps):
            cur_step_str: str = self._cur_step_str(
                steps=steps,
                step_num=step_num + batch_i,
            )
            out = f'{DataSplit.TRAIN.name.capitalize()} ' \
                  f'{cur_step_str} ' \
                  f'took {StringUtil.readable_seconds(end - start)}.'
        if len(train_step_metrics) > 0:
            out += f' {StringUtil.stringify(train_step_metrics)}'
        return out

    @abstractmethod
    def _train_start_msg(self, **kwargs) -> str:
        pass

    @abstractmethod
    def _train_end_msg(self, **kwargs) -> str:
        pass

    @classmethod
    def _cur_epoch_str(cls, epoch_num: int, epochs: int, prefix: str = 'epoch') -> str:
        epoch_num_str, epochs_str = cls._pad_epoch_strs(epoch_num=epoch_num, epochs=epochs)
        return f'{prefix.strip()} [{epoch_num_str}/{epochs_str}]'

    @classmethod
    def _pad_epoch_strs(cls, epoch_num: int, epochs: int) -> Tuple[str, str]:
        epoch_num_str: str = StringUtil.pad_zeros(epoch_num, epochs)  ## epoch_num is in [1, epochs], inclusive.
        epochs_str: str = StringUtil.pad_zeros(epochs, epochs)
        return epoch_num_str, epochs_str

    @classmethod
    def _cur_step_str(
            cls,
            step_num: int,
            steps: int,
            steps_increment: Optional[int] = None,
            prefix: str = 'step',
    ) -> str:
        step_num_str, steps_str, step_num_increment_str = cls._pad_step_strs(
            step_num=step_num,
            steps=steps,
            steps_increment=steps_increment,
        )
        if step_num_increment_str is None:
            return f'{prefix.strip()} [{step_num_str}/{steps_str}]'
        else:
            return f'{prefix.strip()} [{step_num_increment_str}/{steps_str}]'

    @classmethod
    def _cur_step_str_with_increment(cls, step_num: int, steps: int, steps_increment: int, prefix: str = 'step') -> str:
        step_num_str, steps_str, step_num_increment_str = cls._pad_step_strs(
            step_num=step_num,
            steps=steps,
            steps_increment=steps_increment,
        )
        return f'{prefix.strip()}s [{step_num_str}/{steps_str}]-[{step_num_increment_str}/{steps_str}]'

    @classmethod
    def _pad_step_strs(cls, step_num: int, steps: int, steps_increment: Optional[int]) -> Tuple[
        str, str, Optional[str]]:
        step_num_str: str = StringUtil.pad_zeros(step_num, steps + 1)  ## step_num is in [1, steps], inclusive.
        step_num_increment_str: Optional[str] = None
        steps_str: str = StringUtil.pad_zeros(steps, steps + 1)
        if steps_increment is not None:
            step_num_increment_str: str = StringUtil.pad_zeros(  ## step_num is in [1, steps], inclusive.
                _step_num_after_iter(
                    step_num=step_num,
                    steps=steps,
                    steps_increment=steps_increment,
                ),
                steps + 1
            )
        return step_num_str, steps_str, step_num_increment_str

    @classmethod
    def _cur_batch_str(cls, batch_i: int, batches: int, prefix: str = 'step') -> str:
        batch_num_str, batches_str = cls._pad_batch_strs(batch_i=batch_i, batches=batches)
        return f'{prefix.strip()} [{batch_num_str}/{batches_str}]'

    @classmethod
    def _pad_batch_strs(cls, batch_i: int, batches: int) -> Tuple[str, str]:
        batch_num_str: str = StringUtil.pad_zeros(batch_i + 1, batches)  ## batch_i is in [0, num_batches], inclusive.
        batches_str: str = StringUtil.pad_zeros(batches, batches)
        return batch_num_str, batches_str
