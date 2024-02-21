from typing import *
import types
from abc import ABC, abstractmethod
import numpy as np
import time, traceback, pickle, gc, os, tempfile
from synthesizrr.base.constants import FileFormat, DataLayout, MLType, DataSplit, MLTypeSchema, Storage
from synthesizrr.base.util import Registry, MutableParameters, Parameters, FractionalBool, resolve_fractional_bool, as_list, \
    random_sample, safe_validate_arguments, Log, any_are_none, as_set, str_normalize, \
    all_are_none, all_are_not_none, is_abstract, flatten1d, get_default, Schema, format_exception_msg, \
    remove_nulls, StringUtil, is_function, get_fn_args
from synthesizrr.base.util.aws import S3Util
from synthesizrr.base.data import FileMetadata, ScalableDataFrame, ScalableSeries, Asset, ScalableDataFrameOrRaw
from synthesizrr.base.framework import Predictions, Dataset, Datasets
from synthesizrr.base.framework.metric import Metric, Metrics
from synthesizrr.base.framework.mixins import TaskRegistryMixin, TaskOrStr
from pydantic import root_validator, Extra, conint

MODEL_PARAMS_FILE_NAME: str = f'__model_params__.pkl'


class Algorithm(TaskRegistryMixin, Registry, ABC):
    _allow_multiple_subclasses: ClassVar[bool] = True  ## Allows multiple subclasses registered to the same task.
    _allow_subclass_override: ClassVar[bool] = True  ## Allows replacement of subclass with same name.

    dataset_statistics: ClassVar[Tuple[Union[str, Dict], ...]] = ()
    namespace: ClassVar[Optional[str]] = None
    description: ClassVar[Optional[str]] = None
    citation: ClassVar[Optional[str]] = None
    url: ClassVar[Optional[str]] = None
    stats: Optional[Metrics] = None
    num_steps_trained: int = 0
    num_rows_trained: int = 0

    inputs: ClassVar[Type[Dataset]]
    feature_mltypes: ClassVar[Optional[Tuple[MLType, ...]]] = None
    outputs: ClassVar[Type[Predictions]]

    default_batching_params: ClassVar[Dict[str, Any]] = {}

    class Config(MutableParameters.Config):
        extra = Extra.allow  ## Mutable+Extra = allows dynamically adding new items.

    @classmethod
    def _pre_registration_hook(cls):
        cls.inputs = cls._validate_inputs(cls.inputs)
        cls.outputs = cls._validate_outputs_type(cls.outputs)
        cls.dataset_statistics = tuple(as_set('row_count') | as_set(cls.dataset_statistics))

    @classmethod
    def _registry_keys(cls) -> Optional[Union[List[Any], Any]]:
        tasks: List = as_list(cls.tasks)
        return tasks + \
               [(task, str_normalize(cls.class_name)) for task in tasks] + \
               [
                   (task, str_normalize(alias))
                   for task in tasks
                   for alias in cls.aliases
               ]

    @classmethod
    def _validate_inputs(cls, inputs: Type[Dataset]) -> Type[Dataset]:
        for task in as_list(cls.tasks):
            if task not in as_list(inputs.tasks):
                raise ValueError(
                    f'{Algorithm.__name__} "{cls.class_name}" wants to support task "{task}" but this is not supported '
                    f'by input data class "{inputs.class_name}"; please either modify "{cls.class_name}" to '
                    f'remove this task, or modify "{inputs.class_name}" to support it.'
                )
        return inputs

    @classmethod
    def _validate_outputs_type(cls, outputs: Type[Predictions]) -> Type[Predictions]:
        for task in as_list(cls.tasks):
            if task not in as_list(outputs.tasks):
                raise ValueError(
                    f'{Algorithm.__name__} "{cls.class_name}" wants to support task "{task}" but this is not supported '
                    f'by output data class "{outputs.class_name}"; please either modify "{cls.class_name}" to '
                    f'remove this task, or modify "{outputs.class_name}" to support it.'
                )
        return outputs

    def __init__(
            self,
            *,
            stats: Optional[Metrics] = None,
            **kwargs
    ):
        super(Algorithm, self).__init__(stats=stats, **kwargs)
        self.stats = stats

    def __str__(self):
        params_str: str = self.json(indent=4, include={'hyperparams'})
        out: str = f'{self.class_name} with params:\n{params_str}'
        return out

    class Hyperparameters(Parameters):
        seed: Optional[int] = None  ## Seed used for randomization.
        batch_size: Optional[conint(ge=1)] = None  ## Training batch size. None allows inference-only models
        epochs: Optional[conint(ge=1)] = None  ## Number of epochs to train. None allows inference-only models
        steps: Optional[conint(ge=1)] = None  ## Number of steps to train. None allows inference-only models

        class Config(Parameters.Config):
            extra = Extra.allow

        @root_validator(pre=True)
        def check_params(cls, params: Dict) -> Dict:
            if all_are_not_none(params.get('epochs'), params.get('steps')):
                raise ValueError(f'Must pass at most one of `epochs` and `steps`; both were passed.')
            return params

        def dict(
                self,
                include: Optional[Union[Tuple[str, ...], Set[str], Callable]] = None,
                **kwargs,
        ) -> Dict:
            if is_function(include):
                include: Tuple[str, ...] = get_fn_args(include)
            if include is not None:
                include: Set[str] = as_set(include)
            return super().dict(include=include, **kwargs)

        def __str__(self) -> str:
            params_str: str = self.json(indent=4)
            out: str = f'{self.class_name}:\n{params_str}'
            return out

    hyperparams: Hyperparameters = {}

    @property
    def hyperparams_str(self) -> str:
        return ';'.join([f'{k}={v}' for k, v in self.hyperparams.dict().items()])

    @classmethod
    def create_hyperparams(cls, hyperparams: Optional[Dict] = None) -> Hyperparameters:
        hyperparams: Dict = get_default(hyperparams, {})
        return cls.Hyperparameters(**hyperparams)

    @root_validator(pre=False)
    def convert_params(cls, params: Dict):
        ## This allows us to create a new Algorithm instance without specifying `hyperparams`.
        ## If it is specified, we will pick cls.Hyperparameters, which can be overridden by the subclass.
        params.setdefault('hyperparams', {})
        params['hyperparams'] = cls._convert_params(cls.Hyperparameters, params['hyperparams'])
        if not isinstance(params['hyperparams'], Algorithm.Hyperparameters):
            raise ValueError(
                f'Custom hyperparameters class does not inherit from the base class version. '
                f'Please ensure your custom class for hyperparameters is called "{cls.Hyperparameters.class_name}" '
                f'and inherits from "{cls.Hyperparameters.class_name}" defined in the base class.'
            )
        return params

    @classmethod
    def of(
            cls,
            name: Optional[Union[str, Type['Algorithm'], FileMetadata, Dict]] = None,
            *,
            task: Optional[TaskOrStr] = None,
            init: bool = True,
            post_init: bool = True,
            model_dir: Optional[Union[FileMetadata, Dict, str]] = None,
            **kwargs,
    ) -> 'Algorithm':
        kwargs: Dict = remove_nulls(kwargs)
        if 'algorithm' in kwargs and name is None:
            name = kwargs.pop('algorithm')
        if isinstance(name, (FileMetadata, dict)):
            model_dir: Union[FileMetadata, Dict] = name
            name: Optional[str] = None
        if model_dir is not None:
            model_dir: FileMetadata = FileMetadata.of(model_dir)
        cache_dir: Optional[str] = kwargs.get('cache_dir', None)
        if cache_dir is None:
            cache_dir: str = tempfile.TemporaryDirectory().name  ## Does not exist yet
        cache_dir: FileMetadata = FileMetadata.of(cache_dir)
        if all_are_none(name, task) and model_dir is not None:
            # print(f'(pid={os.getpid()}) Loading "{AlgorithmClass}" model from dir: "{model_dir.path}"')
            model_params: Dict = {
                **get_default(Algorithm.load_params(model_dir=model_dir, raise_error=False, tmpdir=cache_dir.path), {}),
                **kwargs,
            }
            try:
                name: str = model_params['algorithm']
                task: str = model_params['task']
            except Exception as e:
                raise ValueError(
                    f'Cannot load algorithm and task from "{model_dir.path}"; '
                    f'found model params file with contents:\n{model_params}.'
                    f'\nError: {format_exception_msg(e)}'
                )
        if all_are_not_none(name, task):
            if isinstance(name, type):
                if not issubclass(name, Algorithm):
                    raise ValueError(f'`name` should be a string or subclass of "{cls.class_name}"')
                AlgorithmClass: Type[Algorithm] = name
            else:
                key = (task, name)
                AlgorithmClass: Type[Algorithm] = Algorithm.get_subclass(key)
        else:
            if isinstance(name, type) and issubclass(name, Algorithm):
                AlgorithmClass: Type[Algorithm] = name
            else:
                AlgorithmClass: Type[Algorithm] = cls
        if is_abstract(AlgorithmClass):
            ## Throw an error:
            abstract_task_subclasses: Set[str] = set()
            concrete_subclasses: Set[str] = set()
            for key, subclasses_dict in cls._registry.items():
                for subclass_name, Subclass in subclasses_dict.items():
                    if not is_abstract(Subclass):
                        concrete_subclasses.add(Subclass.__class__.__name__)
                    elif is_abstract(Subclass) and Subclass != Algorithm:
                        abstract_task_subclasses.add(Subclass.__class__.__name__)
            algorithm_error: str = f'{Algorithm.class_name}.of(name=..., task=..., etc)'
            abstract_task_subclasses_error: str = ', '.join([
                f'{s}.of(name=..., task=..., etc)' for s in random_sample(
                    list(abstract_task_subclasses),
                    n=2,
                    replacement=False
                )
            ])
            concrete_subclasses_error: str = ', '.join([
                f'{s}.of(**hyperparams)' for s in random_sample(
                    list(concrete_subclasses),
                    n=2,
                    replacement=False
                )
            ])
            raise ValueError(
                f'"{Algorithm.class_name}" is an abstract class and cannot be instantised. '
                f'To create an instance, please invoke .of(...) in one of the following ways:\n'
                f'(1) Using Algorithm base class: {algorithm_error}\n'
                f'(2) Using a task-specific base subclass: e.g. {abstract_task_subclasses_error}, etc.\n'
                f'(3) Using a concrete subclass: e.g. {concrete_subclasses_error}, etc.'
            )
        if not issubclass(AlgorithmClass, cls):
            ## Ensures that when you call Embedder.of(), you actually get an embedder.
            raise ValueError(
                f'Invoking {cls.class_name}.of(...) has retrieved {AlgorithmClass.class_name}, which is not '
                f'a subclass of {cls.class_name} as required. To retrieve any subclass of `{Algorithm.class_name}`, '
                f'please invoke {Algorithm.class_name}.of(...) instead.'
            )
        if task is None:
            if len(AlgorithmClass.tasks) > 1:
                raise ValueError(
                    f'"{cls.class_name}" has multiple tasks, please specify one by passing '
                    f'`task=...` when invoking {cls.class_name}.of(...)'
                )
            task: TaskOrStr = AlgorithmClass.tasks[0]
        if model_dir is None:
            # print(f'(pid={os.getpid()}) Creating "{AlgorithmClass}" model from scratch.')
            model: Algorithm = AlgorithmClass(
                task=task,
                **kwargs,
            )
        else:
            # print(f'(pid={os.getpid()}) Loading "{AlgorithmClass}" model from dir: "{model_dir.path}"')
            model_params: Dict = {
                **get_default(Algorithm.load_params(model_dir=model_dir, raise_error=False, tmpdir=cache_dir.path), {}),
                **kwargs,
            }
            model_params.pop('algorithm', None)
            model: Algorithm = AlgorithmClass(**model_params)
        if init:
            if model_dir is not None and model_dir.is_remote_storage():
                ## Copy to local cache directory:
                assert cache_dir is not None
                cache_dir.mkdir()
                model_dir.copy_to_dir(cache_dir)
                model_dir: FileMetadata = cache_dir
            model.initialize(model_dir=model_dir)
        if post_init:
            model.post_initialize()
        return model

    @abstractmethod
    def initialize(self, model_dir: Optional[FileMetadata] = None):
        """
        Should initialize a new "raw" model (or load it from a folder) and store it in private variables.
        :param model_dir: (optional) FileMetadata from which to load the model. Useful when we want to continue training
         an existing model. Use `model_dir.path` to access the folder where you should load your model.
        """
        pass

    def post_initialize(self):
        pass

    @classmethod
    @safe_validate_arguments
    def calculate_dataset_stats(
            cls,
            dataset: Dataset,
            *,
            data_split: Optional[DataSplit] = None,
            batch_size: Optional[conint(ge=1)] = None,
            **kwargs
    ) -> Optional[Metrics]:
        data_split: DataSplit = get_default(data_split, dataset.data_split)
        if data_split is None:
            raise ValueError(f'Must pass data_split in either {Dataset.class_name} or explicitly.')
        stats: Optional[Metrics] = None
        if len(cls.dataset_statistics) > 0:
            ## Get from algorithm class's `dataset_statistics` (always calculate num_rows) and dedupe:
            dataset_stats: List[Metric] = [Metric.of(s).clear() for s in
                                           ['row_count'] + as_list(cls.dataset_statistics)]
            dataset_stats: Dict[str, Metric] = {statistic.display_name: statistic for statistic in dataset_stats}
            dataset_stats: List[Metric] = list(dataset_stats.values())
            required_assets: Set[MLType] = set(flatten1d([m.required_assets for m in dataset_stats]))
            ## Do not fetch assets unless we need to.
            kwargs['fetch_assets'] = False if len(required_assets) == 0 else True
            kwargs['steps'] = None  ## Do not pass steps to the iteration, instead run for exactly one epoch.
            if not all([statistic.is_rolling for statistic in dataset_stats]):
                raise ValueError(
                    f'All `dataset_statistics` must be rolling; following are non-rolling: '
                    f'{[stat for stat in dataset_stats if stat.is_rolling is False]}'
                )
            for batch_i, batch in enumerate(cls.dataset_iter(
                    dataset,
                    data_split=data_split,
                    batch_size=batch_size,
                    validate_inputs=False,
                    shuffle=False,
                    **kwargs
            )):
                dataset_stats: List[Metric] = [
                    statistic.evaluate(batch, rolling=True)
                    for statistic in dataset_stats
                ]
            stats: Metrics = Metrics(metrics={data_split: dataset_stats})
        return stats

    @safe_validate_arguments
    def train(
            self,
            datasets: Optional[Union[Dataset, Datasets]] = None,
            metrics: Optional[Union[List[Union[Metric, Dict, str]], Metrics]] = None,
            *,
            trainer: str = 'local',
            **kwargs,
    ):
        """
        Creates a Trainer and calls train.
        :param datasets: the training dataset, or a collection of datasets.
        :param metrics: the training metrics, or a collection of datasets.
        :param trainer: the trainer to use. Defaults to "local" for LocalTrainer.
        :param kwargs: additional keyword args to pass to Trainer.of and .train() function itself.
        """
        if datasets is None and 'dataset' in kwargs:
            datasets = kwargs['dataset']
        from synthesizrr.base.framework.trainer.Trainer import Trainer
        if not isinstance(datasets, Datasets):
            datasets: Datasets = Datasets.of(train=datasets)
        if metrics is not None and not isinstance(metrics, Metrics):
            metrics: Metrics = Metrics.of(train=metrics)

        trainer = Trainer.of(
            trainer=trainer,
            algorithm=self,
            **kwargs
        )
        trainer.train(
            datasets=datasets,
            metrics=metrics,
            model=self,
            **kwargs
        )

    @safe_validate_arguments
    def train_iter(
            self,
            dataset: Any,
            **kwargs,
    ) -> Generator[Dict, None, None]:
        """
        Creates a generator for a single iteration of training loop, i.e. an epoch.
        This is NOT meant to be overridden unless you know what you're doing.
        """
        if type(self).train_step == Algorithm.train_step:  ## Ref: https://stackoverflow.com/a/59762827
            ## No-op, don't even load data.
            return
        train_batch_generator: Any = dataset
        if isinstance(dataset, Dataset):
            kwargs['data_split'] = DataSplit.TRAIN
            train_batch_generator: Generator = self.dataset_iter(dataset=dataset, **kwargs)
        for batch_i, batch in enumerate(train_batch_generator):
            batch: Dataset = self._task_preprocess(batch)
            if not isinstance(batch, self.inputs):
                raise ValueError(
                    f'{self.class_name} can only train on data of type {self.inputs}; '
                    f'{type(batch)} was passed. Use the from_* methods to create an instance of {self.inputs}.'
                )
            try:
                # print(f'(pid={os.getpid()}) Train Batch#{batch_i + 1}, length={len(batch)}')
                train_step_metrics: Optional[Dict] = self.train_step(batch)  ## Actually train on the batch.
                self.num_steps_trained += 1
                self.num_rows_trained += len(batch)
            except Exception as e:
                Log.error(format_exception_msg(e))
                raise e
            train_step_metrics: Dict = get_default(train_step_metrics, {})
            if not isinstance(train_step_metrics, dict):
                raise ValueError(
                    f'Expected output of {self.class_name}.train_step() to be a dict of batch-level training metrics '
                    f'(e.g. batch loss); found object of type: {type(train_step_metrics)}'
                )
            train_step_metrics['batch_size'] = len(batch)
            if isinstance(dataset, Dataset) and not dataset.in_memory():
                del batch
            yield train_step_metrics

    def _task_preprocess(self, batch: Dataset, **kwargs) -> Dataset:
        """
        Do any last-minute task-specific preprocessing, just before passing data to train_step.
        """
        return batch

    def train_step(self, batch: Dataset, **kwargs):
        pass

    def __call__(self, *args, **kwargs) -> Any:
        return self.predict(*args, **kwargs)

    @safe_validate_arguments
    def predict(
            self,
            dataset: Any,
            *,
            data_schema: Optional[MLTypeSchema] = None,
            yield_partial: bool = False,
            validate_outputs: Optional[FractionalBool] = None,
            **kwargs
    ) -> Union[Iterator[Predictions], Predictions]:
        dataset: Any = self.prepare_prediction_dataset(dataset, data_schema=data_schema, **kwargs)
        generator = self.predict_iter(
            dataset=dataset,
            validate_outputs=validate_outputs,
            **kwargs,
        )
        if yield_partial:
            ## You cannot "yield" & "return" in the same function, so we must return a generator or the realized object.
            return generator
        else:
            predictions: List[Predictions] = list(generator)
            predictions: Predictions = Predictions.concat(predictions)
            if not isinstance(predictions, self.outputs):
                raise ValueError(
                    f'{self.class_name} should return outputs of type {self.outputs}; '
                    f'found {type(predictions)} after concatenation.'
                )
            return predictions

    def prepare_prediction_dataset(
            self,
            dataset: Any,
            data_schema: Optional[MLTypeSchema] = None,
            **kwargs,
    ) -> Any:
        if isinstance(dataset, Predictions):
            dataset: Dataset = dataset.as_task_data()
        if not isinstance(dataset, Dataset):
            ## Try to convert to ScalableDataFrame:
            dataset: ScalableDataFrame = ScalableDataFrame.of(dataset)
            if data_schema is None:
                raise ValueError(
                    f'Error calling .predict(): when passing raw data instead of a {Dataset} or {Predictions} object, '
                    f'you must pass `data_schema` as well.'
                )
            dataset: Dataset = Dataset.of(
                task=self.task,  ## Copy the task from the Algorithm object
                data=dataset,
                data_schema=data_schema,
                **kwargs
            )
        return dataset

    @safe_validate_arguments
    def predict_iter(
            self,
            dataset: Any,
            *,
            batch_size: Optional[conint(ge=1)] = None,
            shuffle: bool = False,
            data_split: Optional[DataSplit] = None,
            validate_outputs: Optional[FractionalBool],
            **kwargs
    ) -> Generator[Predictions, None, None]:
        shuffle: bool = False  ## Do not shuffle while predicting.
        batch_size: Optional[conint(ge=1)] = get_default(
            batch_size,
            self.hyperparams.batch_size,
            len(dataset) if dataset.in_memory() else None,
        )
        predict_batch_generator: Any = dataset
        if isinstance(dataset, Dataset):
            data_split: DataSplit = get_default(data_split, dataset.data_split, DataSplit.PREDICT)
            predict_batch_generator: Generator = self.dataset_iter(
                dataset=dataset,
                data_split=data_split,
                batch_size=batch_size,
                shuffle=shuffle,
                **kwargs
            )
        for batch in predict_batch_generator:
            if not isinstance(batch, self.inputs):
                raise ValueError(
                    f'{self.class_name} can only predict using input data of type {self.inputs}; '
                    f'found {type(batch)}. Use from_* methods to create an instance of {self.inputs}.'
                )
            should_validate: bool = resolve_fractional_bool(validate_outputs)  ## Do NOT pass "seed" here
            batch: Dataset = self._task_preprocess(batch)
            try:
                # print(f'(pid={os.getpid()}): predicting on batch of size {len(batch)}')
                predictions: Any = self.predict_step(batch, **kwargs)  ## Actually predict on the batch.
            except Exception as e:
                Log.error(format_exception_msg(e))
                raise e
            predictions: Predictions = self._create_predictions(
                batch,
                predictions=predictions,
                validated=should_validate,
                **kwargs
            )
            predictions: Predictions = self.postprocess(
                predictions,
                **kwargs
            )
            if not isinstance(predictions, self.outputs):
                raise ValueError(
                    f'{self.class_name} should return outputs of type {self.outputs}; '
                    f'found {type(predictions)}. Use from_* methods to create an instance of {self.outputs}.'
                )
            yield predictions
            if isinstance(dataset, Dataset) and not dataset.in_memory():
                del batch

    @abstractmethod
    def predict_step(self, batch: Dataset, **kwargs) -> Any:
        pass

    @abstractmethod
    def _create_predictions(self, batch: Dataset, predictions: Any, **kwargs) -> Predictions:
        pass

    @classmethod
    @safe_validate_arguments
    def dataset_iter(
            cls,
            dataset: Dataset,
            *,
            data_split: Optional[DataSplit] = None,
            validate_inputs: Optional[FractionalBool] = None,
            fetch_assets: bool = True,
            **kwargs
    ) -> Generator[Union[Dataset, Any], None, None]:
        """
        Retrieves Dataset batches from a dataset.
        """
        data_split: DataSplit = get_default(data_split, dataset.data_split)
        if data_split is None:
            raise ValueError(
                f'Must pass data_split in either {Dataset.class_name}.of(), '
                f'or explicitly when calling .dataset_iter()'
            )

        feature_mltypes: Tuple[MLType, ...] = get_default(cls.feature_mltypes, ())  ## By default, gets all features
        if len(feature_mltypes) > 0:  ## If we have zero feature-MLTypes, keep all features.
            filtered_features_schema: MLTypeSchema = dict(dataset.columns(
                mltypes=feature_mltypes,
                schema_portion='features',
                return_mltypes=True
            ))
            ## Setting this will cause the reader to filter out other columns in dataset.read_batches()
            dataset.data_schema: Schema = dataset.data_schema.set_features(filtered_features_schema, override=True)

        batching_params: Dict = {
            **cls.default_batching_params,
            **kwargs,
            **dict(
                data_split=data_split,
            ),
        }
        return dataset.iter(
            validate=validate_inputs,
            fetch_assets=fetch_assets,
            map=cls.preprocess,
            map_apply='batch',
            **batching_params,
        )

    @staticmethod
    def preprocess(batch: Dataset, **kwargs) -> Dataset:
        return batch

    @staticmethod
    def postprocess(batch: Predictions, **kwargs) -> Predictions:
        return batch

    @safe_validate_arguments
    def evaluate(
            self,
            dataset: Any,
            metrics: Optional[Union[Union[Metric, Dict, str], List[Union[Metric, Dict, str]]]] = None,
            **kwargs
    ) -> List[Metric]:
        if metrics is None and 'metric' in kwargs:
            metrics = kwargs['metric']
        if metrics is None:
            raise ValueError(f'Must pass argument `metrics` to .evaluate')
        metrics: List[Union[Metric, Dict, str]] = as_list(metrics)
        if any_are_none(metrics):
            raise ValueError(f'Metrics list is empty for {dataset.data_split.name.capitalize()} dataset')
        metrics: List[Metric] = [Metric.of(metric) for metric in metrics]
        if np.all([metric.is_rolling for metric in metrics]):
            evaluated_metrics: List[Metric] = [metric.clear() for metric in metrics]
            ## If all metrics support rolling calculation, then call calculate_rolling_metric...this saves memory as we
            ## only need one batch to be in memory at a time.
            for partial_predictions in self.predict(
                    dataset,
                    yield_partial=True,
                    **kwargs
            ):
                evaluated_metrics: List[Metric] = [
                    metric.evaluate(partial_predictions, rolling=True)
                    for metric in evaluated_metrics
                ]
        else:
            predictions: Predictions = self.predict(
                dataset,
                yield_partial=False,
                **kwargs
            )
            evaluated_metrics: List[Metric] = [
                self._evaluate_metric(
                    predictions=predictions,
                    metric=metric,
                )
                for metric in metrics
            ]
        return evaluated_metrics

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

    @classmethod
    def save_param_names(cls) -> Set[str]:
        BaseClasses: List[Type] = as_list(cls.__bases__) + [cls]
        abstract_base_class_param_names: Set[str] = set()
        for BaseClass in BaseClasses:
            ## TODO: is is_abstract really needed?
            if is_abstract(BaseClass) and issubclass(BaseClass, Algorithm):  ## Catches `Algorithm` class.
                abstract_base_class_param_names.update(BaseClass.param_names())
        return abstract_base_class_param_names

    def save_params(
            self,
            model_dir: Union[FileMetadata, str],
            model_params_file_name: str = MODEL_PARAMS_FILE_NAME,
    ):
        model_dir: FileMetadata = FileMetadata.of(model_dir)
        with model_dir.open(file_name=model_params_file_name, mode="wb+") as out:
            pickle.dump(self._model_params(), out)

    def _model_params(self) -> Dict[str, Any]:
        model_params: Dict[str, Any] = self.dict(include=self.save_param_names())
        model_params['algorithm'] = self.class_name
        return model_params

    def __hash__(self) -> str:
        return StringUtil.hash(self._model_params())

    @classmethod
    def load_params(
            cls,
            model_dir: Union[FileMetadata, str],
            model_params_file_name: str = MODEL_PARAMS_FILE_NAME,
            raise_error: bool = True,
            tmpdir: Optional[str] = None,
            **kwargs,
    ) -> Optional[Dict]:
        model_dir: FileMetadata = FileMetadata.of(model_dir)
        with model_dir.open(file_name=model_params_file_name, mode="rb", tmpdir=tmpdir) as inp:
            model_params: Dict = pickle.load(inp)
            if not isinstance(model_params, dict):
                raise ValueError(f'Loaded data must be a dict; found data of type {type(model_params)}')
            return model_params

    def save(self, model_dir: FileMetadata):
        pass

    def post_train_cleanup(self):
        """Should delete all heavy in-memory values. Does not need to delete hyperparams or other lightweight values."""
        gc.collect()

    def cleanup(self):
        """Should delete all heavy in-memory values. Does not need to delete hyperparams or other lightweight values."""
        self.post_train_cleanup()
        gc.collect()
