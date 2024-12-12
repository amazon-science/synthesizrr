from typing import *
import warnings
import time, os, gc, math, random, numpy as np, pandas as pd, ray, logging
from contextlib import contextmanager, ExitStack
from functools import partial
from ray import tune, air
from ray.runtime_env import RuntimeEnv as RayRuntimeEnv
from synthergent.base.util import Parameters, set_param_from_alias, safe_validate_arguments, format_exception_msg, StringUtil, \
    Timer, get_default, is_list_like, is_empty_list_like, run_concurrent, \
    Timeout24Hr, Timeout1Hr, Timeout, as_list, accumulate, get_result, wait, AutoEnum, auto, ProgressBar, only_item, \
    ignore_all_output, ignore_logging, ignore_warnings, ignore_stdout
from synthergent.base.util.aws import S3Util
from synthergent.base.data import FileMetadata, ScalableDataFrame
from synthergent.base.data.sdf import DaskScalableDataFrame
from synthergent.base.constants import Task, DataSplit, Storage, REMOTE_STORAGES, DataLayout, FILE_FORMAT_TO_FILE_ENDING_MAP, \
    FailureAction
from synthergent.base.framework.evaluator.Evaluator import Evaluator, save_predictions, load_predictions
from synthergent.base.framework.task_data import Datasets, Dataset
from synthergent.base.framework.predictions import Predictions
from synthergent.base.framework.algorithm import Algorithm, TaskOrStr
from synthergent.base.framework.metric import Metric, Metrics
from synthergent.base.framework.tracker.Tracker import Tracker
from synthergent.base.framework.ray_base import RayInitConfig, max_num_resource_actors, ActorComposite, RequestCounter
from pydantic import Extra, conint, confloat, constr, root_validator, validator
from pydantic.typing import Literal


@ray.remote
class RowCounter:
    def __init__(self):
        self.rows_completed: int = 0

    def increment_rows(self, num_rows: int):
        self.rows_completed += num_rows
        # print(f'Shard {shard[0]} slept for {time_slept:.3f} sec and completed {num_rows} rows')

    def get_rows_completed(self) -> int:
        return self.rows_completed


class ShardingStrategy(AutoEnum):
    COARSE = auto()
    GRANULAR = auto()


class DataLoadingStrategy(AutoEnum):
    LOCAL = auto()
    DASK = auto()


class LoadBalancingStrategy(AutoEnum):
    ROUND_ROBIN = auto()
    LEAST_USED = auto()
    UNUSED = auto()
    RANDOM = auto()


ALGORITHM_EVALUATOR_VERBOSITY_IGNORE: Dict[int, List[Callable]] = {
    0: [ignore_all_output],
    1: [ignore_stdout, ignore_warnings, partial(ignore_logging, disable_upto=logging.WARNING)],
    2: [partial(ignore_logging, disable_upto=logging.DEBUG)],
    3: [partial(ignore_logging, disable_upto=logging.NOTSET)],
    4: [partial(ignore_logging, disable_upto=logging.NOTSET)],
    5: [partial(ignore_logging, disable_upto=logging.NOTSET)],
}


@contextmanager
def algorithm_evaluator_verbosity(verbosity: int):
    if verbosity not in ALGORITHM_EVALUATOR_VERBOSITY_IGNORE:
        raise ValueError(
            f'Expected `verbosity` to be one of {list(ALGORITHM_EVALUATOR_VERBOSITY_IGNORE.keys())}; '
            f'found: {verbosity}'
        )
    ignore_fns: List[Callable] = ALGORITHM_EVALUATOR_VERBOSITY_IGNORE[verbosity]
    with ExitStack() as es:
        for ignore_fn in ignore_fns:
            es.enter_context(ignore_fn())
        yield


@ray.remote
class AlgorithmEvaluator:
    def __init__(
            self,
            evaluator: Dict,
            actor: Tuple[int, int],
            request_counter: RequestCounter,
            verbosity: int,
    ):
        from synthergent.base.framework import Algorithm, Evaluator
        import synthergent.base.algorithm  ## Registers all algorithms
        self.verbosity = verbosity
        self.evaluator: Optional[Evaluator] = None  ## Set this temporarily while loading when calling Evaluator.of(...)
        with algorithm_evaluator_verbosity(self.verbosity):
            self.evaluator: Evaluator = Evaluator.of(**evaluator)
        self.actor = actor
        self.request_counter: RequestCounter = request_counter

    def is_available(self) -> bool:
        try:
            return self.evaluator is None
        except Exception as e:
            return False

    def get_ip_address(self) -> Optional[str]:
        try:
            ## Ref: https://discuss.ray.io/t/get-current-executor-ip/4916
            return ray.util.get_node_ip_address()
        except Exception as e:
            return None

    def evaluate_shard(
            self,
            data: Any,
            *,
            dataset_params: Dict,
            input_len: int,
            batch_size: int,
            batches_per_save: int,
            predictions_destination: Optional[FileMetadata],
            return_predictions: bool,
            row_counter: Optional[RowCounter],
            failure_action: FailureAction,
            data_loading_strategy: DataLoadingStrategy,
            **kwargs,
    ):
        import pandas as pd
        from synthergent.base.util import accumulate
        from synthergent.base.data import FileMetadata
        from synthergent.base.framework import Dataset, Datasets, Predictions, Metric, Metrics
        from concurrent.futures._base import Future

        ## Stops Pandas SettingWithCopyWarning in output. Ref: https://stackoverflow.com/a/20627316
        pd.options.mode.chained_assignment = None

        self.request_counter.started_request.remote()
        predicted_num_rows: int = 0
        predicted_num_batches: int = 0
        predictions: List[Union[Predictions, Future]] = []
        save_futures: List[Future] = []
        error_to_raise: Optional[Exception] = None
        data: ScalableDataFrame = ScalableDataFrame.of(data)
        failure_action: FailureAction = FailureAction(failure_action)
        data_loading_strategy: DataLoadingStrategy = DataLoadingStrategy(data_loading_strategy)
        if data_loading_strategy is DataLoadingStrategy.DASK:
            ## `data` is the entire dataset, so predict only the relevant shard.
            shard: Tuple[int, int] = self.actor
            make_fname = lambda predicted_num_rows, macro_batch, input_len: \
                f'shard-{StringUtil.pad_zeros(*self.actor)}' \
                f'-rows-{StringUtil.pad_zeros(predicted_num_rows, input_len)}' \
                f'-to-{StringUtil.pad_zeros(predicted_num_rows + len(macro_batch), input_len)}'
        elif data_loading_strategy is DataLoadingStrategy.LOCAL:
            ## `data` here is one shard of the dataset, so predict it entirely.
            shard: Tuple[int, int] = (0, 1)
            make_fname = lambda predicted_num_rows, macro_batch, input_len: \
                f'part-{StringUtil.pad_zeros(dataset_params["data_idx"] + 1, int(1e9))}' \
                f'-rows-{StringUtil.pad_zeros(predicted_num_rows, input_len)}' \
                f'-to-{StringUtil.pad_zeros(predicted_num_rows + len(macro_batch), input_len)}'
        else:
            raise NotImplementedError(f'Unsupported `data_loading_strategy`: {data_loading_strategy}')
        for macro_batch in data.stream(
                shard=shard,
                batch_size=batch_size * batches_per_save,
                shuffle=False,
                stream_as=DataLayout.PANDAS,
        ):
            try:
                if error_to_raise is None:
                    assert isinstance(macro_batch, ScalableDataFrame) and macro_batch.layout == DataLayout.PANDAS
                    macro_batch_save_file: Optional[FileMetadata] = None  ## Saves a macro batch
                    if predictions_destination is not None:
                        file_ending: str = as_list(
                            FILE_FORMAT_TO_FILE_ENDING_MAP[predictions_destination.format]
                        )[0]
                        fname: str = make_fname(
                            predicted_num_rows=predicted_num_rows,
                            macro_batch=macro_batch,
                            input_len=input_len,
                        )
                        macro_batch_save_file: FileMetadata = predictions_destination.file_in_dir(
                            fname,
                            return_metadata=True,
                            file_ending=file_ending,
                        )
                    if macro_batch_save_file is None or not S3Util.s3_object_exists(macro_batch_save_file.path):
                        ## We should predict:
                        kwargs['tracker'] = Tracker.noop_tracker()  ## Do not track.
                        macro_batch: Dataset = Dataset.of(
                            **dataset_params,
                            data=macro_batch,
                        )
                        with algorithm_evaluator_verbosity(self.verbosity):
                            macro_batch_predictions: Predictions = self.evaluator.evaluate(
                                macro_batch,
                                batch_size=batch_size,
                                return_predictions=True,
                                metrics=None,
                                progress_bar=None,
                                failure_action=FailureAction.ERROR,
                                **kwargs,
                            )
                        if macro_batch_save_file is not None:
                            ## Save the predictions to disk/S3:
                            save_futures.append(
                                run_concurrent(
                                    save_predictions,
                                    predictions=macro_batch_predictions,
                                    predictions_destination=macro_batch_save_file,
                                )
                            )
                        if return_predictions:
                            predictions.append(macro_batch_predictions)
                    else:
                        ## Load the file from disk/S3:
                        if return_predictions:
                            predictions.append(
                                run_concurrent(
                                    load_predictions,
                                    macro_batch_save_file.path,
                                )
                            )
            except Exception as e:
                if failure_action is FailureAction.ERROR:
                    with algorithm_evaluator_verbosity(self.verbosity):
                        logging.error(format_exception_msg(e))
                    ## Error immediately
                    raise e
                elif failure_action is FailureAction.ERROR_DELAYED:
                    ## Continue iterating to the end to update `row_counter`, then raise an error.
                    error_to_raise: Exception = e
                elif failure_action is FailureAction.WARN:
                    with algorithm_evaluator_verbosity(self.verbosity):
                        logging.warning(format_exception_msg(e))
                elif failure_action is FailureAction.IGNORE:
                    pass
                else:
                    raise NotImplementedError(f'Unsupported `failure_action`: {failure_action}')
            finally:
                predicted_num_rows += len(macro_batch)
                predicted_num_batches += batches_per_save
                if row_counter is not None:
                    row_counter.increment_rows.remote(
                        num_rows=len(macro_batch),
                    )
        accumulate(save_futures)
        self.request_counter.completed_request.remote()
        if error_to_raise is not None:
            with algorithm_evaluator_verbosity(self.verbosity):
                logging.error(format_exception_msg(error_to_raise))
            raise error_to_raise
        if return_predictions:
            # print(f'Size of predictions dataframe: {preds_df.memory_usage(deep=True)}')
            # print(f'Actor#{self.actor} predicted {len(preds_df)} rows.')
            predictions: Predictions = Predictions.concat(
                accumulate(predictions),
                layout=DataLayout.PANDAS,
            )
            return predictions
        return None


class RayEvaluator(Evaluator):
    aliases = ['ray']

    class Config(Evaluator.Config):
        extra = Extra.allow

    class RunConfig(Evaluator.RunConfig):
        ray_init: RayInitConfig = {}

    nested_evaluator_name: Optional[str] = None
    num_models: Optional[conint(ge=1)] = None
    model: Optional[List[ActorComposite]] = None  ## Stores the actors.
    resources_per_model: Dict[
        Literal['cpu', 'gpu'],
        Union[confloat(ge=0.0, lt=1.0), conint(ge=0)]
    ] = {'cpu': 1, 'gpu': 0}
    progress_update_frequency: int = 5
    ## By default, do not cache the model:
    cache_timeout: Optional[Union[Timeout, confloat(gt=0)]] = None

    @root_validator(pre=True)
    def ray_evaluator_params(cls, params: Dict) -> Dict:
        set_param_from_alias(params, param='nested_evaluator_name', alias=['nested_evaluator'])
        set_param_from_alias(params, param='num_models', alias=[
            'max_models', 'max_num_models', 'model_copies', 'num_copies', 'num_actors',
        ])
        set_param_from_alias(params, param='resources_per_model', alias=[
            'model_resources', 'resources',
        ])
        set_param_from_alias(params, param='progress_update_frequency', alias=[
            'progress_update_freq', 'max_report_frequency', 'progress_update_seconds', 'progress_update_sec',
        ])
        if params.get('device') is not None:
            raise ValueError(
                f'Do not pass "device" to {cls.class_name}.of(), '
                f'instead pass it as: {cls.class_name}.evaluate(device=...)'
            )

        ## Remove extra param names:
        params: Dict = cls._clear_extra_params(params)

        if params.get('resources_per_model') is not None:
            for resource_name, resource_requirement in params['resources_per_model'].items():
                if resource_requirement > 1.0 and round(resource_requirement) != resource_requirement:
                    raise ValueError(
                        f'When specifying `resources_per_model`, fractional resource-requirements are only allowed '
                        f'when specifying a value <1.0; found fractional resource-requirement '
                        f'"{resource_name}"={resource_requirement}. To fix this error, set an integer value for '
                        f'"{resource_name}" in `resources_per_model`.'
                    )
        return params

    def initialize(self, reinit_ray: bool = False, **kwargs):
        ## Connect to the Ray cluster
        if not ray.is_initialized() or reinit_ray is True:
            ray.init(
                ignore_reinit_error=True,
                address=self.run_config.ray_init.address,
                _temp_dir=str(self.run_config.ray_init.temp_dir),
                runtime_env=self.run_config.ray_init.runtime_env,
                **self.run_config.ray_init.dict(exclude={'address', 'temp_dir', 'include_dashboard', 'runtime_env'}),
            )
        if not ray.is_initialized():
            raise SystemError(f'Could not initialize ray.')

    def _load_model(
            self,
            *,
            num_actors: Optional[int] = None,
            progress_bar: Optional[Union[Dict, bool]] = True,
            **kwargs,
    ) -> List[ActorComposite]:
        num_actors: int = get_default(num_actors, self.num_actors)
        progress_bar: Optional[Dict] = self._run_evaluation_progress_bar(progress_bar)
        nested_evaluator_params: Dict = self._create_nested_evaluator_params(**kwargs)

        def actor_factory(*, request_counter: Any, actor_i: int, actor_id: str, **kwargs):
            return AlgorithmEvaluator.options(
                num_cpus=self.model_num_cpus,
                num_gpus=self.model_num_gpus,
            ).remote(
                evaluator=nested_evaluator_params,
                actor=(actor_i, num_actors),
                request_counter=request_counter,
                verbosity=self.verbosity,
            )

        return ActorComposite.create_actors(
            actor_factory,
            num_actors=num_actors,
            progress_bar=progress_bar,
        )

    def cleanup_model(self, ):
        self._kill_actors()

    def _kill_actors(self):
        try:
            if self.model is not None:
                actor_composites: List[ActorComposite] = self.model
                self.model: Optional[List[ActorComposite]] = None
                for actor_comp in actor_composites:
                    actor_comp.kill()
                    del actor_comp
                del actor_composites
        finally:
            gc.collect()

    @property
    def max_num_actors(self) -> int:
        cluster_resources: Dict = ray.cluster_resources()
        RAY_NUM_CPUS: int = int(cluster_resources['CPU'])
        RAY_NUM_GPUS: int = int(cluster_resources.get('GPU', 0))

        max_num_cpu_actors: int = max_num_resource_actors(
            self.resources_per_model.get('cpu', 1),
            RAY_NUM_CPUS,
        )
        max_num_gpu_actors: Union[int, float] = max_num_resource_actors(
            self.resources_per_model.get('gpu', 0),
            RAY_NUM_GPUS,
        )
        max_num_actors: int = min(max_num_gpu_actors, max_num_cpu_actors)
        return max_num_actors

    @property
    def model_num_cpus(self) -> Union[conint(ge=1), confloat(ge=0.0, lt=1.0)]:
        return self.resources_per_model.get('cpu', 1)

    @property
    def model_num_gpus(self) -> Union[conint(ge=0), confloat(ge=0.0, lt=1.0)]:
        return self.resources_per_model.get('gpu', 0)

    @property
    def num_actors(self) -> int:
        cluster_resources: Dict = ray.cluster_resources()
        RAY_NUM_CPUS: int = int(cluster_resources['CPU'])
        RAY_NUM_GPUS: int = int(cluster_resources.get('GPU', 0))

        model_num_cpus: Union[conint(ge=1), confloat(ge=0.0, lt=1.0)] = self.model_num_cpus
        model_num_gpus: Union[conint(ge=0), confloat(ge=0.0, lt=1.0)] = self.model_num_gpus
        max_num_actors: int = self.max_num_actors
        num_actors: Optional[int] = self.num_models
        if num_actors is None:
            warnings.warn(
                f'`num_models` is not specified. Since each model-copy requires '
                f'{model_num_cpus} cpus and {model_num_gpus} gpus, we create {max_num_actors} model-copies so as '
                f'to utilize the entire Ray cluster (having {RAY_NUM_CPUS} cpus and {RAY_NUM_GPUS} gpus). '
                f'To reduce the cluster-utilization, explicitly pass `num_models`.'
            )
            num_actors: int = max_num_actors
        elif num_actors > max_num_actors:
            warnings.warn(
                f'Requested {num_actors} model-copies (each with {model_num_cpus} cpus and {model_num_gpus} gpus); '
                f'however, the Ray cluster only has {RAY_NUM_CPUS} cpus and {RAY_NUM_GPUS} gpus, '
                f'thus we can create at most {max_num_actors} model-copies.'
            )
        num_actors: int = min(num_actors, max_num_actors)
        return num_actors

    def _create_nested_evaluator_params(self, **kwargs) -> Dict:
        nested_evaluator_name: str = get_default(
            self.nested_evaluator_name,
            'accelerate' if self.model_num_gpus > 1 else 'local',
        )
        if self.model_dir is not None and not self.model_dir.is_remote_storage():
            raise ValueError(
                f'When passing `model_dir` to {self.class_name}.of(...), the model directory '
                f'must be on a remote storage, i.e. one of: {REMOTE_STORAGES}'
            )
        if 'cache_dir' in kwargs and kwargs['cache_dir'] is None:
            kwargs.pop('cache_dir')
        if 'model_dir' in kwargs and kwargs['model_dir'] is None:
            kwargs.pop('model_dir')
        nested_evaluator: Dict = Evaluator.of(**{
            **dict(
                evaluator=nested_evaluator_name,
                task=self.task,
                AlgorithmClass=self.AlgorithmClass,
                hyperparams=self.hyperparams,
                model_dir=self.model_dir,
                cache_dir=self.cache_dir,
                cache_timeout=self.cache_timeout,
                validate_inputs=self.validate_inputs,
                validate_outputs=self.validate_outputs,
                custom_definitions=self.custom_definitions,
            ),
            **kwargs,
            **dict(
                init=False,  ## Do not initialize the evaluator on the local machine.
                init_model=False,  ## Do not initialize the evaluator model on the local machine.
                verbosity=0,  ## Ensures we do not print anything from the nested evaluator.
            ),
        }).dict()
        if 'stats' in kwargs:
            nested_evaluator['stats'] = kwargs['stats']
        # print(f'nested_evaluator dict:\n{nested_evaluator}')
        nested_evaluator['evaluator']: str = nested_evaluator_name
        if self.model_num_gpus > 0:
            nested_evaluator.setdefault('device', 'cuda')
        return nested_evaluator

    @staticmethod
    def ray_logger(text: str, should_log: bool, tracker: Tracker):
        text: str = f'{text}'
        if should_log is False:  ## Don't log anything.
            return
        else:
            tracker.info(text)

    @safe_validate_arguments
    def _run_evaluation(
            self,
            dataset: Dataset,
            *,
            tracker: Tracker,
            metrics: Optional[List[Metric]],
            return_predictions: bool,
            predictions_destination: Optional[FileMetadata],
            progress_bar: Optional[Dict],
            failure_action: FailureAction = FailureAction.ERROR_DELAYED,
            sharding_strategy: ShardingStrategy = ShardingStrategy.COARSE,
            data_loading_strategy: DataLoadingStrategy = DataLoadingStrategy.LOCAL,
            load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_USED,
            read_as: Optional[DataLayout] = DataLayout.PANDAS,
            submission_batch_size: Optional[conint(ge=1)] = None,
            submission_max_queued: conint(ge=0) = 2,
            submission_batch_wait: confloat(ge=0) = 15,
            evaluation_timeout: confloat(ge=0, allow_inf_nan=True) = math.inf,
            allow_partial_predictions: bool = False,
            **kwargs
    ) -> Tuple[Optional[Predictions], Optional[List[Metric]]]:
        ## TODO: add rows per save, SaveStrategy and UpdateStrategy:
        set_param_from_alias(kwargs, param='batches_per_save', alias=['batches_per_update'], default=1)
        set_param_from_alias(
            kwargs,
            param='batch_size',
            alias=['predict_batch_size', 'eval_batch_size', 'nrows', 'num_rows'],
            default=self._create_hyperparams().batch_size,
        )
        batch_size: Optional[int] = kwargs.pop('batch_size', None)

        evaluated_predictions: Optional[Predictions] = None
        evaluated_metrics: Optional[List[Metric]] = None

        try:
            timer: Timer = Timer(silent=True)
            timer.start()
            ## Verbosity >= 1: progress bars
            progress_bar: Optional[Dict] = self._run_evaluation_progress_bar(progress_bar)
            ## Verbosity >= 2: basic logging
            main_logger: Callable = partial(
                self.ray_logger,
                ## Unless we request silence (verbosity=0), print important information.
                should_log=self.verbosity >= 2,
                tracker=tracker,
            )
            ## Verbosity >= 3: detailed logging
            debug_logger: Callable = partial(
                self.ray_logger,
                ## Unless we request silence (verbosity=0), print important information.
                should_log=self.verbosity >= 3,
                tracker=tracker,
            )
            main_logger(self._evaluate_start_msg(tracker=tracker, **kwargs))
            if batch_size is None:
                raise ValueError(
                    f'Could not find batch_size in model hyperparams; '
                    f'please pass it explicitly like so: {self.class_name}.evaluate(batch_size=...)'
                )
            if predictions_destination is not None:
                if predictions_destination.storage is not Storage.S3:
                    raise ValueError(
                        f'Results can only be saved to {Storage.S3}; '
                        f'found storage {predictions_destination.storage} having path: {predictions_destination.path}'
                    )
                if not predictions_destination.is_path_valid_dir():
                    raise ValueError(
                        f'Expected predictions destination to be a valid directory; '
                        f'found: "{predictions_destination.path}"...did you forget a "/" at the end?'
                    )
                assert predictions_destination.format is not None  ## Checked in .evaluate().

            actors_were_created_in_this_call: bool = self.init_model(progress_bar=progress_bar, **kwargs)
            num_actors_created: int = len(self.model)
            if actors_were_created_in_this_call:
                resource_req_str: str = StringUtil.join_human([
                    f"{resource_req} {resource_name}(s)"
                    for resource_name, resource_req in self.resources_per_model.items()
                ])
                main_logger(f'Created {num_actors_created} copies of the model, each using {resource_req_str}.')
            dataset: Dataset = dataset.read(read_as=read_as, npartitions=num_actors_created)
            data: ScalableDataFrame = dataset.data.persist(wait=True)
            input_len: int = len(data)
            input_len_str: str = StringUtil.readable_number(input_len, decimals=1, short=True)
            if sharding_strategy is ShardingStrategy.GRANULAR:
                if not isinstance(data, DaskScalableDataFrame):
                    raise ValueError(
                        f'Can only use sharding_strategy={ShardingStrategy.GRANULAR} when read_as={DataLayout.DASK}; '
                        f'found read_as={read_as}.'
                    )
                sharding_timer: Timer = Timer(silent=True)
                sharding_timer.start()
                _, _ = data.set_shard_divisions(
                    num_shards=num_actors_created,
                    num_rows=batch_size,
                    inplace=True,
                )
                sharding_timer.stop()
                debug_logger(f'Set shard divisions in {sharding_timer.time_taken_human}.')

            ## Submit data to be predicted:
            if self.model_num_gpus > 0:
                kwargs.setdefault('device', 'cuda')
            row_counter: ray.actor.ActorHandle = RowCounter.options(
                num_cpus=0.1,
                max_concurrency=max(
                    num_actors_created + 2,
                    submission_max_queued * num_actors_created + 2,
                ),
            ).remote()
            dataset_params: Dict = dataset.dict(exclude={'data'})
            if data_loading_strategy is DataLoadingStrategy.DASK:
                submissions_progress_bar: ProgressBar = ProgressBar.of(
                    progress_bar,
                    total=num_actors_created,
                    desc=f'Submitting {input_len_str} rows',
                    unit='submissions',
                )
                ## Each actor streams data from Dask dataframe on the cluster:
                if not isinstance(data, DaskScalableDataFrame):
                    raise ValueError(
                        f'Can only use data_loading_strategy={DataLoadingStrategy.DASK} when read_as={DataLayout.DASK}; '
                        f'found read_as={read_as}.'
                    )
                predictions: List[Optional[Predictions]] = []
                ## When using DataLoadingStrategy.DASK, each actor will evaluate a fixed set of set, so
                ## LoadBalancingStrategy does not come into play.
                for actor_i, actor_comp in enumerate(self.model):
                    predictions.append(
                        actor_comp.actor.evaluate_shard.remote(
                            data,
                            dataset_params=dataset_params,
                            input_len=input_len,
                            batch_size=batch_size,
                            predictions_destination=predictions_destination,
                            return_predictions=return_predictions or metrics is not None,
                            row_counter=row_counter,
                            failure_action=failure_action,
                            data_loading_strategy=data_loading_strategy,
                            **kwargs,
                        )
                    )
                    submissions_progress_bar.update(1)
                ## Initialize with number of rows completed so far:
                rows_completed: int = ray.get(row_counter.get_rows_completed.remote())
                rows_completed_progress_bar: ProgressBar = ProgressBar.of(
                    progress_bar,
                    total=input_len,
                    desc=f'Evaluating {input_len_str} rows',
                    initial=rows_completed,
                )
                submissions_progress_bar.success('Completed submissions')
            elif data_loading_strategy is DataLoadingStrategy.LOCAL:
                ## Load each shard of data on the calling machine, and send to the cluster:
                if submission_batch_size is None:
                    submission_batch_size: int = batch_size * kwargs['batches_per_save']  ## Heuristic
                submissions_progress_bar: ProgressBar = ProgressBar.of(
                    progress_bar,
                    total=math.ceil(input_len / submission_batch_size),
                    desc=f'Submitting {input_len_str} rows',
                    unit='submissions',
                )
                ## Initialize to zero:
                rows_completed: int = ray.get(row_counter.get_rows_completed.remote())
                rows_completed_progress_bar: ProgressBar = ProgressBar.of(
                    progress_bar,
                    total=input_len,
                    desc=f'Evaluating {input_len_str} rows',
                    initial=rows_completed,
                )
                predictions: List[Optional[Predictions]] = []
                for part_i, part_data in enumerate(data.stream(
                        batch_size=submission_batch_size,
                        shuffle=False,
                        stream_as=DataLayout.PANDAS,
                        fetch_partitions=1,
                )):
                    ## When using DataLoadingStrategy.LOCAL, we can pick which actor to send the data to based on
                    ## the LoadBalancingStrategy.
                    if load_balancing_strategy is LoadBalancingStrategy.ROUND_ROBIN:
                        actor_comp: ActorComposite = self.model[part_i % num_actors_created]
                    elif load_balancing_strategy is LoadBalancingStrategy.RANDOM:
                        rnd_actor_i: int = random.choice(list(range(0, num_actors_created)))
                        actor_comp: ActorComposite = self.model[rnd_actor_i]
                    elif load_balancing_strategy is LoadBalancingStrategy.LEAST_USED:
                        ## When all actors unused, latest_last_completed_timestamp is -1, so we will pick a random actor
                        ## After that, we will pick the actor with the least load which has most-recently completed
                        actor_usages: List[Tuple[int, float, str]] = self._get_actor_usages()
                        min_actor_usage: int = min([actor_usage for actor_usage, _, _ in actor_usages])
                        while min_actor_usage > submission_max_queued:
                            debug_logger(
                                f'Actor usages:\n{actor_usages}\n'
                                f'(All are above submission_least_used_threshold={submission_max_queued}, '
                                f'waiting for {submission_batch_wait} seconds).'
                            )
                            time.sleep(submission_batch_wait)
                            actor_usages: List[Tuple[int, float, str]] = self._get_actor_usages()
                            min_actor_usage: int = min([actor_usage for actor_usage, _, _ in actor_usages])

                        ## last_completed_timestamp = Most-recently used (it will be -1 if actor was unused).
                        ## We do this to ensure faster prediction...if we use the least-recently used actor, it will
                        ## have to load the model into memory.
                        latest_last_completed_timestamp: float = max([
                            last_completed_timestamp for actor_usage, last_completed_timestamp, _ in actor_usages
                            if actor_usage == min_actor_usage
                        ])
                        ## Select an actor randomly among those with min usage and latest_last_completed_timestamp.
                        actor_id: str = random.choice([
                            actor_id
                            for actor_usage, last_completed_timestamp, actor_id in actor_usages
                            if actor_usage == min_actor_usage \
                               and last_completed_timestamp == latest_last_completed_timestamp
                        ])
                        actor_comp: ActorComposite = only_item([
                            actor_comp for actor_comp in self.model
                            if actor_comp.actor_id == actor_id
                        ])
                        if self.verbosity >= 3:
                            actor_ip_address: str = accumulate(actor_comp.actor.get_ip_address.remote())
                            debug_logger(
                                f'Actor usages:\n{actor_usages}\n'
                                f'(Submitting part#{part_i} ({len(part_data)} rows, prediction batch_size={batch_size}) '
                                f'to actor "{actor_comp.actor_id}" on IP address {actor_ip_address})'
                            )
                    else:
                        raise NotImplementedError(f'Unsupported `load_balancing_strategy`: {load_balancing_strategy}')

                    predictions.append(
                        actor_comp.actor.evaluate_shard.remote(
                            part_data,
                            dataset_params={
                                **dataset_params,
                                **dict(data_idx=part_i),
                            },
                            input_len=input_len,
                            batch_size=batch_size,
                            predictions_destination=predictions_destination,
                            return_predictions=return_predictions or metrics is not None,
                            row_counter=row_counter,
                            failure_action=failure_action,
                            data_loading_strategy=data_loading_strategy,
                            **kwargs,
                        )
                    )
                    submissions_progress_bar.update(1)
                    ## Track progress while submitting, since submitting can take upto an hour:
                    new_rows_completed: int = ray.get(row_counter.get_rows_completed.remote())
                    rows_completed_progress_bar.update(new_rows_completed - rows_completed)
                    rows_completed: int = new_rows_completed
            else:
                raise NotImplementedError(f'Unsupported `data_loading_strategy`: {data_loading_strategy}')

            ## Track till all rows are completed:
            rows_completed_start_time: float = time.time()
            while rows_completed < input_len and time.time() < rows_completed_start_time + evaluation_timeout:
                time.sleep(self.progress_update_frequency)
                new_rows_completed: int = ray.get(row_counter.get_rows_completed.remote())
                rows_completed_progress_bar.update(new_rows_completed - rows_completed)
                rows_completed: int = new_rows_completed
            rows_completed_progress_bar.success(f'Evaluated {input_len_str} rows')

            if return_predictions or metrics is not None:
                debug_logger(f'Collecting {len(predictions)} predictions...')
                accumulate_progress_bar: ProgressBar = ProgressBar.of(
                    progress_bar,
                    total=input_len,
                    desc=f'Collecting {input_len_str} rows',
                    initial=0,
                )
                evaluated_predictions: List[Predictions] = []
                evaluated_metrics: List[Metric] = []
                for pred_i, pred in enumerate(predictions):
                    debug_logger(f'Collecting prediction#{pred_i}: type={type(pred)}, val={pred}')
                    try:
                        pred: Optional[Predictions] = get_result(pred, wait=10.0)
                    except Exception as e:
                        main_logger(f'Error while collecting prediction#{pred_i}:\n{format_exception_msg(e)}')
                        raise e
                    if pred is None:
                        debug_logger(f'Collected prediction#{pred_i}: found None.')
                    else:
                        debug_logger(f'Collected prediction#{pred_i}: {type(pred)}.')
                        evaluated_predictions.append(pred)
                        accumulate_progress_bar.update(len(pred))
                if len(evaluated_predictions) == 0:
                    debug_logger(f'No results. evaluated_predictions={evaluated_predictions}')
                    accumulate_progress_bar.failed(f'No results')
                    raise ValueError(f'All predictions returned from actors were None.')
                evaluated_predictions: Predictions = Predictions.concat(evaluated_predictions, error_on_empty=True)
                debug_logger(f'Concatenated into {len(evaluated_predictions)} rows of predictions.')
                if len(evaluated_predictions) != input_len:
                    num_failed_rows: int = input_len - len(evaluated_predictions)
                    num_failed_rows_str: str = StringUtil.readable_number(
                        num_failed_rows,
                        decimals=1,
                        short=True
                    )
                    accumulate_progress_bar.failed(f'Failed for {num_failed_rows_str} rows')
                    if allow_partial_predictions is False:
                        raise ValueError(
                            f'Partial predictions returned: expected {input_len} rows, '
                            f'but only got {len(evaluated_predictions)} rows from actors.'
                        )
                else:
                    accumulate_progress_bar.success(f'Collected {input_len_str} rows')
                if metrics is not None:
                    for metric in metrics:
                        evaluated_metrics.append(evaluated_predictions.evaluate(metric=metric))
            else:
                ## Wait for predictions to complete:
                wait(predictions)
            timer.stop()
            main_logger(
                self._evaluate_end_msg(
                    input_len=input_len,
                    timer=timer,
                    num_actors_created=num_actors_created,
                    tracker=tracker,
                )
            )
            return evaluated_predictions, evaluated_metrics
        except KeyboardInterrupt as e:
            raise e
        finally:
            if 'row_counter' in locals():
                accumulate(ray.kill(row_counter))
                del row_counter
            if self.cache_timeout is None:  ## If we don't have a timeout, delete actors after every execution.
                self.cleanup_model()
            return evaluated_predictions, evaluated_metrics

    def _get_actor_usages(self) -> List[Tuple[int, float, str]]:
        actor_usages: List[Tuple[int, float, str]] = accumulate([
            (
                actor_comp.request_counter.num_pending_requests.remote(),
                actor_comp.request_counter.last_completed_timestamp.remote(),
                actor_comp.actor_id,
            )
            for actor_comp in self.model
        ])
        return actor_usages

    def _run_evaluation_progress_bar(self, progress_bar: Optional[Dict], **kwargs) -> Optional[Dict]:
        if self.verbosity >= 2:
            return progress_bar
        return None

    def _evaluate_start_msg(self, *, tracker: Tracker, **kwargs) -> str:
        if tracker.tracker_name == 'noop':
            tracker_msg: str = 'Logs will not be tracked.'
        else:
            tracker_msg: str = f'{tracker.class_name}@{tracker.id} will save logs to: "{tracker.log_dir}"'
        return f'\nEvaluating using nested evaluator: ' \
               f'{StringUtil.pretty(self._create_nested_evaluator_params(**kwargs))}' \
               f'\n{tracker_msg}'

    def _evaluate_end_msg(
            self,
            *,
            input_len: int,
            timer: Timer,
            num_actors_created: int,
            tracker: Tracker,
            **kwargs,
    ) -> str:
        if tracker.tracker_name == 'noop':
            tracker_msg: str = 'Logs have not been tracked.'
        else:
            tracker_msg: str = f'{tracker.class_name}@{tracker.id} has saved logs to "{tracker.log_dir}"'
        return f'Evaluated {input_len} rows in {timer.time_taken_str} ' \
               f'using {num_actors_created} model-copies ' \
               f'({input_len / timer.time_taken_sec:.3f} rows/sec or ' \
               f'{input_len / (num_actors_created * timer.time_taken_sec):.3f} rows/sec/copy)\n{tracker_msg}'
