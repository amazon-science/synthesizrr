import json
import logging
import warnings
from typing import *
import types
from types import ModuleType
import time, os, gc, io, pickle, math, copy, pandas as pd, numpy as np, random
import ray
from ray import tune, air
from ray.runtime_env import RuntimeEnv as RayRuntimeEnv
from ray.tune.search import SEARCH_ALG_IMPORT, Searcher, Repeater
from synthergent.base.util import set_param_from_alias, MappedParameters, str_normalize, append_to_keys, \
    parameterized_flatten, FileSystemUtil, get_default, all_are_not_none, safe_validate_arguments, StringUtil, Log, \
    all_are_none, pd_partial_column_order, Timer, format_exception_msg, retry, any_are_not_none, is_null, type_str
from synthergent.base.util.aws import S3Util
from synthergent.base.data import FileMetadata
from synthergent.base.constants import DataLayout, DataSplit, MLTypeSchema, Storage, Task
from synthergent.base.framework.dl.torch import PyTorch
from synthergent.base.framework.tracker.Tracker import Tracker
from synthergent.base.framework.trainer.Trainer import Trainer, KFold
from synthergent.base.framework.task_data import DataSplit, Datasets, Dataset
from synthergent.base.framework.predictions import Predictions
from synthergent.base.framework.algorithm import Algorithm, TaskOrStr
from synthergent.base.framework.metric import Metric, TabularMetric, PercentageMetric, CountingMetric, Metrics, metric_stats_str
from ray.tune.experiment import Trial, Experiment
from synthergent.base.framework.ray_base import RayInitConfig
from pydantic import conint, confloat, root_validator, validator, Extra
from pydantic.typing import Literal
from functools import partial

RayTuneTrainer = "RayTuneTrainer"

_RAY_TRIAL_ID: str = 'trial_id'
_RAY_EXPERIMENT_ID: str = 'experiment_id'
_RAY_KFOLD_CURRENT_FOLD_NAME: str = 'current_fold_name'
_RAY_EPOCH_NUM: str = 'epoch_num'
_RAY_STEPS_COMPLETED: str = 'steps_completed'
_RAY_EST_TIME_REMAINING: str = 'est_time_remaining'
_RAY_TRAINING_ITERATION: str = 'training_iteration'
_RAY_HYPERPARAMS_STR: str = 'hyperparams_str'
_RAY_HYPERPARAMS: str = 'hyperparams'
_RAY_METRIC_IS_DATAFRAME_PREFIX: str = 'RAY_METRIC_DATAFRAME::'


class RayTuneTrainerError(Exception):
    pass


class RayTuneTrainerTuneError(Exception):
    pass


class RayTuneTrainerFinalModelsError(Exception):
    pass


## Overview of what happens when you call tuner.fit():
## https://docs.ray.io/en/latest/tune/tutorials/tune-lifecycle.html#the-execution-of-a-trainable
##
## TL;DR:
## (1) Your python script (or Jupyter notebook) creates an instance of ray.tune.Tuner, and calls Tuner.fit()
##
## (2) ray.tune.Tuner uses Ray Actors to parallelize the evaluation of multiple hyperparameter configurations.
##  - Each Ray Actor is a Python process, whose job is to execute an instance of the Trainable.
##  - The definition of the Trainable will be serialized (via cloudpickle) before being sent to each Ray actor.
##  This allows you to execute simple Python classes (rather than Dockers) on the Ray cluster.
##
## (3) Each instance of the Trainable has an associated instance of ray.tune.experiment.Trial, which holds
##  the metadata for one model training execution (this is a 1:1 association from Trainable to Trial).
##  Trials are not themselves distributed; each runs on a corresponding actor.
##
## ==>> INTERNAL WORKING OF RAY COMPONENTS:
##  Calling tune.Tuner.fit() creates an object of ray.tune.execution.trial_runner.TrialRunner, which manages
##  all Trials.
##  TrialRunner is the main "driver" for the training loop. It uses other ray components as follows:
##  (a) Uses a `TrialScheduler` instance to prioritize and execute trials.
##      Ref: https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
##      In Ray Tune, by default we submit models-training jobs one-by-one, in FIFO order. However, this is
##      not the only option. Some hyperparameter optimization algorithms use sophisticated scheduling algorithms,
##      which in ray are called "trial schedulers". TrialSchedulers operate over a set of possible trials to run,
##      prioritizing trial execution given available cluster resources. TrialSchedulers are given the ability to
##      pause trials, reorder/prioritize incoming trials, clone trials, kill poor-performing trials,
##      alter hyperparameters of a running trial, etc.
##      - Examples of TrialSchedulers (from ray.tune.schedulers): FIFOScheduler (default), HyperBandScheduler,
##      ASHAScheduler, etc.
##      - ray.tune.schedulers.trial_scheduler.TrialScheduler is the abstract base class for all scheduling algos.
##  (b) Uses a `SearchAlgorithm` instance to retrieve new hyperparameter-combinations to evaluate:
##      Ref: https://docs.ray.io/en/latest/tune/tutorials/tune-lifecycle.html#searchalg
##      TL;DR:
##      In Ray Tune, the SearchAlgorithm is a user-provided object that is used for querying new hyperparameter
##      configurations to evaluate. SearchAlgorithms will be notified every time a trial finishes executing one
##      training step (of train()), every time a trial errors, and every time a trial completes.
##  (c) Handles the fault tolerance logic.
##
##  ==>> LIFECYCLE OF A SINGLE TRIAL:
##  A trial’s life cycle consists of 6 stages:
##  1. Initialization (generation): A trial is first generated as a hyperparameter sample, and its
##  parameters are configured according to what was provided in Tuner. Trials are then assigned PENDING
##  status, and placed into a queue to be executed.
##  2. PENDING: A pending trial is a trial to be executed on the machine. Every trial is configured with
##  resource values (cpus and gpus).
##  Before running a trial, the TrialRunner will check whether there are available resources on the
##  cluster (see Specifying Task or Actor Resource Requirements). It will compare the available resources
##  with the resources required by the trial.
##  Whenever the trial’s resource values are available, TrialRunner will run the trial (by starting a ray
##  actor holding the resources config, and the training function).
##  3. RUNNING: A running trial is assigned a Ray Actor. There can be multiple running trials in parallel.
##      ==>> EXECUTION OF A TRIAL (i.e. a Trainable instance):
##      Ref: https://docs.ray.io/en/latest/tune/tutorials/tune-lifecycle.html#the-execution-of-a-trainable
##      TL;DR:
##      - If the Trainable is a class, it will be executed iteratively by calling train/step.
##      After each invocation, the driver is notified that a “result dict” is ready. The driver will then
##      pull the result via ray.get.
##      !IMP Do not use session.report within a Trainable class, instead use
##      Ref: https://docs.ray.io/en/latest/tune/api_docs/trainable.html#function-api
##      - If the trainable is a callable or a function, it will be executed on the Ray actor process on a
##      separate execution thread. Whenever session.report is called, the execution thread is paused and
##      waits for the driver to pull a result. After pulling, the actor’s execution thread will resume.
##  4. ERRORED: If a running trial throws an exception, Tune will catch that exception and mark the trial as
##  errored. Note that exceptions can be propagated from an actor to the main Tune driver process.
##  If max_retries is set, Tune will set the trial back into “PENDING” and later start it from the last
##  checkpoint.
##  5. TERMINATED: A trial is terminated if it is stopped by a Stopper/Scheduler. If using the Function API,
##  the trial is also terminated when the function stops.
##  6. PAUSED: A trial can be paused by a Trial scheduler. This means that the trial’s actor will be
##  stopped. A paused trial can later be resumed from the most recent checkpoint.

def _ray_metric_str(data_split: DataSplit, metric: Metric) -> str:
    return f'{data_split.capitalize()}/{metric.display_name}'


def _ray_col_detect_data_split(col: str) -> Optional[DataSplit]:
    ## Returns None if it is not a metric column name, otherwise returns the data-split.
    for data_split in list(DataSplit):
        if col.startswith(f'{data_split.capitalize()}/'):
            return data_split
    return None


def _ray_col_is_metric(col: str) -> bool:
    return _ray_col_detect_data_split(col) is not None


def _ray_put_metric_value(metric: Metric) -> Any:
    if isinstance(metric, TabularMetric):
        assert isinstance(metric.value, pd.DataFrame)
        return _RAY_METRIC_IS_DATAFRAME_PREFIX + metric.value.to_json(orient='records')
    value: Any = metric.value
    if isinstance(value, (int, float, str)) or np.issubdtype(type(value), np.number):
        return value
    buf = io.BytesIO()
    pickle.dump(value, buf)
    buf.seek(0)  ## necessary to start reading at the beginning of the "file"
    return buf


def _ray_get_metric_value(value: Optional[Any]) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, str) and value.startswith(_RAY_METRIC_IS_DATAFRAME_PREFIX):
        return pd.DataFrame(json.loads(value.removeprefix(_RAY_METRIC_IS_DATAFRAME_PREFIX)))
    if isinstance(value, (int, float, str)) or np.issubdtype(type(value), np.number):
        return value
    if not isinstance(value, io.BytesIO):
        raise ValueError(f'Expected metric value to be of type BytesIO, found type: {type_str(value)}')
    value.seek(0)
    return pickle.load(value)


def _ray_convert_metrics_dataframe(metrics_dataframe: pd.DataFrame) -> pd.DataFrame:
    for col in metrics_dataframe.columns:
        if _ray_col_is_metric(col):
            metrics_dataframe[col] = metrics_dataframe[col].apply(_ray_get_metric_value)
    return metrics_dataframe


def _ray_logger(text: str, verbosity: int, logging_fn: Callable, prefix: Optional[str] = None):
    prefix: str = get_default(prefix, '')
    text: str = f'{prefix}{text}'
    if verbosity == 0:  ## Don't log anything.
        return
    logging_fn(text)


def _ray_agg_final_model_metric_stats(
        trialwise_final_model_metrics: Dict[str, Metrics],
        data_split: DataSplit,
) -> Dict[str, Dict[str, Union[int, float]]]:
    student_metrics: Dict[str, Dict[str, Any]] = {}
    for trial_id, trial_metrics in trialwise_final_model_metrics.items():
        for student_metric in trial_metrics[data_split]:
            assert isinstance(student_metric, Metric)
            student_metric_name: str = student_metric.display_name
            student_metrics.setdefault(student_metric_name, {})
            student_metrics[student_metric_name].setdefault('values', [])
            student_metrics[student_metric_name]['values'].append(student_metric.value)
            student_metrics[student_metric_name].setdefault('metric_class', None)
            if student_metrics[student_metric_name]['metric_class'] is None:
                student_metrics[student_metric_name]['metric_class'] = type(student_metric)
            else:
                assert student_metrics[student_metric_name]['metric_class'] == type(student_metric)
    student_metrics_stats: Dict[str, Dict[str, Union[int, float]]] = {}
    for student_metric_name, student_metric_d in student_metrics.items():
        if issubclass(student_metric_d['metric_class'], (PercentageMetric, CountingMetric)):
            student_metrics_stats.setdefault(student_metric_name, {})
            vals: pd.Series = pd.Series(student_metric_d['values'])
            student_metrics_stats[student_metric_name]['mean'] = vals.mean()
            student_metrics_stats[student_metric_name]['std'] = vals.std()
            student_metrics_stats[student_metric_name]['mean+std'] = vals.mean() + vals.std()
            student_metrics_stats[student_metric_name]['mean-std'] = vals.mean() - vals.std()
            student_metrics_stats[student_metric_name]['min'] = vals.min()
            student_metrics_stats[student_metric_name]['max'] = vals.max()
            student_metrics_stats[student_metric_name]['median'] = vals.median()
            student_metrics_stats[student_metric_name]['count'] = len(vals)
    return student_metrics_stats


class HyperparameterSearchSpace(MappedParameters):
    _mapping = append_to_keys(
        prefix='tune.',
        d={
            ## Ref: https://docs.ray.io/en/latest/tune/api_docs/search_space.html
            ## uniform(-5, -1) => Sample a float uniformly between -5.0 and -1.0
            "uniform": tune.uniform,

            ## quniform(3.2, 5.4, q=0.2) -> Sample a float uniformly between 3.2 and 5.4, rounding to multiples of 0.2
            "quniform": tune.quniform,

            ## loguniform(1e-4, 1e-2) -> Sample a float uniformly between 0.0001 and 0.01, while sampling in log space
            "loguniform": tune.loguniform,

            ## qloguniform(1e-4, 1e-1, q=5e-5, base=10) -> Sample a float uniformly between 0.0001 and 0.1, while
            ## sampling in base-10 log space and rounding to multiples of 0.00005
            "qloguniform": tune.qloguniform,

            ## randn(10, 2) -> Sample a random float from a normal distribution with mean=10 and sd=2
            "randn": tune.randn,

            ## qrandn(10, 2, q=0.2) -> Sample a random float from a normal distribution with mean=10 and sd=2, rounding
            ## to multiples of 0.2
            "qrandn": tune.qrandn,

            ## tune.randint(-9, 15) -> Sample an integer uniformly between -9 (inclusive) and 15 (exclusive)
            "randint": tune.randint,

            ## qrandint(-21, 12, q=3) -> Sample a random uniformly between -21 (inclusive) and 12 (inclusive (!))
            ## rounding to multiples of 3 (includes 12).
            ## If q=1, then randint is called instead with the upper bound exclusive
            "qrandint": tune.qrandint,

            ## lograndint(1, 10, base=10) -> Sample an integer uniformly between 1 (inclusive) and 10 (exclusive), while
            ## sampling in base-10 log space
            "lograndint": tune.lograndint,

            ## qlograndint(1, 10, q=2, base=10) -> Sample an integer uniformly between 1 (inclusive)
            ## and 10 (inclusive (!)), while sampling in base-10 log space and rounding to multiples of 2.
            ## If q is 1, then lograndint is called instead with the upper bound exclusive.
            "qlograndint": tune.qlograndint,

            ## choice(["a", "b", "c"]) -> Sample an option uniformly from the specified choices.
            "choice": tune.choice,

            ## NOTE: `tune.grid_search` has the following weird behavior with num_samples:
            ##      `tune.grid_search`: Every value will be sampled ``num_samples`` times (``num_samples`` is the
            ##                          parameter you pass to ``tune.TuneConfig`` in ``Tuner``)
            ##      For this reason, if `tune.grid_search` is passed, we check that num_samples=1
            ##      and all other search spaces are also `tune.grid_search`.
            "grid_search": tune.grid_search,

            ## NOTE: we do not support `tune.sample_from` since it would require defining functions
        }
    )


class SearchAlgorithm(MappedParameters):
    _mapping = {
        **{
            k: importer()
            for k, importer in SEARCH_ALG_IMPORT.items()
        },
        **{
            'grid': SEARCH_ALG_IMPORT['variant_generator']()
        }
    }


class MaximumStepsStopper(tune.stopper.Stopper):
    """Stop trials after reaching a maximum number of steps.
    Ref: https://docs.ray.io/en/latest/_modules/ray/tune/stopper/maximum_iteration.html

    Args:
        max_steps: Number of steps before stopping a trial.
    """

    def __init__(self, max_steps: int, steps_increment: int):
        self._max_steps = max_steps
        self._steps_increment = steps_increment

    def __call__(self, trial_id: str, result: Dict) -> bool:
        # print(f'Trial id: {trial_id}, results: {result}')
        return result[_RAY_STEPS_COMPLETED] >= self._max_steps

    def stop_all(self):
        return False


class AlgorithmTrainable(tune.Trainable):
    """
    Ray Trainable for a single instance of Algorithm.
    Ref: https://docs.ray.io/en/latest/tune/api_docs/trainable.html#trainable-class-api
    TL;DR: these are the steps followed:
    (1) Ray Tune will create a Trainable instance on a separate process (using the Ray Actor API).
    (2) `setup` function is invoked once training starts.
    (3) `step` is invoked multiple times. Each time, the Trainable object executes one logical iteration of training in
    the tuning process, which may include one or more iterations of actual training.
    As a rule of thumb, the execution time of step should be large enough to avoid overheads (i.e. more than a few
    seconds), but short enough to report progress periodically (i.e. at most a few minutes).
    In Deep Learning models, `step` should be an entire epoch. Refer to the PyTorch examples:
        1. def train(): https://docs.ray.io/en/latest/tune/examples/includes/mnist_pytorch.html
        2. Trainable: https://docs.ray.io/en/latest/tune/examples/includes/mnist_pytorch_trainable.html
    (4) `cleanup` is invoked when training is finished.
    """

    def setup(
            self,
            config: Dict,
            ray_tune_trainer: Dict,
            AlgorithmClass: str,
            datasets: Dict,
            metrics: Optional[Dict],
            k_fold: Dict,
            save_model: Optional[Dict],
            **kwargs
    ):
        self.timer: Timer = Timer(silent=True)
        self.k_fold: KFold = KFold(**k_fold)
        k_fold_current_fold: int = config.pop(tune.search.repeater.TRIAL_INDEX, 0)
        self.shard: Tuple[int, int] = (k_fold_current_fold, self.k_fold.num_folds)
        self.ray_tune_trainer: RayTuneTrainer = RayTuneTrainer(**ray_tune_trainer)
        self.trainable_logger: Callable = partial(
            _ray_logger,
            verbosity=1 if self.ray_tune_trainer.verbosity >= 2 else 0,  ## Only log when asking for detailed logging.
            logging_fn=print,
            prefix=f'(trial_id={self.trial_id}) ',
        )
        ## Register custom user-defined classes on the Ray Actor which will be used to train the model.
        for custom_fn in self.ray_tune_trainer.custom_definitions:
            custom_fn()

        try:
            AlgorithmClass: Type[Algorithm] = Algorithm.get_subclass(AlgorithmClass)
            self.AlgorithmClass = AlgorithmClass
        except Exception as e:
            raise ValueError(
                f'Could not fetch Algorithm class using key "{str(AlgorithmClass)}" using Algorithm Registry having '
                f'following classes:\n{Algorithm.subclasses()}\nError encountered:\n{format_exception_msg(e)}'
            )
        try:
            self.hyperparams: Algorithm.Hyperparameters = AlgorithmClass.Hyperparameters(**config)
        except Exception as e:
            raise ValueError(
                f'Could not create Hyperparameters instance for Algorithm "{str(self.AlgorithmClass)}" '
                f'using following config:\n{config}\nError encountered:\n{format_exception_msg(e)}'
            )

        self.epochs: Optional[int] = None
        self.epoch_num: Optional[int] = None
        self.steps: Optional[int] = None
        self.step_range: Optional[Tuple[int, int]] = None
        if self.hyperparams.epochs is not None:
            self.epochs: int = self.hyperparams.epochs
            self.epoch_num: int = 0
        elif self.hyperparams.steps is not None:
            self.steps: int = self.hyperparams.steps
            ## Will start from 1 & metric_eval_frequency on first call to the `step` function:
            self.step_range: Tuple[int, int] = (1 - self.ray_tune_trainer.metric_eval_frequency, 0)
        else:
            raise ValueError(f'Either `epochs` or `steps` hyperparams must be non-None')
        # datasets['datasets']: Dict[DataSplit, Dataset] = {
        #     DataSplit(data_split): Dataset.of(
        #         data=FileMetadata(**dataset['data']),
        #         **remove_keys(dataset, ['data'])
        #     )
        #     for data_split, dataset in datasets['datasets'].items()
        # }
        self.datasets: Datasets = Datasets(**datasets)
        self.metrics: Optional[Metrics] = None if metrics is None else Metrics(**metrics)
        self.save_model: Optional[FileMetadata] = None if save_model is None else FileMetadata(**save_model)
        self.last_saved_model_dir: Optional[FileMetadata] = None
        self.kwargs: Dict = kwargs

        train_stats: Optional[Metrics] = self.AlgorithmClass.calculate_dataset_stats(
            dataset=self.datasets[DataSplit.TRAIN],
            batch_size=self.ray_tune_trainer.stats_batch_size,
            data_split=DataSplit.TRAIN,
        )
        train_dataset_length: Metric = train_stats.find(data_split=DataSplit.TRAIN, select='row_count')
        self.train_dataset_length: int = train_dataset_length.value
        self.model: Algorithm = Algorithm.of(**{
            'name': self.AlgorithmClass,
            'task': self.ray_tune_trainer.task,
            'hyperparams': self.hyperparams,
            'stats': train_stats,
            **self.kwargs,
        })

    def update_iter_count(self):
        if not self.timer.has_started:
            self.timer.start()
        if self.epochs is not None:
            self.epoch_num += 1
        elif self.steps is not None:
            self.step_range: Tuple[int, int] = (
                min(  ## E.g. 1
                    self.step_range[0] + self.ray_tune_trainer.metric_eval_frequency,
                    self.steps
                ),
                min(  ## E.g. 10 when metric_eval_frequency=10
                    self.step_range[1] + self.ray_tune_trainer.metric_eval_frequency,
                    self.steps
                )
            )

    def cur_iter_str(self) -> str:
        if self.epochs is not None:
            return self.ray_tune_trainer._cur_epoch_str(
                epoch_num=self.epoch_num,
                epochs=self.epochs,
            )
        elif self.steps is not None:
            step_num: int = self.step_range[0]  ## Starting step num
            return self.ray_tune_trainer._cur_step_str(
                step_num=step_num,
                steps=self.steps,
                steps_increment=self.ray_tune_trainer.metric_eval_frequency,
            )
        raise ValueError(f'Either `epochs` or `steps` should be non-None')

    def step(self):
        ## Here, one "step" is an epoch over the dataset, after which we might return various metrics.
        ## Note that we do not need to writing the training for-loop ourselves, since we have set a stopping criterion
        ## as tune.stopper.MaximumIterationStopper(max_iter=...), it will stop automatically.
        self.update_iter_count()
        iter_timer: Timer = Timer(silent=True)
        iter_timer.start()

        train_dataset, train_metrics, \
        validation_dataset, validation_metrics, \
        test_dataset, test_metrics = self.ray_tune_trainer._extract_datasets_and_metrics(
            datasets=self.datasets,
            metrics=self.metrics,
        )
        train_kfold_kwargs: Dict = {}
        validation_kfold_kwargs: Dict = {}
        if self.k_fold.num_folds > 1:
            validation_dataset: Dataset = train_dataset
            ## For the train split of k-fold, use the k_fold seed for sharding, and the regular seed for shuffling
            ## within a shard. We use reverse-sharding to pick pieces NOT in the validation split:
            train_kfold_kwargs['shard'] = self.shard
            train_kfold_kwargs['reverse_sharding'] = True
            ## Always shuffle shards. This ensures we get consistent train-val splits for both
            ## training and metric-calculation:
            train_kfold_kwargs['shard_shuffle'] = True
            train_kfold_kwargs['shard_seed'] = self.k_fold.seed  ## Always set.
            train_kfold_kwargs['seed'] = self.ray_tune_trainer.seed
            ## For the validation split of k-fold, use the k_fold seed for sharding, and the regular seed for shuffling
            ## within a shard:
            validation_kfold_kwargs['shard'] = self.shard
            validation_kfold_kwargs['reverse_sharding'] = False
            ## Always shuffle shards. This ensures we get consistent train-val splits for both
            ## training and metric-calculation:
            validation_kfold_kwargs['shard_shuffle'] = True
            validation_kfold_kwargs['shard_seed'] = self.k_fold.seed  ## Always set.
            validation_kfold_kwargs['seed'] = self.ray_tune_trainer.seed
            # self.trainable_logger(
            #     f'Epoch: {self.epoch_num}, '
            #     f'\ntrain_kfold_kwargs: {train_kfold_kwargs}'
            #     f'\nvalidation_kfold_kwargs: {validation_kfold_kwargs}'
            # )
        cur_iter_str: str = self.cur_iter_str()
        self.trainable_logger(f'>> Running {cur_iter_str}...')

        ## Run training phase:
        training_timer: Timer = Timer(silent=True)
        training_timer.start()
        step_num: Optional[int] = self.step_range[0] if self.steps is not None else None  ## Starting step num
        self.ray_tune_trainer._train_loop(
            model=self.model,
            train_dataset=train_dataset,
            train_dataset_length=self.train_dataset_length,
            epochs=self.epochs,
            epoch_num=self.epoch_num,
            steps=self.steps,
            step_num=step_num,
            batch_size=self.model.hyperparams.batch_size,
            batch_logger=None,  ## Do not log when running tuning.
            **dict(
                **self.kwargs,
                **train_kfold_kwargs,
            ),
        )
        training_timer.stop()
        self.trainable_logger(f'...training phase of {cur_iter_str} took {training_timer.time_taken_str}.')

        ## Run metric-evaluation phase:
        metric_eval_timer: Timer = Timer(silent=True)
        metric_eval_timer.start()
        self.trainable_logger('Train metrics:')
        train_metrics_evaluated: Optional[List[Metric]] = self.ray_tune_trainer._evaluate_metrics(
            model=self.model,
            dataset=train_dataset,
            metrics=train_metrics,
            epoch_num=self.epoch_num,
            step_num=step_num,
            **dict(
                **self.kwargs,
                **train_kfold_kwargs,
            ),
        )
        self.trainable_logger('Validation metrics:')
        validation_metrics_evaluated: Optional[List[Metric]] = self.ray_tune_trainer._evaluate_metrics(
            model=self.model,
            dataset=validation_dataset,
            metrics=validation_metrics,
            epoch_num=self.epoch_num,
            step_num=step_num,
            **dict(
                **self.kwargs,
                **validation_kfold_kwargs,
            ),
        )
        test_metrics_evaluated: Optional[List[Metric]] = None
        if self.ray_tune_trainer.is_final_iter(
                epoch_num=self.epoch_num,
                epochs=self.epochs,
                step_num=step_num,
                steps=self.steps,
                steps_increment=self.ray_tune_trainer.metric_eval_frequency,
        ) or self.ray_tune_trainer.evaluate_test_metrics == 'each_iter':
            test_metrics_evaluated: Optional[List[Metric]] = self.ray_tune_trainer._evaluate_metrics(
                model=self.model,
                dataset=test_dataset,
                metrics=test_metrics,
                epoch_num=self.epoch_num,
                step_num=step_num,
                force=True,
                **self.kwargs,
            )
        metric_eval_timer.stop()
        if any_are_not_none(train_metrics_evaluated, validation_metrics_evaluated, test_metrics_evaluated):
            self.trainable_logger(f'...evaluation phase of {cur_iter_str} took {metric_eval_timer.time_taken_str}.')

        ## Increment epochs or steps:
        if self.epochs is not None:
            epochs_completed: int = self.epoch_num
            epochs_remaining: int = max(self.epochs - epochs_completed, 0)
            epochs_speed: float = epochs_completed / self.timer.time_taken_sec
            est_time_remaining: float = epochs_remaining / epochs_speed
        elif self.steps is not None:
            steps_completed: int = self.step_range[1]
            steps_remaining: int = self.steps - steps_completed
            steps_speed: float = steps_completed / self.timer.time_taken_sec
            est_time_remaining: float = steps_remaining / steps_speed
        else:
            raise ValueError(f'Either `epochs` or `steps` should be non-None')

        iter_timer.stop()
        self.trainable_logger(f'...{cur_iter_str} took {iter_timer.time_taken_str}.')
        self.trainable_logger('')
        return {
            **{
                _ray_metric_str(
                    data_split=DataSplit.TRAIN,
                    metric=train_metric,
                ): _ray_put_metric_value(train_metric)
                for train_metric in get_default(train_metrics_evaluated, [])
            },
            **{
                _ray_metric_str(
                    data_split=DataSplit.VALIDATION,
                    metric=validation_metric,
                ): _ray_put_metric_value(validation_metric)
                for validation_metric in get_default(validation_metrics_evaluated, [])
            },
            **{
                _ray_metric_str(
                    data_split=DataSplit.TEST,
                    metric=test_metric,
                ): _ray_put_metric_value(test_metric)
                for test_metric in get_default(test_metrics_evaluated, [])
            },
            **{
                _RAY_EPOCH_NUM: self.epoch_num,
                _RAY_STEPS_COMPLETED: None if self.steps is None else self.step_range[1],
                _RAY_EST_TIME_REMAINING: StringUtil.readable_seconds(est_time_remaining, short=True, decimals=1),
            },
        }

    # @property
    # def is_final_iter(self) -> bool:
    #     if self.epochs is not None:
    #         return self.epoch_num == self.epochs
    #     elif self.steps is not None:
    #         return self.step_range[1] == self.steps
    #     raise ValueError(f'One of `epochs` and `steps` must be non-None.')

    def save_checkpoint(self, tmp_checkpoint_dir: str):
        if self.epochs is not None:
            self.trainable_logger(f'Saving model to "{tmp_checkpoint_dir}" at epoch {self.epoch_num}')
        elif self.steps is not None:
            self.trainable_logger(f'Saving model to "{tmp_checkpoint_dir}" at step {self.step_range[1]}')
        FileSystemUtil.mkdir_if_does_not_exist(tmp_checkpoint_dir)
        model_dir: FileMetadata = FileMetadata.of(tmp_checkpoint_dir)
        self.model.save_params(model_dir=model_dir)
        self.model.save(model_dir=model_dir)
        if self.epochs is not None:
            self.trainable_logger(f'Saved model to "{tmp_checkpoint_dir}" at epoch {self.epoch_num}')
        elif self.steps is not None:
            self.trainable_logger(f'Saved model to "{tmp_checkpoint_dir}" at step {self.step_range[1]}')
        self.last_saved_model_dir: FileMetadata = model_dir
        return tmp_checkpoint_dir

    def load_checkpoint(self, checkpoint_path: str):
        self.model: Algorithm = Algorithm.of(
            name=self.AlgorithmClass,
            hyperparams=self.hyperparams,
            model_dir=FileMetadata.of(FileSystemUtil.get_dir(checkpoint_path)),
            **self.kwargs
        )

    def cleanup(self):
        ## Copy model checkpoint to destination.
        if all_are_not_none(self.save_model, self.last_saved_model_dir):
            try:
                trial_id: str = self.trial_id
                save_model_destination_dir: FileMetadata = self.save_model.subdir_in_dir(
                    trial_id,
                    mkdir=True,
                    return_metadata=True,
                )
                if not self.last_saved_model_dir.copy_to_dir(save_model_destination_dir):
                    raise OSError(
                        f'Error copying final trained model for trial_id={trial_id} '
                        f'to "{save_model_destination_dir.path}".'
                    )
                self.trainable_logger(
                    f'Successfully copied final trained model to "{save_model_destination_dir.path}".',
                )
            except Exception as e:
                print(format_exception_msg(e))
        self.model.cleanup()


class RayTuneTrainer(Trainer):
    aliases = ['ray', 'ray_tune']

    class Config(Trainer.Config):
        extra = Extra.allow

    class RunConfig(Trainer.RunConfig):
        ray_init: RayInitConfig = {}

    search_algorithm: Optional[SearchAlgorithm] = None
    search_space: Dict[str, Union[HyperparameterSearchSpace, str]] = {}
    allow_repeated_grid_search_sampling: bool = False
    num_models: conint(ge=1) = 1
    num_final_models: conint(ge=1) = 1  ## When we tune & then retrain.
    max_parallel_models: conint(ge=0) = 0  ## 0 = no limits to concurrency.
    timeout_seconds: Optional[conint(ge=1)] = None
    objective_metric: Optional[Union[Metric, Dict, str]] = None
    objective_type: Optional[Literal['min', 'minimize', 'max', 'maximize']] = None
    objective_dataset: Optional[DataSplit] = None
    max_dataset_rows_in_memory: int = int(1e6)
    progress_update_frequency: conint(ge=1) = 60
    resources_per_model: Dict[
        Literal['cpu', 'gpu'],
        Union[confloat(ge=0.0, lt=1.0), conint(ge=0)]
    ] = {'cpu': 1, 'gpu': 0}
    retrain_final_model: bool = True
    model_failure_retries: int = 0
    final_model_failure_behavior: Literal['warn', 'error'] = 'warn'
    tune_failure_retries: int = 0
    tune_failure_retry_wait: int = 60  ## Seconds

    @root_validator(pre=True)
    def ray_trainer_params(cls, params: Dict) -> Dict:
        ## Aliases for search_alg:
        set_param_from_alias(params, param='search_algorithm', alias=['search_alg'])
        set_param_from_alias(params, param='num_models', alias=[
            'max_models', 'max_samples', 'num_samples',
            'tune_num_models', 'tune_max_models', 'tune_num_samples', 'tune_max_samples',
        ])
        set_param_from_alias(params, param='num_final_models', alias=[
            'num_final_model',
            'max_final_models', 'max_final_model',
        ])
        set_param_from_alias(params, param='timeout_seconds', alias=['timeout', 'time_budget_s'])
        set_param_from_alias(params, param='objective_metric', alias=['hpo_metric'])
        set_param_from_alias(params, param='objective_dataset', alias=['hpo_dataset'])
        set_param_from_alias(params, param='objective_type', alias=['mode'])
        set_param_from_alias(params, param='max_parallel_models', alias=[
            ## Creates every combination as an alias, e.g. num_parallel_trials, max_parallel_jobs, etc.
            f'{x}_{y}_{z}' if x != '' else f'{y}_{z}'
            for x, y, z in
            parameterized_flatten(
                ['', 'num', 'max'],
                ['concurrent', 'parallel'],
                ['jobs', 'trials', 'models', 'workers']
            )
        ])
        set_param_from_alias(params, param='resources_per_model', alias=[
            'model_resources', 'resources',
        ])
        set_param_from_alias(params, param='retrain_final_model', alias=[
            'retrain_final_models',
            'retrain_final_model_after_tuning',
            'retrain_after_tuning',
            'train_final_model_after_tuning', 'train_final_models_after_tuning',
            'train_final_model', 'train_final_models',
            'train_after_tuning',
        ])
        set_param_from_alias(params, param='progress_update_frequency', alias=[
            'progress_update_freq', 'max_report_frequency', 'progress_update_seconds', 'progress_update_sec',
        ])
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

        ## Process K-fold:
        k_fold = params.get('k_fold', None)
        if k_fold is None:
            params['k_fold'] = KFold.NO_TUNING_K_FOLD
        elif isinstance(k_fold, int):
            params['k_fold'] = KFold(num_folds=k_fold)
        elif isinstance(k_fold, dict):
            params['k_fold'] = KFold(**k_fold)

        ## Convert values:
        if params.get('search_algorithm') is not None and params.get('search_space', {}) != {}:
            params['search_algorithm'] = SearchAlgorithm.of(params['search_algorithm'])
            params['objective_metric'] = Metric.of(params['objective_metric'])
            params['objective_dataset'] = DataSplit.from_str(params['objective_dataset'])
            if params['k_fold'].num_folds > 1:
                if params['objective_dataset'] is not DataSplit.VALIDATION:
                    warnings.warn(
                        f'Overriding `objective_dataset` to {DataSplit.VALIDATION} '
                        f'for K-Fold cross-validation'
                    )
                params['objective_dataset'] = DataSplit.VALIDATION
            params['objective_type'] = {  ## Map to Ray values
                'min': 'min',
                'minimize': 'min',
                'max': 'max',
                'maximize': 'max',
            }[str_normalize(params['objective_type'])]
            params['search_space']: Dict[str, HyperparameterSearchSpace] = {
                hp_name: HyperparameterSearchSpace.of(hp_search_space)
                for hp_name, hp_search_space in params['search_space'].items()
            }

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

    @property
    def ray_search_space(self) -> Dict[str, Any]:
        return {
            hp_name: hp_search_space.initialize()
            for hp_name, hp_search_space in self.search_space.items()
        }

    @property
    def epochs(self) -> Optional[int]:
        return self._create_hyperparams().epochs

    @property
    def steps(self) -> Optional[int]:
        return self._create_hyperparams().steps

    @property
    def objective_metric_display_name(self) -> str:
        return _ray_metric_str(data_split=self.objective_dataset, metric=self.objective_metric)

    def is_n_models_without_tuning(self) -> bool:
        return self.search_algorithm is None or len(self.search_space) == 0

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
    ) -> Tuple[Optional[tune.ResultGrid], Optional[tune.ResultGrid]]:
        final_model_results: Optional[tune.ResultGrid] = None
        tune_results: Optional[tune.ResultGrid] = None
        run_training_error: Optional[Exception] = None
        try:
            timer: Timer = Timer(silent=True)
            timer.start()

            cluster_resources: Dict = ray.cluster_resources()
            RAY_NUM_CPUS: int = int(cluster_resources['CPU'])
            RAY_NUM_GPUS: int = int(cluster_resources.get('GPU', 0))

            for data_split, dataset in datasets.datasets.items():
                if dataset.in_memory() and not dataset.data.is_lazy():
                    dataset_num_rows: int = len(dataset.data)
                    if dataset_num_rows > self.max_dataset_rows_in_memory:
                        raise ValueError(
                            f'Cannot train on in-memory datasets larger than {self.max_dataset_rows_in_memory} rows; '
                            f'please put the data in disk or S3 and process it lazily (e.g. using {DataLayout.DASK}).'
                        )
            main_logger: Callable = partial(
                _ray_logger,
                ## Unless we request silence (verbosity=0), print important information.
                verbosity=0 if self.verbosity == 0 else 1,
                logging_fn=tracker.info,
            )
            progress_bar: Optional[Dict] = self._ray_tune_trainer_progress_bar(progress_bar)
            if self.is_n_models_without_tuning():
                warnings.warn(
                    f'No tuning parameters set, hence parameters `search_algorithm`, `search_space`, and `k_fold` '
                    f'will be ignored, and only {self.num_models} models will be trained without hyperparameter tuning.'
                    f'\nAdditionally, num_final_models={self.num_final_models} will be ignored, '
                    f'num_models={self.num_models} models will be trained instead.'
                )
                tuner: tune.Tuner = self._create_tuner_n_models_without_tuning(
                    datasets=datasets,
                    metrics=metrics,
                    hyperparams=self._create_hyperparams(),
                    save_model=save_model,
                    num_samples=self.num_models,
                    **kwargs
                )
                main_logger(self._train_start_msg(completed=False, tracker=tracker, save_model=save_model))
                ## We use Tuner.fit() rather than tune.run() as the latter will be deprecated soon.
                ## Ref: https://discuss.ray.io/t/tune-run-vs-tuner-fit/7041/4
                results: tune.ResultGrid = retry(
                    tuner.fit,
                    wait=self.tune_failure_retry_wait,
                    retries=self.tune_failure_retries,
                )
                main_logger(self._train_start_msg(completed=True, tracker=tracker, save_model=save_model))
                final_model_results: tune.ResultGrid = results
                assert len(list(final_model_results)) == self.num_models

                try:
                    self.log_final_metrics(
                        final_model_results=final_model_results,
                        metrics=metrics,
                        data_split=DataSplit.TRAIN,
                        logger=main_logger,
                        is_n_models_without_tuning=True,
                        final_model_failure_behavior=self.final_model_failure_behavior,
                    )
                    self.log_final_metrics(
                        final_model_results=final_model_results,
                        metrics=metrics,
                        data_split=DataSplit.TEST,
                        logger=main_logger,
                        is_n_models_without_tuning=True,
                        final_model_failure_behavior=self.final_model_failure_behavior,
                    )
                except Exception as e:
                    main_logger(f'Error while logging metrics:\n{format_exception_msg(e)}')
                    pass
                timer.stop()
                main_logger(self._train_end_msg(timer=timer, tracker=tracker))
            else:
                ## Run tuning to get the best hyperparameters:
                if self.evaluate_test_metrics == 'each_iter' or self.retrain_final_model is False:
                    ## When not re-training final model, or we want to evaluate each iter,
                    ## pass the test dataset to the tuning jobs:
                    tuning_datasets: Datasets = datasets
                else:
                    ## Don't pass the test dataset to the tuning jobs:
                    tuning_datasets: Datasets = datasets.drop(DataSplit.TEST)
                tuner: tune.Tuner = self._create_tuner_for_tuning(
                    datasets=tuning_datasets,
                    metrics=metrics,
                    **kwargs
                )
                main_logger(self._train_start_msg(completed=False, tracker=tracker, save_model=None))
                tune_results: tune.ResultGrid = retry(
                    tuner.fit,
                    wait=self.tune_failure_retry_wait,
                    retries=self.tune_failure_retries,
                )
                main_logger(self._train_start_msg(completed=True, tracker=tracker, save_model=None))
                best_hyperparams, best_objective_metric_value = self.get_best_hyperparams(tune_results)

                ## Log messages before retraining final model:
                main_logger(self.tune_results_message(tune_results, save_model=save_model))

                if self.retrain_final_model:
                    ## Retrain final models:
                    ## TODO: expose option to combine train & validation dataset into single dataset for model retraining.
                    retrain_timer: Timer = Timer(silent=True)
                    retrain_timer.start()
                    tuner: tune.Tuner = self._create_tuner_n_models_without_tuning(
                        datasets=datasets,
                        metrics=metrics,
                        hyperparams=best_hyperparams,
                        save_model=save_model,
                        num_samples=self.num_final_models,
                        **kwargs
                    )
                    final_model_results: tune.ResultGrid = retry(tuner.fit, retries=2)
                    assert len(list(final_model_results)) == self.num_final_models
                    retrain_timer.stop()
                    main_logger(self.tune_retrain_message(
                        final_model_results,
                        retrain_timer=retrain_timer,
                        save_model=save_model,
                    ))
                    try:
                        self.log_final_metrics(
                            final_model_results=final_model_results,
                            metrics=metrics,
                            data_split=DataSplit.TRAIN,
                            logger=main_logger,
                            is_n_models_without_tuning=False,
                            final_model_failure_behavior=self.final_model_failure_behavior,
                        )
                        self.log_final_metrics(
                            final_model_results=final_model_results,
                            metrics=metrics,
                            data_split=DataSplit.TEST,
                            logger=main_logger,
                            is_n_models_without_tuning=False,
                            final_model_failure_behavior=self.final_model_failure_behavior,
                        )
                    except Exception as e:
                        main_logger(f'Error while logging metrics:\n{format_exception_msg(e)}')
                        pass
                timer.stop()
                main_logger(self._train_end_msg(timer=timer, tracker=tracker))
        except Exception as e:
            error_msg: str = f'Ray tuning job failed with the following error:' \
                             f'\n{format_exception_msg(e, short=False)}'
            if final_model_results is not None:
                error_msg += f'\nWe were able to capture the {tune.ResultGrid} for the final model; ' \
                             f'it is the first returned result from {self.class_name}.train(...) function.'
            if tune_results is not None:
                error_msg += f'\nWe were able to capture the {tune.ResultGrid} for the Tuning job; ' \
                             f'it is the second returned result from {self.class_name}.train(...) function.'
            if isinstance(e, RayTuneTrainerFinalModelsError) and self.final_model_failure_behavior == 'error':
                run_training_error = RayTuneTrainerError(error_msg)
            else:
                Log.error(error_msg)
        finally:
            if 'tuner' in locals():
                del tuner
            # del datasets
            if run_training_error is not None:
                raise run_training_error
            return final_model_results, tune_results

    def _ray_tune_trainer_progress_bar(self, progress_bar: Optional[Dict], **kwargs) -> Optional[Dict]:
        # if progress_bar is not None:
        #     warnings.warn(
        #         f'We cannot show a progress bar for multiple jobs. '
        #         f'A dashboard with all the jobs will be shown instead.'
        #     )
        return None

    def tune_results_message(
            self,
            tune_results: tune.ResultGrid,
            *,
            save_model: Optional[FileMetadata] = None,
    ) -> str:
        best_hyperparams, best_objective_metric_value = self.get_best_hyperparams(tune_results)

        ## Log messages before retraining final model:
        if self.k_fold.num_folds > 1:
            msg: str = f'{self.k_fold.num_folds}-fold cross-validation was used to '
        else:
            msg: str = f'{self.objective_dataset.capitalize()} dataset was used to '
        msg: str = f'\n{msg}{self.objective_type}imize ' \
                   f'objective metric "{self.objective_metric.display_name}" ' \
                   f'to {best_objective_metric_value:.6e}' \
                   f'\nBest {best_hyperparams}'
        if tune_results.num_errors > 0:
            msg += f'\n{tune_results.num_errors} model failures were encountered during tuning:'
            msg += '\n'.join([f'\t{err.args}' for err in tune_results.errors]) + '\n'
        msg += f'\nRetraining {self.num_final_models} final models using above hyperparams.\n'
        if save_model is not None:
            msg += f'Final models will be saved to sub-directories within the directory: "{save_model.path}"\n'
        return msg

    def tune_retrain_message(
            self,
            final_model_results: tune.ResultGrid,
            *,
            retrain_timer: Timer,
            save_model: Optional[FileMetadata] = None,
    ) -> str:
        msg: str = f'Using above hyperparameters, {len(list(final_model_results))} final model(s) ' \
                   f'were retrained on complete {DataSplit.TRAIN.capitalize()} dataset ' \
                   f'in {retrain_timer.time_taken_str}.\n'
        if save_model is not None:
            msg += f'\nFinal model(s) were saved to sub-directories within the directory: "{save_model.path}"\n'
        return msg

    def _create_tuner_n_models_without_tuning(
            self,
            datasets: Datasets,
            metrics: Optional[Metrics],
            hyperparams: Algorithm.Hyperparameters,
            save_model: Optional[FileMetadata],
            num_samples: int,
            **kwargs,
    ) -> tune.Tuner:
        tuner: tune.Tuner = tune.Tuner(
            trainable=self._create_trainable(
                AlgorithmClass=self.AlgorithmClass,
                datasets=datasets,
                metrics=metrics,
                resources=self.resources_per_model,
                save_model=save_model,
                is_n_models_without_tuning=True,
                **kwargs,
            ),
            param_space=hyperparams.dict(),
            ## Don't tune, just train num_samples on the Ray cluster:
            tune_config=tune.TuneConfig(
                num_samples=num_samples,
                max_concurrent_trials=self.max_parallel_models,
                time_budget_s=self.timeout_seconds,
                ## Let's not reuse actors for now, since it seems complex.
                ## Refs:
                ## - https://github.com/ray-project/ray/issues/10101
                ## - https://docs.ray.io/en/latest/tune/api_docs/trainable.html#advanced-reusing-actors
                reuse_actors=False,
            ),
            run_config=self._create_run_config(
                epochs=hyperparams.epochs,
                steps=hyperparams.steps,
                save_model=save_model,
                is_n_models_without_tuning=True,
                metrics=metrics,
            ),
        )
        return tuner

    def _create_tuner_for_tuning(
            self,
            datasets: Datasets,
            metrics: Optional[Metrics],
            **kwargs,
    ) -> tune.Tuner:
        tuner: tune.Tuner = tune.Tuner(
            trainable=self._create_trainable(
                AlgorithmClass=self.AlgorithmClass,
                datasets=datasets,
                metrics=metrics,
                resources=self.resources_per_model,
                save_model=None,
                is_n_models_without_tuning=False,
                **kwargs,
            ),
            param_space=self._create_hyperparams_search_space_for_tuning(),
            tune_config=self._create_tune_config_for_tuning(datasets=datasets, metrics=metrics),
            run_config=self._create_run_config(
                epochs=self.epochs,
                steps=self.steps,
                save_model=None,
                is_n_models_without_tuning=False,
                metrics=metrics,
            ),
            ## Do not persist tuning models.
        )
        return tuner

    def _create_trainable(
            self,
            AlgorithmClass: Type[Algorithm],
            datasets: Datasets,
            metrics: Optional[Metrics],
            resources: Dict[Literal['cpu', 'gpu'], confloat(ge=0)],  ## E.g. {"cpu": 3, "gpu": 1}
            save_model: Optional[FileMetadata],
            is_n_models_without_tuning: bool,
            **kwargs
    ) -> Type[tune.Trainable]:
        trainable: Type[tune.Trainable] = AlgorithmTrainable
        if PyTorch is not None and issubclass(AlgorithmClass, PyTorch):
            if resources.get('gpu', 0.0) > 0:
                kwargs.setdefault('device', 'cuda')
        if is_n_models_without_tuning:
            k_fold: KFold = KFold.NO_TUNING_K_FOLD
        else:
            k_fold: KFold = self.k_fold
            if k_fold.num_folds > 1:
                if DataSplit.VALIDATION in datasets.datasets:
                    raise ValueError(
                        f'Cannot pass a {DataSplit.VALIDATION.capitalize()} dataset when using '
                        f'k_fold={self.k_fold.num_folds}; please only pass the {DataSplit.TRAIN.capitalize()} '
                        f'dataset, we will automatically split and perform K-Fold cross-validation.'
                    )
                if self.shuffle is True:
                    ## Priority: global K-Fold seed > random seed.
                    k_fold_seed: int = get_default(k_fold.seed, random.randint(0, int(1e9)))
                    if k_fold.seed is None:
                        warnings.warn(
                            f'When tuning K-Fold with shuffling, to get consistent results, please pass either a '
                            f'global `seed` to {self.class_name}, or a local `seed` to the `k_fold` dict. This seed is '
                            f'needed to create consistent dataset-splits across processes. As no seed is passed, we set '
                            f'the local K-Fold seed as {k_fold_seed}'
                        )
                    k_fold: KFold = k_fold.update_params(seed=k_fold_seed)

        trainable: Type[tune.Trainable] = tune.with_parameters(
            trainable,
            ray_tune_trainer=dict(
                **self.dict(exclude={'tracker'}),
                tracker=dict(tracker='noop'),  ## Don't track these runs.
            ),
            AlgorithmClass=AlgorithmClass.class_name,
            datasets=datasets.dict(),
            metrics=None if metrics is None else metrics.dict(),
            k_fold=k_fold.dict(),
            save_model=None if save_model is None else save_model.dict(),
            **kwargs
        )
        ## Ref: https://docs.ray.io/en/latest/tune/tutorials/tune-resources.html
        trainable: Type[tune.Trainable] = tune.with_resources(
            trainable,
            resources=resources,
        )
        return trainable

    def _create_hyperparams_search_space_for_tuning(self) -> Dict[str, Any]:
        if 'epochs' in self.ray_search_space:
            raise ValueError(
                f'Hyperparameter `epochs` does not need to be tuned; instead, we will train all models for `epochs`, '
                f'iterations over the {DataSplit.TRAIN} dataset, then select the best hyperparameter-set based on '
                f'the objective metric score across all epochs. Thus, `epochs` must be set to a fixed value and '
                f'not a range.'
            )
        grid_search_hp_names: Set[str] = set()
        for hp_name, hp_search_space in self.search_space.items():
            if hp_search_space.mapped_callable() == tune.grid_search:
                grid_search_hp_names.add(hp_name)
        if self.search_algorithm.name == 'grid' and len(grid_search_hp_names) == 0 and len(self.search_space) > 0:
            raise ValueError(
                f'When passing `search_algorithm="grid"` (to run a Grid Search), the search-space for all '
                f'hyperparams must be a "grid_search" search-space (such as a fixed list); '
                f'however, none of the hyperparams is a valid "grid_search" search-space.'
            )

        if len(grid_search_hp_names) > 0:
            if self.search_algorithm.name != 'grid':
                raise ValueError(
                    f'''When passing even one hyperparam's search-space as "grid_search", you must pass '''
                    f'''`search_algorithm="grid"` to run a Grid Search.'''
                )
            if len(grid_search_hp_names) != len(self.search_space):
                raise ValueError(
                    f'''When running a Grid Search, the search-space for all hyperparams should be "grid_search"; '''
                    f'''the search-space of the following hyperparams is not "grid_search": '''
                    f'''{set(self.search_space.keys()) - grid_search_hp_names}.'''
                )
            ## NOTE: `tune.grid_search` has the following weird behavior with num_samples:
            ##      `tune.grid_search`: Every value will be sampled ``num_samples`` times (``num_samples`` is the
            ##                          parameter you pass to ``tune.TuneConfig`` in ``Tuner``)
            ##      For this reason, if `tune.grid_search` is passed, we check that num_samples=1
            ##      and all other search spaces are also `tune.grid_search`.
            if self.num_models != 1 and self.allow_repeated_grid_search_sampling is False:
                raise ValueError(
                    f'When running a Grid Search, you must pass the number of models as 1. This will explore each '
                    f'combination in the grid exactly once. '
                    f'To explore each combination multiple times, pass allow_repeated_grid_search_sampling=True and '
                    f'set num_models > 1 in the {Trainer.class_name}.'
                )

        hyperparams_search_space: Dict[str, Any] = {
            **self._create_hyperparams().dict(),  ## Sets "default" values.
            **self.ray_search_space,  ## Override with values from ray search space.
        }
        if self.search_algorithm.mapped_callable() == tune.search.BasicVariantGenerator:
            ## Temporary fix until Repeater supports BasicVariantGenerator: github.com/ray-project/ray/issues/33677
            hyperparams_search_space[tune.search.repeater.TRIAL_INDEX] = tune.grid_search(range(self.k_fold.num_folds))
        return hyperparams_search_space

    def _create_tune_config_for_tuning(self, datasets: Datasets, metrics: Optional[Metrics]) -> tune.TuneConfig:
        assert self.is_n_models_without_tuning() is False
        num_models: int = self.num_models
        if self.search_algorithm.mapped_callable() == tune.search.BasicVariantGenerator:
            ## Temporary fix until Repeater supports BasicVariantGenerator: github.com/ray-project/ray/issues/33677
            search_alg: Searcher = self.search_algorithm.initialize(constant_grid_search=True)
        else:
            search_alg: Searcher = self.search_algorithm.initialize()
            ## Note: technically, using Repeater to create a new trial for every fold is not ideal. The "correct" way
            ## to use K-fold CrossValidation in search algorithms like BayesOptSearch is to maintain K parallel models,
            ## and at each epoch, use their mean validation metric to guide the Search procedure.
            ## However, this is not implemented by Ray's BayesOptSearch, so we can't do much and except to take the
            ## average validation metric at each epoch over multiple (independent) search procedures, for each fold.
            search_alg: Repeater = Repeater(search_alg, repeat=self.k_fold.num_folds)
            ## When using Repeater, `num_models` becomes a limiter unless we multiply it by the number of folds.
            num_models *= self.k_fold.num_folds
        if self.objective_dataset is not None:
            if self.k_fold.num_folds > 1:
                if self.objective_dataset is not DataSplit.VALIDATION:
                    raise ValueError(
                        f'When using K-Fold cross-validation, '
                        f'`objective_dataset` must be {DataSplit.VALIDATION} '
                    )
            elif self.objective_dataset not in datasets.datasets and self.objective_dataset:
                raise ValueError(
                    f'Attempting to tune on {self.objective_dataset.capitalize()} dataset, '
                    f'but no such dataset was found. Current datasets: {list(datasets.datasets.keys())}'
                )
            if metrics is None or self.objective_dataset not in metrics.metrics:
                error_msg: str = f'Attempting to tune on {self.objective_dataset.capitalize()} dataset, '
                f'but no metrics were configured for this dataset.'
                if metrics is not None:
                    error_msg += f' Current metrics: {metrics}'
                raise ValueError(error_msg)

        ## Ref: https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html
        return tune.TuneConfig(
            metric=self.objective_metric_display_name,
            mode=self.objective_type,
            search_alg=search_alg,
            num_samples=num_models,
            max_concurrent_trials=self.max_parallel_models,
            time_budget_s=self.timeout_seconds,
            ## Let's not reuse actors for now, since it seems complex.
            ## Refs:
            ## - https://github.com/ray-project/ray/issues/10101
            ## - https://docs.ray.io/en/latest/tune/api_docs/trainable.html#advanced-reusing-actors
            reuse_actors=False,
        )

    def _create_run_config(
            self,
            *,
            epochs: Optional[int],
            steps: Optional[int],
            save_model: Optional[FileMetadata],
            is_n_models_without_tuning: bool,
            metrics: Optional[Metrics],
    ) -> air.RunConfig:
        if epochs is not None:
            metric_columns: Dict[str, str] = {
                'time_total_s': 'total time (s)',
                _RAY_EPOCH_NUM: 'epoch',
                'time_this_iter_s': 'current epoch time (s)',
                _RAY_EST_TIME_REMAINING: 'est. time remaining',
            }
        elif steps is not None:
            metric_columns: Dict[str, str] = {
                'time_total_s': 'total time (s)',
                _RAY_STEPS_COMPLETED: 'steps',
                _RAY_EST_TIME_REMAINING: 'est. time remaining',
            }
        else:
            raise ValueError(f'Either `epochs` or `steps` must be non-None.')
        if metrics is not None:
            for data_split, dataset_metrics in metrics.metrics.items():
                for dataset_metric in dataset_metrics:
                    ray_metrics_name: str = _ray_metric_str(data_split=data_split, metric=dataset_metric)
                    metric_columns[ray_metrics_name] = ray_metrics_name

        ## Ref for all keys in air.RunConfig: https://docs.ray.io/en/latest/train/config_guide.html
        if epochs is not None:
            stopper = tune.stopper.CombinedStopper(
                tune.stopper.MaximumIterationStopper(max_iter=epochs),
                ## TODO: support additional stopping criterion
            )
        elif steps is not None:
            stopper = tune.stopper.CombinedStopper(
                MaximumStepsStopper(max_steps=steps, steps_increment=self.metric_eval_frequency),
                ## TODO: support additional stopping criterion
            )
        else:
            raise ValueError(f'Must pass either `epochs` or `steps` in hyperparams.')
        run_config: Dict = dict(
            ## Ref: https://docs.ray.io/en/latest/tune/api_docs/stoppers.html?highlight=ray.tune.stopper.Stopper
            stop=stopper,
            ## Do not checkpoint intermediate results when tuning:
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=1,  ## Minimum value is 1 because of a validation error in Ray.
                checkpoint_at_end=False,
                checkpoint_frequency=0,
                ## TODO: support more keys
            ),
            ## Ref: https://docs.ray.io/en/latest/ray-air/package-ref.html#ray.air.config.FailureConfig
            failure_config=air.FailureConfig(
                max_failures=self.model_failure_retries,  ## How many times to retry failed Trials
                fail_fast=False,  ## Does not stop the entire Tune run if a single trial fails.
            ),
            verbose=self.verbosity,
            ## Ref: https://docs.ray.io/en/latest/tune/tutorials/tune-output.html#how-to-redirect-stdout-and-stderr-to-files
            log_to_file=True,
            ## Refs for progress_reporter:
            ## - https://docs.ray.io/en/latest/tune/api/doc/ray.tune.JupyterNotebookReporter.html
            ## - https://docs.ray.io/en/latest/tune/api/reporters.html
            progress_reporter=tune.JupyterNotebookReporter(
                metric_columns=metric_columns,
                overwrite=False,  ## Do not overwrite the cell contents before initialization
                max_progress_rows=500,  ## For large tuning jobs.
                max_column_length=200,  ## For long hyperparams.
                max_report_frequency=self.progress_update_frequency,  ## Update every N seconds.
            )
        )
        if save_model is not None:
            if is_n_models_without_tuning:
                ## When we are training the final model(s), save checkpoint at end, but don't sync to S3.
                run_config: Dict = {
                    **run_config,
                    ## Ref: https://docs.ray.io/en/latest/ray-air/package-ref.html#ray.air.config.CheckpointConfig
                    **dict(
                        checkpoint_config=air.CheckpointConfig(
                            num_to_keep=1,
                            checkpoint_at_end=True,
                            checkpoint_frequency=self.checkpoint_frequency,
                            ## TODO: support more keys
                        ),
                    ),
                }
            else:
                ## When we are tuning, do not save model checkpoints.
                run_config: Dict = {
                    **run_config,
                    **dict(
                        checkpoint_config=air.CheckpointConfig(
                            num_to_keep=1,  ## This must be 1 because of a validation error in Ray.
                            checkpoint_at_end=False,
                            checkpoint_frequency=0,
                            ## TODO: support more keys
                        ),
                    ),
                }
                # if save_model.storage is Storage.S3:
                #     if not S3Util.is_path_valid_s3_dir(save_model.path):
                #         raise ValueError(f'Following path is not a valid S3 directory: "{save_model.path}"')
                #     run_config: Dict = {
                #         **run_config,
                #         **dict(
                #             sync_config=tune.SyncConfig(
                #                 upload_dir=save_model.path,
                #                 syncer="auto",
                #                 sync_period=15,
                #                 sync_timeout=180,
                #             ),
                #         ),
                #     }
        return air.RunConfig(**run_config)

    def get_detailed_metrics(self, results: tune.ResultGrid) -> pd.DataFrame:
        trial_hyperparams: Dict[str, Dict] = {
            result.metrics[_RAY_TRIAL_ID]: result.config
            for result in results
        }
        column_order: List = [
            _RAY_EXPERIMENT_ID, _RAY_TRIAL_ID, _RAY_HYPERPARAMS, _RAY_KFOLD_CURRENT_FOLD_NAME, _RAY_TRAINING_ITERATION,
        ]
        column_sort: List = [_RAY_HYPERPARAMS_STR, _RAY_TRIAL_ID, _RAY_KFOLD_CURRENT_FOLD_NAME, _RAY_TRAINING_ITERATION]

        detailed_metrics: pd.DataFrame = pd.concat([
            _ray_convert_metrics_dataframe(result.metrics_dataframe)
            for result in results
        ]).reset_index(drop=True)
        detailed_metrics[_RAY_HYPERPARAMS]: pd.Series = detailed_metrics[_RAY_TRIAL_ID].map(
            lambda trial_id: {
                hp_name: hp_val
                for hp_name, hp_val in trial_hyperparams[trial_id].items()
                if hp_name != tune.search.repeater.TRIAL_INDEX
            }
        )
        detailed_metrics[_RAY_HYPERPARAMS_STR] = detailed_metrics[_RAY_HYPERPARAMS].apply(StringUtil.stringify)
        current_fold_name_from_trial_id: Callable = lambda trial_id: \
            None if tune.search.repeater.TRIAL_INDEX not in trial_hyperparams[trial_id] \
                else self.k_fold.fold_names()[trial_hyperparams[trial_id][tune.search.repeater.TRIAL_INDEX]]
        detailed_metrics[_RAY_KFOLD_CURRENT_FOLD_NAME] = detailed_metrics[_RAY_TRIAL_ID].map(
            current_fold_name_from_trial_id)
        if detailed_metrics[_RAY_KFOLD_CURRENT_FOLD_NAME].isna().all() \
                or detailed_metrics[_RAY_KFOLD_CURRENT_FOLD_NAME].nunique() == 1:
            ## In this case we are either not tuning models, or not doing K-fold.
            detailed_metrics: pd.DataFrame = detailed_metrics.drop([_RAY_KFOLD_CURRENT_FOLD_NAME], axis=1)
            column_order.pop(column_order.index(_RAY_KFOLD_CURRENT_FOLD_NAME))
            column_sort.pop(column_sort.index(_RAY_KFOLD_CURRENT_FOLD_NAME))
        if _RAY_EXPERIMENT_ID not in detailed_metrics.columns:
            column_order.pop(column_order.index(_RAY_EXPERIMENT_ID))
        detailed_metrics: pd.DataFrame = pd_partial_column_order(
            detailed_metrics,
            columns=column_order
        ).sort_values(column_sort, ascending=True).reset_index(drop=True)
        return detailed_metrics

    def get_best_hyperparams(self, tune_results: tune.ResultGrid) -> Tuple[Algorithm.Hyperparameters, float]:
        tune_metrics: pd.DataFrame = self.get_detailed_metrics(tune_results)
        if _RAY_EPOCH_NUM in tune_metrics.columns \
                and tune_metrics[_RAY_EPOCH_NUM].isna().value_counts().get(False, 0) > 0:
            iter_col: str = _RAY_EPOCH_NUM
            iter_key: str = 'epochs'
        elif _RAY_STEPS_COMPLETED in tune_metrics.columns and \
                tune_metrics[_RAY_STEPS_COMPLETED].isna().value_counts().get(False, 0) > 0:
            iter_col: str = _RAY_STEPS_COMPLETED
            iter_key: str = 'steps'
        else:
            raise ValueError(f'Either `epochs` or `steps` must be present in the hyperparams dataframe.')
        best_hyperparams_metrics: List[Dict] = []
        for hyperparams_str, hp_df in tune_metrics.groupby(_RAY_HYPERPARAMS_STR):
            if _RAY_KFOLD_CURRENT_FOLD_NAME in hp_df.columns:
                detected_num_folds: int = hp_df[_RAY_KFOLD_CURRENT_FOLD_NAME].nunique()
                detected_epochs: int = hp_df[iter_col].nunique()
                ## Allows for the case where the same set of hyperparams might be repeated multiple times,
                ## so we get multiple folds:
                if len(hp_df) % (detected_num_folds * detected_epochs) != 0 or \
                        hp_df[_RAY_KFOLD_CURRENT_FOLD_NAME].nunique() != self.k_fold.num_folds:
                    ## Skip this set of hyperparams, it has a failed fold/epoch, thus we cannot measure for it.
                    continue
            hyperparams: Dict = hp_df[_RAY_HYPERPARAMS].iloc[0]
            iterwise_mean_objective_metrics: Dict[int, float] = {}
            for iter_num, hp_iter_num_df in hp_df.groupby(iter_col):
                iter_num: int = int(iter_num)
                if _RAY_KFOLD_CURRENT_FOLD_NAME in hp_iter_num_df.columns:
                    assert hp_df[_RAY_KFOLD_CURRENT_FOLD_NAME].nunique() == self.k_fold.num_folds
                    ## Allows for the case where the same set of hyperparams might be repeated multiple times,
                    ## so we get multiple folds for a training iteration:
                    hp_epoch_objective_metric_per_fold: List[float] = []
                    for current_fold_name, hp_epoch_fold_df in hp_iter_num_df.groupby(_RAY_KFOLD_CURRENT_FOLD_NAME):
                        ## If we don't have repeated hyperparams, this should have only 1 row:
                        hp_epoch_objective_metric_per_fold.append(
                            hp_epoch_fold_df[self.objective_metric_display_name].mean())
                    hp_epoch_objective_metric_per_fold: pd.Series = pd.Series(hp_epoch_objective_metric_per_fold)
                    iterwise_mean_objective_metrics[iter_num] = hp_epoch_objective_metric_per_fold.mean()
                else:
                    iterwise_mean_objective_metrics[iter_num] = hp_iter_num_df[
                        self.objective_metric_display_name].mean()
            iterwise_mean_objective_metrics: pd.Series = pd.Series(
                iterwise_mean_objective_metrics,
                name=self.objective_metric_display_name
            )
            iterwise_mean_objective_metrics.index.name = iter_key
            iterwise_mean_objective_metrics: pd.DataFrame = iterwise_mean_objective_metrics.reset_index()
            best_epoch_metric: Dict[str, Union[float, int]] = iterwise_mean_objective_metrics.sort_values(
                self.objective_metric_display_name,
                ascending=True if self.objective_type == 'min' else False
            ).iloc[0].to_dict()
            best_epoch: int = int(best_epoch_metric[iter_key])
            best_metric_val: float = best_epoch_metric[self.objective_metric_display_name]
            best_hyperparams_metrics.append({
                _RAY_HYPERPARAMS: {**hyperparams, iter_key: best_epoch},
                self.objective_metric_display_name: best_metric_val,
            })
        best_hyperparams_metrics: pd.DataFrame = pd.DataFrame(best_hyperparams_metrics)
        best_hyperparams_metrics: Dict = best_hyperparams_metrics.sort_values(
            self.objective_metric_display_name,
            ascending=True if self.objective_type == 'min' else False
        ).iloc[0].to_dict()
        best_objective_metric_value: float = best_hyperparams_metrics[self.objective_metric_display_name]
        best_hyperparams: Algorithm.Hyperparameters = self._create_hyperparams(
            best_hyperparams_metrics[_RAY_HYPERPARAMS]
        )
        ## Dict like {'hyperparams': {...}, self.objective_metric_display_name: ...}
        return best_hyperparams, best_objective_metric_value

    @classmethod
    def get_trialwise_final_model_metrics(
            cls,
            final_model_results: tune.ResultGrid,
            *,
            metrics: Metrics,
    ) -> Dict[str, Metrics]:
        ## Gets trial-wise metrics.
        final_model_metrics: Dict[str, Union[Metrics, Dict[DataSplit, List[Metric]]]] = dict()
        for final_model_result in final_model_results:
            trial_id: str = final_model_result.metrics[_RAY_TRIAL_ID]
            final_model_metrics[trial_id] = dict()
            for data_split, dataset_metrics in metrics.metrics.items():
                if dataset_metrics is None or len(dataset_metrics) == 0:
                    continue
                final_model_metrics[trial_id][data_split] = []
                ## Each trial should have all the metrics, but not guaranteed:
                for dataset_trial_metric in dataset_metrics:
                    dataset_trial_metric: Metric = dataset_trial_metric.clear()
                    ## "result.metrics" gets the metric value at the final epoch/step:
                    dataset_trial_metric.value = _ray_get_metric_value(
                        final_model_result.metrics.get(
                            _ray_metric_str(data_split=data_split, metric=dataset_trial_metric)
                        )
                    )
                    final_model_metrics[trial_id][data_split].append(dataset_trial_metric)
            final_model_metrics[trial_id] = Metrics.of(**final_model_metrics[trial_id])
        return final_model_metrics

    @classmethod
    def get_final_metrics_stats(
            cls,
            final_model_results: tune.ResultGrid,
            *,
            metrics: Metrics,
            data_split: DataSplit,
    ) -> Optional[Dict[str, Dict[str, Union[int, float]]]]:
        dataset_metrics: Optional[List[Metric]] = metrics[data_split]
        metric_vals_are_numeric = lambda metric_vals: np.all([
            isinstance(v, (int, float)) and not is_null(v)
            for v in metric_vals
        ])
        if dataset_metrics is None or len(dataset_metrics) == 0:
            return None
        final_dataset_metrics: Dict[str, Union[List, Dict]] = {}
        for dataset_metric in dataset_metrics:
            final_dataset_metrics[dataset_metric.display_name] = []
            for final_model_result in final_model_results:
                ## "result.metrics" gets the metric value at the final epoch/step:
                final_dataset_metrics[dataset_metric.display_name].append(final_model_result.metrics.get(
                    _ray_metric_str(data_split=data_split, metric=dataset_metric)
                ))
            if not metric_vals_are_numeric(final_dataset_metrics[dataset_metric.display_name]):
                final_dataset_metrics.pop(dataset_metric.display_name)
                continue
            final_dataset_metrics[dataset_metric.display_name]: Dict[str, Union[int, float, Dict]] = {
                'mean': np.mean(final_dataset_metrics[dataset_metric.display_name]),
                'median': np.median(final_dataset_metrics[dataset_metric.display_name]),
                'std': np.std(final_dataset_metrics[dataset_metric.display_name], ddof=1),  ## Unbiased
                'min': np.min(final_dataset_metrics[dataset_metric.display_name]),
                'max': np.max(final_dataset_metrics[dataset_metric.display_name]),
                'num_samples': len(final_dataset_metrics[dataset_metric.display_name]),
                'metric_dict': dataset_metric.dict(),
            }
            if final_dataset_metrics[dataset_metric.display_name]['num_samples'] == 0:
                final_dataset_metrics.pop(dataset_metric.display_name)  ## Remove it from the set of metrics
        final_dataset_metrics: Dict[str, Dict[str, Union[int, float]]] = {
            metric_display_name: metric_stats
            for metric_display_name, metric_stats in sorted(
                list(final_dataset_metrics.items()),
                key=lambda x: x[0],
            )
        }
        if len(final_dataset_metrics) == 0:
            return None
        return final_dataset_metrics

    @classmethod
    def log_final_metrics(
            cls,
            final_model_results: tune.ResultGrid,
            *,
            metrics: Metrics,
            data_split: DataSplit,
            logger: Callable,
            is_n_models_without_tuning: bool,
            final_model_failure_behavior: Literal['warn', 'error'],
    ):
        if final_model_results.num_errors > 0:
            msg = f'\n{final_model_results.num_errors} model failures were encountered during training final models:'
            msg += '\n'.join([f'\t{err.args}' for err in final_model_results.errors]) + '\n'
            if final_model_failure_behavior == 'error':
                raise RayTuneTrainerFinalModelsError(msg)
            elif final_model_failure_behavior == 'warn':
                Log.error(msg)
        final_dataset_metrics: Optional[Dict[str, Dict[str, Union[int, float, Dict]]]] = \
            cls.get_final_metrics_stats(
                final_model_results=final_model_results,
                metrics=metrics,
                data_split=data_split,
            )
        if final_dataset_metrics is not None:
            logger(f'▁' * 80)
            logger(f'Aggregated {data_split.capitalize()} metrics for final model(s):')
            for metric_display_name, metric_stats in final_dataset_metrics.items():
                logger(f'    {metric_stats_str(metric_display_name, metric_stats)}')
        logger(f'\nTrial-wise {data_split.capitalize()} metrics for final model(s):')
        trialwise_final_model_metrics: Dict[str, Metrics] = cls.get_trialwise_final_model_metrics(
            final_model_results,
            metrics=metrics,
        )
        for trial_i, (trial_id, trial_metrics) in enumerate(trialwise_final_model_metrics.items()):
            trial_dataset_metrics: Optional[List[Metric]] = trial_metrics[data_split]
            if trial_dataset_metrics is not None and len(trial_dataset_metrics) > 0:
                logger(f'\n(trial_id={trial_id}) {data_split.capitalize()} metrics:')
                for trial_dataset_metric in trial_dataset_metrics:
                    logger(trial_dataset_metric)
        logger('')

    def _train_start_msg(self, completed: bool, tracker: Tracker, save_model: Optional[FileMetadata], **kwargs) -> str:
        if self.is_n_models_without_tuning():
            if not completed:
                out: str = f'\nTraining {"single" if self.num_models == 1 else self.num_models} ' \
                           f'{self.task_display_name} model{"" if self.num_models == 1 else "(s)"} using Ray.' \
                           f'\nAlgorithm: {self.algorithm_display_name}'
            else:
                out: str = f'\nDone training {"single" if self.num_models == 1 else self.num_models} ' \
                           f'{self.task_display_name} model{"" if self.num_models == 1 else "(s)"} using Ray.'
        else:
            if not completed:
                start_str: str = 'Tuning'
                out: str = f'\n{start_str} {self.task_display_name} model using Ray Tune.' \
                           f'\nAlgorithm: {self.algorithm_display_name}' \
                           f'\nSearch algorithm: {self.search_algorithm.to_call_str()}' \
                           f'\nSearch space: {StringUtil.pretty(self._create_hyperparams_search_space_for_tuning())}'
            else:
                start_str: str = 'Done tuning'
                out: str = f'\n{start_str} {self.task_display_name} model using Ray Tune.'
        save_model_msg: str = ''
        if save_model is not None:
            if not completed:
                save_model_msg: str = f'\nFinal models will be saved to: "{save_model.path}"'
            else:
                save_model_msg: str = f'\nFinal models were saved to: "{save_model.path}"'
        if tracker.tracker_name == 'noop':
            if not completed:
                tracker_msg: str = '\nLogs will not be tracked.'
            else:
                tracker_msg: str = '\nLogs were not tracked.'
        else:
            if not completed:
                tracker_msg: str = f'\n{tracker.class_name}@{tracker.id} will save logs to: "{tracker.log_dir}"'
            else:
                tracker_msg: str = f'\n{tracker.class_name}@{tracker.id} saved logs to: "{tracker.log_dir}"'

        return f'{out}{save_model_msg}{tracker_msg}'

    def _train_end_msg(self, *, timer: Timer, tracker: Tracker, **kwargs) -> str:
        if tracker.tracker_name == 'noop':
            tracker_msg: str = 'Logs have not been tracked.'
        else:
            tracker_msg: str = f'{tracker.class_name}@{tracker.id} has saved logs to "{tracker.log_dir}"'
        if self.is_n_models_without_tuning():
            return f'...training completed in {timer.time_taken_str}.\n' \
                   f'{tracker_msg}'
        else:
            return f'...tuning job completed in {timer.time_taken_str}.\n' \
                   f'{tracker_msg}'
