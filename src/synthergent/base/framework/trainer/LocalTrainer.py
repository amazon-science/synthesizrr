from typing import *
import time, traceback, gc, os, warnings
from functools import partial
from synthergent.base.util import Log, FileSystemUtil, StringUtil, get_default, all_are_not_none, safe_validate_arguments, Timer
from synthergent.base.data import FileMetadata
from synthergent.base.framework.trainer.Trainer import Trainer
from synthergent.base.framework.tracker.Tracker import Tracker
from synthergent.base.framework.task_data import DataSplit, Datasets, Dataset
from synthergent.base.framework.predictions import Predictions
from synthergent.base.framework.algorithm import Algorithm, TaskOrStr
from synthergent.base.framework.metric import Metric, Metrics


class LocalTrainer(Trainer):
    aliases = ['local', 'SimpleTrainer', 'simple']

    def initialize(self, **kwargs):
        pass

    @staticmethod
    def local_logger(text: str, verbosity: int, tracker: Tracker):
        pid: int = os.getpid()
        text: str = f'(pid={pid}): {text}'
        if verbosity == 0:  ## Don't log anything.
            return
        else:
            tracker.info(text)

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
            model: Optional[Algorithm] = None,
            return_model: bool = False,
            **kwargs,
    ) -> Optional[Algorithm]:
        timer: Timer = Timer(silent=True)
        timer.start()
        if self.k_fold.num_folds != 1:
            warnings.warn(f'{self.class_name} does not support the `k_folds` parameter.')

        train_dataset, train_metrics, \
        validation_dataset, validation_metrics, \
        test_dataset, test_metrics = self._extract_datasets_and_metrics(
            datasets=datasets,
            metrics=metrics,
        )
        train_stats: Optional[Metrics] = self.AlgorithmClass.calculate_dataset_stats(
            dataset=train_dataset,
            batch_size=self.stats_batch_size,
            data_split=DataSplit.TRAIN,
        )
        train_dataset_length: Metric = train_stats.find(data_split=DataSplit.TRAIN, select='row_count')
        train_dataset_length: int = train_dataset_length.value
        trainer_created_model: bool = False
        if model is None:
            ## Create a new model. Otherwise, continue training.
            model: Algorithm = self.AlgorithmClass.of(
                hyperparams=self._create_hyperparams(),
                model_dir=load_model,
                stats=train_stats,
                **kwargs
            )
            trainer_created_model: bool = True
        logger: Callable = partial(
            self.local_logger,
            ## Unless we request silence (verbosity=0), print important information.
            verbosity=0 if self.verbosity == 0 else 1,
            tracker=tracker,
        )
        batch_logger: Callable = partial(
            self.local_logger,
            ## Skip batch-level logging unless we explicitly ask for output from all batches (verbosity=2).
            verbosity=2 if self.verbosity == 2 else 0,
            tracker=tracker,
        )
        progress_bar: Optional[Dict] = self._local_trainer_progress_bar(
            progress_bar,
            train_dataset_length=train_dataset_length,
        )
        try:
            logger(self._train_start_msg(model=model, train_dataset=train_dataset, tracker=tracker))
            ## Call chain: ._train_single_model() -> ._train_epoch() -> ._train_loop() -> Algorithm.train_iter()
            self._train_single_model(
                model=model,
                epochs=model.hyperparams.epochs,
                steps=model.hyperparams.steps,
                train_dataset=train_dataset,
                train_dataset_length=train_dataset_length,
                train_metrics=train_metrics,
                validation_dataset=validation_dataset,
                validation_metrics=validation_metrics,
                test_dataset=test_dataset,
                test_metrics=test_metrics,
                logger=logger,
                batch_logger=batch_logger,
                progress_bar=progress_bar,
            )
            if save_model is not None:
                save_model.mkdir(raise_error=True)
                model.save_params(model_dir=save_model)
                model.save(model_dir=save_model)

            timer.stop()
            logger(self._train_end_msg(model=model, timer=timer, tracker=tracker))
            if return_model:
                return model
            elif trainer_created_model:
                model.cleanup()
        except KeyboardInterrupt as e:
            if model is not None and trainer_created_model:
                model.cleanup()
                del model
                gc.collect()
            raise e

    def _local_trainer_progress_bar(
            self,
            progress_bar: Optional[Dict],
            *,
            train_dataset_length: int,
            **kwargs,
    ) -> Optional[Dict]:
        ## Show epoch's progress bar only when printing important information.
        if self.verbosity == 1 and progress_bar is not False:
            if not isinstance(progress_bar, dict):
                progress_bar: Dict = dict()
            progress_bar.setdefault('total', train_dataset_length)
        else:
            progress_bar: Optional[Dict] = None
        return progress_bar

    def _train_start_msg(self, *, model: Algorithm, train_dataset: Dataset, tracker: Tracker, **kwargs) -> str:
        if tracker.tracker_name == 'noop':
            tracker_msg: str = 'Logs will not be tracked.'
        else:
            tracker_msg: str = f'{tracker.class_name}@{tracker.id} will save logs to: "{tracker.log_dir}"'
        return StringUtil.dedupe(
            f'\nTraining following {model.task} model '
            f'for {model.hyperparams.epochs} epochs '
            f'on {train_dataset.display_name} dataset:',
            dedupe=StringUtil.SPACE,
        ) + f'\n{str(model)}' \
            f'\n{tracker_msg}'

    def _train_end_msg(self, *, model: Algorithm, timer: Timer, tracker: Tracker, **kwargs) -> str:
        if tracker.tracker_name == 'noop':
            tracker_msg: str = 'Logs have not been tracked.'
        else:
            tracker_msg: str = f'{tracker.class_name}@{tracker.id} has saved logs to "{tracker.log_dir}"'
        return f'...training completed in {timer.time_taken_str}.\n' \
               f'{tracker_msg}'
