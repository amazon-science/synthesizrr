from typing import *
import time, warnings, traceback, os, math, numpy as np
from functools import partial
from synthesizrr.base.util import Log, optional_dependency, safe_validate_arguments, any_are_none, get_default, \
    StringUtil, FileSystemUtil, Timer
from synthesizrr.base.data import FileMetadata
from synthesizrr.base.framework.trainer.Trainer import Trainer
from synthesizrr.base.framework.task_data import DataSplit, Datasets, Dataset
from synthesizrr.base.framework.algorithm import Algorithm, TaskOrStr
from synthesizrr.base.framework.metric import Metric, Metrics
from synthesizrr.base.framework.tracker import Tracker
from pydantic import root_validator

with optional_dependency('accelerate', 'torch'):
    from synthesizrr.base.framework.dl.torch import PyTorch, is_accelerator
    from torch.utils.data import DataLoader as TorchDataLoader
    from accelerate import Accelerator, notebook_launcher
    from accelerate.utils import DistributedDataParallelKwargs
    from accelerate.utils import broadcast_object_list


    class AccelerateTrainer(Trainer):
        aliases = ['accelerate']

        def initialize(self, **kwargs):
            pass

        accelerator: Dict = {  ## Params for accelerate.Accelerator
            'split_batches': True,
            'kwargs_handlers': [
                ## Have to pass this to avoid https://github.com/pytorch/pytorch/issues/43259
                ## Ref: huggingface.co/docs/accelerate/v0.16.0/en/package_reference/kwargs#accelerate.DistributedDataParallelKwargs
                DistributedDataParallelKwargs(find_unused_parameters=True)
            ]
        }
        worker_rank: int = 0
        num_workers: int = 1
        distributed_evaluation: bool = False

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
                **kwargs,
        ):
            if self.k_fold.num_folds != 1:
                warnings.warn(f'{self.class_name} does not support the `k_folds` parameter.')
            if model is not None:
                raise ValueError(
                    f'Cannot pass `model` to {self.class_name}; instead, pass `load_model` to continue training from '
                    f'an existing model checkpoint.'
                )
            if not issubclass(self.AlgorithmClass, PyTorch):
                raise ValueError(
                    f'Can only use {self.class_name} with subclasses of {PyTorch}; '
                    f'passed `AlgorithmClass` has type: {type(self.AlgorithmClass)}'
                )

            kwargs.pop('name', None)  ## We pass algorithm as that passed in trainer
            kwargs.pop('algorithm', None)  ## We pass algorithm as that passed in trainer
            kwargs.pop('task', None)  ## We pass task as that passed in trainer
            device: Optional = kwargs.pop('device', None)  ## We pass device as accelerator
            if device is not None and self.verbosity > 0:
                Log.warning(
                    f'Device "{device}" will be ignored as we use accelerate.Accelerator '
                    f'to automatically place data and models.'
                )
            del device

            ## Set a random master port to allow running multiple accelerate jobs in parallel.
            ## Ref: https://discuss.pytorch.org/t/run-multiple-distributed-training-jobs-on-one-machine/123383/3
            master_port: str = f'{np.random.randint(20_000, 30_000)}'
            os.environ['MASTER_PORT']: str = master_port
            Log.warning(f'Running accelerate using environment variable MASTER_PORT={master_port}')

            notebook_launcher(
                self._accelerate_training_loop,
                args=(
                    self.dict(),
                    self.AlgorithmClass,
                    datasets.dict(),
                    None if metrics is None else metrics.dict(),
                    None if load_model is None else load_model.dict(),
                    None if save_model is None else save_model.dict(),
                    master_port,
                    dict(
                        experiment=f'{tracker.experiment}-accelerate',
                        **tracker.dict(exclude={'experiment'}),
                    ),
                    progress_bar,
                    kwargs,
                ),
                num_processes=self.num_workers,
                ## Set a random master port to allow running multiple accelerate jobs in parallel on the same machine:
                use_port=str(master_port),
            )

        @staticmethod
        def _accelerate_training_loop(
                accelerate_trainer: Dict,
                AlgorithmClass: Type[Algorithm],
                datasets: Dict,
                metrics: Optional[Dict],
                load_model: Optional[Dict],
                save_model: Optional[Dict],
                master_port: str,
                tracker: Dict,
                progress_bar: Optional[Dict],
                kwargs: Dict,
        ):
            ## In accelerate, anything run/called from inside this function is by default executed on all worker
            ## processes.
            ## To execute code on specific machines/processes:
            ## 1) accelerator.is_main_process -> true for only a single process across all machines.
            ## 2) accelerator.is_local_main_process -> true for the "main" process on each machine.
            ## 3) @accelerator.on_process(process_index=i) -> decorator wrapping a function. The function takes no
            ##  args and will run only on a single process with "global" process-index "i" i.e. 0<=i<num_workers
            ## 4) @accelerator.on_local_process(local_process_idx=i) -> decorator wrapping a function. The function
            ##  takes no args, and will run only on the process with process-index "i" on each machine,
            ##  i.e. 0<=i<num_workers_per_machine
            import pandas as pd
            ## Stops Pandas SettingWithCopyWarning in output. Ref: https://stackoverflow.com/a/20627316
            pd.options.mode.chained_assignment = None  # default='warn'
            timer: Timer = Timer(silent=True)
            timer.start()

            accelerator: Accelerator = Accelerator(**accelerate_trainer['accelerator'])
            accelerate_trainer['worker_rank']: int = accelerator.process_index
            accelerate_trainer['num_workers']: int = accelerator.num_processes
            accelerate_trainer: AccelerateTrainer = AccelerateTrainer(**accelerate_trainer)

            ## Set a random master port to allow running multiple accelerate jobs in parallel.
            ## Ref: https://discuss.pytorch.org/t/run-multiple-distributed-training-jobs-on-one-machine/123383/3
            os.environ['MASTER_PORT']: str = master_port

            ## Verbosity settings for AccelerateTrainer:
            ## verbosity=0: Don't log for each batch
            ## verbosity=1: Don't log for each batch
            ## verbosity=2: Log for each batch only from main process
            ## verbosity=3: Log for each batch from all processes
            tracker['capture_terminal_logs'] = True
            if accelerate_trainer.verbosity == 0:
                tracker: Tracker = Tracker.noop_tracker()  ## Don't track.
            else:  ## verbosity >= 1
                if accelerator.is_main_process:
                    tracker: Tracker = Tracker.of(tracker)
                else:
                    if accelerate_trainer.verbosity >= 3:  ## Log for each batch from all processes.
                        tracker['experiment']: str = \
                            f"{tracker['experiment']}" \
                            f"-worker={accelerate_trainer.worker_rank + 1}/{accelerate_trainer.num_workers}"
                    else:
                        tracker: Tracker = Tracker.noop_tracker()  ## Don't track.
                    tracker: Tracker = Tracker.of(tracker)
            assert isinstance(tracker, Tracker)

            logger: Callable = partial(
                AccelerateTrainer._accelerate_logger,
                worker_rank=accelerate_trainer.worker_rank,
                num_workers=accelerate_trainer.num_workers,
                ## Unless we request silence (verbosity=0), print important information from the main process.
                verbosity=0 if accelerate_trainer.verbosity == 0 else 1,
                tracker=tracker,
                accelerator=accelerator,
            )
            batch_logger: Callable = partial(
                AccelerateTrainer._accelerate_logger,
                worker_rank=accelerate_trainer.worker_rank,
                num_workers=accelerate_trainer.num_workers,
                ## Skip batch-level logging unless we explicitly ask for output from all batches (verbosity=2 and 3).
                verbosity={
                    0: 0,  ## verbosity=0: Don't log for each batch
                    1: 0,  ## verbosity=1: Don't log for each batch
                    2: 1,  ## verbosity=2: Log for each batch only from main process
                    3: 2,  ## verbosity=3: Log for each batch from all processes
                }[accelerate_trainer.verbosity],
                tracker=tracker,
                accelerator=accelerator,
            )
            # datasets['datasets']: Dict[DataSplit, Dataset] = {
            #     data_split: Dataset.of(**{
            #         **remove_keys(dataset, ['data']),
            #         'data': FileMetadata(**dataset['data']),
            #     })
            #     for data_split, dataset in datasets['datasets'].items()
            # }
            datasets: Datasets = Datasets(**datasets)
            metrics: Optional[Metrics] = None if metrics is None else Metrics(**metrics)
            load_model: Optional[FileMetadata] = None if load_model is None else FileMetadata(**load_model)
            save_model: Optional[FileMetadata] = None if save_model is None else FileMetadata(**save_model)

            train_stats: Optional[Metrics] = AlgorithmClass.calculate_dataset_stats(
                dataset=datasets[DataSplit.TRAIN],
                batch_size=accelerate_trainer.stats_batch_size,
                data_split=DataSplit.TRAIN,
            )
            train_dataset_length: Metric = train_stats.find(data_split=DataSplit.TRAIN, select='row_count')
            train_dataset_length: int = train_dataset_length.value
            pt_model: Algorithm = Algorithm.of(
                name=AlgorithmClass,
                task=accelerate_trainer.task,
                hyperparams=accelerate_trainer._create_hyperparams(),
                stats=train_stats,
                model_dir=load_model,
                init=True,  ## Ensure we call initialize(), which load the torch.nn.Module into `pt_model.model`
                post_init=False,  ## When using accelerate, first init training components then manually transfer model.
                device=accelerator,
                **kwargs
            )
            if not isinstance(pt_model, PyTorch):
                raise ValueError(
                    f'Can only use {accelerate_trainer.class_name} with subclasses of {PyTorch}; '
                    f'found model with type: {type(pt_model)}'
                )
            ## With accelerate, first init training components then manually transfer model, optimizer & scheduler:
            pt_model.init_training_components()
            pt_model.model, pt_model.optimizer, pt_model.lr_scheduler = accelerator.prepare(
                pt_model.model, pt_model.optimizer, pt_model.lr_scheduler
            )
            # logger(f'Prepared model, optimizer, lr_scheduler using accelerator: {accelerator}')

            train_dataset, train_metrics, \
            validation_dataset, validation_metrics, \
            test_dataset, test_metrics = accelerate_trainer._extract_datasets_and_metrics(
                datasets=datasets,
                metrics=metrics,
            )

            ## Call chain: ._train_single_model() -> ._train_epoch() -> ._train_loop() -> PyTorch.train_iter()
            logger(accelerate_trainer._train_start_msg(model=pt_model, tracker=tracker))
            progress_bar: Optional[Dict] = AccelerateTrainer._accelerate_trainer_progress_bar(
                progress_bar,
                accelerate_trainer=accelerate_trainer,
                accelerator=accelerator,
                train_dataset_length=train_dataset_length,
            )
            accelerate_trainer._train_single_model(
                model=pt_model,
                epochs=pt_model.hyperparams.epochs,
                steps=pt_model.hyperparams.steps,
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
                ## Extra batching params used by DataLoader:
            )
            pt_model.model = accelerator.unwrap_model(pt_model.model)
            if save_model is not None:
                FileSystemUtil.mkdir_if_does_not_exist(path=save_model.path)
                pt_model.save_params(model_dir=save_model)
                pt_model.save(model_dir=save_model)

            timer.stop()
            logger(accelerate_trainer._train_end_msg(model=pt_model, timer=timer, tracker=tracker))
            accelerator.wait_for_everyone()

        @staticmethod
        def _accelerate_trainer_progress_bar(
                progress_bar: Optional[Dict],
                *,
                accelerate_trainer: Any,
                accelerator: Accelerator,
                train_dataset_length: int,
                **kwargs
        ) -> Optional[Dict]:
            ## Show epoch's progress bar only when printing important information from the main process:
            if accelerate_trainer.verbosity == 1 and accelerator.is_main_process and progress_bar is not None:
                if not isinstance(progress_bar, dict):
                    progress_bar: Dict = dict()
                progress_bar.setdefault('total', train_dataset_length)
            else:
                progress_bar: Optional[Dict] = None
            return progress_bar

        @staticmethod
        def _accelerate_logger(
                text: str,
                worker_rank: int,
                num_workers: int,
                verbosity: int,
                tracker: Tracker,
                accelerator: Accelerator,
        ):
            pid: int = os.getpid()
            now: str = StringUtil.now()
            text: str = f'(pid={pid}, worker={worker_rank + 1}/{num_workers}, {now}): {text}'
            if verbosity == 0:  ## Don't log anything.
                return
            elif verbosity == 1:  ## Only log once globally, log using debug mode locally.
                if accelerator.is_main_process:
                    tracker.info(text)
            else:
                ## Print on all worker processes
                tracker.info(text)

        def _train_loop(
                self,
                model: PyTorch,
                train_dataset: Dataset,
                train_dataset_length: int,  ## Number of rows in training dataset.
                batch_size: int,
                **kwargs
        ):
            accelerator: Accelerator = model.device
            if not is_accelerator(accelerator):
                raise ValueError(
                    f'Expected model.device to be accelerator for model of type {type(model)}; '
                    f'found: {type(accelerator)}'
                )
            ## When we have "S" workers and batch-size "B", a dataset with "N" rows will only perfectly assign a
            ## batch of "B" to each worker if N%(S*B) == 0 (which is highly unlikely). When N%(S*B) != 0, we end up
            ## in the situation where for the final step, some workers get "B" rows, some workers get <= B rows, and
            ## some workers get no rows at all. In such a case, it makes sense to just skip the final step.
            ## This does not lose us too much training data: if N=1MM, B=16 and S=24, we will complete 2604 gradient
            ## update steps before reaching the "final" step where we encounter this issue. Assuming N >> S*B and
            ## we shuffle the dataset before each epoch, skipping the last step should have negligible impact on the
            ## training procedure.
            ## NOTE: we should set drop_last=False during prediction since we don't want to lose any data there.
            kwargs['drop_last'] = True
            train_dataset: TorchDataLoader = train_dataset.torch_dataloader(
                batch_size=batch_size,
                shard=(self.worker_rank, self.num_workers),
                **kwargs
            )
            train_dataset: TorchDataLoader = accelerator.prepare([train_dataset])[0]
            super(AccelerateTrainer, self)._train_loop(
                model=model,
                train_dataset=train_dataset,
                batch_size=batch_size,
                train_dataset_length=train_dataset_length,
                **kwargs,
            )
            # logger(f'Waiting for everyone using accelerator: {accelerator}')
            accelerator.wait_for_everyone()

        def _train_step_message(
                self,
                batches: int,
                batch_size: int,
                train_dataset_length: int,
                **kwargs,
        ) -> str:
            ## As we set drop_last=True during training, we will only do upto the last fully-completed set of W*B
            batches: int = math.floor(train_dataset_length / (self.num_workers * batch_size))
            out: str = super(AccelerateTrainer, self)._train_step_message(
                batches=batches,
                batch_size=batch_size,
                train_dataset_length=train_dataset_length,
                **kwargs,
            )
            return f'\n{out}'

        def _evaluate_metrics(
                self,
                model: PyTorch,
                dataset: Optional[Dataset],
                metrics: Optional[List[Metric]],
                eval_batch_size: Optional[int] = None,
                **kwargs
        ) -> Optional[List[Metric]]:
            accelerator: Accelerator = model.device
            if not is_accelerator(accelerator):
                raise ValueError(
                    f'Expected model.device to be accelerator for model of type {type(model)}; '
                    f'found: {type(accelerator)}'
                )
            if any_are_none(dataset, metrics):
                return None
            kwargs.pop('batch_size', None)  ## Remove the training batch size.
            eval_batch_size: Optional[int] = get_default(
                eval_batch_size,
                self.eval_batch_size,
                model.hyperparams.batch_size,
            )
            if self.distributed_evaluation is True:
                kwargs['drop_last'] = False
                dataset: TorchDataLoader = dataset.torch_dataloader(
                    batch_size=eval_batch_size,
                    shard=(self.worker_rank, self.num_workers),
                    **kwargs
                )
                dataset: TorchDataLoader = accelerator.prepare([dataset])[0]
                raise NotImplementedError(f'We have not implemented distributed evaluation using sharded dataloaders')
            else:
                ## Calculate all metrics on all workers. Note: it is required to predict on ALL workers, because
                ## torch distributed has a bug where evaluating on only one worker causes other workers to hang
                ## when calling accelerator.wait_for_everyone() (which internally calls torch.distributed.barrier()).
                ## Link to this bug: https://github.com/pytorch/pytorch/issues/54059
                ## In either case, doing prediction on only the main worker and making the other workers wait, will not
                ## lead to a speedup in wall-clock time, as compared to doing prediction on all workers.
                metrics: List[Metric] = super(AccelerateTrainer, self)._evaluate_metrics(
                    model=model,
                    dataset=dataset,
                    metrics=metrics,
                    eval_batch_size=eval_batch_size,
                    **kwargs,
                )
                accelerator.wait_for_everyone()
                return metrics

        def _train_start_msg(self, *, model: Algorithm, tracker: Tracker, **kwargs) -> str:
            if tracker.tracker_name == 'noop':
                tracker_msg: str = 'Logs will not be tracked.'
            else:
                tracker_msg: str = f'{tracker.class_name}@{tracker.id} will save logs to: "{tracker.log_dir}"'
            if model.hyperparams.epochs is not None:
                return f'\nTraining following {model.task} model ' \
                       f'for {model.hyperparams.epochs} epochs ' \
                       f'using {self.num_workers} workers...\n' \
                       f'{str(model)}\n{tracker_msg}'
            elif model.hyperparams.steps is not None:
                return f'\nTraining following {model.task} model ' \
                       f'for {model.hyperparams.steps} steps ' \
                       f'using {self.num_workers} workers...\n' \
                       f'{str(model)}\n{tracker_msg}'
            raise ValueError(f'Either `epochs` or `steps` in hyperparams must be not-None.')

        def _train_end_msg(self, *, model: Algorithm, timer: Timer, tracker: Tracker, **kwargs) -> str:
            if tracker.tracker_name == 'noop':
                tracker_msg: str = 'Logs have not been tracked.'
            else:
                tracker_msg: str = f'{tracker.class_name}@{tracker.id} has saved logs to "{tracker.log_dir}"'
            return f'...training completed in {timer.time_taken_str}.\n{tracker_msg}'
