from typing import *
import types, warnings, logging, threading
from abc import ABC, abstractmethod
import numpy as np
import time, traceback, pickle, gc, os, json
from synthergent.base.util import Registry, MutableParameters, Parameters, set_param_from_alias, is_list_like, as_list, \
    random_sample, safe_validate_arguments, format_exception_msg, StringUtil, get_fn_spec, Timer, type_str, \
    run_concurrent, get_result, Future, get_default, FunctionSpec, dispatch, is_function, Log, ThreadPoolExecutor, \
    ProgressBar, stop_executor, only_item, worker_ids, dispatch_executor
from synthergent.base.data import FileMetadata, ScalableDataFrame, ScalableSeries, Asset, ScalableDataFrameOrRaw
from synthergent.base.constants import Status, Parallelize, Alias, COMPLETED_STATUSES
from synthergent.base.util.notify import Notifier
from synthergent.base.framework.tracker import Tracker
from functools import partial
from pydantic import root_validator, Extra, conint, Field, constr, confloat
from pydantic.typing import Literal
from datetime import datetime

Step = "Step"
StepExecution = "StepExecution"
Chain = "Chain"
ChainExecution = "ChainExecution"


class Step(MutableParameters, Registry, ABC):
    _allow_multiple_subclasses = False
    _allow_subclass_override = True

    run_fn_spec: Optional[FunctionSpec] = None
    input_aliases: Dict[str, List[str]] = dict()
    tracker: Optional[Tracker] = None
    verbosity: conint(ge=0) = 1

    class Config(Parameters.Config):
        extra = Extra.ignore

    @classmethod
    def _pre_registration_hook(cls):
        run_fn_spec: FunctionSpec = get_fn_spec(cls.run)
        if len(run_fn_spec.args) > 0:
            warnings.warn(
                f'{cls.class_name}.run(...) must take only keyword args. Currently it has the following '
                f'non-keyword args: {run_fn_spec.args}. In general, the function signature of '
                f'{cls.class_name}.run(...) should have syntax: '
                f'`def run(self, *, a, b=16.0, c=("x", "y", "z")) -> Dict`.'
            )

    @classmethod
    def of(
            cls,
            step: Optional[Union[Type[Step], Step, Chain, Dict, str, Callable]] = None,
            **kwargs,
    ) -> Step:
        """Creates a new Step object. If the input is a Step, it creates a copy."""
        if step is None and 'name' in kwargs:
            step: Any = kwargs.pop('name')
        if is_function(step):
            return FunctionStep(fn=step, run_fn_spec=get_fn_spec(step))
        elif isinstance(step, Chain):
            step: ChainStep = ChainStep(chain=step)
            step.run_fn_spec = step.chain.steps[0].run_fn_spec
            return step
        elif isinstance(step, Step):
            step: Dict = step.dict()  ## We want to make a copy.
        if isinstance(step, dict):
            return cls.of(**{**step, **kwargs})

        if step is not None:
            if isinstance(step, type) and issubclass(step, Step):
                StepClass: Type[Step] = step
            elif isinstance(step, str):
                StepClass: Type[Step] = Step.get_subclass(step)
            else:
                raise NotImplementedError(f'Unsupported value for `step`: {type_str(step)} having value: {step}')
        else:
            StepClass: Type[Step] = cls
        if StepClass == Step:
            subclasses: List[str] = random_sample(
                as_list(Step.subclasses),
                n=3,
                replacement=False
            )
            raise ValueError(
                f'"{Step.class_name}" is an abstract class. '
                f'To create an instance, please either pass `step`, '
                f'or call .of(...) on a subclass of {Step.class_name}, e.g. {", ".join(subclasses)}'
            )
        if kwargs.get('run_fn_spec') is None:
            kwargs['run_fn_spec']: Tuple[str, ...] = get_fn_spec(StepClass.run)
        return StepClass(**kwargs)

    def copy(self) -> Step:
        return self.of(self.dict())

    def __call__(self, **kwargs):
        return self.run(**kwargs)

    @abstractmethod
    def run(self, **kwargs) -> Dict:
        pass

    def error(self, data: Any):
        if self.verbosity >= 1 and self.tracker is not None:
            self.tracker.info(data)

    def warning(self, data: Any):
        if self.verbosity >= 1 and self.tracker is not None:
            self.tracker.info(data)

    def info(self, data: Any):
        if self.verbosity >= 2 and self.tracker is not None:
            self.tracker.info(data)

    def debug(self, data: Any):
        if self.verbosity >= 3 and self.tracker is not None:
            self.tracker.info(data)

    def dict(self, **kwargs) -> Dict:
        d: Dict = super(Step, self).dict(**kwargs)
        ## Don't actually convert nested values to dicts, keep them as-is:
        d: Dict = {k: getattr(self, k) for k in d}
        d['step'] = self.class_name
        return d


class Chain(MutableParameters):
    """A template for creating chain executions."""

    steps: List[Union[Step, Dict, str]]
    inputs: Optional[Dict] = None
    outputs: Optional[Dict] = None
    verbosity: conint(ge=0) = 1

    @classmethod
    def _pre_registration_hook(cls):
        run_fn_spec: FunctionSpec = get_fn_spec(cls.run)
        if len(run_fn_spec.args) > 0:
            warnings.warn(
                f'{cls.class_name}.run(...) must take only keyword args. Currently it has the following '
                f'non-keyword args: {run_fn_spec.args}. In general, the function signature of '
                f'{cls.class_name}.run(...) should have syntax: '
                f'`def run(self, *, a, b=16.0, c=("x", "y", "z")) -> Dict`.'
            )

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @classmethod
    def of(cls, *steps, **kwargs) -> Chain:
        if len(steps) == 1 and is_list_like(steps[0]):
            steps: List = as_list(steps[0])
        if len(steps) == 0:
            raise ValueError(f'You must pass at least one step when calling {cls.class_name}.of(...)')
        return cls(steps=steps, **kwargs)

    @root_validator(pre=True)
    def set_chain_params(cls, params: Dict) -> Dict:
        set_param_from_alias(params, param='verbosity', alias=['verbose'])
        params['steps']: List[Step] = [
            Step.of(step)
            for step in params['steps']
        ]

        # for i in range(1, len(steps)):
        #     step_i_1_outputs: Set[str] = set(steps[i - 1].outputs)
        #     step_i_inputs: Set[str] = set(steps[i].run_fn_spec)
        #     if len(step_i_1_outputs - step_i_inputs) > 0:
        #         raise ValueError(
        #             f'Mismatch of inputs and outputs: '
        #             f'output of step index {i - 1} is {step_i_1_outputs}, '
        #             f'whereas input of step index {i} needs {step_i_inputs}; missing keys: {}'
        #         )
        return params

    @classmethod
    def _log_chain(
            cls,
            text: str,
            *,
            log_notifier: bool,
            tracker_fn: Callable,
            notifier_fn: Callable,
    ):
        ## Store full trace in log file, but don't always print it on sysout:
        tracker_fn(text)
        if log_notifier:
            notifier_fn(f'[{StringUtil.now(human=True, microsec=False)}] {text.strip()}')

    @classmethod
    def _get_chain_loggers(
            cls,
            verbosity: conint(ge=0),
            tracker: Tracker,
            notifier: Notifier,
    ) -> Tuple[Callable, Callable, Callable, Callable]:
        ## Tracker verbosity:
        ## Verbosity = 0: Nothing
        ## Verbosity = 1: Errors and warnings, no info or debug.
        ## Verbosity = 2: Errors, warnings, and info.
        ## Verbosity = 3: Errors, warnings, info, and debug.
        ## Default logging level is INFO. Thus, when verbosity=1 we should pass tracker.debug to info_logger.
        error_logger: Callable = partial(
            cls._log_chain,
            tracker_fn=partial(
                tracker.log,
                level=logging.ERROR if verbosity >= 1 else logging.DEBUG,
                prefix='[ERROR] ',
                flush=True,
            ),
            notifier_fn=notifier.send,
            log_notifier=(verbosity >= 1),
        )
        warning_logger: Callable = partial(
            cls._log_chain,
            tracker_fn=partial(
                tracker.log,
                level=logging.WARNING if verbosity >= 1 else logging.DEBUG,
                prefix='[WARN] ',
                flush=True,
            ),
            notifier_fn=notifier.send,
            log_notifier=(verbosity >= 1),
        )
        info_logger: Callable = partial(
            cls._log_chain,
            # log_tracker=(verbosity >= 2),  ## We will show the progress bar instead of logging to console.
            tracker_fn=partial(
                tracker.log,
                level=logging.INFO if verbosity >= 2 else logging.DEBUG,
                prefix='',
            ),
            notifier_fn=notifier.send,
            log_notifier=(verbosity >= 1),
        )
        debug_logger: Callable = partial(
            cls._log_chain,
            ## Unless we request silence (verbosity=0), print important information.
            tracker_fn=partial(
                tracker.log,
                level=logging.INFO if verbosity >= 3 else logging.DEBUG,
                prefix='[DEBUG] ',
            ),
            notifier_fn=notifier.send,
            log_notifier=(verbosity >= 3),
        )
        return error_logger, warning_logger, info_logger, debug_logger

    def __call__(self, **kwargs) -> Union[ChainExecution, Future]:
        return self.run(**kwargs)

    @safe_validate_arguments
    def run(
            self,
            *args,
            exn_name: Optional[constr(min_length=1)] = None,
            verbosity: conint(ge=0) = 1,
            tracker: Optional[Union[Tracker, Dict, str]] = None,
            notifier: Optional[Union[Notifier, Dict, str]] = None,
            background: bool = False,
            after: Optional[ChainExecution] = None,
            after_wait: conint(ge=0) = 60,
            step_wait: confloat(ge=0.0) = 0.0,
            step_wait_jitter: confloat(gt=0.0) = 0.8,
            **kwargs,
    ) -> Union[ChainExecution, Future]:
        if len(args) > 0:
            raise ValueError(f'Cannot only pass keyword arguments to {self.class_name}.run(...)')
        if after is not None:
            assert isinstance(after, ChainExecution)
        tracker: Tracker = Tracker.of(get_default(tracker, 'noop'))
        notifier: Union[Notifier, Dict, str] = get_default(notifier, 'noop')
        if isinstance(notifier, Notifier):
            notifier: Dict = notifier.dict()
        elif isinstance(notifier, str):
            notifier: Dict = dict(notifier=notifier)

        chain_exn: ChainExecution = ChainExecution(
            chain_template=self.clone(),
            steps=[],
            inputs={**kwargs},
            outputs=None,
            status=Status.PENDING,
        )
        try:
            parallelize: Parallelize = Parallelize.threads if background else Parallelize.sync
            chain_exn._executor: Optional[ThreadPoolExecutor] = dispatch_executor(
                parallelize=parallelize,
                max_workers=1,
            )
            fut: Future = dispatch(
                self._execute_chain,
                exn_name=exn_name,
                verbosity=verbosity,
                tracker=tracker.dict(),
                notifier=notifier,
                chain_exn=chain_exn,
                after=after,
                after_wait=after_wait,
                step_wait=step_wait,
                step_wait_jitter=step_wait_jitter,
                executor=chain_exn._executor,
                parallelize=parallelize,
                **kwargs,
            )
        except KeyboardInterrupt as e:
            chain_exn.stop(force=True)
            raise e
        finally:
            return chain_exn

    def _execute_chain(
            self,
            *,
            exn_name: Optional[constr(min_length=1)],
            verbosity: conint(ge=0),
            tracker: Dict,
            notifier: Dict,
            chain_exn: ChainExecution,
            after: Optional[ChainExecution],
            after_wait: conint(ge=0),
            step_wait: confloat(ge=0.0),
            step_wait_jitter: confloat(gt=0.0),
            **kwargs,
    ) -> NoReturn:
        if after is not None:
            while after.status not in COMPLETED_STATUSES:
                time.sleep(after_wait)
        timer: Timer = Timer(silent=True)
        timer.start()
        exn_name: str = get_default(exn_name, 'Chain')
        notifier: Notifier = Notifier.of(**notifier)
        tracker: Tracker = Tracker.of({**tracker, **dict(init_msg=False)})
        progress_bar: Optional[Dict] = Alias.get_progress_bar(kwargs)
        chain_exn_pbar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=self.num_steps,
            desc=exn_name,
            unit='step',
            disable=False if verbosity >= 1 else True,
        )
        error_logger, warning_logger, info_logger, debug_logger = self._get_chain_loggers(
            verbosity=verbosity,
            tracker=tracker,
            notifier=notifier,
        )
        all_kwargs: Dict = {**kwargs}
        step_i, step = 0, self.steps[0]
        try:
            chain_exn.started_at = timer.start_datetime
            info_logger(
                f'{exn_name}: '
                f'Execution started in process#{os.getpid()}, '
                f'thread#{threading.get_ident()} '
                f'at {StringUtil.now()}.'
            )
            for step_i, step in enumerate(self.steps):
                if chain_exn.status is Status.STOPPED:
                    break
                chain_exn.status = Status.RUNNING
                debug_logger(
                    f'Running {exn_name}'
                    f'Step {StringUtil.pad_zeros(step_i + 1, self.num_steps)}/{self.num_steps}: '
                    f'"{step.class_name}"...'
                )
                chain_exn_pbar.set_description(
                    desc=f'{exn_name}: Running "{step.class_name}" '
                )
                step: Step = step.clone()
                ## Start running the step:
                step_timer: Timer = Timer(silent=True)
                step_timer.start()
                step_inputs: Dict = self._create_step_inputs(
                    exn_name=exn_name,
                    step=step,
                    step_i=step_i,
                    num_steps=self.num_steps,
                    all_kwargs=all_kwargs,
                )
                step_exn: StepExecution = StepExecution(
                    step_template=step,
                    inputs=step_inputs,
                    started_at=step_timer.start_datetime,
                    status=Status.RUNNING,
                )
                step.tracker = tracker  ## Set the tracker for use by the step's exeuction.
                step.verbosity = verbosity
                step.info('')  ## Adds a newline
                try:
                    ## Actually run the step:
                    step_outputs: Dict = step.run(**step_inputs)
                    if not isinstance(step_outputs, dict):
                        raise ValueError(
                            f'{exn_name} Step {StringUtil.pad_zeros(step_i + 1, self.num_steps)}/{self.num_steps} '
                            f'"{step.class_name}" '
                            f'should return a dict; found {type_str(step_outputs)}'
                        )
                    step_exn.outputs = step_outputs
                    step_exn.status = Status.SUCCEEDED
                    all_kwargs: Dict = {
                        **all_kwargs,
                        **step_outputs,
                    }
                    debug_logger(
                        f'...Step '
                        f'[{StringUtil.pad_zeros(step_i + 1, self.num_steps)}/{self.num_steps}] '
                        f'({step.class_name}) '
                        f'completed in {step_timer.time_taken_str}.'
                    )
                    debug_logger(
                        f'Outputs:\n{str(step_outputs)}'
                    )
                    debug_logger('═' * 72)
                    time.sleep(np.random.uniform(
                        step_wait - step_wait * step_wait_jitter,
                        step_wait + step_wait * step_wait_jitter,
                    ))
                except Exception as e:
                    step_exn.status = Status.FAILED
                    raise e
                finally:
                    step_timer.stop()
                    step_exn.completed_at = step_timer.end_datetime
                    step.tracker = None  ## Unset the tracker
                    chain_exn.steps.append(step_exn)
                    chain_exn_pbar.update(1)
                    gc.collect()
            if chain_exn.num_executed_steps == self.num_steps:
                ## All steps were executed, successfully.
                ## If we stopped, it means we stopped during the last step, which executed successfully...in this case
                ## it makes more sense to mark the chain as SUCCEEDED rather than STOPPED.
                chain_exn.status = Status.SUCCEEDED
                chain_exn.outputs = {**all_kwargs}
        except Exception as e:
            chain_exn.status = Status.FAILED
            error_logger(f'Error in {exn_name}:\n{format_exception_msg(e)}')
            chain_exn.error = e
        finally:
            timer.stop()
            chain_exn.completed_at = timer.end_datetime
            if chain_exn.success():
                info_logger(
                    f'\n{exn_name}: Execution succeeded. '
                    f'{self.num_steps} Chain steps completed in {timer.time_taken_str}.'
                )
                chain_exn_pbar.success(f'{exn_name}', append_desc=False)
            elif chain_exn.stopped():
                info_logger(
                    f'{exn_name}: Execution was stopped before starting '
                    f'Step ({StringUtil.pad_zeros(step_i + 1, self.num_steps)}/{self.num_steps}) '
                    f'"{step.class_name}". Previous steps ran for {timer.time_taken_str}.'
                )
                chain_exn_pbar.stopped()
            elif chain_exn.failed():
                error_logger(
                    f'{exn_name}: Execution failed during '
                    f'Step {StringUtil.pad_zeros(step_i + 1, self.num_steps)}/{self.num_steps} '
                    f'"{step.class_name}". Execution ran for {timer.time_taken_str}.'
                )
                chain_exn_pbar.failed(
                    f'Failed at {exn_name} {StringUtil.pad_zeros(step_i + 1, self.num_steps)}/{self.num_steps}: '
                    f'"{step.class_name}"',
                    append_desc=False,
                )
            if chain_exn.success() or chain_exn.failed():
                _executor = chain_exn._executor
                chain_exn._executor = None
                stop_executor(_executor, force=True)  ## Stop the thread forcefully

    def copy(self) -> Chain:
        return self.update_params(steps=[Step.of(step.dict()) for step in self.steps])

    @classmethod
    def _create_step_inputs(
            cls,
            exn_name: Optional[constr(min_length=1)],
            step: Step,
            step_i: int,
            num_steps: int,
            all_kwargs: Dict[str, Any],
    ) -> Dict:
        all_kwargs: Dict[str, Any] = {**all_kwargs}
        for param, aliases in step.input_aliases.items():
            set_param_from_alias(all_kwargs, param=param, alias=aliases)
        step_inputs: Dict[str, Any] = {
            param: val
            for param, val in all_kwargs.items()
            if param in step.run_fn_spec.args_and_kwargs
        }
        if isinstance(step, ChainStep):
            step_inputs['exn_name']: str = \
                f'({exn_name}: Step {StringUtil.pad_zeros(step_i + 1, num_steps)}/{num_steps})'
        missing_keys: Set[str] = set(step.run_fn_spec.required_args_and_kwargs) - set(step_inputs.keys())
        if len(missing_keys) > 0:
            raise ValueError(
                f'Step {step_i + 1}/{num_steps} "{step.class_name}" needs inputs {missing_keys} '
                f'which are not present in the data keys.'
            )
        return step_inputs

    def all_step_inputs(self, *, required_only: bool = True) -> Set[str]:
        all_step_input_keys: Set[str] = set()
        for step in self.steps:
            if required_only:
                all_step_input_keys.update(set(step.run_fn_spec.required_args_and_kwargs))
            else:
                all_step_input_keys.update(set(step.run_fn_spec.args_and_kwargs))
        return all_step_input_keys

    # @safe_validate_arguments
    # def rerun(
    #         self,
    #         *,
    #         start: Optional[conint(ge=0)] = None,  ## Zero-indexed, inclusive
    #         background: Optional[bool] = None,
    #         **kwargs,
    # ) -> Optional[Chain]:
    #     if background is not None:
    #         raise ValueError(f'Cannot rerun chain in the background.')
    #     if not self.is_called:
    #         raise ValueError(f'Cannot rerun a {self.class_name} which has not been called.')
    #     set_param_from_alias(kwargs, param='start', alias=['from_idx', 'from', 'start_at'])
    #     start: Optional[int] = kwargs.pop('start', start)
    #     start: int = get_default(start, 0)
    #     if start > (len(self.steps) - 1):
    #         raise ValueError(
    #             f'Chain has {len(self.steps)} steps, thus permissible values of `start` are [{0}, {len(self.steps)}] '
    #             f'(inclusive); found: start={start}.'
    #         )
    #     end_chain_inputs: Dict = {**self.inputs}
    #     chain_new_exn_steps: List[Step] = []
    #     for step_i, step in enumerate(self.steps):
    #         if step_i < start:
    #             chain_new_exn_steps.append(step)
    #             end_chain_inputs: Dict = {**end_chain_inputs, **step.outputs}
    #     end_chain_inputs: Dict = {**self.inputs, **kwargs}
    #     end_chain_steps: List[Dict] = [
    #         step.dict()
    #         for step_i, step in enumerate(self.steps)
    #         if step_i >= start
    #     ]
    #     end_chain: Chain = self.clone(steps=end_chain_steps)
    #     outputs: Dict = end_chain._execute_chain(**{
    #         **end_chain_inputs,
    #         **kwargs,
    #         **dict(_rerun_step_increment=start)
    #     })
    #     for step in end_chain.steps:
    #         chain_new_exn_steps.append(step)
    #
    #     chain_new_exn: Chain = self.clone()
    #     chain_new_exn.inputs: Dict = {**self.inputs}
    #     chain_new_exn.outputs: Dict = {**end_chain_inputs, **outputs}
    #     chain_new_exn.steps: List[Step] = chain_new_exn_steps
    #
    #     return chain_new_exn


class StepExecution(MutableParameters):
    step_template: Step
    inputs: Dict
    outputs: Optional[Dict] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: Status

    def done(self) -> bool:
        return self.status in COMPLETED_STATUSES

    def running(self) -> bool:
        return self.status in {Status.RUNNING}

    def success(self) -> bool:
        return self.status is Status.SUCCEEDED

    def failed(self) -> bool:
        return self.status is Status.FAILED


class ChainExecution(MutableParameters):
    uuid: Optional[constr(min_length=6)] = None
    chain_template: Chain
    steps: List[StepExecution]
    inputs: Dict
    outputs: Optional[Dict] = None
    status: Status
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    _executor: Optional[ThreadPoolExecutor] = None
    error: Optional[Exception] = None

    @root_validator(pre=False)
    def _chain_execution_params(cls, params: Dict) -> Dict:
        if params.get('uuid') is None:
            chain_template: Chain = params['chain_template']
            uuid_dict: Dict = dict(
                start_dt=StringUtil.now(),
                start_time_ns=str(time.time_ns()),
                step_classes='_'.join([step.class_name for step in chain_template.steps]),
            )
            params['uuid'] = f"chain-exn-{StringUtil.hash(uuid_dict, max_len=12)}"
        return params

    @property
    def num_executed_steps(self) -> int:
        return len(self.steps)

    @property
    def num_successful_steps(self) -> int:
        return sum([True for step in self.steps if step.status is Status.SUCCEEDED])

    @property
    def current_running_step(self) -> Optional[Step]:
        if not self.running():
            return None
        return self.chain_template.steps[self.num_executed_steps].copy()

    @property
    def is_not_completed(self) -> bool:
        return not self.done()

    def done(self) -> bool:
        return self.status in COMPLETED_STATUSES

    def running(self) -> bool:
        return self.status in {Status.RUNNING}

    def success(self) -> bool:
        return self.status is Status.SUCCEEDED

    def failed(self) -> bool:
        return self.status is Status.FAILED

    def stopped(self) -> bool:
        return self.status is Status.STOPPED

    def wait(self, *, timeout: Optional[confloat(ge=0)] = None, pause: confloat(ge=0) = 10):
        start: float = time.time()
        while not self.done():
            time.sleep(pause)
            if timeout is not None and time.time() - start > timeout:
                return

    def stop(self, force: bool = True):
        """Stops the Chain's execution *after* the currently-running step succeeds."""
        _executor = self._executor
        stop_executor(_executor, force=force)
        self._executor = None
        if self.status not in {Status.SUCCEEDED, Status.FAILED}:
            self.status = Status.STOPPED  ## This will cause the chain execution loop to exit after the current step


class FunctionStep(Step):
    fn: Any

    def run(self, **kwargs) -> Dict:
        return self.fn(**kwargs)


class ChainStep(Step):
    chain: Chain

    def run(self, **kwargs) -> Dict:
        kwargs['background']: bool = False
        kwargs['verbosity']: int = 1
        kwargs['tracker']: Optional[Tracker] = None
        kwargs['notifier']: Optional[Notifier] = None
        chain_exn: ChainExecution = self.chain.run(**kwargs)
        chain_exn.wait()
        if chain_exn.error is not None:
            raise chain_exn.error
        outputs: Dict = chain_exn.outputs
        assert isinstance(outputs, dict)
        outputs['__exn__'] = chain_exn
        return outputs
