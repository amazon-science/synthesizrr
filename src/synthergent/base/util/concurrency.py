"""A collection of concurrency utilities to augment the Python language:"""
import logging
from typing import *
import time, traceback, random, sys, math, gc
from datetime import datetime
from math import inf
import numpy as np
import asyncio, ctypes
from threading import Semaphore, Thread, Lock
import multiprocessing as mp
from concurrent.futures._base import Future, Executor
from concurrent.futures.thread import BrokenThreadPool
from concurrent.futures.process import BrokenProcessPool
import ray
from ray.exceptions import GetTimeoutError
from ray.util.dask import RayDaskCallback
from pydantic import conint, confloat, Extra, root_validator
from synthergent.base.util.language import ProgressBar, set_param_from_alias, type_str, get_default, first_item, Parameters, \
    is_list_or_set_like, is_dict_like, PandasSeries, filter_kwargs, format_exception_msg, AutoEnum, auto
from synthergent.base.util.profiling import Timer
from synthergent.base.constants.DataProcessingConstants import Parallelize, FailureAction, Status, COMPLETED_STATUSES
from functools import partial
## Jupyter-compatible asyncio usage:
import asyncio
import threading
import time, inspect, queue, uuid, cloudpickle, warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait as wait_future


class ThreadKilledSystemException(BaseException):
    """Custom exception for killing threads."""
    pass


class ThreadKilledSystemExceptionFilter(logging.Filter):
    def filter(self, record):
        if record.exc_info:
            exc_type = record.exc_info[0]
            if exc_type.__name__ == 'ThreadKilledSystemException':
                return False
        return True


def suppress_ThreadKilledSystemException():
    for _logger_module in ['concurrent.futures', 'ipykernel', 'ipykernel.ipykernel']:
        _logger = logging.getLogger(_logger_module)
        _filter_exists: bool = False
        for _filter in _logger.filters:
            if _filter.__class__.__name__ == 'ThreadKilledSystemExceptionFilter':
                _filter_exists: bool = True
                # print(f'{_filter.__class__.__name__} exists in {_logger_module} filters')
                break
        if not _filter_exists:
            _logger.addFilter(ThreadKilledSystemExceptionFilter())
            # print(f'{ThreadKilledSystemExceptionFilter} added to {_logger_module} filters')


suppress_ThreadKilledSystemException()

_RAY_ACCUMULATE_ITEM_WAIT: float = 10e-3  ## 10ms
_LOCAL_ACCUMULATE_ITEM_WAIT: float = 1e-3  ## 1ms

_RAY_ACCUMULATE_ITER_WAIT: float = 1000e-3  ## 1000ms
_LOCAL_ACCUMULATE_ITER_WAIT: float = 100e-3  ## 100ms


class LoadBalancingStrategy(AutoEnum):
    ROUND_ROBIN = auto()
    LEAST_USED = auto()
    UNUSED = auto()
    RANDOM = auto()


def _asyncio_start_event_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


# Create a new loop and a thread running this loop
_ASYNCIO_EVENT_LOOP = asyncio.new_event_loop()
_ASYNCIO_EVENT_LOOP_THREAD = threading.Thread(target=_asyncio_start_event_loop, args=(_ASYNCIO_EVENT_LOOP,))
_ASYNCIO_EVENT_LOOP_THREAD.start()


## Async wrapper to run a synchronous function in the event loop
async def __run_fn_async(fn, *args, run_sync_in_executor: bool = True, **kwargs):
    if inspect.iscoroutinefunction(fn):
        ## If fn is defined with `def async`, run this using asyncio mechanism,
        ## meaning code inside fn is run in an sync way, except for the "await"-marked lines, which will
        ## be run asynchronously. Note that "await"-marked lines must call other functions defined using "def async".
        result = await fn(*args, **kwargs)
    else:
        ## The function is a sync function.
        if run_sync_in_executor:
            ## Run in the default executor (thread pool) for the event loop, otherwise it blocks the event loop
            ## until the function execution completes.
            ## The executor lives for the lifetime of the event loop. Ref: https://stackoverflow.com/a/33399896/4900327
            ## This basically is the same as run_concurrent, but with no control on the number of threads.
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, partial(fn, *args, **kwargs))
        else:
            ## Run the sync function synchronously. This will block the event loop until the function completes.
            result = fn(*args, **kwargs)
    return result


## Function to submit the coroutine to the asyncio event loop
def run_asyncio(fn, *args, **kwargs):
    ## Create a coroutine (i.e. Future), but do not actually start executing it.
    coroutine = __run_fn_async(fn, *args, **kwargs)
    ## Schedule the coroutine to execute on the event loop (which is running on thread _ASYNCIO_EVENT_LOOP_THREAD).
    return asyncio.run_coroutine_threadsafe(coroutine, _ASYNCIO_EVENT_LOOP)


async def async_http_get(url):
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


def concurrent(max_workers: int = 10, max_calls_per_second: float = inf):
    """
    Decorator which runs function calls concurrently via multithreading.
    When decorating an IO-bound function with @concurrent(MAX_THREADS), and then invoking the function
    N times in a loop, it will run min(MAX_THREADS, N) invocations of the function concurrently.
    For example, if your function calls another service, and you must invoke the function N times, decorating with
    @concurrent(3) ensures that you only have 3 concurrent function-calls at a time, meaning you only make
    3 concurrent requests at a time. This reduces the number of connections you are making to the downstream service.
    As this uses multi-threading and not multi-processing, it is suitable for IO-heavy functions, not CPU-heavy.

    Each call  to the decorated function returns a future. Calling .result() on that future will return the value.
    Generally, you should call the decorated function N times in a loop, and store the futures in a list/dict. Then,
    call .result() on all the futures, saving the results in a new list/dict. Each .result() call is synchronous, so the
    order of items is maintained between the lists. When doing this, at most min(MAX_THREADS, N) function calls will be
    running concurrently.
    Note that if the function calls throws an exception, then calling .result() will raise the exception in the
    orchestrating code. If multiple function calls raise an exception, the one on which .result() was called first will
    throw the exception to the orchestrating code.  You should add try-catch logic inside your decorated function to
    ensure exceptions are handled.
    Note that decorated function `a` can call another decorated function `b` without issues; it is upto the function A
    to determine whether to call .result() on the futures it gets from `b`, or return the future to its own invoker.

    `max_calls_per_second` controls the rate at which we can call the function. This is particularly important for
    functions which execute quickly: e.g. suppose the decorated function calls a downstream service, and we allow a
    maximum concurrency of 5. If each function call takes 100ms, then we end up making 1000/100*5 = 50 calls to the
    downstream service each second. We thus should pass `max_calls_per_second` to restrict this to a smaller value.

    :param max_workers: the max number of threads which can be running the function at one time. This is thus
    them max concurrency factor.
    :param max_calls_per_second: controls the rate at which we can call the function.
    :return: N/A, this is a decorator.
    """

    ## Refs:
    ## 1. ThreadPoolExecutor: docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor.submit
    ## 2. Decorators: www.datacamp.com/community/tutorials/decorators-python
    ## 3. Semaphores: www.geeksforgeeks.org/synchronization-by-using-semaphore-in-python/
    ## 4. Overall code: https://gist.github.com/gregburek/1441055#gistcomment-1294264
    def decorator(function):
        ## Each decorated function gets its own executor and semaphore. These are defined at the function-level, so
        ## if you write two decorated functions `def say_hi` and `def say_bye`, they each gets a separate executor and
        ## semaphore. Then, if you invoke `say_hi` 30 times and `say_bye` 20 times, all 30 calls to say_hi will use the
        ## same executor and semaphore, and all 20 `say_bye` will use a different executor and semaphore. The value of
        ## `max_workers` will determine how many function calls actually run concurrently, e.g. if say_hi has
        ## max_workers=5, then the 30 calls will run 5 at a time (this is enforced by the semaphore).
        executor = ThreadPoolExecutor(max_workers=max_workers)
        semaphore = Semaphore(max_workers)

        ## The minimum time between invocations.
        min_time_interval_between_calls = 1 / max_calls_per_second
        ## This only stores a single value, but it must be a list (mutable) for Python's function scoping to work.
        time_last_called = [0.0]

        def wrapper(*args, **kwargs) -> Future:
            semaphore.acquire()
            time_elapsed_since_last_called = time.time() - time_last_called[0]
            time_to_wait_before_next_call = max(0.0, min_time_interval_between_calls - time_elapsed_since_last_called)
            time.sleep(time_to_wait_before_next_call)

            def run_function(*args, **kwargs):
                try:
                    result = function(*args, **kwargs)
                finally:
                    semaphore.release()  ## If the function call throws an exception, release the semaphore.
                return result

            time_last_called[0] = time.time()
            return executor.submit(run_function, *args, **kwargs)  ## return a future

        return wrapper

    return decorator


class RestrictedConcurrencyThreadPoolExecutor(ThreadPoolExecutor):
    """
    This executor restricts concurrency (max active threads) and, optionally, rate (max calls per second).
    It is similar in functionality to the @concurrent decorator, but implemented at the executor level.
    """

    def __init__(
            self,
            max_workers: Optional[int] = None,
            *args,
            max_calls_per_second: float = float('inf'),
            **kwargs,
    ):
        if max_workers is None:
            max_workers: int = min(32, (mp.cpu_count() or 1) + 4)
        if not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError(f'Expected `max_workers`to be a non-negative integer.')
        kwargs['max_workers'] = max_workers
        super().__init__(*args, **kwargs)
        self._semaphore = Semaphore(max_workers)
        self._max_calls_per_second = max_calls_per_second

        # If we have an infinite rate, don't enforce a delay
        self._min_time_interval_between_calls = 1 / self._max_calls_per_second

        # Tracks the last time a call was started (not finished, just started)
        self._time_last_called = 0.0
        self._lock = Lock()  # Protects access to _time_last_called

    def submit(self, fn, *args, **kwargs):
        # Enforce concurrency limit
        self._semaphore.acquire()

        # Rate limiting logic: Before starting a new call, ensure we wait long enough if needed
        if self._min_time_interval_between_calls > 0.0:
            with self._lock:
                time_elapsed_since_last_called = time.time() - self._time_last_called
                time_to_wait = max(0.0, self._min_time_interval_between_calls - time_elapsed_since_last_called)

            # Wait the required time
            if time_to_wait > 0:
                time.sleep(time_to_wait)

            # Update the last-called time after the wait
            with self._lock:
                self._time_last_called = time.time()
        else:
            # No rate limiting, just update the last-called time
            with self._lock:
                self._time_last_called = time.time()

        future = super().submit(fn, *args, **kwargs)
        # When the task completes, release the semaphore to allow another task to start
        future.add_done_callback(lambda _: self._semaphore.release())
        return future


_GLOBAL_THREAD_POOL_EXECUTOR: ThreadPoolExecutor = RestrictedConcurrencyThreadPoolExecutor(max_workers=16)


def run_concurrent(
        fn,
        *args,
        executor: Optional[ThreadPoolExecutor] = None,
        **kwargs,
):
    global _GLOBAL_THREAD_POOL_EXECUTOR
    if executor is None:
        executor: ThreadPoolExecutor = _GLOBAL_THREAD_POOL_EXECUTOR
    try:
        # logging.debug(f'Running {fn_str(fn)} using {Parallelize.threads} with max_workers={executor._max_workers}')
        return executor.submit(fn, *args, **kwargs)  ## return a future
    except BrokenThreadPool as e:
        if executor is _GLOBAL_THREAD_POOL_EXECUTOR:
            executor = RestrictedConcurrencyThreadPoolExecutor(max_workers=_GLOBAL_THREAD_POOL_EXECUTOR._max_workers)
            del _GLOBAL_THREAD_POOL_EXECUTOR
            _GLOBAL_THREAD_POOL_EXECUTOR = executor
            return executor.submit(fn, *args, **kwargs)  ## return a future
        raise e


def actor_process_main(cls_bytes, init_args, init_kwargs, command_queue, result_queue):
    cls = cloudpickle.loads(cls_bytes)
    instance = None
    while True:
        command = command_queue.get()
        if command is None:
            break
        request_id, method_name, args, kwargs = command
        try:
            if method_name == "__initialize__":
                instance = cls(*init_args, **init_kwargs)
                result_queue.put((request_id, "ok", None))
                continue
            if instance is None:
                raise RuntimeError("Actor instance not initialized.")
            method = getattr(instance, method_name, None)
            if method is None:
                raise AttributeError(f"Method '{method_name}' not found.")
            result = method(*args, **kwargs)
            result_queue.put((request_id, "ok", result))
        except Exception as e:
            tb_str = traceback.format_exc()
            result_queue.put((request_id, "error", (e, tb_str)))


class ActorProxy:
    def __init__(self, cls, init_args, init_kwargs, mp_context: Literal['fork', 'spawn']):
        assert mp_context in {"fork", "spawn"}
        ctx = mp.get_context(mp_context)

        self._uuid = str(uuid.uuid4())

        self._command_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        self._num_submitted: int = 0
        self._task_status: Dict[Status, int] = {
            Status.PENDING: 0,
            Status.RUNNING: 0,
            Status.SUCCEEDED: 0,
            Status.FAILED: 0,
        }

        self._futures = {}
        self._futures_lock = threading.Lock()

        # Create the process using the fork context
        cls_bytes = cloudpickle.dumps(cls)
        self._cls_name = cls.__name__
        self._process: ctx.Process = ctx.Process(
            target=actor_process_main,
            args=(cls_bytes, init_args, init_kwargs, self._command_queue, self._result_queue)
        )
        self._process.start()

        # Synchronous initialization
        self._invoke_sync_initialize()

        self._stopped = False

        # Now start the asynchronous result handling using a thread:
        self._result_thread = threading.Thread(target=self._handle_results, daemon=True)
        self._result_thread.start()

    def _handle_results(self):
        while True:
            if not self._process.is_alive() and self._result_queue.empty():
                self._task_status[Status.RUNNING] = 0
                return
            try:
                item = self._result_queue.get(timeout=1)
            except queue.Empty:
                self._task_status[Status.RUNNING] = 0
                continue
            if item is None:  # Sentinel to stop the results-handling thread.
                return
            request_id, status, payload = item
            with self._futures_lock:
                future = self._futures.pop(request_id, None)
            if future is not None:
                if status == "ok":
                    future.set_result(payload)
                    self._task_status[Status.SUCCEEDED] += 1
                else:
                    e, tb_str = payload
                    future.set_exception(RuntimeError(f"Remote call failed:\n{tb_str}"))
                    self._task_status[Status.FAILED] += 1
                self._task_status[Status.PENDING] -= 1

    def _invoke_sync_initialize(self):
        request_id = self._uuid
        self._command_queue.put((request_id, "__initialize__", (), {}))
        # Direct, blocking call to get the response
        rid, status, payload = self._result_queue.get()
        if status == "error":
            e, tb_str = payload
            raise RuntimeError(f"Remote init failed:\n{tb_str}")

    def stop(self, timeout: int = 10, cancel_futures: bool = True):
        if self._stopped is True:
            return
        self._stopped = True
        self._command_queue.put(None)
        self._process.join(timeout=timeout)
        self._command_queue.close()
        self._result_queue.close()
        # Fail any remaining futures
        if cancel_futures:
            with self._futures_lock:
                for fut in self._futures.values():
                    if not fut.done():
                        fut.set_exception(RuntimeError("Actor stopped before completion."))
                self._futures.clear()
        self._task_status[Status.RUNNING] = 0

    def _invoke(self, method_name, *args, **kwargs):
        if self._stopped is True:
            raise RuntimeError("Cannot invoke methods on a stopped actor.")
        future = Future()
        request_id = str(uuid.uuid4())
        with self._futures_lock:
            self._futures[request_id] = future
        self._command_queue.put((request_id, method_name, args, kwargs))
        self._num_submitted += 1
        self._task_status[Status.PENDING] += 1
        if self._process.is_alive():
            self._task_status[Status.RUNNING] = 1
        return future

    def submitted(self) -> int:
        return self._num_submitted

    def pending(self) -> int:
        return self._task_status[Status.PENDING]

    def running(self) -> int:
        return self._task_status[Status.RUNNING]

    def succeeded(self) -> int:
        return self._task_status[Status.SUCCEEDED]

    def failed(self) -> int:
        return self._task_status[Status.FAILED]

    def __getattr__(self, name):
        # Instead of returning a direct callable, we return a RemoteMethod wrapper
        return RemoteMethod(self, name, self._cls_name)

    def __del__(self):
        try:
            if not self._stopped and self._process.is_alive():
                self.stop()
        except Exception:
            pass


class RemoteMethod:
    """
    A wrapper object returned by ActorProxy.__getattr__.
    To call the method remotely, use .remote(*args, **kwargs).
    """

    def __init__(self, proxy, method_name, cls_name):
        self._proxy = proxy
        self._method_name = method_name
        self._cls_name = cls_name

    def remote(self, *args, **kwargs):
        return self._proxy._invoke(self._method_name, *args, **kwargs)

    def options(self, *args, **kwargs):
        warnings.warn(f'The process-based Actor "{self._cls_name}" cannot use .options(); this call will be ignored.')
        return self


"""
Note: By default we use a `mp_context="fork"` for Actor creation.
Process creation is much slower under spawn than forking. For example:
- On a MacOS machine, Actor creation time is 20 milliseconds (forking) vs 7 seconds (spawn).
- On a Linux machine, Actor creation time is 20 milliseconds (forking) vs 17 seconds (spawn).

However, forking comes with caveats which are not present in spawn:
1. Copy-on-Write Memory Behavior:
On Unix-like systems (including MacOS), forked processes share the same memory pages as the parent initially.
These pages are not immediately copied; instead, they are marked copy-on-write.
This means:
- No immediate bulk copy: Your large data structures (like Pandas DataFrames) do not get physically copied into memory
right away.
- Copies on modification: If either the parent or the child modifies a shared page, only then is that page actually
copied. Thus, if the child process reads from large data structures without writing to them, the overhead remains
relatively low. But if it modifies them, the memory cost could jump significantly.

2. Potential Resource and Concurrency Issues:
Forking a process that already has multiple threads, open file descriptors, or other system resources can lead to
subtle bugs. Some libraries, particularly those relying on threading or certain system calls, may not be “fork-safe.”
Common issues include:
- Thread State: The child process starts with a copy of the parent’s memory but only one thread running (the one that
called fork). Any locks or conditions held by threads in the parent at the time of fork can lead to deadlocks or
inconsistent states.
- External Resources: Network sockets, open database connections, or other system resources may not be safe to use in
the child after fork without an exec. They might appear duplicated but can behave unexpectedly or lead to errors if
not reinitialized.
- Library Incompatibilities: Some libraries are not tested or guaranteed to work correctly in forked children. They
might rely on internal threading, which can break post-fork.
"""
_DEFAULT_ACTOR_PROCESS_CREATION_METHOD: Literal['fork', 'spawn'] = 'fork'


class Actor:
    @classmethod
    def remote(cls, *args, mp_context: Literal['fork', 'spawn'] = _DEFAULT_ACTOR_PROCESS_CREATION_METHOD, **kwargs):
        return ActorProxy(
            cls,
            init_args=args,
            init_kwargs=kwargs,
            mp_context=mp_context,
        )

    @classmethod
    def options(cls, *args, **kwargs):
        warnings.warn(f'The process-based Actor "{cls.__name__}" cannot use .options(); this call will be ignored.')
        return cls


def actor(cls, mp_context: Literal['fork', 'spawn'] = _DEFAULT_ACTOR_PROCESS_CREATION_METHOD):
    """
    Class decorator that transforms a regular class into an actor-enabled class.
    The decorated class gains a .remote(*args, **kwargs) class method that
    returns an ActorProxy running in a separate process.
    """

    def remote(*args, **kwargs):
        return ActorProxy(
            cls,
            init_args=args,
            init_kwargs=kwargs,
            mp_context=mp_context,
        )

    def options(cls, *args, **kwargs):
        warnings.warn(f'The process-based Actor "{cls.__name__}" cannot use .options(); this call will be ignored.')
        return cls

    cls.remote = remote
    cls.options = options
    return cls


@actor
class TaskActor:
    """
    A generic actor that can run an arbitrary callable passed to it.
    We'll send (func, args, kwargs) as serialized objects and it will run them.
    """

    def __init__(self):
        pass

    def run_callable(self, func_bytes, args, kwargs):
        func = cloudpickle.loads(func_bytes)
        return func(*args, **kwargs)


class ActorPoolExecutor(Executor):
    """
    A simple ActorPoolExecutor that mimics the ProcessPoolExecutor interface,
    but uses a pool of TaskActor instances for parallel execution.
    """

    def __init__(
            self,
            max_workers: Optional[int] = None,
            *,
            load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
    ):
        if max_workers is None:
            max_workers = mp.cpu_count() - 1
        self._actors: List[ActorProxy] = [TaskActor.remote() for _ in range(max_workers)]
        self._actor_index = 0
        self._max_workers = max_workers
        self._load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy(load_balancing_strategy)
        self._shutdown_lock = threading.Lock()
        self._futures = []
        self._shutdown = False

    def submit(self, fn, *args, **kwargs):
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("Cannot submit tasks after shutdown")

        func_bytes = cloudpickle.dumps(fn)
        if self._load_balancing_strategy is LoadBalancingStrategy.ROUND_ROBIN:
            actor = self._actors[self._actor_index]
            self._actor_index = (self._actor_index + 1) % self._max_workers
        elif self._load_balancing_strategy is LoadBalancingStrategy.RANDOM:
            actor = random.choice(self._actors)
        elif self._load_balancing_strategy is LoadBalancingStrategy.LEAST_USED:
            actor = sorted([(_actor, _actor.pending()) for _actor in self._actors], key=lambda x: x[1])[0]
        elif self._load_balancing_strategy is LoadBalancingStrategy.UNUSED:
            actor = sorted([(_actor, _actor.running()) for _actor in self._actors], key=lambda x: x[1])[0]
        else:
            raise NotImplementedError(f'Unsupported load_balancing_strategy: {self._load_balancing_strategy}')
        future = actor.run_callable.remote(func_bytes, args, kwargs)
        self._remove_completed_futures()
        self._futures.append(future)
        return future

    def _remove_completed_futures(self):
        self._futures = [fut for fut in self._futures if not fut.done()]

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = True) -> None:
        with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True

        # If wait=True, wait for all futures to complete
        if wait:
            for fut in self._futures:
                fut.result()  # blocks until future is done or raises
        self._remove_completed_futures()
        # Stop all actors
        for actor in self._actors:
            actor.stop(cancel_futures=cancel_futures)

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        if chunksize != 1:
            raise NotImplementedError("chunksize other than 1 is not implemented")

        inputs = zip(*iterables)
        futures = [self.submit(fn, *args) for args in inputs]

        # Yield results in order
        for fut in futures:
            yield fut.result(timeout=timeout)


_GLOBAL_PROCESS_POOL_EXECUTOR: ActorPoolExecutor = ActorPoolExecutor(
    max_workers=max(1, min(32, mp.cpu_count() - 1))
)


def run_parallel(
        fn,
        *args,
        executor: Optional[Union[ProcessPoolExecutor, ActorPoolExecutor]] = None,
        **kwargs,
):
    global _GLOBAL_PROCESS_POOL_EXECUTOR
    if executor is None:
        executor: ActorPoolExecutor = _GLOBAL_PROCESS_POOL_EXECUTOR
    try:
        # print(f'Running {fn_str(fn)} using {Parallelize.threads} with max_workers={executor._max_workers}')
        return executor.submit(fn, *args, **kwargs)  ## return a future
    except BrokenProcessPool as e:
        if executor is _GLOBAL_PROCESS_POOL_EXECUTOR:
            executor = ActorPoolExecutor(max_workers=_GLOBAL_PROCESS_POOL_EXECUTOR._max_workers)
            del _GLOBAL_PROCESS_POOL_EXECUTOR
            _GLOBAL_PROCESS_POOL_EXECUTOR = executor
            return executor.submit(fn, *args, **kwargs)  ## return a future
        raise e


def kill_thread(tid: int):
    """
    Dirty hack to *actually* stop a thread: raises an exception in threads with this thread id.
    How it works:
    - kill_thread function uses ctypes.pythonapi.PyThreadState_SetAsyncExc to raise an exception in a thread.
    - By passing exctype, it attempts to terminate the thread.

    Risks and Considerations
    - Resource Leaks: If the thread holds a lock or other resources, these may not be properly released.
    - Data Corruption: If the thread is manipulating shared data, partial updates may lead to data corruption.
    - Deadlocks: If the thread is killed while holding a lock that other threads are waiting on, it can cause a
        deadlock.
    - Undefined Behavior: The Python runtime does not expect threads to be killed in this manner, which may cause
        undefined behavior.
    """
    exctype: Type[BaseException] = ThreadKilledSystemException
    if not issubclass(exctype, BaseException):
        raise TypeError("Only types derived from BaseException are allowed")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
    logging.debug(f'...killed thread ID: {tid}')
    if res == 0:
        raise ValueError(f"Invalid thread ID: {tid}")
    elif res != 1:
        # If it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def worker_ids(executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor, ActorPoolExecutor]]) -> Set[int]:
    if isinstance(executor, ThreadPoolExecutor):
        return {th.ident for th in executor._threads}
    elif isinstance(executor, ProcessPoolExecutor):
        return {p.pid for p in executor._processes.values()}
    elif isinstance(executor, ActorPoolExecutor):
        return {_actor._process.pid for _actor in executor._actors}
    raise NotImplementedError(f'Cannot get worker ids for executor of type: {executor}')


def stop_executor(
        executor: Optional[Executor],
        force: bool = True,  ## Forcefully terminate, might lead to work being lost.
):
    if executor is not None:
        if isinstance(executor, ThreadPoolExecutor):
            suppress_ThreadKilledSystemException()
            if force:
                executor.shutdown(wait=False)  ## Cancels pending items
                for tid in worker_ids(executor):
                    kill_thread(tid)  ## Note; after calling this, you can still submit
                executor.shutdown(wait=False)  ## Note; after calling this, you cannot submit
            else:
                executor.shutdown(wait=True)
            del executor
        elif isinstance(executor, ProcessPoolExecutor):
            if force:
                for process in executor._processes.values():  # Internal Process objects
                    process.terminate()  # Forcefully terminate the process

                # Wait for the processes to clean up
                for process in executor._processes.values():
                    process.join()
                executor.shutdown(wait=True, cancel_futures=True)
            else:
                executor.shutdown(wait=True, cancel_futures=True)
            del executor
        elif isinstance(executor, ActorPoolExecutor):
            for actor in executor._actors:
                assert isinstance(actor, ActorProxy)
                actor.stop(cancel_futures=force)
                del actor
            del executor


@ray.remote(num_cpus=1)
def _run_parallel_ray_executor(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def _ray_asyncio_start_event_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


class RayPoolExecutor(Executor, Parameters):
    max_workers: Union[int, Literal[inf]]
    iter_wait: float = _RAY_ACCUMULATE_ITER_WAIT
    item_wait: float = _RAY_ACCUMULATE_ITEM_WAIT
    _asyncio_event_loop: Optional = None
    _asyncio_event_loop_thread: Optional = None
    _submission_executor: Optional[ThreadPoolExecutor] = None
    _running_tasks: Dict = {}
    _latest_submit: Optional[int] = None

    def _set_asyncio(self):
        # Create a new loop and a thread running this loop
        if self._asyncio_event_loop is None:
            self._asyncio_event_loop = asyncio.new_event_loop()
            # print(f'Started _asyncio_event_loop')
        if self._asyncio_event_loop_thread is None:
            self._asyncio_event_loop_thread = threading.Thread(
                target=_ray_asyncio_start_event_loop,
                args=(self._asyncio_event_loop,),
            )
            self._asyncio_event_loop_thread.start()
            # print(f'Started _asyncio_event_loop_thread')

    def submit(
            self,
            fn,
            *args,
            scheduling_strategy: str = "SPREAD",
            num_cpus: int = 1,
            num_gpus: int = 0,
            max_retries: int = 0,
            retry_exceptions: Union[List, bool] = True,
            **kwargs,
    ):
        # print(f'Running {fn_str(fn)} using {Parallelize.ray} with num_cpus={num_cpus}, num_gpus={num_gpus}')
        def _submit_task():
            return _run_parallel_ray_executor.options(
                scheduling_strategy=scheduling_strategy,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                max_retries=max_retries,
                retry_exceptions=retry_exceptions,
            ).remote(fn, *args, **kwargs)

        _task_uid = str(time.time_ns())

        if self.max_workers == inf:
            return _submit_task()  ## Submit to Ray directly
        self._set_asyncio()
        ## Create a coroutine (i.e. Future), but do not actually start executing it.
        coroutine = self._ray_run_fn_async(
            submit_task=_submit_task,
            task_uid=_task_uid,
        )

        ## Schedule the coroutine to execute on the event loop (which is running on thread _asyncio_event_loop).
        fut = asyncio.run_coroutine_threadsafe(coroutine, self._asyncio_event_loop)
        # while _task_uid not in self._running_tasks:  ## Ensure task has started scheduling
        #     time.sleep(self.item_wait)
        return fut

    async def _ray_run_fn_async(
            self,
            submit_task: Callable,
            task_uid: str,
    ):
        # self._running_tasks[task_uid] = None
        while len(self._running_tasks) >= self.max_workers:
            for _task_uid in sorted(self._running_tasks.keys()):
                if is_done(self._running_tasks[_task_uid]):
                    self._running_tasks.pop(_task_uid, None)
                    # print(f'Popped {_task_uid}')
                    if len(self._running_tasks) < self.max_workers:
                        break
                time.sleep(self.item_wait)
            if len(self._running_tasks) < self.max_workers:
                break
            time.sleep(self.iter_wait)
        fut = submit_task()
        self._running_tasks[task_uid] = fut
        # print(f'Started {task_uid}. Num running: {len(self._running_tasks)}')

        # ## Cleanup any completed tasks:
        # for k in list(self._running_tasks.keys()):
        #     if is_done(self._running_tasks[k]):
        #         self._running_tasks.pop(k, None)
        #     time.sleep(self.item_wait)
        return fut


def run_parallel_ray(
        fn,
        *args,
        scheduling_strategy: str = "SPREAD",
        num_cpus: int = 1,
        num_gpus: int = 0,
        max_retries: int = 0,
        retry_exceptions: Union[List, bool] = True,
        executor: Optional[RayPoolExecutor] = None,
        **kwargs,
):
    # print(f'Running {fn_str(fn)} using {Parallelize.ray} with num_cpus={num_cpus}, num_gpus={num_gpus}')
    if executor is not None:
        assert isinstance(executor, RayPoolExecutor)
        return executor.submit(
            fn,
            *args,
            scheduling_strategy=scheduling_strategy,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            max_retries=max_retries,
            retry_exceptions=retry_exceptions,
            **kwargs,
        )
    else:
        return _run_parallel_ray_executor.options(
            scheduling_strategy=scheduling_strategy,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            max_retries=max_retries,
            retry_exceptions=retry_exceptions,
        ).remote(fn, *args, **kwargs)


class ExecutorConfig(Parameters):
    class Config(Parameters.Config):
        extra = Extra.ignore

    parallelize: Parallelize
    max_workers: Optional[int] = None
    max_calls_per_second: float = float('inf')

    @root_validator(pre=True)
    def _set_params(cls, params: Dict) -> Dict:
        set_param_from_alias(params, param='max_workers', alias=['num_workers'], default=None)
        return params


def dispatch(
        fn: Callable,
        *args,
        parallelize: Parallelize,
        forward_parallelize: bool = False,
        delay: float = 0.0,
        executor: Optional[Executor] = None,
        **kwargs
) -> Any:
    parallelize: Parallelize = Parallelize.from_str(parallelize)
    if forward_parallelize:
        kwargs['parallelize'] = parallelize
    time.sleep(delay)
    if parallelize is Parallelize.sync:
        return fn(*args, **kwargs)
    elif parallelize is Parallelize.asyncio:
        return run_asyncio(fn, *args, **kwargs)
    elif parallelize is Parallelize.threads:
        return run_concurrent(fn, *args, executor=executor, **kwargs)
    elif parallelize is Parallelize.processes:
        return run_parallel(fn, *args, executor=executor, **kwargs)
    elif parallelize is Parallelize.ray:
        return run_parallel_ray(fn, *args, executor=executor, **kwargs)
    raise NotImplementedError(f'Unsupported parallelization: {parallelize}')


def dispatch_executor(
        *,
        config: Optional[Union[ExecutorConfig, Dict]] = None,
        **kwargs
) -> Optional[Executor]:
    if config is None:
        config: Dict = dict()
    else:
        assert isinstance(config, ExecutorConfig)
        config: Dict = config.dict(exclude=True)
    config: ExecutorConfig = ExecutorConfig(**{**config, **kwargs})
    if config.max_workers is None:
        ## Uses the default executor for threads/processes/ray.
        return None
    if config.parallelize is Parallelize.sync:
        return None
    elif config.parallelize is Parallelize.threads:
        return RestrictedConcurrencyThreadPoolExecutor(
            max_workers=config.max_workers,
            max_calls_per_second=config.max_calls_per_second,
        )
    elif config.parallelize is Parallelize.processes:
        return ActorPoolExecutor(
            max_workers=config.max_workers,
        )
    elif config.parallelize is Parallelize.ray:
        return RayPoolExecutor(
            max_workers=config.max_workers,
        )
    else:
        raise NotImplementedError(f'Unsupported: you cannot create an executor with {parallelize} parallelization.')


def dispatch_apply(
        struct: Union[List, Tuple, np.ndarray, PandasSeries, Set, frozenset, Dict],
        *args,
        fn: Callable,
        parallelize: Parallelize,
        forward_parallelize: bool = False,
        item_wait: Optional[float] = None,
        iter_wait: Optional[float] = None,
        iter: bool = False,
        **kwargs
) -> Any:
    parallelize: Parallelize = Parallelize.from_str(parallelize)
    item_wait: float = get_default(
        item_wait,
        {
            Parallelize.ray: _RAY_ACCUMULATE_ITEM_WAIT,
            Parallelize.processes: _LOCAL_ACCUMULATE_ITEM_WAIT,
            Parallelize.threads: _LOCAL_ACCUMULATE_ITEM_WAIT,
            Parallelize.asyncio: 0.0,
            Parallelize.sync: 0.0,
        }[parallelize]
    )
    iter_wait: float = get_default(
        iter_wait,
        {
            Parallelize.ray: _RAY_ACCUMULATE_ITER_WAIT,
            Parallelize.processes: _LOCAL_ACCUMULATE_ITER_WAIT,
            Parallelize.threads: _LOCAL_ACCUMULATE_ITER_WAIT,
            Parallelize.asyncio: 0.0,
            Parallelize.sync: 0.0,
        }[parallelize]
    )
    if forward_parallelize:
        kwargs['parallelize'] = parallelize
    executor: Optional = dispatch_executor(
        parallelize=parallelize,
        **kwargs,
    )
    try:
        set_param_from_alias(kwargs, param='progress_bar', alias=['progress', 'pbar'], default=True)
        progress_bar: Union[ProgressBar, Dict, bool] = kwargs.pop('progress_bar', False)
        submit_pbar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=len(struct),
            desc='Submitting',
            prefer_kwargs=False,
            unit='item',
        )
        collect_pbar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=len(struct),
            desc='Collecting',
            prefer_kwargs=False,
            unit='item',
        )
        if is_list_or_set_like(struct):
            futs = []
            for v in struct:
                def submit_task(item, **dispatch_kwargs):
                    return fn(item, **dispatch_kwargs)

                futs.append(
                    dispatch(
                        fn=submit_task,
                        item=v,
                        parallelize=parallelize,
                        executor=executor,
                        delay=item_wait,
                        **filter_kwargs(fn, **kwargs),
                    )
                )
                submit_pbar.update(1)
        elif is_dict_like(struct):
            futs = {}
            for k, v in struct.items():
                def submit_task(item, **dispatch_kwargs):
                    return fn(item, **dispatch_kwargs)

                futs[k] = dispatch(
                    fn=submit_task,
                    key=k,
                    item=v,
                    parallelize=parallelize,
                    executor=executor,
                    delay=item_wait,
                    **filter_kwargs(fn, **kwargs),
                )
                submit_pbar.update(1)
        else:
            raise NotImplementedError(f'Unsupported type: {type_str(struct)}')
        submit_pbar.success()
        if iter:
            return accumulate_iter(
                futs,
                item_wait=item_wait,
                iter_wait=iter_wait,
                progress_bar=collect_pbar,
                **kwargs
            )
        else:
            return accumulate(
                futs,
                item_wait=item_wait,
                iter_wait=iter_wait,
                progress_bar=collect_pbar,
                **kwargs
            )
    finally:
        stop_executor(executor)


def get_result(
        x,
        *,
        wait: float = 1.0,  ## 1000 ms
) -> Optional[Any]:
    if isinstance(x, Future):
        return get_result(x.result(), wait=wait)
    if isinstance(x, ray.ObjectRef):
        while True:
            try:
                return ray.get(x, timeout=wait)
            except GetTimeoutError as e:
                pass
    return x


def get_status(x) -> Status:
    if is_running(x):
        return Status.RUNNING
    if not is_done(x):  ## Not running and not done, thus pending i.e. scheduled
        return Status.PENDING
    ## The future is done:
    if is_successful(x):
        return Status.SUCCEEDED
    if is_failed(x):
        return Status.FAILED


def is_future(x) -> bool:
    return isinstance(x, Future) or isinstance(x, ray.ObjectRef)


def is_running(x) -> bool:
    if isinstance(x, Future):
        return x.running()  ## It might be scheduled but not running.
    if isinstance(x, ray.ObjectRef):
        return not is_done(x)
    return False


def is_done(x) -> bool:
    if isinstance(x, Future):
        return x.done()
    if isinstance(x, ray.ObjectRef):
        ## Ref: docs.ray.io/en/latest/ray-core/tasks.html#waiting-for-partial-results
        done, not_done = ray.wait([x], timeout=0)  ## Immediately check if done.
        return len(done) > 0 and len(not_done) == 0
    return True


def is_successful(x, *, pending_returns_false: bool = False) -> Optional[bool]:
    if not is_done(x):
        if pending_returns_false:
            return False
        else:
            return None
    try:
        get_result(x)
        return True
    except Exception as e:
        return False


def is_failed(x, *, pending_returns_false: bool = False) -> Optional[bool]:
    if not is_done(x):
        if pending_returns_false:
            return False
        else:
            return None
    try:
        get_result(x)
        return False
    except Exception as e:
        return True


def accumulate(
        futures: Union[Tuple, List, Set, Dict, Any],
        *,
        check_done: bool = True,
        item_wait: Optional[float] = None,
        iter_wait: Optional[float] = None,
        succeeded_only: bool = False,
        **kwargs,
) -> Union[List, Tuple, Set, Dict, Any]:
    """Join operation on a single future or a collection of futures."""
    set_param_from_alias(kwargs, param='progress_bar', alias=['progress', 'pbar'])
    progress_bar: Union[ProgressBar, Dict, bool] = kwargs.pop('progress_bar', False)
    if isinstance(futures, (list, set, tuple)) and len(futures) > 0:
        if isinstance(first_item(futures), Future):
            item_wait: float = get_default(item_wait, _LOCAL_ACCUMULATE_ITEM_WAIT)
            iter_wait: float = get_default(iter_wait, _LOCAL_ACCUMULATE_ITER_WAIT)
        else:
            item_wait: float = get_default(item_wait, _RAY_ACCUMULATE_ITEM_WAIT)
            iter_wait: float = get_default(iter_wait, _RAY_ACCUMULATE_ITER_WAIT)
        if succeeded_only:
            return type(futures)([
                accumulate(fut, progress_bar=False, check_done=check_done, succeeded_only=succeeded_only)
                for fut in futures
                if is_successful(fut)
            ])
        completed_futures: List[bool] = [
            is_done(fut) if check_done else False
            for fut in futures
        ]
        accumulated_futures: List = [
            accumulate(fut, progress_bar=False, check_done=check_done) if future_is_complete else fut
            for future_is_complete, fut in zip(completed_futures, futures)
        ]
        pbar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=len(futures),
            initial=sum(completed_futures),
            desc='Collecting',
            prefer_kwargs=False,
            unit='item',
        )
        while not all(completed_futures):
            for i, fut in enumerate(accumulated_futures):
                if completed_futures[i] is False:
                    completed_futures[i] = is_done(fut)
                    if completed_futures[i] is True:
                        accumulated_futures[i] = accumulate(fut, progress_bar=False, check_done=check_done)
                        pbar.update(1)
                    time.sleep(item_wait)
            time.sleep(iter_wait)
        pbar.success('Done', close=False)
        return type(futures)(accumulated_futures)  ## Convert
    elif isinstance(futures, dict) and len(futures) > 0:
        if isinstance(first_item(futures)[0], Future) or isinstance(first_item(futures)[1], Future):
            item_wait: float = get_default(item_wait, _LOCAL_ACCUMULATE_ITEM_WAIT)
            iter_wait: float = get_default(iter_wait, _LOCAL_ACCUMULATE_ITER_WAIT)
        else:
            item_wait: float = get_default(item_wait, _RAY_ACCUMULATE_ITEM_WAIT)
            iter_wait: float = get_default(iter_wait, _RAY_ACCUMULATE_ITER_WAIT)
        futures: List[Tuple] = list(futures.items())
        if succeeded_only:
            return dict([
                (
                    accumulate(fut_k, progress_bar=False, check_done=check_done, succeeded_only=succeeded_only),
                    accumulate(fut_v, progress_bar=False, check_done=check_done, succeeded_only=succeeded_only),
                )
                for fut_k, fut_v in futures
                if (is_successful(fut_k) and is_successful(fut_v))
            ])
        completed_futures: List[bool] = [
            (is_done(fut_k) and is_done(fut_v)) if check_done else False
            for fut_k, fut_v in futures
        ]
        accumulated_futures: List[Tuple] = [
            (
                accumulate(fut_k, progress_bar=False, check_done=check_done),
                accumulate(fut_v, progress_bar=False, check_done=check_done)
            ) if future_is_complete else (fut_k, fut_v)
            for future_is_complete, (fut_k, fut_v) in zip(completed_futures, futures)
        ]
        pbar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=len(futures),
            initial=sum(completed_futures),
            desc='Collecting',
            prefer_kwargs=False,
            unit='item',
        )
        while not all(completed_futures):
            for i, (fut_k, fut_v) in enumerate(accumulated_futures):
                if completed_futures[i] is False:
                    completed_futures[i] = is_done(fut_k) and is_done(fut_v)
                    if completed_futures[i] is True:
                        accumulated_futures[i] = (
                            accumulate(fut_k, progress_bar=False, check_done=check_done),
                            accumulate(fut_v, progress_bar=False, check_done=check_done)
                        )
                        pbar.update(1)
                    time.sleep(item_wait)
            time.sleep(iter_wait)
        pbar.success('Done', close=False)
        return dict(accumulated_futures)
    else:
        return get_result(futures)


def accumulate_iter(
        futures: Union[Tuple, List, Set, Dict],
        *,
        item_wait: Optional[float] = None,
        iter_wait: Optional[float] = None,
        allow_partial_results: bool = False,
        **kwargs,
):
    """
    Here we iteratively accumulate and yield completed futures as they have completed.
    This might return them out-of-order.
    """
    set_param_from_alias(kwargs, param='progress_bar', alias=['progress', 'pbar'])
    progress_bar: Union[ProgressBar, Dict, bool] = kwargs.pop('progress_bar', False)
    pbar: ProgressBar = ProgressBar.of(
        progress_bar,
        total=len(futures),
        desc='Iterating',
        prefer_kwargs=False,
        unit='item',
    )
    if isinstance(futures, (list, set, tuple)) and len(futures) > 0:
        if isinstance(first_item(futures), Future):
            item_wait: float = get_default(item_wait, _LOCAL_ACCUMULATE_ITEM_WAIT)
            iter_wait: float = get_default(iter_wait, _LOCAL_ACCUMULATE_ITER_WAIT)
        else:
            item_wait: float = get_default(item_wait, _RAY_ACCUMULATE_ITEM_WAIT)
            iter_wait: float = get_default(iter_wait, _RAY_ACCUMULATE_ITER_WAIT)
        ## Copy as list:
        futures: List = [
            fut
            for fut in futures
        ]
        yielded_futures: List[bool] = [
            False
            for fut in futures
        ]
        while not all(yielded_futures):
            for i, fut in enumerate(futures):
                if yielded_futures[i] is False and is_done(fut):
                    try:
                        yielded_futures[i] = True
                        pbar.update(1)
                        yield get_result(fut)
                        time.sleep(item_wait)
                    except Exception as e:
                        if not allow_partial_results:
                            pbar.failed(close=False)
                            raise e
                        yield fut
            time.sleep(iter_wait)
        pbar.success('Done', close=False)
    elif isinstance(futures, dict) and len(futures) > 0:
        ## Copy as list:
        futures: List[Tuple[Any, Any]] = [
            (fut_k, fut_v)
            for fut_k, fut_v in futures.items()
        ]
        if isinstance(first_item(futures)[0], Future) or isinstance(first_item(futures)[1], Future):
            item_wait: float = get_default(item_wait, _LOCAL_ACCUMULATE_ITEM_WAIT)
            iter_wait: float = get_default(iter_wait, _LOCAL_ACCUMULATE_ITER_WAIT)
        else:
            item_wait: float = get_default(item_wait, _RAY_ACCUMULATE_ITEM_WAIT)
            iter_wait: float = get_default(iter_wait, _RAY_ACCUMULATE_ITER_WAIT)
        yielded_futures: List[bool] = [
            False
            for fut_k, fut_v in futures
        ]
        while not all(yielded_futures):
            for i, (fut_k, fut_v) in enumerate(futures):
                if yielded_futures[i] is False and (is_done(fut_k) and is_done(fut_v)):
                    try:
                        yielded_futures[i] = True
                        pbar.update(1)
                        yield (get_result(fut_k), get_result(fut_v))
                        pbar.update(1)
                        time.sleep(item_wait)
                    except Exception as e:
                        if not allow_partial_results:
                            pbar.failed(close=False)
                            raise e
                        yield (fut_k, fut_v)
            time.sleep(iter_wait)
        pbar.success('Done', close=False)
    else:
        if not isinstance(futures, (list, set, tuple, dict)):
            raise NotImplementedError(f'Cannot iteratively collect from object of type: {type_str(futures)}.')


def wait_if_future(x):
    if isinstance(x, Future):
        wait_future([x])
    elif isinstance(x, ray.ObjectRef):
        ray.wait([x])


def wait(
        futures: Union[Tuple, List, Set, Dict, Any],
        *,
        check_done: bool = True,
        item_wait: float = 0.1,  ## 100 ms
        iter_wait: float = 1.0,  ## 1000 ms
        **kwargs,
) -> NoReturn:
    """Join operation on a single future or a collection of futures."""
    set_param_from_alias(kwargs, param='progress_bar', alias=['progress', 'pbar'])
    progress_bar: Union[ProgressBar, Dict, bool] = kwargs.pop('progress_bar', False)

    if isinstance(futures, (list, tuple, set, np.ndarray)):
        futures: List[Any] = list(futures)
        completed_futures: List[bool] = [
            is_done(fut) if check_done else False
            for fut in futures
        ]
        pbar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=len(futures),
            initial=sum(completed_futures),
            desc='Waiting',
            prefer_kwargs=False,
            unit='item',
        )
        while not all(completed_futures):
            for i, fut in enumerate(futures):
                if completed_futures[i] is False:
                    completed_futures[i] = is_done(fut)
                    if completed_futures[i] is True:
                        pbar.update(1)
                    time.sleep(item_wait)
            time.sleep(iter_wait)
        pbar.success('Done', close=False)
    elif isinstance(futures, dict):
        futures: List[Tuple[Any, Any]] = list(futures.items())
        completed_futures: List[bool] = [
            (is_done(fut_k) and is_done(fut_v)) if check_done else False
            for fut_k, fut_v in futures
        ]
        pbar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=len(futures),
            initial=sum(completed_futures),
            desc='Waiting',
            prefer_kwargs=False,
            unit='item',
        )
        while not all(completed_futures):
            for i, (fut_k, fut_v) in enumerate(futures):
                if completed_futures[i] is False:
                    completed_futures[i] = is_done(fut_k) and is_done(fut_v)
                    if completed_futures[i] is True:
                        pbar.update(1)
                    time.sleep(item_wait)
            time.sleep(iter_wait)
        pbar.success('Done', close=False)
    else:
        wait_if_future(futures)


def retry(
        fn,
        *args,
        retries: int = 5,
        wait: float = 10.0,
        jitter: float = 0.5,
        silent: bool = True,
        return_num_failures: bool = False,
        **kwargs
) -> Union[Any, Tuple[Any, int]]:
    """
    Retries a function call a certain number of times, waiting between calls (with a jitter in the wait period).
    :param fn: the function to call.
    :param retries: max number of times to try. If set to 0, will not retry.
    :param wait: average wait period between retries
    :param jitter: limit of jitter (+-). E.g. jitter=0.1 means we will wait for a random time period in the range
        (0.9 * wait, 1.1 * wait) seconds.
    :param silent: whether to print an error message on each retry.
    :param kwargs: keyword arguments forwarded to the function.
    :param return_num_failures: whether to return the number of times failed.
    :return: the function's return value if any call succeeds. If return_num_failures is set, returns this as the second result.
    :raise: RuntimeError if all `retries` calls fail.
    """
    assert isinstance(retries, int) and 0 <= retries
    assert isinstance(wait, (int, float)) and 0 <= wait
    assert isinstance(jitter, (int, float)) and 0 <= jitter <= 1
    wait: float = float(wait)
    latest_exception = None
    num_failures: int = 0
    for retry_num in range(retries + 1):
        try:
            out = fn(*args, **kwargs)
            if return_num_failures:
                return out, num_failures
            else:
                return out
        except Exception as e:
            num_failures += 1
            latest_exception = format_exception_msg(e)
            if not silent:
                print(f'Function call failed with the following exception (attempts: {retry_num + 1}):\n{latest_exception}')
                if retry_num < (retries - 1):
                    print(f'Retrying {retries - (retry_num + 1)} more time(s)...\n')
            time.sleep(np.random.uniform(wait - wait * jitter, wait + wait * jitter))
    raise RuntimeError(f'Function call failed {retries + 1} time(s).\nLatest exception:\n{latest_exception}\n')


def daemon(wait: float, exit_on_error: bool = False, sentinel: Optional[List] = None, **kwargs):
    """
    A decorator which runs a function as a daemon process in a background thread.

    You do not need to invoke this function directly: simply decorating the daemon function will start running it
    in the background.

    Example using class method: your daemon should be marked with @staticmethod. Example:
        class Printer:
            DATA_LIST = []
            @staticmethod
            @daemon(wait=3, mylist=DATA_LIST)
            def printer_daemon(mylist):
                if len(mylist) > 0:
                    print(f'Contents of list: {mylist}', flush=True)

    Example using sentinel:
        run_sentinel = [True]
        @daemon(wait=1, sentinel=run_sentinel)
        def run():
            print('Running', flush=True)
        time.sleep(3)  ## Prints "Running" 3 times.
        run_sentinel.pop()  ## Stops "Running" from printing any more.

    :param wait: the wait time in seconds between invocations to the @daemon decorated function.
    :param exit_on_error: whether to stop the daemon if an error is raised.
    :sentinel: can be used to stop the executor. When not passed, the daemon runs forever. When passed, `sentinel` must
        be a list with exactly one element (it can be anything). To stop the daemon, run "sentinel.pop()". It is
        important to pass a list (not a tuple), since lists are mutable, and thus the same exact object is used by
        both the executor and by the caller.
    :param kwargs: list of arguments passed to the decorator, which are forwarded to the decorated function as kwargs.
        These values will never change for the life of the daemon. However, if you pass references to mutables such as
        lists, dicts, objects etc to the decorator and use them in the daemon function, you can run certain tasks at a
        regular cadence on fresh data.
    :return: None
    """

    ## Refs on how decorators work:
    ## 1. https://www.datacamp.com/community/tutorials/decorators-python
    def decorator(function):
        ## Each decorated function gets its own executor. These are defined at the function-level, so
        ## if you write two decorated functions `def say_hi` and `def say_bye`, they each gets a separate
        ## executor. The executor for `say_hi` will call `say_hi` repeatedly, and the executor for `say_bye` will call
        ## `say_bye` repeatedly; they will not interact.
        executor = RestrictedConcurrencyThreadPoolExecutor(max_workers=1)

        def run_function_forever(sentinel):
            while sentinel is None or len(sentinel) > 0:
                start = time.perf_counter()
                try:
                    function(**kwargs)
                except Exception as e:
                    logging.debug(traceback.format_exc())
                    if exit_on_error:
                        raise e
                end = time.perf_counter()
                time_to_wait: float = max(0.0, wait - (end - start))
                time.sleep(time_to_wait)
            del executor  ## Cleans up the daemon after it finishes running.

        if sentinel is not None:
            if not isinstance(sentinel, list) or len(sentinel) != 1:
                raise ValueError(f'When passing `sentinel`, it must be a list with exactly one item.')
        completed: Future = executor.submit(run_function_forever, sentinel=sentinel)

        ## The wrapper here should do nothing, since you cannot call the daemon explicitly.
        def wrapper(*args, **kwargs):
            raise RuntimeError('Cannot call daemon function explicitly')

        return wrapper

    return decorator


## Dict of daemon ids to their sentinels
_DAEMONS: Dict[str, List[bool]] = {}


def start_daemon(
        fn,
        wait: float,
        daemon_id: Optional[str] = None,
        daemons: Dict[str, List[bool]] = _DAEMONS,
        **kwargs,
) -> str:
    assert isinstance(daemons, dict)
    assert isinstance(wait, (int, float)) and wait >= 0.0
    if daemon_id is None:
        dt: datetime = datetime.now()
        dt: datetime = dt.replace(tzinfo=dt.astimezone().tzinfo)
        if dt.tzinfo is not None:
            daemon_id: str = dt.strftime('%Y-%m-%d %H:%M:%S.%f UTC%z').strip()
        else:
            daemon_id: str = dt.strftime('%Y-%m-%d %H:%M:%S.%f').strip()
    assert isinstance(daemon_id, str) and len(daemon_id) > 0
    assert daemon_id not in daemons, f'Daemon with id "{daemon_id}" already exists.'

    daemon_sentinel: List[bool] = [True]

    @daemon(wait=wait, sentinel=daemon_sentinel)
    def run():
        fn(**kwargs)

    daemons[daemon_id] = daemon_sentinel
    return daemon_id


def stop_daemon(daemon_id: str, daemons: Dict[str, List[bool]] = _DAEMONS) -> bool:
    assert isinstance(daemons, dict)
    assert isinstance(daemon_id, str) and len(daemon_id) > 0
    daemon_sentinel: List[bool] = daemons.pop(daemon_id, [False])
    assert len(daemon_sentinel) == 1
    return daemon_sentinel.pop()


## Ref: https://docs.ray.io/en/latest/data/dask-on-ray.html#callbacks
class RayDaskPersistWaitCallback(RayDaskCallback):
    ## Callback to wait for computation to complete when .persist() is called with block=True
    def _ray_postsubmit_all(self, object_refs, dsk):
        wait(object_refs)
