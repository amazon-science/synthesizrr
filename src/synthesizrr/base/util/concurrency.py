"""A collection of concurrency utilities to augment the Python language:"""
from typing import *
import time, traceback, random, sys, math, gc
from datetime import datetime
from math import inf
import numpy as np
import asyncio, ctypes
from threading import Semaphore, Thread
import multiprocessing as mp
from concurrent.futures._base import Future
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait as wait_future
from concurrent.futures.thread import BrokenThreadPool
from concurrent.futures.process import BrokenProcessPool
import ray
from ray.exceptions import GetTimeoutError
from ray.util.dask import RayDaskCallback
from pydantic import validate_arguments, conint, confloat
from synthesizrr.base.util.language import ProgressBar, set_param_from_alias, type_str, get_default, first_item, if_else
from synthesizrr.base.constants.DataProcessingConstants import Parallelize, FailureAction, Status, COMPLETED_STATUSES

from functools import partial
## Jupyter-compatible asyncio usage:
import asyncio
import threading
import time, inspect
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait as wait_future


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


def concurrent(max_active_threads: int = 10, max_calls_per_second: float = inf):
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

    :param max_active_threads: the max number of threads which can be running the function at one time. This is thus
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
        ## `max_active_threads` will determine how many function calls actually run concurrently, e.g. if say_hi has
        ## max_active_threads=5, then the 30 calls will run 5 at a time (this is enforced by the semaphore).
        executor = ThreadPoolExecutor(max_workers=max_active_threads)
        semaphore = Semaphore(max_active_threads)

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


_GLOBAL_THREAD_POOL_EXECUTOR: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=16)


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
        # print(f'Running {fn_str(fn)} using {Parallelize.threads} with max_workers={executor._max_workers}')
        return executor.submit(fn, *args, **kwargs)  ## return a future
    except BrokenThreadPool as e:
        if executor is _GLOBAL_THREAD_POOL_EXECUTOR:
            executor = ThreadPoolExecutor(max_workers=_GLOBAL_THREAD_POOL_EXECUTOR._max_workers)
            del _GLOBAL_THREAD_POOL_EXECUTOR
            _GLOBAL_THREAD_POOL_EXECUTOR = executor
            return executor.submit(fn, *args, **kwargs)  ## return a future
        raise e


_GLOBAL_PROCESS_POOL_EXECUTOR: ProcessPoolExecutor = ProcessPoolExecutor(
    max_workers=max(1, min(32, mp.cpu_count() - 1))
)


def run_parallel(
        fn,
        *args,
        executor: Optional[ProcessPoolExecutor] = None,
        **kwargs,
):
    global _GLOBAL_PROCESS_POOL_EXECUTOR
    if executor is None:
        executor: ProcessPoolExecutor = _GLOBAL_PROCESS_POOL_EXECUTOR
    try:
        # print(f'Running {fn_str(fn)} using {Parallelize.threads} with max_workers={executor._max_workers}')
        return executor.submit(fn, *args, **kwargs)  ## return a future
    except BrokenProcessPool as e:
        if executor is _GLOBAL_PROCESS_POOL_EXECUTOR:
            executor = ProcessPoolExecutor(max_workers=_GLOBAL_PROCESS_POOL_EXECUTOR._max_workers)
            del _GLOBAL_PROCESS_POOL_EXECUTOR
            _GLOBAL_PROCESS_POOL_EXECUTOR = executor
            return executor.submit(fn, *args, **kwargs)  ## return a future
        raise e


def kill_thread(tid: int, exctype: Type[BaseException]):
    """
    Dirty hack to *actually* stop a thread: raises an exception in threads with this thread id.
    How it works:
    - kill_thread function uses ctypes.pythonapi.PyThreadState_SetAsyncExc to raise an exception in a thread.
    - By passing SystemExit, it attempts to terminate the thread.

    Risks and Considerations
    - Resource Leaks: If the thread holds a lock or other resources, these may not be properly released.
    - Data Corruption: If the thread is manipulating shared data, partial updates may lead to data corruption.
    - Deadlocks: If the thread is killed while holding a lock that other threads are waiting on, it can cause a
        deadlock.
    - Undefined Behavior: The Python runtime does not expect threads to be killed in this manner, which may cause
        undefined behavior.
    """
    if not issubclass(exctype, BaseException):
        raise TypeError("Only types derived from BaseException are allowed")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
    print(f'...killed thread ID: {tid}')
    if res == 0:
        raise ValueError(f"Invalid thread ID: {tid}")
    elif res != 1:
        # If it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def worker_ids(executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]]) -> Set[int]:
    if isinstance(executor, ThreadPoolExecutor):
        return {th.ident for th in executor._threads}
    elif isinstance(executor, ProcessPoolExecutor):
        return {p.pid for p in executor._processes.values()}
    raise NotImplementedError(f'Cannot get worker ids for executor of type: {executor}')


def stop_executor(
        executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]],
        force: bool = True,  ## Forcefully terminate, might lead to work being lost.
):
    if executor is not None:
        if isinstance(executor, ThreadPoolExecutor):
            if force:
                executor.shutdown(wait=False)  ## Cancels pending items
                for tid in worker_ids(executor):
                    kill_thread(tid, SystemExit)  ## Note; after calling this, you can still submit
                executor.shutdown(wait=False)  ## Note; after calling this, you cannot submit
            else:
                executor.shutdown(wait=True)
            del executor
        elif isinstance(executor, ProcessPoolExecutor):
            executor.shutdown(wait=True)
            del executor


@ray.remote(num_cpus=1)
def __run_parallel_ray_executor(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def run_parallel_ray(
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
    return __run_parallel_ray_executor.options(
        scheduling_strategy=scheduling_strategy,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        max_retries=max_retries,
        retry_exceptions=retry_exceptions,
    ).remote(fn, *args, **kwargs)


def dispatch(
        fn: Callable,
        *args,
        parallelize: Parallelize,
        forward_parallelize: bool = False,
        delay: float = 0.0,
        executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None,
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
        return run_parallel_ray(fn, *args, **kwargs)
    raise NotImplementedError(f'Unsupported parallelization: {parallelize}')


def dispatch_executor(
        parallelize: Parallelize,
        **kwargs
) -> Optional[Union[ProcessPoolExecutor, ThreadPoolExecutor]]:
    parallelize: Parallelize = Parallelize.from_str(parallelize)
    set_param_from_alias(kwargs, param='max_workers', alias=['num_workers'], default=None)
    max_workers: Optional[int] = kwargs.pop('max_workers', None)
    if max_workers is None:
        ## Uses the default executor for threads/processes.
        return None
    if parallelize is Parallelize.threads:
        return ThreadPoolExecutor(max_workers=max_workers)
    elif parallelize is Parallelize.processes:
        return ProcessPoolExecutor(max_workers=max_workers)
    else:
        return None


def get_result(
        x,
        *,
        wait: float = 1.0,  ## 1000 ms
) -> Optional[Any]:
    if isinstance(x, Future):
        return x.result()
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


_RAY_ACCUMULATE_ITEM_WAIT: float = 1000e-3  ## 1000ms
_LOCAL_ACCUMULATE_ITEM_WAIT: float = 10e-3  ## 10ms

_RAY_ACCUMULATE_ITER_WAIT: float = 10e0  ## 10 sec
_LOCAL_ACCUMULATE_ITER_WAIT: float = 100e-3  ## 100ms


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


@validate_arguments
def retry(
        fn,
        *args,
        retries: conint(ge=0) = 5,
        wait: confloat(ge=0.0) = 10.0,
        jitter: confloat(gt=0.0) = 0.5,
        silent: bool = True,
        **kwargs
):
    """
    Retries a function call a certain number of times, waiting between calls (with a jitter in the wait period).
    :param fn: the function to call.
    :param retries: max number of times to try. If set to 0, will not retry.
    :param wait: average wait period between retries
    :param jitter: limit of jitter (+-). E.g. jitter=0.1 means we will wait for a random time period in the range
        (0.9 * wait, 1.1 * wait) seconds.
    :param silent: whether to print an error message on each retry.
    :param kwargs: keyword arguments forwarded to the function.
    :return: the function's return value if any call succeeds.
    :raise: RuntimeError if all `retries` calls fail.
    """
    wait: float = float(wait)
    latest_exception = None
    for retry_num in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            latest_exception = traceback.format_exc()
            if not silent:
                print(f'Function call failed with the following exception:\n{latest_exception}')
                if retry_num < (retries - 1):
                    print(f'Retrying {retries - (retry_num + 1)} more times...\n')
            time.sleep(np.random.uniform(wait - wait * jitter, wait + wait * jitter))
    raise RuntimeError(f'Function call failed {retries} times.\nLatest exception:\n{latest_exception}\n')


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
        executor = ThreadPoolExecutor(max_workers=1)

        def run_function_forever(sentinel):
            while sentinel is None or len(sentinel) > 0:
                start = time.perf_counter()
                try:
                    function(**kwargs)
                except Exception as e:
                    print(traceback.format_exc())
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
