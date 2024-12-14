import logging
from typing import *
from abc import abstractmethod, ABC
import math, time, warnings, json, gzip, base64
import numpy as np, pandas as pd, multiprocessing as mp, threading as th
from concurrent.futures._base import Future
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pandas.core.frame import DataFrame as PandasDataFrame
import dask.dataframe as dd
from dask.dataframe.core import DataFrame as DaskDataFrame
from synthergent.base.util import as_list, resolve_sample_size, SampleSizeType, Registry, StringUtil, get_default, \
    classproperty, accumulate, dispatch, MutableParameters, safe_validate_arguments, is_done, optional_dependency, \
    multiple_are_not_none, all_are_not_none, is_list_of_dict_like, Parameters
from synthergent.base.constants import DataLayout, SDF_DATA_LAYOUT_PRIORITY, LAZY_SDF_DATA_LAYOUTS, Parallelize, \
    CompressionEngine, Alias
from synthergent.base.data.sdf.ScalableSeries import ScalableSeries, ScalableSeriesOrRaw
from pydantic import conint, constr, root_validator
from pydantic.typing import Literal
from collections import deque

ScalableDataFrame = "ScalableDataFrame"
CompressedScalableDataFrame = "CompressedScalableDataFrame"
ScalableDataFrameRawType = Union[Dict, List[Dict], np.recarray, PandasDataFrame, DaskDataFrame]
ScalableDataFrameOrRaw = Union[ScalableDataFrame, ScalableDataFrameRawType]
ScalableOrRaw = Union[ScalableSeriesOrRaw, ScalableDataFrameOrRaw]
RAW_DATA_MEMBER = '_data'

is_scalable: Callable = lambda data: isinstance(data, (ScalableSeries, ScalableDataFrame))


class DataFrameShardingError(Exception):
    """A custom exception used to report errors in sharding."""
    pass


class ScalableDataFrameDisplay(MutableParameters):
    max_rows: int = 10
    min_rows: int = 10


class ScalableDataFrame(Registry, ABC):
    """
    Class to interact with mutable "DataFrames" of various underlying data layouts.
    """
    layout: ClassVar[DataLayout]
    ## Callable typing: stackoverflow.com/a/39624147/4900327
    layout_validator: ClassVar[Callable[[Any, bool], bool]]
    ScalableSeriesClass: ClassVar[Type[ScalableSeries]]
    chunk_prefix: ClassVar[str] = 'part'
    display: ClassVar[ScalableDataFrameDisplay] = ScalableDataFrameDisplay()

    def __init__(self, data: Optional[ScalableDataFrameOrRaw] = None, name: Optional[str] = None, **kwargs):
        self._data = data
        self._name: Optional[str] = name

    def __del__(self):
        _data = self._data
        self._data = None
        del _data

    @property
    def name(self) -> Optional[str]:
        return self._name

    @classproperty
    def _constructor(cls) -> Type[ScalableDataFrame]:
        return cls

    ## Required to serialize ScalableDataFrames using Ray.
    ## Ref: https://docs.ray.io/en/latest/ray-core/objects/serialization.html#customized-serialization
    def __reduce__(self):
        deserializer = self.__class__
        serialized_data = (self._data,)
        return deserializer, serialized_data

    @classmethod
    def _registry_keys(cls) -> Optional[Union[List[Any], Any]]:
        return cls.layout

    def to_frame(self, **kwargs) -> ScalableDataFrame:
        return self

    @classmethod
    def of(
            cls,
            data: ScalableOrRaw,
            layout: Optional[DataLayout] = None,
            **kwargs,
    ) -> ScalableDataFrame:
        """
        Factory to create ScalableDataFrames.
        :param data: ScalableDataFrame or "raw" data to be used as the underlying object.
        :param layout: target layout of the returned ScalableDataFrame.
        :return: ScalableDataFrame instance.
        """
        if data is None:
            raise ValueError(f'Input data cannot be None.')
        if not is_scalable(data):
            ## Try to detect the layout, first as a ScalableSeries, then as a ScalableDataFrame.
            detected_layout: Optional[DataLayout] = ScalableSeries.detect_layout(data, raise_error=False)
            if detected_layout is not None:
                data: ScalableSeries = ScalableSeries.get_subclass(detected_layout)(data=data)
            else:
                detected_layout: Optional[DataLayout] = ScalableDataFrame.detect_layout(data, raise_error=False)
                if detected_layout is None:
                    raise NotImplementedError(f'Cannot infer layout of data with type: {type(data)}.')
                data: ScalableDataFrame = ScalableDataFrame.get_subclass(detected_layout)(data=data)
        if isinstance(data, ScalableSeries):
            data: ScalableDataFrame = data.to_frame()
        assert isinstance(data, ScalableDataFrame)
        if layout is None:
            return data
        ScalableDataFrameClass: Optional[Type[ScalableDataFrame]] = cls.get_subclass(layout, raise_error=False)
        if ScalableDataFrameClass is None:
            raise ValueError(
                f'Cannot create {ScalableDataFrame} subclass having layout "{layout}"; '
                f'available subclasses are: {ScalableDataFrame.subclasses()}'
            )
        ## When passed either raw data (in the correct format) or a ScalableDataFrame, the respective
        ## ScalableDataFrame subclass should be able to accept it.
        return ScalableDataFrameClass(data=data, **kwargs)

    @property
    def hvplot(self) -> Any:
        with optional_dependency('hvplot', error='raise'):
            import hvplot.pandas
            return self.pandas().hvplot

    @classmethod
    def detect_layout(cls, data: Any, raise_error: bool = True) -> Optional[DataLayout]:
        if is_scalable(data):
            return data.layout
        for possible_layout in SDF_DATA_LAYOUT_PRIORITY:
            ScalableDataFrameClass: Optional[Type[ScalableDataFrame]] = cls.get_subclass(
                possible_layout,
                raise_error=False,
            )
            if ScalableDataFrameClass is None:
                continue
            if ScalableDataFrameClass.layout_validator(data, raise_error=False):
                return possible_layout
        if raise_error:
            raise NotImplementedError(
                f'Cannot infer layout of data having type: {type(data)}. '
                f'Please pass `layout=...` using one of the following: {list(DataLayout)}'
            )
        else:
            return None

    def to_layout(self, layout: Optional[DataLayout] = None, **kwargs) -> ScalableDataFrame:
        return self.as_layout(layout=layout, **kwargs)

    @safe_validate_arguments
    def as_layout(self, layout: Optional[DataLayout] = None, **kwargs) -> ScalableDataFrame:
        if layout is None:
            return self
        return self.of(self, layout=layout, **kwargs)

    @classmethod
    def is_record(cls, data: Any, raise_error: bool = True) -> bool:
        if data is None:
            if raise_error:
                raise ValueError(f'Input data cannot be None.')
            return False
        if not isinstance(data, dict):
            if raise_error:
                raise ValueError(f'Expected input to be dict with scalar values; found input with type: {type(data)}')
            return False
        ## data is a dict
        for k, v in data.items():
            v_is_not_datum: bool = not ScalableSeries.is_datum(v)
            if v_is_not_datum:
                if raise_error:
                    err_msg: str = f'Expected keys and values of input dict to be scalars; found '
                    if v_is_not_datum:
                        err_msg += f'non-scalar value {v} (type {type(v)})'
                    raise ValueError(err_msg)
                return False
        return True

    @classmethod
    def is_dict(cls, data: Any, raise_error: bool = True) -> bool:
        if data is None:
            if raise_error:
                raise ValueError(f'Input data cannot be None.')
            else:
                return False
        valid: bool = isinstance(data, dict)
        if not valid and raise_error:
            raise ValueError(f'Expected input to be dict; found input with type: {type(data)}')
        return valid

    @classmethod
    def is_list_of_dict(cls, data: Any, raise_error: bool = True) -> bool:
        if data is None:
            if raise_error:
                raise ValueError(f'Input data cannot be None.')
            else:
                return False
        valid: bool = is_list_of_dict_like(data)
        if not valid and raise_error:
            raise ValueError(f'Expected input to be dict or list of dicts; found input with type: {type(data)}')
        return valid

    @classmethod
    def is_numpy_record_array(cls, data: Any, raise_error: bool = True) -> bool:
        if data is None:
            if raise_error:
                raise ValueError(f'Input data cannot be None.')
            else:
                return False
        valid: bool = isinstance(data, np.recarray)
        if not valid and raise_error:
            raise ValueError(f'Expected input to be NumPy record array; found input with type: {type(data)}')
        return valid

    @classmethod
    def is_pandas(cls, data: Any, raise_error: bool = True) -> bool:
        if data is None:
            if raise_error:
                raise ValueError(f'Input data cannot be None.')
            else:
                return False
        valid: bool = isinstance(data, PandasDataFrame)
        if not valid and raise_error:
            raise ValueError(f'Expected input to be Pandas DataFrame; found input with type: {type(data)}')
        return valid

    @classmethod
    def is_dask(cls, data: Any, raise_error: bool = True) -> bool:
        if data is None:
            if raise_error:
                raise ValueError(f'Input data cannot be None.')
            else:
                return False
        valid: bool = isinstance(data, DaskDataFrame)
        if not valid and raise_error:
            raise ValueError(f'Expected input to be Dask DataFrame; found input with type: {type(data)}')
        return valid

    @safe_validate_arguments
    def valid(
            self,
            validator: Callable[[Any, bool], bool],
            sample_size: Union[SampleSizeType, Literal[False], Literal[0.0], Literal[0]] = True,
            seed: Optional[int] = None,
            return_failed: bool = False,
            **kwargs,
    ) -> Union[bool, Tuple]:
        """
        Runs a validator function element-wise on the data, and throws an exception if any element is invalid.
        :param validator: function to call element-wise.
        :param sample_size: amount of data to validate.
            If False, it will not validate data. If True, it will validate entire data.
            If 0.0 < sample_size <= 1.0, then we will validate a fraction of the data.
            If 1 < sample_size, we will validate these many rows of data.
        :param seed: random seed for sampling a fraction.
        :param return_failed: whether to return the rows which failed validation.
        """
        if sample_size in {False, 0.0, 0}:
            return True
        sample: ScalableDataFrame = self
        if sample_size in {True, 1.0}:
            sample: ScalableDataFrame = self
        elif 0.0 < sample_size < 1.0:
            sample: ScalableDataFrame = self.sample(frac=float(sample_size), random_state=seed)
        elif 1 < sample_size:
            length: int = len(self)
            n: int = resolve_sample_size(sample_size=sample_size, length=length)
            sample: ScalableDataFrame = self.sample(n=int(n), random_state=seed)
        kwargs['na_action'] = None  ## Always pass the cell-value to the validator function.
        valid: ScalableDataFrame = sample.applymap(validator, **kwargs)
        is_valid: bool = valid.mean().all().all()
        if return_failed:
            failed_sample: ScalableDataFrame = sample.loc[~valid.all(axis=1)]
            return is_valid, failed_sample
        return is_valid

    @safe_validate_arguments
    def compress(
            self,
            *,
            layout: DataLayout = DataLayout.LIST_OF_DICT,
            engine: Union[Tuple[CompressionEngine, ...], CompressionEngine] = (
                    CompressionEngine.BROTLI, CompressionEngine.GZIP,
            ),

            compression_kwargs: Optional[Dict] = None,
            base64_encoding: bool = False,
            **kwargs,
    ) -> CompressedScalableDataFrame:
        engine: List[CompressionEngine] = as_list(engine)
        compression_kwargs: Dict = get_default(compression_kwargs, dict())
        allowed_compression_layouts: Set[DataLayout] = {DataLayout.DICT, DataLayout.LIST_OF_DICT}
        if layout not in allowed_compression_layouts:
            raise ValueError(
                f'Cannot compress using layout: {layout}; can only use '
                f'{StringUtil.join_human(allowed_compression_layouts)}'
            )
        data_bytes: bytes = StringUtil.jsonify(
            self.to_layout(layout=layout, **kwargs).raw(),
            minify=True,
        ).encode('utf-8')
        for eng in engine:
            if eng is CompressionEngine.BROTLI:
                compression_kwargs.setdefault('quality', 11)  ## Smallest compressed size
                with optional_dependency('brotli', error='ignore'):
                    import brotli
                    payload: bytes = brotli.compress(data_bytes, **compression_kwargs)
                    if base64_encoding:
                        payload: str = base64.urlsafe_b64encode(payload).decode('utf-8')
                        # print(f'Type of data post b64 encoding: {type(payload)}')

                    return CompressedScalableDataFrame(
                        payload=payload,
                        compression_engine=eng,
                        layout=layout,
                        base64_encoding=base64_encoding,
                    )
                continue  ## If brotli is not found.
            elif eng is CompressionEngine.GZIP:
                compression_kwargs.setdefault('compresslevel', 9)  ## Smallest compressed size
                payload: bytes = gzip.compress(data_bytes, **compression_kwargs)
                if base64_encoding:
                    payload: str = base64.urlsafe_b64encode(payload).decode('utf-8')
                    # print(f'Type of data post b64 encoding: {type(payload)}')
                return CompressedScalableDataFrame(
                    payload=payload,
                    compression_engine=eng,
                    layout=layout,
                    base64_encoding=base64_encoding,
                )
            else:
                raise NotImplementedError(f'Unsupported compression engine: "{eng}"')
        raise NotImplementedError(f'Could not compress using any of these engines: {engine}')

    @safe_validate_arguments
    def split(
            self,
            prefix: constr(min_length=1) = chunk_prefix,
            **kwargs,
    ) -> Dict[str, Any]:
        chunks_list: List[ScalableDataFrameOrRaw] = list(self.stream(**kwargs))
        length: int = sum([len(chunk) for chunk in chunks_list])
        rows_covered_so_far = 0
        chunks_dict: Dict[str, ScalableDataFrame] = {}
        for chunk_i in range(0, len(chunks_list)):
            idx_start: int = rows_covered_so_far
            idx_end: int = rows_covered_so_far + len(chunks_list[chunk_i])
            chunk_name = f'{prefix}-{StringUtil.pad_zeros(chunk_i + 1, len(chunks_list))}'
            chunk_name += f'-rows-{StringUtil.pad_zeros(idx_start + 1, length)}-{StringUtil.pad_zeros(idx_end, length)}'
            chunks_dict[chunk_name] = chunks_list[chunk_i]
            rows_covered_so_far += len(chunks_list[chunk_i])
        return chunks_dict

    def stream(self, **kwargs) -> Generator[ScalableDataFrameOrRaw, None, None]:
        Alias.set_num_rows(kwargs)
        Alias.set_num_chunks(kwargs)
        Alias.set_seed(kwargs)
        Alias.set_shard_seed(kwargs)
        Alias.set_num_workers(kwargs)
        Alias.set_parallelize(kwargs)
        Alias.set_mapper(kwargs, param='map')
        Alias.set_map_executor(kwargs)
        Alias.set_map_failure(kwargs)
        Alias.set_shard_rank(kwargs)
        Alias.set_num_shards(kwargs)

        if kwargs.get('parallelize') == Parallelize.sync:
            if kwargs.get('num_workers', 1) != 1:
                warnings.warn(
                    f'When setting parallelize={Parallelize.sync}, we implicitly use only one worker (the main thread). '
                    f'The entered value of `num_workers` will be ignored.'
                )
            kwargs['num_workers'] = 1

        if kwargs.get('stream_as') in LAZY_SDF_DATA_LAYOUTS:
            raise AttributeError(
                f'Cannot stream data as a {kwargs["stream_as"]}, `stream_as` must be an in-memory datatype '
                f'on the client, such as {DataLayout.LIST_OF_DICT} or {DataLayout.PANDAS}'
            )
        if multiple_are_not_none(kwargs.get('num_rows'), kwargs.get('num_chunks')):
            raise ValueError(f'Only one of `num_rows` or `num_chunks` can be non-None.')
        if all_are_not_none(kwargs.get('shard_rank'), kwargs.get('num_shards')):
            kwargs['shard']: Tuple[int, int] = (kwargs.pop('shard_rank'), kwargs.pop('num_shards'))
        if kwargs.get('shard') is not None:
            shard_rank, num_shards = kwargs['shard']
            if not (isinstance(shard_rank, int) and isinstance(num_shards, int) and 0 <= shard_rank < num_shards):
                raise ValueError(
                    f'Expected `shard` to be a tuple of two elements: (shard_rank, num_shards), '
                    f'where 0 <= shard_rank < num_shards; found: shard_rank={shard_rank}, num_shards={num_shards}'
                )
        kwargs.setdefault('map_kwargs', {})
        return self._stream_chunks(**kwargs)

    @safe_validate_arguments
    def _stream_chunks(
            self,
            map_kwargs: Dict,
            num_rows: Optional[conint(ge=1)] = None,
            num_chunks: Optional[conint(ge=1)] = None,
            stream_as: Optional[DataLayout] = None,
            raw: bool = False,
            shuffle: bool = False,
            seed: Optional[int] = None,
            map: Optional[Callable] = None,
            num_workers: conint(ge=1) = 1,
            parallelize: Parallelize = Parallelize.sync,
            map_failure: Literal['raise', 'drop'] = 'raise',
            map_executor: Literal['spawn'] = 'spawn',
            shard: Tuple[conint(ge=0), conint(ge=1)] = (0, 1),
            shard_shuffle: bool = False,
            shard_seed: Optional[int] = None,
            reverse_sharding: bool = False,
            drop_last: Optional[bool] = None,
            **kwargs,
    ) -> Generator[ScalableDataFrameOrRaw, None, None]:
        """
        Retrieve data from the ScalableDataframe in batches (either ScalableDataFrame objects or raw data).
        :param num_rows: maximum number of rows to fetch in each ScalableDataFrame.
        :param num_chunks: number of ScalableDataFrame chunks to create.
        :param stream_as: which layout should the returned ScalableDataFrame chunks.
            Defaults to most convenient layout.
        :param raw: whether to yield the raw data (e.g. dict, Pandas DataFrame, Dask DataFrame) or ScalableDataFrame.
            Defaults to use ScalableDataFrame.
        :param shuffle: whether to shuffle the data while streaming.
        :param shard_shuffle: whether to shuffle shards.
        :param seed: the random seed to use for shuffling.
        :param map: optional function to process the ScalableDataFrame before yielding it.
            Processing can be both CPU and IO intensive: an example is reading image data from disk/S3 and converting
            it to a PyTorch tensor. The input and output of this callabe should be a ScalableDataFrame.
        :param map_kwargs: keyword arguments to pass to the processing function.
        :param num_workers: when passing `map`, this determines how many chunks to map simultaneously.
        :param parallelize: when passing `map`, this controls how mapping of chunks is parallelized, e.g. via
            threads, processes, Ray, etc.
        :param map_executor: how to handle the executor:
            (a) spawn: creates a new Executor instance for every call to .stream()
            (b) global: uses the globally-available executor
        :param shard: how to shard the data which is streamed. This should be a 2-tuple of (shard_rank, num_shards).
            Defaults to (0, 1).
        :param shard_seed: when using shuffle=True and sharding, this determines the selection of the shards.
        :param reverse_sharding: if set to True, then pick all shards *except* the specified rank.
            E.g. if shard=(2, 5), we pick all shards where (shard_i%5 != 2).
            This is useful for K-fold Cross-validation, as we can select train and val sets as follows:
            for fold_i in range(K):  ## num_folds is the "K" in K-fold CV
                val_dataset_iter = dataset.read_batches(shard=(fold_i, K))
                train_dataset_iter = dataset.read_batches(shard=(fold_i, K), reverse_sharding=True)
            Defaults to False.
        :param drop_last: determines whether to align the batches and drop the last batch.
            (a) None: default option. We will not align batches.
            (b) False: aligns the batches across shards (meaning, we ensure each shard has an equal number of batches).
                Ensures that all batches have the same number of rows, except the last batch which might have a
                different number of rows between 1 and 2*batch_size. Here, we ensure no rows are dropped.
                drop_last=False is suitable for distributed Deep Learning inference, where we want to ensure each worker
                gets a batch, but we are flexible on the size of the batch.
            (c) True: aligns the batches across shards (meaning, we ensure each shard has an equal number of batches).
                Ensures that all batches have exactly the same number of rows. To do this, we might drop rows, with
                maximum of num_shards*batch_size rows being dropped.
                drop_last=True is suitable for distributed Deep Learning training, where we want to ensure each worker
                gets a batch, and all batches have exactly the same number of rows (for gradient synchronization). This
                might mean we drop a negligible number of rows, which should not affect the overall training procuedure.
        :return: yield a single smaller ScalableDataFrame.
        """
        ## TODO: implement chunk_size: Optional[Union[conint(ge=1), constr(regex=StringUtil.FILE_SIZE_REGEX)]] = None
        ## docstring for chunk_size: maximum size of each ScalableDataFrame in bytes (int) or string (e.g. "10MB").
        try:
            mapped_sdf_chunks: Deque[Dict[str, Union[int, Future]]] = deque()
            executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = self._stream_get_executor(
                map=map,
                parallelize=parallelize,
                num_workers=num_workers,
                map_executor=map_executor,
            )

            sdf: ScalableDataFrame = self._stream_in_memory_sdf_convert_to_streaming_layout(stream_as=stream_as)
            ## Get the complete list of batches which we must stream.
            ## Here, `row_idxs` is a list of numpy arrays, where each numpy array has the row-indexes which make
            ## up the chunk which will be yielded.
            chunks_row_idxs: List[np.ndarray] = self._stream_in_memory_sdf_get_shard_chunks_row_idxs(
                length=len(sdf),
                shard=shard,
                num_rows=num_rows,
                num_chunks=num_chunks,
                drop_last=drop_last,
                shuffle=shuffle,
                shard_shuffle=shard_shuffle,
                seed=seed,
                shard_seed=shard_seed,
                reverse_sharding=reverse_sharding,
            )
            ## Stream chunks in the target layout:
            chunks_returned: int = 0
            for chunk_row_idxs in chunks_row_idxs:
                out_sdf_chunk: ScalableDataFrame = self._stream_in_memory_sdf_select_row_idxs(
                    sdf,
                    chunk_row_idxs=chunk_row_idxs,
                    stream_as=stream_as,
                )
                ## Convert the raw chunk to an SDF of the required layout, and enqueue it for processing.
                ## We might also dequeue the head chunk, if it has finished processing or the queue length is
                ## exceeded.
                out_sdf_chunk: Optional[ScalableDataFrame] = self._stream_loop_enqueue_dequeue_map_chunk(
                    out_sdf_chunk=out_sdf_chunk,
                    mapped_sdf_chunks=mapped_sdf_chunks,
                    chunks_returned=chunks_returned,
                    map=map,
                    map_kwargs=map_kwargs,
                    parallelize=parallelize,
                    num_workers=num_workers,
                    map_failure=map_failure,
                    stream_as=stream_as,
                    raw=raw,
                    executor=executor,
                )
                ## Only yield if we actually dequeued the head chunk (i.e. it finished processing).
                if out_sdf_chunk is not None:
                    yield out_sdf_chunk
                chunks_returned += 1
            while len(mapped_sdf_chunks) != 0:
                out_sdf_chunk: Optional[ScalableDataFrame] = self._stream_dequeue_mapped_chunk(
                    mapped_sdf_chunks=mapped_sdf_chunks,
                    stream_as=stream_as,
                    map_failure=map_failure,
                )
                if out_sdf_chunk is not None:
                    yield self._stream_chunk_to_raw(out_sdf_chunk, raw=raw)
        finally:
            del mapped_sdf_chunks
            del map_kwargs
            self._stream_cleanup_executor(
                executor=executor,
                map_executor=map_executor,
            )
        return

    stream.__doc__ = _stream_chunks.__doc__  ## Clone docstring. Ref: https://stackoverflow.com/a/68901244/4900327

    @classmethod
    @safe_validate_arguments
    def _stream_in_memory_sdf_get_shard_chunks_row_idxs(
            cls,
            length: conint(ge=1),
            shard: Tuple[conint(ge=0), conint(ge=1)],
            num_rows: Optional[conint(ge=1)],
            num_chunks: Optional[conint(ge=1)],
            drop_last: Optional[bool],
            shuffle: bool,
            shard_shuffle: bool,
            seed: Optional[int],
            shard_seed: Optional[int],
            reverse_sharding: bool,
    ) -> List[np.ndarray]:
        shard_rank, num_shards = shard
        if not (0 <= shard_rank < num_shards):
            raise DataFrameShardingError(
                f'`shard_rank` must be in range [0, {num_shards}); '
                f'found shard_rank={shard_rank}.'
            )

        if num_shards > length:
            raise DataFrameShardingError(
                f'Cannot shard DataFrame of {length} rows into {num_shards} shards; DataFrame length is insufficient. '
                f'Please reduce the number of shards.'
            )

        if num_shards > 1:
            if shard_shuffle and shard_seed is None:
                raise DataFrameShardingError(
                    f'When calling .stream() with {num_shards} shards and shard_shuffle=True, '
                    f'you must pass `shard_seed` to ensure you get consistent results.'
                )
        if num_rows is not None:
            ## Here, we split the DataFrame into batches of <= num_rows
            ## If we set drop_last=True, then we will get exactly equal-sized batches of `num_rows` (the last batch
            ## might be dropped if its size is < num_rows).
            ## If we set drop_last={False, None}, all batches will have size of `num_rows` except the last batch (which
            ## might have size < num_rows).
            num_rows: int = min(num_rows, length)
            if num_shards * num_rows > length:
                raise DataFrameShardingError(
                    f'Cannot shard DataFrame into {num_shards} shards into batches of size {num_rows}; '
                    f'{num_shards}*{num_rows} is more than the length of the DataFrame ({length} rows), '
                    f'so we cannot even create one batch for each shard. '
                    f'Please reduce the number of shards and/or reduce the number of rows per batch.'
                )

            sharding_num_rows: int = num_rows
            if drop_last in {False, None}:
                sharding_drop_last: bool = False
            else:
                sharding_drop_last: bool = True
        elif num_chunks is not None:
            ## Here, we always keep all the rows, and later re-shuffle the partitions.
            num_chunks: int = min(num_chunks, length)
            if num_shards * num_chunks > length:
                raise DataFrameShardingError(
                    f'Cannot shard DataFrame into {num_shards} shards into {num_chunks} chunks; '
                    f'{num_shards}*{num_chunks} is more than the length of the DataFrame ({length} rows), '
                    f'so we cannot put even one row for each shard. '
                    f'Please reduce the number of shards and/or reduce the number of chunks.'
                )
            if drop_last in {False, None}:
                sharding_drop_last: bool = False
                sharding_num_rows: int = 1
            else:
                sharding_drop_last: bool = True
                ## Ensure we get a fixed number of chunks.
                sharding_num_rows: int = math.floor(length / (num_shards * num_chunks))
                if num_shards * num_chunks > length:
                    raise DataFrameShardingError(
                        f'Cannot shard DataFrame into {num_shards} shards with {num_chunks} per shard; '
                        f'{num_shards}*{num_chunks} is more than the length of the DataFrame ({length} rows), '
                        f'so we cannot even create one batch of 1 row for each shard. '
                        f'Please reduce the number of shards and/or reduce the number of chunks per shard.'
                    )
            # else:
            # raise ValueError(
            #     f'Cannot use `num_chunks` with non-None value of `drop_last`; there are situations where it is '
            #     f'not mathematically possible to create chunks of the same size (except the last chunk). '
            #     f'An example is length=63, num_chunks=10, num_shards=1; in this case, if we try to create exactly '
            #     f'10 chunks of equal size except the last, it is not possible...we will always get 9 chunks having '
            #     f'7 rows each.'
            # )

        else:
            raise NotImplementedError(f'Must pass exactly one of `num_rows` or `num_chunks`; found both to be None')
        ## Get the (shuffled) row idxs which should be in the current shard:
        shard_row_idxs, num_chunks_per_shard = cls._stream_in_memory_get_balanced_shard_row_idxs(
            length=length,
            shard=shard,
            num_rows=sharding_num_rows,
            drop_last=sharding_drop_last,
            shuffle=shuffle,
            shard_shuffle=shard_shuffle,
            seed=seed,
            shard_seed=shard_seed,
            reverse_sharding=reverse_sharding,
        )
        shard_length: int = len(shard_row_idxs)

        shard_chunks_row_idxs: List[np.ndarray] = []
        shard_chunks_row_lens: List[int] = []
        chunks_returned: int = 0
        rows_i: int = 0
        while rows_i < shard_length:
            if num_rows is not None:
                shard_chunk_row_idxs: np.array = shard_row_idxs[rows_i: rows_i + num_rows]
                if drop_last is False and chunks_returned == num_chunks_per_shard - 1:
                    ## We are in the last chunk/batch, so we should pad it with all the remaining rows.
                    shard_chunk_row_idxs: np.array = shard_row_idxs[rows_i: length]
            elif num_chunks is not None:
                if drop_last in {None, False}:
                    ## In this condition, `num_chunks` is not None, so we want to ensure we return `num_chunks` chunks,
                    ## which should all be nearly the same size (since this keeps the processing speed consistent).
                    ## We use function _stream_update_num_rows_according_to_num_chunks to set the number of rows
                    ## to return...for the first length % num_chunks iterations, we will return
                    ## math.ceil(length / num_chunks) rows, and for the remaining, we will return
                    ## math.floor(length / num_chunks).
                    __num_rows: int = cls._stream_update_num_rows_according_to_num_chunks(
                        length=shard_length,
                        chunks_returned=chunks_returned,
                        num_chunks=num_chunks,
                    )
                    shard_chunk_row_idxs: np.array = shard_row_idxs[rows_i: rows_i + __num_rows]
                elif drop_last is True:
                    __num_rows: int = math.floor(shard_length / num_chunks)
                    shard_chunk_row_idxs: np.array = shard_row_idxs[rows_i: rows_i + __num_rows]
            rows_i += len(shard_chunk_row_idxs)  ## Handles the case where rows_i + num_rows exceeds shard_length
            chunks_returned += 1
            shard_chunks_row_idxs.append(shard_chunk_row_idxs)
            shard_chunks_row_lens.append(len(shard_chunk_row_idxs))
        if num_chunks is not None and num_chunks <= shard_length and drop_last in {None, False}:
            if len(shard_chunks_row_idxs) != num_chunks:
                raise ValueError(
                    f'Expected shard {shard} of {shard_length} rows would be split into {num_chunks} chunks; however, '
                    f'created {len(shard_chunks_row_idxs)} chunks with following lengths: {shard_chunks_row_lens}'
                )
        if drop_last is True:
            shard_chunks_row_lens: np.ndarray = np.array(shard_chunks_row_lens)
            if not np.all(shard_chunks_row_lens == shard_chunks_row_lens[0]):
                raise ValueError(
                    f'(Shard: {shard}): When setting drop_last=True (num_rows={num_rows}, num_chunks={num_chunks}), '
                    f'expected all chunks to be the same size; however, created {len(shard_chunks_row_idxs)} chunks '
                    f'with the following lengths (from shard of {shard_length} rows): {shard_chunks_row_lens}'
                )
        return shard_chunks_row_idxs

    def _stream_in_memory_sdf_convert_to_streaming_layout(self, stream_as: DataLayout) -> ScalableDataFrame:
        ## Convert to the target data layout, then select chunks.
        return self.as_layout(layout=stream_as)

    @classmethod
    def _stream_in_memory_sdf_select_row_idxs(
            cls,
            sdf: ScalableDataFrame,
            chunk_row_idxs: np.ndarray,
            stream_as: DataLayout,
    ) -> ScalableDataFrame:
        out_sdf_chunk: ScalableDataFrame = ScalableDataFrame.of(
            sdf.iloc[chunk_row_idxs],
            layout=stream_as
        )
        return out_sdf_chunk

    @classmethod
    @safe_validate_arguments
    def _stream_in_memory_get_balanced_shard_row_idxs(
            cls,
            length: conint(ge=1),
            shard: Tuple[conint(ge=0), conint(ge=1)],
            num_rows: conint(ge=1),
            drop_last: bool,
            shuffle: bool,
            shard_shuffle: bool,
            seed: Optional[int],
            shard_seed: Optional[int],
            reverse_sharding: bool,
    ) -> Tuple[np.ndarray, int]:
        """
        For in-memory ScalableDataFrames, we should use this function to get an equal number of batches per shard.
        Depending on the value of drop_last, we might also ensure an equal number of rows per batch:
        - drop_last=True: we ensure an equal number of rows per batch, by possibly dropping rows from the end of the
        dataset.
        - drop_last=False: we get an equal number of rows per batch for all batches except in the very last batch in
        each shard. The size of the last batch differs by at most 1 row across shards.

        As we have in-memory dataframes (where we assume indexing is cheap), this function uses
        _stream_get_balanced_shard_intervals to first get shard intervals, then create shuffled row-indexes based on the
        intervals. We set npartitions=1 as the data is in-memory in one big object (PandasDataFrame, list-of-dict, etc).
        """
        shard_rank, num_shards = shard
        _, intervals, num_chunks_per_shard = cls._stream_get_balanced_shard_intervals(
            length=length,
            npartitions=1,
            num_shards=num_shards,
            num_rows=num_rows,
            drop_last=drop_last,
        )

        ## Select the intervals to make up the shard:
        if reverse_sharding:
            shard_intervals: List[Tuple[int, int]] = [
                interval
                for interval_i, interval in enumerate(intervals)
                if interval_i % num_shards != shard_rank  ## Only pick intervals which do NOT belong to this rank.
            ]
        else:
            shard_intervals: List[Tuple[int, int]] = [
                interval
                for interval_i, interval in enumerate(intervals)
                if interval_i % num_shards == shard_rank  ## Only pick intervals which belong to this rank.
            ]

        ## Use the sharded intervals to select from the (shuffled) row indexes.
        ## This ensures we select truly random rows in each shard.
        row_idxs: np.ndarray = np.arange(0, length)
        if shard_shuffle:
            ## Set a random seed to ensure we get the same selection of shards.
            ## Ref: https://stackoverflow.com/a/47742676
            np_random_sharding = np.random.RandomState(seed=shard_seed)
            ## IMPORTANT: shuffled row_idxs should be the same across all shards.
            ## This is done by using RandomState. Ref: https://stackoverflow.com/a/47742676
            np_random_sharding.shuffle(row_idxs)

        ## Select shard:
        shard_row_idxs: np.ndarray = np.concatenate([
            row_idxs[shard_interval_start: shard_interval_end]
            for (shard_interval_start, shard_interval_end) in shard_intervals
        ])

        if shuffle:
            ## Set a random seed to ensure we get the same ordering of examples within a shard.
            ## Ref: https://stackoverflow.com/a/47742676
            np_random_within_shard = np.random.RandomState(seed=seed)
            ## IMPORTANT: when setting the same seed, we should get the same ordering within a shard.
            ## This is done by using RandomState. Ref: https://stackoverflow.com/a/47742676
            np_random_within_shard.shuffle(shard_row_idxs)

        # print(f'shard_seed={shard_seed}, seed={seed}, shard_row_idxs: {repr(shard_row_idxs)}')
        return shard_row_idxs, num_chunks_per_shard

    @classmethod
    @safe_validate_arguments
    def get_closest_npartitions(cls, npartitions: int, num_shards: int) -> int:
        S: int = num_shards
        if npartitions % S == 0:
            new_P: int = npartitions
        else:
            ## Pick the number closest to the current number of partitions.
            ## E.g. if we have 83 partitions, and we request 16 shards, then we should pick 80 rather than 96,
            ## since 80 is closet to 83 (our current number of partitions).
            npartitions_low: int = S * max(1, npartitions // S)
            npartitions_high: int = npartitions_low + S
            if abs(npartitions - npartitions_low) < abs(npartitions - npartitions_high):
                new_P: int = npartitions_low
            else:
                new_P: int = npartitions_high
        return new_P

    @classmethod
    @safe_validate_arguments
    def _stream_get_balanced_shard_intervals(
            cls,
            length: conint(ge=1),
            npartitions: conint(ge=1),
            num_shards: conint(ge=1),
            num_rows: conint(ge=1),
            drop_last: bool,
    ) -> Tuple[
        List[int],
        List[Tuple[int, int]],
        int,
    ]:
        """
        Here we need "balanced" shards, i.e. for each shard, we get **exactly the same number of batches**.
        This must also work for DataFrames with multiple partitions, e.g. a DaskDataFrame.

        This is important in two cases:
        1. Distributed training with "W" workers (DistributedDataParallel): here, in each step of training, we must
        do a forward-backward pass and then synchronize gradients across all workers. Thus, in each step, we want to
        ensure all workers get a batch of data, with the exact same number of rows. If in any step a worker does not
        get a batch, or if the different workers get a different same number of rows in their batch, then our gradient
        synchronization will not work as expected.
        In this situation, it is recommended to set drop_last=True, as there is a tolerance for dropping a tiny amount
        of data, e.g. dropping the last few rows in order to get equal number of batches of equal sizes.
        This not affect our training procedure significantly e.g. with N=1,000,000 rows, B=16 batch-size & W=24 workers,
        we will  run a total of 2,604 gradient-update steps before we reach the last (incomplete) column of batched
        which we drop.
        2. Distributed inference with "W" workers: here, in each step of inference, each worker loads a batch of data,
        does a forward pass, then broadcasts their predictions-batch to all other workers. If in any step a worker does
        not get a batch of data, then the broadcast operation will fail.
        In this situation, it is recommended to set drop_last=False, as there is a zero tolerance for dropping rows, but
        it is okay for the batches to have a different number of rows...it is only important that we have the same
        number of batches for each worker.

        Note that this functions requires we know our DataFrame length "N" in advance, and also we are doing sharding
        with a batch_size B (we cannot us this for num_chunks style sharding). Thus, we should invoke this function
        mostly for distributed training or distributed inference, where we perform some batch-wise synchronization
        operation (e.g. gradient synchronization or broadcast).

        =======================================================================
        Below is a detailed explanation of the procedure followed in this function.

        In the most general case, assume we are working with a DaskDataFrame with P partitions currently.
        For the case of in-memory dataframes like Pandas, ListOfDict, etc we have P=1.
        Let:
        N = total number of rows in the DaskDataFrame
        B = batch size
        S = num shards
        P = current number of partitions

        Currently, the DaskDataFrame can be partitioned in any-which way.
        We want to rebalance the DaskDataFrame such that we get the situation below (num_shards=4):
           s0:  █   █   █   █   █   █   █   █   █   █    ... ... █   █   █   █   █   █   █   █
           s1:  █   █   █   █   █   █   █   █   █   █    ... ... █   █   █   █   █   █   █   ▀  <- last batch in
           s2:  █   █   █   █   █   █   █   █   █   █    ... ... █   █   █   █   █   █   █   ░     incomplete, as
           s3:  █   █   █   █   █   █   █   █   █   █    ... ... █   █   █   █   █   █   █   ░     N%B != 0
        batch:  0   1   2   3   4   5   6   7   8   9    ... ...                             last
                                                                                             ⮑ last column is
                                                                                                   incomplete, as
                                                                                                   N%(S*B) != 0
        Where partitions are grouped like this:
           s0: <█===█===█===█===█> <█===█===█===█===█>   ... ...<█===█===█===█===█> <█===█>
           s1: <█===█===█===█===█> <█===█===█===█===█>   ... ...<█===█===█===█===█> <█===▀>
           s2: <█===█===█===█===█> <█===█===█===█===█>   ... ...<█===█===█===█===█> <█===░>
           s3: <█===█===█===█===█> <█===█===█===█===█>   ... ...<█===█===█===█===█> <█===░>
        group:          1                   2            ... ...         ?           last

        To do so, we must ensure all shards are within "B" rows of each other. So, we must create new divisions of the
        DaskDataFrame i.e. "rebalance it".

        Steps:
        1. We will first ensure that the number of partitions is a multiple of the number of shards
        - However, we don't want to go too far from the original number of partitions, e.g. if the dataframe currently
        has P=83 partitions currently, and we require S=4 shards, we don't want to re-partition the dataframe to have
        new_P=4 partitions, because 83 is very different from 4. The user had probably set 83 partitions to ensure good
        load-balancing.
        - Instead, we want to set the new number of partitions to the multiple of S, which is closest to P. This can
        be achieved by setting new_P as S*math.floor(P/S) or S*math.ceil(P/S), whichever is closer to P.

        2. Next, we should ensure that all partitions have an equal number of rows, except the final partition for each
        shard (depicted by the last "group" of partitions in the diagram above). In the final partition for each shard,
        we want to ensure that the partitions do not differ by more than "B" rows.
        - Let "M" be the batches per partition (in diagram above, M=5). Notice that in the diagram above, each "group"
        has M*S batches, i.e. M*S*B rows. Thus, each partition has <=M*B rows.
        - We know that we want new_P partitions total, where the last group might be incomplete, but all other
        groups have M*S*B rows.
        Thus, we need to find M > 1 such that:
            math.floor(N / (M*S*B)) == (new_P-1)
        The value for "M" which satisfies this is:
            M = 1 + (           ## Correction value
                    (N//(S*B))  ## Number of "complete" columns of S*B rows
                )// (new_P//S)  ## Number of groups we want i.e. new_P // S (this is perfect division)
            You can verify this empirically:
                for M in range(1, int(1e9)):
                    if math.ceil(N / (M*S*B)) == (new_P // S):
                        break
                assert 0 <= (N - M*S*B*((new_P//S)-1)) <= M*S*B
        We use this value of "M" iteratively select partitions of size M*B rows.

        3. Finally, we must process the last group, while ensuring we get a roughly equal distribution
        We can detect we are in the last group if we have less than M*S*B rows left from the total N rows in our
        DataFrame, i.e. the last group has (N - M*S*B*((new_P//S)-1)) rows. If we have 0 rows remaining, our dataset
        was perfectly divisible by (M*S*B), so we are done.
        - When we are in the last group, we should select complete columns of S*B batches. Here, the value of
        (N - M*S*B*((new_P//S)-1)) // (S*B) will tell us how many complete columns we have, and thus we should create
        partitions with B*(N - M*S*B*((new_P//S)-1)) rows each. Here, we will have exactly "S" such partitions.

        4. The final column of the last group might be incomplete (i.e. have < S*B rows).
        We can detect we are in the last column when we have less than S*B rows remaining. If we have 0 rows remaining,
        our dataset was perfectly divisible by (S*B), so we are done.
        - When setting drop_last=True (e.g. in distributed training), we want to drop this completely, as it has a
        negligible impact on the training procedure.
        - When setting drop_last=False (e.g. in distributed inference), we cannot drop even one row, but still want to
        ensure each shard gets an equal number of batches. What we do is evenly "pad" the previous shards with rows
        from the last (incomplete) column, i.e. we will try to assign the remaining rows < S*B evenly over the batches
        in the "last group". This ensures that in the last group (i.e. last partition of each shard) we have batches
        which are close to perfectly balanced, i.e. < 2*B rows, and with at most 1 row difference between batches.

        =======================================================================
        ## Below code plots the outputs of this function.
        ## Edit these values:
        length = 25003   ## Length of the dataset.
        npartitions = 39  ## Current number of partitions. This will be rounded to the nearest multiple of num_shards
        num_shards = 8  ## Number of workers/shards
        batch_size = 256  ## Batch size
        drop_last = False

        shard_intervals: Dict[int, List[Tuple[int, int]]] = _stream_shard_partition_intervals(
            length=length,
            npartitions=npartitions,
            num_shards=num_shards,
            batch_size=batch_size,
            drop_last=drop_last,
        )

        from datetime import datetime
        import holoviews as hv
        Viz.init('hvplot_bokeh')
        interval_data = []
        for shard_i, intervals in shard_intervals.items():
            for i, (interval_start, interval_end) in enumerate(intervals):
                interval_data.append({
                    'interval_start_shard': str(shard_i),
                    'interval_end_shard': str(shard_i),
                    'interval_start': interval_start,
                    'interval_end': interval_end,
                    'partition_size': interval_end - interval_start,
                })
        interval_data = pd.DataFrame(interval_data)

        hv.Segments(
            data=interval_data,
            kdims=['interval_start', 'interval_start_shard', 'interval_end', 'interval_end_shard'],
        ).opts(
            line_width=10.,
            width=800, height=300,
            ylabel='Shard', xlabel='Row index',
            title='Distribution of rows across shards',
            tools=['hover'],
            invert_yaxis=True,
        )
        """
        N: int = length
        B: int = num_rows
        S: int = num_shards

        if N < S:
            raise ValueError(f'Cannot shard intervals when number of rows ({N}) is less than number of shards ({S})')

        new_P: int = cls.get_closest_npartitions(npartitions=npartitions, num_shards=num_shards)
        if N < new_P:
            raise ValueError(
                f'Cannot shard intervals when number of rows ({N}) is less than the '
                f'desired number of partitions (new_P={new_P}). Please set a smaller number of partitions, '
                f'e.g. npartitions=1 is recommended for <1,000 rows.'
            )

        ## M is the number of "groups" as explained above.
        M: int = 1 + (  ## Correction value
            (N // (S * B))  ## Number of "complete" columns of S*B rows
        ) // (new_P // S)  ## Number of groups we want

        divisions: List[int] = [0]
        num_batches_per_shard: int = 0
        rows_completed: int = 0
        remaining_num_rows: int = N

        num_complete_groups: int = N // (M * S * B)
        for _ in range(num_complete_groups):
            for s_i in range(S):
                rows_completed += M * B
                remaining_num_rows -= M * B
                divisions.append(rows_completed)
                # print(f'[{divisions[-2]}, {divisions[-1]}], remaining={remaining_num_rows}')
            num_batches_per_shard += M
        assert remaining_num_rows < M * S * B
        # assert remaining_num_rows == (N - M*S*B*((new_P//S)-1))

        # print(f'\nRows remaining in last group: {remaining_num_rows} (group size = {M * S * B}):')
        last_group_num_rows: Dict[int, int] = {s_i: 0 for s_i in range(S)}
        num_remaining_complete_columns: int = remaining_num_rows // (S * B)
        if num_remaining_complete_columns > 0:
            for s_i in range(S):
                last_group_num_rows[s_i] += num_remaining_complete_columns * B
                remaining_num_rows -= num_remaining_complete_columns * B
            num_batches_per_shard += num_remaining_complete_columns
        assert remaining_num_rows < S * B

        # print(f'\nRows remaining in last column: {remaining_num_rows} (column size = {S * B}):')
        if drop_last is False:
            ## When drop_last=False, we are typically in the distributed inference paradigm, so we want to ensure all
            ## the N rows are consumed, and we are not super picky about the batch size.
            ## So, we pad each of the "S" partitions in the last group so that the rows in the last column are consumed.

            ## We must distribute < S*B rows among "S" partitions. We do so by a similar logic as the function
            ## _stream_update_num_rows_according_to_num_chunks:
            final_col_num_rows: int = 0
            for s_i in range(S):
                n_rows = cls._stream_update_num_rows_according_to_num_chunks(
                    length=remaining_num_rows,
                    chunks_returned=s_i,
                    num_chunks=S,
                )
                last_group_num_rows[s_i] += n_rows
                final_col_num_rows += n_rows
                ## Note! We do not update num_batches_per_shard, since we are padding the final batch of each shard,
                ## and not changing the number of batches.
            remaining_num_rows -= final_col_num_rows
            assert remaining_num_rows == 0

        for s_i, num_rows in last_group_num_rows.items():
            if num_rows > 0:
                rows_completed += num_rows
                divisions.append(rows_completed)
                # print(f'[{divisions[-2]}, {divisions[-1]}], remaining={remaining_num_rows}')

        if divisions[0] != 0:
            raise ValueError(f'The first division should be 0; created divisions: {divisions}')
        if drop_last is False and divisions[-1] != length:
            raise ValueError(
                f'When drop_last=False, the final division should be length={length}; '
                f'created divisions: {divisions}'
            )
        if len(set(divisions)) != len(divisions) or sorted(divisions) != divisions:
            raise ValueError(
                f'Expected all values in divisions to be unique and sorted; '
                f'created divisions: {divisions}'
            )

        ## Split the divisions into intervals.
        intervals: List[Tuple[int, int]] = []
        for i in range(len(divisions[:-1])):
            # print(f'Partition#{i + 1}: [{divisions[i]}, {divisions[i + 1]}]')
            intervals.append((divisions[i], divisions[i + 1]))

        if len(intervals) % num_shards != 0:
            raise ValueError(
                f'Expected the number of created intervals to be a multiple of num_shards={num_shards}; '
                f'instead {len(intervals)} intervals were created (using length={N}; npartitions={npartitions}; '
                f'num_shards={S}; num_rows={B}; drop_last={drop_last}). '
                f'Consider using a smaller number of partitions. Typically, you should set `npartitions` such that '
                f'each partition has at least 1,000 rows.\nCreated intervals: {intervals}'
            )

        interval_lengths: np.ndarray = np.array([
            interval_end - interval_start
            for (interval_start, interval_end) in intervals[:-num_shards]
        ])
        ## Except for the last "S" intervals, all intervals should have the same length.
        if len(interval_lengths) > 0 and not np.all(interval_lengths == interval_lengths[0]):
            interval_lengths_str: str = ", ".join([
                f'(idx={i}, start={interval_start}, end={interval_end}, len={interval_end - interval_start})'
                for i, (interval_start, interval_end) in enumerate(intervals[:-num_shards])
            ])
            raise ValueError(
                f'Expected all intervals (except last num_shards={num_shards}) to have the same length;'
                f'found intervals of unequal lengths:\n{interval_lengths_str}'
            )
        return divisions, intervals, num_batches_per_shard

    @classmethod
    def _stream_get_executor(
            cls,
            map: Callable,
            parallelize: Parallelize,
            num_workers: conint(ge=1),
            map_executor: Literal['spawn'],
    ) -> Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]]:
        if map is not None:
            if parallelize is Parallelize.sync:
                return None
            if parallelize is Parallelize.processes:
                assert map_executor == 'spawn'
                return ProcessPoolExecutor(max_workers=num_workers)
            elif parallelize is Parallelize.threads:
                assert map_executor == 'spawn'
                return ThreadPoolExecutor(max_workers=num_workers)
            elif parallelize is Parallelize.ray:
                return None
            else:
                raise NotImplementedError(f'Unsupported: you cannot stream data with {parallelize} parallelization.')
        return None

    @classmethod
    def _stream_cleanup_executor(
            cls,
            executor: Union[ThreadPoolExecutor, ProcessPoolExecutor],
            map_executor: Literal['spawn'] = 'spawn',
    ):
        if map_executor == 'spawn':
            del executor

    @classmethod
    def _stream_loop_enqueue_dequeue_map_chunk(
            cls,
            out_sdf_chunk: ScalableDataFrame,
            mapped_sdf_chunks: Deque[Dict[str, Union[int, Future]]],
            chunks_returned: int,
            map: Optional[Callable],
            map_kwargs: Dict,
            parallelize: Parallelize,
            num_workers: int,
            map_failure: Literal['raise', 'drop'],
            stream_as: Optional[DataLayout],
            raw: bool,
            executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]],
    ) -> Optional[ScalableDataFrame]:
        ## Push the chunk onto the queue for processing. At the same time, dequeue the head chunk if it has
        ## finished processing, or if the queue length has been exceeded. If neither of these conditions is
        ## met, we will not dequeue anything, i.e. None is returned
        if map is None:
            ## Do not preprocess
            return cls._stream_chunk_to_raw(out_sdf_chunk, raw=raw)
        else:
            cls._stream_enqueue_chunk_for_mapping(
                out_sdf_chunk=out_sdf_chunk,
                mapped_sdf_chunks=mapped_sdf_chunks,
                chunks_returned=chunks_returned,
                map=map,
                map_kwargs=map_kwargs,
                map_failure=map_failure,
                parallelize=parallelize,
                executor=executor,
            )
            if len(mapped_sdf_chunks) > num_workers or \
                    (len(mapped_sdf_chunks) > 0 and is_done(mapped_sdf_chunks[0]['future'])):
                out_sdf_chunk: Optional[ScalableDataFrame] = cls._stream_dequeue_mapped_chunk(
                    mapped_sdf_chunks=mapped_sdf_chunks,
                    stream_as=stream_as,
                    map_failure=map_failure,
                )
                if out_sdf_chunk is not None:
                    return cls._stream_chunk_to_raw(out_sdf_chunk, raw=raw)
        return None

    @classmethod
    def _stream_enqueue_chunk_for_mapping(
            cls,
            out_sdf_chunk: ScalableDataFrame,
            mapped_sdf_chunks: Deque[Dict[str, Union[int, Future]]],
            chunks_returned: int,
            map: Callable,
            map_kwargs: Dict,
            map_failure: Literal['raise', 'drop'],
            parallelize: Parallelize,
            executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]]
    ):
        ## Adds the chunk to the queue for processing
        try:
            item = dispatch(
                map,
                out_sdf_chunk,
                parallelize=parallelize,
                executor=executor,
                **map_kwargs
            )
            mapped_sdf_chunks.append({
                'i': chunks_returned,
                'future': item,
            })
        except Exception as e:
            if map_failure == 'raise':
                raise e
            assert map_failure == 'drop'
            return

    @classmethod
    def _stream_dequeue_mapped_chunk(
            cls,
            mapped_sdf_chunks: Deque[Dict[str, Union[int, Future]]],
            stream_as: Optional[DataLayout],
            map_failure: Literal['raise', 'drop'],
    ) -> Optional[ScalableDataFrame]:
        f = mapped_sdf_chunks.popleft()['future']
        try:
            mapped_sdf = accumulate(f)
        except Exception as e:
            if map_failure == 'raise':
                raise e
            assert map_failure == 'drop'
            return None
        finally:
            del f  ## Without this line, the inputs to the future, i.e. the chunk, are persisted in memory.
        return mapped_sdf

    @classmethod
    def _stream_chunk_to_raw(cls, out_sdf_chunk: ScalableDataFrame, raw: bool) -> ScalableDataFrameOrRaw:
        if raw:
            return out_sdf_chunk.raw()
        else:
            return out_sdf_chunk

    @classmethod
    def _stream_update_num_rows_according_to_num_chunks(cls, length: int, chunks_returned: int, num_chunks: int) -> int:
        ## This function is used when num_chunks is not None.
        ## When we have N=len(df) rows and want to break them into M chunks, to ensure 
        ## the chunks have a similar number of rows, we should make it such that:
        ## (a) N%M chunks will have math.ceil(N/M) rows, and
        ## (b) M - N%M chunks will have math.floor(N/M) rows.
        if chunks_returned < length % num_chunks:
            num_rows: int = math.ceil(length / num_chunks)
        else:
            num_rows: int = math.floor(length / num_chunks)
        return num_rows

    """
    ==============================================================================================
    Implement the Pandas DataFrame API v1.4.2: https://pandas.pydata.org/docs/reference/frame.html
    ==============================================================================================
    """

    _NOT_IMPLEMENTED_INSTANCE_PROPERTIES: Set[str] = {
        ## Attributes and underlying data
        'index', 'dtypes', 'values', 'axes', 'size',
        ## Indexing, iteration
        'at', 'iat',
        ## Reshaping, sorting, transposing
        'T',
        ## Metadata
        'attrs',
        ## Plotting
        'plot',
        ## Sparse accessor
        'sparse',
    }

    _NOT_IMPLEMENTED_INSTANCE_METHODS: Set[str] = {
        ## Attributes and underlying data:
        'info', 'select_dtypes', 'memory_usage', 'set_flags',
        ## Conversion
        'astype', 'convert_dtypes', 'infer_objects', 'bool',
        ## Indexing, iteration
        'insert', '__iter__', 'items', 'iteritems', 'keys', 'iterrows', 'itertuples', 'lookup', 'pop', 'xs', 'get',
        'where', 'mask',
        ## Binary operator functions
        'add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'mod', 'pow', 'dot', 'radd', 'rsub', 'rmul', 'rdiv',
        'rtruediv', 'rfloordiv', 'rmod', 'rpow', 'lt', 'gt', 'le', 'ge', 'ne', 'eq', 'combine', 'combine_first',
        ## Function application, GroupBy & window
        'pipe', 'transform', 'rolling', 'expanding', 'ewm',
        ## Computations / descriptive stats
        'corr', 'corrwith', 'count', 'cov', 'cummax', 'cummin', 'cumprod', 'cumsum', 'describe', 'diff', 'eval', 'kurt',
        'kurtosis', 'mad', 'pct_change', 'prod', 'product', 'quantile', 'rank', 'sem', 'skew', 'nunique',
        'value_counts',
        ## Reindexing / selection / label manipulation
        'add_prefix', 'add_suffix', 'align', 'at_time', 'between_time', 'filter', 'first', 'last',
        'set_axis', 'take', 'truncate',
        ## Missing data handling
        'backfill', 'bfill', 'ffill', 'interpolate', 'pad',
        ## Reshaping, sorting, transposing
        'droplevel', 'pivot', 'pivot_table', 'reorder_levels', 'sort_index', 'nlargest', 'nsmallest',
        'swaplevel', 'stack', 'unstack', 'swapaxes', 'melt', 'explode', 'squeeze', 'to_xarray', 'transpose',
        ## Combining / comparing / joining / merging
        'compare', 'join', 'update',
        ## Time Series-related
        'asfreq', 'asof', 'shift', 'slice_shift', 'tshift', 'first_valid_index', 'last_valid_index', 'resample',
        'to_period', 'to_timestamp', 'tz_convert', 'tz_localize',
        ## Flags
        'Flags',
        ## Serialization / IO / conversion:
        'from_dict', 'from_records',
        'to_pickle', 'to_hdf', 'to_sql', 'to_dict', 'to_excel', 'to_html', 'to_feather', 'to_latex', 'to_stata',
        'to_gbq', 'to_records', 'to_string', 'to_clipboard', 'to_markdown',
    }

    _NOT_IMPLEMENTED_REASONS: Dict[str, str] = {
        'append': 'as it is deprecated in newer versions of Pandas',
    }

    _ALTERNATIVE: Dict[str, str] = {
        'values': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'ndim': '.shape',
        'at': '.loc or .iloc',
        'iat': '.loc or .iloc',
        'insert': '.loc or .iloc',
        'keys': '.columns',
        'get': '.loc or .iloc',
        'join': '.merge',
        'sparse': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'plot': f'.{RAW_DATA_MEMBER} to access the raw data for plotting',

        ## Computations / descriptive stats
        'abs': 'the corresponding method on the Series',
        'all': 'the corresponding method on the Series',
        'any': 'the corresponding method on the Series',
        'clip': 'the corresponding method on the Series',
        'corr': 'the corresponding method on the Series',
        'corrwith': 'the corresponding method on the Series',
        'count': 'the corresponding method on the Series',
        'cov': 'the corresponding method on the Series',
        'cummax': 'the corresponding method on the Series',
        'cummin': 'the corresponding method on the Series',
        'cumprod': 'the corresponding method on the Series',
        'cumsum': 'the corresponding method on the Series',
        'describe': 'the corresponding method on the Series',
        'diff': 'the corresponding method on the Series',
        'eval': 'the corresponding method on the Series',
        'kurt': 'the corresponding method on the Series',
        'kurtosis': 'the corresponding method on the Series',
        'mad': 'the corresponding method on the Series',
        'max': 'the corresponding method on the Series',
        'mean': 'the corresponding method on the Series',
        'median': 'the corresponding method on the Series',
        'min': 'the corresponding method on the Series',
        'mode': 'the corresponding method on the Series',
        'pct_change': 'the corresponding method on the Series',
        'prod': 'the corresponding method on the Series',
        'product': 'the corresponding method on the Series',
        'quantile': 'the corresponding method on the Series',
        'rank': 'the corresponding method on the Series',
        'round': 'the corresponding method on the Series',
        'sem': 'the corresponding method on the Series',
        'skew': 'the corresponding method on the Series',
        'sum': 'the corresponding method on the Series',
        'std': 'the corresponding method on the Series',
        'var': 'the corresponding method on the Series',
        'nunique': 'the corresponding method on the Series',
        'value_counts': 'the corresponding method on the Series',

        ## Combining / comparing / joining / merging
        'append': f'ScalableDataFrame.concat',

        ## Serialization / IO / conversion:
        'from_dict': f'.create with layout="{DataLayout.DICT}" or layout="{DataLayout.LIST_OF_DICT}"',
        'from_records': f'.create with layout="{DataLayout.NUMPY_RECORD_ARRAY}"',
        'to_pickle': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'to_hdf': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'to_sql': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'to_excel': 'CsvDataFrameWriter or TsvDataFrameWriter',  ## TODO: add ExcelDataFrameWriter
        'to_html': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'to_feather': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'to_latex': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'to_stata': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'to_gbq': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'to_records': f'as_numpy',
        'to_string': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'to_clipboard': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'to_markdown': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
    }

    def __getattr__(self, attr_name: str):
        if attr_name in self._NOT_IMPLEMENTED_INSTANCE_PROPERTIES:
            raise self._get_attribute_not_implemented_error(
                attr_name,
                is_property=True,
                reason=self._NOT_IMPLEMENTED_REASONS.get(attr_name),
                alternative=self._ALTERNATIVE.get(attr_name)
            )
        elif attr_name in self._NOT_IMPLEMENTED_INSTANCE_METHODS:
            raise self._get_attribute_not_implemented_error(
                attr_name,
                is_property=False,
                reason=self._NOT_IMPLEMENTED_REASONS.get(attr_name),
                alternative=self._ALTERNATIVE.get(attr_name)
            )
        ## Forwards calls to the respective method of the data class.
        try:
            data = self.__dict__[RAW_DATA_MEMBER]
        except KeyError as e:
            raise AttributeError(
                f'Raw_data field `{RAW_DATA_MEMBER}` does not exist in {self.__class__}.\n'
                f'__dict__ keys:{list(self.__dict__.keys())}'
            )
        if not hasattr(data, attr_name):
            raise AttributeError(
                f'Neither {self.__class__.__name__} nor {type(self._data)} classes have attribute "{attr_name}"'
            )
        return getattr(data, attr_name)

    def _get_attribute_not_implemented_error(
            self,
            attr_name: str,
            is_property: bool,
            reason: Optional[str] = None,
            alternative: Optional[str] = None,
    ) -> AttributeError:
        if is_property:
            fn_type = 'Property'
        else:
            fn_type = 'Method'
        if reason is None:
            reason = f'to maintain compatibility between {self.__class__.__name__} subclasses with different layouts'
        if alternative is not None:
            alternative = f' Please use {alternative} instead.'
        return AttributeError(
            f'{fn_type} .{attr_name} has not been implemented {reason}.{alternative}')

    """
    ---------------------------------------------
    Attributes and underlying data
    ---------------------------------------------
    """

    @property
    @abstractmethod
    def columns(self) -> List:
        """Idempotent operation to return the list of columns in sorted order."""
        pass

    @property
    @abstractmethod
    def columns_set(self) -> Set:
        """Idempotent operation to return the set of columns."""
        pass

    @property
    def ndim(self) -> int:
        return 2

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self), len(self.columns_set))

    @property
    def empty(self) -> bool:
        return len(self) == 0 and len(self.columns_set) == 0

    @abstractmethod
    def __len__(self):
        """
        Should return the number of rows in the "DataFrame".
        :return:
        """
        pass

    @abstractmethod
    def __str__(self):
        """
        Gets a string representation of the "DataFrame".
        :return:
        """
        pass

    def __repr__(self):
        return str(self)

    """
    ---------------------------------------------
    Indexing, iteration
    ---------------------------------------------
    """

    def head(self, n: int) -> ScalableDataFrame:
        return self.iloc[:n]

    def tail(self, n: int) -> ScalableDataFrame:
        return self.iloc[:-n]

    def __getitem__(self, key: Any):
        raise NotImplementedError(
            f'We do not allow getting values from {ScalableDataFrame} instances directly; use .loc or .iloc instead.'
        )

    def __setitem__(self, key: Any, value: Any):
        raise NotImplementedError(
            f'We do not allow setting values on {ScalableDataFrame} instances directly; use .loc or .iloc instead.'
        )

    @property
    @abstractmethod
    def loc(self) -> Any:
        pass

    def items(self) -> Generator[Tuple[str, ScalableSeries], None, None]:
        for col_name, col_ss in self._sorted_items_dict().items():
            yield col_name, col_ss

    @abstractmethod
    def _sorted_items_dict(self) -> Dict[str, ScalableSeries]:
        pass

    """
    ---------------------------------------------
    Function application, GroupBy & window
    ---------------------------------------------
    """

    @abstractmethod
    def apply(self, func, axis=1, args=(), **kwargs):
        pass

    @abstractmethod
    def applymap(self, func, na_action=None, **kwargs) -> ScalableDataFrame:
        pass

    @abstractmethod
    def agg(self, func=None, axis=0, *args, **kwargs):
        pass

    def aggregate(self, *args, **kwargs):
        return self.agg(*args, **kwargs)

    @abstractmethod
    def groupby(self, by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=False,
                observed=False, dropna=True):
        pass

    """
    ---------------------------------------------
    Computations / descriptive stats
    ---------------------------------------------
    """

    @abstractmethod
    def all(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        pass

    """
    ---------------------------------------------
    Reindexing / selection / label manipulation
    ---------------------------------------------
    """

    @abstractmethod
    def drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'):
        pass

    @abstractmethod
    def drop_duplicates(self, subset=None, keep='first', inplace=False, ignore_index=False):
        pass

    @abstractmethod
    def duplicated(self, subset=None, keep='first'):
        pass

    @abstractmethod
    def equals(self, other):
        pass

    @abstractmethod
    def idxmax(self, axis=0, skipna=True):
        pass

    @abstractmethod
    def idxmin(self, axis=0, skipna=True):
        pass

    @abstractmethod
    @safe_validate_arguments
    def rename(
            self,
            mapper: Optional[Union[Dict, Callable]] = None,
            *,
            index: Optional[Union[Dict, Callable]] = None,
            columns: Optional[Union[Dict, Callable]] = None,
            axis: Literal[1, 'columns'] = 1,
            copy: bool = True,
            inplace: bool = False,
            level: Optional[Union[int, str]] = None,
            errors: Literal['ignore', 'raise'] = 'ignore',
    ) -> Optional[ScalableDataFrame]:
        pass

    @abstractmethod
    def rename_axis(self, mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False):
        pass

    @abstractmethod
    def sample(self, n=None, frac=None, replace=False, weights=None, random_state=None, axis=None,
               ignore_index=False) -> ScalableDataFrame:
        pass

    """
    ---------------------------------------------
    Missing data handling
    ---------------------------------------------
    """

    @abstractmethod
    def dropna(self, axis=0, how='any', thresh=None, subset=None, inplace=False):
        pass

    @abstractmethod
    def fillna(self, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None):
        pass

    @abstractmethod
    def isna(self):
        pass

    def isnull(self):
        return self.isna()

    @abstractmethod
    def notna(self):
        pass

    def notnull(self):
        return self.notna()

    @abstractmethod
    def replace(self, to_replace=None, value=None, inplace=False, limit=None, regex=False, method=None):
        pass

    """
    ---------------------------------------------
    Combining / comparing / joining / merging
    ---------------------------------------------
    """

    @abstractmethod
    def assign(self, **kwargs):
        pass

    @abstractmethod
    def merge(self, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False,
              sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None) -> ScalableDataFrame:
        pass

    @classmethod
    def concat(
            cls,
            sdfs: List[ScalableDataFrameOrRaw],
            axis: Literal[0, 'index'] = 0,
            layout: Optional[DataLayout] = None,
            reset_index: bool = True,
    ) -> ScalableDataFrame:
        sdfs: List[ScalableDataFrame] = [ScalableDataFrame.of(sdf) for sdf in as_list(sdfs)]
        if axis not in {0, 'index'}:
            raise AttributeError(
                f'{cls} only supports concatenating row-wise (i.e. with axis=0). '
                f'Please use .merge(...) for column-wise  concatenation.'
            )
        if layout is None and len({type(sdf) for sdf in sdfs}) > 1:
            raise ValueError(
                f'When concatenating multiple {cls} instances with different data layouts, '
                f'you must specify the resultant data layout by passing the `layout=...` parameter.'
            )
        if layout is not None:
            sdfs: List[ScalableDataFrame] = [cls.of(sdf, layout=layout) for sdf in sdfs]
        else:
            layout: DataLayout = sdfs[0].layout
        ## All dfs now have the same layout:
        ScalableDataFrameSubclass: Type[ScalableDataFrame] = cls.get_subclass(layout)
        return ScalableDataFrameSubclass._concat_sdfs(sdfs, reset_index=reset_index)

    @classmethod
    @abstractmethod
    def _concat_sdfs(cls, sdfs: List[ScalableDataFrame], reset_index: bool) -> ScalableDataFrame:
        pass

    """
    ---------------------------------------------
    Serialization / IO / conversion
    ---------------------------------------------
    """

    @abstractmethod
    def copy(self, deep: bool = True) -> ScalableDataFrame:
        pass

    def raw(self, **kwargs) -> Any:
        """Alias for .as_raw()"""
        return self.as_raw(**kwargs)

    def to_raw(self, **kwargs) -> Any:
        """Alias for .as_raw()"""
        return self.as_raw(**kwargs)

    def as_raw(self, **kwargs) -> Any:
        return self._data

    def dict(self, **kwargs) -> Dict[str, Union[List, np.ndarray, Any]]:
        """Alias for .as_dict()"""
        return self.as_dict(**kwargs)

    def to_dict(self, **kwargs) -> Dict[str, Union[List, np.ndarray, Any]]:
        """Alias for .as_dict()"""
        return self.as_dict(**kwargs)

    def as_dict(
            self,
            col_type: Optional[Literal['list', 'numpy', list, np.ndarray, 'record']] = None,
            **kwargs
    ) -> Dict[str, Union[List, np.ndarray, Any]]:
        if col_type in {'record'}:
            if len(self) > 1:
                raise ValueError(f'Cannot convert DataFrame of length {len(self)} into record')
            return {col: self.iloc[0, col] for col in self.columns_set}
        if col_type is None:
            return {col: self[col].raw(**kwargs) for col in self.columns_set}
        if col_type in {'numpy', np.ndarray}:
            return {col: self[col].numpy(**kwargs) for col in self.columns_set}
        if col_type in {'list', list}:
            return {col: self[col].to_list(**kwargs) for col in self.columns_set}
        raise NotImplementedError(f'Unsupported `col_type`: {col_type}')

    def list_of_dict(self, **kwargs) -> List[Dict]:
        """Alias for .as_list_of_dict()"""
        return self.as_list_of_dict(**kwargs)

    def to_list_of_dict(self, **kwargs) -> List[Dict]:
        """Alias for .as_list_of_dict()"""
        return self.as_list_of_dict(**kwargs)

    def as_list_of_dict(self, **kwargs) -> List[Dict]:
        ## This method was many times faster when we had a large number of columns.
        df: PandasDataFrame = self.pandas(**kwargs)
        df_dict: Dict = {col: df[col].values for col in df.columns}
        cols = set(df_dict.keys())
        out: List[Dict] = []
        for i in range(len(df)):
            out.append({col: df_dict[col][i] for col in cols})
        return out

    def record(self, **kwargs) -> Dict:
        """Alias for .as_record()"""
        return self.as_record(**kwargs)

    def to_record(self, **kwargs) -> Dict:
        """Alias for .as_record()"""
        return self.as_record(**kwargs)

    def as_record(self, **kwargs) -> Dict:
        if len(self) != 1:
            raise ValueError(
                f'Can only convert to {DataLayout.RECORD} when we have exactly 1 row; '
                f'found {len(self)} rows.'
            )
        return self.as_list_of_dict(**kwargs)[0]

    def numpy(self, **kwargs) -> np.recarray:
        """Alias for .as_numpy()"""
        return self.as_numpy(**kwargs)

    def to_numpy(self, **kwargs) -> np.recarray:
        """Alias for .as_numpy()"""
        return self.as_numpy(**kwargs)

    def as_numpy(self, **kwargs) -> np.recarray:
        return self.pandas(**kwargs)[self.columns].to_records(index=False)

    def pandas(self, **kwargs) -> PandasDataFrame:
        """Alias for .as_pandas()"""
        return self.as_pandas(**kwargs)

    def to_pandas(self, **kwargs) -> PandasDataFrame:
        """Alias for .as_pandas()"""
        return self.as_pandas(**kwargs)

    @abstractmethod
    def as_pandas(self, **kwargs) -> PandasDataFrame:
        pass

    def dask(self, **kwargs) -> DaskDataFrame:
        """Alias for .as_dask()"""
        return self.as_dask(**kwargs)

    def to_dask(self, **kwargs) -> PandasDataFrame:
        """Alias for .as_dask()"""
        return self.as_dask(**kwargs)

    def as_dask(self, **kwargs) -> DaskDataFrame:
        if 'npartitions' not in kwargs and 'chunksize' not in kwargs:
            kwargs['npartitions'] = 1  ## Create a dask dataframe with a single partition.
        return dd.from_pandas(self.pandas(), **kwargs)

    def is_lazy(self) -> bool:
        return False

    def persist(self, **kwargs) -> ScalableDataFrame:
        """For lazily-evaluated DataFrames, stores the task graph up to the current DataFrame."""
        return self

    def compute(self, **kwargs) -> ScalableDataFrame:
        """For lazily-evaluated DataFrames, runs the task graph up to the current DataFrame."""
        return self

    @property
    def npartitions(self) -> int:
        """For distributed DataFrames, this gets the number of data partitions."""
        return 1

    def repartition(self, **kwargs) -> ScalableDataFrame:
        """Creates a new ScalableDataFrame with different partition boundaries."""
        return self

    def to_parquet(self, path, **kwargs):
        return self.pandas().to_parquet(path, **kwargs)

    def to_csv(self, path, **kwargs):
        return self.pandas().to_csv(path, **kwargs)

    def to_json(self, path, **kwargs):
        return self.pandas().to_json(path, **kwargs)

    def to_npz(self, path, storage, **kwargs):
        pandas_df = self.pandas()
        return np.savez(path, **{col: pandas_df[col].values for col in pandas_df.columns}, **kwargs)


## Ref: https://stackoverflow.com/a/15920132/4900327
## This deletes abstractmethods from ScalableDataFrame, which means that we can instantiate classes
## like PandasScalableDataFrame/DaskScalableDataFrame even if they have not implemented all these methods.
## This is done because these classes forward to the respective "raw" dataframe (i.e. PandasDataFrame, DaskDataFrame)
## when an implementation is not found on either ScalableDataFrame or PandasScalableDataFrame/DaskScalableDataFrame.
## At the same time, we only need to override __getattr__, not __getattribute__, which is faster during execution.
## Additionally, typing in PyCharm still works as expected.
[delattr(ScalableDataFrame, abs_method_name) for abs_method_name in ScalableDataFrame.__abstractmethods__]

ScalableDataFrameOrRaw = Union[ScalableDataFrame, ScalableDataFrameRawType]
ScalableOrRaw = Union[ScalableSeriesOrRaw, ScalableDataFrameOrRaw]

to_sdf: Callable = ScalableDataFrame.of
to_ss: Callable = ScalableSeries.of


class CompressedScalableDataFrame(Parameters):
    payload: Union[str, bytes]
    compression_engine: CompressionEngine
    layout: DataLayout
    base64_encoding: bool = False

    @root_validator(pre=False)
    def _set_params(cls, params: Dict) -> Dict:
        if params['base64_encoding'] is False and not isinstance(params['payload'], bytes):
            raise ValueError(
                f"Must pass a bytes `payload` when passing `base64_encoding=False`; "
                f"found {type(params['payload'])}"
            )
        elif params['base64_encoding'] is True and not isinstance(params['payload'], str):
            raise ValueError(
                f"Must pass a string `payload` when passing `base64_encoding=True`; "
                f"found {type(params['payload'])}"
            )
        return params

    @safe_validate_arguments
    def decompress(
            self,
            *,
            layout: Optional[DataLayout] = None,
            **kwargs,
    ) -> ScalableDataFrame:
        eng: CompressionEngine = CompressionEngine.from_str(self.compression_engine)
        layout: DataLayout = get_default(layout, self.layout)
        if eng is CompressionEngine.BROTLI:
            with optional_dependency('brotli', error='ignore'):
                import brotli
                payload: Union[bytes, str] = self.payload
                if self.base64_encoding is True:
                    payload: bytes = base64.urlsafe_b64decode(payload.encode('utf-8'))
                return ScalableDataFrame.of(
                    json.loads(brotli.decompress(payload).decode("utf8")),
                    layout=layout,
                    **kwargs,
                )
        elif eng is CompressionEngine.GZIP:
            payload: Union[bytes, str] = self.payload
            if self.base64_encoding is True:
                payload: bytes = base64.urlsafe_b64decode(payload.encode('utf-8'))
            return ScalableDataFrame.of(
                json.loads(gzip.decompress(payload).decode("utf8")),
                layout=layout,
                **kwargs,
            )
        raise NotImplementedError(f'Unsupported compression engine: "{eng}"')
