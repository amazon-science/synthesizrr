from typing import *
import math, io, ray, numpy as np, pandas as pd
from concurrent.futures._base import Future
from pandas.core.frame import Series as PandasSeries, DataFrame as PandasDataFrame
import dask.dataframe as dd
from dask.dataframe.core import Scalar as DaskScalar, Series as DaskSeries, DataFrame as DaskDataFrame
from synthergent.base.util import multiple_are_not_none, all_are_none, is_function, wrap_fn_output, \
    get_default, RayDaskPersistWaitCallback, get_current_fn_name, accumulate, safe_validate_arguments, Log, Executor
from synthergent.base.constants import DataLayout, Parallelize
from synthergent.base.data.sdf.ScalableSeries import ScalableSeries
from synthergent.base.data.sdf.ScalableDataFrame import ScalableDataFrame, ScalableDataFrameOrRaw, is_scalable, \
    DataFrameShardingError
from synthergent.base.data.sdf.DaskScalableSeries import DaskScalableSeries
from pydantic import validate_arguments, conint, constr
from synthergent.base.util import accumulate, run_concurrent
from pydantic.typing import Literal
from collections import deque

DaskScalableDataFrame = "DaskScalableDataFrame"


class DaskScalableDataFrame(ScalableDataFrame):
    layout = DataLayout.DASK
    layout_validator = ScalableDataFrame.is_dask
    ScalableSeriesClass = DaskScalableSeries

    def __init__(self, data: Union[DaskDataFrame, ScalableDataFrame], name: Optional[str] = None, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        if isinstance(data, ScalableDataFrame):
            data: DaskDataFrame = data.to_dask(**kwargs)
        self.layout_validator(data)
        self._data: DaskDataFrame = data
        if name is not None and not isinstance(name, (str, int, float)):
            raise ValueError(
                f'`name` used to construct {self.__class__} can only be int, str or float; '
                f'found object of type: {type(name)} with value: {name}'
            )
        self._name: Optional[str] = name

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self), len(self.columns))

    @property
    def columns(self) -> List:
        return list(self._data.columns)

    @property
    def columns_set(self) -> Set:
        return set(self._data.columns)

    def __len__(self):
        return self._data.shape[0].compute()

    def __str__(self):
        columns: List[str] = self.columns
        return f"Dask DataFrame with {len(columns)} column(s): {columns}:\n{str(self._data)}"

    @property
    def loc(self) -> Any:
        return self._data.loc

    def _sorted_items_dict(self) -> Dict[str, DaskScalableSeries]:
        return {col: self.ScalableSeriesClass(self._data[col], name=col) for col in sorted(self.columns)}

    @classmethod
    def _compute_dask_scalar(cls, scalar: DaskScalar) -> Any:
        return scalar.compute()

    @classmethod
    def _save_dask_to_npz(self, df: DaskDataFrame, partition: int, dir_path: str, name_fn: Callable):
        pandas_df = df.compute()
        np.savez(dir_path + "/" + name_fn(partition), **{col: pandas_df[col].values for col in pandas_df.columns})

    @classmethod
    def _to_scalable(cls, data: Any) -> Union[ScalableDataFrame, ScalableSeries, Any]:
        return DaskScalableSeries._to_scalable(data)

    def __getattr__(self, attr_name: str):
        """Forwards calls to the respective method of Dask Series class."""
        out = super().__getattr__(attr_name)
        if is_function(out):
            return wrap_fn_output(out, wrapper_fn=self._to_scalable)
        return self._to_scalable(out)

    def __getitem__(self, key: Any):
        return self._to_scalable(self._data[key])

    def __setitem__(self, key: Any, value: Any):
        if is_scalable(value):
            if value.layout is not DataLayout.DASK:
                raise ValueError(f'Can only set using {DataLayout.DASK} DataFrame and Series')
            value: Union[DaskSeries, DaskDataFrame] = value._data
        self._data[key] = value
        # raise NotImplementedError(f'Cannot set at the moment')

    def apply(self, func, *, axis=1, args=(), **kwargs):
        return self._data.map_partitions(
            lambda df_part: df_part.apply(func, axis=axis, raw=False, result_type=None, args=args, **kwargs)
        )

    @classmethod
    def _concat_sdfs(cls, sdfs: List[ScalableDataFrame], reset_index: bool) -> DaskScalableDataFrame:
        ddf_list: List[DaskDataFrame] = []
        for sdf in sdfs:
            assert isinstance(sdf, DaskScalableDataFrame)
            ddf_list.append(sdf._data)
        df = dd.concat(ddf_list)
        if reset_index:
            df = df.reset_index(drop=True)
        return cls.of(df, layout=cls.layout)

    def copy(self, deep: bool = False) -> ScalableDataFrame:
        ## Dask does not allow deep copies.
        return self._constructor(self._data.copy(deep=False))

    def as_pandas(self, **kwargs) -> PandasDataFrame:
        return self._data.compute(**kwargs)

    def as_dask(self, **kwargs) -> DaskDataFrame:
        return self._data

    def is_lazy(self) -> bool:
        return True

    def persist(self, wait: bool = False, **kwargs) -> ScalableDataFrame:
        """
        Submits execution of the Dask task graph up to the current DataFrame.
        :param wait: whether to block the main thread until the .persist() execution completes.
        :param kwargs: additional kwargs forwarded to Dask DataFrame's .persist() method.
        :return: ScalableDataFrame.
        """
        if not isinstance(wait, bool):
            raise ValueError(f'Attribute `wait` must be a boolean, found value of type {type(wait)}')
        if wait:
            with RayDaskPersistWaitCallback():
                self._data = self._data.persist(**kwargs)
        else:
            self._data = self._data.persist(**kwargs)
        return self

    def compute(self, **kwargs) -> ScalableDataFrame:
        """For lazily-evaluated DataFrames, runs the task graph up to the current DataFrame."""
        return self.of(self._data.compute(**kwargs))

    @property
    def npartitions(self) -> int:
        """For distributed DataFrames, this gets the number of data partitions."""
        return self._data.npartitions

    def repartition(
            self,
            nrows: Optional[int] = None,
            npartitions: Optional[int] = None,
            partition_size: Optional[int] = None,
            **kwargs
    ) -> ScalableDataFrame:
        """
        Creates a new DaskScalableDataFrame with different partition boundaries.
        We augment the Dask implementation to allow repartitioning by `nrows`.
        """
        nrows: int = get_default(nrows, kwargs.get('batch_size'), kwargs.get('num_rows'))
        if multiple_are_not_none(nrows, npartitions, partition_size):
            raise ValueError(f'Only one of the following can be non-None: `nrows`, `npartitions`, `partition_size`')
        if nrows is not None:
            ## This slightly violates our desire to have all chunks (except the last) have exactly `nrows`.
            ## However, this does ensure that all chunks have a similar number of rows, all of which are <=num_rows
            ## We do not use divisions since we cannot ensure the index is sorted:
            ## https://docs.dask.org/en/latest/dataframe-design.html#partitions
            npartitions: int = math.ceil(len(self) / nrows)
        self._data = self._data.repartition(
            npartitions=npartitions,
            partition_size=partition_size,
            **kwargs
        )
        return self

    def to_parquet(self, path: Union[io.IOBase, str], **kwargs):
        if isinstance(path, io.IOBase):
            ## Patch the Dask writer to write to streams:
            return self._data.compute().to_parquet(path, **kwargs)
        return self._data.to_parquet(path, **kwargs)

    def to_npz(self, path: Union[io.IOBase, str], **kwargs):
        if isinstance(path, io.IOBase):
            ## Patch the Dask writer to write to streams:
            return self._data.compute().to_npz(path, **kwargs)
        accumulate([
            run_concurrent(self._save_dask_to_npz, self._data.get_partition(partition), partition, path,
                           kwargs["name_function"]) for
            partition in range(self._data.npartitions)
        ])

    def to_csv(self, path: Union[io.IOBase, str], **kwargs):
        if isinstance(path, io.IOBase):
            ## Patch the Dask writer to write to streams:
            return self._data.compute().to_csv(path, **kwargs)
        return self._data.to_csv(path, **kwargs)

    def to_json(self, path: Union[io.IOBase, str], **kwargs):
        if isinstance(path, io.IOBase):
            ## Patch the Dask writer to write to streams:
            return self._data.compute().to_json(path, **kwargs)
        return self._data.to_json(path, **kwargs)

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
            fetch_partitions: conint(ge=0) = 1,
            shard: Tuple[conint(ge=0), conint(ge=1)] = (0, 1),
            reverse_sharding: bool = False,
            drop_last: Optional[bool] = None,
            **kwargs,
    ) -> Generator[ScalableDataFrameOrRaw, None, None]:
        def ilen(sdf_list: List[ScalableDataFrame]) -> int:
            ## Add up length of sdfs in the list. These should be all in-memory DFs (e.g. Pandas, Dict, ListOfDict),
            ## not lazy (like Dask) so this function should run in the order of microseconds.
            return sum([len(sdf) for sdf in sdf_list])

        try:
            shard_rank, num_shards = shard
            length: Optional[int] = None
            df_partitions: Deque[PandasDataFrame] = deque()
            sdfs: List[ScalableDataFrame] = []
            mapped_sdf_chunks: Deque[Dict[str, Union[int, Future]]] = deque()
            executor: Optional[Executor] = self._stream_get_executor(
                map=map,
                parallelize=parallelize,
                num_workers=num_workers,
                map_executor=map_executor,
            )
            num_batches_per_shard: Optional[int] = None

            if drop_last is not None:
                if num_rows is None:
                    raise AttributeError(
                        f'Can only run balanced sharding (i.e. using drop_last={drop_last}) on a DaskDataFrame using '
                        f'`num_rows`; however, `num_rows` was found to be None.'
                    )
                _, num_batches_per_shard = self.set_shard_divisions(
                    num_shards=num_shards,
                    num_rows=num_rows,
                    inplace=True,
                )
                self.persist(wait=True)
            elif drop_last is None and num_shards > 1:
                ## We have multiple shards, but we don't care if they are exactly the equal size.
                new_P: int = self.get_closest_npartitions(
                    npartitions=self.npartitions,
                    num_shards=num_shards
                )
                if new_P != self.npartitions:
                    self.repartition(npartitions=new_P)
                    self.persist(wait=True)
            ## Get the shuffled & sharded list of partitions from which we should stream batches.
            ## Here, each "partition" is a DaskDataFrame partition.
            partition_idxs: np.ndarray = self._stream_get_sharded_partition_idxs(
                npartitions=self._data.npartitions,
                shuffle=shuffle,
                seed=seed,
                shard=shard,
                reverse_sharding=reverse_sharding,
            )
            partition_idxs_iter: Iterator[int] = iter(partition_idxs)

            ## Here, we maintain two queues:
            ## (a) `df_partitions`, which is used to fetch PARTITIONS from the Dask DF, and converts them to Pandas.
            ## This is a thread-pool queue, since fetching partitions is IO-bound. A partition is typically of a large
            ## size, e.g. 1MM rows, as it represents a unit of data which must is sent to a different machine for
            ## processing. We fetch partitions according to partition_idxs, which is shuffled and sharded.
            ## (b) `mapped_sdf_chunks`: this stores and processes CHUNKS, after they are selected from the Pandas DF
            ## and converted to the correct layout. After processing completes, we yield the processed chunk. A chunk is
            ## equivalent to a batch in ML workloads, e.g. 1024 rows, and typically much smaller than a partition.
            ## Depending on the condition (num_rows!=None or num_chunks!=None), in order to create each chunk, we might
            ## need to select rows chunks which are across multiple Dask partitions, so we must be careful when
            ## selecting rows to include in each chunk.
            partition_idx: Optional[int] = next(partition_idxs_iter, None)
            fetch_partitions: int = min(fetch_partitions, self._data.npartitions - 1)
            chunks_returned: int = 0
            while partition_idx is not None or len(df_partitions) > 0:
                ## Fill up the `df_partitions` queue.
                while partition_idx is not None and len(df_partitions) != fetch_partitions + 1:
                    df_partitions.append(
                        run_concurrent(
                            self._fetch_partition,
                            ddf=self._data,
                            ddf_i=partition_idx,
                        )
                    )
                    ## Returns None if we have exhausted all partitions. Ref: https://stackoverflow.com/a/15606960
                    partition_idx: Optional[int] = next(partition_idxs_iter, None)
                ## TODO: check if DaskDataFrame.to_delayed can help here by loading partitions in the background.
                ## Ref:
                ## - https://stackoverflow.com/a/46821613
                ## - https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.to_delayed.html
                fut: Future = df_partitions.popleft()
                df_partition: PandasDataFrame = accumulate(fut)
                del fut  ## Without this line, the inputs to the future, i.e. the partition, are persisted in memory.
                # print(f'Loaded partition {partition_i} into memory ({len(df_partition)/1e3}K rows)')

                ## When asked to shuffle, we take the strategy of randomly selecting a partition, then randomly
                ## selecting rows from that partition. This is done efficiently by simply shuffling rows of the
                ## partition (which, by now, is a Pandas DF),
                if shuffle:
                    df_partition: PandasDataFrame = df_partition.sample(frac=1, random_state=seed)
                ## Convert to the target sdf layout, then select chunks.
                ## This is done because selecting small chunks is faster as a list of dicts.
                df_partition: ScalableDataFrame = ScalableDataFrame.of(df_partition, layout=stream_as)
                if all_are_none(num_rows, num_chunks):
                    ## If no arguments are passed, yield each partition as a chunk.
                    out_sdf_chunk = df_partition
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
                    if out_sdf_chunk is not None and chunks_returned != num_batches_per_shard:
                        yield out_sdf_chunk
                        chunks_returned += 1
                if num_chunks is not None:
                    if length is None:
                        length: int = sum([len(self._data.partitions[ddf_i]) for ddf_i in partition_idxs])
                    ## We reuse the logic of selecting num_rows, but we alter the number of rows according to
                    ## how many chunks have been returned so far.
                    num_rows: int = self._stream_update_num_rows_according_to_num_chunks(
                        length=length,
                        chunks_returned=chunks_returned,
                        num_chunks=num_chunks,
                    )

                if num_rows is not None:  ## TODO: add extra elif for chunk_size
                    if ilen(sdfs) <= num_rows:
                        sdfs.append(df_partition)
                    while ilen(sdfs) > num_rows:
                        ## The final sdf caused us to go over the limit of num_rows.
                        ## So, include everything upto the final sdf.
                        ## This also works with a large final sdf and a small num_rows,
                        ## e.g. if final sdf is 1MM, num_rows=100, then we will keep yielding it in this loop.
                        if drop_last is False and chunks_returned == num_batches_per_shard - 1:
                            ## When using drop_last=False, the last batch will need to be padded with the remaining
                            ## rows. However, the remaining rows in this shard will always be <= batch_size*2. Thus,
                            ## we can set it to twice to get the all remaining rows.
                            num_rows: int = num_rows * 2
                        out_sdf_chunk: ScalableDataFrame = ScalableDataFrame.concat(
                            sdfs[:-1] + [sdfs[-1].iloc[:num_rows - ilen(sdfs[:-1]), :]],
                            layout=stream_as,
                        )
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
                        if out_sdf_chunk is not None:
                            yield out_sdf_chunk
                        chunks_returned += 1  ## NOTE: this is purposely outside the above if-statement!
                        if chunks_returned == num_batches_per_shard:
                            return  ## Stop yielding
                        ## Set the sdfs to be the remaining part of the final sdf.
                        sdfs: List[ScalableDataFrame] = [
                            sdfs[-1].iloc[num_rows - ilen(sdfs[:-1]):, :]
                        ]
                        # print(f'An sdf with {ilen(sdfs)} rows remains')
                        if num_chunks is not None:
                            assert length is not None, f'`length` should be set in the upper half of this loop.'
                            num_rows: int = self._stream_update_num_rows_according_to_num_chunks(
                                length=length,
                                chunks_returned=chunks_returned,
                                num_chunks=num_chunks,
                            )
            ## We have exhausted all partitions, now return the remaining rows as a final sdf:
            if len(sdfs) > 0:
                out_sdf_chunk: ScalableDataFrame = ScalableDataFrame.concat(sdfs, layout=stream_as)
                if map is None:
                    yield self._stream_chunk_to_raw(out_sdf_chunk, raw=raw)
                    chunks_returned += 1
                else:
                    self._stream_enqueue_chunk_for_mapping(
                        out_sdf_chunk=out_sdf_chunk,
                        mapped_sdf_chunks=mapped_sdf_chunks,
                        chunks_returned=chunks_returned,
                        map=map,
                        map_kwargs=map_kwargs,
                        map_failure=map_failure,
                        parallelize=parallelize,
                        executor=executor,
                    )
            while len(mapped_sdf_chunks) != 0:
                out_sdf_chunk: Optional[ScalableDataFrame] = self._stream_dequeue_mapped_chunk(
                    mapped_sdf_chunks=mapped_sdf_chunks,
                    stream_as=stream_as,
                    map_failure=map_failure,
                )
                if out_sdf_chunk is not None:
                    yield self._stream_chunk_to_raw(out_sdf_chunk, raw=raw)
                chunks_returned += 1
                if chunks_returned == num_batches_per_shard:
                    return  ## Stop yielding
        finally:
            del df_partitions
            del mapped_sdf_chunks
            del sdfs
            del map_kwargs
            self._stream_cleanup_executor(
                executor=executor,
                map_executor=map_executor,
            )

    @staticmethod
    def _fetch_partition(ddf: DaskDataFrame, ddf_i: int) -> PandasDataFrame:
        df: DaskDataFrame = ddf.partitions[ddf_i]
        df: PandasDataFrame = df.compute()
        if isinstance(df, PandasSeries) and len(df) == 1 and isinstance(df.iloc[0], ray.ObjectRef):
            ## If you pass a Dask-on-Ray-DataFrame to a ray Task/Actor, for some reason it treats each partition
            ## like a Series object with one element. The one element is a ray.ObjectRef of the actual partition's
            ## PandasDataFrame. So we need to fetch the actual PandasDataFrame.
            df: PandasDataFrame = accumulate(df.iloc[0])
        return df

    @classmethod
    @safe_validate_arguments
    def _stream_get_sharded_partition_idxs(
            cls,
            npartitions: conint(ge=1),
            shuffle: bool,
            seed: Optional[int],
            shard: Tuple[conint(ge=0), conint(ge=1)],
            reverse_sharding: bool,
    ) -> np.ndarray:
        """
        Here we are doing "simple" sharding, where drop_last=None.
        This method is appropriate for sharded distribution of the dataframe across workers, where each worker
        should get **approximately** the same load, and we are okay if the total number of shards is not a
        multiple of num_shards (i.e. unbalanced).
        This is appropriate for distributed evaluation across workers **without synchronization**,
        i.e. "embarassigly parallel" workloads.
        E.g. suppose we want to predict 10MM rows using 4 independent workers. We might have a DaskDataFrame
        of these 10MM rows across 83 partitions. In this case, we want to assign partitions to workers such
        that each worker gets **roughly** the same load.
        For 83 partitions and 4  workers, we will assign partitions as:
            shard=(0, 4) => [0, 4,  ... 72, 76, 80]     (21 partitions)
            shard=(1, 4) => [1, 5,  ... 73, 77, 81]     (21 partitions)
            shard=(2, 4) => [2, 6,  ... 74, 78, 82]     (21 partitions)
            shard=(3, 4) => [3, 7,  ... 75, 79]         (20 partitions)
        Thus, we see that the 4th shard will have one less partition. Since we don't care about perfectly
        balanced, shards, this is okay.

        Steps to get randomized shards:
        1. Shuffle (by randomly shuffling the order of partitions).
         - In the case of Pandas, List-of-Dict and Dict, "partitions" are single rows.
         - In the case of Dask, "Partitions" are DaskDataFrame partitions, **which might not be of equal length**.
        2. Shard (by picking every N'th partition based on rank, from the shuffled list)
         E.g. if our original list is [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], and our partitioned lists for num_shards=4 are:
           shard=(0, 4) => [0, 1, 2]
           shard=(1, 4) => [3, 4, 5]
           shard=(2, 4) => [6, 7]
           shard=(3, 4) => [8, 9]
        With shuffle=True, the shuffled list is [8, 1, 5, 0, 7, 2, 9, 4, 3, 6], and our shards are:
           shard=(0, 4) => [8, 7, 3]
           shard=(1, 4) => [1, 2, 6]
           shard=(2, 4) => [5, 9]
           shard=(3, 4) => [0, 4]
         - In the case of Pandas, List-of-Dict and Dict layouts, this gives us true randomization and ensures each
         shard has at most 1 row difference from other shards.
         - In the case of Dask, this gives us true randomization **across partitions**, and ensures each shard has
         at most 1 partitions difference from other shards.
        """
        partition_idxs: np.ndarray = np.arange(0, npartitions)
        if shuffle:
            ## Ref: https://stackoverflow.com/a/47742676
            partition_idxs: np.ndarray = np.random.RandomState(seed=seed).permutation(partition_idxs)
        shard_rank, num_shards = shard
        if npartitions < num_shards:  ## E.g. we ask for 10 shards but there are only 5 partitions
            raise ValueError(
                f'Sharding failed when streaming: requested {num_shards} shards, but the DataFrame only has '
                f'{npartitions} partitions.'
            )
        if not (0 <= shard_rank < num_shards):
            raise ValueError(
                f'When we have {npartitions} partitions, `shard_rank` must be in range [0, {num_shards}); '
                f'found shard_rank={shard_rank}, which is outside this range.'
            )
        if num_shards > 1:
            if shuffle and seed is None:
                raise ValueError(
                    f'When calling .stream() with {num_shards} shards and shuffle=True, '
                    f'you must pass "seed" to ensure you get consistent results.'
                )
            if reverse_sharding:
                partition_idxs: np.ndarray = np.array([
                    idx
                    for idx_i, idx in enumerate(partition_idxs)
                    if idx_i % num_shards != shard_rank  ## Only pick shards which do NOT belong to this rank.
                ])
            else:
                partition_idxs: np.ndarray = np.array([
                    idx
                    for idx_i, idx in enumerate(partition_idxs)
                    if idx_i % num_shards == shard_rank
                ])
        return partition_idxs

    @safe_validate_arguments
    def set_shard_divisions(
            self,
            num_shards: int,
            num_rows: int,
            inplace: bool = True,
            index_col_name: str = '__ROW_UID__',
    ) -> Tuple[Optional[ScalableDataFrame], int]:
        ## Repartition the DaskDataFrame along equal divisions. Otherwise, we cannot shard in a balanced way
        ## (i.e. by ensuring the same number of batches across shards)
        ## Refs:
        ## - https://stackoverflow.com/a/75642406/4900327
        ## - https://www.coiled.io/blog/dask-set-index-dataframe
        ## - https://stackoverflow.com/a/56014192/4900327

        df: DaskDataFrame = self._data
        partition_lens: List[int] = self._partition_lens(df)
        length: int = sum(partition_lens)
        if num_shards > length:
            raise DataFrameShardingError(
                f'Cannot shard DataFrame of {length} rows into {num_shards} shards; DataFrame length is insufficient. '
                f'Please reduce the number of shards.'
            )

        if num_shards * num_rows > length:
            raise DataFrameShardingError(
                f'Cannot shard DataFrame into {num_shards} shards into batches of size {num_rows}; '
                f'{num_shards}*{num_rows} is more than the length of the DataFrame ({length} rows), '
                f'so we cannot even create one batch for each shard. '
                f'Please reduce the number of shards and/or reduce the number of rows per batch.'
            )
        divisions, _, num_batches_per_shard = self._stream_get_balanced_shard_intervals(
            length=length,
            npartitions=df.npartitions,
            num_shards=num_shards,
            num_rows=num_rows,
            ## We need to ensure that the DaskDataFrame before and after partitioning has the same number of rows,
            ## we can't drop rows. drop_last=False will retain all rows and pad the last batches.
            drop_last=False,
        )
        if list(df.divisions) != list(divisions):
            Log.warning(
                f'WARNING: reassigning Dask DataFrame (length={length}, npartitions={df.npartitions}), to '
                f'{len(divisions)} divisions (using num_shards={num_shards}, num_rows={num_rows}, drop_last={False}). '
                f'This might take a while...'
            )
            df: DaskDataFrame = df.map_partitions(
                self._set_unique_row_num,
                partition_lens=partition_lens,
                unique_col_name=index_col_name,
                meta={**df.dtypes.to_dict(), index_col_name: int, }
            )
            df: DaskDataFrame = df.set_index(index_col_name, sorted=True)
            ## TODO: fix drop_last=True case, getting error: "right side of the new division must be equal or larger than old division"
            df: DaskDataFrame = df.repartition(divisions=divisions, force=True)
        if inplace:
            self._data = df
            return None, num_batches_per_shard
        return self._constructor(df, name=self.name), num_batches_per_shard

    @staticmethod
    def _partition_lens(df: DaskDataFrame) -> List[int]:
        partition_lens: List[int] = [
            x[1] for x in
            sorted(list(df.map_partitions(
                lambda df_part, partition_info: (partition_info['number'], len(df_part)),
                meta=tuple,
            ).compute()), key=lambda x: x[0])
        ]
        return partition_lens

    def set_unique_column(self, col_name: str = '__ROW_UID__') -> DaskScalableDataFrame:
        partition_lens: List[int] = DaskScalableDataFrame._partition_lens(self._data)
        self._data = self._data.map_partitions(
            DaskScalableDataFrame._set_unique_row_num,
            partition_lens=partition_lens,
            unique_col_name=col_name,
            meta={**self._data.dtypes.to_dict(), col_name: int, }
        )
        return self

    @staticmethod
    def _set_unique_row_num(
            df_part: PandasDataFrame,
            partition_lens: List[int],
            partition_info: Dict,
            unique_col_name: str,
    ) -> PandasDataFrame:
        partition_number: int = partition_info['number']
        df_part_len: int = len(df_part)
        rows_before_this_partition: int = sum(partition_lens[:partition_number])
        row_uids = [
            row_i
            for row_i in range(rows_before_this_partition, rows_before_this_partition + df_part_len)
        ]
        return df_part.assign(**{unique_col_name: row_uids})
