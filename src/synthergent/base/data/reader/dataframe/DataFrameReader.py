from typing import *
from abc import abstractmethod, ABC
import logging, io, time
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame as PandasDataFrame
from pandas.core.series import Series as PandasSeries
import dask.dataframe as dd
from dask.dataframe.core import DataFrame as DaskDataFrame
from dask.dataframe.core import Series as DaskSeries
from synthergent.base.util import StringUtil, Future, Log, accumulate, dispatch, retry as retry_fn, \
    is_list_like, only_item, get_default, format_exception_msg, Schema, safe_validate_arguments, ProgressBar
from synthergent.base.constants import Storage, FileContents, DataLayout, Parallelize, MLTypeSchema, Alias
from synthergent.base.data.reader.Reader import Reader
from synthergent.base.data.sdf.ScalableDataFrame import ScalableDataFrame, ScalableDataFrameRawType, ScalableDataFrameOrRaw, \
    is_scalable


class DataFrameReader(Reader, ABC):
    """
    Abstract base class for reading DataFrames from various sources.
    - data_schema: (Optional) Dict of string to MLType, denoting the schema expected schema of the DataFrame columns.
     This will be used to optimize reads and filter the DataFrame post-reading.
    - allow_missing_columns: whether to write even if columns from data_schema are missing in the DataFrame.
    - parallelize: how to parallelize when writing multiple files.
    """
    file_contents = [
        FileContents.DATAFRAME,
        FileContents.ALGORITHM_TRAIN_DATASET,
        FileContents.ALGORITHM_INFERENCE_DATASET,
        FileContents.ALGORITHM_PREDICTIONS_DATASET
    ]
    streams = [io.TextIOBase]

    data_schema: Optional[MLTypeSchema] = None
    allow_missing_columns: bool = False
    parallelize: Parallelize = Parallelize.threads

    def _data_columns(self, data_schema: Optional[MLTypeSchema]) -> Optional[List[str]]:
        data_schema: Optional[MLTypeSchema] = get_default(data_schema, self.data_schema)
        if data_schema is None:
            return None
        cols: List[str] = sorted(list(data_schema.keys()))
        if len(cols) == 0:
            return None
        return cols

    @safe_validate_arguments
    def _read_stream(
            self,
            stream: io.IOBase,
            file_contents: Optional[FileContents] = None,
            data_schema: Optional[MLTypeSchema] = None,
            raw: bool = False,
            read_as: Optional[DataLayout] = None,
            **kwargs,
    ) -> ScalableDataFrameOrRaw:
        if read_as not in {None, DataLayout.PANDAS}:
            raise IOError(f'Cannot read from stream as {read_as}')
        raw_sdf: ScalableDataFrameRawType = self._read_raw_sdf_with_retries(
            source=stream,
            storage=Storage.STREAM,
            data_schema=data_schema,
            read_as=read_as,
            **kwargs,
        )
        return self._postprocess(raw_sdf, data_schema=data_schema, raw=raw, read_as=read_as, **kwargs)

    @safe_validate_arguments
    def _read_url(
            self,
            url: str,
            file_contents: Optional[FileContents] = None,
            data_schema: Optional[MLTypeSchema] = None,
            raw: bool = False,
            read_as: Optional[DataLayout] = None,
            **kwargs,
    ) -> ScalableDataFrameOrRaw:
        if read_as not in {None, DataLayout.PANDAS}:
            raise IOError(f'Cannot read from stream as {read_as}')
        raw_sdf: ScalableDataFrameRawType = self._read_raw_sdf_with_retries(
            source=url,
            storage=Storage.URL,
            data_schema=data_schema,
            read_as=read_as,
            **kwargs,
        )
        return self._postprocess(raw_sdf, data_schema=data_schema, raw=raw, read_as=read_as, **kwargs)

    @safe_validate_arguments
    def _read_local(
            self,
            local_path: Union[str, List[str]],
            file_contents: Optional[FileContents] = None,
            data_schema: Optional[MLTypeSchema] = None,
            raw: bool = False,
            read_as: Optional[DataLayout] = None,
            **kwargs
    ) -> ScalableDataFrameOrRaw:
        if not is_list_like(local_path):
            ## Path is a file:
            raw_sdf: ScalableDataFrameRawType = self._read_raw_sdf_single(
                source=local_path,
                storage=Storage.LOCAL_FILE_SYSTEM,
                data_schema=data_schema,
                read_as=read_as,
                **kwargs,
            )
        else:
            ## Path is a list of files:
            raw_sdf: ScalableDataFrameOrRaw = self._read_raw_sdf_multi(
                source=local_path,
                storage=Storage.LOCAL_FILE_SYSTEM,
                data_schema=data_schema,
                read_as=read_as,
                **kwargs,
            )
        return self._postprocess(raw_sdf, data_schema=data_schema, raw=raw, read_as=read_as, **kwargs)

    @safe_validate_arguments
    def _read_s3(
            self,
            s3_path: Union[str, List[str]],
            file_contents: Optional[FileContents] = None,
            data_schema: Optional[MLTypeSchema] = None,
            raw: bool = False,
            read_as: Optional[DataLayout] = None,
            **kwargs
    ) -> ScalableDataFrameOrRaw:
        if not is_list_like(s3_path):
            ## Path is a file:
            raw_sdf: ScalableDataFrameRawType = self._read_raw_sdf_single(
                source=s3_path,
                storage=Storage.S3,
                data_schema=data_schema,
                read_as=read_as,
                **kwargs,
            )
        else:
            ## Path is a list of files:
            raw_sdf: ScalableDataFrameRawType = self._read_raw_sdf_multi(
                source=s3_path,
                storage=Storage.S3,
                data_schema=data_schema,
                read_as=read_as,
                **kwargs,
            )
        return self._postprocess(raw_sdf, data_schema=data_schema, raw=raw, read_as=read_as, **kwargs)

    def _postprocess(
            self,
            data: ScalableDataFrameOrRaw,
            raw: bool,
            read_as: Optional[DataLayout],
            data_schema: Optional[MLTypeSchema] = None,
            persist: bool = False,
            **kwargs,
    ) -> ScalableDataFrameOrRaw:
        data: ScalableDataFrame = ScalableDataFrame.of(data, layout=read_as, **kwargs)
        if persist:
            data: ScalableDataFrame = data.persist(**kwargs)
        data_schema: Optional[MLTypeSchema] = get_default(data_schema, self.data_schema)
        if data_schema is not None:
            ## In addition to returning a non-raw ScalableDataFrame, we need to convert to ScalableDataFrame to filter.
            data: ScalableDataFrame = Schema.filter_df(
                data,
                data_schema=data_schema,
                allow_missing=self.allow_missing_columns,
            )
        if raw:
            data: ScalableDataFrameRawType = data.raw()
        return data

    def _read_raw_sdf_single(
            self,
            source: Union[io.IOBase, str],
            storage: Storage,
            data_schema: Optional[MLTypeSchema],
            read_as: Optional[DataLayout],
            **kwargs
    ) -> ScalableDataFrameRawType:
        if read_as is DataLayout.DASK:
            ## `source` here could be a filepath (local or remote), or a stream.
            return self._read_raw_dask_sdf_with_retries(
                source=source,
                storage=storage,
                data_schema=data_schema,
                read_as=read_as,
                **kwargs,
            )
        else:
            ## `source` here could be a filepath (local or remote), or a stream.
            return self._read_raw_sdf_with_retries(
                source=source,
                storage=storage,
                data_schema=data_schema,
                read_as=read_as,
                **kwargs
            )

    def _read_raw_sdf_multi(
            self,
            source: List[str],
            storage: Storage,
            data_schema: Optional[MLTypeSchema],
            read_as: Optional[DataLayout],
            **kwargs
    ) -> ScalableDataFrameOrRaw:
        if read_as is DataLayout.DASK:
            ## `source` here is a list of filepaths (local or remote).
            return self._read_raw_dask_sdf_with_retries(
                source=source,
                storage=storage,
                data_schema=data_schema,
                read_as=read_as,
                **kwargs,
            )
        else:
            ## `source` here is a list of filepaths (local or remote).
            return self._read_raw_sdf_multi_in_memory(
                source=source,
                storage=storage,
                data_schema=data_schema,
                read_as=read_as,
                **kwargs
            )

    @safe_validate_arguments
    def _read_raw_sdf_multi_in_memory(
            self,
            source: List[str],
            storage: Storage,
            data_schema: Optional[MLTypeSchema],
            read_as: Optional[DataLayout],
            **kwargs,
    ) -> ScalableDataFrame:
        progress_bar: Optional[Dict] = Alias.get_progress_bar(kwargs)
        read_progress_bar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=len(source),
            desc=f'Read 0 row(s)',
            unit='file',
        )

        raw_sdf_chunks: Dict[str, Union[Future, ScalableDataFrameRawType]] = {}
        for file_path in source:
            kwargs['parallelize'] = self.parallelize
            raw_sdf_chunks[file_path] = dispatch(
                self._read_raw_sdf_with_retries,
                source=file_path,
                storage=storage,
                data_schema=data_schema,
                read_as=read_as,
                **kwargs,
            )

        raw_sdf_chunks_list: List[ScalableDataFrameRawType] = []
        raw_sdf_chunks_num_rows_read: int = 0
        failed_read_file_paths: List[str] = []
        for file_i, (file_path, raw_sdf_chunk) in enumerate(raw_sdf_chunks.items()):
            try:
                raw_sdf_chunk: ScalableDataFrameRawType = accumulate(raw_sdf_chunk)
                if isinstance(raw_sdf_chunk, dict):
                    col_lens: Dict[Any, int] = {col: len(col_arr) for col, col_arr in raw_sdf_chunk.items()}
                    if len(set(col_lens.values())) > 1:
                        raise ValueError(f'Columns are not of equal length; found following lengths: {col_lens}')
                    raw_sdf_chunk_len: int = only_item(set(col_lens.values()))
                else:
                    raw_sdf_chunk_len: int = len(raw_sdf_chunk)
                raw_sdf_chunks_num_rows_read += raw_sdf_chunk_len
                read_progress_bar.update(1)  ## Increment number of files
                read_progress_bar.set_description(f'Read {raw_sdf_chunks_num_rows_read} row(s)')
                raw_sdf_chunks_list.append(raw_sdf_chunk)
            except Exception as e:
                Log.error(
                    f'Error reading from file "{file_path}":\n'
                    f'{format_exception_msg(e, short=False)}\n'
                    f'Kwargs used: {kwargs}'
                )
                failed_read_file_paths.append(file_path)
        if len(failed_read_file_paths) > 0:
            read_progress_bar.failed(f'{len(failed_read_file_paths)} of {len(source)} failed to read')
            raise IOError(f'Could not read DataFrame from the following paths:\n{sorted(list(failed_read_file_paths))}')
        else:
            read_progress_bar.success()
        return ScalableDataFrame.concat(raw_sdf_chunks_list, reset_index=True, layout=read_as)

    def _read_raw_sdf_with_retries(
            self,
            source: Union[str, io.IOBase],
            storage: Storage,
            data_schema: Optional[MLTypeSchema],
            read_as: Optional[DataLayout],
            retry: Optional[float] = None,
            retry_wait: Optional[float] = None,
            **kwargs
    ) -> ScalableDataFrameRawType:
        return retry_fn(
            self._read_raw_sdf,
            retries=get_default(retry, self.retry),
            wait=get_default(retry_wait, self.retry_wait),
            source=source,
            storage=storage,
            data_schema=data_schema,
            read_as=read_as,
            **kwargs,
        )

    @abstractmethod
    def _read_raw_sdf(
            self,
            source: Union[str, io.IOBase],
            storage: Storage,
            data_schema: Optional[MLTypeSchema],
            read_as: Optional[DataLayout],
            **kwargs
    ) -> ScalableDataFrameRawType:
        pass

    def _read_raw_dask_sdf_with_retries(
            self,
            source: Union[List[str], str, io.IOBase],
            storage: Storage,
            data_schema: Optional[MLTypeSchema],
            retry: Optional[float] = None,
            retry_wait: Optional[float] = None,
            **kwargs
    ) -> DaskDataFrame:
        return retry_fn(
            self._read_raw_dask_sdf,
            retries=get_default(retry, self.retry),
            wait=get_default(retry_wait, self.retry_wait),
            source=source,
            storage=storage,
            data_schema=data_schema,
            **kwargs,
        )

    @abstractmethod
    def _read_raw_dask_sdf(
            self,
            source: Union[List[str], str, io.IOBase],
            storage: Storage,
            data_schema: Optional[MLTypeSchema],
            **kwargs
    ) -> DaskDataFrame:
        pass
