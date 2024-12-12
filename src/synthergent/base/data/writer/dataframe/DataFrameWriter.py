from typing import *
from abc import abstractmethod, ABC
import io, os, time, math
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame as PandasDataFrame
from pandas.core.series import Series as PandasSeries
import dask.dataframe as dd
from dask.dataframe.core import DataFrame as DaskDataFrame
from dask.dataframe.core import Series as DaskSeries
from synthergent.base.data.FileMetadata import FileMetadata
from synthergent.base.data.writer.Writer import Writer
from synthergent.base.data.sdf.ScalableDataFrame import ScalableDataFrame, ScalableDataFrameRawType
from synthergent.base.data.sdf.DaskScalableDataFrame import DaskScalableDataFrame
from synthergent.base.util import multiple_are_not_none, any_are_not_none, all_are_none, FileSystemUtil, Schema, AutoEnum, auto, \
    accumulate, dispatch, get_default, Future, format_exception_msg, Log, \
    StringUtil, set_param_from_alias, create_progress_bar, safe_validate_arguments, TqdmProgressBar
from synthergent.base.util.aws import S3Util
from synthergent.base.constants import FileContents, FileFormat, DataLayout, MLTypeSchema, Storage, Parallelize
from pydantic import root_validator, conint, constr


class DataFrameWriter(Writer, ABC):
    """
    Abstract base class for writing DataFrames to file in various formats.
    Additional params (over-and-above Writer):
    - data_schema: (Optional) Dict of string to MLType, denoting the schema expected schema of the DataFrame columns.
     This will be used to filter the DataFrame pre-writing and optimize writes.
    - allow_missing_columns: whether to write even if columns from data_schema are missing in the DataFrame.
    - num_rows: max num rows to split dataframe into while writing. Mutually-exclusive with `num_chunks`
    - num_chunks: max num chunks to split dataframe into while writing. Mutually-exclusive with `num_rows`
    - parallelize: how to parallelize when writing multiple files.
    """

    file_contents = [
        FileContents.DATAFRAME,
    ]
    streams = [io.TextIOBase]
    dask_multiple_write_file_suffix: ClassVar[str] = ''

    data_schema: Optional[MLTypeSchema] = None
    allow_missing_columns: bool = False
    num_rows: Optional[conint(ge=1)] = None
    num_chunks: Optional[conint(ge=1)] = None
    ## TODO: implement chunk_size: Optional[Union[conint(ge=1), constr(regex=StringUtil.FILE_SIZE_REGEX)]] = None
    parallelize: Parallelize = Parallelize.threads

    @root_validator(pre=True)
    def check_df_writer_params(cls, params: Dict):
        params: Dict = Writer.convert_params(params)
        set_param_from_alias(params, param='num_rows', alias=[
            'batch_size',
            'write_batch_size',
            'file_batch_size',
            'write_num_rows',
            'file_num_rows',
        ])
        set_param_from_alias(params, param='num_chunks', alias=[
            'num_batches',
            'write_num_batches',
            'file_num_batches',
            'write_num_chunks',
            'file_num_chunks',
            'num_files',
        ])

        if multiple_are_not_none(
                params.get('num_rows'),
                params.get('num_chunks'),
        ):
            raise ValueError(
                f'Only one of `num_rows` or `num_chunks` can be non-None. '
                f'Found: {params}'
            )
        # if all_are_none(
        #         params.get('num_rows'),
        #         params.get('num_chunks'),
        # ):
        #     params['num_rows'] = cls.default_num_rows_per_split
        return params

    @property
    def has_schema(self) -> bool:
        return self.data_schema is not None

    def _preprocess(
            self,
            data: Union[ScalableDataFrame, ScalableDataFrameRawType],
            write_as: Optional[DataLayout] = None,
            allow_missing_columns=False,
            **kwargs,
    ) -> ScalableDataFrame:
        ## Converts data to ScalableDataFrame, and filters it based on the schema (if it exists).
        sdf: ScalableDataFrame = ScalableDataFrame.of(data, layout=write_as)
        sdf: ScalableDataFrame = Schema.filter_df(
            sdf,
            data_schema=self.data_schema,
            allow_missing=allow_missing_columns or self.allow_missing_columns,
        )
        return sdf

    def _write_stream(
            self,
            stream: io.IOBase,
            data: Union[ScalableDataFrame, ScalableDataFrameRawType],
            write_as: Optional[DataLayout] = None,
            **kwargs,
    ) -> NoReturn:
        """
        Writes ScalableDataFrame (or raw dataframe-like object) to a stream.
        :param stream: stream (subclass of io.IOBase) to write to.
        :param data: input ScalableDataFrame (in any format) or "raw" dataframe-like object, e.g. dict, list-of-dict,
            Pandas DataFrame, Dask DataFrame, etc.
        :param write_as: data layout to convert to while writing. We will use the writing logic specific to that
            layout, e.g. by setting write_as=DataLayout.DASK, we will use DaskDataFrame.to_parquet, etc to write.
        :param kwargs: extra kwargs forwarded to the writing function, e.g. to_csv, to_parquet, etc.
        :return: NoReturn.
        """
        sdf: ScalableDataFrame = self._preprocess(data=data, write_as=write_as, **kwargs)
        self._write_sdf_single(
            destination=stream,
            sdf=sdf,
            storage=Storage.STREAM,
            **kwargs,
        )
        ## Do not return anything here.

    def _write_local(
            self,
            local_path: str,
            data: Union[ScalableDataFrame, ScalableDataFrameRawType],
            write_as: Optional[DataLayout] = None,
            file_name: Optional[constr(min_length=1)] = None,
            single_file: Optional[bool] = None,
            **kwargs,
    ) -> Union[str, List[str]]:
        """
        Writes ScalableDataFrame (or raw dataframe-like object) to local path (file or folder).
        :param local_path: file or folder to write to.
        :param data: input ScalableDataFrame (in any format) or "raw" dataframe-like object, e.g. dict, list-of-dict,
            Pandas DataFrame, Dask DataFrame, etc.
        :param write_as: data layout to convert to while writing. We will use the writing logic specific to that
            layout, e.g. by setting write_as=DataLayout.DASK, we will use DaskDataFrame.to_parquet, etc to write.
        :param file_name: (optional) file name to use when writing to a directory.
            When writing a single file to a directory, this is mandatory.
            When writing multiple files to a directory, we default to naming the files 'part'.
        :param kwargs: extra kwargs forwarded to the writing function, e.g. to_csv, to_parquet, etc.
        :return: local path(s) to the file(s) written.
        """
        sdf: ScalableDataFrame = self._preprocess(data=data, write_as=write_as, **kwargs)
        if not FileSystemUtil.is_path_valid_dir(local_path):
            ## Path is a file:
            return self._write_sdf_single(
                destination=local_path,
                sdf=sdf,
                storage=Storage.LOCAL_FILE_SYSTEM,
                **kwargs,
            )
        else:
            ## Path is a directory:
            if all_are_none(self.num_rows, self.num_chunks) or single_file is True:
                ## Do not write multiple files:
                if file_name is None:
                    raise ValueError(f'You must pass `file_name` when writing to local directory "{local_path}".')
                local_file_path: str = FileSystemUtil.construct_file_path_in_dir(
                    path=local_path,
                    name=file_name,
                    file_ending=self.file_ending,
                )
                return self._write_sdf_single(
                    destination=local_file_path,
                    sdf=sdf,
                    storage=Storage.LOCAL_FILE_SYSTEM,
                    file_name=None,
                    **kwargs,
                )
            else:
                ## Write as multiple files:
                return self._write_sdf_multi(
                    destination=local_path,
                    sdf=sdf,
                    storage=Storage.LOCAL_FILE_SYSTEM,
                    file_name=file_name,
                    **kwargs,
                )

    def _write_s3(
            self,
            s3_path: str,
            data: Union[ScalableDataFrame, ScalableDataFrameRawType],
            write_as: Optional[DataLayout] = None,
            file_name: Optional[constr(min_length=1)] = None,
            single_file: Optional[bool] = None,
            **kwargs,
    ) -> Union[str, List[str]]:
        """
        Writes ScalableDataFrame (or raw dataframe-like object) to S3 path (file or folder).
        :param s3_path: S3 file or folder to write to.
        :param data: input ScalableDataFrame (in any format) or "raw" dataframe-like object, e.g. dict, list-of-dict,
            Pandas DataFrame, Dask DataFrame, etc.
        :param write_as: data layout to convert to while writing. We will use the writing logic specific to that
            layout, e.g. by setting write_as=DataLayout.DASK, we will use DaskDataFrame.to_parquet, etc to write.
        :param file_name: (optional) file name to use when writing to a directory.
            When writing a single file to a directory, this is mandatory.
            When writing multiple files to a directory, we default to naming the files 'part'.
        :param kwargs: extra kwargs forwarded to the writing function, e.g. to_csv, to_parquet, etc.
        :return: S3 path(s) to the file(s) written.
        """
        sdf: ScalableDataFrame = self._preprocess(data=data, write_as=write_as, **kwargs)
        if not S3Util.is_path_valid_s3_dir(s3_path):
            ## Path is a file:
            if any_are_not_none(self.num_rows, self.num_chunks):
                raise IOError(
                    f'Cannot write multiple files to a single S3 file "{s3_path}"; '
                    f'please set `num_rows`, and `num_chunks` as None.'
                )
            return self._write_sdf_single(
                destination=s3_path,
                sdf=sdf,
                storage=Storage.S3,
                **kwargs,
            )
        else:
            ## Path is a directory:
            if all_are_none(self.num_rows, self.num_chunks) or single_file is True:
                ## Do not write multiple files:
                if file_name is None:
                    raise ValueError(f'You must pass `file_name` when writing to S3 directory "{s3_path}".')
                s3_file_path: str = S3Util.construct_path_in_s3_dir(
                    s3_path=s3_path,
                    name=file_name,
                    file_ending=self.file_ending,
                    is_dir=True,
                )
                return self._write_sdf_single(
                    destination=s3_file_path,
                    sdf=sdf,
                    storage=Storage.S3,
                    file_name=None,
                    **kwargs,
                )
            else:
                ## Write as multiple files:
                return self._write_sdf_multi(
                    destination=s3_path,
                    sdf=sdf,
                    storage=Storage.S3,
                    file_name=file_name,
                    **kwargs,
                )

    def _write_sdf_single(
            self,
            destination: Union[io.IOBase, str],
            sdf: Union[ScalableDataFrame, DaskScalableDataFrame],
            storage: Storage,
            **kwargs
    ) -> Optional[str]:
        if sdf.layout is DataLayout.DASK:
            ## `destination` here could be a filepath (local or remote), or a stream.
            self._write_dask_sdf(
                destination=destination,
                sdf=sdf,
                storage=storage,
                is_dir=False,
                **kwargs,
            )
        else:
            ## `destination` here could be a filepath (local or remote), or a stream.
            self._write_sdf(
                destination=destination,
                sdf=sdf,
                storage=storage,
                **kwargs
            )
        ## Returns the file path:
        if storage is not Storage.STREAM:
            return destination

    def _write_sdf_multi(
            self,
            destination: str,  ## Do not allow writing multiple files to stream.
            sdf: Union[ScalableDataFrame, DaskScalableDataFrame],
            storage: Storage,
            file_name: Optional[constr(min_length=1)] = None,
            **kwargs
    ) -> List[str]:
        file_name: str = get_default(file_name, ScalableDataFrame.chunk_prefix).strip()
        if sdf.layout is DataLayout.DASK:
            return self._write_sdf_multi_dask(
                destination_dir=destination,
                sdf=sdf,
                storage=storage,
                file_name=file_name,
                **kwargs
            )
        else:
            return self._write_sdf_multi_in_memory(
                destination_dir=destination,
                sdf=sdf,
                storage=storage,
                file_name=file_name,
                **kwargs
            )

    def _write_sdf_multi_dask(
            self,
            destination_dir: str,  ## Local/remote folder path. Do not allow writing multiple files to stream.
            sdf: DaskScalableDataFrame,
            storage: Storage,
            file_name: constr(min_length=1),
            **kwargs,
    ) -> List[str]:
        if self.num_chunks is not None:
            sdf: ScalableDataFrame = sdf.repartition(npartitions=self.num_chunks)
        if self.num_rows is not None:
            sdf: ScalableDataFrame = sdf.repartition(nrows=self.num_rows)
        ## Note: If we do not set `num_chunks` or `num_rows`, then just write the Dask DataFrame into the same number of
        ## files as we have partitions.
        num_zeros: int = StringUtil.get_num_zeros_to_pad(sdf.npartitions)

        ## Function to generate filenames. `idx` indexing starts from 0:
        def name_function(idx) -> str:
            return f'{file_name}-{idx + 1:0{num_zeros}}{self.file_ending}'

        self._write_dask_sdf(
            destination=destination_dir,
            sdf=sdf,
            storage=storage,
            is_dir=True,
            name_function=name_function,
            **kwargs
        )
        if storage is Storage.LOCAL_FILE_SYSTEM:
            return [
                FileSystemUtil.construct_file_path_in_dir(
                    path=destination_dir,
                    name=name_function(idx) + self.dask_multiple_write_file_suffix,
                )
                for idx in range(sdf.npartitions)
            ]
        if storage is Storage.S3:
            return [
                S3Util.construct_path_in_s3_dir(
                    s3_path=destination_dir,
                    name=name_function(idx),
                    is_dir=False,
                )
                for idx in range(sdf.npartitions)
            ]
        raise NotImplementedError(f'Unsupported storage to write multiple files using Dask: {storage}')

    @safe_validate_arguments
    def _write_sdf_multi_in_memory(
            self,
            destination_dir: str,  ## Local/remote folder path. Do not allow writing multiple files to stream.
            sdf: ScalableDataFrame,
            storage: Storage,
            file_name: constr(min_length=1),
            num_workers: Optional[conint(ge=1)] = None,  ## Remove from kwargs
            parallelize: Optional[Parallelize] = None,
            progress_bar: Optional[Union[Dict, bool]] = None,
            **kwargs,
    ) -> List[str]:
        parallelize: Optional[Parallelize] = get_default(parallelize, self.parallelize)
        ## Write multiple ScalableDataFrames using in-memory data-layouts:
        if storage is Storage.LOCAL_FILE_SYSTEM:
            file_path_gen: Callable[[str], str] = lambda fname: FileSystemUtil.construct_file_path_in_dir(
                path=destination_dir,
                name=fname,
                file_ending=self.file_ending,
            )
        elif storage is Storage.S3:
            file_path_gen: Callable[[str], str] = lambda fname: S3Util.construct_path_in_s3_dir(
                s3_path=destination_dir,
                name=fname,
                file_ending=self.file_ending,
                is_dir=False,
            )
        else:
            raise NotImplementedError(f'Unsupported storage: {storage}')
        chunks: Dict[str, ScalableDataFrame] = sdf.split(
            prefix=file_name,
            num_rows=self.num_rows,
            num_chunks=self.num_chunks,
            num_workers=1,
            parallelize=Parallelize.sync,
        )

        pbar: Optional[TqdmProgressBar] = None
        if progress_bar is not None and progress_bar is not False:
            if isinstance(progress_bar, bool):
                progress_bar: Dict = dict()
            progress_bar['unit'] = 'file'
            pbar: TqdmProgressBar = create_progress_bar(**{
                **progress_bar,
                **dict(total=len(sdf), desc=f'Wrote 0 of {len(chunks)} file(s)', unit='row'),
            })
        written_chunks: List[Dict] = []
        for chunk_name, sdf_chunk in chunks.items():
            chunk_file_path: str = file_path_gen(fname=chunk_name)
            fut: Optional = dispatch(
                self._write_sdf,
                parallelize=parallelize,
                destination=chunk_file_path,
                sdf=sdf_chunk,
                storage=storage,
                **kwargs,
            )
            written_chunks.append(dict(
                future=fut,
                chunk_name=chunk_name,
                chunk_file_path=chunk_file_path,
                chunk_len=len(sdf_chunk),
            ))

        failed_chunk_paths: Set[str] = set()
        for file_i, sdf_chunk_write_d in enumerate(written_chunks):
            chunk_file_path: str = sdf_chunk_write_d['chunk_file_path']
            chunk_name: str = sdf_chunk_write_d['chunk_name']
            chunk_len: int = sdf_chunk_write_d['chunk_len']
            try:
                sdf_chunk_write_d: Dict = accumulate(sdf_chunk_write_d)
                if pbar is not None:
                    pbar.update(chunk_len)
                    pbar.set_description(f'Wrote {file_i + 1} of {len(chunks)} file(s)')
            except Exception as e:
                Log.error(
                    f'Error {type(sdf)} to file "{chunk_file_path}":\n'
                    f'{format_exception_msg(e, short=False)}\n'
                    f'Kwargs used: {kwargs}'
                )
                failed_chunk_paths.add(chunk_file_path)
        if pbar is not None:
            pbar.close()
        if len(failed_chunk_paths) > 0:
            raise IOError(f'Could not write DataFrame chunks to following paths:\n{sorted(list(failed_chunk_paths))}')
        return [sdf_chunk_write_d['chunk_file_path'] for sdf_chunk_write_d in written_chunks]

    @abstractmethod
    def _write_sdf(
            self,
            destination: Union[io.IOBase, str],
            sdf: ScalableDataFrame,
            storage: Storage,
            **kwargs,
    ) -> NoReturn:
        """
        Writes to a stream/single file using layout-specific implementations of to_csv, to_parquet, etc.
        :param destination: stream/single file.
        :param sdf: ScalableDataFrame to write.
        :param storage: the storage medium.
        :param kwargs: additional kwargs to pass to .to_csv, .to_parquet, etc.
        :return: NoReturn.
        """
        pass

    @abstractmethod
    def _write_dask_sdf(
            self,
            destination: Union[io.IOBase, str],
            sdf: DaskScalableDataFrame,
            storage: Storage,
            is_dir: bool,
            name_function: Optional[Callable[[int], str]] = None,
            **kwargs,
    ) -> NoReturn:
        """
        Writes to a stream/file/folder using Dask-specific implementations of to_csv, to_parquet, etc.
        :param destination: stream/file/folder.
        :param sdf: DaskScalableDataFrame to write.
        :param storage: the storage medium.
        :param is_dir: whether the destination is a directory or file. When is_dir=False, this method should write a
            single file.
        :param name_function: function which takes in an integer (partition idx) and returns the file name.
        :param kwargs: additional kwargs to pass to .to_csv, .to_parquet, etc.
        :return: NoReturn
        """
        pass
