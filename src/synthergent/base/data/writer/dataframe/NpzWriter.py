from typing import *
import io, os
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame as PandasDataFrame
from pandas.core.series import Series as PandasSeries
import dask.dataframe as dd
from dask.dataframe.core import DataFrame as DaskDataFrame
from dask.array import to_npy_stack as DaskArray_to_npy
from dask.dataframe.core import Series as DaskSeries
from numpy import savez as Pandas_to_npz
from synthergent.base.data.writer.dataframe.DataFrameWriter import DataFrameWriter
from synthergent.base.data.sdf.ScalableDataFrame import ScalableDataFrame, ScalableDataFrameRawType
from synthergent.base.data.sdf.DaskScalableDataFrame import DaskScalableDataFrame
from synthergent.base.constants import FileFormat, DataLayout, Storage
from synthergent.base.data.FileMetadata import FileMetadata
from synthergent.base.util import FileSystemUtil, all_are_none
from synthergent.base.util.aws import S3Util
from pydantic import root_validator, Field
from synthergent.base.util import Log


class NpzWriter(DataFrameWriter):
    aliases = ['NpzDataFrameWriter']  ## Backward compatibility
    file_formats = [FileFormat.NPZ]
    streams = [io.BytesIO]

    class Params(DataFrameWriter.Params):
        compression: str = 'gzip'
        schema_: Optional[Union[str, Dict, Any]] = Field("infer", alias="schema")

    def _write_sdf(
            self,
            destination: Union[io.IOBase, str],
            sdf: ScalableDataFrame,
            storage: Storage,
            **kwargs,
    ) -> NoReturn:
        if storage is Storage.S3:
            raise NotImplementedError(f'Currently we do not support directly writing .npz files to s3 in NpzWriter')
        elif storage is Storage.STREAM:
            raise NotImplementedError(f'Currently we do not support directly writing to stream in NpzWriter')
        sdf.to_npz(destination, storage, **self.filtered_params(Pandas_to_npz))

    def _write_dask_sdf(
            self,
            destination: Union[io.IOBase, str],
            sdf: DaskScalableDataFrame,
            storage: Storage,
            is_dir: bool,
            name_function: Optional[Callable[[int], str]] = None,
            **kwargs,
    ) -> NoReturn:
        if storage is Storage.STREAM:
            ## Convert dask dataframe to Pandas and write to stream:
            self._write_sdf(
                destination=destination,
                sdf=sdf.as_layout(DataLayout.PANDAS),
                storage=storage,
                **kwargs
            )
        elif not is_dir:
            ## Dask's to_parquet always writes a folder, even if you want it to write a single file.
            ## As a result, we must convert to Pandas to write a single file:
            self._write_sdf(
                destination=destination,
                sdf=sdf.as_layout(DataLayout.PANDAS),
                storage=storage,
                **kwargs
            )
        else:
            ## We are writing multiple files to a directory (either in local or remote).
            assert name_function is not None, f'We require a `name_function` when writing to a directory.'
            params = self.filtered_params(DaskDataFrame.compute, DaskDataFrame.get_partition, Pandas_to_npz)
            sdf.to_npz(
                destination,
                name_function=name_function,
                ## Dask .to_parquet params: docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.to_parquet.html
                **params,
            )