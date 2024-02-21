from typing import *
import io, os
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame as PandasDataFrame
from pandas.core.series import Series as PandasSeries
import dask.dataframe as dd
from dask.dataframe.core import DataFrame as DaskDataFrame
from dask.dataframe.core import Series as DaskSeries
from pandas.io.parquet import to_parquet as Pandas_to_parquet
from dask.dataframe.io.parquet.core import to_parquet as Dask_to_parquet
from synthesizrr.base.data.writer.dataframe.DataFrameWriter import DataFrameWriter
from synthesizrr.base.data.sdf.ScalableDataFrame import ScalableDataFrame, ScalableDataFrameRawType
from synthesizrr.base.data.sdf.DaskScalableDataFrame import DaskScalableDataFrame
from synthesizrr.base.constants import FileFormat, DataLayout, Storage
from synthesizrr.base.data.FileMetadata import FileMetadata
from synthesizrr.base.util import FileSystemUtil, all_are_none
from synthesizrr.base.util.aws import S3Util
from pydantic import root_validator, Field


class ParquetWriter(DataFrameWriter):
    aliases = ['ParquetDataFrameWriter']  ## Backward compatibility
    file_formats = [FileFormat.PARQUET]
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
        sdf.to_parquet(
            destination,
            **self.filtered_params(Pandas_to_parquet),
        )

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
            params = self.filtered_params(Dask_to_parquet, Pandas_to_parquet)
            if self.data_schema is not None:
                params['schema'] = None  ## Ref: https://github.com/dask/dask/issues/9247#issuecomment-1177958306
            sdf.to_parquet(
                destination,
                name_function=name_function,
                ## Dask .to_parquet params: docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.to_parquet.html
                **params,
            )
