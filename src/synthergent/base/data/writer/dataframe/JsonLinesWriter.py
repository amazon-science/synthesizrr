from typing import *
import io, os
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame as PandasDataFrame
from pandas.core.series import Series as PandasSeries
import dask.dataframe as dd
from dask.dataframe.core import DataFrame as DaskDataFrame
from dask.dataframe.core import Series as DaskSeries
from pandas.io.json import to_json as Pandas_to_json
from dask.dataframe.io.json import to_json as Dask_to_json
from synthergent.base.data.writer.dataframe.DataFrameWriter import DataFrameWriter
from synthergent.base.data.sdf.ScalableDataFrame import ScalableDataFrame, ScalableDataFrameRawType
from synthergent.base.data.sdf.DaskScalableDataFrame import DaskScalableDataFrame
from synthergent.base.constants import FileFormat, DataLayout, Storage
from synthergent.base.data.FileMetadata import FileMetadata
from synthergent.base.util import FileSystemUtil, all_are_none
from synthergent.base.util.aws import S3Util
from pydantic import root_validator, constr


class JsonLinesWriter(DataFrameWriter):
    aliases = ['JsonLinesDataFrameWriter']  ## Backward compatibility
    file_formats = [FileFormat.JSONLINES]
    dask_multiple_write_file_suffix = '.part'  ## github.com/dask/dask/issues/9044

    class Params(DataFrameWriter.Params):
        orient: constr(min_length=1) = 'records'
        lines: bool = True
        index: bool = True

    def _write_sdf(
            self,
            destination: Union[io.IOBase, str],
            sdf: ScalableDataFrame,
            storage: Storage,
            **kwargs,
    ) -> NoReturn:
        sdf.to_json(
            destination,
            **self.filtered_params(PandasDataFrame.to_json),
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
            ## Dask's to_json always writes a folder, even if you want it to write a single file.
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
            sdf.to_json(
                destination,
                name_function=name_function,
                ## Dask .to_json params: docs.dask.org/en/stable/generated/dask.dataframe.to_json.html
                **self.filtered_params(Dask_to_json, PandasDataFrame.to_json),
            )
