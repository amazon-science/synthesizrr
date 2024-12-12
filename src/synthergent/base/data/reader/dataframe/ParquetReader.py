from typing import *
import io, s3fs, boto3
from botocore.exceptions import NoCredentialsError
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame as PandasDataFrame
from pandas.core.series import Series as PandasSeries
from pandas import read_parquet as Pandas_read_parquet
import dask.dataframe as dd
from dask.dataframe.core import DataFrame as DaskDataFrame
from dask.dataframe.core import Series as DaskSeries
from dask.dataframe import read_parquet as Dask_read_parquet
from synthergent.base.util import optional_dependency, as_list
from synthergent.base.util.aws import S3Util
from synthergent.base.data.sdf.ScalableDataFrame import ScalableDataFrame, ScalableDataFrameRawType
from synthergent.base.data.reader.dataframe import DataFrameReader
from synthergent.base.data.FileMetadata import FileMetadata
from synthergent.base.constants import FileFormat, Storage, DataLayout, MLTypeSchema


class ParquetReader(DataFrameReader):
    file_formats = [FileFormat.PARQUET]
    streams = [io.BytesIO, io.StringIO]

    def _read_raw_sdf(
            self,
            source: Union[str, io.IOBase],
            storage: Storage,
            data_schema: Optional[MLTypeSchema],
            read_as: Optional[DataLayout],
            **kwargs
    ) -> ScalableDataFrameRawType:
        parquet_params: Dict = self.filtered_params(Pandas_read_parquet)
        # print(f'parquet_params: {parquet_params}')
        try:
            return pd.read_parquet(
                source,
                columns=self._filtered_data_columns(source=source, storage=storage, data_schema=data_schema),
                **parquet_params
            )
        except NoCredentialsError:
            assert storage is Storage.S3
            ## Create a new session and read manually:
            bucket_name, file_key = S3Util.s3_path_exploder(source)
            response = boto3.Session().client('s3').get_object(Bucket=bucket_name, Key=file_key)
            parquet_data = response['Body'].read()
            return pd.read_parquet(
                io.BytesIO(parquet_data),
                columns=self._filtered_data_columns(source=source, storage=storage, data_schema=data_schema),
                **parquet_params
            )

    def _read_raw_dask_sdf(
            self,
            source: Union[List[str], str, io.IOBase],
            storage: Storage,
            data_schema: Optional[MLTypeSchema],
            **kwargs
    ) -> DaskDataFrame:
        if storage is Storage.STREAM:
            ## Read as another layout and convert to Dask:
            df: ScalableDataFrameRawType = self._read_raw_sdf_with_retries(
                source=source,
                storage=storage,
                **kwargs
            )
            return ScalableDataFrame.of(df, layout=DataLayout.DASK, **kwargs).raw()
        else:
            return dd.read_parquet(
                source,
                columns=self._filtered_data_columns(source=source, storage=storage, data_schema=data_schema),
                ## Dask .read_parquet params: docs.dask.org/en/stable/generated/dask.dataframe.read_parquet.html
                **self.filtered_params(Dask_read_parquet, Pandas_read_parquet),
            )

    def _filtered_data_columns(
            self,
            *,
            source: Union[List[str], str],
            storage: Storage,
            data_schema: Optional[MLTypeSchema],
    ) -> Optional[List[str]]:
        columns: Optional[List[str]] = self._data_columns(data_schema)
        if columns is not None and self.allow_missing_columns and storage is not Storage.STREAM:
            ## read_parquet throws and exception if we try to read only certain columns and any are missing.
            ## Thus, to allow missing columns, we must first read the parquet file columns.
            ## Ref: https://stackoverflow.com/a/65706742/4900327
            for fpath in as_list(source):
                ## Keep only the common subset of columns...this is what Dask does when reading from multiple Parquet
                ## files with different sets of columns.
                file_columns: Optional[List[str]] = self.detect_columns(fpath, storage=storage, raise_error=False)
                if file_columns is not None:
                    file_columns: Set[str] = set(file_columns)
                    columns: List[str] = [col for col in columns if col in file_columns]
        return columns

    @classmethod
    def detect_columns(
            cls,
            fpath: str,
            storage: Optional[Storage] = None,
            raise_error: bool = True,
    ) -> Optional[List[str]]:
        if storage is None:
            storage: Storage = FileMetadata.detect_storage(fpath)
        if storage is not Storage.LOCAL_FILE_SYSTEM:
            if raise_error:
                raise ValueError(f'Can only detect columns for parquet file on disk, not {storage}')
            return None
        with optional_dependency('pyarrow', warn_every_time=True):
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(fpath)
            columns_in_file: List[str] = [c for c in parquet_file.schema.names]
            return columns_in_file
        return None  ## Returned if pyarrow is not found
