from typing import *
import csv, io, s3fs, boto3
from botocore.exceptions import NoCredentialsError
import pandas as pd
from pandas.core.frame import DataFrame as PandasDataFrame
from pandas import read_csv as Pandas_read_csv
import dask.dataframe as dd
from dask.dataframe.core import DataFrame as DaskDataFrame
from dask.dataframe.core import Series as DaskSeries
from dask.dataframe.io.csv import read_csv as Dask_read_csv
from synthergent.base.data.sdf.ScalableDataFrame import ScalableDataFrame, ScalableDataFrameRawType
from synthergent.base.data.reader.dataframe import DataFrameReader
from synthergent.base.constants import FileFormat, QUOTING_MAP, DataLayout, Storage, MLTypeSchema
from synthergent.base.util import StringUtil
from synthergent.base.util.aws import S3Util
from pydantic import conint, constr, validator


class CsvReader(DataFrameReader):
    file_formats = [FileFormat.CSV]

    class Params(DataFrameReader.Params):
        sep: constr(min_length=1, max_length=3) = StringUtil.COMMA
        quoting: Optional[int] = csv.QUOTE_MINIMAL
        encoding: Optional[str]
        keep_default_na: Optional[bool] = True
        na_values: Optional[List[str]] = []

        @validator('quoting')
        def validate_quoting(cls, quoting):
            if quoting is None:
                return None
            if quoting not in QUOTING_MAP:
                raise ValueError(f'`quoting` must be in {list(QUOTING_MAP.keys())}; found "{quoting}"')
            return QUOTING_MAP[quoting]

    def _read_raw_sdf(
            self,
            source: Union[str, io.IOBase],
            storage: Storage,
            data_schema: Optional[MLTypeSchema],
            read_as: Optional[DataLayout],
            **kwargs
    ) -> ScalableDataFrameRawType:
        csv_params: Dict = self.filtered_params(Pandas_read_csv)
        try:
            return pd.read_csv(
                source,
                usecols=self._filtered_data_columns(source=source, storage=storage, data_schema=data_schema),
                **csv_params
            )
        except NoCredentialsError:
            assert storage is Storage.S3
            ## Create a new session and read manually:
            bucket_name, file_key = S3Util.s3_path_exploder(source)
            response = boto3.Session().client('s3').get_object(Bucket=bucket_name, Key=file_key)
            csv_data = response['Body'].read().decode('utf-8')
            return pd.read_csv(
                io.StringIO(csv_data),
                usecols=self._filtered_data_columns(source=source, storage=storage, data_schema=data_schema),
                **csv_params
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
            return dd.read_csv(
                source,
                usecols=self._filtered_data_columns(source=source, storage=storage, data_schema=data_schema),
                ## Dask .read_csv params: docs.dask.org/en/stable/generated/dask.dataframe.read_csv.html
                **self.filtered_params(Dask_read_csv, Pandas_read_csv),
            )

    def _filtered_data_columns(
            self,
            *,
            source: Union[List[str], str],
            storage: Storage,
            data_schema: Optional[MLTypeSchema],
    ) -> Optional[Union[Callable, List[str]]]:
        columns: Optional[List[str]] = self._data_columns(data_schema)
        if columns is not None and self.allow_missing_columns:
            columns: Set[str] = set(columns)
            should_keep_col: Callable = lambda col: col in columns
            return should_keep_col
        return columns


class TsvReader(CsvReader):
    file_formats = [FileFormat.TSV]

    class Params(CsvReader.Params):
        sep: constr(min_length=1, max_length=3) = StringUtil.TAB
