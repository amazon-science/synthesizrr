from typing import *
import io, json, boto3
from botocore.exceptions import NoCredentialsError
import pandas as pd
from pandas.core.frame import DataFrame as PandasDataFrame
from pandas import read_json as Pandas_read_json
import dask.dataframe as dd
from dask.dataframe.core import DataFrame as DaskDataFrame
from dask.dataframe.core import Series as DaskSeries
from dask.dataframe.io.json import read_json as Dask_read_json
from synthesizrr.base.data.sdf.ScalableDataFrame import ScalableDataFrame, ScalableDataFrameRawType
from synthesizrr.base.util import optional_dependency
from synthesizrr.base.data.reader.dataframe import DataFrameReader
from synthesizrr.base.data.FileMetadata import FileMetadata
from synthesizrr.base.util.aws.s3 import S3Util
from synthesizrr.base.constants import FileFormat, Storage, DataLayout, MLTypeSchema
from pydantic import constr


class JsonLinesReader(DataFrameReader):
    aliases = ['JsonLinesDataFrameReader']  ## Backward compatibility
    file_formats = [FileFormat.JSONLINES, FileFormat.METRICS_JSONLINES]

    class Params(DataFrameReader.Params):
        orient: constr(min_length=1) = 'records'
        lines: bool = True
        index: bool = True

    def _read_raw_sdf(
            self,
            source: Union[str, io.IOBase],
            storage: Storage,
            data_schema: Optional[MLTypeSchema],
            read_as: Optional[DataLayout],
            **kwargs
    ) -> ScalableDataFrameRawType:
        if read_as is DataLayout.LIST_OF_DICT and self.params.orient == 'records' and self.params.lines is True:
            list_of_dict: Optional[List[Dict]] = self.read_list_of_dict(source, storage=storage)
            if list_of_dict is not None:
                return list_of_dict
            ## Fallback to Pandas if we cannot read as list of dict.
        jsonlines_params: Dict = self.filtered_params(Pandas_read_json)
        try:
            return pd.read_json(source, **jsonlines_params)
        except NoCredentialsError:
            assert storage is Storage.S3
            ## Create a new session and read manually:
            bucket_name, file_key = S3Util.s3_path_exploder(source)
            response = boto3.Session().client('s3').get_object(Bucket=bucket_name, Key=file_key)
            jsonlines_data = response['Body'].read().decode('utf-8')
            pd.read_json(io.StringIO(jsonlines_data), **jsonlines_params)

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
            return dd.read_json(
                source,
                ## Dask .read_json params: docs.dask.org/en/stable/generated/dask.dataframe.read_json.html
                **self.filtered_params(Dask_read_json, Pandas_read_json),
            )

    @staticmethod
    def read_list_of_dict(source: Union[str, io.IOBase], storage: Optional[Storage] = None) -> Optional[List[Dict]]:
        if storage is None:
            storage: Storage = FileMetadata.detect_storage(source)
        with optional_dependency("orjson"):  ## Library faster Json parsing: https://github.com/ijl/orjson
            import orjson
            if storage is Storage.STREAM:
                return [orjson.loads(line) for line in source.readlines()]
            elif storage is Storage.LOCAL_FILE_SYSTEM:
                with io.open(source, 'rb') as inp:
                    return [orjson.loads(line) for line in inp]
            elif storage is Storage.S3:
                return [orjson.loads(line) for line in S3Util.stream_s3_object(source)]
        ## Fallback to regular json:
        if storage is Storage.STREAM:
            return [json.loads(line) for line in source.readlines()]
        elif storage is Storage.LOCAL_FILE_SYSTEM:
            list_of_dict: List[Dict] = []
            with io.open(source, 'r') as inp:
                for line in inp:
                    list_of_dict.append(json.loads(line))
            return list_of_dict
        elif storage is Storage.S3:
            return [json.loads(line) for line in S3Util.stream_s3_object(source)]
        return None
