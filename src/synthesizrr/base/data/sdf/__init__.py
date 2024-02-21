from synthesizrr.base.data.sdf.ScalableSeries import *
from synthesizrr.base.data.sdf.TensorScalableSeries import *
from synthesizrr.base.data.sdf.ScalableDataFrame import *
from synthesizrr.base.data.sdf.NumpyArrayScalableSeries import *
from synthesizrr.base.data.sdf.TorchScalableSeries import *
from synthesizrr.base.data.sdf.DatumScalableSeries import *
from synthesizrr.base.data.sdf.PandasScalableSeries import *
from synthesizrr.base.data.sdf.DaskScalableSeries import *
from synthesizrr.base.data.sdf.RecordScalableDataFrame import *
from synthesizrr.base.data.sdf.ListOfDictScalableDataFrame import *
from synthesizrr.base.data.sdf.DictScalableDataFrame import *
from synthesizrr.base.data.sdf.PandasScalableDataFrame import *
from synthesizrr.base.data.sdf.DaskScalableDataFrame import *

#
# def __get_dataframe_reader(DataFrameReaderClass, **kwargs):
#     instance_kwargs: Dict[str, Any] = {k: v for k, v in kwargs.items() if k in DataFrameReaderClass.properties()}
#     instance_kwargs.pop('params', None)
#     params: Dict[str, Any] = {k: v for k, v in kwargs.items() if k not in DataFrameReaderClass.properties()}
#     return DataFrameReaderClass(params=params, **instance_kwargs)
#
#
# def read_csv(source: Any, **kwargs) -> ScalableDataFrameOrRaw:
#     from synthesizrr.base.data.reader import CsvReader
#     return __get_dataframe_reader(CsvReader, **kwargs).read(source, **kwargs)
#
#
# def read_parquet(source: Any, **kwargs) -> ScalableDataFrameOrRaw:
#     from synthesizrr.base.data.reader import ParquetReader
#     return __get_dataframe_reader(ParquetReader, **kwargs).read(source, **kwargs)
#
#
# def read_json(source: Any, **kwargs) -> ScalableDataFrameOrRaw:
#     from synthesizrr.base.data.reader import JsonLinesReader
#     return __get_dataframe_reader(JsonLinesReader, **kwargs).read(source, **kwargs)
