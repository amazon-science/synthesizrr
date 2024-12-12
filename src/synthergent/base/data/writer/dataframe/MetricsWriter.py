from typing import *
import io
import pandas as pd
from synthergent.base.data.writer.dataframe.DataFrameWriter import DataFrameWriter
from synthergent.base.data.sdf.ScalableDataFrame import ScalableDataFrame
from synthergent.base.data.sdf.DaskScalableDataFrame import DaskScalableDataFrame
from synthergent.base.constants import FileFormat, Storage, Task
from synthergent.base.framework.metric import Metric
from synthergent.base.framework.predictions import Predictions
from synthergent.base.constants import FileContents
from pydantic import *
from pandas.core.frame import DataFrame as PandasDataFrame


class MetricsWriter(DataFrameWriter):
    aliases = ['MetricDataFrameWriter']  ## Backward compatibility
    file_formats = [FileFormat.METRICS_JSONLINES]
    dask_multiple_write_file_suffix = '.part'  ## github.com/dask/dask/issues/9044
    file_contents = [FileContents.METRICS_DATAFRAME]

    class Params(DataFrameWriter.Params):
        task: Task
        metrics: List[Metric] = []

        @root_validator(pre=True)
        def convert_params(cls, params: Dict):
            metric_list: List[Metric] = [
                Metric.of(metric_dict.get('metric_name'), **metric_dict.get('metric_params', {}))
                for metric_dict in params.get('metrics_list')]
            params['metrics'] = metric_list
            return params

    class Config(DataFrameWriter.Config):
        extra = Extra.ignore

    @root_validator(pre=True)
    def convert_params(cls, params: Dict):
        params['params'] = cls._convert_params(cls.Params, params)
        return params

    def _write_sdf(
            self,
            destination: Union[io.IOBase, str],
            sdf: ScalableDataFrame,
            storage: Storage,
            **kwargs,
    ) -> NoReturn:
        PredictionsClass: Predictions = Predictions.get_subclass(self.params.task)
        predictions: Predictions = PredictionsClass.from_dataframe(data=sdf, data_schema=self.data_schema)

        evaluated_metrics: List[Metric] = [
            Metric.of(**metric).evaluate(predictions)
            for metric in self.params.metrics
        ]

        metrics_df: PandasDataFrame = pd.DataFrame({
            evaluated_metric.display_name: [evaluated_metric.aiw_format] for evaluated_metric in
            evaluated_metrics
        })
        metrics_df = metrics_df[sorted(metrics_df.columns)]
        metrics_df.to_json(path_or_buf=destination, orient='records')

    def _write_dask_sdf(self, destination: Union[io.IOBase, str], sdf: DaskScalableDataFrame, storage: Storage,
                        is_dir: bool, name_function: Optional[Callable[[int], str]] = None, **kwargs) -> NoReturn:
        self._write_sdf(destination=destination, sdf=sdf, storage=storage, **kwargs)
