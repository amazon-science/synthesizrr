from typing import *
from abc import abstractmethod, ABC
from synthesizrr.base.util import safe_validate_arguments
from synthesizrr.base.data.sdf import ScalableDataFrame, ScalableSeries, ScalableSeriesOrRaw, ScalableOrRaw, is_scalable
from synthesizrr.base.data.processor import DataProcessor
from synthesizrr.base.constants import MLType, MLTypeSchema, DataLayout


class Nto1ColumnProcessor(DataProcessor, ABC):
    """Abstract base class for N:1 data processors."""

    @safe_validate_arguments
    def fit(
            self,
            data: ScalableOrRaw,
            process_as: Optional[DataLayout] = None,
    ):
        data: ScalableDataFrame = ScalableDataFrame.of(data, layout=process_as)
        self._fit_df(data)

    def _fit_df(self, data: ScalableDataFrame):
        """Fit step is a noop by default."""
        pass

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    @safe_validate_arguments
    def transform(
            self,
            data: ScalableOrRaw,
            process_as: Optional[DataLayout] = None,
    ) -> ScalableSeriesOrRaw:
        output_data: ScalableSeries = self._transform_df(ScalableDataFrame.of(data, layout=process_as))
        if is_scalable(data):
            return output_data
        return output_data.raw()

    @abstractmethod
    def _transform_df(self, data: ScalableDataFrame) -> ScalableSeries:
        """N:1 data processors can make optimizations internally as column-wise operations are usually much faster."""
        pass

    @safe_validate_arguments
    def fit_transform(
            self,
            data: ScalableOrRaw,
            process_as: Optional[DataLayout] = None,
    ) -> ScalableSeries:
        self.fit(data, process_as=process_as)
        return self.transform(data, process_as=process_as)
