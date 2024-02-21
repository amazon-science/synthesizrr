from typing import *
from abc import ABC
from synthesizrr.base.util import get_current_fn_name
from synthesizrr.base.data.sdf import ScalableSeries, ScalableSeriesRawType
from synthesizrr.base.data.processor import DataProcessor
from synthesizrr.base.constants import MLType, MLTypeSchema, DataLayout, DASK_APPLY_OUTPUT_MLTYPE_TO_META_MAP


class SingleColumnProcessor(DataProcessor, ABC):
    """Abstract base class for 1:1 data processors."""

    def fit(
            self,
            data: Union[ScalableSeries, ScalableSeriesRawType],
            process_as: Optional[DataLayout] = None,
    ):
        data: ScalableSeries = ScalableSeries.of(data, layout=process_as)
        self._fit_series(data)

    def _fit_series(self, data: ScalableSeries):
        """Fit step is a noop by default."""
        pass

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def transform(
            self,
            data: Union[ScalableSeries, ScalableSeriesRawType],
            process_as: Optional[DataLayout] = None,
    ) -> Union[ScalableSeries, ScalableSeriesRawType]:
        output_data: ScalableSeries = self._transform_series(ScalableSeries.of(data, layout=process_as))
        if isinstance(data, ScalableSeries):
            return output_data
        return output_data.raw()

    def _transform_series(self, data: ScalableSeries) -> ScalableSeries:
        """1:1 data processors can make optimizations internally."""
        kwargs = {}
        if data.layout is DataLayout.DASK:
            if self.output_mltype in DASK_APPLY_OUTPUT_MLTYPE_TO_META_MAP:
                kwargs['meta'] = DASK_APPLY_OUTPUT_MLTYPE_TO_META_MAP[self.output_mltype]
        return data.apply(self.transform_single, **kwargs)

    def transform_single(self, data: Any) -> Any:
        """
        Transforms a single data point using the current data processor.
        :param data: input data point
        :return: transformed value
        """
        raise NotImplementedError(f'{get_current_fn_name()} has not been implemented.')

    def fit_transform(
            self,
            data: Union[ScalableSeries, ScalableSeriesRawType],
            process_as: Optional[DataLayout] = None,
    ) -> ScalableSeries:
        self.fit(data, process_as=process_as)
        return self.transform(data, process_as=process_as)
