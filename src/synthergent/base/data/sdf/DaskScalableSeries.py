from typing import *
import numpy as np
import dask.array as da
from pandas.core.frame import Series as PandasSeries, DataFrame as PandasDataFrame
from dask.dataframe.core import Scalar as DaskScalar, Series as DaskSeries, DataFrame as DaskDataFrame
from synthergent.base.util import wrap_fn_output, is_function, get_default, RayDaskPersistWaitCallback, StringUtil
from synthergent.base.constants import DataLayout
from synthergent.base.data.sdf.ScalableSeries import ScalableSeries, SS_DEFAULT_NAME
from synthergent.base.data.sdf.ScalableDataFrame import ScalableDataFrame
from pydantic.typing import Literal

DaskScalableSeries = "DaskScalableSeries"


class DaskScalableSeries(ScalableSeries):
    layout = DataLayout.DASK
    layout_validator = ScalableSeries.is_dask

    def __init__(self, data: Union[DaskSeries, ScalableSeries], name: Optional[str] = None, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        if isinstance(data, ScalableSeries):
            data: DaskSeries = data.to_dask(**kwargs)
        self.layout_validator(data)
        self._data: DaskSeries = data
        if name is not None and not isinstance(name, (str, int, float)):
            raise ValueError(
                f'`name` used to construct {self.__class__} can only be int, str or float; '
                f'found object of type: {type(name)} with value: {name}'
            )
        else:
            self._data.name = name
        self._name: Optional[str] = name

    def __len__(self):
        return len(self._data)

    def __str__(self):
        name_str: str = '' if self._name is None else f'"{self._name}": '
        out = f"{name_str}Dask Series of dtype `{self._data.dtype}` with {len(self)} items:\n"
        # out += '\n' + '-' * len(out) + '\n'
        out += str(self._data)
        return out

    @classmethod
    def _compute_dask_scalar(cls, scalar: DaskScalar) -> Any:
        return scalar.compute()

    @classmethod
    def _to_scalable(cls, data: Any) -> Union[ScalableDataFrame, ScalableSeries, Any]:
        if isinstance(data, DaskScalar):
            return cls._compute_dask_scalar(data)
        if isinstance(data, np.ndarray):
            return ScalableSeries.get_subclass(DataLayout.NUMPY)(data)
        if isinstance(data, PandasSeries):
            return ScalableSeries.get_subclass(DataLayout.PANDAS)(data)
        if isinstance(data, PandasDataFrame):
            return ScalableDataFrame.get_subclass(DataLayout.PANDAS)(data)
        if isinstance(data, DaskSeries):
            return ScalableSeries.get_subclass(DataLayout.DASK)(data)
        if isinstance(data, DaskDataFrame):
            return ScalableDataFrame.get_subclass(DataLayout.DASK)(data)
        return data

    def __getattr__(self, attr_name: str):
        """Forwards calls to the respective method of Dask Series class."""
        out = super().__getattr__(attr_name)
        if is_function(out):
            return wrap_fn_output(out, wrapper_fn=self._to_scalable)

    def __getitem__(self, key: Any):
        return self._data[key]

    def __setitem__(self, key: Any, value: Any):
        # self._data[key] = value
        raise NotImplementedError(f'Cannot set at the moment')

    def as_pandas(self, **kwargs) -> PandasSeries:
        return self._data.compute()

    def as_dask(self, **kwargs) -> DaskSeries:
        return self._data

    def _to_frame_raw(self, **kwargs):
        kwargs['name']: Any = get_default(self._name, SS_DEFAULT_NAME)
        return self._data.to_frame(**kwargs)

    def loc(self) -> Any:
        return None

    def is_lazy(self) -> bool:
        return True

    def persist(self, wait: bool = False, **kwargs) -> ScalableSeries:
        """
        Submits execution of the Dask task graph up to the current Series.
        :param wait: whether to block the main thread until the .persist() execution completes.
        :param kwargs: additional kwargs forwarded to Dask Series's .persist() method.
        :return: ScalableSeries.
        """
        if not isinstance(wait, bool):
            raise ValueError(f'Attribute `wait` must be a boolean, found value of type {type(wait)}')
        if wait:
            with RayDaskPersistWaitCallback():
                self._data = self._data.persist(**kwargs)
        else:
            self._data = self._data.persist(**kwargs)
        return self

    def compute(self, **kwargs) -> ScalableSeries:
        """For distributed series, runs the task graph upto the current series."""
        return self.of(self._data.compute(**kwargs))

    """
    ---------------------------------------------
    Attributes
    ---------------------------------------------
    """

    @property
    def hasnans(self) -> bool:
        if np.issubdtype(self._data.dtype, int):  ## Ref: https://stackoverflow.com/a/37727662
            return False
        if np.issubdtype(self._data.dtype, float):  ## Ref: https://stackoverflow.com/a/37727662
            return np.isnan(self._data.dot(self.data).compute())
        return da.any(self._data.isna())

    """
    ---------------------------------------------
    Conversion
    ---------------------------------------------
    """

    def bool(self) -> bool:
        length = len(self)
        if length != 1:
            raise ValueError(f'Can only run `.bool()` with Series having one element; found {length}.')
        data = self._data.compute()[0]
        if not (np.issubdtype(self._data.dtype, bool) or isinstance(data, bool)):
            raise ValueError(f'Can only obtain `.bool()` value of Series having True or False data.')
        return bool(data)

    """
    ---------------------------------------------
    Indexing, iteration
    ---------------------------------------------
    """

    def item(self) -> bool:
        length = len(self)
        if length != 1:
            raise ValueError(f'Can only run `.item()` with Series having one element; found {length}.')
        data = self._data.compute()[0]
        return data

    """
    ---------------------------------------------
    Function application, GroupBy & window
    ---------------------------------------------
    """

    def map(
            self,
            arg: Union[Callable, Dict, ScalableSeries],
            *,
            na_action: Optional[Literal['ignore']] = None,
    ) -> ScalableSeries:
        return self._constructor(
            self._data.map_partitions(
                lambda series_part: series_part.map(arg, na_action=na_action),
            )
        )

    def apply(self, func, *, convert_dtype: bool = True, args=(), **kwargs):
        return self._data.map_partitions(
            lambda series_part: series_part.apply(func, convert_dtype=convert_dtype, args=args, **kwargs)
        )

    """
    ---------------------------------------------
    Computations / descriptive stats
    ---------------------------------------------
    """

    def median(self, *args, **kwargs):
        return self.compute().median(*args, **kwargs)
