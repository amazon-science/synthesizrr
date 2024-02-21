from typing import *
import numpy as np
import pandas as pd
import math
from pandas.core.frame import Series as PandasSeries, DataFrame as PandasDataFrame
from pandas.core.indexes.multi import MultiIndex as PandasMultiIndex
from synthesizrr.base.util import multiple_are_not_none, all_are_none, is_function, wrap_fn_output, get_default, StringUtil, \
    safe_validate_arguments
from synthesizrr.base.constants import DataLayout, Parallelize
from synthesizrr.base.data.sdf.ScalableSeries import ScalableSeries
from synthesizrr.base.data.sdf.ScalableDataFrame import ScalableDataFrame, ScalableDataFrameOrRaw, is_scalable
from synthesizrr.base.data.sdf.PandasScalableSeries import PandasScalableSeries
from pydantic import conint
from collections import deque

PandasScalableDataFrame = "PandasScalableDataFrame"


class PandasScalableDataFrame(ScalableDataFrame):
    layout = DataLayout.PANDAS
    layout_validator = ScalableDataFrame.is_pandas
    ScalableSeriesClass = PandasScalableSeries

    def __init__(self, data: Union[PandasDataFrame, ScalableDataFrame], name: Optional[str] = None, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        if isinstance(data, ScalableDataFrame):
            data: PandasDataFrame = data.pandas(**kwargs)
        self.layout_validator(data)
        if isinstance(data.index, PandasMultiIndex):
            raise ValueError(f'Input Pandas DataFrame must not have index of type {PandasMultiIndex}.')
        self._data: PandasDataFrame = data
        if name is not None and not isinstance(name, (str, int, float)):
            raise ValueError(
                f'`name` used to construct {self.__class__} can only be int, str or float; '
                f'found object of type: {type(name)} with value: {name}'
            )
        self._name: Optional[str] = name

    @property
    def shape(self) -> Tuple[int, int]:
        return self._data.shape

    @property
    def columns(self) -> List[str]:
        return list(self._data.columns)

    @property
    def columns_set(self) -> Set:
        return set(self._data.columns)

    def __len__(self):
        return self._data.shape[0]

    def __str__(self):
        columns: List[str] = self.columns
        return f"Pandas DataFrame with {len(self)} row(s) and {len(columns)} column(s): " \
               f"\nColumns:{columns}:\nData:\n{str(self._data)}"

    @property
    def loc(self) -> Any:
        return self._data.loc

    @classmethod
    def _to_scalable(cls, data: Any) -> Union[ScalableDataFrame, ScalableSeries, Any]:
        return PandasScalableSeries._to_scalable(data)

    def __getattr__(self, attr_name: str):
        """Forwards calls to the respective method of Pandas Series class."""
        out = super().__getattr__(attr_name)
        if is_function(out):
            return wrap_fn_output(out, wrapper_fn=self._to_scalable)
        return self._to_scalable(out)

    def __getitem__(self, key: Any):
        return self._to_scalable(self._data[key])

    def __setitem__(self, key: Any, value: Any):
        if is_scalable(value):
            if value.layout is not DataLayout.PANDAS:
                raise ValueError(f'Can only set using {DataLayout.PANDAS} DataFrame and Series')
            value: Union[PandasSeries, PandasDataFrame] = value._data
        ## Stops Pandas SettingWithCopyWarning in output. Ref: https://stackoverflow.com/a/20627316
        pd.options.mode.chained_assignment = None
        self._data[key] = value

    def _sorted_items_dict(self) -> Dict[str, PandasScalableSeries]:
        return {col: self.ScalableSeriesClass(self._data[col], name=col) for col in sorted(self.columns)}

    def apply(self, func, axis=1, args=(), **kwargs):
        return self._data.apply(func, axis=axis, raw=False, result_type=None, args=args, **kwargs)

    @classmethod
    def _concat_sdfs(cls, sdfs: List[ScalableDataFrame], reset_index: bool) -> PandasScalableDataFrame:
        df_list: List[PandasDataFrame] = []
        for sdf in sdfs:
            assert isinstance(sdf, PandasScalableDataFrame)
            df_list.append(sdf._data)
        if reset_index:
            df = pd.concat(df_list, ignore_index=True)
        else:
            df = pd.concat(df_list)
        return cls.of(df, layout=cls.layout)

    def as_pandas(self, **kwargs) -> PandasDataFrame:
        return self._data
