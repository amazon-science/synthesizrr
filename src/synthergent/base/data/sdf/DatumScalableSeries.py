import types
from typing import *
import random, copy
import numpy as np
import pandas as pd
from pandas.core.frame import Series as PandasSeries
from synthergent.base.util import is_function, all_are_none, all_are_not_none, is_null, optional_dependency, infer_np_dtype, \
    get_default, StringUtil
from synthergent.base.constants import DataLayout
from synthergent.base.data.sdf.ScalableSeries import ScalableSeries, SS_DEFAULT_NAME
from synthergent.base.data.sdf.ScalableDataFrame import ScalableDataFrame
from pydantic import conint
from pydantic.typing import Literal

DatumScalableSeries = "DatumScalableSeries"


class DatumScalableSeries(ScalableSeries):
    """A single data point pretending to be a series."""

    layout = DataLayout.DATUM
    layout_validator = ScalableSeries.is_datum

    def __init__(
            self,
            data: Optional[Any] = None,
            name: Optional[str] = None,
            empty: bool = False,
            dtype: Optional[Union[np.dtype, Type]] = None,
            str_to_object: bool = True,
            **kwargs
    ):
        super(self.__class__, self).__init__(**kwargs)
        if isinstance(data, self.__class__):
            data = data._data
        # self.layout_validator(data)  ## Does nothing
        self._data: Any = data
        if name is not None and not isinstance(name, (str, int, float)):
            raise ValueError(
                f'`name` used to construct {self.__class__} can only be int, str or float; '
                f'found object of type: {type(name)} with value: {name}'
            )
        self._name: Optional[str] = name
        ## Lazily-computed attributes:
        self.__is_null: Optional[bool] = None
        self.__dtype: Optional[Union[np.dtype, Type]] = dtype
        self.__empty: bool = empty  ## Length-zero series
        self.__str_to_object: bool = str_to_object

    @property
    def _empty_datum_series(self) -> DatumScalableSeries:
        return DatumScalableSeries(empty=True)

    def __len__(self):
        return 0 if self.__empty else 1

    @property
    def _is_null(self) -> bool:
        ## Whether data is None, NaN or NaT
        if self.__is_null is None:
            self.__is_null = is_null(self._data)
        return self.__is_null

    @property
    def dtype(self) -> Union[np.dtype, Type]:
        if self.__empty:
            return np.float_  ## Default value for numpy when running `np.array([]).dtype`
        if self.__dtype is None:
            self.__dtype = infer_np_dtype(
                self._data,
                str_to_object=self.__str_to_object,
                sample_size=1,
            )
        return self.__dtype

    def __str__(self):
        return StringUtil.pretty(self._data)

    def _repr_html_(self):
        name_str: str = '' if self._name is None else f'"{self._name}": '
        out = f"<b>{name_str}Datum of type <code>{self.dtype}</code>: <code>{self._data}</code></b>"
        return f'<div>{StringUtil.pretty(out)}</div>'

    def _repr_latex_(self):
        return self._repr_html_()

    def __repr__(self):
        return str(self)

    @classmethod
    def _to_scalable(cls, data: Any) -> Union[ScalableSeries, Any]:
        return ScalableSeries.get_subclass(DataLayout.DATUM)(data)

    def as_list(self, **kwargs) -> List:
        return [self._data]

    def as_set(self, **kwargs) -> Set:
        return {self._data}

    def as_numpy(self, **kwargs) -> np.ndarray:
        return np.ndarray([self._data], dtype=self.dtype)

    def as_torch(self, error: Literal['raise', 'warn', 'ignore'] = 'raise', **kwargs) -> Optional[Any]:
        with optional_dependency('torch', error=error):
            import torch
            if isinstance(self._data, torch.Tensor):
                return self._data
            if isinstance(self._data, np.ndarray):
                return torch.from_numpy(self._data)
            return torch.from_numpy(self.numpy(**kwargs))
        return None

    def as_pandas(self, **kwargs) -> PandasSeries:
        return pd.Series([self._data], dtype=self.dtype, name=get_default(self._name, SS_DEFAULT_NAME))

    def _to_frame_raw(self, name: Any = SS_DEFAULT_NAME, **kwargs):
        name: Any = get_default(self._name, name)
        return {name: self}

    """
    ---------------------------------------------
    Attributes
    ---------------------------------------------
    """

    def index(self):
        return [0]

    def hasnans(self) -> bool:
        if self.__empty:  ## `pd.Series(np.array([])).hasnans`
            return False
        return self._is_null

    """
    ---------------------------------------------
    Conversion
    ---------------------------------------------
    """

    def copy(self, deep: bool = False) -> ScalableSeries:
        if deep:
            return self._constructor(copy.deepcopy(self._data))
        return self._constructor(copy.copy(self._data))

    def bool(self) -> bool:
        return bool(self._data)

    """
    ---------------------------------------------
    Indexing, iteration
    ---------------------------------------------
    """

    def __add__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        if isinstance(self._data, str):
            return self._constructor(self._data + other)
        return self._constructor(self._data.__add__(other))

    def __radd__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        if isinstance(self._data, str):
            ## Patches the issue with strings not implementing __radd__: https://stackoverflow.com/a/2213422/4900327
            return self._constructor(other + self._data)
        return self._constructor(self._data.__radd__(other))

    @property
    def loc(self) -> Any:
        return self

    @property
    def iloc(self) -> Any:
        return self

    def __getitem__(self, key) -> Union[Any, DatumScalableSeries]:
        if isinstance(key, int):
            if key != 0:
                raise KeyError(f'For {self.__class__.__name__}, can only index using 0')
            return self._data
        ## Refs for slicing:
        ## - https://stackoverflow.com/a/9951672
        ## - https://stackoverflow.com/a/15669247
        if isinstance(key, slice):
            ## E.g. key is `slice(1,10,3)`.
            ## Note that for slices, list(range(10))[slice(1,10,3)] returns [1,4,7]
            if self.__empty or 0 not in list(range(key.start, key.stop, key.step)):
                return self._empty_datum_series
            return self
        if isinstance(key, np.ndarray):
            if key.ndim > 1:
                raise KeyError(f'Can only index with 1-D NumPy array; found array with shape {key.shape}')
            if len(key) > 1:
                raise KeyError(f'Cannot index {self.__class__.__name__} using multiple elements')
        if isinstance(key, np.ndarray):
            key = key[0]
            if key == 0 or key is True:
                return self
            elif key is False:
                return self._empty_datum_series
            raise KeyError(f'When indexing with a Numpy array, must pass the index as 0 or True')
        raise IndexError(f'Unsupported indexing using: {type(key)} with value: {key}')

    def __setitem__(self, key: Any, value: Any):
        raise NotImplementedError(f'Cannot set at the moment')

    def astype(self, dtype: Union[np.dtype, str, Type]) -> DatumScalableSeries:
        if self.__dtype == dtype:
            return self
        if dtype is object:
            return self._constructor(self._data, dtype=object)
        if dtype in {str, 'str'}:
            return self._constructor(str(self._data), dtype=str)
        if isinstance(dtype, str):
            dtype = np.type(dtype)
        out = self._constructor(dtype.__call__(self._data), dtype=dtype)
        if np.issubdtype(dtype, np.unicode_) and self.__str_to_object:
            ## To get true unicode arrays, pass dtype=np.unicode_
            out = out.astype(object)
        return out

    def item(self) -> Any:
        return self._data

    def prod(self, *args, **kwargs) -> Union[int, float]:
        return self.product(*args, **kwargs)

    def product(self, *args, **kwargs) -> Union[int, float]:
        return float(self._data)

    """
    ---------------------------------------------
    Function application, GroupBy & window
    ---------------------------------------------
    """

    def apply(
            self,
            func: Callable,
            convert_dtype: bool = True,
            args: Tuple[Any, ...] = (),
            **kwargs
    ) -> Union[ScalableSeries, ScalableDataFrame]:
        return self._constructor(func(self._data, *args, **kwargs))

    def map(
            self,
            arg: Union[Callable, Dict, ScalableSeries],
            na_action: Optional[Literal['ignore']] = None,
    ) -> ScalableSeries:
        ## Ref: https://github.com/pandas-dev/pandas/blob/221f6362bc25833da87f00015d4d5418ee316eff/pandas/core/base.py#L820
        mapper: Callable = arg
        if isinstance(mapper, dict):
            if hasattr(mapper, "__missing__"):
                # If a dictionary subclass defines a default value method,
                # convert mapper to a lookup function.
                dict_with_default = mapper
                mapper: Callable = lambda k: dict_with_default[k]
            else:
                dict_without_default = mapper
                mapper: Callable = lambda k: dict_without_default.get(k, None)
        if not is_function(mapper):
            raise ValueError(f'Expected input to be function/method/lambda or dict, found: {type(mapper)}')
        if na_action == 'ignore':
            map_fn: Callable = lambda x: mapper(x) if not is_null(x) else x
        else:
            map_fn: Callable = mapper
        return self._constructor(map_fn(self._data))

    """
    ---------------------------------------------
    Computations / descriptive stats
    ---------------------------------------------
    """

    def mode(self, dropna: bool = True) -> ScalableSeries:
        if self._is_null and dropna is True:
            return self._empty_datum_series
        return self

    def mean(
            self,
            axis: int = 0,
            skipna: bool = True,
            level: Literal[None] = None,
            numeric_only: Literal[None] = None,
            **kwargs
    ) -> Union[int, float]:
        if self.__empty or (skipna and self._is_null):
            return np.nan  ## pd.Series([np.nan]).median()
        return float(self._data)

    def median(
            self,
            axis: int = 0,
            skipna: bool = True,
            level: Literal[None] = None,
            numeric_only: Literal[None] = None,
            **kwargs
    ) -> Union[int, float]:
        if self.__empty or (skipna and self._is_null):
            return np.nan  ## pd.Series([np.nan]).median()
        return float(self._data)

    def max(
            self,
            axis: int = 0,
            skipna: bool = True,
            level: Literal[None] = None,
            numeric_only: Literal[None] = None,
            **kwargs,
    ) -> Union[int, float]:
        if self.__empty or (skipna and self._is_null):
            return np.nan  ## pd.Series([np.nan]).max()
        return float(self._data)

    def min(
            self,
            axis: int = 0,
            skipna: bool = True,
            level: Literal[None] = None,
            numeric_only: Literal[None] = None,
            **kwargs,
    ) -> Union[int, float]:
        if self.__empty or (skipna and self._is_null):
            return np.nan  ## pd.Series([np.nan]).max()
        return float(self._data)

    def unique(self) -> ScalableSeries:
        return self

    """
    ---------------------------------------------
    Reindexing / selection / label manipulation
    ---------------------------------------------
    """

    def dropna(self, axis: int = 0, inplace: bool = False, how=None) -> Optional[ScalableSeries]:
        if self._is_null:
            if inplace:
                self._data = None
                self.__empty = True
                return None
            else:
                return self._empty_datum_series
        return self

    def fillna(
            self,
            value: Optional[Union[Any, Dict, ScalableSeries, ScalableDataFrame]] = None,
            method: Optional[Literal["backfill", "bfill", "ffill", "pad"]] = None,
            axis: Literal[0, 'index'] = 0,
            inplace: bool = False,
            limit: Optional[conint(ge=1)] = None,
            downcast: Optional[Dict] = None,
    ) -> Optional[ScalableSeries]:
        if all_are_none(value, method):
            raise ValueError(f'Must specify a fill `value` or `method`.')
        if all_are_not_none(value, method):
            raise ValueError(f'Cannot specify both `value` and `method`.')
        if method is not None:
            raise NotImplementedError(f'`method` is not currently supported')
        if value is not None:
            if self._is_null:
                if inplace:
                    self._data = value
                    self.__empty = False
                    return None
                else:
                    return self._constructor(value)
            return self

    def isna(self) -> ScalableSeries:
        return self._constructor(self._is_null)

    def notna(self) -> ScalableSeries:
        return self._constructor(not self._is_null)
