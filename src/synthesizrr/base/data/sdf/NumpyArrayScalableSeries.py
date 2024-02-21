from typing import *
import random, copy, math
import numpy as np
from scipy import stats
import pandas as pd
from pandas.core.frame import Series as PandasSeries
from synthesizrr.base.util import wrap_fn_output, is_function, all_are_none, all_are_not_none, StringUtil, infer_np_dtype, \
    SampleSizeType, as_list, is_null, get_default
from synthesizrr.base.constants import DataLayout
from synthesizrr.base.data.sdf.ScalableSeries import ScalableSeries, SS_DEFAULT_NAME
from synthesizrr.base.data.sdf.ScalableDataFrame import ScalableDataFrame
from pydantic import conint
from pydantic.typing import Literal

NumpyArrayScalableSeries = "NumpyArrayScalableSeries"


class NumpyArrayScalableSeries(ScalableSeries):
    layout = DataLayout.NUMPY
    layout_validator = ScalableSeries.is_numpy_array

    def __init__(
            self,
            data: Union[List, Tuple, Set, np.ndarray, ScalableSeries],
            name: Optional[str] = None,
            np_inference_sample_size: SampleSizeType = int(1e3),
            str_to_object: bool = True,
            deepcopy: bool = False,
            **kwargs
    ):
        super(self.__class__, self).__init__(**kwargs)
        self._str_to_object: bool = str_to_object
        if isinstance(data, NumpyArrayScalableSeries):
            ## Do not validate
            if deepcopy:
                self._data = data._data.copy()  ## Deep copy
            else:
                self._data = data._data  ## Shallow copy
        else:
            if isinstance(data, ScalableSeries):
                data: np.ndarray = data.numpy(**kwargs)
            elif isinstance(data, PandasSeries):
                data: np.ndarray = data.values
            elif isinstance(data, np.ndarray):
                if data.ndim > 1:
                    raise ValueError(f'Can only create a series of 1D NumPy arrays.')
            elif self.is_tensor(data):
                data: np.ndarray = self.of(data).numpy(keep_dims=False, **kwargs)
            else:
                data: np.ndarray = self._convert_list_data_to_numpy(
                    as_list(data),
                    sample_size=np_inference_sample_size,
                    str_to_object=self._str_to_object,
                )
            assert isinstance(data, np.ndarray), f'Found type: {type(data)}'
            if np.issubdtype(data.dtype, np.unicode_) and str_to_object:
                data: np.ndarray = data.astype(object)  ## Do not allow unicode arrays with dtype like "U<8"
            self._data: np.ndarray = data
        self.layout_validator(self._data)
        if name is not None and not isinstance(name, (str, int, float)):
            raise ValueError(
                f'`name` used to construct {self.__class__} can only be int, str or float; '
                f'found object of type: {type(name)} with value: {name}'
            )
        self._name: Optional[str] = name
        ## TODO: add self._index

    @staticmethod
    def _convert_list_data_to_numpy(data: List, sample_size: SampleSizeType, str_to_object: bool) -> np.ndarray:
        orig_len = len(data)
        if orig_len == 1:  ## For some reason, 1 element takes longer to process than 2 elements.
            data = [data[0], data[0]]
        inferred_np_dtype = infer_np_dtype(
            data,
            str_to_object=str_to_object,
            sample_size=sample_size,
            return_str_for_collection=True,
        )
        if inferred_np_dtype == 'collection':
            np_data: np.ndarray = np.empty(len(data), dtype=object)
            for i, item in enumerate(data):
                np_data[i] = item
        else:
            np_data: np.ndarray = np.array(
                data,
                dtype=inferred_np_dtype
            )
            if np_data.ndim > 1:  ## We have a nested list/tuple structure
                np_data: np.ndarray = np.empty(len(data), dtype=object)
                np_data[:] = data
            elif not (
                    np.issubdtype(np_data.dtype, bool) or
                    np.issubdtype(np_data.dtype, int) or
                    np.issubdtype(np_data.dtype, float) or
                    np.issubdtype(np_data.dtype, complex) or
                    np.issubdtype(np_data.dtype, object)
            ):
                np_data: np.ndarray = np.array(data, dtype=object)
        if orig_len == 1:
            np_data = np_data[:1]
        return np_data

    def __len__(self):
        return len(self._data)

    def __str__(self):
        name_str: str = '' if self._name is None else f'"{self._name}": '
        out = f"{name_str}Numpy array of dtype `{self._data.dtype}` with {len(self)} items:\n"
        # out += '\n' + '-' * len(out) + '\n'
        length: int = len(self._data)
        if length == 0:
            data_sample = str([])
        elif length == 1:
            data_sample = f'[0]: {StringUtil.pretty(self._data[0])}'
        elif length == 2:
            data_sample = f'[0]: {StringUtil.pretty(self._data[0])}\n[1]: {StringUtil.pretty(self._data[1])}'
        else:
            data_sample = f'[{StringUtil.pad_zeros(0, length - 1)}]: {StringUtil.pretty(self._data[0])}\n' \
                          f'...\n' \
                          f'[{StringUtil.pad_zeros(length - 1, length - 1)}]: {StringUtil.pretty(self._data[-1])}'
        out += data_sample
        return out

    def _repr_html_(self):
        name_str: str = '' if self._name is None else f'"{self._name}": '
        out = f"<b>{name_str}Numpy array of dtype <code>{self._data.dtype}</code> with <code>{len(self)}</code> values.</b>"
        out += '<hr>'
        length: int = len(self._data)
        if length == 0:
            data_sample = ''
        else:
            with np.printoptions(threshold=self.display.max_rows + 1, edgeitems=self.display.max_rows // 2):
                data_sample = f'<pre>{str(self._data)}</pre><br>'
        out += f'{data_sample}'
        return f'<div>{out}</div>'

    def _repr_latex_(self):
        return self._repr_html_()

    def __repr__(self):
        return str(self)

    @classmethod
    def _to_scalable(cls, data: Any) -> Union[ScalableSeries, Any]:
        if isinstance(data, np.ndarray):
            return ScalableSeries.get_subclass(DataLayout.NUMPY)(data)
        return data

    def __getattr__(self, attr_name: str) -> Union[Any, NumpyArrayScalableSeries]:
        """Forwards calls to the respective method of Numpy ndarray class."""
        out = super().__getattr__(attr_name)
        if is_function(out):
            return wrap_fn_output(out, wrapper_fn=self._to_scalable)
        if isinstance(out, np.ndarray):
            return self._constructor(out)
        return out

    def as_set(self) -> Set:
        return set(np.unique(self._data))

    def nunique(self, dropna: bool = True) -> int:
        uniq = np.unique(self._data, equal_nan=True)
        if dropna:
            uniq: np.ndarray = uniq[~pd.isnull(uniq)]
        return len(uniq)

    def as_list(self, **kwargs) -> List:
        return list(self._data.tolist())

    def as_numpy(self, *, stack: bool = False, **kwargs) -> np.ndarray:
        np_arr: np.ndarray = self._data
        if stack:
            np_arr: np.ndarray = np.vstack(np_arr)
        return np_arr

    def as_pandas(self, **kwargs) -> PandasSeries:
        return pd.Series(self._data, name=get_default(self._name, SS_DEFAULT_NAME))

    def _to_frame_raw(self, name: Any = SS_DEFAULT_NAME, layout: Optional[DataLayout] = None, **kwargs):
        layout: Optional[DataLayout] = DataLayout.from_str(layout, raise_error=False)
        name: Any = get_default(self._name, name)
        if layout is None or layout is DataLayout.DICT:
            return {name: self._data}  ## Dict layout
        elif layout is DataLayout.LIST_OF_DICT:
            return [{name: x} for x in self._data]  ## List of Dict layout
        raise NotImplementedError(f'Unsupported layout: {layout}')

    """
    ---------------------------------------------
    Attributes
    ---------------------------------------------
    """

    @property
    def index(self):
        return np.arange(0, stop=len(self))

    @property
    def hasnans(self) -> bool:
        if np.issubdtype(self._data.dtype, int):  ## Ref: https://stackoverflow.com/a/37727662
            return False
        if np.issubdtype(self._data.dtype, float):  ## Ref: https://stackoverflow.com/a/37727662
            return np.isnan(np.dot(self._data, self._data))
        return np.any(self._nulls)

    """
    ---------------------------------------------
    Conversion
    ---------------------------------------------
    """

    def copy(self, deep: bool = False) -> ScalableSeries:
        if bool(deep) is True:
            return self._constructor(copy.deepcopy(self._data))
        return self._constructor(self._data)

    def bool(self) -> bool:
        if len(self) != 1:
            raise ValueError(f'Can only run `.bool()` with Series having one element; found {len(self)}.')
        data = self._data[0]
        if not np.issubdtype(self._data.dtype, bool):
            raise ValueError(f'Can only obtain `.bool()` value of Series having True or False data.')
        return bool(data)

    """
    ---------------------------------------------
    Indexing, iteration
    ---------------------------------------------
    """

    def loc(self) -> Any:
        return self

    @property
    def iloc(self) -> 'ILocIndexer':
        return self.ILocIndexer(self)

    def __getitem__(self, key) -> Union[Any, NumpyArrayScalableSeries]:
        ## Refs for slicing:
        ## - https://stackoverflow.com/a/9951672
        ## - https://stackoverflow.com/a/15669247
        if isinstance(key, int):
            ## E.g. key is `4`
            ## Note that for slices, list(range(10))[slice(1,10,3)] returns [1,4,7]
            return self._data[key]
        if isinstance(key, slice):
            ## E.g. key is `slice(1,10,3)`.
            ## Note that for slices, list(range(10))[slice(1,10,3)] returns [1,4,7]
            return self._constructor(self._data[key])
        if isinstance(key, np.ndarray):
            if key.ndim > 1:
                raise TypeError(f'Can only index with 1-D NumPy array; found array with shape {key.shape}')
            ## Ref: https://stackoverflow.com/a/37727662
            if np.issubdtype(key.dtype, int) or np.issubdtype(key.dtype, bool):
                return self._constructor(self._data[key])
            raise TypeError(f'Indexing with Numpy arrays must be done with integer or bool arrays; '
                            f'found array with dtype: {key.dtype}')

        if isinstance(key, list):
            return self._constructor(self._data[key])
        raise IndexError(f'Unsupported indexing using: {type(key)} with value: {key}')

    def __setitem__(self, key: Any, value: Any):
        raise NotImplementedError(f'Cannot set at the moment')

    def astype(self, dtype: Union[np.dtype, str]) -> NumpyArrayScalableSeries:
        out: np.ndarray = self._data.astype(dtype)
        if dtype in {str, 'str'} and np.issubdtype(out.dtype, np.unicode_):
            ## To get true unicode arrays, pass dtype=np.unicode_
            out = out.astype(object)
        return self._constructor(out)

    def item(self) -> Any:
        if len(self) != 1:
            raise ValueError(f'Can only run `.item()` with Series having one element; found {len(self)}.')
        data = self._data[0]
        return data

    def prod(self, *args, **kwargs) -> Union[int, float]:
        return self.product(*args, **kwargs)

    def product(self, *args, **kwargs) -> Union[int, float]:
        return pd.Series(self._data).product(*args, **kwargs)

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
        return self._constructor([func(x, *args, **kwargs) for x in self._data])

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
        return self._constructor([map_fn(x) for x in self._data])

    """
    ---------------------------------------------
    Computations / descriptive stats
    ---------------------------------------------
    """

    def mode(self, dropna: bool = True) -> ScalableSeries:
        if dropna:
            nan_policy = 'omit'
            data: np.ndarray = self.dropna()._data
        else:
            nan_policy = 'propagate'
            data: np.ndarray = self._data
        return self._constructor(stats.mode(data, nan_policy=nan_policy).mode)

    def mean(
            self,
            axis: int = 0,
            skipna: bool = True,
            level: Literal[None] = None,
            numeric_only: Literal[None] = None,
            **kwargs
    ) -> Union[int, float]:
        if skipna:
            return np.mean(self.dropna()._data)
        return np.mean(self._data)

    def median(
            self,
            axis: int = 0,
            skipna: bool = True,
            level: Literal[None] = None,
            numeric_only: Literal[None] = None,
            **kwargs
    ) -> Union[int, float]:
        if skipna:
            return np.median(self.dropna()._data)
        return np.median(self._data)

    def max(
            self,
            axis: int = 0,
            skipna: bool = True,
            level: Literal[None] = None,
            numeric_only: Literal[None] = None,
            **kwargs,
    ) -> Union[int, float]:
        if skipna:
            return np.max(self.dropna()._data)
        return np.max(self._data)

    def min(
            self,
            axis: int = 0,
            skipna: bool = True,
            level: Literal[None] = None,
            numeric_only: Literal[None] = None,
            **kwargs,
    ) -> Union[int, float]:
        if skipna:
            return np.min(self.dropna()._data)
        return np.min(self._data)

    def unique(self) -> ScalableSeries:
        return self._constructor(np.unique(self._data))  ## As of Numpy>=1.21.0, np.unique returns single NaN

    """
    ---------------------------------------------
    Reindexing / selection / label manipulation
    ---------------------------------------------
    """

    def dropna(self, axis: int = 0, inplace: bool = False, how=None) -> Optional[ScalableSeries]:
        if inplace:
            self._data = self._data[~self._nulls]
            return None
        return self._constructor(self._data[~self._nulls])

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
            if isinstance(value, (dict, ScalableSeries, ScalableDataFrame)):
                raise NotImplementedError(f'Unsupported `value` of type {type(value)}')
            dtype = self._data.dtype
            if (np.issubdtype(dtype, int) and not isinstance(value, int)) or \
                    (np.issubdtype(dtype, float) and not isinstance(value, float)):
                data: np.ndarray = self._data.astype(object)
            else:
                data: np.ndarray = self._data
            if inplace:
                data[self._nulls] = value
                self._data = data
                return None
            else:
                data: np.ndarray = data.copy()
                data[self._nulls] = value
                return self._constructor(data)

    def isna(self) -> ScalableSeries:
        return self._constructor(self._nulls)

    def notna(self) -> ScalableSeries:
        return self._constructor(~self._nulls)

    @property
    def _nulls(self) -> np.ndarray:
        ## Returns a NumPy array, useful for indexing.
        return pd.isnull(self._data)

    class ILocIndexer:
        def __init__(self, ss: NumpyArrayScalableSeries):
            self._ss: NumpyArrayScalableSeries = ss

        def __setitem__(self, key, value):
            raise NotImplementedError(f'Can only use for retrieving.')

        def __getitem__(self, key) -> NumpyArrayScalableSeries:
            return self._ss[key]
