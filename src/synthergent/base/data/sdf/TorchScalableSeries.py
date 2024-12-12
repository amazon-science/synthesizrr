from typing import *
import random, copy, math
import numpy as np
from scipy import stats
import pandas as pd
from pandas.core.frame import Series as PandasSeries
from synthergent.base.util import wrap_fn_output, is_function, all_are_none, all_are_not_none, StringUtil, \
    SampleSizeType, as_list, is_null, optional_dependency
from synthergent.base.constants import DataLayout
from synthergent.base.data.sdf.ScalableSeries import ScalableSeries
from synthergent.base.data.sdf.TensorScalableSeries import TensorScalableSeries
from synthergent.base.data.sdf.NumpyArrayScalableSeries import NumpyArrayScalableSeries
from synthergent.base.data.sdf.ScalableDataFrame import ScalableDataFrame
from pydantic import conint
from pydantic.typing import Literal

TorchScalableSeries = "TorchScalableSeries"
with optional_dependency('torch', error='warn'):
    import torch
    from torch import Tensor as TorchTensor


    class TorchScalableSeries(TensorScalableSeries):
        layout = DataLayout.TORCH
        layout_validator = TensorScalableSeries.is_torch_tensor
        TensorType = torch.Tensor

        def __init__(
                self,
                data: Union[TorchTensor, List, Tuple, Set, np.ndarray, ScalableSeries],
                name: Optional[str] = None,
                **kwargs
        ):
            super(self.__class__, self).__init__(**kwargs)
            if isinstance(data, ScalableSeries):
                data: TorchTensor = data.as_torch(**kwargs)
            elif not isinstance(data, TorchTensor):
                ## Convert to Numpy array first
                data: TorchTensor = NumpyArrayScalableSeries(data, **kwargs).as_torch(**kwargs)
            self.layout_validator(data)
            self._data: TorchTensor = data
            if name is not None and not isinstance(name, (str, int, float)):
                raise ValueError(
                    f'`name` used to construct {self.__class__} can only be int, str or float; '
                    f'found object of type: {type(name)} with value: {name}'
                )
            self._name: Optional[str] = name

        @property
        def tensor_shape(self) -> Tuple[int,]:
            return tuple(self._data.shape)

        @property
        def is_0d(self) -> bool:
            return len(self.tensor_shape) == 0

        def __str__(self):
            name_str: str = '' if self._name is None else f'"{self._name}": '
            out = f"{name_str}PyTorch Tensor of dtype `{self._data.dtype}` with shape {self.tensor_shape}:\n"
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
            out = f"<b>{name_str}PyTorch Tensor of dtype <code>{self._data.dtype}</code> " \
                  f"with shape <code>{self.tensor_shape}</code> values.</b>"
            out += '<hr>'
            if len(self._data) == 0:
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
            if isinstance(data, TorchTensor):
                return ScalableSeries.get_subclass(DataLayout.TORCH)(data)
            elif isinstance(data, np.ndarray):
                return ScalableSeries.get_subclass(DataLayout.TORCH)(torch.from_numpy(data))
            elif isinstance(data, PandasSeries):
                return ScalableSeries.get_subclass(DataLayout.TORCH)(torch.from_numpy(data.values))
            return data

        def as_list(self, **kwargs) -> List:
            return list(self._data.tolist())

        def as_set(self, **kwargs) -> Set:
            return set(self._data.tolist())

        def as_numpy(self, *, stack: bool = False, keep_dims: bool = False, **kwargs) -> np.ndarray:
            if keep_dims:
                ## Keep data as an N-dimensional numpy array.
                return self._data.cpu().detach().numpy()
            ## Return a numpy array of tensors:
            if stack:
                np_arr: np.ndarray = self._data.cpu().detach().numpy()
            else:
                np_arr: np.ndarray = np.array(list(self._data.cpu().detach()), dtype=object)
            return np_arr

        def as_tensor(self, tensor_type: Optional[Literal['torch', 'pt']] = 'torch', **kwargs) -> Optional[Any]:
            return self.as_torch(**kwargs)

        def as_torch(self, error: Literal['raise', 'warn', 'ignore'] = 'raise', **kwargs) -> TorchTensor:
            return self._data

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
            return bool(self._data.isnan().any())

        """
        ---------------------------------------------
        Conversion
        ---------------------------------------------
        """

        def copy(self, deep: bool = False) -> ScalableSeries:
            if bool(deep) is True:
                ## Ref: .clone() vs .clone().detach(): https://discuss.pytorch.org/t/clone-and-detach-in-v0-4-0/16861/21
                return self._constructor(self._data.clone().detach())
            return self._constructor(self._data.clone())

        def bool(self) -> bool:
            data: Optional[Any] = self._convert_0d_tensor_to_python()
            if data is None:
                raise ValueError(
                    f'Can only run `.bool()` with Series having one element; '
                    f'found Torch Tensor {self.tensor_shape}.'
                )
            if not isinstance(data, bool):
                raise ValueError(f'Can only obtain `.bool()` value of Series having True or False data.')
            return data

        def _convert_0d_tensor_to_python(self) -> Optional[Any]:
            if self.is_0d:
                return self._data.item()
            else:
                data: TorchTensor = self._data
                for dim_i in range(data.shape):
                    if data.shape[0] != 1:
                        return None
                    data = data[0]
                return data

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

        def __getitem__(self, key) -> Union[Any, TorchScalableSeries]:
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
            if isinstance(key, TorchTensor):
                ## Ref: https://pytorch.org/docs/stable/tensor_attributes.html#torch-dtype
                if key.dtype in {
                    torch.bool,
                    torch.uint8, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
                }:
                    return self._constructor(self._data[key])
                raise TypeError(f'Indexing with Torch tensors must be done with integer or bool tensors; '
                                f'found tensor with dtype: {key.dtype}')

            if isinstance(key, list):
                return self._constructor(self._data[key])
            raise IndexError(f'Unsupported indexing using: {type(key)} with value: {key}')

        def __setitem__(self, key: Any, value: Any):
            raise NotImplementedError(f'Cannot set at the moment')

        def astype(self, dtype: Union[torch.dtype, str]) -> TorchScalableSeries:
            out: TorchTensor = self._data.to(dtype=dtype)
            return self._constructor(out)

        def item(self) -> Any:
            data: Optional[Any] = self._convert_0d_tensor_to_python()
            if data is None:
                raise ValueError(
                    f'Can only run `.item()` with Series having one element; '
                    f'found Torch Tensor of shape {self.tensor_shape}.'
                )
            return data

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
            return self._constructor(self._data.apply_(lambda x: func(x, *args, **kwargs)))

        """
        ---------------------------------------------
        Computations / descriptive stats
        ---------------------------------------------
        """

        def mode(self, dropna: bool = True) -> ScalableSeries:
            if dropna:
                data: TorchTensor = self.dropna()._data
            else:
                data: TorchTensor = self._data
            return self._constructor(torch.mode(data))

        def mean(
                self,
                axis: int = 0,
                skipna: bool = True,
                level: Literal[None] = None,
                numeric_only: Literal[None] = None,
                **kwargs
        ) -> float:
            if skipna:
                return float(torch.mean(self.dropna()._data))
            return float(torch.mean(self._data))

        def median(
                self,
                axis: int = 0,
                skipna: bool = True,
                level: Literal[None] = None,
                numeric_only: Literal[None] = None,
                **kwargs
        ) -> float:
            if skipna:
                return float(torch.median(self.dropna()._data))
            return float(torch.median(self._data))

        def max(
                self,
                axis: int = 0,
                skipna: bool = True,
                level: Literal[None] = None,
                numeric_only: Literal[None] = None,
                **kwargs,
        ) -> float:
            if skipna:
                return float(np.max(self.dropna()._data))
            return float(np.max(self._data))

        def min(
                self,
                axis: int = 0,
                skipna: bool = True,
                level: Literal[None] = None,
                numeric_only: Literal[None] = None,
                **kwargs,
        ) -> float:
            if skipna:
                return float(np.min(self.dropna()._data))
            return float(np.min(self._data))

        def unique(self) -> ScalableSeries:
            return self._constructor(np.unique(self._data))  ## As of Numpy>=1.21.0, np.unique returns single NaN

        """
        ---------------------------------------------
        Reindexing / selection / label manipulation
        ---------------------------------------------
        """

        def dropna(self, axis: int = 0, inplace: bool = False, how=None) -> Optional[ScalableSeries]:
            ## Ref: https://stackoverflow.com/a/64594975
            def dropna(tensor: TorchTensor):
                shape = tensor.shape
                tensor_reshaped = tensor.reshape(shape[0], -1)
                # Drop all rows containing any nan:
                tensor_reshaped = tensor_reshaped[~torch.any(tensor_reshaped.isnan(), dim=1)]
                # Reshape back:
                tensor = tensor_reshaped.reshape(tensor_reshaped.shape[0], *shape[1:])
                return tensor

            data: TorchTensor = dropna(self._data)
            if inplace:
                self._data = data
                return None
            return self._constructor(data)

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
                data: TorchTensor = self._data.fill_(value)
                if inplace:
                    self._data = data
                    return None
                else:
                    return self._constructor(data)

        def isna(self) -> ScalableSeries:
            return self._constructor(self._nulls)

        def notna(self) -> ScalableSeries:
            return self._constructor(~self._nulls)

        @property
        def _nulls(self) -> TorchTensor:
            ## Returns a boolean Tensor, useful for indexing.
            return torch.isnan(self._data)

        class ILocIndexer:
            def __init__(self, ss: TorchScalableSeries):
                self._ss: TorchScalableSeries = ss

            def __setitem__(self, key, value):
                raise NotImplementedError(f'Can only use for retrieving.')

            def __getitem__(self, key) -> TorchScalableSeries:
                return self._ss[key]

        def __del__(self):
            del self._data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
