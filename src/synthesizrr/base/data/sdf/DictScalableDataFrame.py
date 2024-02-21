from typing import *
import math, copy
import numpy as np
import pandas as pd
from pandas.core.frame import Series as PandasSeries, DataFrame as PandasDataFrame
from synthesizrr.base.util import filter_kwargs, as_set, any_are_not_none, multiple_are_not_none, all_are_none, is_scalar, \
    get_default, safe_validate_arguments, as_list, StringUtil, type_str
from synthesizrr.base.constants import DataLayout, Parallelize
from synthesizrr.base.data.sdf.ScalableSeries import ScalableSeries
from synthesizrr.base.data.sdf.ScalableDataFrame import ScalableDataFrame, ScalableDataFrameOrRaw
from synthesizrr.base.data.sdf.NumpyArrayScalableSeries import NumpyArrayScalableSeries
from synthesizrr.base.data.sdf.TensorScalableSeries import TensorScalableSeries
from pydantic import conint
from pydantic.typing import Literal
from collections import deque

DictScalableDataFrame = "DictScalableDataFrame"


class DictScalableDataFrame(ScalableDataFrame):
    layout = DataLayout.DICT
    layout_validator = ScalableDataFrame.is_dict
    ScalableSeriesClass = NumpyArrayScalableSeries

    def __init__(
            self,
            data: Union[Dict, ScalableDataFrame],
            name: Optional[str] = None,
            **kwargs,
    ):
        super(self.__class__, self).__init__(**kwargs)
        if isinstance(data, ScalableDataFrame):
            data: Dict = data.to_dict(**kwargs)
        else:
            self.layout_validator(data)
        self._data: Dict[Any, Union[NumpyArrayScalableSeries, TensorScalableSeries]] = self._make_dict(data)
        self.column_lengths(check=True)
        if name is not None and not isinstance(name, (str, int, float)):
            raise ValueError(
                f'`name` used to construct {self.__class__} can only be int, str or float; '
                f'found object of type: {type_str(name)} with value: {name}'
            )
        self._name: Optional[str] = name

    @classmethod
    def _make_dict(cls, data: Dict) -> Dict[Any, Union[NumpyArrayScalableSeries, TensorScalableSeries]]:
        data = {col: val for col, val in data.items()}
        for col in data.keys():
            data_col = data[col]
            if isinstance(data_col, (NumpyArrayScalableSeries, TensorScalableSeries)):
                data_col: Union[NumpyArrayScalableSeries, TensorScalableSeries] = data_col
            elif ScalableSeries.is_tensor(data_col):
                data_col: ScalableSeries = ScalableSeries.of(data_col, name=col)
            else:
                data_col: NumpyArrayScalableSeries = NumpyArrayScalableSeries(data_col, name=col)
            data_col._name = col
            data[col]: Union[NumpyArrayScalableSeries, TensorScalableSeries] = data_col
        return data

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self), len(self.columns_set)

    @property
    def columns(self) -> List:
        return sorted(list(self.columns_set))

    @property
    def columns_set(self) -> Set:
        return set(self._data.keys())

    def column_lengths(self, check: bool = True) -> Dict[Any, int]:
        col_lens: Dict[Any, int] = {col: col_arr.shape[0] for col, col_arr in self._data.items()}
        if check and len(set(col_lens.values())) > 1:
            raise ValueError(f'Columns are not of equal length; found following lengths: {col_lens}')
        return col_lens

    def column_dtypes(self) -> Dict[Any, np.dtype]:
        return {col: col_arr.dtype for col, col_arr in self._data.items()}

    def __len__(self):
        col_lens = self.column_lengths(check=True)
        return next(iter(col_lens.values()))

    def __str__(self):
        name_str: str = '' if self._name is None else f'"{self._name}": '
        columns: List = self.columns
        length: int = len(self)
        out = f"{name_str}Dict with {len(columns)} column(s) and {length} items in each column."
        width = len(out)
        out += f'\nColumns: {list(columns)}'
        out += '\n' + '-' * width + '\n'
        data_sample = ''
        for col in columns:
            with np.printoptions(threshold=self.display.max_rows + 1, edgeitems=self.display.max_rows // 2):
                data_sample += f'{repr(self._data[col])}\n\n'
        out += data_sample.strip()
        return out

    def _repr_html_(self):
        name_str: str = '' if self._name is None else f'"{self._name}": '
        columns: List = self.columns
        length: int = len(self)
        out = f"<b>{name_str}Dict with <code>{len(columns)}</code> column(s) and <code>{length}</code> elements in each column</b>"
        width = len(out)
        out += '<hr>'
        out += f'<b>Columns:</b><pre>{list(columns)}</pre>'
        out += '<hr><b>Data:</b>'
        data_sample = ''
        for col in columns:
            with np.printoptions(threshold=self.display.max_rows + 1, edgeitems=self.display.max_rows // 2):
                data_sample += f'<pre>{str(self._data[col])}</pre><br>'
        out += f'{data_sample}'
        return f'<div>{out}</div>'

    def _repr_latex_(self):
        return self._repr_html_()

    def copy(self, deep: bool = True) -> ScalableDataFrame:
        data = self._data
        if deep:
            data = copy.deepcopy(data)
        return self._constructor(data, name=self.name)

    def as_raw(self, **kwargs) -> Any:
        return self.as_dict(col_type=None)

    def as_dict(
            self,
            col_type: Optional[Literal['list', 'numpy', list, np.ndarray, 'record']] = None,
            **kwargs
    ) -> Dict[str, Union[List, np.ndarray, Any]]:
        if col_type is None:
            return {
                col: data_col.raw() if isinstance(data_col, TensorScalableSeries) else data_col.numpy()
                for col, data_col in self._data.items()
            }
        if col_type == 'record':
            length: int = len(self)
            if length > 1:
                raise ValueError(f'Cannot convert {self.__class__} of length {length} into record')
            return {col: data_col[0] for col, data_col in self._data.items()}
        if col_type in {'numpy', np.ndarray}:
            return {col: data_col.numpy() for col, data_col in self._data.items()}
        if col_type in {'list', list}:
            return {col: data_col.to_list() for col, data_col in self._data.items()}
        raise NotImplementedError(f'Unsupported `col_type`: {col_type}')

    def as_list_of_dict(self, **kwargs) -> List[Dict]:
        length = len(self)
        out_list: List[Dict] = [{} for _ in range(length)]
        for col in self.columns_set:
            data_col = self._data[col]
            for i in range(length):
                out_list[i][col] = data_col[i]
        return out_list

    def as_numpy(self, sort_columns: bool = True, **kwargs) -> np.recarray:
        sorted_cols: List = sorted(self.columns)
        data: Dict[Any, np.ndarray] = self._get_data_as_dict_of_numpy(sort_columns=sort_columns, **kwargs)
        data_rec_array = []
        np_dtype_list = []
        for col in sorted_cols:
            data_col = data[col]
            data_rec_array.append(data_col)
            np_dtype_list.append((col, data_col.dtype))
        np_dtype = np.dtype(np_dtype_list)
        return np.rec.array(data_rec_array, dtype=np_dtype)

    def as_pandas(self, sort_columns: bool = True, **kwargs) -> PandasDataFrame:
        data: Dict[Any, np.ndarray] = self._get_data_as_dict_of_numpy(sort_columns=sort_columns, **kwargs)
        return pd.DataFrame(data, **filter_kwargs(pd.DataFrame.__init__, **kwargs))

    def _get_data_as_dict_of_numpy(self, sort_columns: bool, **kwargs) -> Dict[Any, np.ndarray]:
        data: Dict[Any, np.ndarray] = {}
        cols: List = as_list(self.columns_set)
        if sort_columns:
            cols: List = sorted(cols)
        for col in cols:
            data_col: np.ndarray = self._data[col].numpy()
            if not isinstance(data_col, np.ndarray):
                raise ValueError(f'Expected data column `{col}` to be NumPy array; found type {type_str(data_col)}')
            data[col] = data_col
        return data

    @safe_validate_arguments
    def rename(
            self,
            mapper: Optional[Union[Dict, Callable]] = None,
            *,
            index: Optional[Union[Dict, Callable]] = None,
            columns: Optional[Union[Dict, Callable]] = None,
            axis: Literal[1, 'columns'] = 1,
            copy: bool = True,
            inplace: bool = False,
            level: Optional[Union[int, str]] = None,
            errors: Literal['ignore', 'raise'] = 'ignore',
    ) -> Optional[ScalableDataFrame]:
        if axis not in {1, 'columns'}:
            raise AttributeError(f'{self.__class__} only supports column-renaming i.e. axis=1.')
        if all_are_none(mapper, columns):
            raise AttributeError(f'{self.__class__} only supports passing `columns` or `mapper` with axis=1')
        mapper: Union[Dict, Callable] = get_default(mapper, columns)
        renamed_data: Dict = self._rename_single_dict(self._data, mapper=mapper)
        if inplace:
            self._data = renamed_data
            return None
        return self._constructor(renamed_data, name=self.name)

    @staticmethod
    def _rename_single_dict(d: Dict, mapper: Union[Dict, Callable]) -> Dict:
        if isinstance(mapper, dict):
            return {mapper.get(k, k): v for k, v in d.items()}
        return {mapper(k): v for k, v in d.items()}

    @classmethod
    def _concat_sdfs(cls, sdfs: List[DictScalableDataFrame], reset_index: bool) -> DictScalableDataFrame:
        cols: Set[Any] = set()
        for sdf in sdfs:
            assert isinstance(sdf, DictScalableDataFrame)
            cols.update(sdf.columns_set)
        concat_dict: Dict = {}
        for col in cols:
            concat_dict[col] = []
            for sdf in sdfs:
                col_lens: Dict[Any, int] = sdf.column_lengths(check=True)
                df_len: int = next(iter(col_lens.values()))
                if col in col_lens:
                    data_col: Union[NumpyArrayScalableSeries, torch.Tensor] = sdf._data[col]
                    if isinstance(data_col, NumpyArrayScalableSeries):
                        data_col: np.ndarray = data_col.numpy()
                    concat_dict[col].append(data_col)
                else:
                    concat_dict[col].append(np.full(df_len, np.nan))
            concat_dict[col]: np.ndarray = np.hstack(concat_dict[col])
        return cls(concat_dict)

    def _sorted_items_dict(self) -> Dict[str, Union[NumpyArrayScalableSeries, TensorScalableSeries]]:
        return {col: self._data[col] for col in sorted(self.columns)}

    @property
    def loc(self) -> Any:
        return self

    @property
    def iloc(self) -> 'ILocIndexer':
        return self.ILocIndexer(self)

    def __getitem__(self, key) -> Union[Any, DictScalableDataFrame, NumpyArrayScalableSeries]:
        ## Refs for slicing:
        ## - https://stackoverflow.com/a/9951672
        ## - https://stackoverflow.com/a/15669247
        if isinstance(key, (int, slice)):
            ## E.g. key is `4` or `slice(1,10,3)`.
            ## Note that for slices, list(range(10))[slice(1,10,3)] returns [1,4,7]
            d: Dict[str, Union[Any, np.array]] = {
                col: l[key]
                for col, l in self._data.items()
            }
            return self._constructor(d, name=self.name)

        if isinstance(key, np.ndarray):
            if key.ndim > 1:
                raise TypeError(f'Can only index with 1-D NumPy array; found array with shape {key.shape}')
            ## Ref: https://stackoverflow.com/a/37727662
            if np.issubdtype(key.dtype, int) or np.issubdtype(key.dtype, bool):
                d: Dict[str, Union[Any, np.ndarray]] = {
                    col: l[key]
                    for col, l in self._data.items()
                }
                return self._constructor(d, name=self.name)
            raise TypeError(f'Indexing with Numpy arrays must be done with integer or bool arrays; '
                            f'found array with dtype: {key.dtype}')

        if isinstance(key, str):
            ## Access columns:
            cols: Set = as_set(key)
            missing_cols: Set = cols - self.columns_set
            if len(missing_cols) > 0:
                raise KeyError(f'Following columns could not be found in Dataframe: {missing_cols}')
            col = key
            if isinstance(self._data[col], TensorScalableSeries):
                return self._data[col].__class__(self._data[col], name=col)
            else:
                return NumpyArrayScalableSeries(self._data[col], name=col)

        if isinstance(key, (list, set)):
            ## Access columns:
            cols: Set = as_set(key)
            missing_cols: Set = cols - self.columns_set
            if len(missing_cols) > 0:
                raise KeyError(f'Following columns could not be found in Dataframe: {missing_cols}')
            return self._constructor({col: self._data[col] for col in key}, name=self.name)

        if isinstance(key, tuple):
            if len(key) != 2:
                raise NotImplementedError(f'Unsupported indexing with tuple having <2 or >2 entries.')
            i, cols = key
            if isinstance(i, tuple):
                raise IndexError(f'Cannot use tuples for indexing')
            out_df = self[i]
            if isinstance(cols, slice):
                if any_are_not_none(cols.start, cols.stop, cols.step):
                    raise IndexError(
                        f'When indexing, the second argument (i.e. columns argument) '
                        f'should be a list of columns or an empty slice'
                    )
                return out_df
            else:
                return out_df[cols]
        raise IndexError(f'Unsupported indexing using: {type_str(key)} with value: {key}')

    def __setitem__(self, key: Any, value: Any):
        length: int = len(self)
        if isinstance(key, (str, int)):
            if isinstance(value, (NumpyArrayScalableSeries, TensorScalableSeries)):
                data_col: Union[NumpyArrayScalableSeries, TensorScalableSeries] = value
            elif ScalableSeries.is_tensor(value):
                data_col: TensorScalableSeries = ScalableSeries.of(value, name=key)
            else:
                if is_scalar(value):
                    value = np.full(length, value)
                data_col: NumpyArrayScalableSeries = NumpyArrayScalableSeries(value)
            data_col_length: int = len(data_col)
            if data_col_length != length:
                raise ValueError(f'Expected input value to have same length {length}; found length {data_col_length}')
            self._data[key]: Union[NumpyArrayScalableSeries, TensorScalableSeries] = data_col
        elif isinstance(key, tuple):
            ## Supports assignment like: sdf[:, 'colA'] = 123
            if not len(key) == 2:
                raise ValueError(f'Can only set using a tuple of two elements; found {len(key)} elements:\n{key}')
            first = key[0]
            second = key[1]
            if isinstance(first, slice) and first.start is first.step is first.stop is None:
                self[second] = value  ## Call __setitem__ recursively
            else:
                raise NotImplementedError(f'Sliced assignment not supported at the moment')
        else:
            raise NotImplementedError(f'Cannot set using key of type {type_str(key)}.')

    def _stream_in_memory_sdf_convert_to_streaming_layout(self, stream_as: DataLayout) -> ScalableDataFrame:
        ## First select the chunks (as dicts), and then convert it to the target data layout only just before
        ## returning...selecting small chunks is faster in dict than in other formats.
        return self

    class ILocIndexer:
        def __init__(self, sdf: DictScalableDataFrame):
            self._sdf: DictScalableDataFrame = sdf

        def __setitem__(self, key, value):
            raise NotImplementedError(f'Can only use for retrieving.')

        def __getitem__(self, key) -> DictScalableDataFrame:
            return self._sdf[key]

    def applymap(self, func, na_action: Optional[Literal['ignore']] = None, **kwargs) -> ScalableDataFrame:
        return self._constructor(
            {
                col: col_arr.map(lambda x: func(x, **kwargs), na_action=na_action)
                for col, col_arr in self._data.items()
            },
            name=self.name,
        )

    def apply(self, *args, **kwargs):
        return self.to_layout(layout=DataLayout.LIST_OF_DICT).apply(*args, **kwargs)

#
#     def __getattr__(self, attr_name: str):
#         """Forwards calls to the respective method of PandasDataFrame class."""
#         data = self.__dict__[RAW_DATA_MEMBER]
#         data = pd.DataFrame(data)
#         if not hasattr(data, attr_name):
#             raise AttributeError(
#                 f'Neither {self.__class__.__name__} nor {PandasDataFrame} classes have attribute "{attr_name}"'
#             )
#         out = getattr(data, attr_name)
#         if isinstance(out, PandasSeries):
#             return DictScalableSeries(out.to_list())
#         if isinstance(out, PandasDataFrame):
#             return self.__class__(out.to_dict(orient='records'))
#         return out
#
#     @classproperty
#     def layout(self) -> 'DataLayout':
#         return DataLayout.DICT
#
#     @property
#     def columns(self) -> List:
#         return sorted(list(self.columns_set))
#
#     @property
#     def columns_set(self) -> Set:
#         columns: Set[str] = set()
#         for idx, d in enumerate(self._data):
#             columns.update(set(d.keys()))
#         return columns
#
#     @property
#     def shape(self) -> Tuple[int, int]:
#         return (len(self), len(self.columns_set))
#
#     def __str__(self):
#         columns: Set = self.columns_set
#         out = f"List of {len(self)} dicts with {len(columns)} column(s)."
#         width = len(out)
#         out += '\n' + '-' * width + '\n'
#         out += f'Columns: {list(columns)}'
#         out += '\n' + '-' * width + '\n'
#         if len(self._data) == 0:
#             data_sample = str([])
#         elif len(self._data) == 1:
#             data_sample = str(self._data[0])
#         elif len(self._data) == 2:
#             data_sample = f'{str(self._data[0])}\n{str(self._data[1])}'
#         else:
#             data_sample = f'{str(self._data[0])}\n...\n{str(self._data[-1])}'
#         out += data_sample
#         return out
#
#     def _repr_html_(self):
#         columns: Set = self.columns_set
#         out = f"<b>List of <code>{len(self)}</code> dicts with <code>{len(columns)}</code> column(s)</b>"
#         width = len(out)
#         out += '<hr>'
#         out += f'<b>Columns:</b><pre>{list(columns)}</pre>'
#         out += '<hr><b>Data:</b>'
#         if len(self._data) == 0:
#             data_sample = ''
#         elif len(self._data) == 1:
#             data_sample = f'<pre>{str(self._data[0])}</pre>'
#         elif len(self._data) == 2:
#             data_sample = f'<pre>{str(self._data[0])}\n{str(self._data[1])}</pre>'
#         else:
#             data_sample = f'<pre>{str(self._data[0])}\n...\n{str(self._data[-1])}</pre>'
#         out += f'{data_sample}'
#         return f'<div>{out}</div>'
#
#     def _repr_latex_(self):
#         return self._repr_html_()
#
#     def __len__(self):
#         return len(self._data)
#
#     def __getitem__(self, key) -> Union[Any, DictScalableDataFrame, 'DictScalableSeries']:
#         ## Refs for slicing:
#         ## - https://stackoverflow.com/a/9951672
#         ## - https://stackoverflow.com/a/15669247
#         if isinstance(key, (int, slice)):
#             ## E.g. key is `4` or `slice(1,10,3)`.
#             ## Note that for slices, list(range(10))[slice(1,10,3)] returns [1,4,7]
#             return ScalableDataFrame.of(self._data[key], layout=DataLayout.LIST_OF_DICT)
#         if isinstance(key, np.ndarray):
#             if len(key.shape) > 1:
#                 raise TypeError(f'Can only index with 1-D NumPy array; found array with shape {key.shape}')
#             if not np.issubdtype(key.dtype, np.integer):  ## Ref: https://stackoverflow.com/a/37727662
#                 raise TypeError(f'Indexing with Numpy arrays must be done with integer arrays; '
#                                 f'found array with dtype: {key.dtype}')
#             return ScalableDataFrame.of([self._data[idx] for idx in key], layout=DataLayout.LIST_OF_DICT)
#         if isinstance(key, (str, list)):
#             ## Access columns:
#             cols: Set = set(as_list(key))
#             missing_cols: Set = cols - self.columns_set
#             if len(missing_cols) > 0:
#                 raise KeyError(f'Following columns could not be found in Dataframe: {missing_cols}')
#             if len(cols) == 1:
#                 col = cols.pop()
#                 out_vals = [
#                     d[col]
#                     for d in self._data
#                 ]
#                 if len(out_vals) == 1:
#                     return out_vals[0]
#                 return DictScalableSeries(out_vals)
#             else:
#                 out_dicts = [
#                     {
#                         k: v for k, v in d.items()
#                         if k in cols
#                     }
#                     for d in self._data
#                 ]
#                 return DictScalableDataFrame(out_dicts)
#
#         if isinstance(key, tuple):
#             if len(key) != 2:
#                 raise NotImplementedError(f'Unsupported indexing with tuple having <2 or >2 entries.')
#             idx, cols = key
#             if isinstance(idx, tuple):
#                 raise IndexError(f'Cannot use tuples for indexing')
#             out_df = self[idx]
#             if isinstance(cols, slice):
#                 if any_are_not_none(cols.start, cols.stop, cols.step):
#                     raise IndexError(f'Second argument when indexing (i.e. columns) should be a list or an empty slice')
#                 return out_df
#             else:
#                 return out_df[cols]
#         raise IndexError(f'Unsupported indexing using: {type(key)} with value: {key}')
#
#     def __setitem__(self, key: Any, value: Union[List, Tuple, Any]):
#         ## TODO: implement slicing
#         assert isinstance(key, SCALAR_TYPES)
#         if isinstance(value, list):
#             assert len(value) == len(self._data)
#             for i, (d, val) in enumerate(zip(self._data, value)):
#                 d[key] = val
#         elif isinstance(value, (str, int, float, bool, type(None))):
#             for i, d in enumerate(self._data):
#                 d[key] = value
#         else:
#             raise NotImplementedError(f'Unsupported type for value: {type(value)}')
#
#     def apply(self, func, axis=1, args=(), **kwargs):
#         super().apply(func=func, axis=axis, args=args, **kwargs)
#         return DictScalableSeries([
#             func(d)
#             for d in self._data
#         ])

#     def as_list_of_dict(self) -> List[Dict]:
#         return self._data
#
#
#     def as_dask(self, *args, **kwargs) -> DaskDataFrame:
#         import dask.dataframe as dd
#         return dd.from_pandas(self.pandas(), *args, **kwargs)
#
#     def stream(
#             self,
#             num_rows: Optional[int] = None,
#             layout: 'DataLayout' = None,
#             raw: bool = False,
#             shuffle: bool = False,
#             seed: int = DEFAULT_RANDOM_SEED,
#     ) -> Union[ScalableDataFrame, 'ScalableDataFrameRawType']:
#         IntegerRange(1, inf, allow_none=True).check_is_valid(num_rows, name='num_rows')
#         BoolRange().check_is_valid(shuffle, name='shuffle')
#         BoolRange().check_is_valid(raw, name='raw')
#         layout: DataLayout = DataLayout.from_str(layout, raise_error=False)
#         if layout is DataLayout.DASK:
#             raise AttributeError(
#                 f'Cannot stream data as a Dask DataFrame, `layout` must be an in-memory datatype on the client, '
#                 f'such as {DataLayout.LIST_OF_DICT} or {DataLayout.PANDAS}'
#             )
#         idx: np.ndarray = np.arange(0, len(self))
#         if shuffle:
#             ## Ref: https://stackoverflow.com/a/47742676
#             idx = np.random.RandomState(seed=seed).permutation(idx)
#         if num_rows is None:
#             num_rows = 1
#         ## Select chunks, then convert it to the respective data layout...selecting small chunks is faster in list.
#         df: DictScalableDataFrame = DictScalableDataFrame([self._data[i] for i in idx])
#         for i in range(0, len(df), num_rows):
#             df_chunk: ScalableDataFrame = ScalableDataFrame.of(df[i:i + num_rows], layout=layout)
#             if raw:
#                 yield df_chunk._data
#             else:
#                 yield df_chunk
#
#     class ListOfDictScalableIndexer:
#         def __init__(self, df: DictScalableDataFrame):
#             self.__df: DictScalableDataFrame = df
#
#         def __setitem__(self, key, value):
#             raise NotImplementedError(f'Can only use for retrieving')
#
#         def __getitem__(self, key) -> DictScalableDataFrame:
#             return self.__df[key]
#
#
# class DictScalableSeries(ScalableSeries):
#     def __init__(self, data: List[Any]):
#         super(self.__class__, self).__init__()
#         self._data: List[Any] = as_list(data)
#
#     def __getattr__(self, attr_name: str):
#         """Forwards calls to the respective method of PandasSeries class."""
#         data = self.__dict__[RAW_DATA_MEMBER]
#         data = pd.Series(data)
#         if not hasattr(data, attr_name):
#             raise AttributeError(
#                 f'Neither {self.__class__.__name__} nor {type(self._data)} classes have attribute "{attr_name}"'
#             )
#         out = getattr(data, attr_name)
#
#         if isinstance(out, (types.FunctionType, types.LambdaType, types.MethodType)):
#             return wrap_fn_output(out, IsInstanceClass=PandasSeries, WrapperClass=self.__class__)
#         if isinstance(out, PandasSeries):
#             return self.__class__(out.to_list())
#         return out
#
#     def __len__(self):
#         return len(self._data)
#
#     def __str__(self):
#         out = f"List of {len(self)} values."
#         out += '\n' + '-' * len(out) + '\n'
#         if len(self._data) == 0:
#             data_sample = str([])
#         elif len(self._data) == 1:
#             data_sample = str(self._data[0])
#         elif len(self._data) == 2:
#             data_sample = f'{str(self._data[0])}\n{str(self._data[1])}'
#         else:
#             data_sample = f'{str(self._data[0])}\n...\n{str(self._data[-1])}'
#         out += data_sample
#         return out
#
#     def _repr_html_(self):
#         out = f"<b>List of <code>{len(self)}</code> values.</b>"
#         out += '<hr>'
#         if len(self._data) == 0:
#             data_sample = ''
#         elif len(self._data) == 1:
#             data_sample = f'<pre>{str(self._data[0])}</pre>'
#         elif len(self._data) == 2:
#             data_sample = f'<pre>{str(self._data[0])}\n{str(self._data[1])}</pre>'
#         else:
#             data_sample = f'<pre>{str(self._data[0])}\n...\n{str(self._data[-1])}</pre>'
#         out += f'{data_sample}'
#         return f'<div>{out}</div>'
#
#     def _repr_latex_(self):
#         return self._repr_html_()
#
#     def __repr__(self):
#         return str(self)
#
#     def to_tensor(self) -> torch.Tensor:
#         return torch.stack(self._data, dim=0)
#
#     def mean(self) -> float:
#         return float(np.mean(self._data))
#
#     def median(self) -> Union[int, float]:
#         return float(np.median(self._data))
