import copy
from typing import *
import numpy as np
import pandas as pd
import math, pprint
from pandas.core.frame import Series as PandasSeries, DataFrame as PandasDataFrame
from synthergent.base.util import as_list, any_are_not_none, all_are_none, get_current_fn_name, is_scalar, is_null, \
    filter_kwargs, get_default, safe_validate_arguments, StringUtil
from synthergent.base.constants import DataLayout, Parallelize
from synthergent.base.data.sdf.ScalableSeries import ScalableSeries
from synthergent.base.data.sdf import ScalableDataFrame, ScalableDataFrameOrRaw, RAW_DATA_MEMBER
from synthergent.base.data.sdf.NumpyArrayScalableSeries import NumpyArrayScalableSeries
from pydantic import conint
from pydantic.typing import Literal
from collections import deque

ListOfDictScalableDataFrame = "ListOfDictScalableDataFrame"


class ListOfDictScalableDataFrame(ScalableDataFrame):
    layout = DataLayout.LIST_OF_DICT
    layout_validator = ScalableDataFrame.is_list_of_dict
    ScalableSeriesClass = NumpyArrayScalableSeries

    def __init__(self, data: Union[List[Dict], ScalableDataFrame], name: Optional[str] = None, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        if isinstance(data, ScalableDataFrame):
            data: List[Dict] = data.to_list_of_dict(**kwargs)
        else:
            self.layout_validator(data)
        self._data: List[Dict] = as_list(data)
        if name is not None and not isinstance(name, (str, int, float)):
            raise ValueError(
                f'`name` used to construct {self.__class__} can only be int, str or float; '
                f'found object of type: {type(name)} with value: {name}'
            )
        self._name: Optional[str] = name

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self), len(self.columns_set))

    @property
    def columns(self) -> List:
        return sorted(list(self.columns_set))

    @property
    def columns_set(self) -> Set:
        columns: Set[str] = set()
        for _, d in enumerate(self._data):
            columns.update(set(d.keys()))
        return columns

    def __len__(self):
        return len(self._data)

    def __str__(self):
        columns: Set = self.columns_set
        name_str: str = '' if self._name is None else f'"{self._name}": '
        out = f"{name_str}List of {len(self)} dict(s) with {len(columns)} column(s)."
        width = len(out)
        out += '\n' + '-' * width + '\n'
        out += f'Columns: {list(columns)}'
        out += '\n' + '-' * width + '\n'
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
        columns: Set = self.columns_set
        out = f"<b>{name_str}List of <code>{len(self)}</code> dicts with <code>{len(columns)}</code> column(s)</b>"
        width = len(out)
        out += '<hr>'
        out += f'<b>Columns:</b><pre>{list(columns)}</pre>'
        out += '<hr><b>Data:</b>'
        length: int = len(self._data)
        if length == 0:
            data_sample = ''
        elif length == 1:
            data_sample = f'<pre>{StringUtil.pretty(self._data[0])}</pre>'
        else:
            head_len: int = math.ceil(self.display.max_rows / 2)
            tail_len: int = math.floor(self.display.max_rows / 2)
            if len(self) <= head_len + tail_len:
                data_sample = '\n'.join([StringUtil.pretty(x) for x in self._data])
                data_sample: str = f'<pre>{data_sample}</pre>'
            else:
                head_sample: str = '\n'.join([StringUtil.pretty(x) for x in self._data[:head_len]])
                tail_sample: str = '\n'.join([StringUtil.pretty(x) for x in self._data[-tail_len:]])
                data_sample: str = f'<pre>{head_sample}\n...\n{tail_sample}</pre>'
        out += f'{data_sample}'
        return f'<div>{out}</div>'

    def _repr_latex_(self):
        return self._repr_html_()

    def __getattr__(self, attr_name: str):
        attr = super().__getattr__(attr_name)
        if attr is not None:
            return attr
        """Forwards calls to the respective method of PandasDataFrame class."""
        data = self.__dict__[RAW_DATA_MEMBER]
        data = pd.DataFrame(data)
        if not hasattr(data, attr_name):
            raise AttributeError(
                f'Neither {self.__class__.__name__} nor {PandasDataFrame} classes have attribute "{attr_name}"'
            )
        out = getattr(data, attr_name)
        if isinstance(out, PandasSeries):
            return isinstance(out.to_list(), ListOfDictScalableDataFrame)
        if isinstance(out, PandasDataFrame):
            return self._constructor(out.to_dict(orient='records'), name=self.name)
        return out

    @property
    def loc(self) -> Any:
        return self

    @property
    def iloc(self) -> 'ILocIndexer':
        return self.ILocIndexer(self)

    def _sorted_items_dict(self) -> Dict[str, NumpyArrayScalableSeries]:
        return {col: self.ScalableSeriesClass(self[col], name=col) for col in sorted(self.columns)}

    def __getitem__(self, key) -> Union[Any, ListOfDictScalableDataFrame, NumpyArrayScalableSeries]:
        ## Refs for slicing:
        ## - https://stackoverflow.com/a/9951672
        ## - https://stackoverflow.com/a/15669247
        if isinstance(key, (int, slice)):
            ## E.g. key is `4` or `slice(1,10,3)`.
            ## Note that for slices, list(range(10))[slice(1,10,3)] returns [1,4,7]
            return self._constructor(as_list(self._data[key]), name=self.name)

        if isinstance(key, np.ndarray):
            if key.ndim > 1:
                raise TypeError(f'Can only index with 1-D NumPy array; found array with shape {key.shape}')
            ## Ref: https://stackoverflow.com/a/37727662
            if np.issubdtype(key.dtype, int):
                return self._constructor([self._data[i] for i in key], name=self.name)
            elif np.issubdtype(key.dtype, bool):
                return self._constructor(
                    [self._data[i] for i, should_select in enumerate(key) if should_select],
                    name=self.name
                )
            raise TypeError(f'Indexing with Numpy arrays must be done with integer or bool arrays; '
                            f'found array with dtype: {key.dtype}')

        if isinstance(key, str):
            ## Access columns:
            cols: Set = set(as_list(key))
            missing_cols: Set = cols - self.columns_set
            if len(missing_cols) > 0:
                raise KeyError(f'Following columns could not be found in Dataframe: {missing_cols}')
            col = key
            out_vals = [
                d[col]
                for d in self._data
            ]
            return self.ScalableSeriesClass(out_vals, name=col)

        if isinstance(key, (list, set)):
            cols: Set = set(as_list(key))
            missing_cols: Set = cols - self.columns_set
            if len(missing_cols) > 0:
                raise KeyError(f'Following columns could not be found in Dataframe: {missing_cols}')
            return self._constructor(
                [
                    {k: v for k, v in d.items() if k in key}
                    for d in self._data
                ],
                name=self.name
            )

        if isinstance(key, tuple):
            if len(key) != 2:
                raise NotImplementedError(f'Unsupported indexing with tuple having <2 or >2 entries.')
            i, cols = key
            if isinstance(i, tuple):
                raise IndexError(f'Cannot use tuples for indexing')
            out_df = self[i]
            if isinstance(cols, slice):
                if any_are_not_none(cols.start, cols.stop, cols.step):
                    raise IndexError(f'Second argument when indexing (i.e. columns) should be a list or an empty slice')
                return out_df
            else:
                return out_df[cols]
        raise IndexError(f'Unsupported indexing using: {type(key)} with value: {key}')

    def __setitem__(self, key: Any, value: Any):
        length: int = len(self)
        if isinstance(key, str):
            if isinstance(value, ScalableSeries):
                if length != len(value):
                    raise ValueError(f'Expected input {type(value)} to have length={length}; found length={len(value)}')
                for i, val in enumerate(value):
                    self._data[i][key] = val
            elif is_scalar(value) or isinstance(value, set):
                for i, d in enumerate(self._data):
                    d[key] = value
            else:
                if length != len(value):
                    raise ValueError(f'Expected input value to have length={length}; found length={len(value)}')
                for i, val in enumerate(value):
                    self._data[i][key] = val
        elif isinstance(key, tuple):
            ## Supports assignment like: sdf[:, 'colA'] = 123
            if not len(key) == 2:
                raise ValueError(f'Can only set using a tuple of two elements; found {len(key)} elements:\n{key}')
            first = key[0]
            second = key[1]
            if isinstance(first, slice) and first.start is first.step is first.stop is None:
                self[second] = value
            else:
                raise NotImplementedError(f'Sliced assignment not supported at the moment')
        else:
            raise NotImplementedError(f'Cannot set using key of type {type(key)}.')

    def apply(self, func, axis=1, args=(), **kwargs):
        if axis not in {1, 'columns'}:
            raise AttributeError(
                f'{self.__class__} only supports applying a function row-wise (i.e. with axis=1).'
            )
        out: List[Any] = [
            func(d, *args, **kwargs)
            for d in self._data
        ]
        all_are_dict: bool = True
        for d in out:
            if not isinstance(d, dict):
                all_are_dict: bool = False
                break
        if all_are_dict:
            return self._constructor(out)
        return self.ScalableSeriesClass(out)

    @classmethod
    def _concat_sdfs(cls, sdfs: List[ListOfDictScalableDataFrame], reset_index: bool) -> ListOfDictScalableDataFrame:
        list_of_dicts: List[Dict] = []
        for sdf in sdfs:
            assert isinstance(sdf, ListOfDictScalableDataFrame)
            list_of_dicts.extend(sdf._data)
        return cls.of(list_of_dicts, layout=cls.layout)

    def copy(self, deep: bool = True) -> ScalableDataFrame:
        data = self._data
        if deep:
            data = copy.deepcopy(data)
        return self._constructor(data, name=self.name)

    def as_dict(
            self,
            col_type: Optional[Literal['list', 'numpy', list, np.ndarray, 'record']] = None,
            **kwargs
    ) -> Dict[str, Union[List, np.ndarray, Any]]:
        if col_type == 'record':
            length: int = len(self)
            if length > 1:
                raise ValueError(f'Cannot convert {self.__class__} of length {length} into record')
            return self._data[0]
        cols: Set = self.columns_set
        out: Dict[str, Union[List, np.ndarray]] = {col: [] for col in cols}
        for d in self._data:
            for col in cols:
                out[col].append(d.get(col))
        if col_type in {None, 'list', list}:
            return out
        if col_type in {'numpy', np.ndarray}:
            for col in cols:
                out[col]: np.ndarray = self.ScalableSeriesClass(out[col]).numpy()
            return out
        raise NotImplementedError(f'Unsupported `col_type`: {col_type}')

    def as_list_of_dict(self, **kwargs) -> List[Dict]:
        return self._data

    def as_numpy(self, sort_columns: bool = True, **kwargs) -> np.recarray:
        cols: List = list(self.columns)
        if sort_columns:
            cols: List = sorted(cols)
        np_dtype = np.dtype([
            (col, object)
            for col in cols
        ])
        return np.rec.array([
            tuple([d.get(col, np.nan) for col in cols])
            for d in self._data
        ], dtype=np_dtype)

    def as_pandas(self, sort_columns: bool = True, **kwargs) -> PandasDataFrame:
        df: PandasDataFrame = pd.DataFrame(self._data, **filter_kwargs(pd.DataFrame.__init__, **kwargs))
        if sort_columns:
            df = df[sorted(df.columns)]
        return df

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
        renamed_data: List[Dict] = [self._rename_single_dict(d, mapper=mapper) for d in self._data]
        if inplace:
            self._data = renamed_data
            return None
        return self._constructor(renamed_data, name=self.name)

    @staticmethod
    def _rename_single_dict(d: Dict, mapper: Union[Dict, Callable]) -> Dict:
        if isinstance(mapper, dict):
            return {mapper.get(k, k): v for k, v in d.items()}
        return {mapper(k): v for k, v in d.items()}

    def _stream_in_memory_sdf_convert_to_streaming_layout(self, stream_as: DataLayout) -> ScalableDataFrame:
        ## First select the chunks (as a list-of-dict), then convert it to the target data layout only just before
        ## yielding...selecting small chunks is faster in list than in other formats.
        return self

    class ILocIndexer:
        def __init__(self, sdf: ListOfDictScalableDataFrame):
            self._sdf: ListOfDictScalableDataFrame = sdf

        def __setitem__(self, key, value):
            raise NotImplementedError(f'Can only use for retrieving.')

        def __getitem__(self, key) -> ListOfDictScalableDataFrame:
            return self._sdf[key]

    def applymap(self, func, na_action: Optional[Literal['ignore']] = None, **kwargs) -> ScalableDataFrame:
        if na_action == 'ignore':
            mapper: Callable = lambda x: x if is_null(x) else func(x, **kwargs)
        else:
            mapper: Callable = lambda x: func(x, **kwargs)
        return self._constructor(
            [
                {k: mapper(v) for k, v in d.items()}
                for d in self._data
            ],
            name=self.name
        )
