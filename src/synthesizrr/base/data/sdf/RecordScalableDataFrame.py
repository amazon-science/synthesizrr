from typing import *
import math, copy, pprint
import numpy as np
import pandas as pd
from pandas.core.frame import Series as PandasSeries, DataFrame as PandasDataFrame
from synthesizrr.base.util import safe_validate_arguments, as_list, any_are_not_none, get_default, StringUtil
from synthesizrr.base.constants import DataLayout, Parallelize
from synthesizrr.base.data.sdf.ScalableSeries import ScalableSeries
from synthesizrr.base.data.sdf.DatumScalableSeries import DatumScalableSeries
from synthesizrr.base.data.sdf.ScalableDataFrame import ScalableDataFrame, ScalableDataFrameOrRaw
from pydantic import conint
from pydantic.typing import Literal

RecordScalableDataFrame = "RecordScalableDataFrame"


class RecordScalableDataFrame(ScalableDataFrame):
    ## Low-latency implementaion of DictScalableDataFrame, will only contain data of a single record.

    layout = DataLayout.RECORD
    layout_validator = ScalableDataFrame.is_record
    ScalableSeriesClass = DatumScalableSeries

    def __init__(
            self,
            data: Union[Dict, ScalableDataFrame],
            name: Optional[str] = None,
            **kwargs,
    ):
        super(self.__class__, self).__init__(**kwargs)
        if isinstance(data, ScalableDataFrame):
            if len(data) > 1:
                raise ValueError(
                    f'Cannot create {self.__class__} from {type(data)} with {len(data)} records; '
                    f'only single-record {ScalableDataFrame.__class__}s can be passed.'
                )
            data: Dict = data.to_dict(col_type='record', **kwargs)
        self._data: Dict[Any, DatumScalableSeries] = self._make_record(data)
        if name is not None and not isinstance(name, (str, int, float)):
            raise ValueError(
                f'`name` used to construct {self.__class__} can only be int, str or float; '
                f'found object of type: {type(name)} with value: {name}'
            )
        self._name: Optional[str] = name

    @classmethod
    def _make_record(cls, data: Dict) -> Dict[Any, DatumScalableSeries]:
        return {col: cls.ScalableSeriesClass(val, name=col) for col, val in data.items()}

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
        col_lens: Dict[Any, int] = {col: datum.shape[0] for col, datum in self._data.items()}
        if check and len(set(col_lens.values())) > 1:
            raise ValueError(f'Columns are not of equal length; found following lengths: {col_lens}')
        return col_lens

    def column_dtypes(self) -> Dict[Any, np.dtype]:
        return {col: datum.dtype for col, datum in self._data.items()}

    def __len__(self):
        return 1

    def __str__(self):
        name_str: str = '' if self._name is None else f'"{self._name}": '
        columns: List = self.columns
        out = f'{name_str}Single record with {len(columns)} column(s):\n' \
              f'[0]: {StringUtil.pretty(self.raw())}'
        out = out.strip()
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
                data_sample += f'<pre>"{col}": {StringUtil.pretty(self._data[col])}</pre><br>'
        out += f'{data_sample}'
        return f'<div>{out}</div>'

    def _repr_latex_(self):
        return self._repr_html_()

    def copy(self, deep: bool = True) -> ScalableDataFrame:
        data = self._data
        if deep:
            data = copy.deepcopy(data)
        return self._constructor(data)

    def as_raw(self, **kwargs) -> Any:
        return self.as_dict(col_type=None)

    def as_dict(
            self,
            col_type: Optional[Literal['list', 'numpy', list, np.ndarray, 'record']] = None,
            **kwargs
    ) -> Dict[str, Union[List, np.ndarray, Any]]:
        if col_type in {'record', None}:
            return {col: val._data for col, val in self._data.items()}
        if col_type in {'numpy', np.ndarray}:
            return {col: data_col.numpy() for col, data_col in self._data.items()}
        if col_type in {'list', list}:
            return {col: data_col.to_list() for col, data_col in self._data.items()}
        raise NotImplementedError(f'Unsupported `col_type`: {col_type}')

    def as_list_of_dict(self, **kwargs) -> List[Dict]:
        return [self.raw()]

    def as_pandas(self, **kwargs) -> PandasDataFrame:
        return pd.DataFrame(self.raw(), index=[0])

    @classmethod
    def _concat_sdfs(cls, sdfs: List[RecordScalableDataFrame], reset_index: bool) -> ScalableDataFrame:
        list_of_dicts: List[Dict] = []
        for sdf in sdfs:
            assert isinstance(sdf, RecordScalableDataFrame)
            list_of_dicts.append(sdf.raw())
        return cls.of(list_of_dicts, layout=DataLayout.LIST_OF_DICT)

    def _sorted_items_dict(self) -> Dict[str, ScalableSeries]:
        return {col: self.ScalableSeriesClass(self._data[col], name=col) for col in sorted(self.columns)}

    @property
    def loc(self) -> Any:
        return self

    @property
    def iloc(self) -> Any:
        return self

    def __getitem__(self, key) -> Union[Any, RecordScalableDataFrame, DatumScalableSeries]:
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
            if self._empty or 0 not in list(range(key.start, key.stop, key.step)):
                return self._empty_datum_series
            return self

        if isinstance(key, np.ndarray):
            if key.ndim > 1:
                raise KeyError(f'Can only index with 1-D NumPy array; found array with shape {key.shape}')
            if len(key) > 1:
                raise KeyError(f'Cannot index {self.__class__.__name__} using multiple elements')
            key = key[0]
            if key == 0 or key is True:
                return self
            elif key is False:
                return self._empty_datum_series
            raise KeyError(f'When indexing with a Numpy array, must pass the index as 0 or True')

        if isinstance(key, str):
            ## Access columns:
            cols: Set = set(as_list(key))
            missing_cols: Set = cols - self.columns_set
            if len(missing_cols) > 0:
                raise KeyError(f'Following columns could not be found in Dataframe: {missing_cols}')
            col = key
            return self.ScalableSeriesClass(self._data[col], name=col)

        if isinstance(key, (list, set)):
            cols: Set = set(as_list(key))
            missing_cols: Set = cols - self.columns_set
            if len(missing_cols) > 0:
                raise KeyError(f'Following columns could not be found in Dataframe: {missing_cols}')
            return self._constructor({col: self._data[col] for col in key})

        if isinstance(key, tuple):
            if len(key) != 2:
                raise NotImplementedError(f'Unsupported indexing with tuple having <2 or >2 entries.')
            idx, cols = key
            if isinstance(idx, tuple):
                raise IndexError(f'Cannot use tuples for indexing')
            out_df = self[idx]
            if isinstance(cols, slice):
                if any_are_not_none(cols.start, cols.stop, cols.step):
                    raise IndexError(
                        f'When indexing, the second argument (i.e. columns argument) '
                        f'should be a list of columns or an empty slice'
                    )
                return out_df
            else:
                return out_df[cols]
        if isinstance(key, np.ndarray):
            key = key[0]
            if key == 0 or key is True:
                return self
            elif key is False:
                return self._empty_datum_series
            raise KeyError(f'When indexing with a List or Numpy array, must pass the index as 0 or True')
        raise IndexError(f'Unsupported indexing using: {type(key)} with value: {key}')

    def __setitem__(self, key: Any, value: Union[DatumScalableSeries, Any]):
        if isinstance(key, (str, int)):
            if not isinstance(value, self.ScalableSeriesClass):
                value = self.ScalableSeriesClass(value)
            self._data[key] = value
        else:
            raise NotImplementedError(f'Cannot set using key of type {type(key)}.')

    @safe_validate_arguments
    def _stream_chunks(
            self,
            ## All these args are captured (but ignored) to keep kwargs clean.
            num_rows: Optional[conint(ge=1)] = None,
            num_chunks: Optional[conint(ge=1)] = None,
            stream_as: DataLayout = None,
            raw: bool = False,
            shuffle: bool = False,
            seed: Optional[int] = None,
            map: Optional[Callable] = None,
            map_kwargs: Optional[Dict] = None,
            num_workers: conint(ge=1) = 1,
            parallelize: Parallelize = Parallelize.sync,
            map_executor: Literal['spawn'] = 'spawn',
            shard: Tuple[conint(ge=0), conint(ge=1)] = (0, 1),
            reverse_sharding: bool = False,
            drop_last: Optional[bool] = None,
            **kwargs
    ) -> ScalableDataFrameOrRaw:
        ## TODO: add map and other args.
        out = self.as_layout(layout=stream_as, **kwargs)
        if raw:
            return self.raw()
        return out

    def applymap(self, func, na_action: Optional[Literal['ignore']] = None, **kwargs) -> ScalableDataFrame:
        return self._constructor({
            col: datum.map(lambda x: func(x, **kwargs), na_action=na_action)
            for col, datum in self._data.items()
        })
