from typing import *
import collections
from abc import abstractmethod, ABC
import numpy as np
from pandas.core.frame import Series as PandasSeries, DataFrame as PandasDataFrame
import dask.dataframe as dd
from dask.dataframe.core import Scalar as DaskScalar, Series as DaskSeries, DataFrame as DaskDataFrame
from synthergent.base.util import Registry, resolve_sample_size, SampleSizeType, optional_dependency, MutableParameters, \
    get_true, safe_validate_arguments, str_normalize, only_item
from synthergent.base.constants import DataLayout, SS_DATA_LAYOUT_PRIORITY, TENSOR_SS_DATA_LAYOUT_PRIORITY, \
    SHORTHAND_TO_TENSOR_LAYOUT_MAP
from pydantic import conint
from pydantic.typing import Literal

ScalableSeries = "ScalableSeries"
ScalableDataFrame = "ScalableDataFrame"
ScalableSeriesRawType = Union[np.ndarray, List, PandasSeries, DaskSeries]
ScalableSeriesOrRaw = Union[ScalableSeries, ScalableSeriesRawType]
RAW_DATA_MEMBER: str = '_data'
SS_DEFAULT_NAME: str = '0'


class ScalableDataFrameDisplay(MutableParameters):
    max_rows: int = 10
    min_rows: int = 10


class ScalableSeries(Registry, ABC):
    """
    Class to interact with mutable "Series" with various underlying data layouts.
    """
    layout: ClassVar[DataLayout]
    ## Callable typing: stackoverflow.com/a/39624147/4900327
    layout_validator: ClassVar[Callable[[Any, bool], bool]]
    display: ClassVar[ScalableDataFrameDisplay] = ScalableDataFrameDisplay()

    def __init__(
            self,
            data: Optional[Union[ScalableSeries, ScalableSeriesRawType]] = None,
            name: Optional[str] = None,
            **kwargs
    ):
        self._data: Any = data
        self._name: Optional[str] = name

    def __del__(self):
        _data = self._data
        self._data = None
        del _data

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def _constructor(self) -> Type[ScalableSeries]:
        return self.__class__

    ## Required to serialize ScalableSeries using Ray.
    ## Ref: https://docs.ray.io/en/latest/ray-core/objects/serialization.html#customized-serialization
    def __reduce__(self):
        deserializer = self.__class__
        serialized_data = (self._data,)
        return deserializer, serialized_data

    @classmethod
    def _registry_keys(cls) -> Optional[Any]:
        return cls.layout

    @classmethod
    def of(
            cls,
            data: Union[ScalableSeries, ScalableSeriesRawType],
            layout: Optional[DataLayout] = None,
            **kwargs,
    ) -> ScalableSeries:
        """
        Factory to create ScalableSeries.
        :param data: ScalableSeries or "raw" data to be used as the underlying object.
        :param layout: target layout of the returned ScalableSeries.
        :return: ScalableSeries instance.
        """
        if data is None:
            raise ValueError(f'Input data cannot be None.')
        if not isinstance(data, ScalableSeries):
            if layout is None:
                detected_layout: Optional[DataLayout] = cls.detect_layout(data, raise_error=False)
                if detected_layout is None:
                    raise NotImplementedError(f'Cannot infer layout of data with type: {type(data)}.')
                layout: DataLayout = detected_layout
            data: ScalableSeries = cls.get_subclass(layout)(data=data, **kwargs)
        assert isinstance(data, ScalableSeries)
        if layout is None:
            return data  ## If data is ScalableSeries and layout is None, return data (SS) without modification
        ScalableSeriesClass: Optional[Type[ScalableSeries]] = cls.get_subclass(layout, raise_error=False)
        if ScalableSeriesClass is None:
            raise ValueError(
                f'Cannot create {ScalableSeries} subclass having layout "{layout}"; '
                f'available subclasses are: {ScalableSeries.subclasses()}'
            )
        ## When passed either raw data (in the correct format) or a ScalableSeries, the respective
        ## ScalableSeries subclass should be able to accept it.
        return ScalableSeriesClass(data=data, **kwargs)

    @property
    def hvplot(self) -> Any:
        with optional_dependency('hvplot', error='raise'):
            import hvplot.pandas
            return self.pandas().hvplot

    @classmethod
    def detect_layout(cls, data: Any, raise_error: bool = True) -> Optional[DataLayout]:
        from synthergent.base.data.sdf.ScalableDataFrame import ScalableDataFrame
        if isinstance(data, (ScalableSeries, ScalableDataFrame)):
            return data.layout
        for possible_layout in SS_DATA_LAYOUT_PRIORITY + TENSOR_SS_DATA_LAYOUT_PRIORITY:
            ScalableSeriesClass: Optional[Type[ScalableSeries]] = cls.get_subclass(
                possible_layout,
                raise_error=False,
            )
            if ScalableSeriesClass is None:
                continue
            if ScalableSeriesClass.layout_validator(data, raise_error=False):
                return possible_layout
        if raise_error:
            raise NotImplementedError(
                f'Cannot infer layout of data having type: {type(data)}. '
                f'Please pass `layout=...` using one of the following: {list(DataLayout)}'
            )
        else:
            return None

    @safe_validate_arguments
    def as_layout(self, layout: Optional[DataLayout] = None, **kwargs) -> ScalableSeries:
        if layout is None:
            return self
        return self.of(self, layout=layout, **kwargs)

    @classmethod
    def is_datum(cls, data: Optional[Any], raise_error: bool = True) -> bool:
        return True

    @classmethod
    def is_list(cls, data: Any, raise_error: bool = True) -> bool:
        if data is None:
            if raise_error:
                raise ValueError(f'Input data cannot be None.')
            else:
                return False
        valid: bool = isinstance(data, list)
        if not valid and raise_error:
            raise ValueError(f'Expected input to be list; found input with type: {type(data)}')
        return valid

    @classmethod
    def is_numpy_array(cls, data: Any, raise_error: bool = True) -> bool:
        if data is None:
            if raise_error:
                raise ValueError(f'Input data cannot be None.')
            else:
                return False
        valid: bool = isinstance(data, np.ndarray)
        if not valid and raise_error:
            raise ValueError(f'Expected input to be NumPy array; found input with type: {type(data)}')
        return valid

    @classmethod
    def is_tensor(cls, data: Any) -> bool:
        return get_true(
            cls.is_torch_tensor(data, raise_error=False),
        )

    @classmethod
    def is_torch_tensor(cls, data: Any, raise_error: bool = True) -> bool:
        with optional_dependency('torch', error='ignore'):
            import torch
            if data is None:
                if raise_error:
                    raise ValueError(f'Input data cannot be None.')
                else:
                    return False
            valid: bool = isinstance(data, torch.Tensor)
            if not valid and raise_error:
                raise ValueError(f'Expected input to be Torch tensor; found input with type: {type(data)}')
            return valid
        return False

    @classmethod
    def is_pandas(cls, data: Any, raise_error: bool = True) -> bool:
        if data is None:
            if raise_error:
                raise ValueError(f'Input data cannot be None.')
            else:
                return False
        valid: bool = isinstance(data, PandasSeries)
        if not valid and raise_error:
            raise ValueError(f'Expected input to be Pandas Series; found input with type: {type(data)}')
        return valid

    @classmethod
    def is_dask(cls, data: Any, raise_error: bool = True) -> bool:
        if data is None:
            if raise_error:
                raise ValueError(f'Input data cannot be None.')
            else:
                return False
        valid: bool = isinstance(data, DaskSeries)
        if not valid and raise_error:
            raise ValueError(f'Expected input to be Dask Series; found input with type: {type(data)}')
        return valid

    @safe_validate_arguments
    def valid(
            self,
            validator: Callable[[Any, bool], bool],
            sample_size: Union[SampleSizeType, Literal[False], Literal[0.0], Literal[0]] = True,
            seed: Optional[int] = None,
            return_failed: bool = False,
            **kwargs,
    ) -> Union[bool, Tuple]:
        """
        Runs a validator function element-wise on the data, and throws an exception if any element is invalid.
        :param validator: function to call element-wise.
        :param sample_size: amount of data to validate.
            If False, it will not validate data. If True, it will validate entire data.
            If 0.0 < sample_size <= 1.0, then we will validate a fraction of the data.
            If 1 < sample_size, we will validate these many rows of data.
        :param seed: random seed for sampling a fraction.
        :param return_failed: whether to return the rows which failed validation.
        :raises: ValueError if arguments are incorrect types or any row is invalid.
        """
        if sample_size in {False, 0.0, 0}:
            return True
        sample: ScalableSeries = self
        if sample_size in {True, 1.0}:
            sample: ScalableSeries = self
        elif 0.0 < sample_size < 1.0:
            sample: ScalableSeries = self.sample(frac=float(sample_size), random_state=seed)
        elif 1 < sample_size:
            length: int = len(self)
            n: int = resolve_sample_size(sample_size=sample_size, length=length)
            sample: ScalableSeries = self.sample(n=int(n), random_state=seed)
        valid: ScalableSeries = sample.map(
            lambda x: validator(x, **kwargs),
            na_action=None,  ## Always pass the cell-value to the validator function.
        )
        valid: bool = valid.all()
        if return_failed:
            failed_sample: ScalableDataFrame = sample.loc[~valid]
            return valid, failed_sample
        return valid

    """
    =======================================================================================
    Implement the Pandas Series API: https://pandas.pydata.org/docs/reference/frame.html
    =======================================================================================
    """

    _NOT_IMPLEMENTED_INSTANCE_PROPERTIES: Set[str] = {
        ## Attributes
        'array', 'values', 'nbytes', 'dtypes', 'name', 'flags',
        ## Indexing, iteration
        'at', 'iat',
        ## Accessors
        'dt', 'str', 'cat', 'sparse',
        ## Metadata
        'attrs',
        ## Plotting
        'plot',
    }

    _NOT_IMPLEMENTED_INSTANCE_METHODS: Set[str] = {
        ## Attributes:
        'memory_usage', 'set_flags',
        ## Conversion
        'convert_dtypes', 'infer_objects', 'to_numpy', 'to_period', 'to_timestamp', '__array__',
        ## Indexing, iteration
        'get', 'items', 'iteritems', 'keys', 'pop', 'xs',
        ## Binary operator functions
        'add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'mod', 'pow', 'radd', 'rsub', 'rmul', 'rdiv', 'rtruediv',
        'rfloordiv', 'rmod', 'rpow', 'combine', 'combine_first', 'lt', 'gt', 'le', 'ge', 'ne', 'eq',
        ## Function application, GroupBy & window
        'agg', 'aggregate', 'transform', 'groupby', 'rolling', 'expanding', 'ewm', 'pipe',
        ## Computations / descriptive stats
        'autocorr', 'cummax', 'cummin', 'cumprod', 'cumsum', 'describe', 'diff', 'factorize', 'pct_change', 'prod',
        'rank', 'sem', 'skew',
        ## Reindexing / selection / label manipulation
        'align', 'droplevel', 'first', 'last', 'rename', 'rename_axis',
        'set_axis', 'take', 'truncate', 'add_prefix', 'add_suffix', 'filter', 'where', 'mask',
        ## Missing data handling
        'interpolate',
        ## Reshaping, sorting
        'reorder_levels', 'sort_index', 'swaplevel', 'unstack', 'explode', 'searchsorted', 'ravel', 'repeat', 'squeeze',
        'view',
        ## Combining / comparing / joining / merging
        'append', 'combine',
        ## Time Series-related
        'asfreq', 'asof', 'shift', 'first_valid_index', 'last_valid_index', 'resample', 'tz_convert', 'tz_localize',
        'at_time', 'between_time', 'tshift', 'slice_shift',
        ## Flags
        'Flags',
        ## Serialization / IO / conversion
        'to_pickle', 'to_csv', 'to_excel', 'to_xarray', 'to_hdf', 'to_sql', 'to_json', 'to_string', 'to_clipboard',
        'to_latex', 'to_markdown', 'to_dict',
    }

    _NOT_IMPLEMENTED_REASONS: Dict[str, str] = {
        'append': 'as it is deprecated in newer versions of Pandas',
    }

    _ALTERNATIVE: Dict[str, str] = {
        'values': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'at': '.loc or .iloc',
        'iat': '.loc or .iloc',
        'get': '.loc or .iloc',

        ## Combining / comparing / joining / merging
        'append': f'ScalableSeries.concat',

        ## Accessors
        'dt': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'str': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'cat': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'sparse': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',

        ## Serialization / IO / conversion:
        'to_pickle': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'to_csv': '.to_frame() followed by CsvDataFrameWriter or TsvDataFrameWriter',
        'to_dict': 'as_dict',
        'to_excel': '.to_frame() followed by CsvDataFrameWriter or TsvDataFrameWriter',
        ## TODO: add ExcelDataFrameWriter
        'to_xarray': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'to_hdf': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'to_sql': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'to_json': '.to_frame() followed by JsonlinesDataFrameWriter',
        'to_string': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'to_clipboard': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'to_latex': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
        'to_markdown': f'.{RAW_DATA_MEMBER} to get the raw data in the respective layout',
    }

    def __getattr__(self, attr_name: str):
        if attr_name in self._NOT_IMPLEMENTED_INSTANCE_PROPERTIES:
            raise self._get_attribute_not_implemented_error(
                attr_name,
                is_property=True,
                reason=self._NOT_IMPLEMENTED_REASONS.get(attr_name),
                alternative=self._ALTERNATIVE.get(attr_name)
            )
        elif attr_name in self._NOT_IMPLEMENTED_INSTANCE_METHODS:
            raise self._get_attribute_not_implemented_error(
                attr_name,
                is_property=False,
                reason=self._NOT_IMPLEMENTED_REASONS.get(attr_name),
                alternative=self._ALTERNATIVE.get(attr_name)
            )
        ## Forwards calls to the respective method of the data class.
        data = self.__dict__[RAW_DATA_MEMBER]
        if not hasattr(data, attr_name):
            raise AttributeError(
                f'Neither {self.__class__.__name__} nor {type(self._data)} classes have attribute "{attr_name}"'
            )
        return getattr(data, attr_name)

    def _get_attribute_not_implemented_error(
            self,
            attr_name: str,
            is_property: bool,
            reason: Optional[str] = None,
            alternative: Optional[str] = None,
    ) -> AttributeError:
        if is_property:
            fn_type = 'Property'
        else:
            fn_type = 'Method'
        if reason is None:
            reason = f'to maintain compatibility between {self.__class__.__name__} subclasses with different layouts'
        if alternative is not None:
            alternative = f' Please use {alternative} instead.'
        return AttributeError(
            f'{fn_type} .{attr_name} has not been implemented {reason}.{alternative}')

    """
    ---------------------------------------------
    Attributes
    ---------------------------------------------
    """

    @property
    def shape(self) -> Tuple[int, ...]:
        return (len(self),)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def size(self) -> int:
        return len(self)

    @property
    def T(self) -> ScalableSeries:
        return self

    @property
    def empty(self) -> bool:
        return len(self) == 0

    @abstractmethod
    def __len__(self):
        """
        Should return the number of rows in the "Series".
        :return:
        """
        pass

    @abstractmethod
    def __str__(self):
        """
        Gets a string representation of the "Series".
        :return:
        """
        pass

    def __repr__(self):
        return str(self)

    """
    ---------------------------------------------
    Conversion
    ---------------------------------------------
    """

    def to_frame(self, **kwargs):
        from synthergent.base.data.sdf import ScalableDataFrame
        return ScalableDataFrame.of(self._to_frame_raw(**kwargs))

    @abstractmethod
    def _to_frame_raw(self, **kwargs):
        pass

    def raw(self, **kwargs) -> Any:
        """Alias for .as_raw()"""
        return self.as_raw(**kwargs)

    def to_raw(self, **kwargs) -> Any:
        """Alias for .as_raw()"""
        return self.as_raw(**kwargs)

    def as_raw(self, **kwargs) -> Any:
        return self._data

    def list(self, **kwargs) -> List:
        """Alias for .as_list()"""
        return self.as_list(**kwargs)

    def to_list(self, **kwargs) -> List:
        """Alias for .as_list()"""
        return self.as_list(**kwargs)

    def tolist(self, **kwargs) -> List:
        """Alias for .as_list()"""
        return self.as_list(**kwargs)

    def as_list(self, **kwargs) -> List:
        return list(self.pandas(**kwargs).values)

    def set(self, **kwargs) -> Set:
        """Alias for .as_set()"""
        return self.as_set(**kwargs)

    def to_set(self, **kwargs) -> Set:
        """Alias for .as_set()"""
        return self.as_set(**kwargs)

    def as_set(self, **kwargs) -> Set:
        return set(np.unique(self.pandas(**kwargs).values))

    def numpy(self, **kwargs) -> Optional[Any]:
        """Alias for .as_numpy()"""
        return self.as_numpy(**kwargs)

    def to_numpy(self, **kwargs) -> Optional[Any]:
        """Alias for .as_numpy()"""
        return self.as_numpy(**kwargs)

    def as_numpy(self, *, stack: bool = False, **kwargs) -> np.ndarray:
        np_arr: np.ndarray = np.array(self.pandas(**kwargs).values)
        if stack:
            np_arr: np.ndarray = np.vstack(np_arr)
        return np_arr

    def tensor(self, *, tensor_type: str, **kwargs) -> Optional[Any]:
        """Alias for .as_tensor()"""
        return self.as_tensor(tensor_type=tensor_type, **kwargs)

    def to_tensor(self, *, tensor_type: str, **kwargs) -> Optional[Any]:
        """Alias for .as_tensor()"""
        return self.as_tensor(tensor_type=tensor_type, **kwargs)

    def as_tensor(self, *, tensor_type: str, **kwargs) -> Optional[Any]:
        tensor_layout: DataLayout = SHORTHAND_TO_TENSOR_LAYOUT_MAP[str_normalize(tensor_type)]
        if tensor_layout is DataLayout.NUMPY:
            return self.numpy(**kwargs)
        if tensor_layout is DataLayout.TORCH:
            return self.as_torch(**kwargs)
        raise NotImplementedError(f'Unsupported value of `tensor_type`: {tensor_type}')

    def torch(self, **kwargs) -> Optional[Any]:
        """Alias for .as_torch()"""
        return self.as_torch(**kwargs)

    def to_torch(self, **kwargs) -> Optional[Any]:
        """Alias for .as_torch()"""
        return self.as_torch(**kwargs)

    def as_torch(
            self,
            error: Literal['raise', 'warn', 'ignore'] = 'raise',
            **kwargs
    ) -> Optional[Any]:
        with optional_dependency('torch', error=error):
            import torch
            arr: np.ndarray = self.numpy(**kwargs)
            return torch.from_numpy(np.stack(arr))
        return None

    def pandas(self, **kwargs) -> PandasSeries:
        """Alias for .as_pandas()"""
        return self.as_pandas(**kwargs)

    def to_pandas(self, **kwargs) -> PandasSeries:
        """Alias for .as_pandas()"""
        return self.as_pandas(**kwargs)

    @abstractmethod
    def as_pandas(self, **kwargs) -> PandasSeries:
        pass

    def dask(self, **kwargs) -> DaskSeries:
        """Alias for .as_dask()"""
        return self.as_dask(**kwargs)

    def to_dask(self, **kwargs) -> DaskSeries:
        """Alias for .as_dask()"""
        return self.as_dask(**kwargs)

    def as_dask(self, **kwargs) -> DaskSeries:
        if 'npartitions' not in kwargs and 'chunksize' not in kwargs:
            kwargs['npartitions'] = 1  ## Create a dask series with a single partition.
        return dd.from_pandas(self.pandas(), **kwargs)

    def is_lazy(self) -> bool:
        return False

    def persist(self, **kwargs) -> ScalableDataFrame:
        """For lazily-evaluated DataFrames, stores the task graph up to the current DataFrame."""
        return self

    def compute(self, **kwargs) -> ScalableDataFrame:
        """For lazily-evaluated DataFrames, runs the task graph up to the current DataFrame."""
        return self

    @property
    def npartitions(self) -> int:
        """For distributed DataFrames, this gets the number of data partitions."""
        return 1

    def repartition(self, **kwargs) -> ScalableDataFrame:
        """Creates a new ScalableSeries with different partition boundaries."""
        return self

    """
    ---------------------------------------------
    Indexing, iteration
    ---------------------------------------------
    """

    @property
    @abstractmethod
    def loc(self) -> Any:
        pass

    def __iter__(self):
        return self._data.__iter__()

    @property
    @abstractmethod
    def hasnans(self) -> bool:
        pass

    def stream(self, **kwargs) -> Generator[ScalableSeriesOrRaw, None, None]:
        sdf: ScalableDataFrame = self.to_frame()
        col: str = only_item(sdf.columns_set)
        for data_batch in sdf.stream(**kwargs):
            yield data_batch[col]

    """
    ---------------------------------------------
    Binary operator functions
    ---------------------------------------------
    """

    def __add__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__add__(other))

    def __radd__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__radd__(other))

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__sub__(other))

    def __rsub__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__rsub__(other))

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__mul__(other))

    def __rmul__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__rmul__(other))

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__truediv__(other))

    def __rtruediv__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__rtruediv__(other))

    def __floordiv__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__floordiv__(other))

    def __rfloordiv__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__rfloordiv__(other))

    def __mod__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__mod__(other))

    def __rmod__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__rmod__(other))

    def __divmod__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__divmod__(other))

    def __rdivmod__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__rdivmod__(other))

    def __pow__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__pow__(other))

    def __rpow__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__rpow__(other))

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__lt__(other))

    def __gt__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__gt__(other))

    def __le__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__le__(other))

    def __ge__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__ge__(other))

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__ne__(other))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        return self.__class__(self._data.__eq__(other))

    def dot(self, other):
        if isinstance(other, self.__class__):
            other = other._data
        out = self._data.dot(other)
        if isinstance(out, DaskScalar):
            out = out.compute()
        return out

    """
    ---------------------------------------------
    Function application, GroupBy & window
    ---------------------------------------------
    """

    @abstractmethod
    def apply(
            self,
            func: Callable,
            convert_dtype: bool = True,
            args: Tuple[Any, ...] = (),
            **kwargs
    ) -> Union[ScalableSeries, ScalableDataFrame]:
        pass

    @abstractmethod
    def map(
            self,
            arg: Union[Callable, Dict, ScalableSeries],
            na_action: Optional[Literal['ignore']] = None,
    ) -> ScalableSeries:
        pass

    """
    ---------------------------------------------
    Computations / descriptive stats
    ---------------------------------------------
    """

    @abstractmethod
    def abs(self):
        pass

    @abstractmethod
    def all(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        pass

    @abstractmethod
    def any(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        pass

    @abstractmethod
    def between(self, left, right, inclusive='both'):
        pass

    @abstractmethod
    def clip(self, lower=None, upper=None, axis=None, inplace=False, *args, **kwargs):
        pass

    @abstractmethod
    def corr(self, other, method='pearson', min_periods=None):
        pass

    @abstractmethod
    def count(self, level=None):
        pass

    @abstractmethod
    def cov(self, other, min_periods=None, ddof=1):
        pass

    @abstractmethod
    def kurt(self, axis=0, skipna=True, level=None, numeric_only=None, **kwargs):
        pass

    @abstractmethod
    def mad(self, axis=None, skipna=True, level=None):
        pass

    @abstractmethod
    def max(self, axis=0, skipna=True, level=None, numeric_only=None, **kwargs):
        pass

    @abstractmethod
    def mean(self, axis=0, skipna=True, level=None, numeric_only=None, **kwargs):
        pass

    @abstractmethod
    def median(self, axis=0, skipna=True, level=None, numeric_only=None, **kwargs):
        pass

    @abstractmethod
    def min(self, axis=0, skipna=True, level=None, numeric_only=None, **kwargs):
        pass

    @abstractmethod
    def mode(self, dropna: bool = True) -> ScalableSeries:
        pass

    @abstractmethod
    def nlargest(self, n=5, keep='first'):
        pass

    @abstractmethod
    def nsmallest(self, n=5, keep='first'):
        pass

    @abstractmethod
    def quantile(self, q=0.5, interpolation='linear'):
        pass

    @abstractmethod
    def std(self, axis=None, skipna=True, level=None, ddof=1, numeric_only=None, **kwargs):
        pass

    @abstractmethod
    def sum(self, axis=None, skipna=True, level=None, numeric_only=None, min_count=0, **kwargs):
        pass

    @abstractmethod
    def var(self, axis=None, skipna=True, level=None, ddof=1, numeric_only=None, **kwargs):
        pass

    @abstractmethod
    def kurtosis(self, axis=0, skipna=True, level=None, numeric_only=None, **kwargs):
        pass

    @abstractmethod
    def unique(self):
        pass

    @abstractmethod
    def nunique(self, dropna=True):
        pass

    @property
    @abstractmethod
    def is_unique(self):
        pass

    @property
    @abstractmethod
    def is_monotonic_increasing(self):
        pass

    @property
    def is_monotonic(self):
        return self.is_monotonic_increasing

    @property
    @abstractmethod
    def is_monotonic_decreasing(self):
        pass

    @abstractmethod
    def value_counts(self, normalize=False, sort=True, ascending=False, bins=None, dropna=True):
        pass

    """
    ---------------------------------------------
    Reindexing / selection / label manipulation
    ---------------------------------------------
    """

    @abstractmethod
    def drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'):
        pass

    @abstractmethod
    def drop_duplicates(self, keep='first', inplace=False):
        pass

    @abstractmethod
    def duplicated(self, keep='first'):
        pass

    @abstractmethod
    def equals(self, other):
        pass

    @abstractmethod
    def head(self, n=5):
        pass

    @abstractmethod
    def idxmax(self, axis=0, skipna=True, *args, **kwargs):
        pass

    @abstractmethod
    def idxmin(self, axis=0, skipna=True, *args, **kwargs):
        pass

    @abstractmethod
    def isin(self, values):
        pass

    @abstractmethod
    def sample(self, n=None, frac=None, replace=False, weights=None, random_state=None, axis=None, ignore_index=False):
        pass

    @abstractmethod
    def tail(self, n=5):
        pass

    """
    ---------------------------------------------
    Missing data handling
    ---------------------------------------------
    """

    @abstractmethod
    def backfill(self, axis=None, inplace=False, limit=None, downcast=None):
        pass

    @abstractmethod
    def bfill(self, axis=None, inplace=False, limit=None, downcast=None):
        pass

    @abstractmethod
    def dropna(self, axis: int = 0, inplace: bool = False, how=None) -> ScalableSeries:
        pass

    @abstractmethod
    def ffill(self, axis=None, inplace=False, limit=None, downcast=None):
        pass

    @abstractmethod
    def fillna(
            self,
            value: Optional[Union[Any, Dict, ScalableSeries, ScalableDataFrame]] = None,
            method: Optional[Literal["backfill", "bfill", "ffill", "pad"]] = None,
            axis: Literal[0, 'index'] = 0,
            inplace: bool = False,
            limit: Optional[conint(ge=1)] = None,
            downcast: Optional[Dict] = None,
    ) -> Optional[ScalableSeries]:
        pass

    @abstractmethod
    def isna(self) -> ScalableSeries:
        pass

    def isnull(self) -> ScalableSeries:
        return self.isna()

    @abstractmethod
    def notna(self) -> ScalableSeries:
        pass

    def notnull(self) -> ScalableSeries:
        return self.notna()

    @abstractmethod
    def pad(self, axis=None, inplace=False, limit=None, downcast=None):
        pass

    @abstractmethod
    def replace(self, to_replace=None, value=None, inplace=False, limit=None, regex=False, method=None):
        pass

    """
    ---------------------------------------------
    Reshaping, sorting
    ---------------------------------------------
    """

    @abstractmethod
    def argsort(self, axis=0, kind='quicksort', order=None):
        pass

    @abstractmethod
    def argmin(self, axis=None, skipna=True, *args, **kwargs):
        pass

    @abstractmethod
    def argmax(self, axis=None, skipna=True, *args, **kwargs):
        pass

    @abstractmethod
    def sort_values(self, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last',
                    ignore_index=False, key=None):
        pass

    """
    ---------------------------------------------
    Combining / comparing / joining / merging
    ---------------------------------------------
    """

    @abstractmethod
    def update(self, other):
        pass

    def is_lazy(self) -> bool:
        return False

    def persist(self, **kwargs) -> ScalableSeries:
        """For lazily-evaluated Series, stores the task graph up to the current Series."""
        return self

    def compute(self, **kwargs) -> ScalableSeries:
        """For lazily-evaluated Series, runs the task graph up to the current Series."""
        return self


## Ref: https://stackoverflow.com/a/15920132/4900327
## This deletes abstractmethods from ScalableSeries, which means that we can instantiate classes
## like PandasScalableSeries/DaskScalableSeries even if they have not implemented all these methods.
## This is done because these classes forward to the respective "raw" series (i.e. PandasSeries, DaskSeries)
## when an implementation is not found on either ScalableSeries or PandasScalableSeries/DaskScalableSeries.
## At the same time, we only need to override __getattr__, not __getattribute__, which is faster during execution.
## Additionally, typing in PyCharm still works as expected.
[delattr(ScalableSeries, abs_method_name) for abs_method_name in ScalableSeries.__abstractmethods__]

ScalableSeriesOrRaw = Union[ScalableSeries, ScalableSeriesRawType]
