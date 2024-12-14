import gc
from typing import *
import typing, types, typing_extensions
import sys, os, time, functools, datetime as dt, string, inspect, re, random, math, ast, warnings, logging, json, ctypes, \
    tempfile, io
import numpy as np
import pandas as pd
from pandas.api.types import is_scalar as pd_is_scalar
from pandas.core.frame import Series as PandasSeries, DataFrame as PandasDataFrame
from dask.dataframe.core import Series as DaskSeries, DataFrame as DaskDataFrame
from abc import ABC, abstractmethod
from enum import Enum, auto
from pydantic import BaseModel, validate_arguments, Field, root_validator, Extra, confloat, conint, constr, \
    create_model_from_typeddict
from pydantic.typing import Literal
from pydantic.fields import Undefined
from itertools import product, permutations
from contextlib import contextmanager
from collections import defaultdict
from collections.abc import KeysView, ValuesView, ItemsView
from ast import literal_eval
from datetime import datetime
from tqdm.auto import tqdm as AutoTqdmProgressBar
from tqdm.autonotebook import tqdm as NotebookTqdmProgressBar
from tqdm.std import tqdm as StdTqdmProgressBar

TqdmProgressBar = Union[AutoTqdmProgressBar, NotebookTqdmProgressBar, StdTqdmProgressBar]

"""A collection of utilities to augment the Python language:"""

ListOrTuple = Union[List, Tuple]
DataFrameOrSeries = Union[PandasSeries, PandasDataFrame]
SeriesOrArray1D = Union[PandasSeries, List, Tuple, np.ndarray]
DataFrameOrArray2D = Union[PandasSeries, PandasDataFrame, List, List[List], np.ndarray]
SeriesOrArray1DOrDataFrameOrArray2D = Union[SeriesOrArray1D, DataFrameOrArray2D]

FractionalBool = Union[confloat(ge=0.0, le=1.0), bool]
SampleSizeType = Union[confloat(gt=0.0, le=1.0), conint(gt=1)]


def resolve_fractional_bool(fractional_bool: Optional[FractionalBool], seed: int = None) -> bool:
    if fractional_bool in {0.0, False, None}:
        return False
    elif fractional_bool in {1.0, False, True}:
        return True
    else:
        rnd: float = np.random.RandomState(seed=seed).random()
        return rnd <= fractional_bool


def resolve_sample_size(sample_size: Optional[SampleSizeType], length: int) -> conint(ge=0):
    if sample_size in {1.0, True}:
        n = length
    elif 0.0 < sample_size < 1.0:
        n: int = math.ceil(sample_size * length)  ## Use at least 1 row.
    elif isinstance(sample_size, int) and 1 < sample_size:
        n: int = sample_size
    else:
        raise ValueError(f'Invalid value for `sample_size`: {sample_size}')
    n: int = min(n, length)
    return n


def get_default(*vals) -> Optional[Any]:
    for x in vals:
        if not is_null(x):
            return x
    return None


def unset(obj, attr_name: str, new_val: Any = None, delete: bool = True):
    attr: Any = getattr(obj, attr_name)
    setattr(obj, attr_name, new_val)
    if delete:
        del attr


def get_true(*vals) -> bool:
    for x in vals:
        if x is True:
            return x
    return False


if_else = lambda cond, x, y: (x if cond is True else y)  ## Ternary operator
is_series = lambda x: isinstance(x, PandasSeries)
is_df = lambda x: isinstance(x, PandasDataFrame)
is_int_in_floats_clothing = lambda x: isinstance(x, int) or (isinstance(x, float) and int(x) == x)


## ======================== None utils ======================== ##
def any_are_none(*args) -> bool:
    for x in args:
        if x is None:
            return True
    return False


def all_are_not_none(*args) -> bool:
    return not any_are_none(*args)


def all_are_none(*args) -> bool:
    for x in args:
        if x is not None:
            return False
    return True


def any_are_not_none(*args) -> bool:
    return not all_are_none(*args)


def all_are_true(*args) -> bool:
    for x in args:
        assert x in {True, False}
        if not x:  ## Check for falsy values
            return False
    return True


def all_are_false(*args) -> bool:
    for x in args:
        assert x in {True, False}
        if x:  ## Check for truthy values
            return False
    return True


def none_count(*args) -> int:
    none_count: int = 0
    for x in args:
        if x is None:
            none_count += 1
    return none_count


def not_none_count(*args) -> int:
    return len(args) - none_count(*args)


def multiple_are_none(*args) -> bool:
    return none_count(*args) >= 2


def multiple_are_not_none(*args) -> bool:
    return not_none_count(*args) >= 2


def check_isinstance_or_none(x: Optional[Any], y: Type, raise_error: bool = True):
    if x is None:
        return True
    return check_isinstance(x, y, raise_error=raise_error)


is_null = lambda z: pd.isnull(z) if is_scalar(z) else (z is None)
is_not_null = lambda z: not is_null(z)


def equal(*args) -> bool:
    if len(args) == 0:
        raise ValueError(f'Cannot find equality for zero arguments')
    if len(args) == 1:
        return True
    first_arg = args[0]
    for arg in args[1:]:
        if arg != first_arg:
            return False
    return True


## ======================== String utils ======================== ##
def str_format_args(x: str, named_only: bool = True) -> List[str]:
    ## Ref: https://stackoverflow.com/a/46161774/4900327
    args: List[str] = [
        str(tup[1]) for tup in string.Formatter().parse(x)
        if tup[1] is not None
    ]
    if named_only:
        args: List[str] = [
            arg for arg in args
            if not arg.isdigit() and len(arg) > 0
        ]
    return args


def str_normalize(x: str, *, remove: Optional[Union[str, Tuple, List, Set]] = (' ', '-', '_')) -> str:
    ## Found to be faster than .translate() and re.sub() on Python 3.10.6
    if remove is None:
        remove: Set[str] = set()
    if isinstance(remove, str):
        remove: Set[str] = set(remove)
    assert isinstance(remove, (list, tuple, set))
    if len(remove) == 0:
        return str(x).lower()
    out: str = str(x)
    for rem in set(remove).intersection(set(out)):
        out: str = out.replace(rem, '')
    out: str = out.lower()
    return out


_PUNCTUATION_REMOVAL_TABLE = str.maketrans(
    '', '',
    string.punctuation  ## Will be removed
)
_PUNCTUATION_REMOVAL_TABLE_WITH_LOWERCASE = str.maketrans(
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz',
    string.punctuation  ## Will be removed
)
_PUNCTUATION_REMOVAL_TABLE_WITH_SPACE = str.maketrans(
    '', '',
    ' ' + string.punctuation  ## Will be removed
)
_PUNCTUATION_REMOVAL_TABLE_WITH_LOWERCASE_AND_SPACE = str.maketrans(
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz',
    ' ' + string.punctuation  ## Will be removed
)

_PUNCTUATION_REMOVAL_TABLE_WITH_NUMBERS = str.maketrans(
    '', '',
    '1234567890' + string.punctuation  ## Will be removed
)
_PUNCTUATION_REMOVAL_TABLE_WITH_LOWERCASE_AND_NUMBERS = str.maketrans(
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz',
    '1234567890' + string.punctuation  ## Will be removed
)
_PUNCTUATION_REMOVAL_TABLE_WITH_SPACE_AND_NUMBERS = str.maketrans(
    '', '',
    '1234567890 ' + string.punctuation  ## Will be removed
)
_PUNCTUATION_REMOVAL_TABLE_WITH_LOWERCASE_AND_SPACE_AND_NUMBERS = str.maketrans(
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz',
    '1234567890 ' + string.punctuation  ## Will be removed
)


def punct_normalize(x: str, *, lowercase: bool = True, space: bool = True, numbers: bool = False) -> str:
    punct_table = {
        (False, False, False): _PUNCTUATION_REMOVAL_TABLE,
        (True, False, False): _PUNCTUATION_REMOVAL_TABLE_WITH_LOWERCASE,
        (False, True, False): _PUNCTUATION_REMOVAL_TABLE_WITH_SPACE,
        (True, True, False): _PUNCTUATION_REMOVAL_TABLE_WITH_LOWERCASE_AND_SPACE,

        (False, False, True): _PUNCTUATION_REMOVAL_TABLE_WITH_NUMBERS,
        (True, False, True): _PUNCTUATION_REMOVAL_TABLE_WITH_LOWERCASE_AND_NUMBERS,
        (False, True, True): _PUNCTUATION_REMOVAL_TABLE_WITH_SPACE_AND_NUMBERS,
        (True, True, True): _PUNCTUATION_REMOVAL_TABLE_WITH_LOWERCASE_AND_SPACE_AND_NUMBERS,
    }[(lowercase, space, numbers)]
    return str(x).translate(punct_table)


def whitespace_normalize(text: str, remove_newlines: bool = False):
    ## Remove trailing whitespace at the end of each line
    text: str = re.sub(r'\s+$', '', text, flags=re.MULTILINE)

    if remove_newlines:
        text: str = text.replace('\n', '')
    else:
        ## Replace double newlines with single newlines
        text: str = re.sub(r'\n\n+', '\n', text)

    ## Replace double spaces with single spaces
    text: str = re.sub(r'  +', ' ', text)
    return text.strip()


def type_str(data: Any) -> str:
    if isinstance(data, type):
        if issubclass(data, Parameters):
            out: str = data.class_name
        else:
            out: str = str(data.__name__)
    else:
        out: str = str(type(data))
    ## Crocodile brackets mess up Aim's logging, they are treated as HTML tags.
    out: str = out.replace('<', '').replace('>', '')
    return out


def fn_str(fn):
    return f'{get_fn_spec(fn).resolved_name}'


def format_exception_msg(ex: Exception, short: bool = False, prefix: str = '[ERROR]') -> str:
    ## Ref: https://stackoverflow.com/a/64212552
    tb = ex.__traceback__
    trace = []
    while tb is not None:
        trace.append({
            "filename": tb.tb_frame.f_code.co_filename,
            "function_name": tb.tb_frame.f_code.co_name,
            "lineno": tb.tb_lineno
        })
        tb = tb.tb_next
    out = f'{prefix}: {type(ex).__name__}: "{str(ex)}"'
    if short:
        out += '\nTrace: '
        for trace_line in trace:
            out += f'{trace_line["filename"]}#{trace_line["lineno"]}; '
    else:
        out += '\nTraceback:'
        for trace_line in trace:
            out += f'\n\t{trace_line["filename"]} line {trace_line["lineno"]}, in {trace_line["function_name"]}...'
    return out.strip()


## ======================== Function utils ======================== ##
get_current_fn_name = lambda n=0: sys._getframe(n + 1).f_code.co_name  ## Ref: https://stackoverflow.com/a/31615605


def is_function(fn: Any) -> bool:
    ## Ref: https://stackoverflow.com/a/69823452/4900327
    return isinstance(fn, (
        types.FunctionType,
        types.MethodType,
        types.BuiltinFunctionType,
        types.BuiltinMethodType,
        types.LambdaType,
        functools.partial,
    ))


def call_str_to_params(
        call_str: str,
        callable_name_key: str = 'name',
        max_len: int = 1024,
) -> Tuple[List, Dict]:
    """Creates params dict from a call string."""
    if len(call_str) > max_len:  ## To prevent this attack: https://stackoverflow.com/a/54763776/4900327
        raise ValueError(f'We cannot parse `call_str` beyond {max_len} chars; found {len(call_str)} chars')
    call_str: str = call_str.strip()
    if not (call_str.find('(') < call_str.find(')')):
        raise ValueError(
            f'`call_str` must have one opening paren, followed by one closing paren; '
            f'found: `call_str`="{call_str}"'
        )
    if not call_str.endswith(')'):
        raise ValueError(f'`call_str` must end with a closing paren; found: `call_str`="{call_str}"')
    name: str = call_str.split('(')[0]
    args: List = []
    kwargs: Dict = {callable_name_key: name}
    if call_str != f'{name}()':
        ## We have some params:
        params_str: str = call_str.replace(f'{name}(', '')
        assert params_str.endswith(')')
        params_str: str = params_str[:-1]
        for param_str in params_str.split(','):
            param_str: str = param_str.strip()
            if '=' not in param_str:
                ## Not an arg-value pair, instead just arg:
                args.append(literal_eval(param_str))
            elif len(param_str.split('=')) != 2:
                ## Cannot resolve arg-value pair:
                raise ValueError(f'Found invalid arg-value pair "{param_str}" in `call_str`="{call_str}"')
            else:
                k, v = param_str.split('=')
                ## No, this is not a security issue. Ref: https://stackoverflow.com/a/7689085/4900327
                if k == name:
                    raise ValueError(f'Argument name and callable name overlap: "{name}"')
                kwargs[k] = literal_eval(v)
    return args, kwargs


def params_to_call_str(callable_name: str, args: List, kwargs: Dict) -> str:
    sep: str = ', '
    stringified = []
    if len(args) > 0:
        stringified.append(sep.join(args))
    if len(kwargs) > 0:
        stringified.append(sep.join([f'{k}={v}' for k, v in sorted(list(kwargs.items()), key=lambda x: x[0])]))
    return f'{callable_name}({sep.join(stringified)})'


def wrap_fn_output(fn: Callable, wrapper_fn: Callable) -> Callable:
    """
    Ensures a function always returns objects of a particular class.
    :param fn: original function to invoke.
    :param wrapper_fn: wrapper which takes as input the original function output and returns a different value.
    :return: wrapped function object.
    """

    def do(*args, **kwargs):
        return wrapper_fn(fn(*args, **kwargs))

    return do


def parsed_fn_source(function) -> Tuple[str, str]:
    # Get the source code of the function
    # Parse the source code into an AST
    parsed_source = ast.parse(inspect.getsource(function))
    # The first element of the body should be the FunctionDef node for the function
    function_node = parsed_source.body[0]
    # Extract the body of the FunctionDef node
    fn_source: str = ast.unparse(function_node)
    # Convert the body back to source code strings
    fn_body: str = '\n'.join([ast.unparse(stmt) for stmt in function_node.body])
    return fn_source, fn_body


class FunctionSpec(BaseModel):
    name: str
    qualname: str
    resolved_name: str
    source: str
    source_body: str
    args: Tuple[str, ...]
    varargs_name: Optional[str]
    kwargs: Tuple[str, ...]
    varkwargs_name: Optional[str]
    default_args: Dict[str, Any]
    default_kwargs: Dict[str, Any]
    ignored_args: Tuple[str, ...] = ('self', 'cls')

    class Config:
        ## Ref for Pydantic mutability: https://pydantic-docs.helpmanual.io/usage/models/#faux-immutability
        allow_mutation = False
        ## Ref for Extra.forbid: https://pydantic-docs.helpmanual.io/usage/model_config/#options
        extra = Extra.forbid
        ## Ref for Pydantic private attributes: https://pydantic-docs.helpmanual.io/usage/models/#private-model-attributes
        underscore_attrs_are_private = True
        ## Validates default values. Ref: https://pydantic-docs.helpmanual.io/usage/model_config/#options
        validate_all = True
        ## Validates typing by `isinstance` check. Ref: https://pydantic-docs.helpmanual.io/usage/model_config/#options
        arbitrary_types_allowed = True

    @root_validator(pre=False)
    def _remove_ignored(cls, params: Dict) -> Dict:
        ignored_args: Tuple[str, ...] = params['ignored_args']
        params['args'] = tuple(arg_name for arg_name in params['args'] if arg_name not in ignored_args)
        params['kwargs'] = tuple(arg_name for arg_name in params['kwargs'] if arg_name not in ignored_args)
        params['default_args'] = dict(
            (arg_name, default_val) for arg_name, default_val in params['default_args'].items()
            if arg_name not in ignored_args
        )
        params['default_kwargs'] = dict(
            (arg_name, default_val) for arg_name, default_val in params['default_kwargs'].items()
            if arg_name not in ignored_args
        )
        return params

    @property
    def args_and_kwargs(self) -> Tuple[str, ...]:
        return self.args + self.kwargs

    @property
    def default_args_and_kwargs(self) -> Dict[str, Any]:
        return {**self.default_args, **self.default_kwargs}

    @property
    def required_args_and_kwargs(self) -> Tuple[str, ...]:
        default_args_and_kwargs: Dict[str, Any] = self.default_args_and_kwargs
        return tuple(
            arg_name
            for arg_name in self.args_and_kwargs
            if arg_name not in default_args_and_kwargs
        )

    @property
    def num_args(self) -> int:
        return len(self.args)

    @property
    def num_kwargs(self) -> int:
        return len(self.kwargs)

    @property
    def num_args_and_kwargs(self) -> int:
        return self.num_args + self.num_kwargs

    @property
    def num_default_args(self) -> int:
        return len(self.default_args)

    @property
    def num_default_kwargs(self) -> int:
        return len(self.default_kwargs)

    @property
    def num_default_args_and_kwargs(self) -> int:
        return self.num_default_args + self.num_default_kwargs

    @property
    def num_required_args_and_kwargs(self) -> int:
        return self.num_args_and_kwargs - self.num_default_args_and_kwargs


def get_fn_spec(fn: Callable) -> FunctionSpec:
    if hasattr(fn, '__wrapped__'):
        """
        if a function is wrapped with decorators, unwrap and get all args
        eg: pd.read_csv.__code__.co_varnames returns (args, kwargs, arguments) as its wrapped by a decorator @deprecate_nonkeyword_arguments
        This line ensures to unwrap all decorators recursively
        """
        return get_fn_spec(fn.__wrapped__)
    argspec: inspect.FullArgSpec = inspect.getfullargspec(fn)  ## Ref: https://stackoverflow.com/a/218709

    args: Tuple[str, ...] = tuple(get_default(argspec.args, []))
    varargs_name: Optional[str] = argspec.varargs

    kwargs: Tuple[str, ...] = tuple(get_default(argspec.kwonlyargs, []))
    varkwargs_name: Optional[str] = argspec.varkw

    default_args: Tuple[Any, ...] = get_default(argspec.defaults, tuple())
    default_args: Dict[str, Any] = dict(zip(
        argspec.args[-len(default_args):],  ## Get's last len(default_args) values from the args list.
        default_args,
    ))
    default_kwargs: Dict[str, Any] = get_default(argspec.kwonlydefaults, dict())

    try:
        source, source_body = parsed_fn_source(fn)
    except IndentationError:
        source = inspect.getsource(fn)
        source_args_and_body = re.sub(r'^\s*(def\s+\w+\()', '', source, count=1, flags=re.MULTILINE).strip()
        source_body: str = source_args_and_body  ## Better than nothing.
    return FunctionSpec(
        name=fn.__name__,
        qualname=fn.__qualname__,
        resolved_name=fn.__module__ + "." + fn.__qualname__,
        source=source,
        source_body=source_body,
        args=args,
        varargs_name=varargs_name,
        kwargs=kwargs,
        varkwargs_name=varkwargs_name,
        default_args=default_args,
        default_kwargs=default_kwargs,
    )


def get_fn_args(
        fn: Union[Callable, FunctionSpec],
        *,
        ignore: Tuple[str, ...] = ('self', 'cls', 'kwargs'),
        include_args: bool = True,
        include_kwargs: bool = True,
        include_default: bool = True,
) -> Tuple[str, ...]:
    if isinstance(fn, FunctionSpec):
        fn_spec: FunctionSpec = fn
    else:
        fn_spec: FunctionSpec = get_fn_spec(fn)
    arg_names: List[str] = list()
    if include_args:
        arg_names.extend(fn_spec.args)
    if include_kwargs:
        arg_names.extend(fn_spec.kwargs)
    if include_default is False:
        ignore: List[str] = list(ignore) + list(fn_spec.default_args.keys()) + list(fn_spec.default_kwargs.keys())
    ignore: Set[str] = set(ignore)
    arg_names: Tuple[str, ...] = tuple(a for a in arg_names if a not in ignore)
    return arg_names


def filter_kwargs(fns: Union[Callable, List[Callable], Tuple[Callable, ...]], **kwargs) -> Dict[str, Any]:
    to_keep: Set = set()
    for fn in as_list(fns):
        fn_args: Tuple[str, ...] = get_fn_args(fn)
        to_keep.update(as_set(fn_args))
    filtered_kwargs: Dict[str, Any] = {
        k: kwargs[k]
        for k in kwargs
        if k in to_keep
    }
    return filtered_kwargs


## ======================== Class utils ======================== ##
def is_abstract(Class: Type) -> bool:
    return ABC in Class.__bases__


## Ref: https://stackoverflow.com/a/13624858/4900327
class classproperty(property):
    def __get__(self, obj, objtype=None):
        return super(classproperty, self).__get__(objtype)

    def __set__(self, obj, value):
        super(classproperty, self).__set__(type(obj), value)

    def __delete__(self, obj):
        super(classproperty, self).__delete__(type(obj))


## ======================== Typing utils ======================== ##
def safe_validate_arguments(f):
    names_to_fix = {n for n in BaseModel.__dict__ if not n.startswith('_')}

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        kwargs = {n[:-1] if n[:-1] in names_to_fix else n: v for n, v in kwargs.items()}
        return f(*args, **kwargs)

    def _create_param(p: inspect.Parameter) -> inspect.Parameter:
        default = Undefined if p.default is inspect.Parameter.empty else p.default
        return p.replace(name=f"{p.name}_", default=Field(default, alias=p.name))

    sig = inspect.signature(f)
    sig = sig.replace(parameters=[_create_param(p) if n in names_to_fix else p for n, p in sig.parameters.items()])

    wrapper.__signature__ = sig
    wrapper.__annotations__ = {f"{n}_" if n in names_to_fix else n: v for n, v in f.__annotations__.items()}

    try:
        return validate_arguments(
            wrapper,
            config={
                "allow_population_by_field_name": True,
                "arbitrary_types_allowed": True,
            }
        )
    except Exception as e:
        raise ValueError(
            f'Error creating model for function {get_fn_spec(f).resolved_name}.'
            f'\nEncountered Exception: {format_exception_msg(e)}'
        )


def not_impl(
        param_name: str,
        param_val: Any,
        supported: Optional[Union[List, Set, Tuple, Any]] = None,
) -> Exception:
    if not isinstance(param_name, str):
        raise ValueError(f'First value `param_name` must be a string.')
    param_val_str: str = str(param_val)
    if len(param_val_str) > 100:
        param_val_str: str = '\n' + param_val_str
    if supported is not None:
        supported: List = as_list(supported)
        return NotImplementedError(
            f'Unsupported value for param `{param_name}`. Valid values are: {supported}; '
            f'found {type_str(param_val)} having value: {param_val_str}'
        )

    return NotImplementedError(
        f'Unsupported value for param `{param_name}`; '
        f'found {type_str(param_val)} having value: {param_val_str}'
    )


def check_isinstance(x: Optional[Any], y: Union[List[Type], Tuple[Type, ...], Type], raise_error: bool = True):
    if x is None and y is type(None):
        return True
    assert isinstance(y, type) or (isinstance(y, (list, tuple)) and np.all([isinstance(z, type) for z in y]))
    if (isinstance(y, type) and isinstance(x, y)) or (isinstance(y, list) and np.any([isinstance(x, z) for z in y])):
        return True
    if raise_error:
        y_str: str = ', '.join([type_str(_y) for _y in as_list(y)])
        raise TypeError(
            f'Input parameter must be of type `{y_str}`; found type `{type_str(x)}` with value:\n{x}'
        )
    return False


def check_issubclass_or_none(x: Optional[Any], y: Type, raise_error: bool = True):
    if x is None:
        return True
    return check_issubclass(x, y, raise_error=raise_error)


def check_issubclass(x: Optional[Any], y: Type, raise_error: bool = True):
    if x is None:
        return False
    assert isinstance(x, type)
    assert isinstance(y, type) or (isinstance(y, list) and np.all([isinstance(z, type) for z in y]))
    if (isinstance(y, type) and issubclass(x, y)) or (isinstance(y, list) and np.any([issubclass(x, z) for z in y])):
        return True
    if raise_error:
        raise TypeError(f'Input parameter must be a subclass of type {str(y)}; found type {type(x)} with value {x}')
    return False


def is_scalar(x: Any, method: Literal['numpy', 'pandas'] = 'pandas') -> bool:
    if method == 'pandas':
        ## Ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.types.is_scalar.html
        ## Actual code: github.com/pandas-dev/pandas/blob/0402367c8342564538999a559e057e6af074e5e4/pandas/_libs/lib.pyx#L162
        return pd_is_scalar(x)
    if method == 'numpy':
        ## Ref: https://numpy.org/doc/stable/reference/arrays.scalars.html#built-in-scalar-types
        return np.isscalar(x)
    raise NotImplementedError(f'Unsupported method: "{method}"')


def get_classvars(cls) -> List[str]:
    return [
        var_name
        for var_name, typing_ in typing.get_type_hints(cls).items()
        if typing_.__origin__ is typing.ClassVar
    ]


def get_classvars_typing(cls) -> Dict[str, Any]:
    return {
        var_name: typing_.__args__[0]
        for var_name, typing_ in typing.get_type_hints(cls).items()
        if typing.get_origin(typing_) is typing.ClassVar
    }


## ======================== Import utils ======================== ##
@contextmanager
def optional_dependency(
        *names: Union[List[str], str],
        error: Literal['raise', 'warn', 'ignore'] = "ignore",
        warn_every_time: bool = False,
        __WARNED_OPTIONAL_MODULES: Set[str] = set()  ## "Private" argument
) -> Optional[Union[Tuple[types.ModuleType, ...], types.ModuleType]]:
    """
    A contextmanager (used with "with") which passes code if optional dependencies are not present.
    Ref: https://stackoverflow.com/a/73838546/4900327

    Parameters
    ----------
    names: str or list of strings.
        The module name(s) which are optional.
    error: str {'raise', 'warn', 'ignore'}
        What to do when a dependency is not found in the "with" block:
        * raise : Raise an ImportError.
        * warn: print a warning (see `warn_every_time`).
        * ignore: do nothing.
    warn_every_time: bool
        Whether to warn every time an import is tried. Only applies when error="warn".
        Setting this to True will result in multiple warnings if you try to
        import the same library multiple times.

    Usage
    -----
    ## 1. Only run code if modules exist, otherwise ignore:
        with optional_dependency("pydantic", "sklearn", error="ignore"):
            from pydantic import BaseModel
            from sklearn.metrics import accuracy_score
            class AccuracyCalculator(BaseModel):
                decimals: int = 5
                def calculate(self, y_pred: List, y_true: List) -> float:
                    return round(accuracy_score(y_true, y_pred), self.decimals)
            print("Defined AccuracyCalculator in global context")
        print("Will be printed finally")  ## Always prints

    ## 2. Print warnings with error="warn". Multiple warings are be printed via `warn_every_time=True`.
        with optional_dependency("pydantic", "sklearn", error="warn"):
            from pydantic import BaseModel
            from sklearn.metrics import accuracy_score
            class AccuracyCalculator(BaseModel):
                decimals: int = 5
                def calculate(self, y_pred: List, y_true: List) -> float:
                    return round(accuracy_score(y_true, y_pred), self.decimals)
            print("Defined AccuracyCalculator in global context")
        print("Will be printed finally")  ## Always prints

    ## 3. Raise ImportError warnings with error="raise":
        with optional_dependency("pydantic", "sklearn", error="raise"):
            from pydantic import BaseModel
            from sklearn.metrics import accuracy_score
            class AccuracyCalculator(BaseModel):
                decimals: int = 5
                def calculate(self, y_pred: List, y_true: List) -> float:
                    return round(accuracy_score(y_true, y_pred), self.decimals)
            print("Defined AccuracyCalculator in global context")
        print("Will be printed finally")  ## Always prints
    """
    assert error in {"raise", "warn", "ignore"}
    names: Optional[Set[str]] = set(names)
    try:
        yield None
    except (ImportError, ModuleNotFoundError) as e:
        missing_module: str = e.name
        if len(names) > 0 and missing_module not in names:
            raise e  ## A non-optional dependency is missing
        if error == "raise":
            raise e
        if error == "warn":
            if missing_module not in __WARNED_OPTIONAL_MODULES or warn_every_time is True:
                msg = f'Missing optional dependency "{missing_module}". Use pip or conda to install.'
                print(f'Warning: {msg}')
                __WARNED_OPTIONAL_MODULES.add(missing_module)


class alias(auto):
    def __init__(self, *aliases):
        if len(aliases) == 0:
            raise ValueError('Cannot have empty alias() call.')
        for a in aliases:
            if not isinstance(a, str):
                raise ValueError(f'All aliases for must be strings; found alias of type {type(a)} having value: {a}')
        self.names = aliases
        self.enum_name = None

    def __repr__(self) -> str:
        return str(self)

    def __str__(self):
        if self.enum_name is not None:
            return self.enum_name
        return self.alias_repr

    @property
    def alias_repr(self) -> str:
        return str(f'alias:{list(self.names)}')

    def __setattr__(self, attr_name: str, attr_value: Any):
        if attr_name == 'value':
            ## because alias subclasses auto and does not set value, enum.py:143 will try to set value
            self.enum_name = attr_value
        else:
            super(alias, self).__setattr__(attr_name, attr_value)

    def __getattribute__(self, attr_name: str):
        """
        Refer these lines in Python 3.10.9 enum.py:

        class _EnumDict(dict):
            ...
            def __setitem__(self, key, value):
                ...
                elif not _is_descriptor(value):
                    ...
                    if isinstance(value, auto):
                        if value.value == _auto_null:
                            value.value = self._generate_next_value(
                                    key,
                                    1,
                                    len(self._member_names),
                                    self._last_values[:],
                                    )
                            self._auto_called = True
                        value = value.value
                    ...
                ...
            ...

        """
        if attr_name == 'value':
            if object.__getattribute__(self, 'enum_name') is None:
                ## Gets _auto_null as alias inherits auto class but does not set `value` class member; refer enum.py:142
                try:
                    return object.__getattribute__(self, 'value')
                except Exception as e:
                    from enum import _auto_null
                    return _auto_null
            return self
        return object.__getattribute__(self, attr_name)


_DEFAULT_REMOVAL_TABLE = str.maketrans(
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz',
    ' -_.:;,'  ## Will be removed
)


class AutoEnum(str, Enum):
    """
    Utility class which can be subclassed to create enums using auto() and alias().
    Also provides utility methods for common enum operations.
    """

    def __init__(self, value: Union[str, alias]):
        self.aliases: Tuple[str, ...] = tuple()
        if isinstance(value, alias):
            self.aliases: Tuple[str, ...] = value.names

    @classmethod
    def _missing_(cls, enum_value: Any):
        ## Ref: https://stackoverflow.com/a/60174274/4900327
        ## This is needed to allow Pydantic to perform case-insensitive conversion to AutoEnum.
        return cls.from_str(enum_value=enum_value, raise_error=True)

    def _generate_next_value_(name, start, count, last_values):
        return name

    @property
    def str(self) -> str:
        return self.__str__()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.__class__.__name__ + '.' + self.name)

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def matches(self, enum_value: str) -> bool:
        return self is self.from_str(enum_value, raise_error=False)

    @classmethod
    def matches_any(cls, enum_value: str) -> bool:
        return cls.from_str(enum_value, raise_error=False) is not None

    @classmethod
    def does_not_match_any(cls, enum_value: str) -> bool:
        return not cls.matches_any(enum_value)

    @classmethod
    def display_names(cls, **kwargd) -> str:
        return str([enum_value.display_name(**kwargd) for enum_value in list(cls)])

    def display_name(self, *, sep: str = ' ') -> str:
        return sep.join([
            word.lower() if word.lower() in ('of', 'in', 'the') else word.capitalize()
            for word in str(self).split('_')
        ])

    @classmethod
    def _initialize_lookup(cls):
        if '_value2member_map_normalized_' not in cls.__dict__:  ## Caching values for fast retrieval.
            cls._value2member_map_normalized_ = {}

            def _set_normalized(e, normalized_e_name):
                if normalized_e_name in cls._value2member_map_normalized_:
                    raise ValueError(
                        f'Cannot register enum "{e.name}"; '
                        f'another enum with the same normalized name "{normalized_e_name}" already exists.'
                    )
                cls._value2member_map_normalized_[normalized_e_name] = e

            for e in list(cls):
                _set_normalized(e, cls._normalize(e.name))
                if len(e.aliases) > 0:
                    ## Add the alias-repr to the lookup:
                    _set_normalized(e, cls._normalize(alias(*e.aliases).alias_repr))
                    for e_alias in e.aliases:
                        _set_normalized(e, cls._normalize(e_alias))

    @classmethod
    def from_str(cls, enum_value: str, raise_error: bool = True) -> Optional:
        """
        Performs a case-insensitive lookup of the enum value string among the members of the current AutoEnum subclass.
        :param enum_value: enum value string
        :param raise_error: whether to raise an error if the string is not found in the enum
        :return: an enum value which matches the string
        :raises: ValueError if raise_error is True and no enum value matches the string
        """
        if isinstance(enum_value, cls):
            return enum_value
        if enum_value is None and raise_error is False:
            return None
        if not isinstance(enum_value, str) and raise_error is True:
            raise ValueError(f'Input should be a string; found type {type(enum_value)}')
        cls._initialize_lookup()
        enum_obj: Optional[AutoEnum] = cls._value2member_map_normalized_.get(cls._normalize(enum_value))
        if enum_obj is None and raise_error is True:
            raise ValueError(f'Could not find enum with value {repr(enum_value)}; available values are: {list(cls)}.')
        return enum_obj

    @classmethod
    def _normalize(cls, x: str) -> str:
        ## Found to be faster than .translate() and re.sub() on Python 3.10.6
        return str(x).translate(_DEFAULT_REMOVAL_TABLE)

    @classmethod
    def convert_keys(cls, d: Dict) -> Dict:
        """
        Converts string dict keys to the matching members of the current AutoEnum subclass.
        Leaves non-string keys untouched.
        :param d: dict to transform
        :return: dict with matching string keys transformed to enum values
        """
        out_dict = {}
        for k, v in d.items():
            if isinstance(k, str) and cls.from_str(k, raise_error=False) is not None:
                out_dict[cls.from_str(k, raise_error=False)] = v
            else:
                out_dict[k] = v
        return out_dict

    @classmethod
    def convert_keys_to_str(cls, d: Dict) -> Dict:
        """
        Converts dict keys of the current AutoEnum subclass to the matching string key.
        Leaves other keys untouched.
        :param d: dict to transform
        :return: dict with matching keys of the current AutoEnum transformed to strings.
        """
        out_dict = {}
        for k, v in d.items():
            if isinstance(k, cls):
                out_dict[str(k)] = v
            else:
                out_dict[k] = v
        return out_dict

    @classmethod
    def convert_values(
            cls,
            d: Union[Dict, Set, List, Tuple],
            raise_error: bool = False
    ) -> Union[Dict, Set, List, Tuple]:
        """
        Converts string values to the matching members of the current AutoEnum subclass.
        Leaves non-string values untouched.
        :param d: dict, set, list or tuple to transform.
        :param raise_error: raise an error if unsupported type.
        :return: data structure with matching string values transformed to enum values.
        """
        if isinstance(d, dict):
            return cls.convert_dict_values(d)
        if isinstance(d, list):
            return cls.convert_list(d)
        if isinstance(d, tuple):
            return tuple(cls.convert_list(d))
        if isinstance(d, set):
            return cls.convert_set(d)
        if raise_error:
            raise ValueError(f'Unrecognized data structure of type {type(d)}')
        return d

    @classmethod
    def convert_dict_values(cls, d: Dict) -> Dict:
        """
        Converts string dict values to the matching members of the current AutoEnum subclass.
        Leaves non-string values untouched.
        :param d: dict to transform
        :return: dict with matching string values transformed to enum values
        """
        out_dict = {}
        for k, v in d.items():
            if isinstance(v, str) and cls.from_str(v, raise_error=False) is not None:
                out_dict[k] = cls.from_str(v, raise_error=False)
            else:
                out_dict[k] = v
        return out_dict

    @classmethod
    def convert_list(cls, l: Union[List, Tuple]) -> List:
        """
        Converts string list itmes to the matching members of the current AutoEnum subclass.
        Leaves non-string items untouched.
        :param l: list to transform
        :return: list with matching string items transformed to enum values
        """
        out_list = []
        for item in l:
            if isinstance(item, str) and cls.matches_any(item):
                out_list.append(cls.from_str(item))
            else:
                out_list.append(item)
        return out_list

    @classmethod
    def convert_set(cls, s: Set) -> Set:
        """
        Converts string list itmes to the matching members of the current AutoEnum subclass.
        Leaves non-string items untouched.
        :param s: set to transform
        :return: set with matching string items transformed to enum values
        """
        out_set = set()
        for item in s:
            if isinstance(item, str) and cls.matches_any(item):
                out_set.add(cls.from_str(item))
            else:
                out_set.add(item)
        return out_set

    @classmethod
    def convert_values_to_str(cls, d: Dict) -> Dict:
        """
        Converts dict values of the current AutoEnum subclass to the matching string value.
        Leaves other values untouched.
        :param d: dict to transform
        :return: dict with matching values of the current AutoEnum transformed to strings.
        """
        out_dict = {}
        for k, v in d.items():
            if isinstance(v, cls):
                out_dict[k] = str(v)
            else:
                out_dict[k] = v
        return out_dict


## ======================== List utils ======================== ##
def is_list_like(l: Union[List, Tuple, np.ndarray, PandasSeries, DaskSeries]) -> bool:
    if isinstance(l, (list, tuple, ValuesView, ItemsView, PandasSeries, DaskSeries)):
        return True
    if isinstance(l, np.ndarray) and l.ndim == 1:
        return True
    return False


def is_not_empty_list_like(l: ListOrTuple) -> bool:
    return is_list_like(l) and len(l) > 0


def is_empty_list_like(l: ListOrTuple) -> bool:
    return not is_not_empty_list_like(l)


def assert_not_empty_list(l: List):
    assert is_not_empty_list(l)


def assert_not_empty_list_like(l: ListOrTuple, error_message=''):
    assert is_not_empty_list_like(l), error_message


def is_not_empty_list(l: List) -> bool:
    return isinstance(l, list) and len(l) > 0


def is_empty_list(l: List) -> bool:
    return not is_not_empty_list(l)


def as_list(l) -> List:
    if is_list_or_set_like(l):
        return list(l)
    return [l]


def list_pop_inplace(l: List, *, pop_condition: Callable) -> List:
    assert isinstance(l, list)  ## Needs to be a mutable
    ## Iterate backwards to preserve indexes while iterating
    for i in range(len(l) - 1, -1, -1):  # Iterate backwards
        if pop_condition(l[i]):
            l.pop(i)  ## Remove the item inplace
    return l


def set_union(*args) -> Set:
    _union: Set = set()
    for s in args:
        if isinstance(s, (pd.Series, np.ndarray)):
            s: List = s.tolist()
        s: Set = set(s)
        _union: Set = _union.union(s)
    return _union


def set_intersection(*args) -> Set:
    _intersection: Optional[Set] = None
    for s in args:
        if isinstance(s, (pd.Series, np.ndarray)):
            s: List = s.tolist()
        s: Set = set(s)
        if _intersection is None:
            _intersection: Set = s
        else:
            _intersection: Set = _intersection.intersection(s)
    return _intersection


def filter_string_list(l: List[str], pattern: str, ignorecase: bool = False) -> List[str]:
    """
    Filter a list of strings based on an exact match to a regex pattern. Leaves non-string items untouched.
    :param l: list of strings
    :param pattern: Regex pattern used to match each item in list of strings.
    Strings which are not a regex pattern will be expected to exactly match.
    E.g. the pattern 'abcd' will only match the string 'abcd'.
    To match 'abcdef', pattern 'abcd.*' should be used.
    To match 'xyzabcd', patterm '.*abcd' should be used.
    To match 'abcdef', 'xyzabcd' and 'xyzabcdef', patterm '.*abcd.*' should be used.
    :param ignorecase: whether to ignore case while matching the pattern to the strings.
    :return: filtered list of strings which match the pattern.
    """
    if not pattern.startswith('^'):
        pattern = '^' + pattern
    if not pattern.endswith('$'):
        pattern = pattern + '$'
    flags = 0
    if ignorecase:
        flags = flags | re.IGNORECASE
    return [x for x in l if not isinstance(x, str) or len(re.findall(pattern, x, flags=flags)) > 0]


def keep_values(
        a: Union[List, Tuple, Set, Dict],
        values: Any,
) -> Union[List, Tuple, Set, Dict]:
    values: Set = as_set(values)
    if isinstance(a, list):
        return list(x for x in a if x in values)
    elif isinstance(a, tuple):
        return tuple(x for x in a if x in values)
    elif isinstance(a, set):
        return set(x for x in a if x in values)
    elif isinstance(a, dict):
        return {k: v for k, v in a.items() if v in values}
    raise NotImplementedError(f'Unsupported data structure: {type(a)}')


def remove_values(
        a: Union[List, Tuple, Set, Dict],
        values: Any,
) -> Union[List, Tuple, Set, Dict]:
    values: Set = as_set(values)
    if isinstance(a, list):
        return list(x for x in a if x not in values)
    elif isinstance(a, tuple):
        return tuple(x for x in a if x not in values)
    elif isinstance(a, set):
        return set(x for x in a if x not in values)
    elif isinstance(a, dict):
        return {k: v for k, v in a.items() if v not in values}
    raise NotImplementedError(f'Unsupported data structure: {type(a)}')


def remove_nulls(
        a: Union[List, Tuple, Set, Dict],
) -> Union[List, Tuple, Set, Dict]:
    if isinstance(a, list):
        return list(x for x in a if is_not_null(x))
    elif isinstance(a, tuple):
        return tuple(x for x in a if is_not_null(x))
    elif isinstance(a, set):
        return set(x for x in a if is_not_null(x))
    elif isinstance(a, dict):
        return {k: v for k, v in a.items() if is_not_null(v)}
    raise NotImplementedError(f'Unsupported data structure: {type(a)}')


def elvis(d: Optional[Union[Dict, Any]], *args) -> Optional[Any]:
    if len(args) == 0:
        raise ValueError('Must pass non-empty list of keys to match when using elvis operator')
    val: Union[Dict, Any] = get_default(d, {})
    for k in args:
        val: Union[Dict, Any] = get_default(val, {})
        if isinstance(val, dict):
            val: Union[Dict, Any] = val.get(k)
        else:
            return val
    return val


## ======================== Tuple utils ======================== ##
def as_tuple(l) -> Tuple:
    if is_list_or_set_like(l):
        return tuple(l)
    return (l,)


## ======================== Set utils ======================== ##
def is_set_like(l: Any) -> bool:
    return isinstance(l, (set, frozenset, KeysView))


def is_list_or_set_like(l: Union[List, Tuple, np.ndarray, PandasSeries, Set, frozenset]):
    return is_list_like(l) or is_set_like(l)


def get_subset(small_list: ListOrTuple, big_list: ListOrTuple) -> Set:
    assert is_list_like(small_list)
    assert is_list_like(big_list)
    return set.intersection(set(small_list), set(big_list))


def is_subset(small_list: ListOrTuple, big_list: ListOrTuple) -> bool:
    return len(get_subset(small_list, big_list)) == len(small_list)


def as_set(s) -> Set:
    if isinstance(s, set):
        return s
    if is_list_or_set_like(s):
        return set(s)
    return {s}


## ======================== Dict utils ======================== ##
@safe_validate_arguments
def append_to_keys(d: Dict, prefix: Union[List[str], str] = '', suffix: Union[List[str], str] = '') -> Dict:
    keys = set(d.keys())
    for k in keys:
        new_keys = {f'{p}{k}' for p in as_list(prefix)} \
                   | {f'{k}{s}' for s in as_list(suffix)} \
                   | {f'{p}{k}{s}' for p in as_list(prefix) for s in as_list(suffix)}
        for k_new in new_keys:
            d[k_new] = d[k]
    return d


@safe_validate_arguments
def transform_keys_case(d: Dict, case: Literal['lower', 'upper'] = 'lower'):
    """
    Converts string dict keys to either uppercase or lowercase. Leaves non-string keys untouched.
    :param d: dict to transform
    :param case: desired case, either 'lower' or 'upper'
    :return: dict with case-transformed keys
    """
    out = {}
    for k, v in d.items():
        if isinstance(k, str):
            if case == 'lower':
                out[k.lower()] = v
            elif case == 'upper':
                out[k.upper()] = v
        else:
            out[k] = v
    return out


@safe_validate_arguments
def transform_values_case(d: Dict, case: Literal['lower', 'upper'] = 'lower'):
    """
    Converts string dict values to either uppercase or lowercase. Leaves non-string values untouched.
    :param d: dict to transform
    :param case: desired case, either 'lower' or 'upper'
    :return: dict with case-transformed values
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, str):
            if case == 'lower':
                out[k] = v.lower()
            elif case == 'upper':
                out[k] = v.upper()
        else:
            out[k] = v
    return out


def dict_set_default(d: Dict, default_params: Dict) -> Dict:
    """
    Sets default values in a dict for missing keys
    :param d: input dict
    :param default_params: dict of default values
    :return: input dict with default values populated for missing keys
    """
    if d is None:
        d = {}
    assert isinstance(d, dict)
    if default_params is None:
        return d
    assert isinstance(default_params, dict)
    for k, v in default_params.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            ## We need to go deeper:
            d[k] = dict_set_default(d[k], v)
        else:
            d.setdefault(k, v)
    return d


def sorted_dict(
        d: Dict,
        *,
        by: Literal['key', 'value'] = 'key',
        reverse: bool = False,
        order: Optional[List] = None,
) -> List[Tuple]:
    assert by in {'key', 'value'}
    if order is not None:
        order: List = as_list(order)
        assert by == 'key'
        out_d: Dict = {}
        for k in order:
            ## In order
            out_d[k] = d[k]
        for k in set(d.keys()) - set(order):
            ## Unordered
            out_d[k] = d[k]
        return list(out_d.items())
    else:
        if by == 'key':
            return sorted(d.items(), key=lambda x: str(x[0]), reverse=reverse)
        elif by == 'value':
            return sorted(d.items(), key=lambda x: str(x[1]), reverse=reverse)
        else:
            raise not_impl('by', by)


def dict_key_with_best_value(
        d: Dict,
        *,
        how: Literal['max', 'min'],
) -> Any:
    assert how in {'max', 'min'}
    sorted_items: List[Tuple] = sorted_dict(
        d,
        by='value',
        reverse={
            'min': False,
            'max': True,
        }[how]
    )
    return sorted_items[0][0]


@safe_validate_arguments
def filter_keys(
        d: Dict,
        keys: Union[List, Tuple, Set, str],
        how: Literal['include', 'exclude'] = 'include',
) -> Dict:
    """
    Filter values in a dict based on a list of keys.
    :param d: dict to filter
    :param keys: list of keys to include/exclude.
    :param how: whether to keep or remove keys in filtered_keys list.
    :return: dict with filtered list of keys
    """
    keys: Set = as_set(keys)
    if how == 'include':
        return keep_keys(d, keys)
    elif how == 'exclude':
        return remove_keys(d, keys)
    else:
        raise NotImplementedError(f'Invalid value for parameter `how`: "{how}"')


def filter_values(
        struct: Union[List, Tuple, Set, Dict, str],
        fn: Callable,
        *,
        raise_error: bool = True,
) -> Optional[Any]:
    if (is_list_like(struct) or is_set_like(struct)) and len(struct) > 0:
        return type(struct)([x for x in struct if fn(x)])
    elif is_dict_like(struct):
        return dict({k: v for k, v in struct.items() if fn(v)})
    if raise_error:
        raise ValueError(f'Unsupported structure: {type_str(struct)}')
    return None


def keep_keys(d: Dict, keys: Union[List, Tuple, Set, str]) -> Dict:
    keys: Set = as_set(keys)
    return {k: d[k] for k in keys if k in d}


def remove_keys(d: Dict, keys: Union[List, Tuple, Set, str]) -> Dict:
    keys: Set = as_set(keys)
    return {k: d[k] for k in d if k not in keys}


class UniqueDict(dict):
    def __setitem__(self, key, value):  ## Dict which rejects updates for existing keys.
        if key not in self:
            dict.__setitem__(self, key, value)
        else:
            raise KeyError("Key already exists")


@safe_validate_arguments
def convert_and_filter_keys_on_enum(
        d: Dict,
        AutoEnumClass: AutoEnum.__class__,
        how: Literal['include', 'exclude'] = 'include',
) -> Dict:
    """
    Filter values in a dict based on those matching an enum.
    :param d: dict to filter.
    :param AutoEnumClass: AutoEnum class on which to filter.
    :param how: whether to keep or remove keys in the AutoEnum class.
    :return: dict with filtered list of keys
    """
    if AutoEnumClass is None:
        return {}
    assert isinstance(AutoEnumClass, AutoEnum.__class__)
    d = AutoEnumClass.convert_keys(d)
    return filter_keys(d, list(AutoEnumClass), how=how)


def filter_keys_on_pattern(
        d: Dict,
        key_pattern: str,
        ignorecase: bool = False,
        how: Literal['include', 'exclude'] = 'include',
):
    """
    Filter string keys in a dict based on a regex pattern.
    :param d: dict to filter
    :param key_pattern: regex pattern used to match keys.
    :param how: whether to keep or remove keys.
    Follows same rules as `filter_string_list` method, i.e. only checks string keys and retains non-string keys.
    :return: dict with filtered keys
    """
    keys: List = list(d.keys())
    filtered_keys: List = filter_string_list(keys, key_pattern, ignorecase=ignorecase)
    return filter_keys(d, filtered_keys, how=how)


def is_not_empty_dict(d: Dict) -> bool:
    return is_dict_like(d) and len(d) > 0


def is_empty_dict(d: Dict) -> bool:
    return not is_not_empty_dict(d)


def assert_not_empty_dict(d: Dict):
    assert is_not_empty_dict(d)


def is_dict_like(d: Union[Dict, defaultdict]) -> bool:
    return isinstance(d, (dict, defaultdict))


def is_list_or_dict_like(d: Any) -> bool:
    return is_list_like(d) or is_dict_like(d)


def is_list_of_dict_like(d: List[Dict]) -> bool:
    if not is_list_like(d):
        return False
    for x in d:
        if not is_dict_like(x):
            return False
    return True


def is_dict_like_or_list_of_dict_like(d: Union[Dict, List[Dict]]) -> bool:
    if is_dict_like(d):
        return True
    elif is_list_like(d):
        return is_list_of_dict_like(d)
    return False


def eval_dict_values(params: Dict):
    if not isinstance(params, dict):
        raise ValueError(f"{params} should be of type dict")
    updated_dict = {}
    for parameter, value in params.items():
        try:
            updated_dict[parameter] = literal_eval(value)
        except:
            updated_dict[parameter] = value
    return updated_dict


def invert_dict(d: Dict) -> Dict:
    if not isinstance(d, dict):
        raise ValueError(f'{d} should be of type dict')
    d_inv: Dict = {v: k for k, v in d.items()}
    if len(d_inv) != len(d):
        raise ValueError(f'Dict is not invertible as values are not unique.')
    return d_inv


def iter_dict(d, depth: int = 1, *, _cur_depth: int = 0):
    """
    Recursively iterate over nested dictionaries and yield keys at each depth.

    :param d: The dictionary to iterate over.
    :param depth: The current depth of recursion (used for tracking depth of keys).
    :return: Yields tuples where the first elements are keys at different depths, and the last element is the value.
    """
    assert isinstance(d, dict), f'Input must be a dictionary, found: {type(d)}'
    assert isinstance(depth, int) and depth >= 1, f'depth must be an integer (1 or more)'

    for k, v in d.items():
        if isinstance(v, dict) and _cur_depth < depth - 1:
            # If the value is a dictionary, recurse
            for subkeys in iter_dict(v, _cur_depth=_cur_depth + 1, depth=depth):
                yield (k,) + subkeys
        else:
            # If the value is not a dictionary, yield the key-value pair
            yield (k, v)


## ======================== NumPy utils ======================== ##
def is_numpy_integer_array(data: Any) -> bool:
    if not isinstance(data, np.ndarray):
        return False
    return issubclass(data.dtype.type, np.integer)


def is_numpy_float_array(data: Any) -> bool:
    if not isinstance(data, np.ndarray):
        return False
    return issubclass(data.dtype.type, float)


def is_numpy_string_array(data: Any) -> bool:
    if not isinstance(data, np.ndarray):
        return False
    return issubclass(data.dtype.type, str)


## Ref (from Pytorch tests):
## github.com/pytorch/pytorch/blob/e180ca652f8a38c479a3eff1080efe69cbc11621/torch/testing/_internal/common_utils.py#L349
NUMPY_TO_TORCH_DTYPE_MAP = {}
with optional_dependency('torch'):
    import torch

    NUMPY_TO_TORCH_DTYPE_MAP = {
        np.bool_: torch.bool,
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128
    }
    TORCH_TO_NUMPY_DTYPE_MAP = {v: k for k, v in NUMPY_TO_TORCH_DTYPE_MAP.items()}


def infer_np_dtype(
        data: Union[List, np.ndarray, pd.Series, 'torch.Tensor'],
        sample_size: SampleSizeType = True,
        str_to_object: bool = True,
        return_str_for_collection: bool = False,
) -> Optional[Union[np.dtype, Type, str]]:
    """
    Fast inference of the numpy dtype in a list.
    Note: we cannot use pandas.api.types.infer_dtype because it returns Pandas dtypes, not numpy.

    :param data: data collection (usually a list or tuple).
    :param sample_size: amount of data to subsample (without replacement) in order to determine the dtype.
        If False, it will not subsample data. If True, it will use entire data.
        If 0.0 < sample < 1.0, then we will subsample a fraction of the data.
        If 1 <= sample, we will subsample these many rows of data.
    :param str_to_object: whether to treat string as objects rather than np.unicode_ (like "U<1").
    :param return_str_for_collection: whether to return the string 'collection' for collections like list, set,
        numpy array, etc.
    :return:
    """
    if isinstance(data, (np.ndarray, pd.Series)):
        return data.dtype
    with optional_dependency('torch'):
        if isinstance(data, torch.Tensor):
            return TORCH_TO_NUMPY_DTYPE_MAP[data.dtype]

    data: List = as_list(data)
    dtypes: Set[Union[Type, np.dtype]] = set()
    has_nulls: bool = False
    for x in random_sample(data, n=sample_size, replacement=False):
        if str_to_object and np.issubdtype(type(x), np.character):
            ## Fast convert str, np.str_ and np.unicode_ to object:
            return object
        if not is_scalar(x):
            ## Fast return for collections such as list, tuple, dict, set, np.ndarray, Tensors.
            if return_str_for_collection:
                return 'collection'
            return object
        if is_null(x):  ## Checks NaNs, None, and pd.NaT
            has_nulls: bool = True
        else:
            dtypes.add(type(x))
    if len(dtypes) == 0:
        ## All NaNs / None
        return None
    elif len(dtypes) == 1:
        dtype = next(iter(dtypes))
        ## Ref: https://numpy.org/doc/stable/reference/arrays.dtypes.html#Built-in%20Python%20types
        if dtype in {bool, np.bool_, float, np.float_, complex, np.complex_, bytes}:
            return np.dtype(dtype)
    return _np_dtype_fallback(dtypes, has_nulls=has_nulls, str_to_object=str_to_object)


def _np_dtype_fallback(dtypes: Union[Type, Set[Type]], has_nulls: bool, str_to_object: bool):
    ## We have one or more dtypes, which might be Python types or Numpy dtypes.
    ## We will now check if all the dtypes have a common parent, based on the NumPy scalar types hierarchy:
    ## i.e. https://numpy.org/doc/stable/reference/arrays.scalars.html
    if _all_are_np_subtypes(dtypes, {np.bool_, }):
        if has_nulls:
            return np.float_  ## Converts None to NaN, and True/False to 1.0/0.0
        return np.bool_
    elif _all_are_np_subtypes(dtypes, {np.bool_, np.integer}):
        if has_nulls:
            return np.float_  ## Converts None to NaN, True/False to 1.0/0.0, and 123 to 123.0
        return np.int_
    elif _all_are_np_subtypes(dtypes, {np.bool_, np.integer, np.floating}):
        return np.float_
    elif _all_are_np_subtypes(dtypes, {np.character, }):
        if str_to_object:
            return object
        return np.unicode_
    elif _all_are_np_subtypes(dtypes, {np.bool_, np.integer, np.floating, np.complex_}):
        return np.complex_
    ## Multiple, heterogeneous and incompatible types, return as object
    return object


def _all_are_np_subtypes(

        dtypes: Union[Type, Set[Type]],
        parent_dtypes: Union[Type, Set[Type]],
) -> bool:
    ## Note: the following hold for Python types when checking with np.issubdtype:
    ## np.issubdtype(bool, np.bool_) is True
    ## np.issubdtype(int, np.integer) is True (however, np.issubdtype(bool, np.integer) is False)
    ## np.issubdtype(float, np.floating) is True (however, np.issubdtype(int, np.floating) is False)
    ## np.issubdtype(complex, np.complex_) is True (however, np.issubdtype(float, np.complex_) is False)
    ## np.issubdtype(str, np.character) is True
    dtypes: Set[Type] = as_set(dtypes)
    parent_dtypes: Set[Type] = as_set(parent_dtypes)
    return all({
        any({np.issubdtype(dtype, parent_dtype) for parent_dtype in parent_dtypes})
        for dtype in dtypes
    })


is_even = lambda x: x % 2 == 0
is_odd = lambda x: x % 2 == 1


## ======================== Pandas utils ======================== ##
def get_num_non_null_columns_per_row(df: PandasDataFrame) -> PandasSeries:
    ## Ref: https://datascience.stackexchange.com/a/16801/35826
    assert isinstance(df, PandasDataFrame)
    return (~df.isna()).sum(axis=1)


def get_max_num_non_null_columns_per_row(df: PandasDataFrame) -> int:
    assert isinstance(df, PandasDataFrame)
    return get_num_non_null_columns_per_row(df).max()


## ======================== Utils for multiple collections ======================== ##
def only_item(
        d: Union[Dict, List, Tuple, Set, np.ndarray, PandasSeries],
        raise_error: bool = True,
) -> Union[Dict, List, Tuple, Set, np.ndarray, PandasSeries, Any]:
    if not (is_list_or_set_like(d) or is_dict_like(d)):
        return d
    if len(d) == 1:
        if is_dict_like(d):
            return next(iter(d.items()))
        return next(iter(d))
    if raise_error:
        raise ValueError(f'Expected input {type(d)} to have only one item; found {len(d)} elements.')
    return d


def only_key(d: Dict, raise_error: bool = True) -> Union[Any]:
    if not is_dict_like(d):
        return d
    if len(d) == 1:
        return next(iter(d.keys()))
    if raise_error:
        raise ValueError(f'Expected input {type(d)} to have only one item; found {len(d)} elements.')
    return d


def only_value(d: Dict, raise_error: bool = True) -> Union[Any]:
    if not is_dict_like(d):
        return d
    if len(d) == 1:
        return next(iter(d.values()))
    if raise_error:
        raise ValueError(f'Expected input {type(d)} to have only one item; found {len(d)} elements.')
    return d


def is_1d_array(l: Union[List, Tuple]):
    return is_list_like(l) and len(l) > 0 and not is_list_like(l[0])


def is_2d_array(l: Union[List, Tuple]):
    return is_list_like(l) and len(l) > 0 and is_list_like(l[0])


def convert_1d_or_2d_array_to_dataframe(data: SeriesOrArray1DOrDataFrameOrArray2D) -> PandasDataFrame:
    if is_1d_array(data):
        data: PandasSeries = convert_1d_array_to_series(data)
    if isinstance(data, PandasSeries) or is_2d_array(data):
        data: PandasDataFrame = pd.DataFrame(data)
    assert isinstance(data, PandasDataFrame)
    return data


def convert_1d_array_to_series(data: SeriesOrArray1D):
    if len(data) == 0:
        raise ValueError(f'Cannot convert empty data structure to series')
    if isinstance(data, PandasSeries):
        return data
    if not is_list_like(data):
        raise ValueError(f'Cannot convert non list-like data structure to series')
    return pd.Series(data)


def flatten1d(
        l: Union[List, Tuple, Set, Any],
        output_type: Type = list
) -> Union[List, Set, Tuple]:
    assert output_type in {list, set, tuple}
    if not is_list_or_set_like(l):
        return l
    out = []
    for x in l:
        out.extend(as_list(flatten1d(x)))
    return output_type(out)


def flatten2d(
        l: Union[List, Tuple, Set, Any],
        outer_type: Type = list,
        inner_type: Type = tuple,
) -> Union[List, Tuple, Set, Any]:
    assert outer_type in {list, set, tuple}
    assert inner_type in {list, set, tuple}
    if not is_list_or_set_like(l):
        return l
    out: List[Union[List, Set, Tuple]] = [
        flatten1d(x, output_type=inner_type)
        for x in l
    ]
    return outer_type(out)


def get_unique(
        data: SeriesOrArray1DOrDataFrameOrArray2D,
        exclude_nans: bool = True
) -> Set[Any]:
    if data is None:
        return set()
    if isinstance(data, PandasSeries) or isinstance(data, PandasDataFrame):
        data: np.ndarray = data.values
    if is_2d_array(data):
        data: np.ndarray = convert_1d_or_2d_array_to_dataframe(data).values
    if not isinstance(data, np.ndarray):
        data: np.ndarray = np.array(data)
    flattened_data = data.ravel('K')  ## 1-D array of all data (w/ nans). Ref: https://stackoverflow.com/a/26977495
    if len(flattened_data) == 0:
        return set()
    if exclude_nans:
        flattened_data = flattened_data[~pd.isnull(flattened_data)]
    flattened_data = np.unique(flattened_data)
    return set(flattened_data)


def any_item(
        struct: Union[List, Tuple, Set, Dict, ValuesView, str],
        *,
        seed: Optional[int] = None,
        raise_error: bool = True,
) -> Optional[Any]:
    py_random: random.Random = random.Random(seed)
    if (is_list_like(struct) or is_set_like(struct)) and len(struct) > 0:
        return py_random.choice(tuple(struct))
    elif is_dict_like(struct):
        k: Any = any_key(struct, seed=seed, raise_error=raise_error)
        v: Any = struct[k]
        return k, v  ## Return an item
    elif isinstance(struct, str):
        return py_random.choice(struct)
    if raise_error:
        raise ValueError(f'Unsupported structure: {type_str(struct)}')
    return None


def any_key(d: Dict, *, seed: Optional[int] = None, raise_error: bool = True) -> Optional[Any]:
    py_random: random.Random = random.Random(seed)
    if is_not_empty_dict(d):
        return py_random.choice(sorted(list(d.keys())))
    if raise_error:
        raise ValueError(
            f'Expected input to be a non-empty dict; '
            f'found {type_str(d) if not is_dict_like(d) else "empty dict"}.'
        )
    return None


def any_value(d: Dict, *, seed: Optional[int] = None, raise_error: bool = True) -> Optional[Any]:
    k: Any = any_key(d, seed=seed, raise_error=raise_error)
    return d[k]


def first_item(
        struct: Union[List, Tuple, Set, Dict, str],
        *,
        raise_error: bool = True,
) -> Optional[Any]:
    if is_dict_like(struct):
        k: Any = first_key(struct, raise_error=raise_error)
        v: Any = struct[k]
        return k, v  ## Return an item
    elif is_list_like(struct) or is_set_like(struct) or isinstance(struct, str):
        return list(struct)[0]
    if raise_error:
        raise ValueError(f'Unsupported structure: {type_str(struct)}')
    return None


def first_key(d: Dict, *, raise_error: bool = True) -> Optional[Any]:
    if is_not_empty_dict(d):
        return list(d.keys())[0]
    if raise_error:
        raise ValueError(
            f'Expected input to be a non-empty dict; '
            f'found {type_str(d) if not is_dict_like(d) else "empty dict"}.'
        )
    return None


def first_value(d: Dict, *, raise_error: bool = True) -> Optional[Any]:
    k: Any = first_key(d, raise_error=raise_error)
    return d[k]


def partial_sort(
        struct: Union[List[Any], Tuple[Any]],
        order: Union[List[Any], Tuple[Any], Any],
) -> Union[List[Any], Tuple[Any]]:
    """
    Partialls sorts a list or tuple.
    Only workds
    """
    ## Dictionary to store the count of each element in order
    order: List[Any] = as_list(order)
    order_count: Dict[Any, int] = {item: 0 for item in order}

    # Two lists: one for elements in order and one for the rest
    ordered_part: List[Any] = []
    rest_part: List[Any] = []

    for item in struct:
        if item in order_count:
            # If the item is in order, increment the count and add to ordered_part
            order_count[item] += 1
        else:
            # Otherwise, add to rest_part
            rest_part.append(item)

    ## Construct the final ordered part based on the count
    for item in order:
        ordered_part.extend([item] * order_count[item])

    ## Combine the ordered part with the rest
    out: List[Any] = ordered_part + rest_part
    if isinstance(struct, tuple):
        return tuple(out)
    return out


def irange(low: Union[float, int], high: Union[float, int], step: Union[float, int] = 1):
    """Inclusive range, useful for coding up math notation."""
    if not (isinstance(low, int) or (isinstance(low, float) and low.is_integer())):
        raise ValueError(f'low={low} is not a valid integer.')
    if not (isinstance(high, int) or (isinstance(high, float) and high.is_integer())):
        raise ValueError(f'high={high} is not a valid integer.')
    if not (isinstance(step, int) or (isinstance(step, float) and step.is_integer())):
        raise ValueError(f'step={step} is not a valid integer.')
    return range(int(low), int(high) + 1, int(step))


def frange(low: float, high: float, step: float, *, limits: bool = True) -> List[float]:
    """Inclusive range, useful for coding up math notation."""
    assert isinstance(low, (int, float)) and isinstance(high, (int, float)) and isinstance(step, (int, float))
    out: List[float] = [
        x for x in [round(float(x) / step, 0) * step for x in np.arange(low, high + step, step)]
        if low <= x <= high
    ]
    if limits:
        out: List[float] = sorted(set(out).union({low, high}))
    return out


def is_valid_idx(
        l: Union[List, Tuple, np.ndarray, pd.Series, pd.DataFrame],
        idx: int,
        *,
        raise_error: bool = True,
) -> bool:
    assert isinstance(l, (list, tuple, np.ndarray, pd.Series, pd.DataFrame))
    assert idx >= 0, f'Can only check validity of non-negative indexes'
    if len(l) == 0:
        if raise_error:
            raise ValueError(f'Cannot check validity of index for empty {str(type(l))}')
        return False  ## No index is valid
    return idx in range(0, len(l))


def iter_batches(
        struct: Union[List, Tuple, Set, Dict, np.ndarray, pd.Series, pd.DataFrame, int],
        batch_size: int,
) -> Generator[List[Any], None, None]:
    assert isinstance(batch_size, int) and batch_size > 0
    if is_int_in_floats_clothing(struct):
        struct: List[int] = list(range(int(struct)))
    if is_set_like(struct):
        struct_type: Type = set
    elif is_dict_like(struct):
        struct_type: Type = dict
    else:
        struct_type: Optional[Type] = None
    if struct_type is not None:
        buf: List[Any] = []
        if isinstance(struct, dict):
            struct: ItemsView = struct.items()
        for x in struct:
            buf.append(x)
            if len(buf) == batch_size:
                yield struct_type(buf)
                buf: List[Any] = []
        if len(buf) > 0:
            yield struct_type(buf)
    else:
        struct_len: int = len(struct)
        for i in range(0, struct_len, batch_size):
            if isinstance(struct, (pd.Series, pd.DataFrame)):
                yield struct.iloc[i: min(i + batch_size, struct_len)]
            else:
                yield struct[i: min(i + batch_size, struct_len)]


def mean(vals):
    return sum(vals) / len(vals)


def random_sample(
        data: Union[List, Tuple, Set, np.ndarray],
        n: SampleSizeType,
        *,
        replacement: bool = False,
        seed: Optional[int] = None,
) -> Union[List, np.ndarray]:
    """
    Sample data randomly from a list or numpy array, with or without replacement.
    :param data: list or numpy array to randomly subsample.
    :param n: size of the sample to return.
    :param replacement: whether to sample with replacement or not.
    :param seed: optional random seed to use for reproducibility.
    :return: list or numpy array of randomly-sampled data.
    """
    np_random = np.random.RandomState(seed)
    py_random = random.Random(seed)
    if is_set_like(data):
        data: List = list(data)
    if not is_list_like(data):
        raise ValueError(
            f'Input `data` must be {list}, {tuple} or {np.ndarray}; '
            f'found object of type {type(data)}'
        )
    if len(data) == 1:
        return data
    l: Union[List, np.ndarray] = data
    length: int = len(l)
    n: int = resolve_sample_size(sample_size=n, length=length)
    if replacement:
        ## Subsample with replacement:
        ## Ref: https://stackoverflow.com/a/71892814/4900327
        if isinstance(l, (list, tuple)):
            if n < 50:
                return py_random.choices(l, k=n)
            else:
                return [l[idx] for idx in np_random.randint(0, len(l), n)]
        elif isinstance(l, np.ndarray):
            if n < 25:
                return [l[idx] for idx in (py_random.randrange(0, len(l)) for _ in range(n))]
            else:
                return np_random.choice(l, n, replace=True)
    else:
        ## Subsample without replacement:
        ## Ref: https://stackoverflow.com/a/71892814/4900327
        if isinstance(l, (list, tuple)):
            return py_random.sample(l, n)
        elif isinstance(l, np.ndarray):
            return np_random.choice(l, n, replace=False)
    raise NotImplementedError(f'Unsupported input data type: {type(data)}')


def values_dist(vals: Union[List, Tuple, np.ndarray, pd.Series]) -> pd.Series:
    assert isinstance(vals, (list, tuple, np.ndarray, pd.Series))
    val_counts: pd.Series = pd.Series(Counter(vals))  ## Includes nan and None as keys.
    return val_counts / val_counts.sum()


def sample_idxs_match_distribution(
        source: Union[List, Tuple, np.ndarray, pd.Series],
        target: Union[List, Tuple, np.ndarray, pd.Series],
        n: Optional[int] = None,
        seed: Optional[int] = None,
        shuffle: bool = True,
        target_is_dist: bool = False,
) -> np.ndarray:
    """
    Values from current series based on another distribution, and return randomly-shuffled indexes from the source.
    Selecting these indexes will give a distribution from the source whicha matches that of the target distribution.
    """
    if not target_is_dist:
        target_prob_dist: pd.Series = values_dist(target)
    else:
        target_prob_dist: pd.Series = target
    assert isinstance(target_prob_dist, pd.Series)
    assert abs(float(target_prob_dist.sum()) - 1.0) <= 1e-2  ## Sum of probs should be exactly or very close to 1.

    assert isinstance(source, (list, tuple, np.ndarray, pd.Series))
    source_vc: pd.Series = pd.Series(Counter(source))
    # print(f'\nsource_vc:\n{source_vc}')
    # print(f'\ntarget_prob_dist:\n{target_prob_dist}')
    missing_source_vals: Set = set(target_prob_dist.index) - set(source_vc.index)
    if len(missing_source_vals) > 0:
        raise ValueError(f'Cannot sample; the following values are missing in the source: {missing_source_vals}')

    n: int = get_default(n, len(source))
    max_n_sample: pd.Series = (source_vc / target_prob_dist).apply(
        lambda max_n_sample_category: min(max_n_sample_category, n),
    )
    # print(f'\n\nmax_n_sample:\n{max_n_sample}')
    max_n_sample: int = math.floor(min(max_n_sample.dropna()))
    # print(f'Max possible sample size: {max_n_sample}')
    source_value_wise_count_to_sample: pd.Series = (target_prob_dist * max_n_sample).round(0).astype(int)
    source_value_wise_count_to_sample: Dict[Any, int] = source_value_wise_count_to_sample.to_dict()
    ## Select random indexes:
    source_val_idxs: Dict[Any, List[int]] = {val: [] for val in source_vc.index}
    for idx, val in enumerate(source):
        if val in source_value_wise_count_to_sample:
            source_val_idxs[val].append(idx)
    sampled_idxs: np.array = np.array(flatten1d([
        random_sample(source_val_idxs[val], n=req_source_val_count, seed=seed)
        for val, req_source_val_count in source_value_wise_count_to_sample.items()
    ]))
    if shuffle:
        sampled_idxs: np.ndarray = np.random.RandomState(seed).permutation(sampled_idxs)
    return sampled_idxs


def entropy(probabilities: np.ndarray) -> float:
    # Remove zero probabilities to avoid issues with logarithm
    if not isinstance(probabilities, np.ndarray):
        probabilities: np.ndarray = np.array(probabilities)
        assert is_numpy_float_array(probabilities)
    prob_sum: float = float(probabilities.sum())
    if abs(1 - prob_sum) > 1e-2:
        raise ValueError(f'Probabilities sum to {prob_sum}, should sum to 1')
    probabilities = probabilities[probabilities > 0]
    # probabilities += 1e-9
    _entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    return _entropy


def shuffle_items(
        struct: Union[List, Tuple, Set, Dict, str],
        *,
        seed: Optional[int] = None,
        dict_return_values: bool = False,
) -> Generator[Any, None, None]:
    if isinstance(struct, set):
        struct: Tuple = tuple(struct)
    elif isinstance(struct, dict):
        if dict_return_values:
            struct: Tuple = tuple(struct.values())
        else:
            struct: Tuple = tuple(struct.items())
    rnd_idxs: List[int] = list(range(len(struct)))
    random.Random(seed).shuffle(rnd_idxs)
    for rnd_idx in rnd_idxs:
        yield struct[rnd_idx]


def random_cartesian_product(*lists, seed: Optional[int] = None, n: int):
    rnd = random.Random(seed)
    cartesian_idxs: Set[Tuple[int, ...]] = set()
    list_lens: List[int] = [len(l) for l in lists]
    max_count: int = 1
    for l_len in list_lens:
        max_count *= l_len
    if max_count < n:
        raise ValueError(f'At most {max_count} cartesian product elements can be created.')
    while len(cartesian_idxs) < n:
        rnd_idx: Tuple[int, ...] = tuple(
            rnd.randint(0, l_len - 1)
            for l_len in list_lens
        )
        if rnd_idx not in cartesian_idxs:
            cartesian_idxs.add(rnd_idx)
            elem = []
            for l_idx, l in zip(rnd_idx, lists):
                elem.append(l[l_idx])
            yield elem


def argmax(d: Union[List, Tuple, np.ndarray, Dict, Set]) -> Any:
    if is_set_like(d):
        raise ValueError(f'Cannot get argmax from a {type_str(d)}.')
    if is_dict_like(d):
        ## Get key pertaining to max value:
        return max(d, key=d.get)
    assert is_list_like(d)
    return max([(i, x) for (i, x) in enumerate(d)], key=lambda x: x[1])[0]


def argmin(d: Union[List, Tuple, np.ndarray, Dict, Set]) -> Any:
    if is_set_like(d):
        raise ValueError(f'Cannot get argmin from a {type_str(d)}.')
    if is_dict_like(d):
        ## Get key pertaining to max value:
        return min(d, key=d.get)
    assert is_list_like(d)
    return min([(i, x) for (i, x) in enumerate(d)], key=lambda x: x[1])[0]


def best_k(
        vals: np.ndarray,
        k: int,
        *,
        how: Literal['min', 'max'],
        sort: Optional[Literal['ascending', 'descending']] = None,
        indexes_only: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Efficiently gets the top-k elements from a numpy array."""
    assert isinstance(k, int) and k > 0
    ## np.argpartition creates a new array with the top-k/bottom-k scores in the head/tail k elements,
    ## but these k are not actually sorted.
    if how == 'min':
        sort: str = sort if sort is not None else 'ascending'
        bottom_k_idxs: np.ndarray = np.argpartition(vals, k, axis=0)[:k]
        ## Index vals to get bottom-k values, unsorted:
        bottom_k_vals: np.ndarray = vals[bottom_k_idxs]
        ## Get argsorted indexes for the bottom-k values (between 1 & k).
        ## We then use this to index the bottom-k-indexes array:
        if sort == 'ascending':
            bottom_k_idxs_sorted: np.ndarray = bottom_k_idxs[bottom_k_vals.argsort(axis=0)]
            bottom_k_vals_sorted = np.sort(bottom_k_vals, axis=0)
        elif sort == 'descending':
            bottom_k_idxs_sorted: np.ndarray = bottom_k_idxs[bottom_k_vals.argsort(axis=0)[::-1]]
            bottom_k_vals_sorted = np.sort(bottom_k_vals, axis=0)[::-1]
        else:
            raise NotImplementedError(f'Unsupported value of `sort`: {sort}')
        # print(f'bottom_k_vals_sorted: {bottom_k_vals_sorted}')
        # print(f'bottom_k_idxs_sorted: {bottom_k_idxs_sorted}')
        # assert bool((vals[bottom_k_idxs_sorted] == bottom_k_vals_sorted).all())
        if indexes_only:
            return bottom_k_idxs_sorted
        return bottom_k_idxs_sorted, bottom_k_vals_sorted
    elif how == 'max':
        sort: str = sort if sort is not None else 'descending'
        top_k_idxs: np.ndarray = np.argpartition(vals, -k, axis=0)[-k:]
        ## Index vals to get top-k values, unsorted:
        top_k_vals: np.ndarray = vals[top_k_idxs]
        ## Get argsorted indexes for the top-k values (between 1 & k).
        ## We then use this to index the top-k-indexes array:
        if sort == 'ascending':
            top_k_idxs_sorted: np.ndarray = top_k_idxs[top_k_vals.argsort(axis=0)]
            top_k_vals_sorted = np.sort(top_k_vals, axis=0)
        elif sort == 'descending':
            top_k_idxs_sorted: np.ndarray = top_k_idxs[top_k_vals.argsort(axis=0)[::-1]]
            top_k_vals_sorted = np.sort(top_k_vals, axis=0)[::-1]
        else:
            raise NotImplementedError(f'Unsupported value of `sort`: {sort}')
        # print(f'top_k_vals_sorted: {top_k_vals_sorted}')
        # print(f'top_k_idxs_sorted: {top_k_idxs_sorted}')
        # assert bool((vals[top_k_idxs_sorted] == top_k_vals_sorted).all())
        if indexes_only:
            return top_k_idxs_sorted
        return top_k_idxs_sorted, top_k_vals_sorted
    else:
        raise ValueError(f'Unsupported value for `how`: {how}')


def clip(low: Union[int, float], val: Union[int, float], high: Union[int, float]):
    assert isinstance(low, (int, float, np.integer, np.float_))
    assert isinstance(high, (int, float, np.integer, np.float_))
    assert isinstance(val, (int, float, np.integer, np.float_))
    assert low <= high
    return max(low, min(val, high))


def pad_interval(low: Union[int, float], high: Union[int, float], pad: float) -> Tuple[float, float]:
    assert isinstance(low, (int, float, np.integer, np.float_))
    assert isinstance(high, (int, float, np.integer, np.float_))
    assert isinstance(pad, (int, float, np.integer, np.float_)) and 0.0 <= pad <= 1.0
    assert low <= high
    width: float = float(high) - float(low)
    pad: float = float(pad)
    return (low - width * pad, high + width * pad)


def rolling_avg(iterator: Union[Iterable, Iterator, Generator]) -> float:
    if not hasattr(iterator, '__iter__'):
        raise ValueError(
            f'Cannot calculate rolling average from an object which is neither an iterator or generator; '
            f'found object of type {type_str(iterator)}.'
        )
    avg: float = 0
    for i, x in enumerate(iterator):
        avg = update_rolling_avg(avg, i, x)
    return avg


def update_rolling_avg(avg_i: float, i: int, val_i: float) -> float:
    """
    Calculates a rolling average.
    :param avg_i: the current average.
    :param i: the i'th index (starting from 0)
    :param val_i: the i'th value.
    :return: the updated average.

    Example usage:
    n: int = 1_000_000
    l: List[int] = list(range(1, n+1))  ## We know this adds up to n*(n+1)/2, thus the average is (n+1)/2
    avg: float = 0
    for i, x in enumerate(l):
        avg = update_rolling_avg(avg, i, x)
    assert avg == sum(l)/n == (n+1)/2
    """
    n: int = i + 1
    return ((n - 1) * avg_i + val_i) / n


@safe_validate_arguments
def relative_increase(
        prev: float,
        cur: float,
        *,
        how: Literal['ratio', 'pct'] = 'ratio',
        decimals: Optional[int] = None,
) -> float:
    increase_frac: float = cur / prev
    if how == 'ratio':
        if decimals is None:
            decimals: int = 5
        return round(increase_frac - 1, decimals)
    elif how == 'pct':
        if decimals is None:
            decimals: int = 2
        return round(100 * (increase_frac - 1), decimals)
    elif how == 'bps':
        if decimals is None:
            decimals: int = 1
        return round(100 * 100 * (increase_frac - 1), decimals)
    else:
        raise NotImplementedError(f'Unsupported `method`: "{how}"')


class Registry(ABC):
    """
    A registry for subclasses. When a base class extends Registry, its subclasses will automatically be registered,
     without any code in the base class to do so explicitly.
    This coding trick allows us to maintain the Dependency Inversion Principle, as the base class does not have to
     depend on any subclass implementation; in the base class code, we can instead retrieve the subclass in the registry
     using a key, and then interact with the retrieved subclass using the base class interface methods (which we assume
     the subclass has implemented as per the Liskov Substitution Principle).

    Illustrative example:
        Suppose we have abstract base class AbstractAnimal.
        This is registered as a registry via:
            class AbstractAnimal(Parameters, Registry, ABC):
                pass
        Then, subclasses of AbstractAnimal will be automatically registered:
            class Dog(AbstractAnimal):
                name: str
        Now, we can extract the subclass using the registered keys (of which the class-name is always included):
            AbstractAnimalSubclass = AbstractAnimal.get_subclass('Dog')
            dog = AbstractAnimalSubclass(name='Sparky')

        We can also set additional keys to register the subclass against:
            class AnimalType(AutoEnum):
                CAT = auto()
                DOG = auto()
                BIRD = auto()

            class Dog(AbstractAnimal):
                aliases = [AnimalType.DOG]

            AbstractAnimalSubclass = AbstractAnimal.get_subclass(AnimalType.DOG)
            dog = AbstractAnimalSubclass(name='Sparky')

        Alternately, the registry keys can be set using the _registry_keys() classmethod:
            class Dog(AbstractAnimal):
                @classmethod
                def _registry_keys(cls) -> List[Any]:
                    return [AnimalType.DOG]
    """
    _registry: ClassVar[Dict[Any, Dict[str, Type]]] = {}  ## Dict[key, Dict[classname, Class]
    _registry_base_class: ClassVar[Optional[Type[BaseModel]]] = None
    _classvars_typing_dict: ClassVar[Optional[Dict[str, Any]]] = None
    _classvars_BaseModel: ClassVar[Optional[Type[BaseModel]]] = None
    _allow_multiple_subclasses: ClassVar[bool] = False
    _allow_subclass_override: ClassVar[bool] = False
    _dont_register: ClassVar[bool] = False
    aliases: ClassVar[Tuple[str, ...]] = tuple()

    def __init_subclass__(cls, **kwargs):
        """
        Register any subclass with the base class. A child class is registered as long as it is imported/defined.
        """
        super().__init_subclass__(**kwargs)
        if cls in Registry.__subclasses__():
            ## The current class is a direct subclass of Registry (i.e. it is the base class of the hierarchy).
            cls._registry: Dict[Any, Dict[str, Type]] = {}
            cls._registry_base_class: Type = cls
            cls.__set_classvars_typing()
        else:
            ## The current class is a subclass of a Registry-subclass, and is not abstract; register this.
            if not is_abstract(cls) and not cls._dont_register:
                cls._pre_registration_hook()
                cls.__set_classvars_typing()
                cls.__validate_classvars_BaseModel()
                cls.__register_subclass()

    @classmethod
    def __set_classvars_typing(cls):
        classvars_typing_dict: Dict[str, Any] = {
            var_name: typing_
            for var_name, typing_ in get_classvars_typing(cls).items()
            if not var_name.startswith('_')
        }
        cls._classvars_typing_dict: ClassVar[Dict[str, Any]] = classvars_typing_dict

        class Config(Parameters.Config):
            extra = Extra.ignore

        cls._classvars_BaseModel: ClassVar[Type[BaseModel]] = create_model_from_typeddict(
            typing_extensions.TypedDict(f'{cls.__name__}_ClassVarsBaseModel', classvars_typing_dict),
            warnings=False,
            __config__=Config
        )

    @classmethod
    def __validate_classvars_BaseModel(cls):
        ## Gives the impression of validating ClassVars on concrete subclasses in the hierarchy.
        classvar_values: Dict[str, Any] = {}
        for classvar, type_ in cls._classvars_typing_dict.items():
            if not hasattr(cls, classvar):
                if ABC not in cls.__bases__:
                    ## Any concrete class must have all classvars set with values.
                    raise ValueError(
                        f'You must set a value for class variable "{classvar}" on subclass "{cls.__name__}".\n'
                        f'Custom type-hints might be one reason why "{classvar}" is not recognized. '
                        f'If you have added custom type-hints, please try removing them and set "{classvar}" like so: `{classvar} = <value>`'
                    )
            else:
                classvar_value = getattr(cls, classvar)
                if hasattr(type_, '__origin__'):
                    if type_.__origin__ == typing.Union and len(type_.__args__) == 2 and type(None) in type_.__args__:
                        ## It is something like Optional[str], Optional[List[str]], etc.
                        args = set(type_.__args__)
                        args.remove(type(None))
                        classvar_type = next(iter(args))
                    else:
                        classvar_type = type_.__origin__
                    if classvar_type in {set, list, tuple} and classvar_value is not None:
                        classvar_value = classvar_type(as_list(classvar_value))
                classvar_values[classvar] = classvar_value
        classvar_values: BaseModel = cls._classvars_BaseModel(**classvar_values)
        for classvar, type_ in cls._classvars_typing_dict.items():
            if not hasattr(cls, classvar):
                if ABC not in cls.__bases__:
                    ## Any concrete class must have all classvars set with values.
                    raise ValueError(
                        f'You must set a value for class variable "{classvar}" on subclass "{cls.__name__}".\n'
                        f'Custom type-hints might be one reason why "{classvar}" is not recognized. '
                        f'If you have added custom type-hints, please try removing them and set "{classvar}" like so: `{classvar} = <value>`'
                    )
            else:
                setattr(cls, classvar, classvar_values.__getattribute__(classvar))

    @classmethod
    def _pre_registration_hook(cls):
        pass

    @classmethod
    def __register_subclass(cls):
        subclass_name: str = str(cls.__name__).strip()
        cls.__add_to_registry(subclass_name, cls)  ## Always register subclass name
        for k in set(as_list(cls.aliases) + as_list(cls._registry_keys())):
            if k is not None:
                cls.__add_to_registry(k, cls)

    @classmethod
    @validate_arguments
    def __add_to_registry(cls, key: Any, subclass: Type):
        subclass_name: str = subclass.__name__
        if isinstance(key, (str, AutoEnum)):
            ## Case-insensitive matching:
            keys_to_register: List[str] = [str_normalize(key)]
        elif isinstance(key, tuple):
            keys_to_register: List[Tuple] = [tuple(
                ## Case-insensitive matching:
                str_normalize(k) if isinstance(k, (str, AutoEnum)) else k
                for k in key
            )]
        else:
            keys_to_register: List[Any] = [key]
        for k in keys_to_register:
            if k not in cls._registry:
                cls._registry[k] = {subclass_name: subclass}
                continue
            ## Key is in the registry
            registered: Dict[str, Type] = cls._registry[k]
            registered_names: Set[str] = set(registered.keys())
            assert len(registered_names) > 0, f'Invalid state: key {k} is registered to an empty dict'
            if subclass_name in registered_names and cls._allow_subclass_override is False:
                raise KeyError(
                    f'A subclass with name {subclass_name} is already registered against key {k} for registry under '
                    f'{cls._registry_base_class}; overriding subclasses is not permitted.'
                )
            elif subclass_name not in registered_names and cls._allow_multiple_subclasses is False:
                assert len(registered_names) == 1, \
                    f'Invalid state: _allow_multiple_subclasses is False but we have multiple subclasses registered ' \
                    f'against key {k}'
                raise KeyError(
                    f'Key {k} already is already registered to subclass {next(iter(registered_names))}; registering '
                    f'multiple subclasses to the same key is not permitted.'
                )
            cls._registry[k] = {
                **registered,
                ## Add or override the subclass names
                subclass_name: subclass,
            }

    @classmethod
    def get_subclass(
            cls,
            key: Any,
            raise_error: bool = True,
            *args,
            **kwargs,
    ) -> Optional[Union[Type, List[Type]]]:
        if isinstance(key, (str, AutoEnum)):
            Subclass: Optional[Dict[str, Type]] = cls._registry.get(str_normalize(key))
        else:
            Subclass: Optional[Dict[str, Type]] = cls._registry.get(key)
        if Subclass is None:
            if raise_error:
                raise KeyError(
                    f'Could not find subclass of {cls} using key: "{key}" (type={type(key)}). '
                    f'Available keys are: {set(cls._registry.keys())}'
                )
            return None
        if len(Subclass) == 1:
            return next(iter(Subclass.values()))
        return list(Subclass.values())

    @classmethod
    def subclasses(cls, keep_abstract: bool = False) -> Set[Type]:
        available_subclasses: Set[Type] = set()
        for k, d in cls._registry.items():
            for subclass in d.values():
                if subclass == cls._registry_base_class:
                    continue
                if is_abstract(subclass) and keep_abstract is False:
                    continue
                if isinstance(subclass, type) and issubclass(subclass, cls):
                    available_subclasses.add(subclass)
        return available_subclasses

    @classmethod
    def remove_subclass(cls, subclass: Union[Type, str]):
        name: str = subclass
        if isinstance(subclass, type):
            name: str = subclass.__name__
        for k, d in cls._registry.items():
            for subclass_name, subclass in list(d.items()):
                if str_normalize(subclass_name) == str_normalize(name):
                    d.pop(subclass_name, None)

    @classmethod
    def _registry_keys(cls) -> Optional[Union[List[Any], Any]]:
        return None


def set_param_from_alias(
        params: Dict,
        param: str,
        alias: Union[Tuple[str, ...], List[str], Set[str], str],
        remove_alias: bool = True,
        prioritize_aliases: bool = False,
        default: Optional[Any] = None,
):
    if prioritize_aliases:
        param_names: List = as_list(alias) + [param]
    else:
        param_names: List = [param] + as_list(alias)
    if remove_alias:
        value: Optional[Any] = get_default(*[params.pop(param_name, None) for param_name in param_names], default)
    else:
        value: Optional[Any] = get_default(*[params.get(param_name, None) for param_name in param_names], default)
    if value is not None:
        ## If none are set, use default value:
        params[param] = value


## Ref: https://stackoverflow.com/q/6760685/4900327, Method 2 base class.
## The metaclass method in the above link did not work well with multiple inheritance.
class Singleton:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.__instance, cls):
            cls.__instance = super(Singleton, cls).__new__(cls)
        return cls.__instance

    @classproperty
    def instance(cls):
        return cls.__instance


class Utility:
    def __init__(self):
        raise TypeError(f'Cannot instantiate utility class "{str(self.__class__)}"')


ParametersSubclass = TypeVar('ParametersSubclass', bound='Parameters')


class NeverFailJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        # print(f'Running NeverFailJsonEncoder')
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, pd.Series, list, set, tuple)):
            return obj.tolist()
        elif isinstance(obj, complex):
            return obj.real, obj.imag
        elif isinstance(obj, (
                types.FunctionType,
                types.MethodType,
                types.BuiltinFunctionType,
                types.BuiltinMethodType,
                types.LambdaType,
                functools.partial,
        )):
            return {
                '<function>': f'{obj.__module__}.{obj.__qualname__}{inspect.signature(obj)}'
            }
        with optional_dependency('torch'):
            import torch
            if isinstance(obj, torch.dtype):
                return str(obj)
        try:
            return super(NeverFailJsonEncoder, self).default(obj)
        except TypeError as e:
            obj_members: List[str] = []
            for k, v in obj.__dict__.items():
                if is_function(v):
                    continue
                k_str: str = str(k)
                # try:
                #     v_str: str = json.dumps(
                #         v,
                #         cls=NeverFailJsonEncoder,
                #         skipkeys=self.skipkeys,
                #         ensure_ascii=self.ensure_ascii,
                #         check_circular=self.check_circular,
                #         allow_nan=self.allow_nan,
                #         sort_keys=self.sort_keys,
                #         indent=self.indent,
                #         separators=(self.item_separator, self.key_separator),
                #     )
                # except TypeError as e:
                #     v_str: str = '...'
                v_str: str = '...'
                obj_members.append(f'{k_str}={v_str}')
            obj_members_str: str = ', '.join(obj_members)
            return f'{obj.__class__.__name__}({obj_members_str})'


class Parameters(BaseModel, ABC):
    ## Ref on Pydantic + ABC: https://pydantic-docs.helpmanual.io/usage/models/#abstract-base-classes
    ## Needed to work with Registry.alias...this needs to be on a subclass of `BaseModel`.
    aliases: ClassVar[Tuple[str, ...]] = tuple()
    dict_exclude: ClassVar[Tuple[str, ...]] = tuple()

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except Exception as e:
            raise ValueError(
                f'Cannot create Pydantic instance of type "{self.class_name}".'
                f'\nEncountered exception: {format_exception_msg(e)}'
            )

    @classproperty
    def class_name(cls) -> str:
        return str(cls.__name__)  ## Will return the child class name.

    @classmethod
    def param_names(cls, **kwargs) -> Set[str]:
        # superclass_params: Set[str] = set(super(Parameters, cls).schema(**kwargs)['properties'].keys())
        class_params: Set[str] = set(cls.schema(**kwargs)['properties'].keys())
        return class_params  # .union(superclass_params)

    @classmethod
    def param_default_values(cls, **kwargs) -> Dict:
        return {
            param: param_schema['default']
            for param, param_schema in cls.schema(**kwargs)['properties'].items()
            if 'default' in param_schema  ## The default value might be None
        }

    @classmethod
    def _clear_extra_params(cls, params: Dict) -> Dict:
        return {k: v for k, v in params.items() if k in cls.param_names()}

    def dict(self, *args, exclude: Optional[Any] = None, **kwargs) -> Dict:
        exclude: Set[str] = as_set(get_default(exclude, [])).union(as_set(self.dict_exclude))
        return super(Parameters, self).dict(*args, exclude=exclude, **kwargs)

    def json(self, *args, encoder: Optional[Any] = None, indent: Optional[int] = None, **kwargs) -> str:
        if encoder is None:
            encoder = functools.partial(json.dumps, cls=NeverFailJsonEncoder, indent=indent)
        return super(Parameters, self).json(*args, encoder=encoder, **kwargs)

    @classproperty
    def _constructor(cls) -> Type["Parameters"]:
        return cls

    def __str__(self) -> str:
        params_str: str = self.json(indent=4)
        out: str = f'{self.class_name} with params:\n{params_str}'
        return out

    class Config:
        ## Ref for Pydantic mutability: https://pydantic-docs.helpmanual.io/usage/models/#faux-immutability
        allow_mutation = False
        ## Ref for Extra.forbid: https://pydantic-docs.helpmanual.io/usage/model_config/#options
        extra = Extra.forbid
        ## Ref for Pydantic private attributes: https://pydantic-docs.helpmanual.io/usage/models/#private-model-attributes
        underscore_attrs_are_private = True
        ## Validates default values. Ref: https://pydantic-docs.helpmanual.io/usage/model_config/#options
        validate_all = True
        ## Validates typing by `isinstance` check. Ref: https://pydantic-docs.helpmanual.io/usage/model_config/#options
        arbitrary_types_allowed = True

    @staticmethod
    def _convert_params(Class: Type[BaseModel], d: Union[Type[BaseModel], Dict]):
        if type(d) == Class:
            return d
        if isinstance(d, BaseModel):
            return Class(**d.dict(exclude=None))
        if d is None:
            return Class()
        if isinstance(d, dict):
            return Class(**d)
        raise NotImplementedError(f'Cannot convert object of type {type(d)} to {Class.__class__}')

    def update_params(self, **new_params) -> Generic[ParametersSubclass]:
        ## Since Parameters class is immutable, we create a new one:
        overidden_params: Dict = {
            **self.dict(exclude=None),
            **new_params,
        }
        return self._constructor(**overidden_params)

    def copy(self, **kwargs) -> Generic[ParametersSubclass]:
        return super(Parameters, self).copy(**kwargs)

    def clone(self, **kwargs) -> Generic[ParametersSubclass]:
        return self.copy(**kwargs)


class UserEnteredParameters(Parameters):
    """
    Case-insensitive Parameters class.
    Use this for configs classes where you expect to read from user-entered input, which might have any case.
    IMPORTANT: the param names in the subclass must be in LOWERCASE ONLY.
    Ref: https://github.com/samuelcolvin/pydantic/issues/1147#issuecomment-571109376
    """

    @root_validator(pre=True)
    def convert_params_to_lowercase(cls, params: Dict):
        return {str(k).strip().lower(): v for k, v in params.items()}


class MutableParameters(Parameters):
    class Config(Parameters.Config):
        ## Ref on mutability: https://pydantic-docs.helpmanual.io/usage/models/#faux-immutability
        allow_mutation = True


class MutableUserEnteredParameters(UserEnteredParameters, MutableParameters):
    pass


class MappedParameters(Parameters, ABC):
    """
    Allows creation of a Parameters instance by mapping from a dict.
    From this dict, the 'name' key will be used to look up the cls._mapping dictionary, and retrieve the corresponding
    class. This class will be instantiated using the other values in the dict.
    """
    _mapping: ClassVar[Dict[Union[Tuple[str, ...], str], Any]]

    class Config(Parameters.Config):
        extra = Extra.allow

    name: constr(min_length=1)
    args: Tuple = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not isinstance(cls._mapping, dict) or len(cls._mapping) == 0:
            raise ValueError(f'Lookup must be a non-empty dict; found: {cls._mapping}')
        for key, val in list(cls._mapping.items()):
            if is_list_like(key):
                for k in key:
                    cls._mapping[str_normalize(k)] = val
            else:
                cls._mapping[str_normalize(key)] = val

    @root_validator(pre=True)
    def check_mapped_params(cls, params: Dict) -> Dict:
        if not str_normalize(params['name']) in cls._mapping:
            raise ValueError(
                f'''`name`="{params['name']}" was not found in the lookup. '''
                f'''Valid values for `name`: {set(cls._mapping.keys())}'''
            )
        return params

    def dict(self, *args, exclude: Optional[Any] = None, **kwargs) -> Dict:
        params: Dict = super(Parameters, self).dict(*args, exclude=exclude, **kwargs)
        if exclude is not None and 'name' in exclude:
            params.pop('name', None)
        else:
            params['name'] = self.name
        return params

    def __str__(self) -> str:
        params_str: str = self.json(indent=4)
        out: str = f'{self.class_name} with params:\n{params_str}'
        return out

    @classmethod
    def from_call_str(cls, call_str: str) -> Any:
        args, kwargs = call_str_to_params(call_str)
        return cls(args=args, **kwargs)

    def mapped_callable(self) -> Any:
        return self._mapping[str_normalize(self.name)]

    @property
    def kwargs(self) -> Dict:
        return self.dict(exclude={'name', 'args'} | set(self.dict_exclude))

    def to_call_str(self) -> str:
        args: List = list(self.args)
        kwargs: Dict = self.kwargs
        callable: Callable = self.mapped_callable()
        if is_function(callable) or isinstance(callable, type):
            callable_name: str = callable.__name__
        else:
            callable_name: str = str(callable)
        return params_to_call_str(
            callable_name=callable_name,
            args=args,
            kwargs=kwargs,
        )

    @classmethod
    @safe_validate_arguments
    def of(
            cls,
            name: Optional[Union[Parameters, Dict, str]],
            **params,
    ) -> Optional[Any]:
        if name is None:
            return None
        if isinstance(name, cls):
            return name
        if isinstance(name, dict):
            return cls(**name)
        if isinstance(name, str):
            if '(' in name or ')' in name:
                return cls.from_call_str(name)
            else:
                return cls(**{'name': name, **params})
        raise ValueError(f'Unsupported value for `name`: {name}')

    def initialize(self, **kwargs) -> Any:
        return self.mapped_callable()(
            *self.args,
            **self.kwargs,
            **kwargs
        )


## Test Utils:
def parameterized_name_func(test, _, param):
    from parameterized import parameterized
    ## Ref: https://kracekumar.com/post/618264170735009792/parameterize-python-tests/
    return f"{test.__name__}_{parameterized.to_safe_name('_'.join([str(x) for x in param.args]))}"


def parameterized_flatten(*args) -> List:
    return flatten2d(list(product(*args)))


class Timeout(MutableParameters):
    timeout: confloat(gt=0)  ## In seconds.
    last_used_time: float = time.time()

    @property
    def has_expired(self) -> bool:
        return self.last_used_time + self.timeout < time.time()

    def reset_timeout(self):
        self.last_used_time: float = time.time()


class Timeout1Min(Timeout):
    timeout: confloat(gt=0, le=60)


class Timeout15Min(Timeout):
    timeout: confloat(gt=0, le=60 * 15)


class Timeout1Hr(Timeout):
    timeout: confloat(gt=0, le=60 * 60)


class Timeout24Hr(Timeout):
    timeout: confloat(gt=0, le=60 * 60 * 24)


class TimeoutNever(Timeout):
    timeout: float = math.inf


class Timeout1Week(Timeout):
    timeout: confloat(gt=0, le=60 * 60 * 24 * 7)


@contextmanager
def pd_display(**kwargs):
    """
    Use pd.describe_option('display') to see all options.
    """
    try:
        from IPython.display import display
    except ImportError:
        display = print
    set_param_from_alias(params=kwargs, param='max_rows', alias=['num_rows', 'nrows', 'rows'], default=None)
    set_param_from_alias(params=kwargs, param='max_cols', alias=['num_cols', 'ncols', 'cols'], default=None)
    set_param_from_alias(params=kwargs, param='max_colwidth', alias=[
        'max_col_width',
        'max_columnwidth', 'max_column_width',
        'columnwidth', 'column_width',
        'colwidth', 'col_width',
    ], default=None)
    set_param_from_alias(params=kwargs, param='vertical_align', alias=['valign'], default='top')
    set_param_from_alias(params=kwargs, param='text_align', alias=['textalign'], default='left')
    set_param_from_alias(params=kwargs, param='ignore_css', alias=['css'], default=False)

    max_rows: Optional[int] = kwargs.get('max_rows')
    max_cols: Optional[int] = kwargs.get('max_cols')
    max_colwidth: Optional[int] = kwargs.get('max_colwidth')
    vertical_align: str = kwargs['vertical_align']
    text_align: str = kwargs['text_align']
    ignore_css: bool = kwargs['ignore_css']

    # print(kwargs)

    def disp(df: pd.DataFrame):
        css = [
            ## Align header to center
            {
                'selector': 'th',
                'props': [
                    ('vertical-align', 'center'),
                    ('text-align', 'center'),
                    ('padding', '10px'),
                ]
            },
            ## Align cell to top and left/center
            {
                'selector': 'td',
                'props': [
                    ('vertical-align', vertical_align),
                    ('text-align', text_align),
                    ('padding', '10px'),
                ]
            },

        ]
        if not ignore_css and isinstance(df, pd.DataFrame):
            df = df.style.set_table_styles(css)
        display(df)

    with pd.option_context(
            'display.max_rows', max_rows,
            'display.max_columns', max_cols,
            'max_colwidth', max_colwidth,
            'display.expand_frame_repr', False,
    ):
        yield disp


pd_extended_display = pd_display


def print_md(x):
    try:
        from IPython.display import display, Markdown
        x = Markdown(x)
    except ImportError:
        display = print
    display(x)


def print_math(x):
    try:
        from IPython.display import display, Math
        x = Math(x)
    except ImportError:
        display = print
    display(x)


def display_colors(colors: Union[Set[str], Tuple[str, ...], List[str], str]):
    """Displays colors from the given list with their names or codes."""
    # Start the HTML string for the colored divs
    html_str: str = "<div style='display: flex; flex-wrap: wrap; padding: 5px;'>"

    # Loop through the colors, adding each as a small colored div with a label
    for color in as_list(colors):
        html_str += f"""
        <div style='margin: 10px; text-align: center;'>
            <div style='background: {color}; width: 50px; height: 50px;'></div>
            <div style='margin-top: 5px;'>{color.lower()}</div>
        </div>
        """

    # Close the main div
    html_str += "</div>"

    # Display the HTML
    try:
        from IPython.display import display, HTML
    except ImportError:
        display = print
        HTML = lambda x: str(x)
    display(HTML(html_str))


def pd_partial_column_order(df: pd.DataFrame, columns: List) -> pd.DataFrame:
    columns: List = as_list(columns)
    df_columns: List = list(df.columns)
    final_columns: List = []
    for col in columns:
        if col not in df_columns:
            raise ValueError(f'Column "{col}" not found in current {pd.DataFrame} columns: {df.columns}')
        final_columns.append(col)
    for col in df_columns:  ## Add all the remaining columns
        if col not in final_columns:
            final_columns.append(col)
    assert set(final_columns) == set(df_columns)
    return df[final_columns]


ProgressBar = "ProgressBar"


class ProgressBar(MutableParameters):
    pbar: Optional[TqdmProgressBar] = None
    style: Literal['auto', 'notebook', 'std', 'ray'] = 'auto'
    unit: str = 'row'
    color: str = '#0288d1'  ## Bluish
    ncols: int = 100
    smoothing: float = 0.15
    total: Optional[int] = None
    disable: bool = False
    miniters: conint(ge=1) = 1
    _pending_updates: int = 0

    class Config(Parameters.Config):
        extra = Extra.allow

    @root_validator(pre=False)
    def _set_params(cls, params: Dict) -> Dict:
        set_param_from_alias(params, param='disable', alias=['disabled'])
        pbar: TqdmProgressBar = cls._create_pbar(**remove_keys(params, ['pbar', 'color']))
        pbar.color = params['color']
        pbar.refresh()
        params['pbar']: TqdmProgressBar = pbar
        return params

    @classmethod
    def _create_pbar(
            cls,
            style: Literal['auto', 'notebook', 'std', 'ray'],
            **kwargs,
    ) -> TqdmProgressBar:
        if style == 'auto':
            with optional_dependency('ipywidgets'):
                kwargs['ncols']: Optional[int] = None
            return AutoTqdmProgressBar(**kwargs)
        elif style == 'notebook':
            with optional_dependency('ipywidgets'):
                kwargs['ncols']: Optional[int] = None
            return NotebookTqdmProgressBar(**kwargs)
        elif style == 'ray':
            from ray.experimental import tqdm_ray
            kwargs = filter_keys(
                kwargs,
                keys=set(get_fn_spec(tqdm_ray.tqdm).args + get_fn_spec(tqdm_ray.tqdm).kwargs),
                how='include',
            )
            from ray.experimental import tqdm_ray
            return tqdm_ray.tqdm(**kwargs)
        else:
            return StdTqdmProgressBar(**kwargs)

    @classmethod
    def iter(cls, iterable: Union[Generator, Iterator, List, Tuple, Set, Dict, ItemsView], **kwargs):
        if is_list_or_set_like(iterable) or is_dict_like(iterable):
            kwargs['total'] = len(iterable)
        if is_dict_like(iterable):
            iterable: ItemsView = iterable.items()
        pbar: ProgressBar = ProgressBar.of(**kwargs)
        try:
            for item in iterable:
                yield item
                pbar.update(1)
            pbar.success()
        except Exception as e:
            pbar.failed()
            raise e

    @classmethod
    def of(
            cls,
            progress_bar: Optional[Union[ProgressBar, Dict, bool]] = True,
            *,
            prefer_kwargs: bool = True,
            **kwargs
    ) -> ProgressBar:
        if isinstance(progress_bar, ProgressBar):
            if prefer_kwargs:
                if 'total' in kwargs:
                    progress_bar.set_total(kwargs['total'])
                if 'initial' in kwargs:
                    progress_bar.set_n(kwargs['initial'])
                if 'desc' in kwargs:
                    progress_bar.set_description(kwargs['desc'])
                if 'unit' in kwargs:
                    progress_bar.set_description(kwargs['unit'])
            return progress_bar
        if progress_bar is not None and not isinstance(progress_bar, (bool, dict)):
            raise ValueError(f'You must pass `progress_bar` as either a bool, dict or None. None or False disables it.')
        if progress_bar is True:
            progress_bar: Optional[Dict] = dict()
        elif progress_bar is False:
            progress_bar: Optional[Dict] = None
        if progress_bar is not None and not isinstance(progress_bar, dict):
            raise ValueError(f'You must pass `progress_bar` as either a bool, dict or None. None or False disables it.')
        if progress_bar is None:
            progress_bar: Dict = dict(disable=True)
        elif isinstance(progress_bar, dict) and len(kwargs) > 0:
            if prefer_kwargs is True:
                progress_bar: Dict = {
                    **progress_bar,
                    **kwargs,
                }
            else:
                progress_bar: Dict = {
                    **kwargs,
                    **progress_bar,
                }
        assert isinstance(progress_bar, dict)
        return ProgressBar(**progress_bar)

    def update(self, n: int = 1) -> Optional[bool]:
        self._pending_updates += n
        if abs(self._pending_updates) >= self.miniters:
            out = self.pbar.update(n=self._pending_updates)
            self.refresh()
            self._pending_updates = 0
            return out
        else:
            return None

    def set_n(self, new_n: int):
        self.pbar.update(n=new_n - self.pbar.n)
        self._pending_updates = 0  ## Clear all updates after setting new value
        self.refresh()

    def set_total(self, new_total: int):
        self.pbar.total = new_total
        self._pending_updates = 0  ## Clear all updates after setting new value
        self.refresh()

    def set_description(self, desc: Optional[str] = None, refresh: Optional[bool] = True):
        out = self.pbar.set_description(desc=desc, refresh=refresh)
        self.refresh()
        return out

    def set_unit(self, new_unit: str):
        self.pbar.unit = new_unit
        self.refresh()

    def success(self, desc: Optional[str] = None, close: bool = True, append_desc: bool = True):
        self._complete_with_status(
            color='#43a047',  ## Dark Green
            desc=desc,
            close=close,
            append_desc=append_desc,
        )

    def stopped(self, desc: Optional[str] = None, close: bool = True, append_desc: bool = True):
        self._complete_with_status(
            color='#b0bec5',  ## Dark Grey
            desc=desc,
            close=close,
            append_desc=append_desc,
        )

    def failed(self, desc: Optional[str] = None, close: bool = True, append_desc: bool = True):
        self._complete_with_status(
            color='#e64a19',  ## Dark Red
            desc=desc,
            close=close,
            append_desc=append_desc,
        )

    def _complete_with_status(
            self,
            color: str,
            desc: Optional[str],
            close: bool,
            append_desc: bool,
    ):
        if not self.pbar.disable:
            self.pbar.update(n=self._pending_updates)
            self._pending_updates = 0
            self.color = color
            self.pbar.colour = color
            if desc is not None:
                if append_desc:
                    desc: str = f'[{desc}] {self.pbar.desc}'
                self.pbar.desc = desc
            self.pbar.refresh()
            if close:
                self.close()

    def refresh(self):
        self.pbar.colour = self.color
        self.pbar.refresh()

    def close(self):
        self.pbar.refresh()
        self.pbar.close()
        self.pbar.refresh()

    def __del__(self):
        self.pbar.close()


def create_progress_bar(
        *,
        style: Optional[Literal['auto', 'notebook', 'std']] = 'auto',
        unit: str = 'row',
        ncols: int = 100,
        smoothing: float = 0.1,
        **kwargs
) -> TqdmProgressBar:
    try:
        if style == 'auto':
            with optional_dependency('ipywidgets'):
                ncols: Optional[int] = None
            return AutoTqdmProgressBar(
                ncols=ncols,
                unit=unit,
                smoothing=smoothing,
                **kwargs
            )
        elif style == 'notebook':
            with optional_dependency('ipywidgets'):
                ncols: Optional[int] = None
            return NotebookTqdmProgressBar(
                ncols=ncols,
                unit=unit,
                smoothing=smoothing,
                **kwargs
            )
        elif style == 'ray':
            from ray.experimental import tqdm_ray
            kwargs = filter_keys(
                kwargs,
                keys=set(get_fn_spec(tqdm_ray.tqdm).args + get_fn_spec(tqdm_ray.tqdm).kwargs),
                how='include',
            )
            from ray.experimental import tqdm_ray
            return tqdm_ray.tqdm(**kwargs)
        else:
            return StdTqdmProgressBar(
                ncols=ncols,
                unit=unit,
                smoothing=smoothing,
                **kwargs
            )
    except Exception as e:
        kwargs['style'] = style
        kwargs['unit'] = unit
        kwargs['ncols'] = ncols
        kwargs['smoothing'] = smoothing
        raise ValueError(
            f'Error: could not create progress bar using settings: {kwargs}. Stack trace:\n{format_exception_msg(e)}'
        )


@contextmanager
def ignore_warnings():
    pd_chained_assignment: Optional[str] = pd.options.mode.chained_assignment  # default='warn'
    with warnings.catch_warnings():  ## Ref: https://stackoverflow.com/a/14463362
        warnings.simplefilter("ignore")
        ## Stops Pandas SettingWithCopyWarning in output. Ref: https://stackoverflow.com/a/20627316
        pd.options.mode.chained_assignment = None
        yield
    pd.options.mode.chained_assignment = pd_chained_assignment


@contextmanager
def ignore_stdout():
    devnull = open(os.devnull, "w")
    stdout = sys.stdout
    sys.stdout = devnull
    try:
        yield
    finally:
        sys.stdout = stdout


@contextmanager
def ignore_stderr():
    devnull = open(os.devnull, "w")
    stderr = sys.stderr
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stderr = stderr


@contextmanager
def ignore_stdout_and_stderr():
    with ignore_stdout():
        with ignore_stderr():
            yield


@contextmanager
def ignore_warnings_and_stdout():
    with ignore_warnings():
        with ignore_stdout():
            with ignore_stderr():
                yield


@contextmanager
def ignore_logging(disable_upto: int = logging.CRITICAL):
    prev_disable_level: int = logging.root.manager.disable
    logging.disable(disable_upto + 1)
    try:
        yield
    finally:
        logging.disable(prev_disable_level)


@contextmanager
def ignore_all_output():
    with ignore_stdout():
        with ignore_warnings():
            with ignore_stderr():
                with ignore_logging():
                    yield


@contextmanager
def ignore_nothing():
    yield


# from pydantic import Field, AliasChoices
# def Alias(*, default: Optional[Any] = None, alias: Union[Tuple[str, ...], List[str], Set[str], str]):
#     alias: AliasChoices = AliasChoices(*as_tuple(alias))
#     return Field(default=default, validation_alias=alias, serialization_alias=alias)

_Comparable = Union[int, float, str]


def is_sorted(l: Union[List[Any], Tuple[Any, ...]], *, reverse: bool = False) -> bool:
    assert isinstance(l, (list, tuple))
    length = len(l)
    assert length > 0
    if length == 1:
        return True
    if reverse:
        l: List[Any] = list(l)[::-1]
    for x, x_next in zip(l[0:length - 1], l[1:length]):
        if x > x_next:
            return False
    return True


def binary_search(
        l: Union[List[_Comparable], Tuple[_Comparable, ...]],
        target: _Comparable,
        *,
        return_tuple: bool = False,
) -> Union[
    Tuple[Optional[_Comparable], Optional[_Comparable]],
    _Comparable
]:
    if not is_sorted(l):
        l: List[_Comparable] = sorted(l)
    low: int = 0
    high: int = len(l) - 1
    while low <= high:
        mid = (low + high) // 2
        if l[mid] == target:
            if return_tuple:
                return l[mid], l[mid]
            return l[mid]
        elif l[mid] < target:
            low: int = mid + 1
        else:
            high: int = mid - 1

    ## When the target is not found, set lower and upper bounds
    lower: _Comparable = l[high] if high >= 0 else None
    upper: _Comparable = l[low] if low < len(l) else None

    return lower, upper


@safe_validate_arguments
def plotsum(
        plots_list: Union[List[Tuple[str, Any]], List[Any]],
        *,
        order: Optional[List[str]] = None,
        how: Literal['overlay', 'grid'] = 'grid',
        legend: Literal['first', 'last', 'none'] = 'none',
        update_layout: Optional[Dict] = None,
        backend: Literal['plotly'] = 'plotly',
):
    import holoviews as hv
    if order is not None:
        assert len(plots_list) > 0
        assert len(order) == len(plots_list)
        assert len(set(p[0] for p in plots_list)) == len(order)
        ordered_plots_list: List[Any] = []
        for order_item in order:
            plot_str: Optional = None
            for plot_str, plot in plots_list:
                if plot_str == order_item:
                    break
                plot_str = None
            if plot_str is None:
                raise ValueError(f'No plot found with name: "{order_item}"')
            ordered_plots_list.append(plot)
        plots_list = ordered_plots_list

    plots = None
    for plot in plots_list:
        if isinstance(plot, tuple):
            assert len(plot) == 2
            plot = plot[1]
        if plots is None:
            plots = plot
        else:
            if how == 'grid':
                plots += plot
            elif how == 'overlay':
                plots *= plot
            else:
                raise not_impl('how', how)
    return plots


def to_pct(counts: pd.Series):  ## Converts value counts to percentages
    _sum = counts.sum()
    return pd.DataFrame({
        'value': counts.index.tolist(),
        'count': counts.tolist(),
        'pct': counts.apply(lambda x: 100 * x / _sum).tolist(),
        'count_str': counts.apply(lambda x: f'{x} of {_sum}').tolist(),
    })
