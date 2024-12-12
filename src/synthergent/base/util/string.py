from typing import *
from ast import literal_eval
import math, re, json, sys, inspect, io, pprint, random, types, functools
import numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pydantic import conint, constr, confloat, validate_arguments
from collections import defaultdict

StructuredBlob = Union[List, Dict, List[Dict]]  ## used for type hints.
KERNEL_START_DT: datetime = datetime.now()


class _JsonEncoder(json.JSONEncoder):
    def default(self, obj):
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
        return super(_JsonEncoder, self).default(obj)


class StringUtil:
    def __init__(self):
        raise TypeError(f'Cannot instantiate utility class "{str(self.__class__)}"')

    EMPTY: str = ''
    SPACE: str = ' '
    DOUBLE_SPACE: str = SPACE * 2
    FOUR_SPACE: str = SPACE * 4
    TAB: str = '\t'
    NEWLINE: str = '\n'
    WINDOWS_NEWLINE: str = '\r'
    BACKSLASH: str = '\\'
    SLASH: str = '/'
    PIPE: str = '|'
    SINGLE_QUOTE: str = "'"
    DOUBLE_QUOTE: str = '"'
    COMMA: str = ','
    COMMA_SPACE: str = ', '
    COMMA_NEWLINE: str = ',\n'
    HYPHEN: str = '-'
    DOUBLE_HYPHEN: str = '--'
    DOT: str = '.'
    ASTERISK: str = '*'
    DOUBLE_ASTERISK: str = '**'
    QUESTION_MARK: str = '?'
    CARET: str = '^'
    DOLLAR: str = '$'
    UNDERSCORE: str = '_'
    COLON: str = ':'
    SEMICOLON: str = ';'
    EQUALS: str = '='
    LEFT_PAREN: str = '('
    RIGHT_PAREN: str = ')'
    BACKTICK: str = '`'
    TILDE: str = '~'

    MATCH_ALL_REGEX_SINGLE_LINE: str = CARET + DOT + ASTERISK + DOLLAR
    MATCH_ALL_REGEX_MULTI_LINE: str = DOT + ASTERISK

    S3_PREFIX: str = 's3://'
    FILE_PREFIX: str = 'file://'
    HTTP_PREFIX: str = 'http://'
    HTTPS_PREFIX: str = 'https://'
    PORT_REGEX: str = ':(\d+)'
    DOCKER_REGEX: str = '\d+\.dkr\.ecr\..*.amazonaws\.com/.*'

    DEFAULT_CHUNK_NAME_PREFIX: str = 'part'

    FILES_TO_IGNORE: str = ['_SUCCESS', '.DS_Store']

    UTF_8: str = 'utf-8'

    FILE_SIZE_UNITS: Tuple[str, ...] = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    ## FILE_SIZE_REGEX taken from: https://rgxdb.com/r/4IG91ZFE
    ## Matches: "2", "2.5", "2.5b", "2.5B", "2.5k", "2.5K", "2.5kb", "2.5Kb", "2.5KB", "2.5kib", "2.5KiB", "2.5kiB"
    ## Does not match: "2.", "2ki", "2ib", "2.5KIB"
    FILE_SIZE_REGEX = r'^(\d*\.?\d+)((?=[KMGTkgmt])([KMGTkgmt])(?:i?[Bb])?|[Bb]?)$'

    ALPHABET: Tuple[str, ...] = tuple('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    ALPHABET_CAPS: Tuple[str, ...] = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    ALPHABET_CAPS_NO_DIGITS: Tuple[str, ...] = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    ## Taken from: https://github.com/django/django/blob/master/django/utils/baseconv.py#L101
    BASE2_ALPHABET: str = '01'
    BASE16_ALPHABET: str = '0123456789ABCDEF'
    BASE56_ALPHABET: str = '23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz'
    BASE36_ALPHABET: str = '0123456789abcdefghijklmnopqrstuvwxyz'
    BASE62_ALPHABET: str = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    BASE64_ALPHABET: str = BASE62_ALPHABET + '-_'

    class BaseConverter:
        decimal_digits: str = '0123456789'

        def __init__(self, digits, sign='-'):
            self.sign = sign
            self.digits = digits
            if sign in self.digits:
                raise ValueError('Sign character found in converter base digits.')

        def __repr__(self):
            return "<%s: base%s (%s)>" % (self.__class__.__name__, len(self.digits), self.digits)

        def encode(self, i):
            neg, value = self.convert(i, self.decimal_digits, self.digits, '-')
            if neg:
                return self.sign + value
            return value

        def decode(self, s):
            neg, value = self.convert(s, self.digits, self.decimal_digits, self.sign)
            if neg:
                value = '-' + value
            return int(value)

        def convert(self, number, from_digits, to_digits, sign):
            if str(number)[0] == sign:
                number = str(number)[1:]
                neg = 1
            else:
                neg = 0

            # make an integer out of the number
            x = 0
            for digit in str(number):
                x = x * len(from_digits) + from_digits.index(digit)

            # create the result in base 'len(to_digits)'
            if x == 0:
                res = to_digits[0]
            else:
                res = ''
                while x > 0:
                    digit = x % len(to_digits)
                    res = to_digits[digit] + res
                    x = int(x // len(to_digits))
            return neg, res

    BASE_CONVERTER_MAP: Dict[int, BaseConverter] = {
        2: BaseConverter(BASE2_ALPHABET),
        16: BaseConverter(BASE16_ALPHABET),
        36: BaseConverter(BASE36_ALPHABET),
        56: BaseConverter(BASE56_ALPHABET),
        62: BaseConverter(BASE62_ALPHABET),
        64: BaseConverter(BASE64_ALPHABET, sign='$'),
    }

    @classmethod
    def assert_not_empty_and_strip(cls, string: str, error_message: str = '') -> str:
        cls.assert_not_empty(string, error_message)
        return string.strip()

    @classmethod
    def strip_if_not_empty(cls, string: str) -> str:
        if cls.is_not_empty(string):
            return string.strip()
        return string

    @classmethod
    def is_not_empty(cls, string: str) -> bool:
        return isinstance(string, str) and len(string.strip()) > 0

    @classmethod
    def is_not_empty_bytes(cls, string: bytes) -> bool:
        return isinstance(string, bytes) and len(string.strip()) > 0

    @classmethod
    def is_not_empty_str_or_bytes(cls, string: Union[str, bytes]) -> bool:
        return cls.is_not_empty(string) or cls.is_not_empty_bytes(string)

    @classmethod
    def is_empty(cls, string: Any) -> bool:
        return not cls.is_not_empty(string)

    @classmethod
    def is_empty_bytes(cls, string: Any) -> bool:
        return not cls.is_not_empty_bytes(string)

    @classmethod
    def is_empty_str_or_bytes(cls, string: Any) -> bool:
        return not cls.is_not_empty_str_or_bytes(string)

    @classmethod
    def assert_not_empty(cls, string: Any, error_message: str = ''):
        assert cls.is_not_empty(string), error_message

    @classmethod
    def assert_not_empty_bytes(cls, string: Any, error_message: str = ''):
        assert cls.is_not_empty_str_or_bytes(string), error_message

    @classmethod
    def assert_not_empty_str_or_bytes(cls, string: Any, error_message: str = ''):
        assert cls.is_not_empty_str_or_bytes(string), error_message

    @classmethod
    def is_int(cls, string: Any) -> bool:
        """
        Checks if an input string is an integer.
        :param string: input string
        :raises: error when input is not a string
        :return: True for '123', '-123' but False for '123.0', '1.23', '-1.23' and '1e2'
        """
        try:
            int(string)
            return True
        except Exception as e:
            return False

    @classmethod
    def is_float(cls, string: Any) -> bool:
        """
        Checks if an input string is a floating-point value.
        :param string: input string
        :raises: error when input is not a string
        :return: True for '123', '1.23', '123.0', '-123', '-123.0', '1e2', '1.23e-5', 'NAN' & 'nan'; but False for 'abc'
        """
        try:
            float(string)  ## Will return True for NaNs as well.
            return True
        except Exception as e:
            return False

    @classmethod
    def is_prefix(cls, prefix: str, strings: Union[List[str], Set[str]]) -> bool:
        cls.assert_not_empty(prefix)
        if isinstance(strings, str):
            strings = [strings]
        return True in {string.startswith(prefix) for string in strings}

    @classmethod
    def remove_prefix(cls, string: str, prefix: str) -> str:
        cls.assert_not_empty(prefix)
        if string.startswith(prefix):
            string = string[len(prefix):]
        return string

    @classmethod
    def remove_suffix(cls, string: str, suffix: str) -> str:
        cls.assert_not_empty(suffix)
        if string.endswith(suffix):
            string = string[:-len(suffix)]
        return string

    @classmethod
    def join_human(
            cls,
            l: Union[List, Tuple, Set],
            sep: str = ',',
            final_join: str = 'and',
            oxford_comma: bool = False,
    ) -> str:
        l: List = list(l)
        if len(l) == 1:
            return str(l[0])
        out: str = ''
        for x in l[:-1]:
            out += ' ' + str(x) + sep
        if not oxford_comma:
            out: str = cls.remove_suffix(out, sep)
        x = l[-1]
        out += f' {final_join} ' + str(x)
        return out.strip()

    @classmethod
    def convert_str_to_type(cls, val: str, expected_type: Type) -> Any:
        assert isinstance(expected_type, type)
        if isinstance(val, expected_type):
            return val
        if expected_type == str:
            return str(val)
        if expected_type == bool and isinstance(val, str):
            val = val.lower().strip().capitalize()  ## literal_eval does not parse "false", only "False".
        out = literal_eval(StringUtil.assert_not_empty_and_strip(str(val)))
        if expected_type == float and isinstance(out, int):
            out = float(out)
        if expected_type == int and isinstance(out, float) and int(out) == out:
            out = int(out)
        if expected_type == tuple and isinstance(out, list):
            out = tuple(out)
        if expected_type == list and isinstance(out, tuple):
            out = list(out)
        if expected_type == set and isinstance(out, (list, tuple)):
            out = set(out)
        if expected_type == bool and out in [0, 1]:
            out = bool(out)
        if type(out) != expected_type:
            raise ValueError(f'Input value {val} cannot be converted to {str(expected_type)}')
        return out

    @classmethod
    def readable_bytes(cls, size_in_bytes: int, decimals: int = 3) -> str:
        sizes: Dict[str, float] = cls.convert_size_from_bytes(size_in_bytes, unit=None, decimals=decimals)
        sorted_sizes: List[Tuple[str, float]] = [
            (k, v) for k, v in sorted(sizes.items(), key=lambda item: item[1])
        ]
        size_unit, size_val = None, None
        for size_unit, size_val in sorted_sizes:
            if size_val >= 1:
                break
        return f'{size_val} {size_unit}'

    @classmethod
    def convert_size_from_bytes(
            cls,
            size_in_bytes: int,
            unit: Optional[str] = None,
            decimals: int = 3,
    ) -> Union[Dict, float]:
        size_in_bytes = float(size_in_bytes)
        cur_size = size_in_bytes
        sizes = {}
        if size_in_bytes == 0:
            for size_name in cls.FILE_SIZE_UNITS:
                sizes[size_name] = 0.0
        else:
            for size_name in cls.FILE_SIZE_UNITS:
                val: float = round(cur_size, decimals)
                i = 1
                while val == 0:
                    val = round(cur_size, decimals + i)
                    i += 1
                sizes[size_name] = val
                i = int(math.floor(math.log(cur_size, 1024)))
                cur_size = cur_size / 1024
        if unit is not None:
            assert isinstance(unit, str)
            unit = unit.upper()
            assert unit in cls.FILE_SIZE_UNITS
            return sizes[unit]
        return sizes

    @classmethod
    def convert_size_to_bytes(cls, size_in_human_readable: str) -> int:
        size_in_human_readable: str = cls.assert_not_empty_and_strip(size_in_human_readable).upper()
        size_selection_regex = f"""(\d+(?:\.\d+)?) *({cls.PIPE.join(cls.FILE_SIZE_UNITS)})"""  ## This uses a non-capturing group: https://stackoverflow.com/a/3512530/4900327
        matches = re.findall(size_selection_regex, size_in_human_readable)
        if len(matches) != 1 or len(matches[0]) != 2:
            raise ValueError(f'Cannot convert value "{size_in_human_readable}" to bytes.')
        val, unit = matches[0]
        val = float(val)
        for file_size_unit in cls.FILE_SIZE_UNITS:
            if unit == file_size_unit:
                return int(round(val))
            val = val * 1024
        raise ValueError(f'Cannot convert value "{size_in_human_readable}" to bytes.')

    @classmethod
    def readable_seconds(
            cls,
            time_in_seconds: Union[float, timedelta],
            *,
            decimals: int = 2,
            short: bool = False,
    ) -> str:
        if isinstance(time_in_seconds, timedelta):
            time_in_seconds: float = time_in_seconds.total_seconds()
        times: Dict[str, float] = cls.convert_time_from_seconds(
            time_in_seconds,
            unit=None,
            decimals=decimals,
            short=short,
        )
        sorted_times: List[Tuple[str, float]] = [
            (k, v) for k, v in sorted(times.items(), key=lambda item: item[1])
        ]
        time_unit, time_val = None, None
        for time_unit, time_val in sorted_times:
            if time_val >= 1:
                break
        if decimals <= 0:
            time_val = int(time_val)
        if short:
            return f'{time_val}{time_unit}'
        return f'{time_val} {time_unit}'

    @classmethod
    def convert_time_from_seconds(
            cls,
            time_in_seconds: float,
            unit: Optional[str] = None,
            decimals: int = 3,
            short: bool = False,
    ) -> Union[Dict, float]:
        TIME_UNITS = {
            "nanoseconds": 1e-9,
            "microseconds": 1e-6,
            "milliseconds": 1e-3,
            "seconds": 1.0,
            "mins": 60,
            "hours": 60 * 60,
            "days": 24 * 60 * 60,
        }
        if short:
            TIME_UNITS = {
                "ns": 1e-9,
                "us": 1e-6,
                "ms": 1e-3,
                "s": 1.0,
                "min": 60,
                "hr": 60 * 60,
                "d": 24 * 60 * 60,
            }
        time_in_seconds = float(time_in_seconds)
        times: Dict[str, float] = {
            time_unit: round(time_in_seconds / TIME_UNITS[time_unit], decimals)
            for time_unit in TIME_UNITS
        }
        if unit is not None:
            assert isinstance(unit, str)
            unit = unit.lower()
            assert unit in TIME_UNITS
            return times[unit]
        return times

    @classmethod
    def readable_number(
            cls,
            n: Union[float, int],
            decimals: int = 3,
            short: bool = True,
            scientific: bool = False,
    ) -> str:
        if n == 0:
            return '0'
        assert abs(n) > 0
        if 0 < abs(n) < 1:
            scientific: bool = True
        if scientific:
            n_unit: str = ''
            n_val: str = f'{n:.{decimals}e}'
        else:
            numbers: Dict[str, float] = cls.convert_number(
                abs(n),
                unit=None,
                decimals=decimals,
                short=short,
            )
            sorted_numbers: List[Tuple[str, float]] = [
                (k, v) for k, v in sorted(numbers.items(), key=lambda item: item[1])
            ]
            n_unit, n_val = None, None
            for n_unit, n_val in sorted_numbers:
                if n_val >= 1:
                    break
            if decimals <= 0:
                n_val: int = int(n_val)
            if n_val == int(n_val):
                n_val: int = int(n_val)
        if n < 0:
            n_val: str = f'-{n_val}'
        if short:
            return f'{n_val}{n_unit}'.strip()
        return f'{n_val} {n_unit}'.strip()

    @classmethod
    def convert_number(
            cls,
            n: float,
            unit: Optional[str] = None,
            decimals: int = 3,
            short: bool = False,
    ) -> Union[Dict, float]:
        assert n >= 0
        N_UNITS = {
            "": 1e0,
            "thousand": 1e3,
            "million": 1e6,
            "billion": 1e9,
            "trillion": 1e12,
            "quadrillion": 1e15,
            "quintillion": 1e18,
        }
        if short:
            N_UNITS = {
                "": 1e0,
                "K": 1e3,
                "M": 1e6,
                "B": 1e9,
                "T": 1e12,
                "Qa": 1e15,
                "Qi": 1e18,
            }
        n: float = float(n)
        numbers: Dict[str, float] = {
            n_unit: round(n / N_UNITS[n_unit], decimals)
            for n_unit in N_UNITS
        }
        if unit is not None:
            assert isinstance(unit, str)
            unit = unit.lower()
            assert unit in N_UNITS
            return numbers[unit]
        return numbers

    @classmethod
    def jsonify(
            cls,
            blob: StructuredBlob,
            *,
            minify: bool = False,
    ) -> str:
        if minify:
            return json.dumps(blob, indent=None, separators=(cls.COMMA, cls.COLON), cls=_JsonEncoder)
        else:
            return json.dumps(blob, cls=_JsonEncoder, indent=4)

    @classmethod
    def get_num_zeros_to_pad(cls, max_i: int) -> int:
        assert isinstance(max_i, int) and max_i >= 1
        num_zeros = math.ceil(math.log10(max_i))  ## Ref: https://stackoverflow.com/a/51837162/4900327
        if max_i == 10 ** num_zeros:  ## If it is a power of 10
            num_zeros += 1
        return num_zeros

    @classmethod
    def pad_zeros(cls, i: int, max_i: int = int(1e12)) -> str:
        assert isinstance(i, int)
        assert i >= 0
        assert isinstance(max_i, int)
        assert max_i >= i, f'Expected max_i to be >= current i; found max_i={max_i}, i={i}'
        num_zeros: int = cls.get_num_zeros_to_pad(max_i)
        return f'{i:0{num_zeros}}'

    @classmethod
    def stringify(
            cls,
            d: Union[Dict, List, Tuple, Set, Any],
            *,
            sep: str = ',',
            key_val_sep: str = '=',
            literal: bool = False,
            nested_literal: bool = True,
    ) -> str:
        if isinstance(d, (dict, defaultdict)):
            if nested_literal:
                out: str = sep.join([
                    f'{k}'
                    f'{key_val_sep}'
                    f'{cls.stringify(v, sep=sep, key_val_sep=key_val_sep, literal=True, nested_literal=True)}'
                    for k, v in sorted(list(d.items()), key=lambda x: x[0])
                ])
            else:
                out: str = sep.join([
                    f'{k}'
                    f'{key_val_sep}'
                    f'{cls.stringify(v, sep=sep, key_val_sep=key_val_sep, literal=False, nested_literal=False)}'
                    for k, v in sorted(list(d.items()), key=lambda x: x[0])
                ])
        elif isinstance(d, (list, tuple, set, frozenset, np.ndarray, pd.Series)):
            try:
                s = sorted(list(d))
            except TypeError:  ## Sorting fails
                s = list(d)
            out: str = sep.join([
                f'{cls.stringify(x, sep=sep, key_val_sep=key_val_sep, literal=nested_literal, nested_literal=nested_literal)}'
                for x in s
            ])
        else:
            out: str = repr(d)
        if literal:
            if isinstance(d, list):
                out: str = f'[{out}]'
            elif isinstance(d, np.ndarray):
                out: str = f'np.array([{out}])'
            elif isinstance(d, pd.Series):
                out: str = f'pd.Series([{out}])'
            elif isinstance(d, tuple):
                if len(d) == 1:
                    out: str = f'({out},)'
                else:
                    out: str = f'({out})'
            elif isinstance(d, (set, frozenset)):
                out: str = f'({out})'
            elif isinstance(d, (dict, defaultdict)):
                out: str = f'dict({out})'
        return out

    @classmethod
    def destringify(cls, s: str) -> Any:
        if isinstance(s, str):
            try:
                val = literal_eval(s)
            except ValueError:
                val = s
        else:
            val = s
        if isinstance(val, float):
            if val.is_integer():
                return int(val)
            return val
        return val

    @classmethod
    @validate_arguments
    def random(
            cls,
            shape: Tuple = (1,),
            length: Union[conint(ge=1), Tuple[conint(ge=1), conint(ge=1)]] = 6,
            spaces_prob: Optional[confloat(ge=0.0, le=1.0)] = None,
            alphabet: Tuple = ALPHABET,
            seed: Optional[int] = None,
            unique: bool = False,
    ) -> Union[str, np.ndarray]:
        if isinstance(length, int):
            min_num_chars: int = length
            max_num_chars: int = length
        else:
            min_num_chars, max_num_chars = length
        assert min_num_chars <= max_num_chars, \
            f'Must have min_num_chars ({min_num_chars}) <= max_num_chars ({max_num_chars})'
        if spaces_prob is not None:
            num_spaces_to_add: int = int(round(len(alphabet) * spaces_prob / (1 - spaces_prob), 0))
            alphabet = alphabet + num_spaces_to_add * (cls.SPACE,)

        ## Ref: https://stackoverflow.com/a/25965461/4900327
        np_random = np.random.RandomState(seed=seed)
        random_alphabet_lists = np_random.choice(alphabet, shape + (max_num_chars,))
        random_strings: np.ndarray = np.apply_along_axis(
            arr=random_alphabet_lists,
            func1d=lambda random_alphabet_list:
            ''.join(random_alphabet_list)[:np_random.randint(min_num_chars, max_num_chars + 1)],
            axis=len(shape),
        )
        if shape == (1,):
            return random_strings[0]
        if unique:
            random_strings_flatten1d: np.ndarray = random_strings.ravel()
            if len(set(random_strings_flatten1d)) != len(random_strings_flatten1d):
                ## Call it recursively:
                random_strings: np.ndarray = cls.random(
                    shape=shape,
                    length=length,
                    spaces_prob=spaces_prob,
                    alphabet=alphabet,
                    seed=seed,
                    unique=unique,
                )
        return random_strings

    @classmethod
    def random_name(
            cls,
            count: int = 1,
            *,
            sep: str = HYPHEN,
            order: Tuple[str, ...] = ('adjective', 'verb', 'noun'),
            seed: Optional[int] = None,
    ) -> Union[List[str], str]:
        cartesian_product_parts: List[List[str]] = []
        assert len(order) > 0
        for order_part in order:
            if order_part == 'verb':
                cartesian_product_parts.append(cls.RANDOM_VERBS)
            elif order_part == 'adjective':
                cartesian_product_parts.append(cls.RANDOM_ADJECTIVES)
            elif order_part == 'noun':
                cartesian_product_parts.append(cls.RANDOM_NOUNS)
            else:
                raise NotImplementedError(f'Unrecognized part of the order sequence: "{order_part}"')

        out: List[str] = [
            sep.join(parts)
            for parts in cls.__random_cartesian_product(*cartesian_product_parts, seed=seed, n=count)
        ]
        if count == 1:
            return out[0]
        return out

    @staticmethod
    def __random_cartesian_product(*lists, seed: Optional[int] = None, n: int):
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

    @classmethod
    def parse_datetime(cls, dt: Union[str, int, float, datetime]) -> datetime:
        if isinstance(dt, datetime):
            return dt
        elif type(dt) in [int, float]:
            return datetime.fromtimestamp(dt)
        elif isinstance(dt, str):
            return datetime.fromisoformat(dt)
        raise NotImplementedError(f'Cannot parse datetime from value {dt} with type {type(dt)}')

    @classmethod
    def now(cls, **kwargs) -> str:
        dt: datetime = datetime.now()
        return cls.readable_datetime(dt, **kwargs)

    @classmethod
    def kernel_start_time(cls, **kwargs) -> str:
        return cls.readable_datetime(KERNEL_START_DT, **kwargs)

    @classmethod
    def readable_datetime(
            cls,
            dt: datetime,
            *,
            human: bool = False,
            microsec: bool = True,
            tz: bool = True,
            **kwargs,
    ) -> str:
        dt: datetime = dt.replace(tzinfo=dt.astimezone().tzinfo)
        if human:
            format_str: str = '%d%b%Y-%H:%M:%S'
            microsec: bool = False
        else:
            format_str: str = '%Y-%m-%dT%H:%M:%S'
        if microsec:
            format_str += '.%f'
        split_tz_colon: bool = False
        if tz and dt.tzinfo is not None:
            if human:
                format_str += '+%Z'
            else:
                format_str += '%z'
                split_tz_colon: bool = True
        out: str = dt.strftime(format_str).strip()
        if split_tz_colon:  ## Makes the output exactly like dt.isoformat()
            out: str = out[:-2] + ':' + out[-2:]
        return out

    @classmethod
    def convert_integer_to_base_n_str(cls, integer: int, base: int) -> str:
        assert isinstance(integer, int)
        assert isinstance(base, int) and base in cls.BASE_CONVERTER_MAP, \
            f'Param `base` must be an integer in {list(cls.BASE_CONVERTER_MAP.keys())}; found: {base}'
        return cls.BASE_CONVERTER_MAP[base].encode(integer)

    @classmethod
    def hash(cls, val: Union[str, int, float, List, Dict], max_len: int = 256, base: int = 62) -> str:
        """
        Constructs a hash of a JSON object or value.
        :param val: any valid JSON value (including str, int, float, list, and dict).
        :param max_len: the maximum length of the output hash (will truncate upto this length).
        :param base: the base of the output hash.
            Defaults to base56, which encodes the output in a ASCII-chars
        :return: SHA256 hash.
        """

        def hash_rec(val, base):
            if isinstance(val, list):
                return hash_rec(','.join([hash_rec(x, base=base) for x in val]), base=base)
            elif isinstance(val, dict):
                return hash_rec(
                    [
                        f'{hash_rec(k, base=base)}:{hash_rec(v, base=base)}'
                        for k, v in sorted(val.items(), key=lambda kv: kv[0])
                    ],
                    base=base
                )
            return cls.convert_integer_to_base_n_str(int(sha256(str(val).encode('utf8')).hexdigest(), 16), base=base)

        return hash_rec(val, base)[:max_len]

    @classmethod
    def fuzzy_match(
            cls,
            string: str,
            strings_to_match: Union[str, List[str]],
            replacements: Tuple = (SPACE, HYPHEN, SLASH),
            repl_char: str = UNDERSCORE,
    ) -> Optional[str]:
        """Gets the closest fuzzy-matched string from the list, or else returns None."""
        if not isinstance(strings_to_match, list) and not isinstance(strings_to_match, tuple):
            assert isinstance(strings_to_match, str), f'Input must be of a string or list of strings; found type ' \
                                                      f'{type(strings_to_match)} with value: {strings_to_match}'
            strings_to_match: List[str] = [strings_to_match]
        string: str = str(string).lower()
        strings_to_match_repl: List[str] = [str(s).lower() for s in strings_to_match]
        for repl in replacements:
            string: str = string.replace(repl, repl_char)
            strings_to_match_repl: List[str] = [s.replace(repl, repl_char) for s in strings_to_match_repl]
        for i, s in enumerate(strings_to_match_repl):
            if string == s:
                return strings_to_match[i]
        return None

    @classmethod
    def is_fuzzy_match(cls, string: str, strings_to_match: List[str]) -> bool:
        """Returns whether or not there is a fuzzy-matched string in the list"""
        return cls.fuzzy_match(string, strings_to_match) is not None

    @classmethod
    def header(cls, text: str, width: int = 65, border: str = '=') -> str:
        out = ''
        out += border * width + cls.NEWLINE
        out += ('{:^' + str(width) + 's}').format(text) + cls.NEWLINE
        out += border * width + cls.NEWLINE
        return out

    @classmethod
    def is_stream(cls, obj) -> bool:
        return isinstance(obj, io.IOBase) and hasattr(obj, 'read')

    @classmethod
    def pretty(cls, d: Any, max_width: int = 100) -> str:
        if isinstance(d, dict):
            return pprint.pformat(d, indent=4, width=max_width)
        return pprint.pformat(d, width=max_width)

    @classmethod
    def dedupe(cls, text: str, dedupe: str) -> str:
        while (2 * dedupe) in text:
            text: str = text.replace(2 * dedupe, dedupe)
        return text

    ## Taken from: https://github.com/moby/moby/blob/0ad2293d0e5bbf4c966a0e8b27c3ac3835265577/pkg/namesgenerator/names-generator.go
    RANDOM_NAME_LEFT: List[str] = [
        "admiring", "adoring", "affectionate", "agitated", "amazing", "angry", "awesome", "beautiful", "blissful",
        "bold", "boring", "brave", "busy", "charming", "clever", "cool", "compassionate", "competent", "condescending",
        "confident", "cranky", "crazy", "dazzling", "determined", "distracted", "dreamy", "eager", "ecstatic",
        "elastic", "elated", "elegant", "eloquent", "epic", "exciting", "fervent", "festive", "flamboyant", "focused",
        "friendly", "frosty", "funny", "gallant", "gifted", "goofy", "gracious", "great", "happy", "hardcore",
        "heuristic", "hopeful", "hungry", "infallible", "inspiring", "interesting", "intelligent", "jolly", "jovial",
        "keen", "kind", "laughing", "loving", "lucid", "magical", "mystifying", "modest", "musing", "naughty",
        "nervous", "nice", "nifty", "nostalgic", "objective", "optimistic", "peaceful", "pedantic", "pensive",
        "practical", "priceless", "quirky", "quizzical", "recursing", "relaxed", "reverent", "romantic", "sad",
        "serene", "sharp", "silly", "sleepy", "stoic", "strange", "stupefied", "suspicious", "sweet", "tender",
        "thirsty", "trusting", "unruffled", "upbeat", "vibrant", "vigilant", "vigorous", "wizardly", "wonderful",
        "xenodochial", "youthful", "zealous", "zen",
    ]
    RANDOM_NAME_RIGHT: List[str] = [
        "albattani", "allen", "almeida", "antonelli", "agnesi", "archimedes", "ardinghelli", "aryabhata", "austin",
        "babbage", "banach", "banzai", "bardeen", "bartik", "bassi", "beaver", "bell", "benz", "bhabha", "bhaskara",
        "black", "blackburn", "blackwell", "bohr", "booth", "borg", "bose", "bouman", "boyd", "brahmagupta", "brattain",
        "brown", "buck", "burnell", "cannon", "carson", "cartwright", "carver", "cerf", "chandrasekhar", "chaplygin",
        "chatelet", "chatterjee", "chebyshev", "cohen", "chaum", "clarke", "colden", "cori", "cray", "curran", "curie",
        "darwin", "davinci", "dewdney", "dhawan", "diffie", "dijkstra", "dirac", "driscoll", "dubinsky", "easley",
        "edison", "einstein", "elbakyan", "elgamal", "elion", "ellis", "engelbart", "euclid", "euler", "faraday",
        "feistel", "fermat", "fermi", "feynman", "franklin", "gagarin", "galileo", "galois", "ganguly", "gates",
        "gauss", "germain", "goldberg", "goldstine", "goldwasser", "golick", "goodall", "gould", "greider",
        "grothendieck", "haibt", "hamilton", "haslett", "hawking", "hellman", "heisenberg", "hermann", "herschel",
        "hertz", "heyrovsky", "hodgkin", "hofstadter", "hoover", "hopper", "hugle", "hypatia", "ishizaka", "jackson",
        "jang", "jemison", "jennings", "jepsen", "johnson", "joliot", "jones", "kalam", "kapitsa", "kare", "keldysh",
        "keller", "kepler", "khayyam", "khorana", "kilby", "kirch", "knuth", "kowalevski", "lalande", "lamarr",
        "lamport", "leakey", "leavitt", "lederberg", "lehmann", "lewin", "lichterman", "liskov", "lovelace", "lumiere",
        "mahavira", "margulis", "matsumoto", "maxwell", "mayer", "mccarthy", "mcclintock", "mclaren", "mclean",
        "mcnulty", "mendel", "mendeleev", "meitner", "meninsky", "merkle", "mestorf", "mirzakhani", "montalcini",
        "moore", "morse", "murdock", "moser", "napier", "nash", "neumann", "newton", "nightingale", "nobel", "noether",
        "northcutt", "noyce", "panini", "pare", "pascal", "pasteur", "payne", "perlman", "pike", "poincare", "poitras",
        "proskuriakova", "ptolemy", "raman", "ramanujan", "ride", "ritchie", "rhodes", "robinson", "roentgen",
        "rosalind", "rubin", "saha", "sammet", "sanderson", "satoshi", "shamir", "shannon", "shaw", "shirley",
        "shockley", "shtern", "sinoussi", "snyder", "solomon", "spence", "stonebraker", "sutherland", "swanson",
        "swartz", "swirles", "taussig", "tereshkova", "tesla", "tharp", "thompson", "torvalds", "tu", "turing",
        "varahamihira", "vaughan", "visvesvaraya", "volhard", "villani", "wescoff", "wilbur", "wiles", "williams",
        "williamson", "wilson", "wing", "wozniak", "wright", "wu", "yalow", "yonath", "zhukovsky",
    ]

    ## Taken from: https://github.com/mrmaxguns/wonderwordsmodule/tree/master/wonderwords/assets
    RANDOM_VERBS: List[str] = [
        "abide", "accelerate", "accept", "accomplish", "achieve", "acquire", "acted", "activate", "adapt", "add",
        "address", "administer", "admire", "admit", "adopt", "advise", "afford", "agree", "alert", "alight", "allow",
        "altered", "amuse", "analyze", "announce", "annoy", "answer", "anticipate", "apologize", "appear", "applaud",
        "applied", "appoint", "appraise", "appreciate", "approve", "arbitrate", "argue", "arise", "arrange", "arrest",
        "arrive", "ascertain", "ask", "assemble", "assess", "assist", "assure", "attach", "attack", "attain", "attempt",
        "attend", "attract", "audited", "avoid", "awake", "back", "bake", "balance", "ban", "bang", "bare", "bat",
        "bathe", "battle", "be", "beam", "bear", "beat", "become", "beg", "begin", "behave", "behold", "belong", "bend",
        "beset", "bet", "bid", "bind", "bite", "bleach", "bleed", "bless", "blind", "blink", "blot", "blow", "blush",
        "boast", "boil", "bolt", "bomb", "book", "bore", "borrow", "bounce", "bow", "box", "brake", "branch", "break",
        "breathe", "breed", "brief", "bring", "broadcast", "bruise", "brush", "bubble", "budget", "build", "bump",
        "burn", "burst", "bury", "bust", "buy", "buze", "calculate", "call", "camp", "care", "carry", "carve", "cast",
        "catalog", "catch", "cause", "challenge", "change", "charge", "chart", "chase", "cheat", "check", "cheer",
        "chew", "choke", "choose", "chop", "claim", "clap", "clarify", "classify", "clean", "clear", "cling", "clip",
        "close", "clothe", "coach", "coil", "collect", "color", "comb", "come", "command", "communicate", "compare",
        "compete", "compile", "complain", "complete", "compose", "compute", "conceive", "concentrate", "conceptualize",
        "concern", "conclude", "conduct", "confess", "confront", "confuse", "connect", "conserve", "consider",
        "consist", "consolidate", "construct", "consult", "contain", "continue", "contract", "control", "convert",
        "coordinate", "copy", "correct", "correlate", "cost", "cough", "counsel", "count", "cover", "crack", "crash",
        "crawl", "create", "creep", "critique", "cross", "crush", "cry", "cure", "curl", "curve", "cut", "cycle", "dam",
        "damage", "dance", "dare", "deal", "decay", "deceive", "decide", "decorate", "define", "delay", "delegate",
        "delight", "deliver", "demonstrate", "depend", "describe", "desert", "deserve", "design", "destroy", "detail",
        "detect", "determine", "develop", "devise", "diagnose", "dig", "direct", "disagree", "disappear", "disapprove",
        "disarm", "discover", "dislike", "dispense", "display", "disprove", "dissect", "distribute", "dive", "divert",
        "divide", "do", "double", "doubt", "draft", "drag", "drain", "dramatize", "draw", "dream", "dress", "drink",
        "drip", "drive", "drop", "drown", "drum", "dry", "dust", "dwell", "earn", "eat", "edited", "educate",
        "eliminate", "embarrass", "employ", "empty", "enacted", "encourage", "end", "endure", "enforce", "engineer",
        "enhance", "enjoy", "enlist", "ensure", "enter", "entertain", "escape", "establish", "estimate", "evaluate",
        "examine", "exceed", "excite", "excuse", "execute", "exercise", "exhibit", "exist", "expand", "expect",
        "expedite", "experiment", "explain", "explode", "express", "extend", "extract", "face", "facilitate", "fade",
        "fail", "fancy", "fasten", "fax", "fear", "feed", "feel", "fence", "fetch", "fight", "file", "fill", "film",
        "finalize", "finance", "find", "fire", "fit", "fix", "flap", "flash", "flee", "fling", "float", "flood", "flow",
        "flower", "fly", "fold", "follow", "fool", "forbid", "force", "forecast", "forego", "foresee", "foretell",
        "forget", "forgive", "form", "formulate", "forsake", "frame", "freeze", "frighten", "fry", "gather", "gaze",
        "generate", "get", "give", "glow", "glue", "go", "govern", "grab", "graduate", "grate", "grease", "greet",
        "grin", "grind", "grip", "groan", "grow", "guarantee", "guard", "guess", "guide", "hammer", "hand", "handle",
        "handwrite", "hang", "happen", "harass", "harm", "hate", "haunt", "head", "heal", "heap", "hear", "heat",
        "help", "hide", "hit", "hold", "hook", "hop", "hope", "hover", "hug", "hum", "hunt", "hurry", "hurt",
        "hypothesize", "identify", "ignore", "illustrate", "imagine", "implement", "impress", "improve", "improvise",
        "include", "increase", "induce", "influence", "inform", "initiate", "inject", "injure", "inlay", "innovate",
        "input", "inspect", "inspire", "install", "institute", "instruct", "insure", "integrate", "intend", "intensify",
        "interest", "interfere", "interlay", "interpret", "interrupt", "interview", "introduce", "invent", "inventory",
        "investigate", "invite", "irritate", "itch", "jail", "jam", "jog", "join", "joke", "judge", "juggle", "jump",
        "justify", "keep", "kept", "kick", "kill", "kiss", "kneel", "knit", "knock", "knot", "know", "label", "land",
        "last", "laugh", "launch", "lay", "lead", "lean", "leap", "learn", "leave", "lecture", "led", "lend", "let",
        "level", "license", "lick", "lie", "lifted", "light", "lighten", "like", "list", "listen", "live", "load",
        "locate", "lock", "log", "long", "look", "lose", "love", "maintain", "make", "man", "manage", "manipulate",
        "manufacture", "map", "march", "mark", "market", "marry", "match", "mate", "matter", "mean", "measure",
        "meddle", "mediate", "meet", "melt", "melt", "memorize", "mend", "mentor", "milk", "mine", "mislead", "miss",
        "misspell", "mistake", "misunderstand", "mix", "moan", "model", "modify", "monitor", "moor", "motivate",
        "mourn", "move", "mow", "muddle", "mug", "multiply", "murder", "nail", "name", "navigate", "need", "negotiate",
        "nest", "nod", "nominate", "normalize", "note", "notice", "number", "obey", "object", "observe", "obtain",
        "occur", "offend", "offer", "officiate", "open", "operate", "order", "organize", "oriented", "originate",
        "overcome", "overdo", "overdraw", "overflow", "overhear", "overtake", "overthrow", "owe", "own", "pack",
        "paddle", "paint", "park", "part", "participate", "pass", "paste", "pat", "pause", "pay", "peck", "pedal",
        "peel", "peep", "perceive", "perfect", "perform", "permit", "persuade", "phone", "photograph", "pick", "pilot",
        "pinch", "pine", "pinpoint", "pioneer", "place", "plan", "plant", "play", "plead", "please", "plug", "point",
        "poke", "polish", "pop", "possess", "post", "pour", "practice", "praised", "pray", "preach", "precede",
        "predict", "prefer", "prepare", "prescribe", "present", "preserve", "preset", "preside", "press", "pretend",
        "prevent", "prick", "print", "process", "procure", "produce", "profess", "program", "progress", "project",
        "promise", "promote", "proofread", "propose", "protect", "prove", "provide", "publicize", "pull", "pump",
        "punch", "puncture", "punish", "purchase", "push", "put", "qualify", "question", "queue", "quit", "race",
        "radiate", "rain", "raise", "rank", "rate", "reach", "read", "realign", "realize", "reason", "receive",
        "recognize", "recommend", "reconcile", "record", "recruit", "reduce", "refer", "reflect", "refuse", "regret",
        "regulate", "rehabilitate", "reign", "reinforce", "reject", "rejoice", "relate", "relax", "release", "rely",
        "remain", "remember", "remind", "remove", "render", "reorganize", "repair", "repeat", "replace", "reply",
        "report", "represent", "reproduce", "request", "rescue", "research", "resolve", "respond", "restored",
        "restructure", "retire", "retrieve", "return", "review", "revise", "rhyme", "rid", "ride", "ring", "rinse",
        "rise", "risk", "rob", "rock", "roll", "rot", "rub", "ruin", "rule", "run", "rush", "sack", "sail", "satisfy",
        "save", "saw", "say", "scare", "scatter", "schedule", "scold", "scorch", "scrape", "scratch", "scream", "screw",
        "scribble", "scrub", "seal", "search", "secure", "see", "seek", "select", "sell", "send", "sense", "separate",
        "serve", "service", "set", "settle", "sew", "shade", "shake", "shape", "share", "shave", "shear", "shed",
        "shelter", "shine", "shiver", "shock", "shoe", "shoot", "shop", "show", "shrink", "shrug", "shut", "sigh",
        "sign", "signal", "simplify", "sin", "sing", "sink", "sip", "sit", "sketch", "ski", "skip", "slap", "slay",
        "sleep", "slide", "sling", "slink", "slip", "slit", "slow", "smash", "smell", "smile", "smite", "smoke",
        "snatch", "sneak", "sneeze", "sniff", "snore", "snow", "soak", "solve", "soothe", "soothsay", "sort", "sound",
        "sow", "spare", "spark", "sparkle", "speak", "specify", "speed", "spell", "spend", "spill", "spin", "spit",
        "split", "spoil", "spot", "spray", "spread", "spring", "sprout", "squash", "squeak", "squeal", "squeeze",
        "stain", "stamp", "stand", "stare", "start", "stay", "steal", "steer", "step", "stick", "stimulate", "sting",
        "stink", "stir", "stitch", "stop", "store", "strap", "streamline", "strengthen", "stretch", "stride", "strike",
        "string", "strip", "strive", "stroke", "structure", "study", "stuff", "sublet", "subtract", "succeed", "suck",
        "suffer", "suggest", "suit", "summarize", "supervise", "supply", "support", "suppose", "surprise", "surround",
        "suspect", "suspend", "swear", "sweat", "sweep", "swell", "swim", "swing", "switch", "symbolize", "synthesize",
        "systemize", "tabulate", "take", "talk", "tame", "tap", "target", "taste", "teach", "tear", "tease",
        "telephone", "tell", "tempt", "terrify", "test", "thank", "thaw", "think", "thrive", "throw", "thrust", "tick",
        "tickle", "tie", "time", "tip", "tire", "touch", "tour", "tow", "trace", "trade", "train", "transcribe",
        "transfer", "transform", "translate", "transport", "trap", "travel", "tread", "treat", "tremble", "trick",
        "trip", "trot", "trouble", "troubleshoot", "trust", "try", "tug", "tumble", "turn", "tutor", "twist", "type",
        "undergo", "understand", "undertake", "undress", "unfasten", "unify", "unite", "unlock", "unpack", "untidy",
        "update", "upgrade", "uphold", "upset", "use", "utilize", "vanish", "verbalize", "verify", "vex", "visit",
        "wail", "wait", "wake", "walk", "wander", "want", "warm", "warn", "wash", "waste", "watch", "water", "wave",
        "wear", "weave", "wed", "weep", "weigh", "welcome", "wend", "wet", "whine", "whip", "whirl", "whisper",
        "whistle", "win", "wind", "wink", "wipe", "wish", "withdraw", "withhold", "withstand", "wobble", "wonder",
        "work", "worry", "wrap", "wreck", "wrestle", "wriggle", "wring", "write", "x-ray", "yawn", "yell", "zip",
        "zoom",
    ]

    RANDOM_ADJECTIVES: List[str] = [
        "quizzical", "highfalutin", "dynamic", "wakeful", "cheerful", "thoughtful", "cooperative", "questionable",
        "abundant", "uneven", "yummy", "juicy", "vacuous", "concerned", "young", "sparkling", "abhorrent", "sweltering",
        "late", "macho", "scrawny", "friendly", "kaput", "divergent", "busy", "charming", "protective", "premium",
        "puzzled", "waggish", "rambunctious", "puffy", "hard", "fat", "sedate", "yellow", "resonant", "dapper",
        "courageous", "vast", "cool", "elated", "wary", "bewildered", "level", "wooden", "ceaseless", "tearful",
        "cloudy", "gullible", "flashy", "trite", "quick", "nondescript", "round", "slow", "spiritual", "brave",
        "tenuous", "abstracted", "colossal", "sloppy", "obsolete", "elegant", "fabulous", "vivacious", "exuberant",
        "faithful", "helpless", "odd", "sordid", "blue", "imported", "ugly", "ruthless", "deeply", "eminent",
        "reminiscent", "rotten", "sour", "volatile", "succinct", "judicious", "abrupt", "learned", "stereotyped",
        "evanescent", "efficacious", "festive", "loose", "torpid", "condemned", "selective", "strong", "momentous",
        "ordinary", "dry", "great", "ultra", "ahead", "broken", "dusty", "piquant", "creepy", "miniature", "periodic",
        "equable", "unsightly", "narrow", "grieving", "whimsical", "fantastic", "kindhearted", "miscreant", "cowardly",
        "cloistered", "marked", "bloody", "chunky", "undesirable", "oval", "nauseating", "aberrant", "stingy",
        "standing", "distinct", "illegal", "angry", "faint", "rustic", "few", "calm", "gorgeous", "mysterious", "tacky",
        "unadvised", "greasy", "minor", "loving", "melodic", "flat", "wretched", "clever", "barbarous", "pretty",
        "endurable", "handsomely", "unequaled", "acceptable", "symptomatic", "hurt", "tested", "long", "warm",
        "ignorant", "ashamed", "excellent", "known", "adamant", "eatable", "verdant", "meek", "unbiased", "rampant",
        "somber", "cuddly", "harmonious", "salty", "overwrought", "stimulating", "beautiful", "crazy", "grouchy",
        "thirsty", "joyous", "confused", "terrible", "high", "unarmed", "gabby", "wet", "sharp", "wonderful", "magenta",
        "tan", "huge", "productive", "defective", "chilly", "needy", "imminent", "flaky", "fortunate", "neighborly",
        "hot", "husky", "optimal", "gaping", "faulty", "guttural", "massive", "watery", "abrasive", "ubiquitous",
        "aspiring", "impartial", "annoyed", "billowy", "lucky", "panoramic", "heartbreaking", "fragile", "purring",
        "wistful", "burly", "filthy", "psychedelic", "harsh", "disagreeable", "ambiguous", "short", "splendid",
        "crowded", "light", "yielding", "hypnotic", "dispensable", "deserted", "nonchalant", "green", "puny",
        "deafening", "classy", "tall", "typical", "exclusive", "materialistic", "mute", "shaky", "inconclusive",
        "rebellious", "doubtful", "telling", "unsuitable", "woebegone", "cold", "sassy", "arrogant", "perfect",
        "adhesive", "industrious", "crabby", "curly", "voiceless", "nostalgic", "better", "slippery", "willing",
        "nifty", "orange", "victorious", "ritzy", "wacky", "vigorous", "spotless", "good", "powerful", "bashful",
        "soggy", "grubby", "moaning", "placid", "permissible", "half", "towering", "bawdy", "measly", "abaft",
        "delightful", "goofy", "capricious", "nonstop", "addicted", "acoustic", "furtive", "erratic", "heavy", "square",
        "delicious", "needless", "resolute", "innocent", "abnormal", "hurried", "awful", "impossible", "aloof", "giddy",
        "large", "pointless", "petite", "jolly", "boundless", "abounding", "hilarious", "heavenly", "honorable",
        "squeamish", "red", "phobic", "trashy", "pathetic", "parched", "godly", "greedy", "pleasant", "small",
        "aboriginal", "dashing", "icky", "bumpy", "laughable", "hapless", "silent", "scary", "shaggy", "organic",
        "unbecoming", "inexpensive", "wrong", "repulsive", "flawless", "labored", "disturbed", "aboard", "gusty",
        "loud", "jumbled", "exotic", "vulgar", "threatening", "belligerent", "synonymous", "encouraging", "fancy",
        "embarrassed", "clumsy", "fast", "ethereal", "chubby", "high-pitched", "plastic", "open", "straight", "little",
        "ancient", "fair", "psychotic", "murky", "earthy", "callous", "heady", "lamentable", "hallowed", "obtainable",
        "toothsome", "oafish", "gainful", "flippant", "tangy", "tightfisted", "damaging", "utopian", "gaudy", "brainy",
        "imperfect", "shiny", "fanatical", "snotty", "relieved", "shallow", "foamy", "parsimonious", "gruesome",
        "elite", "wide", "kind", "bored", "tangible", "depressed", "boring", "screeching", "outrageous", "determined",
        "picayune", "glossy", "historical", "staking", "curious", "gigantic", "wandering", "profuse", "vengeful",
        "glib", "unaccountable", "frightened", "outstanding", "chivalrous", "workable", "modern", "swanky",
        "comfortable", "gentle", "substantial", "brawny", "curved", "nebulous", "boorish", "afraid", "fierce",
        "efficient", "lackadaisical", "recondite", "internal", "absorbed", "squealing", "frail", "thundering",
        "wanting", "cooing", "axiomatic", "debonair", "boiling", "tired", "numberless", "flowery", "mushy",
        "enthusiastic", "proud", "upset", "hungry", "astonishing", "deadpan", "prickly", "mammoth", "absurd", "clean",
        "jittery", "wry", "entertaining", "literate", "lying", "uninterested", "aquatic", "super", "languid", "cute",
        "absorbing", "scattered", "brief", "halting", "bright", "fuzzy", "lethal", "scarce", "aggressive", "obsequious",
        "fine", "giant", "holistic", "pastoral", "stormy", "quaint", "nervous", "wasteful", "grotesque", "loutish",
        "abiding", "unable", "black", "dysfunctional", "knowledgeable", "truculent", "various", "luxuriant", "shrill",
        "spiffy", "guarded", "colorful", "misty", "spurious", "freezing", "glamorous", "famous", "new", "instinctive",
        "nasty", "exultant", "seemly", "tawdry", "maniacal", "wrathful", "shy", "nutritious", "idiotic", "worried",
        "bad", "stupid", "ruddy", "wholesale", "naughty", "thoughtless", "futuristic", "available", "slimy", "cynical",
        "fluffy", "plausible", "nasty", "tender", "changeable", "smiling", "oceanic", "satisfying", "steadfast",
        "ugliest", "crooked", "subsequent", "fascinated", "woozy", "teeny", "quickest", "moldy", "uppity", "sable",
        "horrible", "silly", "ad hoc", "numerous", "berserk", "wiry", "knowing", "lazy", "childlike", "zippy",
        "fearless", "pumped", "weak", "tacit", "weary", "rapid", "precious", "smoggy", "swift", "lyrical", "steep",
        "quack", "direful", "talented", "hesitant", "fallacious", "ill", "quarrelsome", "quiet", "flipped-out",
        "didactic", "fluttering", "glorious", "tough", "sulky", "elfin", "abortive", "sweet", "habitual", "supreme",
        "hollow", "possessive", "inquisitive", "adjoining", "incandescent", "lowly", "majestic", "bizarre", "acrid",
        "expensive", "aback", "unusual", "foolish", "jobless", "capable", "damp", "political", "dazzling", "erect",
        "Early", "immense", "hellish", "omniscient", "reflective", "lovely", "incompetent", "empty", "breakable",
        "educated", "easy", "devilish", "assorted", "decorous", "jaded", "homely", "dangerous", "adaptable", "coherent",
        "dramatic", "tense", "abject", "fretful", "troubled", "diligent", "solid", "plain", "raspy", "irate", "offbeat",
        "healthy", "melted", "cagey", "many", "wild", "venomous", "animated", "alike", "youthful", "ripe", "alcoholic",
        "sincere", "teeny-tiny", "lush", "defeated", "zonked", "foregoing", "dizzy", "frantic", "obnoxious", "funny",
        "damaged", "grandiose", "spectacular", "maddening", "defiant", "makeshift", "strange", "painstaking",
        "merciful", "madly", "clammy", "itchy", "difficult", "clear", "used", "temporary", "abandoned", "null", "rainy",
        "evil", "alert", "domineering", "amuck", "rabid", "jealous", "robust", "obeisant", "overt", "enchanting",
        "longing", "cautious", "motionless", "bitter", "anxious", "craven", "breezy", "ragged", "skillful", "quixotic",
        "knotty", "grumpy", "dark", "draconian", "alluring", "magical", "versed", "humdrum", "accurate", "ludicrous",
        "sleepy", "envious", "lavish", "roasted", "thinkable", "overconfident", "roomy", "painful", "wee", "observant",
        "old-fashioned", "drunk", "royal", "likeable", "adventurous", "eager", "obedient", "attractive", "x-rated",
        "spooky", "poised", "righteous", "excited", "real", "abashed", "womanly", "ambitious", "lacking", "testy",
        "big", "gamy", "early", "auspicious", "blue-eyed", "discreet", "nappy", "vague", "helpful", "nosy",
        "perpetual", "disillusioned", "overrated", "gleaming", "tart", "soft", "agreeable", "therapeutic", "accessible",
        "poor", "gifted", "old", "humorous", "flagrant", "magnificent", "alive", "understood", "economic", "mighty",
        "ablaze", "racial", "tasteful", "purple", "broad", "lean", "legal", "witty", "nutty", "icy", "feigned",
        "redundant", "adorable", "apathetic", "jumpy", "scientific", "combative", "worthless", "tasteless", "voracious",
        "jazzy", "uptight", "utter", "hospitable", "imaginary", "finicky", "shocking", "dead", "noisy", "shivering",
        "subdued", "rare", "zealous", "demonic", "ratty", "snobbish", "deranged", "muddy", "whispering", "credible",
        "hulking", "fertile", "tight", "abusive", "functional", "obscene", "thankful", "daffy", "smelly", "lively",
        "homeless", "secretive", "amused", "lewd", "mere", "agonizing", "sad", "innate", "sneaky", "noxious",
        "illustrious", "alleged", "cultured", "tame", "macabre", "lonely", "mindless", "low", "scintillating",
        "statuesque", "decisive", "rhetorical", "hysterical", "happy", "earsplitting", "mundane", "spicy", "overjoyed",
        "taboo", "peaceful", "forgetful", "elderly", "upbeat", "squalid", "warlike", "dull", "plucky", "handsome",
        "groovy", "absent", "wise", "romantic", "invincible", "receptive", "smooth", "different", "tiny", "cruel",
        "dirty", "mature", "faded", "tiresome", "wicked", "average", "panicky", "detailed", "juvenile", "scandalous",
        "steady", "wealthy", "deep", "sticky", "jagged", "wide-eyed", "tasty", "disgusted", "garrulous", "graceful",
        "tranquil", "annoying", "hissing", "noiseless", "selfish", "onerous", "lopsided", "ossified", "penitent",
        "malicious", "aromatic", "successful", "zany", "evasive", "wet", "naive", "nice", "uttermost", "brash",
        "muddled", "energetic", "accidental", "silky", "guiltless", "important", "drab", "aware", "skinny", "careful",
        "rightful", "tricky", "sore", "rich", "blushing", "stale", "daily", "watchful", "uncovered", "rough", "fresh",
        "hushed", "rural",
    ]

    RANDOM_NOUNS: List[str] = [
        "aardvark", "abacus", "abbey", "abbreviation", "abdomen", "ability", "abnormality", "abolishment", "abrogation",
        "absence", "abundance", "abuse", "academics", "academy", "accelerant", "accelerator", "accent", "acceptance",
        "access", "accessory", "accident", "accommodation", "accompanist", "accomplishment", "accord", "accordance",
        "accordion", "account", "accountability", "accountant", "accounting", "accuracy", "accusation", "acetate",
        "achievement", "achiever", "acid", "acknowledgment", "acorn", "acoustics", "acquaintance", "acquisition",
        "acre", "acrylic", "act", "action", "activation", "activist", "activity", "actor", "actress", "acupuncture",
        "ad", "adaptation", "adapter", "addiction", "addition", "address", "adjective", "adjustment", "admin",
        "administration", "administrator", "admire", "admission", "adobe", "adoption", "adrenalin", "adrenaline",
        "adult", "adulthood", "advance", "advancement", "advantage", "advent", "adverb", "advertisement", "advertising",
        "advice", "adviser", "advocacy", "advocate", "affair", "affect", "affidavit", "affiliate", "affinity", "afoul",
        "afterlife", "aftermath", "afternoon", "aftershave", "aftershock", "afterthought", "age", "agency", "agenda",
        "agent", "aggradation", "aggression", "aglet", "agony", "agreement", "agriculture", "aid", "aide", "aim", "air",
        "airbag", "airbus", "aircraft", "airfare", "airfield", "airforce", "airline", "airmail", "airman", "airplane",
        "airport", "airship", "airspace", "alarm", "alb", "albatross", "album", "alcohol", "alcove", "alder", "ale",
        "alert", "alfalfa", "algebra", "algorithm", "alias", "alibi", "alien", "allegation", "allergist", "alley",
        "alliance", "alligator", "allocation", "allowance", "alloy", "alluvium", "almanac", "almighty", "almond",
        "alpaca", "alpenglow", "alpenhorn", "alpha", "alphabet", "altar", "alteration", "alternative", "altitude",
        "alto", "aluminium", "aluminum", "amazement", "amazon", "ambassador", "amber", "ambience", "ambiguity",
        "ambition", "ambulance", "amendment", "amenity", "ammunition", "amnesty", "amount", "amusement", "anagram",
        "analgesia", "analog", "analogue", "analogy", "analysis", "analyst", "analytics", "anarchist", "anarchy",
        "anatomy", "ancestor", "anchovy", "android", "anesthesiologist", "anesthesiology", "angel", "anger", "angina",
        "angiosperm", "angle", "angora", "angstrom", "anguish", "animal", "anime", "anise", "ankle", "anklet",
        "anniversary", "announcement", "annual", "anorak", "answer", "ant", "anteater", "antecedent", "antechamber",
        "antelope", "antennae", "anterior", "anthropology", "antibody", "anticipation", "anticodon", "antigen",
        "antique", "antiquity", "antler", "antling", "anxiety", "anybody", "anyone", "anything", "anywhere",
        "apartment", "ape", "aperitif", "apology", "app", "apparatus", "apparel", "appeal", "appearance", "appellation",
        "appendix", "appetiser", "appetite", "appetizer", "applause", "apple", "applewood", "appliance", "application",
        "appointment", "appreciation", "apprehension", "approach", "appropriation", "approval", "apricot", "apron",
        "apse", "aquarium", "aquifer", "arcade", "arch", "arch-rival", "archaeologist", "archaeology", "archeology",
        "archer", "architect", "architecture", "archives", "area", "arena", "argument", "arithmetic", "ark", "arm",
        "arm-rest", "armadillo", "armament", "armchair", "armoire", "armor", "armour", "armpit", "armrest", "army",
        "arrangement", "array", "arrest", "arrival", "arrogance", "arrow", "art", "artery", "arthur", "artichoke",
        "article", "artifact", "artificer", "artist", "ascend", "ascent", "ascot", "ash", "ashram", "ashtray", "aside",
        "asparagus", "aspect", "asphalt", "aspic", "assassination", "assault", "assembly", "assertion", "assessment",
        "asset", "assignment", "assist", "assistance", "assistant", "associate", "association", "assumption",
        "assurance", "asterisk", "astrakhan", "astrolabe", "astrologer", "astrology", "astronomy", "asymmetry",
        "atelier", "atheist", "athlete", "athletics", "atmosphere", "atom", "atrium", "attachment", "attack",
        "attacker", "attainment", "attempt", "attendance", "attendant", "attention", "attenuation", "attic", "attitude",
        "attorney", "attraction", "attribute", "auction", "audience", "audit", "auditorium", "aunt", "authentication",
        "authenticity", "author", "authorisation", "authority", "authorization", "auto", "autoimmunity", "automation",
        "automaton", "autumn", "availability", "avalanche", "avenue", "average", "avocado", "award", "awareness", "awe",
        "axis", "azimuth", "babe", "baboon", "babushka", "baby", "bachelor", "back", "back-up", "backbone", "backburn",
        "backdrop", "background", "backpack", "backup", "backyard", "bacon", "bacterium", "badge", "badger",
        "bafflement", "bag", "bagel", "baggage", "baggie", "baggy", "bagpipe", "bail", "bait", "bake", "baker",
        "bakery", "bakeware", "balaclava", "balalaika", "balance", "balcony", "ball", "ballet", "balloon", "balloonist",
        "ballot", "ballpark", "bamboo", "ban", "banana", "band", "bandana", "bandanna", "bandolier", "bandwidth",
        "bangle", "banjo", "bank", "bankbook", "banker", "banking", "bankruptcy", "banner", "banquette", "banyan",
        "baobab", "bar", "barbecue", "barbeque", "barber", "barbiturate", "bargain", "barge", "baritone", "barium",
        "bark", "barley", "barn", "barometer", "barracks", "barrage", "barrel", "barrier", "barstool", "bartender",
        "base", "baseball", "baseboard", "baseline", "basement", "basics", "basil", "basin", "basis", "basket",
        "basketball", "bass", "bassinet", "bassoon", "bat", "bath", "bather", "bathhouse", "bathrobe", "bathroom",
        "bathtub", "battalion", "batter", "battery", "batting", "battle", "battleship", "bay", "bayou", "beach", "bead",
        "beak", "beam", "bean", "beancurd", "beanie", "beanstalk", "bear", "beard", "beast", "beastie", "beat",
        "beating", "beauty", "beaver", "beck", "bed", "bedrock", "bedroom", "bee", "beech", "beef", "beer", "beet",
        "beetle", "beggar", "beginner", "beginning", "begonia", "behalf", "behavior", "behaviour", "beheading",
        "behest", "behold", "being", "belfry", "belief", "believer", "bell", "belligerency", "bellows", "belly", "belt",
        "bench", "bend", "beneficiary", "benefit", "beret", "berry", "best-seller", "bestseller", "bet", "beverage",
        "beyond", "bias", "bibliography", "bicycle", "bid", "bidder", "bidding", "bidet", "bifocals", "bijou", "bike",
        "bikini", "bill", "billboard", "billing", "billion", "bin", "binoculars", "biology", "biopsy", "biosphere",
        "biplane", "birch", "bird", "bird-watcher", "birdbath", "birdcage", "birdhouse", "birth", "birthday", "biscuit",
        "bit", "bite", "bitten", "bitter", "black", "blackberry", "blackbird", "blackboard", "blackfish", "blackness",
        "bladder", "blade", "blame", "blank", "blanket", "blast", "blazer", "blend", "blessing", "blight", "blind",
        "blinker", "blister", "blizzard", "block", "blocker", "blog", "blogger", "blood", "bloodflow", "bloom",
        "bloomer", "blossom", "blouse", "blow", "blowgun", "blowhole", "blue", "blueberry", "blush", "boar", "board",
        "boat", "boatload", "boatyard", "bob", "bobcat", "body", "bog", "bolero", "bolt", "bomb", "bomber", "bombing",
        "bond", "bonding", "bondsman", "bone", "bonfire", "bongo", "bonnet", "bonsai", "bonus", "boogeyman", "book",
        "bookcase", "bookend", "booking", "booklet", "bookmark", "boolean", "boom", "boon", "boost", "booster", "boot",
        "bootee", "bootie", "booty", "border", "bore", "borrower", "borrowing", "bosom", "boss", "botany", "bother",
        "bottle", "bottling", "bottom", "bottom-line", "boudoir", "bough", "boulder", "boulevard", "boundary",
        "bouquet", "bourgeoisie", "bout", "boutique", "bow", "bower", "bowl", "bowler", "bowling", "bowtie", "box",
        "boxer", "boxspring", "boy", "boycott", "boyfriend", "boyhood", "boysenberry", "bra", "brace", "bracelet",
        "bracket", "brain", "brake", "bran", "branch", "brand", "brandy", "brass", "brassiere", "bratwurst", "bread",
        "breadcrumb", "breadfruit", "break", "breakdown", "breakfast", "breakpoint", "breakthrough", "breast",
        "breastplate", "breath", "breeze", "brewer", "bribery", "brick", "bricklaying", "bride", "bridge", "brief",
        "briefing", "briefly", "briefs", "brilliant", "brink", "brisket", "broad", "broadcast", "broccoli", "brochure",
        "brocolli", "broiler", "broker", "bronchitis", "bronco", "bronze", "brooch", "brood", "brook", "broom",
        "brother", "brother-in-law", "brow", "brown", "brownie", "browser", "browsing", "brunch", "brush", "brushfire",
        "brushing", "bubble", "buck", "bucket", "buckle", "buckwheat", "bud", "buddy", "budget", "buffalo", "buffer",
        "buffet", "bug", "buggy", "bugle", "builder", "building", "bulb", "bulk", "bull", "bull-fighter", "bulldozer",
        "bullet", "bump", "bumper", "bun", "bunch", "bungalow", "bunghole", "bunkhouse", "burden", "bureau", "burglar",
        "burial", "burlesque", "burn", "burn-out", "burning", "burrito", "burro", "burrow", "burst", "bus", "bush",
        "business", "businessman", "bust", "bustle", "butane", "butcher", "butler", "butter", "butterfly", "button",
        "buy", "buyer", "buying", "buzz", "buzzard", "c-clamp", "cabana", "cabbage", "cabin", "cabinet", "cable",
        "caboose", "cacao", "cactus", "caddy", "cadet", "cafe", "caffeine", "caftan", "cage", "cake", "calcification",
        "calculation", "calculator", "calculus", "calendar", "calf", "caliber", "calibre", "calico", "call", "calm",
        "calorie", "camel", "cameo", "camera", "camp", "campaign", "campaigning", "campanile", "camper", "campus",
        "can", "canal", "cancer", "candelabra", "candidacy", "candidate", "candle", "candy", "cane", "cannibal",
        "cannon", "canoe", "canon", "canopy", "cantaloupe", "canteen", "canvas", "cap", "capability", "capacity",
        "cape", "caper", "capital", "capitalism", "capitulation", "capon", "cappelletti", "cappuccino", "captain",
        "caption", "captor", "car", "carabao", "caramel", "caravan", "carbohydrate", "carbon", "carboxyl", "card",
        "cardboard", "cardigan", "care", "career", "cargo", "caribou", "carload", "carnation", "carnival", "carol",
        "carotene", "carp", "carpenter", "carpet", "carpeting", "carport", "carriage", "carrier", "carrot", "carry",
        "cart", "cartel", "carter", "cartilage", "cartload", "cartoon", "cartridge", "carving", "cascade", "case",
        "casement", "cash", "cashew", "cashier", "casino", "casket", "cassava", "casserole", "cassock", "cast",
        "castanet", "castle", "casualty", "cat", "catacomb", "catalogue", "catalysis", "catalyst", "catamaran",
        "catastrophe", "catch", "catcher", "category", "caterpillar", "cathedral", "cation", "catsup", "cattle",
        "cauliflower", "causal", "cause", "causeway", "caution", "cave", "caviar", "cayenne", "ceiling", "celebration",
        "celebrity", "celeriac", "celery", "cell", "cellar", "cello", "celsius", "cement", "cemetery", "cenotaph",
        "census", "cent", "center", "centimeter", "centre", "centurion", "century", "cephalopod", "ceramic", "ceramics",
        "cereal", "ceremony", "certainty", "certificate", "certification", "cesspool", "chafe", "chain", "chainstay",
        "chair", "chairlift", "chairman", "chairperson", "chaise", "chalet", "chalice", "chalk", "challenge", "chamber",
        "champagne", "champion", "championship", "chance", "chandelier", "change", "channel", "chaos", "chap", "chapel",
        "chaplain", "chapter", "character", "characteristic", "characterization", "chard", "charge", "charger",
        "charity", "charlatan", "charm", "charset", "chart", "charter", "chasm", "chassis", "chastity", "chasuble",
        "chateau", "chatter", "chauffeur", "chauvinist", "check", "checkbook", "checking", "checkout", "checkroom",
        "cheddar", "cheek", "cheer", "cheese", "cheesecake", "cheetah", "chef", "chem", "chemical", "chemistry",
        "chemotaxis", "cheque", "cherry", "chess", "chest", "chestnut", "chick", "chicken", "chicory", "chief",
        "chiffonier", "child", "childbirth", "childhood", "chili", "chill", "chime", "chimpanzee", "chin", "chinchilla",
        "chino", "chip", "chipmunk", "chit-chat", "chivalry", "chive", "chives", "chocolate", "choice", "choir",
        "choker", "cholesterol", "choosing", "chop", "chops", "chopstick", "chopsticks", "chord", "chorus", "chow",
        "chowder", "chrome", "chromolithograph", "chronicle", "chronograph", "chronometer", "chrysalis", "chub",
        "chuck", "chug", "church", "churn", "chutney", "cicada", "cigarette", "cilantro", "cinder", "cinema",
        "cinnamon", "circadian", "circle", "circuit", "circulation", "circumference", "circumstance", "cirrhosis",
        "cirrus", "citizen", "citizenship", "citron", "citrus", "city", "civilian", "civilisation", "civilization",
        "claim", "clam", "clamp", "clan", "clank", "clapboard", "clarification", "clarinet", "clarity", "clasp",
        "class", "classic", "classification", "classmate", "classroom", "clause", "clave", "clavicle", "clavier",
        "claw", "clay", "cleaner", "clearance", "clearing", "cleat", "cleavage", "clef", "cleft", "clergyman", "cleric",
        "clerk", "click", "client", "cliff", "climate", "climb", "clinic", "clip", "clipboard", "clipper", "cloak",
        "cloakroom", "clock", "clockwork", "clogs", "cloister", "clone", "close", "closet", "closing", "closure",
        "cloth", "clothes", "clothing", "cloud", "cloudburst", "clove", "clover", "cloves", "club", "clue", "cluster",
        "clutch", "co-producer", "coach", "coal", "coalition", "coast", "coaster", "coat", "cob", "cobbler", "cobweb",
        "cockpit", "cockroach", "cocktail", "cocoa", "coconut", "cod", "code", "codepage", "codling", "codon",
        "codpiece", "coevolution", "cofactor", "coffee", "coffin", "cohesion", "cohort", "coil", "coin", "coincidence",
        "coinsurance", "coke", "cold", "coleslaw", "coliseum", "collaboration", "collagen", "collapse", "collar",
        "collard", "collateral", "colleague", "collection", "collectivisation", "collectivization", "collector",
        "college", "collision", "colloquy", "colon", "colonial", "colonialism", "colonisation", "colonization",
        "colony", "color", "colorlessness", "colt", "column", "columnist", "comb", "combat", "combination", "combine",
        "comeback", "comedy", "comestible", "comfort", "comfortable", "comic", "comics", "comma", "command",
        "commander", "commandment", "comment", "commerce", "commercial", "commission", "commitment", "committee",
        "commodity", "common", "commonsense", "commotion", "communicant", "communication", "communion", "communist",
        "community", "commuter", "company", "comparison", "compass", "compassion", "compassionate", "compensation",
        "competence", "competition", "competitor", "complaint", "complement", "completion", "complex", "complexity",
        "compliance", "complication", "complicity", "compliment", "component", "comportment", "composer", "composite",
        "composition", "compost", "comprehension", "compress", "compromise", "comptroller", "compulsion", "computer",
        "comradeship", "con", "concentrate", "concentration", "concept", "conception", "concern", "concert",
        "conclusion", "concrete", "condition", "conditioner", "condominium", "condor", "conduct", "conductor", "cone",
        "confectionery", "conference", "confidence", "confidentiality", "configuration", "confirmation", "conflict",
        "conformation", "confusion", "conga", "congo", "congregation", "congress", "congressman", "congressperson",
        "conifer", "connection", "connotation", "conscience", "consciousness", "consensus", "consent", "consequence",
        "conservation", "conservative", "consideration", "consignment", "consist", "consistency", "console",
        "consonant", "conspiracy", "conspirator", "constant", "constellation", "constitution", "constraint",
        "construction", "consul", "consulate", "consulting", "consumer", "consumption", "contact", "contact lens",
        "contagion", "container", "content", "contention", "contest", "context", "continent", "contingency",
        "continuity", "contour", "contract", "contractor", "contrail", "contrary", "contrast", "contribution",
        "contributor", "control", "controller", "controversy", "convection", "convenience", "convention",
        "conversation", "conversion", "convert", "convertible", "conviction", "cook", "cookbook", "cookie", "cooking",
        "coonskin", "cooperation", "coordination", "coordinator", "cop", "cop-out", "cope", "copper", "copy", "copying",
        "copyright", "copywriter", "coral", "cord", "corduroy", "core", "cork", "cormorant", "corn", "corner",
        "cornerstone", "cornet", "cornflakes", "cornmeal", "corporal", "corporation", "corporatism", "corps", "corral",
        "correspondence", "correspondent", "corridor", "corruption", "corsage", "cosset", "cost", "costume", "cot",
        "cottage", "cotton", "couch", "cougar", "cough", "council", "councilman", "councilor", "councilperson",
        "counsel", "counseling", "counselling", "counsellor", "counselor", "count", "counter", "counter-force",
        "counterpart", "counterterrorism", "countess", "country", "countryside", "county", "couple", "coupon",
        "courage", "course", "court", "courthouse", "courtroom", "cousin", "covariate", "cover", "coverage", "coverall",
        "cow", "cowbell", "cowboy", "coyote", "crab", "crack", "cracker", "crackers", "cradle", "craft", "craftsman",
        "cranberry", "crane", "cranky", "crash", "crate", "cravat", "craw", "crawdad", "crayfish", "crayon", "crazy",
        "cream", "creation", "creationism", "creationist", "creative", "creativity", "creator", "creature", "creche",
        "credential", "credenza", "credibility", "credit", "creditor", "creek", "creme brulee", "crepe", "crest",
        "crew", "crewman", "crewmate", "crewmember", "crewmen", "cria", "crib", "cribbage", "cricket", "cricketer",
        "crime", "criminal", "crinoline", "crisis", "crisp", "criteria", "criterion", "critic", "criticism",
        "crocodile", "crocus", "croissant", "crook", "crop", "cross", "cross-contamination", "cross-stitch", "crotch",
        "croup", "crow", "crowd", "crown", "crucifixion", "crude", "cruelty", "cruise", "crumb", "crunch", "crusader",
        "crush", "crust", "cry", "crystal", "crystallography", "cub", "cube", "cuckoo", "cucumber", "cue", "cuff-link",
        "cuisine", "cultivar", "cultivator", "culture", "culvert", "cummerbund", "cup", "cupboard", "cupcake", "cupola",
        "curd", "cure", "curio", "curiosity", "curl", "curler", "currant", "currency", "current", "curriculum", "curry",
        "curse", "cursor", "curtailment", "curtain", "curve", "cushion", "custard", "custody", "custom", "customer",
        "cut", "cuticle", "cutlet", "cutover", "cutting", "cyclamen", "cycle", "cyclone", "cyclooxygenase", "cygnet",
        "cylinder", "cymbal", "cynic", "cyst", "cytokine", "cytoplasm", "dad", "daddy", "daffodil", "dagger", "dahlia",
        "daikon", "daily", "dairy", "daisy", "dam", "damage", "dame", "dance", "dancer", "dancing", "dandelion",
        "danger", "dare", "dark", "darkness", "darn", "dart", "dash", "dashboard", "data", "database", "date",
        "daughter", "dawn", "day", "daybed", "daylight", "dead", "deadline", "deal", "dealer", "dealing", "dearest",
        "death", "deathwatch", "debate", "debris", "debt", "debtor", "decade", "decadence", "decency", "decimal",
        "decision", "decision-making", "deck", "declaration", "declination", "decline", "decoder", "decongestant",
        "decoration", "decrease", "decryption", "dedication", "deduce", "deduction", "deed", "deep", "deer", "default",
        "defeat", "defendant", "defender", "defense", "deficit", "definition", "deformation", "degradation", "degree",
        "delay", "deliberation", "delight", "delivery", "demand", "democracy", "democrat", "demon", "demur", "den",
        "denim", "denominator", "density", "dentist", "deodorant", "department", "departure", "dependency", "dependent",
        "deployment", "deposit", "deposition", "depot", "depression", "depressive", "depth", "deputy", "derby",
        "derivation", "derivative", "derrick", "descendant", "descent", "description", "desert", "design",
        "designation", "designer", "desire", "desk", "desktop", "dessert", "destination", "destiny", "destroyer",
        "destruction", "detail", "detainee", "detainment", "detection", "detective", "detector", "detention",
        "determination", "detour", "devastation", "developer", "developing", "development", "developmental", "deviance",
        "deviation", "device", "devil", "dew", "dhow", "diabetes", "diadem", "diagnosis", "diagram", "dial", "dialect",
        "dialogue", "diam", "diamond", "diaper", "diaphragm", "diarist", "diary", "dibble", "dickey", "dictaphone",
        "dictator", "diction", "dictionary", "die", "diesel", "diet", "difference", "differential", "difficulty",
        "diffuse", "dig", "digestion", "digestive", "digger", "digging", "digit", "dignity", "dilapidation", "dill",
        "dilution", "dime", "dimension", "dimple", "diner", "dinghy", "dining", "dinner", "dinosaur", "dioxide", "dip",
        "diploma", "diplomacy", "dipstick", "direction", "directive", "director", "directory", "dirndl", "dirt",
        "disability", "disadvantage", "disagreement", "disappointment", "disarmament", "disaster", "discharge",
        "discipline", "disclaimer", "disclosure", "disco", "disconnection", "discount", "discourse", "discovery",
        "discrepancy", "discretion", "discrimination", "discussion", "disdain", "disease", "disembodiment",
        "disengagement", "disguise", "disgust", "dish", "dishwasher", "disk", "disparity", "dispatch", "displacement",
        "display", "disposal", "disposer", "disposition", "dispute", "disregard", "disruption", "dissemination",
        "dissonance", "distance", "distinction", "distortion", "distribution", "distributor", "district", "divalent",
        "divan", "diver", "diversity", "divide", "dividend", "divider", "divine", "diving", "division", "divorce",
        "doc", "dock", "doctor", "doctorate", "doctrine", "document", "documentary", "documentation", "doe", "dog",
        "doggie", "dogsled", "dogwood", "doing", "doll", "dollar", "dollop", "dolman", "dolor", "dolphin", "domain",
        "dome", "domination", "donation", "donkey", "donor", "donut", "door", "doorbell", "doorknob", "doorpost",
        "doorway", "dory", "dose", "dot", "double", "doubling", "doubt", "doubter", "dough", "doughnut", "down",
        "downfall", "downforce", "downgrade", "download", "downstairs", "downtown", "downturn", "dozen", "draft",
        "drag", "dragon", "dragonfly", "dragonfruit", "dragster", "drain", "drainage", "drake", "drama", "dramaturge",
        "drapes", "draw", "drawbridge", "drawer", "drawing", "dream", "dreamer", "dredger", "dress", "dresser",
        "dressing", "drill", "drink", "drinking", "drive", "driver", "driveway", "driving", "drizzle", "dromedary",
        "drop", "drudgery", "drug", "drum", "drummer", "drunk", "dryer", "duck", "duckling", "dud", "dude", "due",
        "duel", "dueling", "duffel", "dugout", "dulcimer", "dumbwaiter", "dump", "dump truck", "dune", "dune buggy",
        "dungarees", "dungeon", "duplexer", "duration", "durian", "dusk", "dust", "dust storm", "duster", "duty",
        "dwarf", "dwell", "dwelling", "dynamics", "dynamite", "dynamo", "dynasty", "dysfunction", "e-book", "e-mail",
        "e-reader", "eagle", "eaglet", "ear", "eardrum", "earmuffs", "earnings", "earplug", "earring", "earrings",
        "earth", "earthquake", "earthworm", "ease", "easel", "east", "eating", "eaves", "eavesdropper", "ecclesia",
        "echidna", "eclipse", "ecliptic", "ecology", "economics", "economy", "ecosystem", "ectoderm", "ectodermal",
        "ecumenist", "eddy", "edge", "edger", "edible", "editing", "edition", "editor", "editorial", "education", "eel",
        "effacement", "effect", "effective", "effectiveness", "effector", "efficacy", "efficiency", "effort", "egg",
        "egghead", "eggnog", "eggplant", "ego", "eicosanoid", "ejector", "elbow", "elderberry", "election",
        "electricity", "electrocardiogram", "electronics", "element", "elephant", "elevation", "elevator", "eleventh",
        "elf", "elicit", "eligibility", "elimination", "elite", "elixir", "elk", "ellipse", "elm", "elongation",
        "elver", "email", "emanate", "embarrassment", "embassy", "embellishment", "embossing", "embryo", "emerald",
        "emergence", "emergency", "emergent", "emery", "emission", "emitter", "emotion", "emphasis", "empire", "employ",
        "employee", "employer", "employment", "empowerment", "emu", "enactment", "encirclement", "enclave", "enclosure",
        "encounter", "encouragement", "encyclopedia", "end", "endive", "endoderm", "endorsement", "endothelium",
        "endpoint", "enemy", "energy", "enforcement", "engagement", "engine", "engineer", "engineering", "enigma",
        "enjoyment", "enquiry", "enrollment", "enterprise", "entertainment", "enthusiasm", "entirety", "entity",
        "entrance", "entree", "entrepreneur", "entry", "envelope", "environment", "envy", "enzyme", "epauliere", "epee",
        "ephemera", "ephemeris", "ephyra", "epic", "episode", "epithelium", "epoch", "eponym", "epoxy", "equal",
        "equality", "equation", "equinox", "equipment", "equity", "equivalent", "era", "eraser", "erection", "erosion",
        "error", "escalator", "escape", "escort", "espadrille", "espalier", "essay", "essence", "essential",
        "establishment", "estate", "estimate", "estrogen", "estuary", "eternity", "ethernet", "ethics", "ethnicity",
        "ethyl", "euphonium", "eurocentrism", "evaluation", "evaluator", "evaporation", "eve", "evening",
        "evening-wear", "event", "everybody", "everyone", "everything", "eviction", "evidence", "evil", "evocation",
        "evolution", "ex-husband", "ex-wife", "exaggeration", "exam", "examination", "examiner", "example",
        "exasperation", "excellence", "exception", "excerpt", "excess", "exchange", "excitement", "exclamation",
        "excursion", "excuse", "execution", "executive", "executor", "exercise", "exhaust", "exhaustion", "exhibit",
        "exhibition", "exile", "existence", "exit", "exocrine", "expansion", "expansionism", "expectancy",
        "expectation", "expedition", "expense", "experience", "experiment", "experimentation", "expert", "expertise",
        "explanation", "exploration", "explorer", "explosion", "export", "expose", "exposition", "exposure",
        "expression", "extension", "extent", "exterior", "external", "extinction", "extreme", "extremist", "eye",
        "eyeball", "eyebrow", "eyebrows", "eyeglasses", "eyelash", "eyelashes", "eyelid", "eyelids", "eyeliner",
        "eyestrain", "eyrie", "fabric", "face", "facelift", "facet", "facility", "facsimile", "fact", "factor",
        "factory", "faculty", "fahrenheit", "fail", "failure", "fairness", "fairy", "faith", "faithful", "fall",
        "fallacy", "falling-out", "fame", "familiar", "familiarity", "family", "fan", "fang", "fanlight", "fanny-pack",
        "fantasy", "farm", "farmer", "farming", "farmland", "farrow", "fascia", "fashion", "fat", "fate", "father",
        "father-in-law", "fatigue", "fatigues", "faucet", "fault", "fav", "fava", "favor", "favorite", "fawn", "fax",
        "fear", "feast", "feather", "feature", "fedelini", "federation", "fedora", "fee", "feed", "feedback", "feeding",
        "feel", "feeling", "fellow", "felony", "female", "fen", "fence", "fencing", "fender", "feng", "fennel",
        "ferret", "ferry", "ferryboat", "fertilizer", "festival", "fetus", "few", "fiber", "fiberglass", "fibre",
        "fibroblast", "fibrosis", "ficlet", "fiction", "fiddle", "field", "fiery", "fiesta", "fifth", "fig", "fight",
        "fighter", "figure", "figurine", "file", "filing", "fill", "fillet", "filly", "film", "filter", "filth",
        "final", "finance", "financing", "finding", "fine", "finer", "finger", "fingerling", "fingernail", "finish",
        "finisher", "fir", "fire", "fireman", "fireplace", "firewall", "firm", "first", "fish", "fishbone", "fisherman",
        "fishery", "fishing", "fishmonger", "fishnet", "fisting", "fit", "fitness", "fix", "fixture", "flag", "flair",
        "flame", "flan", "flanker", "flare", "flash", "flat", "flatboat", "flavor", "flax", "fleck", "fledgling",
        "fleece", "flesh", "flexibility", "flick", "flicker", "flight", "flint", "flintlock", "flip-flops", "flock",
        "flood", "floodplain", "floor", "floozie", "flour", "flow", "flower", "flu", "flugelhorn", "fluke", "flume",
        "flung", "flute", "fly", "flytrap", "foal", "foam", "fob", "focus", "fog", "fold", "folder", "folk", "folklore",
        "follower", "following", "fondue", "font", "food", "foodstuffs", "fool", "foot", "footage", "football",
        "footnote", "footprint", "footrest", "footstep", "footstool", "footwear", "forage", "forager", "foray", "force",
        "ford", "forearm", "forebear", "forecast", "forehead", "foreigner", "forelimb", "forest", "forestry", "forever",
        "forgery", "fork", "form", "formal", "formamide", "format", "formation", "former", "formicarium", "formula",
        "fort", "forte", "fortnight", "fortress", "fortune", "forum", "foundation", "founder", "founding", "fountain",
        "fourths", "fowl", "fox", "foxglove", "fraction", "fragrance", "frame", "framework", "fratricide", "fraud",
        "fraudster", "freak", "freckle", "freedom", "freelance", "freezer", "freezing", "freight", "freighter",
        "frenzy", "freon", "frequency", "fresco", "friction", "fridge", "friend", "friendship", "fries", "frigate",
        "fright", "fringe", "fritter", "frock", "frog", "front", "frontier", "frost", "frosting", "frown", "fruit",
        "frustration", "fry", "fuel", "fugato", "fulfillment", "full", "fun", "function", "functionality", "fund",
        "funding", "fundraising", "funeral", "fur", "furnace", "furniture", "furry", "fusarium", "futon", "future",
        "gadget", "gaffe", "gaffer", "gain", "gaiters", "gale", "gall-bladder", "gallery", "galley", "gallon",
        "galoshes", "gambling", "game", "gamebird", "gaming", "gamma-ray", "gander", "gang", "gap", "garage", "garb",
        "garbage", "garden", "garlic", "garment", "garter", "gas", "gasket", "gasoline", "gasp", "gastronomy",
        "gastropod", "gate", "gateway", "gather", "gathering", "gator", "gauge", "gauntlet", "gavel", "gazebo",
        "gazelle", "gear", "gearshift", "geek", "gel", "gelatin", "gelding", "gem", "gemsbok", "gender", "gene",
        "general", "generation", "generator", "generosity", "genetics", "genie", "genius", "genocide", "genre",
        "gentleman", "geography", "geology", "geometry", "geranium", "gerbil", "gesture", "geyser", "gherkin", "ghost",
        "giant", "gift", "gig", "gigantism", "giggle", "ginger", "gingerbread", "ginseng", "giraffe", "girdle", "girl",
        "girlfriend", "git", "glacier", "gladiolus", "glance", "gland", "glass", "glasses", "glee", "glen", "glider",
        "gliding", "glimpse", "globe", "glockenspiel", "gloom", "glory", "glove", "glow", "glucose", "glue", "glut",
        "glutamate", "gnat", "gnu", "go-kart", "goal", "goat", "gobbler", "god", "goddess", "godfather", "godmother",
        "godparent", "goggles", "going", "gold", "goldfish", "golf", "gondola", "gong", "good", "good-bye", "goodbye",
        "goodie", "goodness", "goodnight", "goodwill", "goose", "gopher", "gorilla", "gosling", "gossip", "governance",
        "government", "governor", "gown", "grab-bag", "grace", "grade", "gradient", "graduate", "graduation",
        "graffiti", "graft", "grain", "gram", "grammar", "gran", "grand", "grandchild", "granddaughter", "grandfather",
        "grandma", "grandmom", "grandmother", "grandpa", "grandparent", "grandson", "granny", "granola", "grant",
        "grape", "grapefruit", "graph", "graphic", "grasp", "grass", "grasshopper", "grassland", "gratitude", "gravel",
        "gravitas", "gravity", "gravy", "gray", "grease", "great-grandfather", "great-grandmother", "greatness",
        "greed", "green", "greenhouse", "greens", "grenade", "grey", "grid", "grief", "grill", "grin", "grip",
        "gripper", "grit", "grocery", "ground", "group", "grouper", "grouse", "grove", "growth", "grub", "guacamole",
        "guarantee", "guard", "guava", "guerrilla", "guess", "guest", "guestbook", "guidance", "guide", "guideline",
        "guilder", "guilt", "guilty", "guinea", "guitar", "guitarist", "gum", "gumshoe", "gun", "gunpowder", "gutter",
        "guy", "gym", "gymnast", "gymnastics", "gynaecology", "gyro", "habit", "habitat", "hacienda", "hacksaw",
        "hackwork", "hail", "hair", "haircut", "hake", "half", "half-brother", "half-sister", "halibut", "hall",
        "halloween", "hallway", "halt", "ham", "hamburger", "hammer", "hammock", "hamster", "hand", "hand-holding",
        "handball", "handful", "handgun", "handicap", "handle", "handlebar", "handmaiden", "handover", "handrail",
        "handsaw", "hanger", "happening", "happiness", "harald", "harbor", "harbour", "hard-hat", "hardboard",
        "hardcover", "hardening", "hardhat", "hardship", "hardware", "hare", "harm", "harmonica", "harmonise",
        "harmonize", "harmony", "harp", "harpooner", "harpsichord", "harvest", "harvester", "hash", "hashtag",
        "hassock", "haste", "hat", "hatbox", "hatchet", "hatchling", "hate", "hatred", "haunt", "haven", "haversack",
        "havoc", "hawk", "hay", "haze", "hazel", "hazelnut", "head", "headache", "headlight", "headline", "headphones",
        "headquarters", "headrest", "health", "health-care", "hearing", "hearsay", "heart", "heart-throb", "heartache",
        "heartbeat", "hearth", "hearthside", "heartwood", "heat", "heater", "heating", "heaven", "heavy", "hectare",
        "hedge", "hedgehog", "heel", "heifer", "height", "heir", "heirloom", "helicopter", "helium", "hellcat", "hello",
        "helmet", "helo", "help", "hemisphere", "hemp", "hen", "hepatitis", "herb", "herbs", "heritage", "hermit",
        "hero", "heroine", "heron", "herring", "hesitation", "heterosexual", "hexagon", "heyday", "hiccups", "hide",
        "hierarchy", "high", "high-rise", "highland", "highlight", "highway", "hike", "hiking", "hill", "hint", "hip",
        "hippodrome", "hippopotamus", "hire", "hiring", "historian", "history", "hit", "hive", "hobbit", "hobby",
        "hockey", "hoe", "hog", "hold", "holder", "hole", "holiday", "home", "homeland", "homeownership", "hometown",
        "homework", "homicide", "homogenate", "homonym", "homosexual", "homosexuality", "honesty", "honey", "honeybee",
        "honeydew", "honor", "honoree", "hood", "hoof", "hook", "hop", "hope", "hops", "horde", "horizon", "hormone",
        "horn", "hornet", "horror", "horse", "horseradish", "horst", "hose", "hosiery", "hospice", "hospital",
        "hospitalisation", "hospitality", "hospitalization", "host", "hostel", "hostess", "hotdog", "hotel", "hound",
        "hour", "hourglass", "house", "houseboat", "household", "housewife", "housework", "housing", "hovel",
        "hovercraft", "howard", "howitzer", "hub", "hubcap", "hubris", "hug", "hugger", "hull", "human", "humanity",
        "humidity", "hummus", "humor", "humour", "hunchback", "hundred", "hunger", "hunt", "hunter", "hunting",
        "hurdle", "hurdler", "hurricane", "hurry", "hurt", "husband", "hut", "hutch", "hyacinth", "hybridisation",
        "hybridization", "hydrant", "hydraulics", "hydrocarb", "hydrocarbon", "hydrofoil", "hydrogen", "hydrolyse",
        "hydrolysis", "hydrolyze", "hydroxyl", "hyena", "hygienic", "hype", "hyphenation", "hypochondria",
        "hypothermia", "hypothesis", "ice", "ice-cream", "iceberg", "icebreaker", "icecream", "icicle", "icing", "icon",
        "icy", "id", "idea", "ideal", "identification", "identity", "ideology", "idiom", "igloo", "ignorance",
        "ignorant", "ikebana", "illegal", "illiteracy", "illness", "illusion", "illustration", "image", "imagination",
        "imbalance", "imitation", "immigrant", "immigration", "immortal", "impact", "impairment", "impala",
        "impediment", "implement", "implementation", "implication", "import", "importance", "impostor", "impress",
        "impression", "imprisonment", "impropriety", "improvement", "impudence", "impulse", "in-joke", "in-laws",
        "inability", "inauguration", "inbox", "incandescence", "incarnation", "incense", "incentive", "inch",
        "incidence", "incident", "incision", "inclusion", "income", "incompetence", "inconvenience", "increase",
        "incubation", "independence", "independent", "index", "indication", "indicator", "indigence", "individual",
        "industrialisation", "industrialization", "industry", "inequality", "inevitable", "infancy", "infant",
        "infarction", "infection", "infiltration", "infinite", "infix", "inflammation", "inflation", "influence",
        "influx", "info", "information", "infrastructure", "infusion", "inglenook", "ingrate", "ingredient",
        "inhabitant", "inheritance", "inhibition", "inhibitor", "initial", "initialise", "initialize", "initiative",
        "injunction", "injury", "injustice", "ink", "inlay", "inn", "innervation", "innocence", "innocent",
        "innovation", "input", "inquiry", "inscription", "insect", "insectarium", "insert", "inside", "insight",
        "insolence", "insomnia", "inspection", "inspector", "inspiration", "installation", "instance", "instant",
        "instinct", "institute", "institution", "instruction", "instructor", "instrument", "instrumentalist",
        "instrumentation", "insulation", "insurance", "insurgence", "insurrection", "integer", "integral",
        "integration", "integrity", "intellect", "intelligence", "intensity", "intent", "intention", "intentionality",
        "interaction", "interchange", "interconnection", "intercourse", "interest", "interface", "interferometer",
        "interior", "interject", "interloper", "internet", "interpretation", "interpreter", "interval", "intervenor",
        "intervention", "interview", "interviewer", "intestine", "introduction", "intuition", "invader", "invasion",
        "invention", "inventor", "inventory", "inverse", "inversion", "investigation", "investigator", "investment",
        "investor", "invitation", "invite", "invoice", "involvement", "iridescence", "iris", "iron", "ironclad",
        "irony", "irrigation", "ischemia", "island", "isogloss", "isolation", "issue", "item", "itinerary", "ivory",
        "jack", "jackal", "jacket", "jackfruit", "jade", "jaguar", "jail", "jailhouse", "jalapeño", "jam", "jar",
        "jasmine", "jaw", "jazz", "jealousy", "jeans", "jeep", "jelly", "jellybeans", "jellyfish", "jerk", "jet",
        "jewel", "jeweller", "jewellery", "jewelry", "jicama", "jiffy", "job", "jockey", "jodhpurs", "joey", "jogging",
        "joint", "joke", "jot", "journal", "journalism", "journalist", "journey", "joy", "judge", "judgment", "judo",
        "jug", "juggernaut", "juice", "julienne", "jumbo", "jump", "jumper", "jumpsuit", "jungle", "junior", "junk",
        "junker", "junket", "jury", "justice", "justification", "jute", "kale", "kamikaze", "kangaroo", "karate",
        "kayak", "kazoo", "kebab", "keep", "keeper", "kendo", "kennel", "ketch", "ketchup", "kettle", "kettledrum",
        "key", "keyboard", "keyboarding", "keystone", "kick", "kick-off", "kid", "kidney", "kielbasa", "kill", "killer",
        "killing", "kilogram", "kilometer", "kilt", "kimono", "kinase", "kind", "kindness", "king", "kingdom",
        "kingfish", "kiosk", "kiss", "kit", "kitchen", "kite", "kitsch", "kitten", "kitty", "kiwi", "knee", "kneejerk",
        "knickers", "knife", "knife-edge", "knight", "knitting", "knock", "knot", "know-how", "knowledge", "knuckle",
        "koala", "kohlrabi", "kumquat", "lab", "label", "labor", "laboratory", "laborer", "labour", "labourer", "lace",
        "lack", "lacquerware", "lad", "ladder", "ladle", "lady", "ladybug", "lag", "lake", "lamb", "lambkin", "lament",
        "lamp", "lanai", "land", "landform", "landing", "landmine", "landscape", "lane", "language", "lantern", "lap",
        "laparoscope", "lapdog", "laptop", "larch", "lard", "larder", "lark", "larva", "laryngitis", "lasagna",
        "lashes", "last", "latency", "latex", "lathe", "latitude", "latte", "latter", "laugh", "laughter", "laundry",
        "lava", "law", "lawmaker", "lawn", "lawsuit", "lawyer", "lay", "layer", "layout", "lead", "leader",
        "leadership", "leading", "leaf", "league", "leaker", "leap", "learning", "leash", "leather", "leave", "leaver",
        "lecture", "leek", "leeway", "left", "leg", "legacy", "legal", "legend", "legging", "legislation", "legislator",
        "legislature", "legitimacy", "legume", "leisure", "lemon", "lemonade", "lemur", "lender", "lending", "length",
        "lens", "lentil", "leopard", "leprosy", "leptocephalus", "lesbian", "lesson", "letter", "lettuce", "level",
        "lever", "leverage", "leveret", "liability", "liar", "liberty", "libido", "library", "licence", "license",
        "licensing", "licorice", "lid", "lie", "lieu", "lieutenant", "life", "lifestyle", "lifetime", "lift", "ligand",
        "light", "lighting", "lightning", "lightscreen", "ligula", "likelihood", "likeness", "lilac", "lily", "limb",
        "lime", "limestone", "limit", "limitation", "limo", "line", "linen", "liner", "linguist", "linguistics",
        "lining", "link", "linkage", "linseed", "lion", "lip", "lipid", "lipoprotein", "lipstick", "liquid",
        "liquidity", "liquor", "list", "listening", "listing", "literate", "literature", "litigation", "litmus",
        "litter", "littleneck", "liver", "livestock", "living", "lizard", "llama", "load", "loading", "loaf", "loafer",
        "loan", "lobby", "lobotomy", "lobster", "local", "locality", "location", "lock", "locker", "locket",
        "locomotive", "locust", "lode", "loft", "log", "loggia", "logic", "login", "logistics", "logo", "loincloth",
        "lollipop", "loneliness", "longboat", "longitude", "look", "lookout", "loop", "loophole", "loquat", "lord",
        "loss", "lot", "lotion", "lottery", "lounge", "louse", "lout", "love", "lover", "lox", "loyalty", "luck",
        "luggage", "lumber", "lumberman", "lunch", "luncheonette", "lunchmeat", "lunchroom", "lung", "lunge", "lute",
        "luxury", "lychee", "lycra", "lye", "lymphocyte", "lynx", "lyocell", "lyre", "lyrics", "lysine", "macadamia",
        "macaroni", "macaroon", "macaw", "machine", "machinery", "macrame", "macro", "macrofauna", "madam", "maelstrom",
        "maestro", "magazine", "maggot", "magic", "magnet", "magnitude", "maid", "maiden", "mail", "mailbox", "mailer",
        "mailing", "mailman", "main", "mainland", "mainstream", "maintainer", "maintenance", "maize", "major",
        "major-league", "majority", "makeover", "maker", "makeup", "making", "male", "malice", "mall", "mallard",
        "mallet", "malnutrition", "mama", "mambo", "mammoth", "man", "manacle", "management", "manager", "manatee",
        "mandarin", "mandate", "mandolin", "mangle", "mango", "mangrove", "manhunt", "maniac", "manicure",
        "manifestation", "manipulation", "mankind", "manner", "manor", "mansard", "manservant", "mansion", "mantel",
        "mantle", "mantua", "manufacturer", "manufacturing", "many", "map", "maple", "mapping", "maracas", "marathon",
        "marble", "march", "mare", "margarine", "margin", "mariachi", "marimba", "marines", "marionberry", "mark",
        "marker", "market", "marketer", "marketing", "marketplace", "marksman", "markup", "marmalade", "marriage",
        "marsh", "marshland", "marshmallow", "marten", "marxism", "mascara", "mask", "masonry", "mass", "massage",
        "mast", "master", "masterpiece", "mastication", "mastoid", "mat", "match", "matchmaker", "mate", "material",
        "maternity", "math", "mathematics", "matrix", "matter", "mattock", "mattress", "max", "maximum", "maybe",
        "mayonnaise", "mayor", "meadow", "meal", "mean", "meander", "meaning", "means", "meantime", "measles",
        "measure", "measurement", "meat", "meatball", "meatloaf", "mecca", "mechanic", "mechanism", "med", "medal",
        "media", "median", "medication", "medicine", "medium", "meet", "meeting", "melatonin", "melody", "melon",
        "member", "membership", "membrane", "meme", "memo", "memorial", "memory", "men", "menopause", "menorah",
        "mention", "mentor", "menu", "merchandise", "merchant", "mercury", "meridian", "meringue", "merit",
        "mesenchyme", "mess", "message", "messenger", "messy", "metabolite", "metal", "metallurgist", "metaphor",
        "meteor", "meteorology", "meter", "methane", "method", "methodology", "metric", "metro", "metronome",
        "mezzanine", "microlending", "micronutrient", "microphone", "microwave", "mid-course", "midden", "middle",
        "middleman", "midline", "midnight", "midwife", "might", "migrant", "migration", "mile", "mileage", "milepost",
        "milestone", "military", "milk", "milkshake", "mill", "millennium", "millet", "millimeter", "million",
        "millisecond", "millstone", "mime", "mimosa", "min", "mincemeat", "mind", "mine", "mineral", "mineshaft",
        "mini", "mini-skirt", "minibus", "minimalism", "minimum", "mining", "minion", "minister", "mink", "minnow",
        "minor", "minor-league", "minority", "mint", "minute", "miracle", "mirror", "miscarriage", "miscommunication",
        "misfit", "misnomer", "misogyny", "misplacement", "misreading", "misrepresentation", "miss", "missile",
        "mission", "missionary", "mist", "mistake", "mister", "misunderstand", "miter", "mitten", "mix", "mixer",
        "mixture", "moai", "moat", "mob", "mobile", "mobility", "mobster", "moccasins", "mocha", "mochi", "mode",
        "model", "modeling", "modem", "modernist", "modernity", "modification", "molar", "molasses", "molding", "mole",
        "molecule", "mom", "moment", "monastery", "monasticism", "money", "monger", "monitor", "monitoring", "monk",
        "monkey", "monocle", "monopoly", "monotheism", "monsoon", "monster", "month", "monument", "mood", "moody",
        "moon", "moonlight", "moonscape", "moonshine", "moose", "mop", "morale", "morbid", "morbidity", "morning",
        "moron", "morphology", "morsel", "mortal", "mortality", "mortgage", "mortise", "mosque", "mosquito", "most",
        "motel", "moth", "mother", "mother-in-law", "motion", "motivation", "motive", "motor", "motorboat", "motorcar",
        "motorcycle", "mound", "mountain", "mouse", "mouser", "mousse", "moustache", "mouth", "mouton", "movement",
        "mover", "movie", "mower", "mozzarella", "mud", "muffin", "mug", "mukluk", "mule", "multimedia", "murder",
        "muscat", "muscatel", "muscle", "musculature", "museum", "mushroom", "music", "music-box", "music-making",
        "musician", "muskrat", "mussel", "mustache", "mustard", "mutation", "mutt", "mutton", "mycoplasma", "mystery",
        "myth", "mythology", "nail", "name", "naming", "nanoparticle", "napkin", "narrative", "nasal", "nation",
        "nationality", "native", "naturalisation", "nature", "navigation", "necessity", "neck", "necklace", "necktie",
        "nectar", "nectarine", "need", "needle", "neglect", "negligee", "negotiation", "neighbor", "neighborhood",
        "neighbour", "neighbourhood", "neologism", "neon", "neonate", "nephew", "nerve", "nest", "nestling", "nestmate",
        "net", "netball", "netbook", "netsuke", "network", "networking", "neurobiologist", "neuron", "neuropathologist",
        "neuropsychiatry", "news", "newsletter", "newspaper", "newsprint", "newsstand", "nexus", "nibble", "nicety",
        "niche", "nick", "nickel", "nickname", "niece", "night", "nightclub", "nightgown", "nightingale", "nightlife",
        "nightlight", "nightmare", "ninja", "nit", "nitrogen", "nobody", "nod", "node", "noir", "noise", "nonbeliever",
        "nonconformist", "nondisclosure", "nonsense", "noodle", "noodles", "noon", "norm", "normal", "normalisation",
        "normalization", "north", "nose", "notation", "note", "notebook", "notepad", "nothing", "notice", "notion",
        "notoriety", "nougat", "noun", "nourishment", "novel", "nucleotidase", "nucleotide", "nudge", "nuke", "number",
        "numeracy", "numeric", "numismatist", "nun", "nurse", "nursery", "nursing", "nurture", "nut", "nutmeg",
        "nutrient", "nutrition", "nylon", "nymph", "oak", "oar", "oasis", "oat", "oatmeal", "oats", "obedience",
        "obesity", "obi", "object", "objection", "objective", "obligation", "oboe", "observation", "observatory",
        "obsession", "obsidian", "obstacle", "occasion", "occupation", "occurrence", "ocean", "ocelot", "octagon",
        "octave", "octavo", "octet", "octopus", "odometer", "odyssey", "oeuvre", "off-ramp", "offence", "offense",
        "offer", "offering", "office", "officer", "official", "offset", "oil", "okra", "oldie", "oleo", "olive",
        "omega", "omelet", "omission", "omnivore", "oncology", "onion", "online", "onset", "opening", "opera",
        "operating", "operation", "operator", "ophthalmologist", "opinion", "opium", "opossum", "opponent",
        "opportunist", "opportunity", "opposite", "opposition", "optimal", "optimisation", "optimist", "optimization",
        "option", "orange", "orangutan", "orator", "orchard", "orchestra", "orchid", "order", "ordinary", "ordination",
        "ore", "oregano", "organ", "organisation", "organising", "organization", "organizing", "orient", "orientation",
        "origin", "original", "originality", "ornament", "osmosis", "osprey", "ostrich", "other", "otter", "ottoman",
        "ounce", "outback", "outcome", "outfielder", "outfit", "outhouse", "outlaw", "outlay", "outlet", "outline",
        "outlook", "output", "outrage", "outrigger", "outrun", "outset", "outside", "oval", "ovary", "oven",
        "overcharge", "overclocking", "overcoat", "overexertion", "overflight", "overhead", "overheard", "overload",
        "overnighter", "overshoot", "oversight", "overview", "overweight", "owl", "owner", "ownership", "ox", "oxford",
        "oxygen", "oyster", "ozone", "pace", "pacemaker", "pack", "package", "packaging", "packet", "pad", "paddle",
        "paddock", "pagan", "page", "pagoda", "pail", "pain", "paint", "painter", "painting", "paintwork", "pair",
        "pajamas", "palace", "palate", "palm", "pamphlet", "pan", "pancake", "pancreas", "panda", "panel", "panic",
        "pannier", "panpipe", "pansy", "panther", "panties", "pantologist", "pantology", "pantry", "pants", "pantsuit",
        "panty", "pantyhose", "papa", "papaya", "paper", "paperback", "paperwork", "parable", "parachute", "parade",
        "paradise", "paragraph", "parallelogram", "paramecium", "paramedic", "parameter", "paranoia", "parcel",
        "parchment", "pard", "pardon", "parent", "parenthesis", "parenting", "park", "parka", "parking", "parliament",
        "parole", "parrot", "parser", "parsley", "parsnip", "part", "participant", "participation", "particle",
        "particular", "partner", "partnership", "partridge", "party", "pass", "passage", "passbook", "passenger",
        "passing", "passion", "passive", "passport", "password", "past", "pasta", "paste", "pastor", "pastoralist",
        "pastry", "pasture", "pat", "patch", "pate", "patent", "patentee", "path", "pathogenesis", "pathology",
        "pathway", "patience", "patient", "patina", "patio", "patriarch", "patrimony", "patriot", "patrol", "patroller",
        "patrolling", "patron", "pattern", "patty", "pattypan", "pause", "pavement", "pavilion", "paw", "pawnshop",
        "pay", "payee", "payment", "payoff", "pea", "peace", "peach", "peacoat", "peacock", "peak", "peanut", "pear",
        "pearl", "peasant", "pecan", "pedal", "peek", "peen", "peer", "peer-to-peer", "pegboard", "pelican", "pelt",
        "pen", "penalty", "pence", "pencil", "pendant", "pendulum", "penguin", "penicillin", "peninsula", "pennant",
        "penny", "pension", "pentagon", "peony", "people", "pepper", "pepperoni", "percent", "percentage", "perception",
        "perch", "perennial", "perfection", "performance", "perfume", "period", "periodical", "peripheral",
        "permafrost", "permission", "permit", "perp", "perpendicular", "persimmon", "person", "personal", "personality",
        "personnel", "perspective", "pest", "pet", "petal", "petition", "petitioner", "petticoat", "pew", "pharmacist",
        "pharmacopoeia", "phase", "pheasant", "phenomenon", "phenotype", "pheromone", "philanthropy", "philosopher",
        "philosophy", "phone", "phosphate", "photo", "photodiode", "photograph", "photographer", "photography",
        "photoreceptor", "phrase", "phrasing", "physical", "physics", "physiology", "pianist", "piano", "piccolo",
        "pick", "pickax", "pickaxe", "picket", "pickle", "pickup", "picnic", "picture", "picturesque", "pie", "piece",
        "pier", "piety", "pig", "pigeon", "piglet", "pigpen", "pigsty", "pike", "pilaf", "pile", "pilgrim",
        "pilgrimage", "pill", "pillar", "pillbox", "pillow", "pilot", "pimp", "pimple", "pin", "pinafore", "pince-nez",
        "pine", "pineapple", "pinecone", "ping", "pink", "pinkie", "pinot", "pinstripe", "pint", "pinto", "pinworm",
        "pioneer", "pipe", "pipeline", "piracy", "pirate", "pistol", "pit", "pita", "pitch", "pitcher", "pitching",
        "pith", "pizza", "place", "placebo", "placement", "placode", "plagiarism", "plain", "plaintiff", "plan",
        "plane", "planet", "planning", "plant", "plantation", "planter", "planula", "plaster", "plasterboard",
        "plastic", "plate", "platelet", "platform", "platinum", "platter", "platypus", "play", "player", "playground",
        "playroom", "playwright", "plea", "pleasure", "pleat", "pledge", "plenty", "plier", "pliers", "plight", "plot",
        "plough", "plover", "plow", "plowman", "plug", "plugin", "plum", "plumber", "plume", "plunger", "plywood",
        "pneumonia", "pocket", "pocket-watch", "pocketbook", "pod", "podcast", "poem", "poet", "poetry", "poignance",
        "point", "poison", "poisoning", "poker", "polarisation", "polarization", "pole", "polenta", "police",
        "policeman", "policy", "polish", "politician", "politics", "poll", "polliwog", "pollutant", "pollution", "polo",
        "polyester", "polyp", "pomegranate", "pomelo", "pompom", "poncho", "pond", "pony", "pool", "poor", "pop",
        "popcorn", "poppy", "popsicle", "popularity", "population", "populist", "porcelain", "porch", "porcupine",
        "pork", "porpoise", "port", "porter", "portfolio", "porthole", "portion", "portrait", "position", "possession",
        "possibility", "possible", "post", "postage", "postbox", "poster", "posterior", "postfix", "pot", "potato",
        "potential", "pottery", "potty", "pouch", "poultry", "pound", "pounding", "poverty", "powder", "power",
        "practice", "practitioner", "prairie", "praise", "pray", "prayer", "precedence", "precedent", "precipitation",
        "precision", "predecessor", "preface", "preference", "prefix", "pregnancy", "prejudice", "prelude",
        "premeditation", "premier", "premise", "premium", "preoccupation", "preparation", "prescription", "presence",
        "present", "presentation", "preservation", "preserves", "presidency", "president", "press", "pressroom",
        "pressure", "pressurisation", "pressurization", "prestige", "presume", "pretzel", "prevalence", "prevention",
        "prey", "price", "pricing", "pride", "priest", "priesthood", "primary", "primate", "prince", "princess",
        "principal", "principle", "print", "printer", "printing", "prior", "priority", "prison", "prisoner", "privacy",
        "private", "privilege", "prize", "prizefight", "probability", "probation", "probe", "problem", "procedure",
        "proceedings", "process", "processing", "processor", "proctor", "procurement", "produce", "producer", "product",
        "production", "productivity", "profession", "professional", "professor", "profile", "profit", "progenitor",
        "program", "programme", "programming", "progress", "progression", "prohibition", "project", "proliferation",
        "promenade", "promise", "promotion", "prompt", "pronoun", "pronunciation", "proof", "proof-reader",
        "propaganda", "propane", "property", "prophet", "proponent", "proportion", "proposal", "proposition",
        "proprietor", "prose", "prosecution", "prosecutor", "prospect", "prosperity", "prostacyclin", "prostanoid",
        "prostrate", "protection", "protein", "protest", "protocol", "providence", "provider", "province", "provision",
        "prow", "proximal", "proximity", "prune", "pruner", "pseudocode", "pseudoscience", "psychiatrist",
        "psychoanalyst", "psychologist", "psychology", "ptarmigan", "pub", "public", "publication", "publicity",
        "publisher", "publishing", "pudding", "puddle", "puffin", "pug", "puggle", "pulley", "pulse", "puma", "pump",
        "pumpernickel", "pumpkin", "pumpkinseed", "pun", "punch", "punctuation", "punishment", "pup", "pupa", "pupil",
        "puppet", "puppy", "purchase", "puritan", "purity", "purple", "purpose", "purr", "purse", "pursuit", "push",
        "pusher", "put", "puzzle", "pyramid", "pyridine", "quadrant", "quail", "qualification", "quality", "quantity",
        "quart", "quarter", "quartet", "quartz", "queen", "query", "quest", "question", "questioner", "questionnaire",
        "quiche", "quicksand", "quiet", "quill", "quilt", "quince", "quinoa", "quit", "quiver", "quota", "quotation",
        "quote", "rabbi", "rabbit", "raccoon", "race", "racer", "racing", "racism", "racist", "rack", "radar",
        "radiator", "radio", "radiosonde", "radish", "raffle", "raft", "rag", "rage", "raid", "rail", "railing",
        "railroad", "railway", "raiment", "rain", "rainbow", "raincoat", "rainmaker", "rainstorm", "rainy", "raise",
        "raisin", "rake", "rally", "ram", "rambler", "ramen", "ramie", "ranch", "rancher", "randomisation",
        "randomization", "range", "ranger", "rank", "rap", "rape", "raspberry", "rat", "rate", "ratepayer", "rating",
        "ratio", "rationale", "rations", "raven", "ravioli", "rawhide", "ray", "rayon", "razor", "reach", "reactant",
        "reaction", "read", "reader", "readiness", "reading", "real", "reality", "realization", "realm", "reamer",
        "rear", "reason", "reasoning", "rebel", "rebellion", "reboot", "recall", "recapitulation", "receipt",
        "receiver", "reception", "receptor", "recess", "recession", "recipe", "recipient", "reciprocity", "reclamation",
        "recliner", "recognition", "recollection", "recommendation", "reconsideration", "record", "recorder",
        "recording", "recovery", "recreation", "recruit", "rectangle", "red", "redesign", "redhead", "redirect",
        "rediscovery", "reduction", "reef", "refectory", "reference", "referendum", "reflection", "reform",
        "refreshments", "refrigerator", "refuge", "refund", "refusal", "refuse", "regard", "regime", "region",
        "regionalism", "register", "registration", "registry", "regret", "regulation", "regulator", "rehospitalisation",
        "rehospitalization", "reindeer", "reinscription", "reject", "relation", "relationship", "relative",
        "relaxation", "relay", "release", "reliability", "relief", "religion", "relish", "reluctance", "remains",
        "remark", "reminder", "remnant", "remote", "removal", "renaissance", "rent", "reorganisation", "reorganization",
        "repair", "reparation", "repayment", "repeat", "replacement", "replica", "replication", "reply", "report",
        "reporter", "reporting", "repository", "representation", "representative", "reprocessing", "republic",
        "republican", "reputation", "request", "requirement", "resale", "rescue", "research", "researcher",
        "resemblance", "reservation", "reserve", "reservoir", "reset", "residence", "resident", "residue", "resist",
        "resistance", "resolution", "resolve", "resort", "resource", "respect", "respite", "response", "responsibility",
        "rest", "restaurant", "restoration", "restriction", "restroom", "restructuring", "result", "resume", "retailer",
        "retention", "rethinking", "retina", "retirement", "retouching", "retreat", "retrospect", "retrospective",
        "retrospectivity", "return", "reunion", "revascularisation", "revascularization", "reveal", "revelation",
        "revenant", "revenge", "revenue", "reversal", "reverse", "review", "revitalisation", "revitalization",
        "revival", "revolution", "revolver", "reward", "rhetoric", "rheumatism", "rhinoceros", "rhubarb", "rhyme",
        "rhythm", "rib", "ribbon", "rice", "riddle", "ride", "rider", "ridge", "riding", "rifle", "right", "rim",
        "ring", "ringworm", "riot", "rip", "ripple", "rise", "riser", "risk", "rite", "ritual", "river", "riverbed",
        "rivulet", "road", "roadway", "roar", "roast", "robe", "robin", "robot", "robotics", "rock", "rocker", "rocket",
        "rocket-ship", "rod", "role", "roll", "roller", "romaine", "romance", "roof", "room", "roommate", "rooster",
        "root", "rope", "rose", "rosemary", "roster", "rostrum", "rotation", "round", "roundabout", "route", "router",
        "routine", "row", "rowboat", "rowing", "rubber", "rubbish", "rubric", "ruby", "ruckus", "rudiment", "ruffle",
        "rug", "rugby", "ruin", "rule", "ruler", "ruling", "rum", "rumor", "run", "runaway", "runner", "running",
        "runway", "rush", "rust", "rutabaga", "rye", "sabre", "sac", "sack", "saddle", "sadness", "safari", "safe",
        "safeguard", "safety", "saffron", "sage", "sail", "sailboat", "sailing", "sailor", "saint", "sake", "salad",
        "salami", "salary", "sale", "salesman", "salmon", "salon", "saloon", "salsa", "salt", "salute", "samovar",
        "sampan", "sample", "samurai", "sanction", "sanctity", "sanctuary", "sand", "sandal", "sandbar", "sandpaper",
        "sandwich", "sanity", "sardine", "sari", "sarong", "sash", "satellite", "satin", "satire", "satisfaction",
        "sauce", "saucer", "sauerkraut", "sausage", "savage", "savannah", "saving", "savings", "savior", "saviour",
        "savory", "saw", "saxophone", "scaffold", "scale", "scallion", "scallops", "scalp", "scam", "scanner",
        "scarecrow", "scarf", "scarification", "scenario", "scene", "scenery", "scent", "schedule", "scheduling",
        "schema", "scheme", "schizophrenic", "schnitzel", "scholar", "scholarship", "school", "schoolhouse", "schooner",
        "science", "scientist", "scimitar", "scissors", "scooter", "scope", "score", "scorn", "scorpion", "scotch",
        "scout", "scow", "scrambled", "scrap", "scraper", "scratch", "screamer", "screen", "screening", "screenwriting",
        "screw", "screw-up", "screwdriver", "scrim", "scrip", "script", "scripture", "scrutiny", "sculpting",
        "sculptural", "sculpture", "sea", "seabass", "seafood", "seagull", "seal", "seaplane", "search", "seashore",
        "seaside", "season", "seat", "seaweed", "second", "secrecy", "secret", "secretariat", "secretary", "secretion",
        "section", "sectional", "sector", "security", "sediment", "seed", "seeder", "seeker", "seep", "segment",
        "seizure", "selection", "self", "self-confidence", "self-control", "self-esteem", "seller", "selling",
        "semantics", "semester", "semicircle", "semicolon", "semiconductor", "seminar", "senate", "senator", "sender",
        "senior", "sense", "sensibility", "sensitive", "sensitivity", "sensor", "sentence", "sentencing", "sentiment",
        "sepal", "separation", "septicaemia", "sequel", "sequence", "serial", "series", "sermon", "serum", "serval",
        "servant", "server", "service", "servitude", "sesame", "session", "set", "setback", "setting", "settlement",
        "settler", "severity", "sewer", "sexuality", "shack", "shackle", "shade", "shadow", "shadowbox", "shakedown",
        "shaker", "shallot", "shallows", "shame", "shampoo", "shanty", "shape", "share", "shareholder", "shark", "shaw",
        "shawl", "shear", "shearling", "sheath", "shed", "sheep", "sheet", "shelf", "shell", "shelter", "sherbet",
        "sherry", "shield", "shift", "shin", "shine", "shingle", "ship", "shipper", "shipping", "shipyard", "shirt",
        "shirtdress", "shoat", "shock", "shoe", "shoe-horn", "shoehorn", "shoelace", "shoemaker", "shoes", "shoestring",
        "shofar", "shoot", "shootdown", "shop", "shopper", "shopping", "shore", "shoreline", "short", "shortage",
        "shorts", "shortwave", "shot", "shoulder", "shout", "shovel", "show", "show-stopper", "shower", "shred",
        "shrimp", "shrine", "shutdown", "sibling", "sick", "sickness", "side", "sideboard", "sideburns", "sidecar",
        "sidestream", "sidewalk", "siding", "siege", "sigh", "sight", "sightseeing", "sign", "signal", "signature",
        "signet", "significance", "signify", "signup", "silence", "silica", "silicon", "silk", "silkworm", "sill",
        "silly", "silo", "silver", "similarity", "simple", "simplicity", "simplification", "simvastatin", "sin",
        "singer", "singing", "singular", "sink", "sinuosity", "sip", "sir", "sister", "sister-in-law", "sitar", "site",
        "situation", "size", "skate", "skating", "skean", "skeleton", "ski", "skiing", "skill", "skin", "skirt",
        "skull", "skullcap", "skullduggery", "skunk", "sky", "skylight", "skyline", "skyscraper", "skywalk", "slang",
        "slapstick", "slash", "slate", "slave", "slavery", "slaw", "sled", "sledge", "sleep", "sleepiness", "sleeping",
        "sleet", "sleuth", "slice", "slide", "slider", "slime", "slip", "slipper", "slippers", "slope", "slot", "sloth",
        "slump", "smell", "smelting", "smile", "smith", "smock", "smog", "smoke", "smoking", "smolt", "smuggling",
        "snack", "snail", "snake", "snakebite", "snap", "snarl", "sneaker", "sneakers", "sneeze", "sniffle", "snob",
        "snorer", "snow", "snowboarding", "snowflake", "snowman", "snowmobiling", "snowplow", "snowstorm", "snowsuit",
        "snuck", "snug", "snuggle", "soap", "soccer", "socialism", "socialist", "society", "sociology", "sock", "socks",
        "soda", "sofa", "softball", "softdrink", "softening", "software", "soil", "soldier", "sole", "solicitation",
        "solicitor", "solidarity", "solidity", "soliloquy", "solitaire", "solution", "solvency", "sombrero", "somebody",
        "someone", "someplace", "somersault", "something", "somewhere", "son", "sonar", "sonata", "song", "songbird",
        "sonnet", "soot", "sophomore", "soprano", "sorbet", "sorghum", "sorrel", "sorrow", "sort", "soul", "soulmate",
        "sound", "soundness", "soup", "source", "sourwood", "sousaphone", "south", "southeast", "souvenir",
        "sovereignty", "sow", "soy", "soybean", "space", "spacing", "spade", "spaghetti", "span", "spandex", "spank",
        "sparerib", "spark", "sparrow", "spasm", "spat", "spatula", "spawn", "speaker", "speakerphone", "speaking",
        "spear", "spec", "special", "specialist", "specialty", "species", "specification", "spectacle", "spectacles",
        "spectrograph", "spectrum", "speculation", "speech", "speed", "speedboat", "spell", "spelling", "spelt",
        "spending", "sphere", "sphynx", "spice", "spider", "spiderling", "spike", "spill", "spinach", "spine", "spiral",
        "spirit", "spiritual", "spirituality", "spit", "spite", "spleen", "splendor", "split", "spokesman",
        "spokeswoman", "sponge", "sponsor", "sponsorship", "spool", "spoon", "spork", "sport", "sportsman", "spot",
        "spotlight", "spouse", "sprag", "sprat", "spray", "spread", "spreadsheet", "spree", "spring", "sprinkles",
        "sprinter", "sprout", "spruce", "spud", "spume", "spur", "spy", "spyglass", "square", "squash", "squatter",
        "squeegee", "squid", "squirrel", "stab", "stability", "stable", "stack", "stacking", "stadium", "staff", "stag",
        "stage", "stain", "stair", "staircase", "stake", "stalk", "stall", "stallion", "stamen", "stamina", "stamp",
        "stance", "stand", "standard", "standardisation", "standardization", "standing", "standoff", "standpoint",
        "star", "starboard", "start", "starter", "state", "statement", "statin", "station", "station-wagon",
        "statistic", "statistics", "statue", "status", "statute", "stay", "steak", "stealth", "steam", "steamroller",
        "steel", "steeple", "stem", "stench", "stencil", "step", "step-aunt", "step-brother", "step-daughter",
        "step-father", "step-grandfather", "step-grandmother", "step-mother", "step-sister", "step-son", "step-uncle",
        "stepdaughter", "stepmother", "stepping-stone", "stepson", "stereo", "stew", "steward", "stick", "sticker",
        "stiletto", "still", "stimulation", "stimulus", "sting", "stinger", "stir-fry", "stitch", "stitcher", "stock",
        "stock-in-trade", "stockings", "stole", "stomach", "stone", "stonework", "stool", "stop", "stopsign",
        "stopwatch", "storage", "store", "storey", "storm", "story", "story-telling", "storyboard", "stot", "stove",
        "strait", "strand", "stranger", "strap", "strategy", "straw", "strawberry", "strawman", "stream", "street",
        "streetcar", "strength", "stress", "stretch", "strife", "strike", "string", "strip", "stripe", "strobe",
        "stroke", "structure", "strudel", "struggle", "stucco", "stud", "student", "studio", "study", "stuff",
        "stumbling", "stump", "stupidity", "sturgeon", "sty", "style", "styling", "stylus", "sub", "subcomponent",
        "subconscious", "subcontractor", "subexpression", "subgroup", "subject", "submarine", "submitter", "subprime",
        "subroutine", "subscription", "subsection", "subset", "subsidence", "subsidiary", "subsidy", "substance",
        "substitution", "subtitle", "suburb", "subway", "success", "succotash", "suck", "sucker", "suede", "suet",
        "suffocation", "sugar", "suggestion", "suicide", "suit", "suitcase", "suite", "sulfur", "sultan", "sum",
        "summary", "summer", "summit", "sun", "sunbeam", "sunbonnet", "sundae", "sunday", "sundial", "sunflower",
        "sunglasses", "sunlamp", "sunlight", "sunrise", "sunroom", "sunset", "sunshine", "superiority", "supermarket",
        "supernatural", "supervision", "supervisor", "supper", "supplement", "supplier", "supply", "support",
        "supporter", "suppression", "supreme", "surface", "surfboard", "surge", "surgeon", "surgery", "surname",
        "surplus", "surprise", "surround", "surroundings", "surrounds", "survey", "survival", "survivor", "sushi",
        "suspect", "suspenders", "suspension", "sustainment", "sustenance", "swallow", "swamp", "swan", "swanling",
        "swath", "sweat", "sweater", "sweatshirt", "sweatshop", "sweatsuit", "sweets", "swell", "swim", "swimming",
        "swimsuit", "swine", "swing", "switch", "switchboard", "switching", "swivel", "sword", "swordfight",
        "swordfish", "sycamore", "symbol", "symmetry", "sympathy", "symptom", "syndicate", "syndrome", "synergy",
        "synod", "synonym", "synthesis", "syrup", "system", "t-shirt", "tab", "tabby", "tabernacle", "table",
        "tablecloth", "tablet", "tabletop", "tachometer", "tackle", "taco", "tactics", "tactile", "tadpole", "tag",
        "tail", "tailbud", "tailor", "tailspin", "take-out", "takeover", "tale", "talent", "talk", "talking",
        "tam-o'-shanter", "tamale", "tambour", "tambourine", "tan", "tandem", "tangerine", "tank", "tank-top", "tanker",
        "tankful", "tap", "tape", "tapioca", "target", "taro", "tarragon", "tart", "task", "tassel", "taste", "tatami",
        "tattler", "tattoo", "tavern", "tax", "taxi", "taxicab", "taxpayer", "tea", "teacher", "teaching", "team",
        "teammate", "teapot", "tear", "tech", "technician", "technique", "technologist", "technology", "tectonics",
        "teen", "teenager", "teepee", "telephone", "telescreen", "teletype", "television", "tell", "teller", "temp",
        "temper", "temperature", "temple", "tempo", "temporariness", "temporary", "temptation", "temptress", "tenant",
        "tendency", "tender", "tenement", "tenet", "tennis", "tenor", "tension", "tensor", "tent", "tentacle", "tenth",
        "tepee", "teriyaki", "term", "terminal", "termination", "terminology", "termite", "terrace", "terracotta",
        "terrapin", "terrarium", "territory", "terror", "terrorism", "terrorist", "test", "testament", "testimonial",
        "testimony", "testing", "text", "textbook", "textual", "texture", "thanks", "thaw", "theater", "theft",
        "theism", "theme", "theology", "theory", "therapist", "therapy", "thermals", "thermometer", "thermostat",
        "thesis", "thickness", "thief", "thigh", "thing", "thinking", "thirst", "thistle", "thong", "thongs", "thorn",
        "thought", "thousand", "thread", "threat", "threshold", "thrift", "thrill", "throat", "throne", "thrush",
        "thrust", "thug", "thumb", "thump", "thunder", "thunderbolt", "thunderhead", "thunderstorm", "thyme", "tiara",
        "tic", "tick", "ticket", "tide", "tie", "tiger", "tights", "tile", "till", "tilt", "timbale", "timber", "time",
        "timeline", "timeout", "timer", "timetable", "timing", "timpani", "tin", "tinderbox", "tinkle", "tintype",
        "tip", "tire", "tissue", "titanium", "title", "toad", "toast", "toaster", "tobacco", "today", "toe", "toenail",
        "toffee", "tofu", "tog", "toga", "toilet", "tolerance", "tolerant", "toll", "tom-tom", "tomatillo", "tomato",
        "tomb", "tomography", "tomorrow", "ton", "tonality", "tone", "tongue", "tonic", "tonight", "tool", "toot",
        "tooth", "toothbrush", "toothpaste", "toothpick", "top", "top-hat", "topic", "topsail", "toque", "toreador",
        "tornado", "torso", "torte", "tortellini", "tortilla", "tortoise", "total", "tote", "touch", "tough-guy",
        "tour", "tourism", "tourist", "tournament", "tow-truck", "towel", "tower", "town", "townhouse", "township",
        "toy", "trace", "trachoma", "track", "tracking", "tracksuit", "tract", "tractor", "trade", "trader", "trading",
        "tradition", "traditionalism", "traffic", "trafficker", "tragedy", "trail", "trailer", "trailpatrol", "train",
        "trainer", "training", "trait", "tram", "tramp", "trance", "transaction", "transcript", "transfer",
        "transformation", "transit", "transition", "translation", "transmission", "transom", "transparency",
        "transplantation", "transport", "transportation", "trap", "trapdoor", "trapezium", "trapezoid", "trash",
        "travel", "traveler", "tray", "treasure", "treasury", "treat", "treatment", "treaty", "tree", "trek", "trellis",
        "tremor", "trench", "trend", "triad", "trial", "triangle", "tribe", "tributary", "trick", "trigger",
        "trigonometry", "trillion", "trim", "trinket", "trip", "tripod", "tritone", "triumph", "trolley", "trombone",
        "troop", "trooper", "trophy", "trouble", "trousers", "trout", "trove", "trowel", "truck", "trumpet", "trunk",
        "trust", "trustee", "truth", "try", "tsunami", "tub", "tuba", "tube", "tuber", "tug", "tugboat", "tuition",
        "tulip", "tumbler", "tummy", "tuna", "tune", "tune-up", "tunic", "tunnel", "turban", "turf", "turkey",
        "turmeric", "turn", "turning", "turnip", "turnover", "turnstile", "turret", "turtle", "tusk", "tussle", "tutu",
        "tuxedo", "tweet", "tweezers", "twig", "twilight", "twine", "twins", "twist", "twister", "twitter", "type",
        "typeface", "typewriter", "typhoon", "ukulele", "ultimatum", "umbrella", "unblinking", "uncertainty", "uncle",
        "underclothes", "underestimate", "underground", "underneath", "underpants", "underpass", "undershirt",
        "understanding", "understatement", "undertaker", "underwear", "underweight", "underwire", "underwriting",
        "unemployment", "unibody", "uniform", "uniformity", "union", "unique", "unit", "unity", "universe",
        "university", "update", "upgrade", "uplift", "upper", "upstairs", "upward", "urge", "urgency", "urn", "usage",
        "use", "user", "usher", "usual", "utensil", "utilisation", "utility", "utilization", "vacation", "vaccine",
        "vacuum", "vagrant", "valance", "valentine", "validate", "validity", "valley", "valuable", "value", "vampire",
        "van", "vanadyl", "vane", "vanilla", "vanity", "variability", "variable", "variant", "variation", "variety",
        "vascular", "vase", "vault", "vaulting", "veal", "vector", "vegetable", "vegetarian", "vegetarianism",
        "vegetation", "vehicle", "veil", "vein", "veldt", "vellum", "velocity", "velodrome", "velvet", "vendor",
        "veneer", "vengeance", "venison", "venom", "venti", "venture", "venue", "veranda", "verb", "verdict",
        "verification", "vermicelli", "vernacular", "verse", "version", "vertigo", "verve", "vessel", "vest",
        "vestment", "vet", "veteran", "veterinarian", "veto", "viability", "vibe", "vibraphone", "vibration",
        "vibrissae", "vice", "vicinity", "victim", "victory", "video", "view", "viewer", "vignette", "villa", "village",
        "vine", "vinegar", "vineyard", "vintage", "vintner", "vinyl", "viola", "violation", "violence", "violet",
        "violin", "virginal", "virtue", "virus", "visa", "viscose", "vise", "vision", "visit", "visitor", "visor",
        "vista", "visual", "vitality", "vitamin", "vitro", "vivo", "vixen", "vodka", "vogue", "voice", "void", "vol",
        "volatility", "volcano", "volleyball", "volume", "volunteer", "volunteering", "vomit", "vote", "voter",
        "voting", "voyage", "vulture", "wad", "wafer", "waffle", "wage", "wagon", "waist", "waistband", "wait",
        "waiter", "waiting", "waitress", "waiver", "wake", "walk", "walker", "walking", "walkway", "wall", "wallaby",
        "wallet", "walnut", "walrus", "wampum", "wannabe", "want", "war", "warden", "wardrobe", "warfare", "warlock",
        "warlord", "warm-up", "warming", "warmth", "warning", "warrant", "warren", "warrior", "wasabi", "wash",
        "washbasin", "washcloth", "washer", "washtub", "wasp", "waste", "wastebasket", "wasting", "watch", "watcher",
        "watchmaker", "water", "waterbed", "watercress", "waterfall", "waterfront", "watermelon", "waterskiing",
        "waterspout", "waterwheel", "wave", "waveform", "wax", "way", "weakness", "wealth", "weapon", "wear", "weasel",
        "weather", "web", "webinar", "webmail", "webpage", "website", "wedding", "wedge", "weed", "weeder",
        "weedkiller", "week", "weekend", "weekender", "weight", "weird", "welcome", "welfare", "well", "well-being",
        "west", "western", "wet-bar", "wetland", "wetsuit", "whack", "whale", "wharf", "wheat", "wheel", "whelp",
        "whey", "whip", "whirlpool", "whirlwind", "whisker", "whiskey", "whisper", "whistle", "white", "whole",
        "wholesale", "wholesaler", "whorl", "wick", "widget", "widow", "width", "wife", "wifi", "wild", "wildebeest",
        "wilderness", "wildlife", "will", "willingness", "willow", "win", "wind", "wind-chime", "windage", "window",
        "windscreen", "windshield", "wine", "winery", "wing", "wingman", "wingtip", "wink", "winner", "winter", "wire",
        "wiretap", "wiring", "wisdom", "wiseguy", "wish", "wisteria", "wit", "witch", "witch-hunt", "withdrawal",
        "witness", "wok", "wolf", "woman", "wombat", "wonder", "wont", "wood", "woodchuck", "woodland", "woodshed",
        "woodwind", "wool", "woolens", "word", "wording", "work", "workbench", "worker", "workforce", "workhorse",
        "working", "workout", "workplace", "workshop", "world", "worm", "worry", "worship", "worshiper", "worth",
        "wound", "wrap", "wraparound", "wrapper", "wrapping", "wreck", "wrecker", "wren", "wrench", "wrestler",
        "wriggler", "wrinkle", "wrist", "writer", "writing", "wrong", "xylophone", "yacht", "yahoo", "yak", "yam",
        "yang", "yard", "yarmulke", "yarn", "yawl", "year", "yeast", "yellow", "yellowjacket", "yesterday", "yew",
        "yin", "yoga", "yogurt", "yoke", "yolk", "young", "youngster", "yourself", "youth", "yoyo", "yurt", "zampone",
        "zebra", "zebrafish", "zen", "zephyr", "zero", "ziggurat", "zinc", "zipper", "zither", "zombie", "zone", "zoo",
        "zoologist", "zoology", "zoot-suit", "zucchini",
    ]
