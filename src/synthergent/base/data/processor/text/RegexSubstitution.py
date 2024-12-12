from typing import *
import pandas as pd
import re
from synthergent.base.data.processor import SingleColumnProcessor, TextInputProcessor, TextOutputProcessor
from synthergent.base.util import is_null
from pydantic import root_validator, constr


class RegexSubstitution(SingleColumnProcessor, TextInputProcessor, TextOutputProcessor):
    """
    Replaces each matched regex pattern in a list with the corresponding substitution pattern.

    Params:
    - SUBSTITUTION_LIST: a list of 2-tuples, where the first element is the regex to match and the second is the
        substitution (which might be string or regex, controllable via SUBSTITUTE_IS_REGEX).
        This list of substitutions will be applied on the input text sequentially.
    - IGNORECASE: whether to ignore case during regex matching.
    - MULTILINE: whether to do multiline mathcing during regex matching.
    - SUBSTITUTE_IS_REGEX: whether the substitution is a regex expression. If set to True, the transformer will compile
        the substitution as regex during replacement, allowing usage of capturing groups etc.
    """

    class Params(SingleColumnProcessor.Params):
        substitution_list: List[Tuple[constr(min_length=1), constr(min_length=0)]]
        ignorecase: bool = False
        multiline: bool = True
        substitute_is_regex: bool = True
        flags: Optional[int] = None
        match_patterns: Dict[constr(min_length=1), Any] = None

        @root_validator(pre=False)
        def set_flags(cls, params):
            flags = 0
            if params['ignorecase']:
                flags |= re.IGNORECASE
            if params['multiline']:
                flags |= re.MULTILINE
            params['flags'] = flags
            params['match_patterns'] = {
                regex_pattern: re.compile(regex_pattern, flags=flags)
                for regex_pattern, _ in params['substitution_list']
            }
            return params

    def transform_single(self, data: Optional[str]) -> Optional[str]:
        if is_null(data):
            return None
        for regex_pattern, sub_str in self.params.substitution_list:
            match_pattern = self.params.match_patterns[regex_pattern]
            sub_pattern = sub_str if not self.params.substitute_is_regex else r'%s' % (sub_str)
            data: str = match_pattern.sub(sub_pattern, data)
        return data
