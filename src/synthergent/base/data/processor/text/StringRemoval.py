from typing import *
from synthergent.base.data.processor import SingleColumnProcessor, TextInputProcessor, TextOutputProcessor
import pandas as pd
from synthergent.base.util import is_list_like, is_null
from pydantic import validator


class StringRemoval(SingleColumnProcessor, TextInputProcessor, TextOutputProcessor):
    """
    Removes certain strings from each text string using str.replace() (no regex matching).

    Params:
    - REMOVAL_LIST: the list of strings to remove.
    """

    class Params(SingleColumnProcessor.Params):
        removal_list: List[str]

        @validator('removal_list')
        def check_removal_list(cls, removal_list: List):
            if len(removal_list) == 0 or not is_list_like(removal_list):
                raise ValueError(f'`removal_list` should be a non-empty list of strings')
            return list(removal_list)

    def transform_single(self, data: Optional[str]) -> Optional[str]:
        if is_null(data):
            return None
        for s in self.params.removal_list:
            data: str = data.replace(s, '')
        return data
