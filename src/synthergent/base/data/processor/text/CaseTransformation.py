from typing import *
from synthergent.base.data.processor import SingleColumnProcessor, TextInputProcessor, TextOutputProcessor
import pandas as pd
from synthergent.base.util import AutoEnum, auto, is_null


class Case(AutoEnum):
    UPPER = auto()
    LOWER = auto()


class CaseTransformation(SingleColumnProcessor, TextInputProcessor, TextOutputProcessor):
    """
    Transforms the text case to uppercase or lowercase.

    Params:
    - CASE: must be the string 'upper' or 'lower'.
    """

    class Params(SingleColumnProcessor.Params):
        case: Case = Case.LOWER

    def transform_single(self, data: Optional[str]) -> Optional[str]:
        if is_null(data):
            return None
        if self.params.case is Case.LOWER:
            return data.lower()
        elif self.params.case is Case.UPPER:
            return data.upper()
        raise NotImplementedError(f'Unsupported case: {self.params.case}')
