from typing import *
from synthergent.base.data.processor import SingleColumnProcessor, TextInputProcessor, TextOutputProcessor
from synthergent.base.util import AutoEnum, auto, StringUtil, is_null
import pandas as pd
import string
from pydantic import constr


class PunctuationCleaner(SingleColumnProcessor, TextInputProcessor, TextOutputProcessor):
    """
    Replaces punctuations with spaces.
    """

    class Params(SingleColumnProcessor.Params):
        replacement_char: constr(min_length=1) = StringUtil.SPACE

    def transform_single(self, data: Optional[str]) -> Optional[str]:
        if is_null(data):
            return None
        return data.translate(
            str.maketrans(
                string.punctuation,
                self.params.replacement_char * len(string.punctuation)
            )
        )
