from typing import *
from synthesizrr.base.data.processor import SingleColumnProcessor, TextInputProcessor, TextOutputProcessor
from synthesizrr.base.util import is_null
import pandas as pd
import re


class HtmlTagRemoval(SingleColumnProcessor, TextInputProcessor, TextOutputProcessor):
    """
    Removes HTML tags from the text. Leaves the content between tags untouched.
    An HTML tag is recognized as anything between a pair of crocodile brackets, e.g. <p>, < p>, < p >, < /p html >, etc.
    """
    HTML_REGEX: ClassVar = re.compile(u'<.*?>')

    def transform_single(self, data: Optional[str]) -> Optional[str]:
        if is_null(data):
            return None
        return self.HTML_REGEX.sub('', data)
