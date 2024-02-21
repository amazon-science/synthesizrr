from typing import *
from synthesizrr.base.data.processor import SingleColumnProcessor, TextOrLabelInputProcessor, EncodedLabelOutputProcessor
import pandas as pd
from synthesizrr.base.util import AutoEnum, auto, StringUtil, is_null
from synthesizrr.base.data.sdf import ScalableSeries
from pydantic import constr


class LabelAffix(SingleColumnProcessor, TextOrLabelInputProcessor, EncodedLabelOutputProcessor):
    """
    Adds a suffix or prefix (or both) to a label.

    Params:
    - PREFIX: option prefix to the label
    - SUFFIX: option suffix to the label
    """

    class Params(SingleColumnProcessor.Params):
        prefix: constr(min_length=0) = ''
        suffix: constr(min_length=0) = ''

    # def _transform_series(self, data: ScalableSeries) -> ScalableSeries:
    #     nulls: ScalableSeries = data.isna()
    #     data = self.params.prefix + data.fillna('').astype(str) + self.params.suffix
    #     data[nulls] = None
    #     return data

    def transform_single(self, data: Optional[Any]) -> Optional[str]:
        if is_null(data):
            return None
        return self.params.prefix + str(data) + self.params.suffix
