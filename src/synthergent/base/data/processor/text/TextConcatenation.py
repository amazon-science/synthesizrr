from typing import *
from synthergent.base.data.processor import Nto1ColumnProcessor, TextInputProcessor, TextOutputProcessor
import pandas as pd
from synthergent.base.data.sdf import ScalableDataFrame, ScalableSeries
from synthergent.base.util import AutoEnum, auto, StringUtil, get_default, is_list_like, is_null
from pydantic import root_validator, validator, constr


class ColumnOrder(AutoEnum):
    SORT_BY_NAME_ASCENDING = auto()
    SORT_BY_NAME_DESCENDING = auto()
    SORT_BY_SHORTEST_FIRST = auto()
    INPUT_ORDER = auto()


class TextConcatenation(Nto1ColumnProcessor, TextInputProcessor, TextOutputProcessor):
    """
    Concatenates text from multiple columns into a single column.
    For non-text columns, converts to string and then concatenates.

    Params:
    - SEP: the separator between columns in the combined text string.
    - COLUMN_ORDER: which way to order columns.
    """

    class Params(Nto1ColumnProcessor.Params):
        sep: constr(min_length=1) = StringUtil.SPACE
        column_order: ColumnOrder = ColumnOrder.SORT_BY_NAME_ASCENDING  ## Do not change this for legacy reasons.
        input_ordering: Optional[List[str]] = None
        prefix_col_name: Optional[bool] = False
        prefix_col_sep: Optional[str] = ": "
        allow_missing: Optional[bool] = False

    ordered_cols: Optional[List[str]] = None

    @root_validator(pre=False)
    def set_ordered_cols(cls, params: Dict):
        if params['params'].column_order is ColumnOrder.INPUT_ORDER:
            if not is_list_like(params['params'].input_ordering):
                raise ValueError(
                    f'`input_ordering` must be a non-empty list when column_order={ColumnOrder.INPUT_ORDER}'
                )
            params['ordered_cols']: List[str] = params['params'].input_ordering
        return params

    def _fit_df(self, data: ScalableDataFrame):
        cols: List[str] = list(data.columns)
        if self.params.column_order is ColumnOrder.SORT_BY_SHORTEST_FIRST:
            avg_column_length: Dict[str, float] = {
                col: data[col].dropna().astype(str).apply(len).mean()
                for col in cols
            }
            ## Sort first by avg. length, then by column name:
            self.ordered_cols: List[str] = [
                col for col, avg_len in sorted(list(avg_column_length.items()), key=lambda x: (x[1], x[0]))
            ]
        elif self.params.column_order is ColumnOrder.SORT_BY_NAME_DESCENDING:
            self.ordered_cols: List[str] = sorted(cols, reverse=True)
        elif self.params.column_order is ColumnOrder.SORT_BY_NAME_ASCENDING:
            self.ordered_cols: List[str] = sorted(cols)
        elif self.params.column_order is ColumnOrder.INPUT_ORDER:
            self.ordered_cols: List[str] = self.params.input_ordering
        else:
            self.ordered_cols = None

    def _transform_single(self, data: List[Any]) -> str:
        """Concatanate a list of data of any type"""
        return self.params.sep.join([str(x) for x in data if not is_null(x)])

    def _transform_df(self, data: ScalableDataFrame) -> ScalableSeries:
        if self.ordered_cols is None:
            raise self.FitBeforeTransformError
        output_series: Optional[ScalableSeries] = None
        for col in self.ordered_cols:
            if col not in data.columns_set:
                if self.params.allow_missing:
                    continue
                raise ValueError(f'Column {col} is required but not found in input data. Input data has columns: {data.columns}')
            to_add_col = col + self.params.prefix_col_sep
            if self.params.prefix_col_name is False:
                to_add_col = ""
            if output_series is None:
                output_series: ScalableSeries = to_add_col + data[col].fillna(StringUtil.EMPTY).astype(str)
            else:
                output_series += self.params.sep + to_add_col + data[col].fillna(StringUtil.EMPTY).astype(str)
        return output_series
