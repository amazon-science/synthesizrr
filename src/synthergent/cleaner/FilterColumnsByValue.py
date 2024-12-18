from typing import *
import pandas as pd
from synthergent.base.util import as_list, as_set
from synthergent.cleaner.Cleaner import Cleaner
from pydantic import root_validator


class FilterColumnsByValue(Cleaner):
    class Params(Cleaner.Params):
        exact_match: Dict[str, List] = {}
        contains: Dict[str, List] = {}

        @root_validator(pre=False)
        def _check_params(cls, params: Dict) -> Dict:
            if len(params['exact_match']) == len(params['contains']) == 0:
                raise ValueError('You must pass at least one parameter')
            return params

    def clean(
            self,
            data: pd.DataFrame,
            **kwargs,
    ) -> pd.DataFrame:
        query_str: List = []

        if len(self.params.exact_match) > 0:
            exact_match_query_str: List = []
            for col, values in self.params.exact_match.items():
                values = as_list(as_set(values))
                exact_match_query_str.append(f'({col} in {values})')
            exact_match_query_str: str = '(' + ' and '.join(exact_match_query_str).strip() + ')'
            query_str.append(exact_match_query_str.strip())

        if len(self.params.contains) > 0:
            contains_query_str: List = []
            for col, values in self.params.contains.items():
                col_contains_query_str: List = []
                for value in as_list(as_set(values)):
                    col_contains_query_str.append(f'{col}.str.contains("{value}", case=False, na=False)')
                col_contains_query_str: str = '(' + ' or '.join(col_contains_query_str).strip() + ')'
                contains_query_str.append(col_contains_query_str)
            contains_query_str: str = '(' + ' and '.join(contains_query_str).strip() + ')'
            query_str.append(contains_query_str.strip())
        query_str: str = ' and '.join(query_str).strip()
        if len(query_str) == 0:
            raise ValueError('Created empty query string')
        return data.query(query_str)
