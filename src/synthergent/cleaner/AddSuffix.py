from typing import *
import pandas as pd
from synthergent.cleaner.Cleaner import Cleaner
from pydantic import root_validator


class AddSuffix(Cleaner):
    class Params(Cleaner.Params):
        reps: int = 1
        col: str = 'generations'
        suffix: str = ' lol'

    def clean(
            self,
            data: pd.DataFrame,
            **kwargs,
    ) -> pd.DataFrame:
        for _ in range(self.params.reps):
            data[self.params.col] = data[self.params.col].astype(str) + self.params.suffix
        return data
