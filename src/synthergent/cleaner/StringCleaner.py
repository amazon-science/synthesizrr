from typing import *
import math, ray, re, multiprocessing as mp
from abc import ABC, abstractmethod
import pandas as pd
from nltk import word_tokenize
from pydantic import root_validator, conint, confloat, constr, Extra
from pydantic.typing import Literal
from sklearn.model_selection import train_test_split
from synthergent.base.constants import FileFormat, DataLayout, DataSplit, Parallelize
from synthergent.base.data import FileMetadata, Reader, Writer
from synthergent.base.data.reader import DataFrameReader
from synthergent.base.framework.ray_base import ActorComposite
from synthergent.base.framework.chain.Chain import Step
from synthergent.base.data.sdf import ScalableDataFrame, ScalableDataFrameRawType, DaskDataFrame
from synthergent.base.framework.metric import Metric, Metrics
from synthergent.base.framework.task.classification import ClassificationData
from synthergent.base.framework.task.text_generation import TextGenerationsPredictionsBase, GENERATED_TEXTS_COL
from synthergent.base.framework.task_data import Dataset
from synthergent.base.util import Parameters, as_list, as_set, \
    safe_validate_arguments, get_default, StringUtil, AutoEnum, auto, alias, not_impl, \
    binary_search, ProgressBar, shuffle_items, only_item, irange, punct_normalize, remove_nulls, all_are_none, \
    dispatch, Timer
from synthergent.base.util.concurrency import accumulate
from synthergent.cleaner.Cleaner import Cleaner


class StringCleaner(Cleaner):
    class Params(Cleaner.Params):
        cleaner: Callable
        col: Union[List[constr(min_length=1)], constr(min_length=1)]

        @root_validator(pre=True)
        def _set_string_cleaner_params(cls, params: Dict) -> Dict:
            params['col'] = as_list(params['col'])
            return params

    def clean(
            self,
            data: pd.DataFrame,
            **kwargs,
    ) -> pd.DataFrame:
        for col in self.params.col:
            data[col] = data[col].apply(self.params.cleaner)
        return data
