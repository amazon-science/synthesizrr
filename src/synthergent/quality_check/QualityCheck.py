from typing import *
import math, ray, re, multiprocessing as mp
from abc import ABC, abstractmethod
import pandas as pd
from nltk import word_tokenize
from pydantic import root_validator, conint, confloat, Extra
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
    dispatch, Timer, Executor
from synthergent.base.util.concurrency import accumulate
from synthergent.config import ScalingConfig


class QualityCheck(Step, ABC):
    class Params(Parameters):
        """
        BaseModel for parameters. Expected to be overridden by subclasses.
        """
        num_cpus: int = 1
        num_gpus: int = 0
        display_exclude: Tuple[str, ...] = ('num_cpus', 'num_gpus')

        class Config(Parameters.Config):
            ## Allow extra keyword parameters to be used when initializing the class.
            extra = Extra.forbid

    params: Params = {}

    @root_validator(pre=True)
    def convert_params(cls, params: Dict) -> Dict:
        params['params'] = cls._convert_params(cls.Params, params.get('params'))
        return params

    @abstractmethod
    def evaluate(
            self,
            data: pd.DataFrame,
            scaling: ScalingConfig,
            executor: Optional[Executor],
            **kwargs,
    ) -> pd.DataFrame:
        pass

    @safe_validate_arguments
    def run(
            self,
            *,
            data: Any,
            scaling: ScalingConfig,
            executor: Optional[Executor],
            step_i: int,
            num_steps: int,
            **kwargs,
    ) -> Dict:
        if isinstance(data, (str, dict, FileMetadata)):
            data: FileMetadata = FileMetadata.of(data)

            available_df_readers: Set[FileFormat] = set()
            for df_reader_cls in DataFrameReader.subclasses():
                assert issubclass(df_reader_cls, DataFrameReader)
                available_df_readers |= as_set(df_reader_cls.file_formats)
            available_df_readers_str: str = StringUtil.join_human([x.lower() for x in available_df_readers])
            if data.format is None:
                raise ValueError(
                    'Unable to determine format for data. '
                    '\n- If passing a folder path, please pass a dict as: {"path": "/path/to/data/", "format": "parquet"}'
                    '\n- If passing a file path, please pass a dict as: {"path": "/path/to/data.parquet", "format": "parquet"}'
                    f'\n Available formats are: {available_df_readers_str}'
                )
            elif data.format not in available_df_readers:
                raise ValueError(
                    f'Unsupported format to read data: {data.format}\n'
                    f'Available formats are: {available_df_readers_str}'
                )
        result: pd.DataFrame = self.evaluate(
            data=data,
            scaling=scaling,
            executor=executor,
            **kwargs
        )
        params_str: str = StringUtil.stringify(self.params.dict(
            exclude=as_list([
                'run_fn_spec', 'input_aliases', 'tracker', 'verbosity', 'display_exclude'
            ]) + as_list(self.params.display_exclude)
        ))
        output_key: str = f'''{self.class_name}({params_str})'''
        return {
            output_key: result,
        }
