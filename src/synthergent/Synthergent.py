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
from synthergent.base.data.writer import DataFrameWriter
from synthergent.base.framework.ray_base import ActorComposite
from synthergent.base.framework.chain.Chain import Step, Chain, ChainExecution, ParallelMap, ChainStep
from synthergent.base.data.sdf import ScalableDataFrame, ScalableDataFrameRawType
from synthergent.base.framework.metric import Metric, Metrics
from synthergent.base.framework.task.classification import ClassificationData
from synthergent.base.framework.task.text_generation import TextGenerationsPredictionsBase, GENERATED_TEXTS_COL
from synthergent.base.framework.task_data import Dataset
from synthergent.base.util import Parameters, as_list, \
    safe_validate_arguments, dispatch_executor, get_default, StringUtil, AutoEnum, auto, alias, not_impl, \
    binary_search, ProgressBar, shuffle_items, only_item, irange, punct_normalize, remove_nulls, all_are_none, \
    Executor, dispatch, Timer, remove_keys
from synthergent.base.util.concurrency import accumulate
from pydantic import constr
from typing import *
import types, warnings, logging, threading
from abc import ABC, abstractmethod
import numpy as np
import time, traceback, pickle, gc, os, json
from synthergent.base.util import Registry, MutableParameters, Parameters, set_param_from_alias, as_set, as_list, \
    random_sample, safe_validate_arguments, format_exception_msg, StringUtil, get_fn_spec, Timer, type_str, \
    run_concurrent, get_result, Future, get_default, FunctionSpec, dispatch, is_function, Log, ThreadPoolExecutor, \
    ProgressBar, stop_executor, only_item, worker_ids, dispatch_executor, is_list_like, only_key
from synthergent.base.data import FileMetadata, ScalableDataFrame, ScalableSeries, Asset, ScalableDataFrameOrRaw
from synthergent.base.constants import Status, Parallelize, Alias, COMPLETED_STATUSES
from synthergent.base.util.notify import Notifier
from synthergent.base.framework.tracker import Tracker
from functools import partial
from pydantic import root_validator, Extra, conint, Field, constr, confloat
from pydantic.typing import Literal
from datetime import datetime
from synthergent.config import ScalingConfig
from synthergent.FinalStep import FinalStep
from synthergent.cleaner.Cleaner import Cleaner
from synthergent.quality_check.QualityCheck import QualityCheck
from synthergent.distillation.Distillation import Distillation


class Synthergent(Chain):
    @root_validator(pre=False)
    def _check_steps(cls, params: Dict) -> Dict:
        num_steps: int = len(params['steps'])
        for step_i, step in enumerate(params['steps']):
            if isinstance(step, ChainStep) and isinstance(step.chain, FinalStep):
                if step_i != num_steps - 1:
                    raise ValueError(f'When creating a FinalStep, it must be the last step.')
                for _nested_step in step.chain.steps:
                    if not isinstance(_nested_step, (QualityCheck, Distillation)):
                        raise ValueError(
                            f'Can only have QualityCheck and Distillation in FinalStep.of(); '
                            f'found: {type_str(_nested_step)}'
                        )
            if isinstance(step, QualityCheck):
                raise ValueError(f'Cannot have QualityCheck as an individual step in pipeline, it must be in FinalStep.of()')
            if isinstance(step, Distillation):
                raise ValueError(f'Cannot have Distillation as an individual step in pipeline, it must be in FinalStep.of()')
        return params

    @safe_validate_arguments
    def run(
            self,
            *args,
            scaling: ScalingConfig = ScalingConfig(
                batch_size=None,
                partition_size=None,
                parallelize=Parallelize.sync,
                max_workers=max(1, min(mp.cpu_count() - 1, 16)),  ## Default: 1-6 processes
            ),
            save: Optional[Union[FileMetadata, Dict, str]] = None,
            return_data_on_save: bool = False,
            verbosity: int = 1,
            return_exn: bool = False,
            **kwargs,
    ) -> Optional[Union[pd.DataFrame, Dict, ChainExecution]]:
        kwargs['exn_name'] = 'Synthergent'
        kwargs['background'] = False
        kwargs['scaling'] = scaling
        kwargs['verbosity'] = verbosity

        if save is not None:
            save: FileMetadata = FileMetadata.of(save)
            available_df_writers: Set[FileFormat] = set()
            for df_writer_cls in DataFrameWriter.subclasses():
                assert issubclass(df_writer_cls, DataFrameWriter)
                available_df_writers |= as_set(df_writer_cls.file_formats)
            available_df_writers_str: str = StringUtil.join_human([x.lower() for x in available_df_writers])
            if save.format is None:
                raise ValueError(
                    'Unable to determine format in which to save data.\n'
                    '- If passing a folder path, please pass a dict as: {"path": "/path/to/data/", "format": "parquet"}\n'
                    '- If passing a file path, please pass a dict as: {"path": "/path/to/data.parquet", "format": "parquet"}\n'
                    f'Available formats are: {available_df_writers_str}'
                )
            elif save.format not in available_df_writers:
                raise ValueError(
                    f'Unsupported format to write data: {save.format}\n'
                    f'Available formats are: {available_df_writers_str}'
                )

        executor: Optional[Executor] = None
        try:
            if scaling.parallelize in {Parallelize.sync, Parallelize.threads, Parallelize.processes}:
                ## Run locally:
                if executor is None:
                    executor: Optional[Executor] = dispatch_executor(
                        parallelize=scaling.parallelize,
                        max_workers=scaling.max_workers,
                    )
            elif scaling.parallelize in {Parallelize.ray}:
                ## Run using Dask-on-Ray:
                pass
            else:
                raise NotImplementedError(f'Unsupported value: `scaling` = {scaling}')

            kwargs['executor'] = executor
            exn: ChainExecution = super(Synthergent, self).run(*args, **kwargs)
            if save is not None:
                writer: Writer = Writer.of(
                    save.format,
                    num_rows={Parallelize.ray: None}.get(scaling.parallelize, scaling.batch_size),
                )
                writer.write(
                    data=exn.outputs['data'],
                    destination=save,
                    file_name='part',
                )
                if return_data_on_save is False:
                    print(f'Saved to: "{save.path}"')
                    if return_exn:
                        return exn
                    else:
                        return None
            if return_exn:
                return exn
            outputs = {}
            if 'final_step_results' in exn.outputs:
                outputs = {
                    **remove_keys(exn.outputs, ['final_step_results']),
                    **exn.outputs['final_step_results'],
                }
            outputs['data']: pd.DataFrame = ScalableDataFrame.of(outputs['data']).pandas()
            if len(outputs) == 1 and only_key(outputs) == 'data':
                return outputs['data']
            return outputs
        finally:
            stop_executor(executor)
