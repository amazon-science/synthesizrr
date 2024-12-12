from typing import *

import numpy as np

from synthergent.base.util import AutoEnum, auto, alias, optional_dependency, as_list
from synthergent.base.constants.MLConstants import MLType
import csv

DEFAULT_RANDOM_SEED: int = 42  ## https://en.wikipedia.org/wiki/42_(number)#The_Hitchhiker's_Guide_to_the_Galaxy


class DataLayout(AutoEnum):
    DATUM = auto()
    LIST_OF_DICT = auto()  ## List dicts with various columns (sparse storage). Fast row-wise access.
    DICT = auto()  ## Single Dict with Numpy Arrays or Tensorts for columns (dense storage). Fast column-wise access.
    RECORD = auto()  ## Single Dict with Numpy Arrays or Tensorts for columns (dense storage). Fast column-wise access.
    NUMPY = auto()  ## Numpy array (dense storage). Useful for row-wise access.
    TORCH = auto()
    TENSORFLOW = auto()
    JAX = auto()
    NUMPY_RECORD_ARRAY = auto()  ## Numpy array of tuples (dense storage). Fast row-wise access.
    PANDAS = auto()  ## Numpy array with extra metadata (dense storage). Fast row-wise or column-wise access.
    DASK = auto()  ## Lazily-evaluated DataFrame (dense storage). Fast column-wise access.


SDF_DATA_LAYOUT_PRIORITY: List[DataLayout] = [
    ## Do not include DataLayout.RECORD in this.
    DataLayout.DICT,
    DataLayout.LIST_OF_DICT,
    DataLayout.PANDAS,
    DataLayout.DASK,
]
LAZY_SDF_DATA_LAYOUTS: List[DataLayout] = [
    DataLayout.DASK,
]

SS_DATA_LAYOUT_PRIORITY: List[DataLayout] = [
    DataLayout.NUMPY,
    DataLayout.PANDAS,
    DataLayout.DASK,
]

TENSOR_SS_DATA_LAYOUT_PRIORITY: List[DataLayout] = [
    DataLayout.TORCH,
    DataLayout.TENSORFLOW,
    DataLayout.JAX,
]

AVAILABLE_TENSOR_TYPES: Dict[DataLayout, Type] = {
    DataLayout.NUMPY: np.ndarray
}
with optional_dependency('torch'):
    import torch

    AVAILABLE_TENSOR_TYPES[DataLayout.TORCH] = torch.Tensor

with optional_dependency('tensorflow'):
    import tensorflow as tf
    from tensorflow import keras as ks

    AVAILABLE_TENSOR_TYPES[DataLayout.TENSORFLOW] = tf.Tensor

with optional_dependency('jax', 'flax'):
    import jax
    import jax.numpy as jnp
    import flax.linen as nn

    AVAILABLE_TENSOR_TYPES[DataLayout.JAX] = jnp.ndarray

AVAILABLE_DEEP_LEARNING_PACKAGES: Set[DataLayout] = set(AVAILABLE_TENSOR_TYPES.keys())

TENSOR_LAYOUT_TO_SHORTHAND_MAP: Dict[DataLayout, List[str]] = {
    DataLayout.NUMPY: ['np', 'numpy'],
    DataLayout.TORCH: ['pt', 'torch', 'pytorch'],
    DataLayout.TENSORFLOW: ['tf', 'tensorflow'],
    DataLayout.JAX: ['jax'],
}
TensorShortHand = Literal['np', 'numpy', 'pt', 'torch', 'pytorch', 'tf', 'tensorflow', 'jax']

SHORTHAND_TO_TENSOR_LAYOUT_MAP: Dict[str, DataLayout] = {}
for tensor_layout, shorthand in TENSOR_LAYOUT_TO_SHORTHAND_MAP.items():
    for sh in as_list(shorthand):
        if sh in SHORTHAND_TO_TENSOR_LAYOUT_MAP:
            raise ValueError(f'Cannot have duplicate file-ending keys: {sh}')
        SHORTHAND_TO_TENSOR_LAYOUT_MAP[sh] = tensor_layout


class ProcessingMode(AutoEnum):
    TRANSFORM = auto()
    FIT_TRANSFORM = auto()
    ZIPPING = auto()
    TRANSFORM_SINGLE_ROW = auto()

    def get_data_layout(self):
        return DataLayout.RECORD if self.name is ProcessingMode.TRANSFORM_SINGLE_ROW else None


class MissingColumnBehavior(AutoEnum):
    ERROR = auto()
    SKIP = auto()
    EXECUTE = auto()


class Parallelize(AutoEnum):
    asyncio = alias('async', 'asynchronous')
    sync = alias('synchronous')
    threads = alias('thread')
    processes = alias('proc', 'process')
    ray = auto()


QUOTING_MAP: Dict = {
    'quote_none': csv.QUOTE_NONE,
    csv.QUOTE_NONE: csv.QUOTE_NONE,
    'quote_minimal': csv.QUOTE_MINIMAL,
    csv.QUOTE_MINIMAL: csv.QUOTE_MINIMAL,
    'quote_nonnumeric': csv.QUOTE_NONNUMERIC,
    csv.QUOTE_NONNUMERIC: csv.QUOTE_NONNUMERIC,
    'quote_all': csv.QUOTE_ALL,
    csv.QUOTE_ALL: csv.QUOTE_ALL,
}

DASK_APPLY_OUTPUT_MLTYPE_TO_META_MAP = {
    MLType.BOOL: bool,
    MLType.TEXT: str,
    MLType.INT: int,
    MLType.FLOAT: float,
    MLType.VECTOR: list,
}


class DataPosition(AutoEnum):
    START = auto()
    MIDDLE = auto()
    END = auto()


class AggregationStrategy(AutoEnum):
    AVERAGE = auto()
    MIN = auto()
    MAX = auto()
    MEDIAN = auto()
    MODE = auto()
    NONE = auto()


class CompressionEngine(AutoEnum):
    BROTLI = auto()
    GZIP = auto()


class Status(AutoEnum):
    PENDING = alias('SCHEDULED')  ## The job has not yet started executing
    RUNNING = auto()  ## The job is currently running.
    STOPPED = auto()  ## The job was intentionally stopped by the user.
    SUCCEEDED = alias('SUCCESS', 'SUCCESSFUL')  ## The job finished successfully.
    FAILED = auto()  ## The job failed.


COMPLETED_STATUSES: Set[Status] = {Status.STOPPED, Status.SUCCEEDED, Status.FAILED}


class FailureAction(AutoEnum):
    ERROR = auto()
    ERROR_DELAYED = auto()
    WARN = auto()
    IGNORE = auto()
