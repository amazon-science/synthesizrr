from typing import *
from abc import ABC, abstractmethod
import random, copy, math
import numpy as np
from scipy import stats
import pandas as pd
from pandas.core.frame import Series as PandasSeries
from synthesizrr.base.util import wrap_fn_output, is_function, get_default
from synthesizrr.base.constants import DataLayout
from synthesizrr.base.data.sdf.ScalableSeries import ScalableSeries, SS_DEFAULT_NAME
from synthesizrr.base.data.sdf.NumpyArrayScalableSeries import NumpyArrayScalableSeries
from synthesizrr.base.data.sdf.ScalableDataFrame import ScalableDataFrame
from pydantic import conint
from pydantic.typing import Literal

TensorScalableSeries = "TensorScalableSeries"


class TensorScalableSeries(ScalableSeries, ABC):
    TensorType: ClassVar[Type]

    @property
    @abstractmethod
    def tensor_shape(self) -> Tuple[int,]:
        pass

    def __len__(self):
        if self.is_0d:
            ## 0-dimensional tensor, e.g. torch.tensor(True) ## Note small "t" in torch.tensor
            return 1
        return self.tensor_shape[0]

    def as_pandas(self, **kwargs) -> PandasSeries:
        return pd.Series(self.numpy(**kwargs), name=get_default(self._name, SS_DEFAULT_NAME))

    def _to_frame_raw(self, **kwargs):
        kwargs['name'] = get_default(self._name, SS_DEFAULT_NAME)
        return self.pandas(**kwargs).to_frame(**kwargs)

    @property
    @abstractmethod
    def is_0d(self) -> bool:
        pass

    def __getattr__(self, attr_name: str) -> Union[Any, TensorScalableSeries]:
        """Forwards calls to the respective method of Tensor class."""
        out = super().__getattr__(attr_name)
        if is_function(out):
            return wrap_fn_output(out, wrapper_fn=self._to_scalable)
        if isinstance(out, self.TensorType):
            return self._constructor(out)
        return out

    """
    ---------------------------------------------
    Function application, GroupBy & window
    ---------------------------------------------
    """

    def map(
            self,
            arg: Union[Callable, Dict, ScalableSeries],
            na_action: Optional[Literal['ignore']] = None,
    ) -> ScalableSeries:
        raise NotImplementedError('Cannot execute .map() over a Tensor series')
