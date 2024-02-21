from typing import *
import pandas as pd
from synthesizrr.base.constants import MLType, DataLayout
from synthesizrr.base.util import AutoEnum, auto, is_null
from synthesizrr.base.data.processor import SingleColumnProcessor, CategoricalInputProcessor, CategoricalOutputProcessor
from synthesizrr.base.data.sdf import ScalableSeries, ScalableSeriesRawType
from pydantic import root_validator


class CategoricalImputationStrategy(AutoEnum):
    MODE = auto()
    CONSTANT = auto()


class CategoricalMissingValueImputation(SingleColumnProcessor, CategoricalInputProcessor, CategoricalOutputProcessor):
    """
    This calculates or fills in the value to be filled in place of nan based on strategy passed as input.
    Params:
    - FILL_VALUE: the value to be filled in when it encounters a NaN (This must be only passed when CONSTANT is strategy)
    - STRATEGY: this indicates what strategy must be used when NaN is encountered
        - MODE: The number which appears most often in a set of numbers
        - CONSTANT: This allows the user to pass in a fill value where that fill value will be imputed
    """

    class Params(SingleColumnProcessor.Params):
        strategy: CategoricalImputationStrategy
        fill_value: Optional[Any] = None

    imputed_value: Optional[Any] = None

    @root_validator(pre=False)
    def set_imputed_value(cls, params: Dict):
        if params['params'].strategy is CategoricalImputationStrategy.CONSTANT:
            if params['params'].fill_value is None:
                raise ValueError(f'Cannot have empty `fill_value` when `strategy` is {CategoricalImputationStrategy.CONSTANT}')
            params['imputed_value'] = params['params'].fill_value
        elif params['params'].fill_value is not None:
            raise ValueError(f'`fill_value` can only be passed when strategy={CategoricalImputationStrategy.CONSTANT}')
        return params

    def _fit_series(self, data: ScalableSeries):
        if self.params.strategy is not CategoricalImputationStrategy.CONSTANT:
            if self.imputed_value is not None:
                raise self.AlreadyFitError
            if self.params.strategy is CategoricalImputationStrategy.MODE:
                self.imputed_value = self._get_mode(data)
            else:
                raise NotImplementedError(f'Unsupported strategy: {self.params.strategy}')

    def _get_mode(self, data: ScalableSeries) -> Any:
        imputed_value: Any = data.mode().compute().iloc[0]
        if not isinstance(imputed_value, str):
            if float(imputed_value).is_integer():
                return int(imputed_value)
        return imputed_value

    def transform_single(self, data: Optional[Any]) -> Any:
        if self.imputed_value is None and self.params.strategy is not CategoricalImputationStrategy.CONSTANT:
            raise self.FitBeforeTransformError
        if is_null(data):
            data = self.imputed_value
        return data
