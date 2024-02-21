from typing import *
from abc import ABC, abstractmethod
import json
from synthesizrr.base.util import MutableParameters, UserEnteredParameters, Registry
from synthesizrr.base.constants import MLType, MLTypeSchema, MissingColumnBehavior, DataLayout
from synthesizrr.base.data.sdf import ScalableDataFrame, ScalableDataFrameRawType, ScalableSeries, ScalableSeriesRawType
from pydantic import validator, root_validator


class DataProcessor(MutableParameters, Registry, ABC):
    """
    Abstract base class for all data processors.

    Subclasses of this class should be serializable via pickling.
    Subclasses must define the following class variables:
    - missing_column_behavior: Used in the context of DataProcessingPipeline. This field determined whether to allow
     skipping of transformations when the columns required for those transformations are not present in the DataFrame.
     E.g. If the pipeline processes the ground truth labels (such as label-encoding), then during inference time ground
     truth labels will not be present and transformations declared on the ground truth column cannot run.
    - input_mltypes: Lists the input MLTypes types of the columns the data processor can act take as input.
    - output_mltype: Lists the output MLType the data processor will return. This is an instance method since it might
     vary depending on the parameters used to initialize the data processor.
    """

    missing_column_behavior: ClassVar[MissingColumnBehavior] = MissingColumnBehavior.ERROR
    input_mltypes: ClassVar[Tuple[MLType, ...]]
    output_mltype: ClassVar[MLType]

    AlreadyFitError: ClassVar[ValueError] = ValueError(f'.fit() has already been called.')
    FitBeforeTransformError: ClassVar[ValueError] = ValueError(f'.fit() must be called before .transform()')

    class Params(UserEnteredParameters):
        """
        BaseModel for parameters. Expected to be overridden by subclasses of DataProcessor.
        Example: 
            class CaseTransformer(DataProcessor):
                class Params(DataProcessor.Params):
                    case: Literal['lowercase', 'uppercase']         
        """
        pass

    name: str = None
    data_schema: Optional[MLTypeSchema] = None
    params: Params = {}

    def __str__(self):
        params_str: str = self.json(
            include={'name': True, 'data_schema': True, 'params': True},
            indent=4
        )
        out: str = f'{self.class_name} with params:\n{params_str}'
        return out

    @root_validator(pre=True)
    def convert_params(cls, params: Dict):
        params['params'] = super(DataProcessor, cls)._convert_params(cls.Params, params.get('params'))
        return params

    @validator('name')
    def set_name(cls, name: Optional[str]):
        if name is None:
            name: str = cls.class_name
        return name

    @abstractmethod
    def fit(
            self,
            data: Union[ScalableDataFrame, ScalableDataFrameRawType, ScalableSeries, ScalableDataFrameRawType],
            process_as: Optional[DataLayout] = None,
    ) -> NoReturn:
        """
        Fits the data processor instance on the input data.
        By default, this is a no-op, i.e. the data processor is assumed to be stateless.
        - Any subclass implementation must not modify the input data.
        - Any subclass implementation must fit data structure(s) which are serializable via pickling.
        :param data: input data which the data processor will use to fit.
        :param process_as: data-layout to use while processing.
        :return: None
        """
        pass

    @abstractmethod
    def transform(
            self,
            data: Union[ScalableDataFrame, ScalableDataFrameRawType, ScalableSeries, ScalableDataFrameRawType],
            process_as: Optional[DataLayout] = None,
    ) -> Union[ScalableDataFrame, ScalableDataFrameRawType, ScalableSeries, ScalableDataFrameRawType]:
        """
        Transforms the input data and returns the result. Any subclass implementation must not modify the input data.
        :param data: input data which the data processor will act on.
        :param process_as: data-layout to use while processing.
        :return: transformed result.
        """
        pass

    def fit_transform(
            self,
            data: Union[ScalableDataFrame, ScalableDataFrameRawType, ScalableSeries, ScalableDataFrameRawType],
            process_as: Optional[DataLayout] = None
    ) -> Union[ScalableDataFrame, ScalableDataFrameRawType, ScalableSeries, ScalableDataFrameRawType]:
        self.fit(data, process_as=process_as)
        return self.transform(data, process_as=process_as)
