from typing import *
import json, io, gc, copy, cloudpickle, time, math
from math import inf
from abc import abstractmethod, ABC
import numpy as np
import pandas as pd
from synthesizrr.base.util import as_list, is_list_like, AutoEnum, auto, Parameters, UserEnteredParameters, StringUtil, \
    safe_validate_arguments, Log, format_exception_msg, FractionalBool, measure_time_ms, Registry, is_subset, \
    get_subset, keep_values, filter_string_list, keep_keys, type_str
from synthesizrr.base.constants import ProcessingMode, MLType, MLTypeSchema, FileContents, MissingColumnBehavior
from synthesizrr.base.data.FileMetadata import FileMetadata
from synthesizrr.base.data.sdf import ScalableDataFrame, ScalableDataFrameRawType, DataLayout
from synthesizrr.base.data.reader import Reader, ConfigReader
from synthesizrr.base.data.writer import Writer, DataFrameWriter
from synthesizrr.base.data.processor import DataProcessor, SingleColumnProcessor, Nto1ColumnProcessor
from pydantic import root_validator, constr, conint, confloat
from collections import OrderedDict

AlgorithmDatasetWriter = "AlgorithmDatasetWriter"
DataProcessingPipeline = "DataProcessingPipeline"
DataProcessingPipelineStep = "DataProcessingPipelineStep"
DataProcessingPipelineStepProcessor = "DataProcessingPipelineStepProcessor"

PipelineWriter = Union[DataFrameWriter, AlgorithmDatasetWriter]


class PersistLevel(AutoEnum):
    DONT_PERSIST = auto()
    BEFORE_PIPELINE = auto()
    AFTER_PIPELINE = auto()
    BEFORE_AFTER_PIPELINE = auto()
    EVERY_PIPELINE_STEP = auto()
    EVERY_PROCESSOR = auto()


class ProcessorPerf(Parameters):
    start_time: confloat(ge=0.0)
    processing_mode: ProcessingMode
    input_columns: List[str]
    output_columns: List[str]
    data_processor_class_name: str
    data_processor_params: Dict
    persist_time_ms: Optional[confloat(ge=0.0)]
    end_time: confloat(ge=0.0)
    time_ms: Optional[confloat(ge=0.0)]

    @root_validator(pre=True)
    def set_time_ms(cls, params):
        params['time_ms'] = 1000 * (params['end_time'] - params['start_time'])
        return params


class PipelineStepPerf(Parameters):
    start_time: confloat(ge=0.0)
    processing_mode: ProcessingMode
    num_rows_processed: Optional[conint(ge=1)]
    length_calculation_ms: Optional[confloat(ge=0.0)]
    processor_perfs: List[ProcessorPerf]
    persist_time_ms: Optional[confloat(ge=0.0)]
    end_time: confloat(ge=0.0)
    time_ms: Optional[confloat(ge=0.0)]

    @root_validator(pre=True)
    def set_time_ms(cls, params):
        params['time_ms'] = 1000 * (params['end_time'] - params['start_time'])
        return params


class PipelineWriterPerf(Parameters):
    start_time: confloat(ge=0.0)
    input_columns: List[str]
    writer_class_name: str
    writer_params: Dict
    end_time: confloat(ge=0.0)
    time_ms: Optional[confloat(ge=0.0)]

    @root_validator(pre=True)
    def set_time_ms(cls, params):
        params['time_ms'] = 1000 * (params['end_time'] - params['start_time'])
        return params


class ProcessingPipelinePerf(Parameters):
    processing_mode: ProcessingMode
    persist: PersistLevel
    is_input_ScalableDataFrame: bool
    should_log_perf: bool

    start_time: confloat(ge=0.0)
    layout_detection_time_ms: confloat(ge=0.0)
    input_data_layout: DataLayout

    persist_read_time_ms: Optional[confloat(ge=0.0)]

    length_calculation_ms: Optional[confloat(ge=0.0)]
    process_as: DataLayout
    layout_conversion_ms: confloat(ge=0.0)

    pipeline_steps_compute_time_ms: Optional[confloat(ge=0.0)]
    pipeline_step_perfs: Optional[List[PipelineStepPerf]]
    persist_compute_time_ms: Optional[confloat(ge=0.0)]

    pipeline_write_time_ms: Optional[confloat(ge=0.0)]
    pipeline_writer_perfs: Optional[List[PipelineWriterPerf]]

    num_rows_processed: conint(ge=1)

    end_time: confloat(ge=0.0)
    time_ms: Optional[confloat(ge=0.0)]

    @root_validator(pre=True)
    def set_time_ms(cls, params):
        params['time_ms'] = 1000 * (params['end_time'] - params['start_time'])
        return params


class DataProcessingPipelineConfig(UserEnteredParameters):
    """Structure in YAML file."""

    class StepConfig(UserEnteredParameters):
        input: Union[List[Union[MLType, str]], MLType, str]
        output: constr(min_length=1, strip_whitespace=True) = '{col_name}'
        params: Optional[Dict[str, Any]] = None
        transformer: constr(min_length=1, strip_whitespace=True)  ## Data Processor name

    class WriterConfig(UserEnteredParameters):
        input: Union[List[Union[MLType, str]], MLType, str]
        writer: constr(min_length=1, strip_whitespace=True)
        params: Dict[str, Any] = {}
        schema_override: MLTypeSchema = {}

    pipeline: List[StepConfig] = []
    writers_config: List[WriterConfig] = []


class DataProcessingPipelineStepProcessor(Parameters, Registry, ABC):
    data_processor_class: ClassVar[Type[DataProcessor]]
    data_processor: DataProcessor
    output_col_name: str
    output_mltype: MLType

    @classmethod
    def _registry_keys(cls) -> Optional[Union[List[Any], Any]]:
        return [cls.data_processor_class, cls.data_processor_class.__name__]

    @classmethod
    @abstractmethod
    def create_pipeline_step_processors(
            cls,
            DataProcessorClass: Type[DataProcessor],
            filtered_input_schema: MLTypeSchema,
            name: str,
            params: Dict,
            output_pattern: str,
    ) -> Dict[Union[str, Tuple[str]], DataProcessingPipelineStepProcessor]:
        """
        Static factory to create a mapping from input column(s) to data processor instances and their output.
        :param filtered_input_schema: input schema with only the relevant columns which we must transform.
                Each key is a column name from the input data, and each value is its corresponding MLType.
        :param name: name of the data processor(s).
        :param params: dict of params to initialize the data processor(s).
        :param output_pattern: used to name the output columns.
        :return: Depending on the type of processor (1:1, N:1, etc), the returned map will have each key as a single
        column or a tuple of columns. The value is the data processor instance which will transform a single column or
        set of columns, respectively.
        E.g. for 1:1 we might return:
        {
            "ASIN_STATIC_ITEM_NAME":
                (<__TFIDFVectorization_at_87da792f>, 'ASIN_STATIC_ITEM_NAME_TFIDF_15000', MLType.VECTOR),
            "ASIN_STATIC_BULLET_POINT":
                (<__TFIDFVectorization_at_adf90eb8>, 'ASIN_STATIC_BULLET_POINT_TFIDF_15000', MLType.VECTOR)
        }
        E.g. for N:1 we might return:
        {
            ("ASIN_STATIC_ITEM_NAME", "ASIN_STATIC_BULLET_POINT"):
                (<__TextConcatenation_at_92ba33e>, 'CONCATENATED_TEXT_COLUMNS', MLType.TEXT)
        }
        """
        pass

    @classmethod
    @abstractmethod
    def get_pipeline_step_output_schema(
            cls,
            input_schema: MLTypeSchema,
            pipeline_step_processors: Dict[Union[str, Tuple[str]], DataProcessingPipelineStepProcessor]
    ) -> MLTypeSchema:
        """
        Obtains the output schema from the input data processors dict.
        :param input_schema: schema with all columns in the current DataFrame.
        :param pipeline_step_processors: map from column(s) which must be transformed, to data processor
            and its outputs. This should be the output of a call to `create_pipeline_step_processors`.
        :return: the updated output schema. Columns which are not in `pipeline_step_processors` are copied as-is.
            For other columns this function will use the output column names and MLTypes in `pipeline_step_processors`
            to add the corresponding columns to the input schema. This becomes the output schema which is returned.
        """
        pass


class DataProcessingPipelineStepSingleColumnProcessor(DataProcessingPipelineStepProcessor):
    data_processor_class: ClassVar[Type[DataProcessor]] = SingleColumnProcessor

    @classmethod
    def create_pipeline_step_processors(
            cls,
            DataProcessorClass: Type[SingleColumnProcessor],
            filtered_input_schema: MLTypeSchema,
            name: str,
            params: Dict,
            output_pattern: str,
    ) -> Dict[Union[str, Tuple[str]], DataProcessingPipelineStepProcessor]:
        pipeline_step_processors: Dict[Union[str, Tuple[str]], DataProcessingPipelineStepProcessor] = {}
        for input_col, input_mltype in filtered_input_schema.items():
            processor_input_schema: MLTypeSchema = {input_col: input_mltype}
            data_processor: SingleColumnProcessor = DataProcessorClass(
                name=name,
                data_schema=processor_input_schema,
                params=params,
            )
            supported_input_mltypes: Tuple[MLType] = data_processor.input_mltypes
            assert input_mltype in supported_input_mltypes, \
                f'"{str(input_mltype)}" not included in supported MLTypes: "{str(supported_input_mltypes)}"'
            ## For 1:1 data processors, the supported input MLType should be a list of MLTypes
            if not all([isinstance(mltype, MLType) for mltype in supported_input_mltypes]):
                raise AttributeError(
                    f'Supported input types for class {str(cls)} (1:1 data processor) ' + \
                    f'should be a list of MLTypes, not: {supported_input_mltypes}'
                )
            ## Converts '{col_name}_XYZ' to 'MyCol_XYZ' but leaves 'XYZ' unchanged.
            output_col_name = output_pattern.format(col_name=input_col)
            output_mltype = data_processor.output_mltype
            pipeline_step_processors[input_col] = cls(
                data_processor=data_processor,
                output_col_name=output_col_name,
                output_mltype=output_mltype
            )
        return pipeline_step_processors

    @classmethod
    def get_pipeline_step_output_schema(
            cls,
            input_schema: MLTypeSchema,
            pipeline_step_processors: Dict[Union[str, Tuple[str]], DataProcessingPipelineStepProcessor]
    ) -> MLTypeSchema:
        output_schema = copy.deepcopy(input_schema)
        for input_cols, step_processor in pipeline_step_processors.items():
            if step_processor.output_col_name is not None and step_processor.output_mltype is not None:
                output_schema[step_processor.output_col_name] = step_processor.output_mltype
        return output_schema


class DataProcessingPipelineStepNto1ColumnProcessor(DataProcessingPipelineStepProcessor):
    data_processor_class: ClassVar[Type[DataProcessor]] = Nto1ColumnProcessor

    @classmethod
    def create_pipeline_step_processors(
            cls,
            DataProcessorClass: Type[Nto1ColumnProcessor],
            filtered_input_schema: MLTypeSchema,
            name: str,
            params: Dict,
            output_pattern: str,
    ) -> Dict[Union[str, Tuple[str]], DataProcessingPipelineStepProcessor]:
        pipeline_step_processors: Dict[Union[str, Tuple[str]], DataProcessingPipelineStepProcessor] = {}
        ## Sorted tuple of columns we want to pass to this data processor.
        input_cols: Tuple[str] = tuple(filtered_input_schema.keys())
        if len(input_cols) > 0:
            processor: Nto1ColumnProcessor = DataProcessorClass(
                name=name,
                data_schema=copy.deepcopy(filtered_input_schema),
                params=params,
            )
            supported_input_mltypes: Tuple[MLType, ...] = processor.input_mltypes
            ## For N:1 data processors, the supported input MLType should be a list of MLTypes
            if not all([isinstance(mltype, MLType) for mltype in supported_input_mltypes]):
                raise AttributeError(
                    f'Supported input types for {str(cls)} (N:1 data processor) ' + \
                    f'should be a list of MLTypes, not: {supported_input_mltypes}'
                )
            if not all([mltype in supported_input_mltypes for mltype in filtered_input_schema.values()]):
                raise AttributeError(
                    f'MLTypes of selected columns passed to {str(cls)} (N:1 data processor) ' + \
                    f'should be supported by this data processor. Supported types are ' + \
                    f'{supported_input_mltypes}, selected columns have MLTypes: ' + \
                    f'{list(filtered_input_schema.values())}'
                )
            output_col_name = output_pattern  ## Assume it does not have {col_name} in it.
            output_mltype: MLType = processor.output_mltype
            pipeline_step_processors[input_cols] = cls(
                data_processor=processor,
                output_col_name=output_col_name,
                output_mltype=output_mltype,
            )
        return pipeline_step_processors

    @classmethod
    def get_pipeline_step_output_schema(
            cls,
            input_schema: MLTypeSchema,
            pipeline_step_processors: Dict[Union[str, Tuple[str]], DataProcessingPipelineStepProcessor]
    ) -> MLTypeSchema:
        output_schema = copy.deepcopy(input_schema)
        ## dict returned by create_pipeline_step_processors should have exactly one item.
        assert len(pipeline_step_processors) <= 1
        for input_cols, step_processor in pipeline_step_processors.items():
            if step_processor.output_col_name is not None and step_processor.output_mltype is not None:
                output_schema[step_processor.output_col_name] = step_processor.output_mltype
        return output_schema


class DataProcessingPipelineStep(Parameters):
    input_schema: MLTypeSchema
    pipeline_step_processors: Dict[Union[str, Tuple], DataProcessingPipelineStepProcessor]
    output_schema: MLTypeSchema

    def __str__(self):
        out_str = f'{self.__class__.__name__}:'
        out_str += '\n  >> Input schema: ' + str(
            MLType.convert_values_to_str(self.input_schema)
        )
        out_str += '\n  >> Data Processors map:'
        for cols_to_transform, step_processor in self.pipeline_step_processors.items():
            out_str += f'\n  - Columns to transform: {str(cols_to_transform)}'
            out_str += f'\n    Data Processor: {step_processor.data_processor.class_name}'
            if len(step_processor.data_processor.params.dict()) > 0:
                out_str += f' ({step_processor.data_processor.params})'
            out_str += f'\n    Output column: {str(step_processor.output_col_name)} ({str(step_processor.output_mltype)})'
        out_str += '\n  >> Output schema: ' + str(
            MLType.convert_values_to_str(self.output_schema)
        )
        return out_str

    @classmethod
    @safe_validate_arguments
    def from_config(
            cls,
            step_cfg: DataProcessingPipelineConfig.StepConfig,
            step_input_schema: MLTypeSchema,
    ) -> Any:
        """
        Static factory to resolve and instantiate a pipeline step object.
        Resolution includes:
        - Add filtered input schema to the pipeline step
        - Add a collection of data processors to the pipeline step
        - Add an output schema to the pipeline step
        :param step_cfg: pipeline step configuration.
        :param step_input_schema: the schema of the DataFrame at this step.
        :return: Serializable DataProcessingPipelineStep instance.
        """
        ## Extract variables:
        DataProcessorClass: Type[DataProcessor] = DataProcessor.get_subclass(step_cfg.transformer)
        if issubclass(DataProcessorClass, SingleColumnProcessor):
            DataProcessorSuperClass: Type[DataProcessor] = SingleColumnProcessor
        elif issubclass(DataProcessorClass, Nto1ColumnProcessor):
            DataProcessorSuperClass: Type[DataProcessor] = Nto1ColumnProcessor
        else:
            raise NotImplementedError(
                f'Unsupported subtype of {DataProcessor}: {DataProcessorClass}, '
                f'with following inheritance: {DataProcessorClass.__mro__}'
            )

        DataProcessingPipelineStepProcessorClass: Type[DataProcessingPipelineStepProcessor] = \
            DataProcessingPipelineStepProcessor.get_subclass(DataProcessorSuperClass)
        ## Create data processors and output schema:
        ## Note: selection of columns from the pipeline config is case insensitive. User might enter 'AbCD' but the
        ## appropriate columns 'abcd' will be picked up from the DataFrame schema.
        filtered_step_input_schema: MLTypeSchema = PipelineUtil.filter_schema_by_input_patterns(
            step_input_schema,
            step_cfg.input,
        )
        try:
            pipeline_step_processors: Dict[Union[str, Tuple[str]], DataProcessingPipelineStepProcessor] = \
                DataProcessingPipelineStepProcessorClass.create_pipeline_step_processors(
                    DataProcessorClass=DataProcessorClass,
                    filtered_input_schema=filtered_step_input_schema,
                    name=step_cfg.transformer,
                    params=step_cfg.params,
                    output_pattern=step_cfg.output,
                )
        except Exception as e:
            print(format_exception_msg(e))
            raise AttributeError(
                f'Error while creating data processor of type "{str(DataProcessorClass)}" '
                f'with params: {str(step_cfg.params)} '
                f'and filtered input schema {str(filtered_step_input_schema)}'
            )
        output_schema: MLTypeSchema = DataProcessingPipelineStepProcessorClass.get_pipeline_step_output_schema(
            input_schema=step_input_schema,
            pipeline_step_processors=pipeline_step_processors
        )
        return DataProcessingPipelineStep(
            input_schema=filtered_step_input_schema,
            pipeline_step_processors=pipeline_step_processors,
            output_schema=output_schema,
        )

    def execute_pipeline_step(
            self,
            sdf: ScalableDataFrame,
            processing_mode: ProcessingMode,
            persist: PersistLevel,
            should_measure_perf: bool,
            should_log_perf: bool,
    ) -> Tuple[ScalableDataFrame, Optional[PipelineStepPerf]]:
        """
        Runs the particular pipeline step on the input ScalableDataFrame.
        :param sdf: input ScalableDataFrame to process.
        :param processing_mode: what this step should do, e.g. fit-transform, transform, etc.
        :param persist: how often to persist the ScalableDataFrame every so often.
        :param should_measure_perf: whether to measure performance information.
        :param should_log_perf: whether to log performance information.
        :return: transformed ScalableDataFrame (or raw data) after executing this step
        """
        step_start_time = time.perf_counter()
        if should_log_perf:
            Log.debug(f'\n>> Running {processing_mode.lower().replace("_", "-")} on pipeline step...')
        _processor_perfs: List[ProcessorPerf] = []
        for input_cols, step_processors in self.pipeline_step_processors.items():
            data_processor: DataProcessor = step_processors.data_processor
            output_col_name: str = step_processors.output_col_name
            input_cols: List[str] = as_list(input_cols)
            sdf_cols: List[str] = list(sdf.columns)
            if is_subset(input_cols, sdf_cols) or \
                    data_processor.missing_column_behavior is MissingColumnBehavior.EXECUTE:
                ## Apply data processor on whatever subset exists, retaining column order:
                cols_to_process_set: Set[str] = get_subset(input_cols, sdf_cols)
                cols_to_process_in_order: List[str] = [
                    col for col in input_cols
                    if col in cols_to_process_set
                ]
                if isinstance(data_processor, SingleColumnProcessor):
                    if len(cols_to_process_in_order) != 1:
                        raise ValueError(f'Expected only one column, found: {cols_to_process_in_order}')
                    cols_to_process_in_order: str = cols_to_process_in_order[0]
                processor_start_time = time.perf_counter()
                if should_log_perf:
                    Log.debug(
                        f'\n>> Running {processing_mode.lower().replace("_", "-")} '
                        f'on {type_str(sdf)}, using:\n{str(data_processor)}'
                    )
                sdf: ScalableDataFrame = self._execute_data_processor(
                    sdf=sdf,
                    cols_to_process_in_order=cols_to_process_in_order,
                    data_processor=data_processor,
                    processing_mode=processing_mode,
                    output_col_name=output_col_name,
                )
                persist_time_ms: Optional[float] = None
                if persist is PersistLevel.EVERY_PROCESSOR:
                    sdf, persist_time_ms = measure_time_ms(
                        lambda: sdf.persist(wait=True)
                    )

                processor_end_time: float = time.perf_counter()
                if should_log_perf:
                    Log.debug(
                        f'\r...processor ran in '
                        f'{StringUtil.readable_seconds(processor_end_time - processor_start_time)}.'
                    )
                if should_measure_perf:
                    _processor_perfs.append(ProcessorPerf(
                        start_time=processor_start_time,
                        processing_mode=processing_mode,
                        input_columns=as_list(cols_to_process_in_order),
                        output_columns=as_list(output_col_name),
                        data_processor_class_name=data_processor.class_name,
                        data_processor_params=data_processor.params.dict(),
                        persist_time_ms=persist_time_ms,
                        end_time=processor_end_time,
                    ))
            elif data_processor.missing_column_behavior is MissingColumnBehavior.SKIP:
                continue
            elif data_processor.missing_column_behavior is MissingColumnBehavior.ERROR:
                raise ValueError(
                    f'Cannot transform {type_str(sdf)} using {data_processor.class_name} due to insufficient columns: '
                    f'columns required for transformation: {input_cols}; '
                    f'columns actually present: {sdf_cols}'
                )
            else:
                raise NotImplementedError(
                    f'Unsupported value for {MissingColumnBehavior}: {data_processor.missing_column_behavior}'
                )
        persist_time_ms: Optional[float] = None
        if persist is PersistLevel.EVERY_PIPELINE_STEP:
            sdf, persist_time_ms = measure_time_ms(
                lambda: sdf.persist(wait=True)
            )

        step_end_time: float = time.perf_counter()
        if should_log_perf:
            Log.debug(
                f'\r...pipeline-step ran in '
                f'{StringUtil.readable_seconds(step_end_time - step_start_time)}.'
            )
        step_end_time: float = time.perf_counter()
        if should_measure_perf:
            if sdf.layout is not DataLayout.DASK:
                sdf_num_rows, length_calculation_ms = measure_time_ms(
                    lambda: len(sdf)
                )
            else:
                sdf_num_rows, length_calculation_ms = None, None
            return sdf, PipelineStepPerf(
                start_time=step_start_time,
                processing_mode=processing_mode,
                num_rows_processed=sdf_num_rows,
                length_calculation_ms=length_calculation_ms,
                processor_perfs=_processor_perfs,
                persist_time_ms=persist_time_ms,
                end_time=step_end_time,
            )
        return sdf, None

    def _execute_data_processor(
            self,
            sdf: ScalableDataFrame,
            cols_to_process_in_order: List[str],
            data_processor: DataProcessor,
            processing_mode: ProcessingMode,
            output_col_name: str,
    ) -> ScalableDataFrame:
        if processing_mode is ProcessingMode.FIT_TRANSFORM:
            sdf[output_col_name] = data_processor.fit_transform(sdf[cols_to_process_in_order])
        elif processing_mode is ProcessingMode.TRANSFORM:
            sdf[output_col_name] = data_processor.transform(sdf[cols_to_process_in_order])
        return sdf


class DataProcessingPipeline(Parameters):
    input_schema: MLTypeSchema
    pipeline: List[DataProcessingPipelineStep]
    output_schema: MLTypeSchema
    writers: Dict[FileContents, PipelineWriter] = {}
    layout_scaling: Optional[Dict[ProcessingMode, Tuple[Tuple[confloat(ge=1), DataLayout], ...]]] = {
        ProcessingMode.FIT_TRANSFORM: (
            ## Determines which layout to use with different number of rows.
            (1_000, DataLayout.DICT),  ## <= 1k rows, use DataLayout.DICT
            (500_000, DataLayout.PANDAS),  ## <= 500k rows, use DataLayout.PANDAS
            (inf, DataLayout.DASK),  ## >500k rows, use DataLayout.DASK
        ),
        ProcessingMode.TRANSFORM: (
            ## Determines which layout to use with different number of rows.
            (5, DataLayout.LIST_OF_DICT),  ## <= 5 rows, use DataLayout.LIST_OF_DICT
            (1_000, DataLayout.DICT),  ## <= 1k rows, use DataLayout.DICT
            (125_000, DataLayout.PANDAS),  ## <= 125k rows, use DataLayout.PANDAS
            (inf, DataLayout.DASK),  ## >125k rows, use DataLayout.DASK
        ),
    }
    _performance: List[ProcessingPipelinePerf] = []  ## For performance tracking

    # @classmethod
    # @safe_validate_arguments
    # def from_steps(
    #         cls,
    #         input_schema: MLTypeSchema,
    #         process: List[Union[
    #             DataProcessor,
    #             Tuple[str, DataProcessor],
    #             Tuple[DataProcessor, str],
    #             Tuple[str, DataProcessor, str]
    #         ]],
    #         select: List[Union[MLType, str]],
    #         write: Optional[List[Writer]] = None
    # ) -> DataProcessingPipeline:
    #     process: List = as_list(process)
    #     select: List = as_list(process)
    #     if write is not None:
    #         write: List = as_list(write)
    #     current_schema: MLTypeSchema = copy.deepcopy(input_schema)
    #     processing_steps: List[DataProcessingPipelineStep] = []
    #     for processor_tuple in process:
    #         if isinstance(processor_tuple, DataProcessor):
    #             DataProcessingPipelineStep(
    #                 input_schema=filtered_step_input_schema,
    #                 data_processors=data_processors,
    #                 output_schema=output_schema,
    #             )

    @classmethod
    @safe_validate_arguments
    def from_config(
            cls,
            config: Union[DataProcessingPipelineConfig, FileMetadata],
            input_schema: MLTypeSchema,
            only_writers: bool = False,
            *args,
            **kwargs,
    ) -> DataProcessingPipeline:
        """
        Static factory to resolve each pipeline step and instantiate the pipeline object.
        :param config: either DataProcessingPipelineConfig or config file (YAML/JSON) with pipeline steps and writers.
        :param input_schema: schema of the input dataframe this pipeline can process.
        :param only_writers: if True, then only the writers will be initialized.
        :return: Serializable DataProcessingPipeline instance.
        """
        if isinstance(config, FileMetadata):
            reader: Reader = Reader.of(config.format)
            assert isinstance(reader, ConfigReader)
            Log.debug('\nReading pipeline config...')
            config: DataProcessingPipelineConfig = DataProcessingPipelineConfig(**reader.read_metadata(config))
            Log.debug('...done reading pipeline config.')
        if not only_writers:
            return cls._resolve_pipeline(
                input_schema=input_schema,
                pipeline_steps=config.pipeline,
                writers=config.writers_config,
                *args,
                **kwargs,
            )
        else:
            return cls._resolve_pipeline(
                input_schema=input_schema,
                pipeline_steps=[],
                writers=config.writers_config,
                *args,
                **kwargs,
            )

    @classmethod
    def _resolve_pipeline(
            cls,
            input_schema: MLTypeSchema,
            pipeline_steps: List[DataProcessingPipelineConfig.StepConfig],
            writers: Optional[List[DataProcessingPipelineConfig.WriterConfig]] = None,
            *args,
            **kwargs,

    ) -> DataProcessingPipeline:
        """
        Static factory to resolve each pipeline step and instantiate the pipeline object.
        :param input_schema: schema of the input dataframe this pipeline can process.
        :param pipeline_steps: list of pipeline steps input by the user.
        :param writers: list of Dataframe or Algorithm writers input by the user.
            Some of these may be invoked when the file with corresponding properties is passed.
        :return: Serializable DataProcessingPipeline instance.
        """

        Log.debug('\nInitializing DataProcessingPipeline...')
        Log.debug(f'\n> Input schema to pipeline: {input_schema}')

        ## Resolve pipeline steps:
        resolved_pipeline: List[DataProcessingPipelineStep] = []
        cur_schema = input_schema
        Log.debug(f'\n> Resolving pipeline transformation steps: {str(pipeline_steps)}')
        for pipeline_step in pipeline_steps:
            resolved_pipeline_step: DataProcessingPipelineStep = DataProcessingPipelineStep.from_config(
                step_cfg=pipeline_step,
                step_input_schema=cur_schema,
            )
            resolved_pipeline.append(resolved_pipeline_step)
            Log.debug(f'Added {str(resolved_pipeline_step)}')
            cur_schema: MLTypeSchema = resolved_pipeline_step.output_schema
        output_schema: MLTypeSchema = cur_schema
        Log.debug(f'...resolved pipeline transformation steps.')
        Log.debug(f'\n> Output schema from pipeline: \n{json.dumps(output_schema, indent=4)}')

        ## Resolve writers:
        if writers is None:
            writers: Dict[FileContents, PipelineWriter] = {}
        else:
            Log.debug(f'\n> Resolving pipeline writers...')
            writers: Dict[FileContents, PipelineWriter] = cls._resolve_pipeline_writers(
                writers=writers,
                output_schema=output_schema,
                *args,
                **kwargs,
            )
            Log.debug('...resolved pipeline writers.')

        ## Instantiate:
        pipeline = DataProcessingPipeline(
            input_schema=input_schema,
            pipeline=resolved_pipeline,
            output_schema=output_schema,
            writers=writers,
        )
        Log.debug('...done initializing pipeline.')
        return pipeline

    @classmethod
    @safe_validate_arguments
    def _resolve_pipeline_writers(
            cls,
            writers: List[DataProcessingPipelineConfig.WriterConfig],
            output_schema: MLTypeSchema,
            *args,
            **kwargs,
    ) -> Dict[FileContents, PipelineWriter]:
        pipeline_writers: Dict[FileContents, PipelineWriter] = {}
        for writer_cfg in writers:
            writer: PipelineWriter = cls._create_pipeline_writer(
                writer_cfg,
                output_schema,
            )
            for supported_file_content in writer.file_contents:
                if supported_file_content in pipeline_writers:
                    raise KeyError(
                        f'Only one writer of {supported_file_content} contents can be present'
                        f'in the pipeline. Found two writers of type {supported_file_content}.'
                    )
                pipeline_writers[supported_file_content] = writer
                Log.debug(f'Set writer of key "{str(supported_file_content)}" as {str(writer)}')
        return pipeline_writers

    @classmethod
    def _create_pipeline_writer(
            cls,
            writer_cfg: DataProcessingPipelineConfig.WriterConfig,
            output_schema: MLTypeSchema,
    ) -> PipelineWriter:
        writer_cfg: DataProcessingPipelineConfig.WriterConfig = writer_cfg.copy(deep=True)
        WriterClass: Type[Writer] = Writer.get_subclass(writer_cfg.writer)
        if not (isinstance(WriterClass, DataFrameWriter.__class__) or
                isinstance(WriterClass, AlgorithmDatasetWriter.__class__)):
            raise TypeError(
                f'Pipeline writers must be of type {str(DataFrameWriter.class_name)} or '
                f'{str(AlgorithmDatasetWriter.class_name)}, found: {WriterClass.class_name}.'
            )

        ## Overwrite keys in the output schema with those present in the writer config (if any):
        writer_data_schema: MLTypeSchema = {
            **output_schema,
            **writer_cfg.schema_override,
        }
        writer_data_schema: MLTypeSchema = PipelineUtil.filter_schema_by_input_patterns(
            schema=writer_data_schema,
            input_patterns=writer_cfg.input
        )
        writer_cfg.params['data_schema'] = writer_data_schema
        return WriterClass(**writer_cfg.params)

    @safe_validate_arguments
    def get_writer_by_file_contents(self, file_contents: FileContents) -> Optional[Writer]:
        return self.writers.get(file_contents)

    # def fit(self, df, )
    # def transform(self, df, )
    # def fit_transform(self, df, )

    @safe_validate_arguments
    def execute(
            self,
            data: Union[ScalableDataFrame, ScalableDataFrameRawType],
            processing_mode: ProcessingMode,
            process_as: Optional[DataLayout] = None,
            measure_perf: FractionalBool = True,
            log_perf: FractionalBool = True,
            persist: PersistLevel = PersistLevel.DONT_PERSIST,
            write_to: Optional[Union[List[FileMetadata], FileMetadata]] = None,
            overwrite: bool = False,
            rnd: Optional[confloat(ge=0.0, le=1.0)] = None,
            **kwargs
    ) -> Union[ScalableDataFrame, ScalableDataFrameRawType]:
        """
        Executes each pipeline step on the input DataFrame in a sequential fashion.
        :param data: input ScalableDataFrame or raw type (Pandas, Dask, List of Dicts, etc).
        :param processing_mode: fit, fit_transform, transform
        :param process_as: data layout to run the pipeline.
        :param measure_perf: how often to measure performance.
        If False, it will not measure performance. If True, it will measure performance.
        If 0.0 < measure_perf < 1.0, then we will measure performance a fraction of the time.
        :param log_perf: how often to log performance.
        If False, it will not log performance. If True, it will always log performance.
        If 0.0 < log_perf < 1.0, then we will log performance a fraction of the time.
        :param persist: how often to persist processed results (for lazily-evaluated dataframes).
        :param write_to: output files to write to using the configured writers.
        :param overwrite: whether to overwrite the path while writing.
        :param rnd: Optional random value (passed to ensure end-to-end logging and performance measurement).
        :return: the transformed DataFrame.
        """
        pipeline_start_time = time.time()
        if rnd is None:
            rnd: float = np.random.random()
        should_measure_perf: bool = rnd <= measure_perf
        should_log_perf: bool = rnd <= log_perf

        if should_measure_perf:
            Log.info(f'\nRunning pipeline in {processing_mode.lower().replace("_", "-")} mode on dataset...')

        ## Detect layout if the input is raw data:
        is_input_ScalableDataFrame: bool = isinstance(data, ScalableDataFrame)
        sdf, layout_detection_time_ms = measure_time_ms(
            lambda: ScalableDataFrame.of(data, layout=None)
        )
        input_data_layout: DataLayout = sdf.layout

        ## For lazy-loaded DataFrames (e.g. Dask, Spark), read data from file:
        persist_read_time_ms: Optional[float] = None
        if persist in {
            PersistLevel.BEFORE_PIPELINE,
            PersistLevel.BEFORE_AFTER_PIPELINE,
            PersistLevel.EVERY_PIPELINE_STEP,
            PersistLevel.EVERY_PROCESSOR,
        }:
            sdf, persist_read_time_ms = measure_time_ms(
                lambda: sdf.persist(wait=True)
            )

        ## Convert to different layout used to process:
        sdf_num_rows: Optional[int] = None
        length_calculation_ms: Optional[float] = None
        if process_as is None:
            if sdf_num_rows is None:
                sdf_num_rows, length_calculation_ms = measure_time_ms(
                    lambda: len(sdf)
                )
            for (sdf_num_rows_limit, process_as) in self.layout_scaling[processing_mode]:
                if sdf_num_rows <= sdf_num_rows_limit:
                    break  ## Sets data_layout
        layout_conversion_ms: float = 0.0
        if process_as is not DataLayout.RECORD:
            sdf, layout_conversion_ms = measure_time_ms(
                lambda: sdf.as_layout(layout=process_as)
            )

        ## Run the pipeline:
        pipeline_step_perfs: Optional[List[PipelineStepPerf]] = None
        pipeline_steps_compute_time_ms: Optional[float] = None
        if len(self.pipeline) > 0:
            pipeline_steps_compute_start_time: float = time.time()
            if processing_mode is ProcessingMode.TRANSFORM and \
                    process_as in {DataLayout.LIST_OF_DICT, DataLayout.RECORD}:
                sdf, pipeline_step_perfs = self._transform_as_records(
                    sdf=sdf,
                    processing_mode=processing_mode,
                    persist=persist,
                    should_measure_perf=should_measure_perf,
                    should_log_perf=should_log_perf,
                )
            else:
                sdf, pipeline_step_perfs = self._execute_as_sdf(
                    sdf=sdf,
                    processing_mode=processing_mode,
                    persist=persist,
                    should_measure_perf=should_measure_perf,
                    should_log_perf=should_log_perf,
                    process_as=process_as,
                )

            pipeline_steps_compute_end_time: float = time.time()
            pipeline_steps_compute_time_ms: float = 1000 * (
                    pipeline_steps_compute_end_time - pipeline_steps_compute_start_time
            )

        ## For lazy-loaded DataFrames (e.g. Dask, Spark), this actually starts the data-processing:
        persist_compute_time_ms: Optional[float] = None
        if persist in {
            PersistLevel.AFTER_PIPELINE,
            PersistLevel.BEFORE_AFTER_PIPELINE,
        }:
            sdf, persist_compute_time_ms = measure_time_ms(
                lambda: sdf.persist(wait=True)
            )

        ## Write data to files
        pipeline_writer_perfs: Optional[List[PipelineWriterPerf]] = None
        pipeline_write_time_ms: Optional[float] = None
        if write_to is not None:
            write_to: List[FileMetadata] = as_list(write_to)
            pipeline_write_start_time: float = time.time()
            pipeline_writer_perfs: List[PipelineWriterPerf] = self._write_processed(
                sdf=sdf,
                processing_mode=processing_mode,
                should_measure_perf=should_measure_perf,
                should_log_perf=should_log_perf,
                write_to=write_to,
                overwrite=overwrite,
                **kwargs
            )
            pipeline_write_end_time: float = time.time()
            pipeline_write_time_ms: float = 1000 * (pipeline_write_end_time - pipeline_write_start_time)

        ## Log and measure performance
        pipeline_end_time: float = time.time()
        if should_log_perf:
            writers_log_str: str = f' and running {len(self.writers)} writers ' if write_to is not None else ' '
            Log.info(
                f'...done running pipeline in {processing_mode.lower().replace("_", "-")} mode{writers_log_str}in '
                f'{StringUtil.readable_seconds(pipeline_end_time - pipeline_start_time)}.'
            )
        if should_measure_perf:
            if sdf_num_rows is None:
                sdf_num_rows, length_calculation_ms = measure_time_ms(
                    lambda: len(sdf)
                )
            pipeline_end_time: float = time.time()
            self._performance.append(ProcessingPipelinePerf(
                processing_mode=processing_mode,
                persist=persist,
                is_input_ScalableDataFrame=is_input_ScalableDataFrame,
                should_log_perf=should_log_perf,

                start_time=pipeline_start_time,
                layout_detection_time_ms=layout_detection_time_ms,
                input_data_layout=input_data_layout,

                persist_read_time_ms=persist_read_time_ms,

                length_calculation_ms=length_calculation_ms,
                process_as=process_as,
                layout_conversion_ms=layout_conversion_ms,

                pipeline_steps_compute_time_ms=pipeline_steps_compute_time_ms,
                pipeline_step_perfs=pipeline_step_perfs,
                persist_compute_time_ms=persist_compute_time_ms,

                pipeline_write_time_ms=pipeline_write_time_ms,
                pipeline_writer_perfs=pipeline_writer_perfs,

                num_rows_processed=sdf_num_rows,
                end_time=pipeline_end_time,
            ))
        if is_input_ScalableDataFrame:
            return sdf
        return sdf._data

    def _transform_as_records(
            self,
            sdf: ScalableDataFrame,
            processing_mode: ProcessingMode,
            persist: PersistLevel,
            should_measure_perf: bool,
            should_log_perf: bool,
    ) -> Tuple[ScalableDataFrame, List[PipelineStepPerf]]:
        record_sdfs: List[ScalableDataFrame] = list(
            sdf.stream(stream_as=DataLayout.RECORD, num_rows=1, shuffle=False, raw=False)
        )
        for i in range(len(record_sdfs)):
            for (pipeline_step_i, pipeline_step) in enumerate(self.pipeline):
                record_sdfs[i], _step_perf = pipeline_step.execute_pipeline_step(
                    sdf=record_sdfs[i],
                    processing_mode=processing_mode,
                    persist=PersistLevel.DONT_PERSIST,
                    should_measure_perf=should_measure_perf,
                    should_log_perf=should_log_perf,
                )
                ## TODO: log perfs
        record_sdf_concat: ScalableDataFrame = ScalableDataFrame.concat(
            record_sdfs,
            reset_index=True,
            layout=sdf.layout,
        )
        if record_sdf_concat.layout != sdf.layout:
            raise ValueError(
                f'Expected the output {ScalableDataFrame.__name__} to have layout '
                f'{sdf.layout}; found layout {record_sdf_concat.layout}'
            )
        return record_sdf_concat, []

    def _execute_as_sdf(
            self,
            sdf: ScalableDataFrame,
            processing_mode: ProcessingMode,
            persist: PersistLevel,
            should_measure_perf: bool,
            should_log_perf: bool,
            process_as: DataLayout,
    ) -> Tuple[ScalableDataFrame, List[PipelineStepPerf]]:
        pipeline_step_perfs: List[PipelineStepPerf] = []
        for pipeline_step in self.pipeline:
            sdf, _step_perf = pipeline_step.execute_pipeline_step(
                sdf=sdf,
                processing_mode=processing_mode,
                persist=persist,
                should_measure_perf=should_measure_perf,
                should_log_perf=should_log_perf,
            )
            if sdf.layout != process_as:
                raise ValueError(
                    f'Expected the output {ScalableDataFrame.__name__} of the following step to have layout '
                    f'{process_as}; found layout {sdf.layout}: {str(pipeline_step)}'
                )
            if should_measure_perf:
                pipeline_step_perfs.append(_step_perf)
        return sdf, pipeline_step_perfs

    def _write_processed(
            self,
            sdf: ScalableDataFrame,
            processing_mode: ProcessingMode,
            should_measure_perf: bool,
            should_log_perf: bool,
            write_to: List[FileMetadata],
            overwrite: bool = False,
            **kwargs
    ) -> Optional[List[PipelineWriterPerf]]:
        writers_start_time: float = time.time()
        if should_log_perf:
            Log.debug(
                f'\nWriting dataset after {processing_mode.lower().replace("_", "-")}, '
                f'using {len(self.writers)} writers...'
            )

        _writer_perfs: List[PipelineWriterPerf] = []
        for file in write_to:
            writer_start_time: float = time.time()
            writer: Writer = self.writers.get(file.contents)
            if writer is None:
                raise KeyError(
                    f'While writing from pipeline, could not find writer for the following output metadata '
                    f'(with contents {file.contents}):\n{str(file)}'
                )
            if should_log_perf:
                Log.debug(f'\n>> Writing processed data using {str(writer)}')
            if not writer.write_metadata(file, sdf, overwrite=overwrite, **kwargs):
                raise IOError(f'Could not write pipeline output to file.')
            writer_end_time: float = time.time()
            if should_log_perf:
                Log.debug(
                    f'\r...writer ran in '
                    f'{StringUtil.readable_seconds(writer_end_time - writer_start_time)}.'
                )
            if should_measure_perf:
                _writer_perfs.append(PipelineWriterPerf(
                    start_time=writer_start_time,
                    input_columns=sorted(list(writer.data_schema.keys())),
                    writer_class_name=writer.class_name,
                    writer_params=writer.params.dict(),
                    end_time=writer_end_time,
                ))
        writers_end_time: float = time.time()
        if should_log_perf:
            if processing_mode is ProcessingMode.FIT_TRANSFORM:
                Log.info(
                    f'...done running writers in '
                    f'{StringUtil.readable_seconds(writers_end_time - writers_start_time)}.'
                )
        return _writer_perfs

    def serialize(self, file: str):
        """
        Serialize the pipeline object (and all its data processors) using the cloudpickle library, which Ray uses.
        Ref: https://github.com/cloudpipe/cloudpickle
        """
        ## TODO: create a writer for pickled objects.
        Log.debug('\nSerializing pipeline...')
        file = StringUtil.assert_not_empty_and_strip(file)
        with io.open(file, 'wb') as out:
            cloudpickle.dump(self, out)
        Log.debug('...done serializing pipeline.')

    @classmethod
    def deserialize(cls, file) -> DataProcessingPipeline:
        ## TODO: create a reader for pickled objects.
        Log.debug('Reading pipeline file from pickle...')
        with io.open(file, 'rb') as inp:
            pipeline: DataProcessingPipeline = cloudpickle.load(inp)
        if not isinstance(pipeline, DataProcessingPipeline):
            raise TypeError(f'Deserialized pipeline is must be an instance of {DataProcessingPipeline.__class__}.',
                            f'Found object of type {type_str(pipeline)}')
        Log.debug('...done reading pipeline file from pickle.')
        return pipeline


class PipelineUtil:
    def __init__(self):
        raise TypeError(f"Cannot create {str(self.__class__)} instances.")

    @classmethod
    def filter_schema_by_input_patterns(cls, schema: MLTypeSchema, input_patterns: Union[str, List[str]]):
        """
        :param schema: Dict where keys are column names and values are strings corresponding to MLTypes.
        :param input_patterns: String or list of strings, like '.*_TFIDF', 'NUMERIC', ['TEXT', '.*_TFIDF'] etc.
        :return: filtered schema, where we filter based on either the key (if string) or value (if MLType).
        """
        filtered_schema: Optional[Dict] = None
        if not isinstance(input_patterns, list):
            input_patterns = [input_patterns]
        filtered_cols = set()
        filtered_cols_ordered = []
        for input_pattern in input_patterns:
            input_mltype = MLType.from_str(input_pattern, raise_error=False)
            if isinstance(input_mltype, MLType):
                filtered_mltype_cols = set(keep_values(schema, input_mltype).keys())
                filtered_mltype_cols_list = list(filtered_mltype_cols)

                # This is used for handling cases when Numeric value is present inside String
                # Example: If there are two columns named TOP_1_PREDICTED_LABEL and TOP_10_PREDICTED_LABEL
                # Then output of sorted would be ['TOP_10_PREDICTED_LABEL', 'TOP_1_PREDICTED_LABEL']
                # This creates problem when using Uncertainty Calculator
                # To solve this, first check if all the column names have any digit present.
                # If yes, then sort it using key (REF: https://stackoverflow.com/a/49232907)
                # If no, then sort lexicographically

                if cls.__do_column_names_have_numeric_values(filtered_mltype_cols_list):
                    filtered_cols_ordered += [col for col in sorted(filtered_mltype_cols_list,
                                                                    key=lambda x: int(
                                                                        "".join([i for i in x if i.isdigit()])))
                                              if col not in filtered_cols_ordered]
                else:
                    filtered_cols_ordered += [col for col in sorted(filtered_mltype_cols_list)
                                              if col not in filtered_cols_ordered]
                filtered_cols = filtered_cols.union(filtered_mltype_cols)
            elif isinstance(input_pattern, str):
                filtered_str_pattern_cols = set(
                    filter_string_list(list(schema.keys()), input_pattern, ignorecase=True)
                )
                filtered_cols = filtered_cols.union(filtered_str_pattern_cols)
                filtered_cols_ordered += [col for col in sorted(list(filtered_str_pattern_cols))
                                          if col not in filtered_cols_ordered]
            else:
                raise AttributeError(f'input_pattern must be a str denoting regex or an MLType, found {input_pattern}')
        filtered_schema: Dict = keep_keys(schema, list(filtered_cols))
        filtered_schema_ordered = OrderedDict()
        for col in filtered_cols_ordered:
            filtered_schema_ordered[col] = filtered_schema[col]

        return filtered_schema_ordered

    @classmethod
    def __do_column_names_have_numeric_values(cls, filtered_cols_list: List[str]) -> bool:
        return all([True if any(char.isdigit() for char in col_name) else False for col_name in filtered_cols_list])
