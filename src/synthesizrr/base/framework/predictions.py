from abc import ABC
from typing import *
import pandas as pd
from copy import deepcopy
from synthesizrr.base.util import Registry, as_list, Schema, SchemaTemplate, equal, safe_validate_arguments, get_default, \
    StringUtil, accumulate, run_concurrent, any_are_none, remove_keys, format_exception_msg
from synthesizrr.base.data.FileMetadata import FileMetadata
from synthesizrr.base.data.sdf import ScalableDataFrame, ScalableSeries, ScalableSeriesOrRaw, ScalableOrRaw, is_scalable
from synthesizrr.base.constants import MLTypeSchema, DataLayout, TaskOrStr, Task, DataSplit, FILE_FORMAT_TO_FILE_ENDING_MAP, \
    FileFormat
from synthesizrr.base.framework.mixins import InputOutputDataMixin, SchemaValidationError
from synthesizrr.base.framework.task_data import Dataset
from pydantic import root_validator
from pydantic.typing import Literal

Predictions = "Predictions"
Visualization = "Visualization"
PredictionsSubclass = TypeVar('PredictionsSubclass', bound=Predictions)

PREDICTIONS_PARAMS_SAVE_FILE_NAME: str = 'predictions-metadata'
PREDICTIONS_PARAMS_SAVE_FILE_ENDING: str = '.predictions-params.json'


class Predictions(InputOutputDataMixin, Registry, ABC):
    _allow_multiple_subclasses: ClassVar[bool] = False
    _allow_subclass_override: ClassVar[bool] = True
    _allow_empty_features_schema: ClassVar[bool] = True

    index_col: ClassVar[Optional[str]] = None
    features_schema: ClassVar[Optional[MLTypeSchema]] = None
    ground_truths_schema: ClassVar[Optional[MLTypeSchema]]
    predictions_schema: ClassVar[MLTypeSchema]

    @classmethod
    def _pre_registration_hook(cls):
        cls.schema_template = SchemaTemplate.from_parts(
            index_col_template=cls.index_col,
            features_schema_template=cls.features_schema,
            ground_truths_schema_template=cls.ground_truths_schema,
            predictions_schema_template=cls.predictions_schema,
        )

    @root_validator(pre=True)
    def _set_predictions_params(cls, params: Dict) -> Dict:
        params['data_schema']: Schema = Schema.of(params['data_schema'], schema_template=cls.schema_template)
        # data_schema: Union[Schema, MLTypeSchema] = params['data_schema']
        # if isinstance(data_schema, dict):
        #     ## We need to infer the schema:
        #     schema_params: Dict = {}
        #     if params.get('data_split') is not None:
        #         data_split: DataSplit = DataSplit(params['data_split'])
        #         if data_split in {
        #             DataSplit.TRAIN,
        #             DataSplit.VALIDATION,
        #             DataSplit.TRAIN_VAL,
        #             DataSplit.TEST,
        #         }:
        #             schema_params: Dict = dict(
        #                 infer_ground_truths=True,
        #                 has_ground_truths=True,
        #             )
        #         elif data_split in {DataSplit.PREDICT, DataSplit.UNSUPERVISED}:
        #             schema_params: Dict = dict(
        #                 infer_ground_truths=False,
        #                 has_ground_truths=False,
        #             )
        #         else:
        #             raise ValueError(f'Unsupported value for `data_split`: {data_split}')
        #     try:
        #         data_schema: Schema = Schema.of(
        #             data_schema,
        #             schema_template=cls.schema_template,
        #             **schema_params
        #         )
        #     except Exception as e:
        #         if params.get('data_split') is None:
        #             raise ValueError(
        #                 f'Schema inference failed; try passing split=... when creating the Dataset instance. '
        #                 f'Schema inference error: {format_exception_msg(e)}'
        #             )
        #         raise e
        # assert isinstance(data_schema, Schema)
        # if not cls._allow_empty_features_schema and len(data_schema.features_schema) == 0:
        #     raise ValueError(
        #         f'Cannot have empty features schema in instance of class "{cls.class_name}". '
        #         f'This error might be resolved by specifying `data_split` when calling {Predictions.class_name}.of(...)'
        #     )
        # params['data_schema']: Schema = data_schema
        return params

    @classmethod
    def of(
            cls,
            data: Union[PredictionsSubclass, ScalableDataFrame, ScalableOrRaw, FileMetadata, Dict, str],
            **kwargs,
    ) -> PredictionsSubclass:
        if isinstance(data, Predictions):
            return data
        if isinstance(data, dict) and 'data' in data:
            return cls.of(**{**data, **kwargs})
        return cls._of(data=data, IOMixinBaseSubclass=Predictions, **kwargs)

    @classmethod
    @safe_validate_arguments
    def validate_schema(cls, data_schema: Schema):
        if data_schema.has_predictions is False:
            raise SchemaValidationError(f'Cannot have empty predictions schema in instance of class "{cls.class_name}"')
        inferred_schema: Schema = cls.schema_template.infer_from_columns(
            data_schema.columns_set,
            index_col=data_schema.index_col,
            has_predictions=True,
        )
        if data_schema.index_col != inferred_schema.index_col or \
                data_schema.predictions_schema != inferred_schema.predictions_schema:
            raise SchemaValidationError(
                f'Passed and inferred data schemas are different:\n'
                f'Passed:\n{data_schema}\n'
                f'Inferred:\n{inferred_schema}'
            )

    @classmethod
    @safe_validate_arguments
    def from_task_data(
            cls,
            data: Dataset,
            predictions: ScalableOrRaw,
            layout: Optional[DataLayout] = None,
            validated: bool = False,
            drop_features: bool = False,
            allow_non_unique_index: bool = False,
            **kwargs
    ) -> PredictionsSubclass:
        data.check_in_memory()
        input_sdf: ScalableDataFrame = data.data.to_frame(**kwargs)
        if layout is None:
            layout: DataLayout = input_sdf.layout
        input_sdf: ScalableDataFrame = input_sdf.as_layout(layout, **kwargs)
        index_col: str = data.data_schema.index_col
        index_col_data: ScalableSeries = input_sdf[index_col]
        if allow_non_unique_index is False and index_col_data.nunique() != len(index_col_data):
            raise ValueError(
                f'Index column must be unique while calling {cls.class_name}.from_task_data(...); '
                f'found index with {len(index_col_data)} elements but only {index_col_data.nunique()} unique values.'
            )
        predictions: ScalableDataFrame = cls._create_predictions(
            predictions,
            index=index_col_data,
            index_col=index_col,
            layout=layout,
            **kwargs,
        )
        pred_cols: Set[str] = predictions.columns_set
        inferred_pred_schema: Schema = cls.schema_template.infer_from_columns(
            pred_cols,
            index_col=index_col,
            infer_predictions=True,
            infer_ground_truths=False,
            infer_features=False,
        )
        if inferred_pred_schema.columns_set < set(pred_cols):
            raise ValueError(
                f'Could not infer how predictions dataframe columns map to MLTypes:\n'
                f'Predictions columns: {pred_cols}\n'
                f'Inferred MLTypes for predictions columns: {inferred_pred_schema.predictions_schema}\n'
                f'All inferred MLTypes: {inferred_pred_schema}'
            )
        predictions_schema: MLTypeSchema = inferred_pred_schema.predictions_schema

        ## TODO: update this line after supporting ScalableDataFrame.merge
        merged_sdf: ScalableDataFrame = ScalableDataFrame.of(
            input_sdf.pandas().merge(
                left_on=index_col,
                right=predictions.pandas(),
                right_on=index_col,
            ),
            layout=layout,
            **kwargs
        )

        merged_data_schema: Schema = data.data_schema.set_predictions(predictions_schema)
        if drop_features:
            remaining_columns: Set = merged_sdf.columns_set - set(merged_data_schema.features_schema.keys())
            merged_sdf: ScalableDataFrame = merged_sdf[sorted(list(remaining_columns))]
            merged_data_schema: Schema = merged_data_schema.drop_features()
        kwargs.pop('task', None)
        kwargs.pop('data_schema', None)
        kwargs['data_split'] = get_default(
            kwargs.get('data_split', None),
            data.data_split,
            DataSplit.PREDICT,
        )
        return cls(
            data=merged_sdf,
            task=data.task,
            data_idx=data.data_idx,
            data_position=data.data_position,
            data_schema=merged_data_schema,
            validated=validated,
            **kwargs
        )

    @classmethod
    @safe_validate_arguments
    def from_dataframe(
            cls,
            data: ScalableOrRaw,
            data_schema: Optional[Union[Schema, MLTypeSchema]] = None,
            validated: bool = False,
            layout: Optional[DataLayout] = None,
            **kwargs
    ) -> PredictionsSubclass:
        data: ScalableDataFrame = ScalableDataFrame.of(data, layout=layout, **kwargs)
        if data_schema is None:
            data_schema: Schema = cls.schema_template.infer_from_columns(data.columns_set, has_predictions=True)
        else:
            data_schema: Schema = Schema.of(data_schema, schema_template=cls.schema_template, has_predictions=True)
        return cls(
            data=data,
            data_schema=data_schema,
            validated=validated,
            **kwargs
        )

    @classmethod
    @safe_validate_arguments
    def from_parts(
            cls,
            index_col: str,
            index: ScalableSeriesOrRaw,
            predictions: ScalableOrRaw,
            features: Optional[ScalableOrRaw] = None,
            ground_truths: Optional[ScalableOrRaw] = None,
            layout: Optional[DataLayout] = None,
            validated: bool = False,
            **kwargs
    ) -> PredictionsSubclass:
        """
        Initializes Predictions instance that has data and data_schema
        Data should be in the predefined format eg:
                        INDEX   predicted_score   ground_truth
                0      ASIN_1   21.456              22.087
                1      ASIN_2   98.41               96.746

        Data schema should also be in a predefined format eg:
            {
                'INDEX' : INDEX
                'predicted_score_1' : PREDICTED
                'ground_truth_1'    : GROUND_TRUTH
            }
        """
        ## TODO: remove the conversion to Pandas after supporting ScalableDataFrame.merge_multi
        index: ScalableSeries = cls._create_index(index, name=index_col, layout=layout, **kwargs)
        sdfs_to_merge: List[ScalableDataFrame] = [
            index.to_frame().as_layout(DataLayout.PANDAS)
        ]
        predictions: ScalableDataFrame = cls._create_predictions(
            predictions,
            index=index,
            index_col=index_col,
            layout=layout,
            **kwargs
        )
        sdfs_to_merge.append(predictions)
        if features is not None:
            features: ScalableDataFrame = cls._create_features(
                features,
                index=index,
                index_col=index_col,
                layout=layout,
                **kwargs
            )
            sdfs_to_merge.append(features)
        if ground_truths is not None:
            ground_truths: ScalableDataFrame = cls._create_ground_truths(
                ground_truths,
                index=index,
                index_col=index_col,
                layout=layout,
                **kwargs
            )
            sdfs_to_merge.append(ground_truths)
        ## TODO: remove conversion to Pandas after supporting ScalableDataFrame.merge_multi
        merged_sdf: ScalableDataFrame = ScalableDataFrame.of(
            pd.concat([sdf.pandas() for sdf in sdfs_to_merge], axis=1)
        ).as_layout(layout, **kwargs)

        merged_cols: List[str] = merged_sdf.columns
        merged_data_schema: Schema = cls.schema_template.infer_from_columns(
            merged_cols,
            index_col=index_col,
        )
        return cls(
            data=merged_sdf,
            data_schema=merged_data_schema,
            validated=validated,
            **kwargs
        )

    @classmethod
    def _create_predictions(
            cls,
            predictions: ScalableOrRaw,
            index: ScalableSeries,
            index_col: str,
            layout: DataLayout,
            **kwargs,
    ) -> ScalableDataFrame:
        if not is_scalable(predictions):
            if ScalableDataFrame.detect_layout(predictions, raise_error=False) is None:
                raise ValueError(
                    f'Cannot detect columns for predictions of type {type(predictions)}; '
                    f'please ensure you set column names for predictions in '
                    f'the .predict_step(...) function.\n'
                    f'If you have a list of predicted values/objects, an easy fix is to return a dict '
                    f'with a single column, named according to the SchemaTemplate: {cls.schema_template}.\n'
                    f'E.g. if are predicting a list of regression scores, then '
                    f'.predict_step(...) should return a dict like this:\n'
                    '{"predicted_score": [123.1, 64.2, 420.69, ...]}'
                )
        predictions: ScalableDataFrame = ScalableDataFrame.of(predictions, layout=layout, **kwargs)
        predictions.loc[:, index_col] = index
        return predictions

    # @classmethod
    # def concat(
    #         cls,
    #         predictions_list: List[Predictions],
    #         layout: Optional[DataLayout] = None,
    # ) -> Predictions:
    #     predictions_list: List[Predictions] = as_list(predictions_list)
    #     if len(predictions_list) == 0:
    #         raise ValueError("Cannot concatenate empty list of predictions.")
    #     subclasses: List[Type[Predictions]] = [type(pred) for pred in predictions_list]
    #     if not equal(*subclasses):
    #         raise ValueError(
    #             f'Cannot concatenate {Predictions.__name__} instances of different subclasses. '
    #             f'Found following {len(subclasses)} unique subclasses: {set(subclasses)}'
    #         )
    #     PredictionsSubclass: Type[Predictions] = subclasses[0]
    #
    #     schemas: List[Schema] = [pred.data_schema for pred in predictions_list]
    #     if not equal(*schemas):
    #         raise ValueError(
    #             f'Cannot concatenate {Predictions.__name__} subclasses with different schemas. '
    #             f'Found {len(schemas)} schemas in total:\n{schemas}'
    #         )
    #     data_schema: Schema = schemas[0]
    #
    #     other_data: List[Dict] = [
    #         pred.dict(exclude={'data', 'data_schema', 'data_idx', 'data_position'})
    #         for pred in predictions_list
    #     ]
    #     if not equal(*other_data):
    #         raise ValueError(
    #             f'Cannot concatenate {Predictions.__name__} subclasses with different data fields. '
    #             f'Found {len(other_data)} data dicts in total:\n{other_data}'
    #         )
    #     other_data: Dict = other_data[0]
    #
    #     data: ScalableDataFrame = ScalableDataFrame.concat(
    #         [pred.data for pred in predictions_list],
    #         reset_index=True,
    #         layout=layout,
    #     )
    #     return PredictionsSubclass(
    #         data=data,
    #         data_schema=data_schema,
    #         **other_data,
    #     )

    @classmethod
    def concat(
            cls,
            data_list: Optional[Union[List[PredictionsSubclass], PredictionsSubclass]],
            **kwargs,
    ) -> Optional[PredictionsSubclass]:
        return cls._concat(io_data_list=data_list, IOMixinBaseSubclass=Predictions, **kwargs)

    def viz(self, *args, **kwargs) -> Visualization:
        """Alias for .visualize()"""
        return self.visualize(*args, **kwargs)

    @safe_validate_arguments
    def visualize(
            self,
            name: Optional[str] = None,
            **kwargs,
    ) -> Visualization:
        from synthesizrr.base.framework.visualize import Visualization
        ## Should show an interactive plot.
        return Visualization.plot(data=self, name=name, **kwargs)

    @safe_validate_arguments
    def as_task_data(
            self,
            task: Optional[TaskOrStr] = None,
            deep: bool = False,
            schema_action: Literal['drop_predictions', 'predictions_to_features'] = 'drop_predictions',
    ) -> Dataset:
        task: Optional[TaskOrStr] = get_default(task, self.task)
        data_schema: Schema = deepcopy(self.data_schema)
        if schema_action == 'drop_predictions':
            data_schema: Schema = data_schema.drop_predictions()
        elif schema_action == 'predictions_to_features':
            data_schema: Schema = data_schema.predictions_to_features()
        return Dataset.of(
            data_split=self.data_split,
            task=task,
            data=self.data.copy(deep=deep),
            data_schema=data_schema,
            index_col=self.index_col,
        )


def load_predictions(
        predictions_source: Optional[Union[FileMetadata, str]],
        file_ending: Optional[str] = None,
        *,
        task: Optional[TaskOrStr] = None,
        **kwargs
) -> Optional[PredictionsSubclass]:
    if predictions_source is None:
        return
    ## Don't want to mistake with similar params used for prediction:
    from synthesizrr.base.data.reader import DataFrameReader, JsonReader
    predictions_source: FileMetadata = FileMetadata.of(predictions_source)
    reader: DataFrameReader = DataFrameReader.of(
        predictions_source.format,
        **kwargs,
    )
    json_reader: JsonReader = JsonReader()
    if not predictions_source.is_path_valid_dir():
        data: ScalableDataFrame = reader.read_metadata(
            file=predictions_source,
            **kwargs
        )
        file_endings: List[str] = as_list(FILE_FORMAT_TO_FILE_ENDING_MAP[predictions_source.format])
        if file_ending is None:
            for file_ending in file_endings:
                if predictions_source.path.endswith(file_ending):
                    break
                file_ending: Optional[str] = None
            if file_ending is None:
                file_ending: Optional[str] = FileMetadata.detect_file_ending(predictions_source.path)
        if file_ending is None:
            raise ValueError(f'Cannot detect file ending from path: "{predictions_source.path}"')
        predictions_params_file: FileMetadata = FileMetadata.of(
            StringUtil.remove_suffix(
                predictions_source.path,
                suffix=file_ending
            ) + PREDICTIONS_PARAMS_SAVE_FILE_ENDING,
            format=FileFormat.JSON,
        )
        predictions_params: Dict = json_reader.read_metadata(predictions_params_file)
        task: TaskOrStr = get_default(task, predictions_params.get('task'))
        return Predictions.of(**{
            **predictions_params,
            **dict(
                task=task,
                data=data,
            ),
        })
    else:
        ## There should be at least one .predictions-params.json file
        predictions_params_fpaths: List[str] = predictions_source.list(
            file_glob=f'*{PREDICTIONS_PARAMS_SAVE_FILE_ENDING}'
        )
        if len(predictions_params_fpaths) == 0:
            raise ValueError(
                f'No file ending in "{PREDICTIONS_PARAMS_SAVE_FILE_ENDING}" was found in "{predictions_source.path}"; '
                f'this file is required to create a {Predictions.class_name} object; please check the directory is '
                f'correct.'
            )
        if len(predictions_params_fpaths) > 1:
            ## Ensure all are the same:
            predictions_params_list: List[Dict] = accumulate([
                run_concurrent(
                    json_reader.read_metadata,
                    FileMetadata.of(
                        predictions_params_fpath,
                        format=FileFormat.JSON,
                    )
                )
                for predictions_params_fpath in predictions_params_fpaths
            ])
            predictions_params_list: List[Dict] = [
                remove_keys(predictions_params_d, ['data_idx', 'data_position', 'validated'])
                for predictions_params_d in predictions_params_list
            ]
            predictions_params_set: Set[str] = set([
                StringUtil.stringify(predictions_params_d)
                for predictions_params_d in predictions_params_list
            ])
            if len(predictions_params_set) > 1:
                raise ValueError(
                    f'Found {len(predictions_params_fpaths)} files ending with "{PREDICTIONS_PARAMS_SAVE_FILE_ENDING}" '
                    f'in directory "{predictions_source.path}", however these files have different parameters and thus '
                    f'the predictions cannot be merged. Different parameters found:'
                    f'\n{predictions_params_set}'
                )
        predictions_params_file: FileMetadata = FileMetadata.of(
            predictions_params_fpaths[0],
            format=FileFormat.JSON,
        )
        predictions_params: Dict = json_reader.read_metadata(predictions_params_file)
        file_endings: List[str] = as_list(FILE_FORMAT_TO_FILE_ENDING_MAP[predictions_source.format])
        if file_ending is None:
            for file_ending in file_endings:
                if len(predictions_source.list(file_glob=f'*{file_ending}')) > 0:
                    break
                file_ending: Optional[str] = None
        if file_ending is None:
            raise ValueError(
                f'No files ending with {StringUtil.join_human(file_endings, final_join="or")} '
                f'exist in directory: "{predictions_source.path}"'
            )
        data: ScalableDataFrame = reader.read_metadata(
            predictions_source,
            file_glob=f'*{file_ending}',
            **kwargs
        )
        return Predictions.of(
            **predictions_params,
            data=data,
        )


def save_predictions(
        predictions: Optional[PredictionsSubclass],
        predictions_destination: Optional[Union[FileMetadata, Dict, str]],
        *,
        overwrite: bool = True,
        **kwargs
) -> NoReturn:
    from synthesizrr.base.data.writer import DataFrameWriter, JsonWriter

    if any_are_none(predictions, predictions_destination):
        return
    predictions_destination: FileMetadata = FileMetadata.of(predictions_destination)

    writer: DataFrameWriter = DataFrameWriter.of(
        predictions_destination.format,
        **kwargs,
    )
    json_writer: JsonWriter = JsonWriter()

    if predictions_destination.is_path_valid_dir():
        predictions_params_file: FileMetadata = predictions_destination.file_in_dir(
            PREDICTIONS_PARAMS_SAVE_FILE_NAME,
            file_ending=PREDICTIONS_PARAMS_SAVE_FILE_ENDING,
            return_metadata=True,
        )
        predictions_params_file: FileMetadata = predictions_params_file.update_params(format=FileFormat.JSON)
    else:
        kwargs['single_file']: bool = True  ## Passed to DataFrameWriter, writes a single file.
        file_endings: List[str] = as_list(FILE_FORMAT_TO_FILE_ENDING_MAP[predictions_destination.format])
        file_ending: Optional[str] = None
        for file_ending in file_endings:
            if predictions_destination.path.endswith(file_ending):
                break
            file_ending: Optional[str] = None
        if file_ending is None:
            file_ending: Optional[str] = FileMetadata.detect_file_ending(predictions_destination.path)
        if file_ending is None:
            raise ValueError(f'Cannot detect file ending from path: "{predictions_destination.path}"')
        predictions_params_file: FileMetadata = FileMetadata.of(
            StringUtil.remove_suffix(
                predictions_destination.path,
                suffix=file_ending
            ) + PREDICTIONS_PARAMS_SAVE_FILE_ENDING,
            format=FileFormat.JSON,
        )

    writer.write_metadata(
        file=predictions_destination,
        data=predictions.data,
        overwrite=overwrite,
        **kwargs
    )
    json_writer.write_metadata(
        file=predictions_params_file,
        data=predictions.dict(exclude={'data', 'data_idx', 'data_position', 'validated'}),
        overwrite=overwrite,
    )

#
# class PredictionsUtil:
#     INDEX_COL_NAME = 'index'
#     INDEX_COL_REGEX = re.compile(INDEX_COL_NAME)
#     PREDICTED_SCORES_I_TEMPLATE = 'predicted_score_%s'
#     GROUND_TRUTH_I_TEMPLATE = 'ground_truth_%s'
#
#     @classmethod
#     def get_index_df(cls, index: ScalableSeries) -> PandasDataFrame:
#         """
#         Returns a Pandas Series with INDEX column named as "INDEX"
#         eg:
#                         INDEX
#             ASIN_1      ASIN_1
#             ASIN_2      ASIN_2
#         @param index:
#         @return: PandasDataFrame
#         """
#         index_df = index.pandas()
#         index_df = index_df.to_frame(cls.INDEX_COL_NAME).set_index(index_df)
#         return index_df
#
#     @classmethod
#     def get_scores_df(cls, index: ScalableSeries, scores: ScalableDataFrame,
#                       predicted_scores_template: str) -> PandasDataFrame:
#         """
#         Returns a PandasDataFrame with INDEX column and predicted_score columns
#         eg:
#                     INDEX   predicted_score_1   predicted_score_2
#             0      ASIN_1   0.09823             0.99873
#             1      ASIN_2   0.56853             0.67352
#         @param predicted_scores_template:
#         @param index:
#         @param scores:
#         @return: PandasDataFrame
#         """
#
#         if isinstance(scores, ScalableSeries):
#             scores = ScalableDataFrame.of(scores)
#
#         if StringUtil.is_empty(predicted_scores_template):
#             predicted_scores_template = cls.PREDICTED_SCORES_I_TEMPLATE
#
#         n_cols = scores.shape[1]
#         scores_cols = cls.create_predicted_scores_col(index=n_cols, predicted_scores_template=predicted_scores_template)
#         return PredictionsUtil.create_df(index=index, data=scores, cols=scores_cols)
#
#     @classmethod
#     def get_ground_truths_df(
#             cls,
#             index: ScalableSeries,
#             ground_truths: ScalableDataFrame,
#             ground_truths_template: str
#     ) -> PandasDataFrame:
#         """
#         Returns a PandasDataFrame with INDEX column and ground_truth columns
#         eg:
#                     INDEX   ground_truth_1   ground_truth_2
#             0      ASIN_1   category_1       category_8
#             1      ASIN_2   category_4       None
#         @param ground_truths_template:
#         @param index:
#         @param ground_truths:
#         @return: PandasDataFrame
#         """
#         if isinstance(ground_truths, ScalableSeries):
#             ground_truths = ScalableDataFrame.of(ground_truths)
#
#         if StringUtil.is_empty(ground_truths_template):
#             ground_truths_template = cls.GROUND_TRUTH_I_TEMPLATE
#
#         n_cols = ground_truths.shape[1]
#         ground_truth_cols = cls.create_ground_truth_col(index=n_cols, ground_truth_template=ground_truths_template)
#         return PredictionsUtil.create_df(index=index, data=ground_truths, cols=ground_truth_cols)
#
#     @staticmethod
#     def create_df(index: ScalableSeries, data: ScalableDataFrame, cols: List[str]) -> PandasDataFrame:
#         """
#         Converting to Pandas due to following reasons
#             1. set_index method is not implemented in sdf. Reasoning behind this is List of Dicts wont have indices
#             2. concat method of sdf doesn't support column-wise concatenation
#             3. #TODO merge method is yet to be implemented by sdf
#             4. SDF currently doesn't support pandas convention of setting column names df.columns = [list_of_cols]
#
#         This is done only once
#         @param index:
#         @param data:
#         @param cols:
#         @return: PandasDataFrame
#         """
#         df = data.pandas()  # Converting to Pandas as list of dict wont have set_index methods
#         index_df = index.pandas()
#         df = df.set_index(index_df)
#         df.columns = cols
#         return df
#
#     @classmethod
#     def create_predicted_scores_col(cls, index: int, predicted_scores_template: str):
#         assert isinstance(index, int)
#         cols = []
#         for i in range(1, 1 + index):
#             cols.append(predicted_scores_template % (i))
#         return cols
#
#     @classmethod
#     def create_ground_truth_col(cls, index: int, ground_truth_template: str):
#         assert isinstance(index, int)
#         cols = []
#         for i in range(1, 1 + index):
#             cols.append(ground_truth_template % (i))
#         return cols
#
#     @classmethod
#     def filter_index_col_names(cls, columns: List[str]) -> List[str]:
#         index_regex = re.compile(cls.INDEX_COL_NAME)
#         return list(filter(index_regex.match, columns))
#
#     @classmethod
#     def filter_ground_truth_col_names(cls, columns: List[str], ground_truth_template: str) -> List[str]:
#         ground_truth_regex = re.compile(ground_truth_template.replace('%s', '\\d+'))
#         return list(filter(ground_truth_regex.match, columns))
#
#     @classmethod
#     def filter_predicted_score_col_names(cls, columns: List[str], predicted_scores_template: str) -> List[str]:
#         predicted_scores_regex = re.compile(predicted_scores_template.replace('%s', '\\d+'))
#         return list(filter(predicted_scores_regex.match, columns))
