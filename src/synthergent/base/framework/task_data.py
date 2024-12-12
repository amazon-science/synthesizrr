from typing import *
import os, math
from abc import ABC
from synthergent.base.data.sdf import ScalableDataFrame, ScalableDataFrameOrRaw, ScalableSeries, ScalableOrRaw, to_sdf
from synthergent.base.util import Parameters, Registry, FractionalBool, resolve_fractional_bool, Schema, SchemaTemplate, \
    StringUtil, safe_validate_arguments, get_default, run_concurrent, accumulate, as_list, \
    str_normalize, optional_dependency, set_param_from_alias, any_are_none, remove_keys, format_exception_msg
from synthergent.base.constants import DataSplit, DataLayout, MLTypeSchema, Parallelize, FileFormat, MLType, ASSET_ML_TYPES, \
    FILE_FORMAT_TO_FILE_ENDING_MAP
from synthergent.base.data import FileMetadata, Asset
from synthergent.base.data.reader import DataFrameReader, AssetReader
from synthergent.base.framework.mixins import InputOutputDataMixin, TaskOrStr, SchemaValidationError
from pydantic import root_validator, conint
from pydantic.typing import Literal

Dataset = "Dataset"
Visualization = "Visualization"
Datasets = "Datasets"
DatasetSubclass = TypeVar('DatasetSubclass', bound=Dataset)


class Dataset(InputOutputDataMixin, Registry, ABC):
    _allow_multiple_subclasses: ClassVar[bool] = False
    _allow_subclass_override: ClassVar[bool] = True
    _allow_empty_features_schema: ClassVar[bool] = False

    index_col: ClassVar[Optional[str]] = None
    features_schema: ClassVar[Optional[MLTypeSchema]] = None
    ground_truths_schema: ClassVar[MLTypeSchema]

    @classmethod
    def _pre_registration_hook(cls):
        cls.schema_template = SchemaTemplate.from_parts(
            index_col_template=cls.index_col,
            features_schema_template=cls.features_schema,
            ground_truths_schema_template=cls.ground_truths_schema,
        )

    @classmethod
    def of(
            cls,
            data: Union[DatasetSubclass, ScalableDataFrame, ScalableOrRaw, FileMetadata, Dict, str],
            **kwargs,
    ) -> DatasetSubclass:
        if isinstance(data, Dataset):
            return data
        if isinstance(data, dict) and 'data' in data:
            return cls.of(**{**data, **kwargs})
        return cls._of(data=data, IOMixinBaseSubclass=Dataset, **kwargs)

    @classmethod
    def concat(
            cls,
            data_list: Optional[Union[List[DatasetSubclass], DatasetSubclass]],
            **kwargs,
    ) -> Optional[DatasetSubclass]:
        return cls._concat(io_data_list=data_list, IOMixinBaseSubclass=Dataset, **kwargs)

    #
    # @classmethod
    # def from_hf(
    #         cls,
    #         name: str,
    #         **kwargs,
    # ) -> Optional[Dataset]:
    #     from datasets import load_dataset
    #     data = to_sdf(load_dataset(name, **kwargs))

    @root_validator(pre=True)
    def _set_dataset_params(cls, params: Dict) -> Dict:
        data_schema: Union[Schema, MLTypeSchema] = params['data_schema']
        if isinstance(data_schema, dict):
            ## We need to infer the schema:
            schema_params: Dict = {}
            if params.get('data_split') is not None:
                data_split: DataSplit = DataSplit(params['data_split'])
                if data_split in {
                    DataSplit.TRAIN,
                    DataSplit.VALIDATION,
                    DataSplit.TRAIN_VAL,
                    DataSplit.TEST,
                }:
                    schema_params: Dict = dict(
                        infer_ground_truths=True,
                        has_ground_truths=True,
                    )
                elif data_split in {DataSplit.PREDICT, DataSplit.UNSUPERVISED}:
                    schema_params: Dict = dict(
                        infer_ground_truths=False,
                        has_ground_truths=False,
                    )
                else:
                    raise ValueError(f'Unsupported value for `data_split`: {data_split}')
            try:
                data_schema: Schema = Schema.of(
                    data_schema,
                    schema_template=cls.schema_template,
                    **schema_params
                )
            except Exception as e:
                if params.get('data_split') is None:
                    raise ValueError(
                        f'Schema inference failed; try passing split=... when creating the Dataset instance. '
                        f'Schema inference error: {format_exception_msg(e)}'
                    )
                raise e
        assert isinstance(data_schema, Schema)
        if not cls._allow_empty_features_schema and len(data_schema.features_schema) == 0:
            raise ValueError(
                f'Cannot have empty features schema in instance of class "{cls.class_name}". '
                f'This error might be resolved by specifying `data_split` when calling {Dataset.class_name}.of(...)'
            )
        params['data_schema']: Schema = data_schema
        return params

    @classmethod
    @safe_validate_arguments
    def validate_schema(cls, data_schema: Schema):
        ## Check index column is same.
        if len(cls.schema_template.index_col_template.matches(data_schema.index_col)) != 1:
            raise SchemaValidationError(
                f'Index column from passed schema does not match schema-template:\n'
                f'Passed Schema:\n{data_schema}\n'
                f'Schema-Template:\n{cls.schema_template}'
            )

        if data_schema.has_ground_truths:
            inferred_schema: Schema = cls.schema_template.infer_from_columns(
                data_schema.columns_set - set(data_schema.features_schema.keys()),
                index_col=data_schema.index_col,
                has_ground_truths=True,
            )
            if data_schema.index_col != inferred_schema.index_col or \
                    (data_schema.ground_truths_schema != inferred_schema.ground_truths_schema):
                raise SchemaValidationError(
                    f'Passed and inferred data schemas are different:\n'
                    f'Passed:\n{data_schema}\n'
                    f'Inferred:\n{inferred_schema}'
                )

    def viz(self, *args, **kwargs) -> Visualization:
        """Alias for .visualize()"""
        return self.visualize(*args, **kwargs)

    def visualize(
            self,
            name: Optional[str] = None,
            **kwargs,
    ) -> Visualization:
        from synthergent.base.framework.visualize import Visualization
        ## Should show an interactive plot.
        return Visualization.plot(data=self, name=name, **kwargs)


class Datasets(Parameters):
    name: Optional[str] = None
    datasets: Dict[DataSplit, Dataset]

    @classmethod
    @safe_validate_arguments
    def of(cls, **datasets) -> Datasets:
        try:
            return cls(datasets=datasets)
        except Exception as e:
            raise ValueError(f'Error creating {cls.class_name}:\n{format_exception_msg(e)}')

    @root_validator(pre=True)
    def set_datasets(cls, params: Dict):
        name: Optional[str] = None
        if params.get('name') is not None:
            name: str = params['name']
        datasets_dict: Dict[DataSplit, Dataset] = {}
        for data_split, dataset in params['datasets'].items():
            if dataset is None:
                continue
            data_split: DataSplit = DataSplit.from_str(data_split)
            if isinstance(dataset, dict):
                dataset['data_split'] = data_split
                if name is not None:
                    dataset['name'] = name  ## Copy Datasets name into each dataset
                dataset: Dataset = Dataset.of(**dataset)
            if not isinstance(dataset, Dataset):
                raise ValueError(
                    f'Unsupported value for dataset: '
                    f'{type(dataset)} with following value:\n{dataset}'
                )
            dataset: Dataset = dataset.update_params(data_split=data_split)
            datasets_dict[data_split] = dataset
        params['datasets'] = datasets_dict
        return params

    def splits(self) -> Set[DataSplit]:
        return set(self.datasets.keys())

    def __getattr__(self, attr_name: Union[str, DataSplit]) -> DatasetSubclass:
        if DataSplit.matches_any(attr_name):
            return self.datasets.get(DataSplit.from_str(attr_name))
        raise AttributeError(
            f'`{attr_name}` is neither an attribute of {self.class_name} '
            f'nor a valid value of {DataSplit}.'
        )

    def __getitem__(self, attr_name: Union[str, DataSplit]) -> DatasetSubclass:
        if DataSplit.matches_any(attr_name):
            return self.datasets.get(DataSplit.from_str(attr_name))
        raise KeyError(f'Unknown value for {DataSplit}: {attr_name}')

    @safe_validate_arguments
    def drop(self, data_split: Union[List[DataSplit], DataSplit]) -> Datasets:
        data_split: List[DataSplit] = as_list(data_split)
        return Datasets(
            **self.dict(exclude={'datasets'}),
            datasets={
                split: dataset
                for split, dataset in self.datasets.items()
                if split not in data_split
            }
        )

    @safe_validate_arguments
    def read(self, **kwargs) -> Datasets:
        return Datasets(
            **self.dict(exclude={'datasets'}),
            datasets={
                split: dataset.read(**kwargs)
                for split, dataset in self.datasets.items()
            }
        )


TaskData = Dataset

DATASET_PARAMS_SAVE_FILE_NAME: str = 'dataset-metadata'
DATASET_PARAMS_SAVE_FILE_ENDING: str = '.dataset-params.json'


def load_dataset(
        dataset_source: Optional[Union[FileMetadata, str]],
        file_ending: Optional[str] = None,
        *,
        task: Optional[TaskOrStr] = None,
        **kwargs
) -> Optional[DatasetSubclass]:
    if dataset_source is None:
        return
    ## Don't want to mistake with similar params used for prediction:
    from synthergent.base.data.reader import DataFrameReader, JsonReader
    dataset_source: FileMetadata = FileMetadata.of(dataset_source)
    reader: DataFrameReader = DataFrameReader.of(
        dataset_source.format,
        **kwargs,
    )
    json_reader: JsonReader = JsonReader()
    if not dataset_source.is_path_valid_dir():
        data: ScalableDataFrame = reader.read_metadata(
            file=dataset_source,
            **kwargs
        )
        file_endings: List[str] = as_list(FILE_FORMAT_TO_FILE_ENDING_MAP[dataset_source.format])
        if file_ending is None:
            for file_ending in file_endings:
                if dataset_source.path.endswith(file_ending):
                    break
                file_ending: Optional[str] = None
            if file_ending is None:
                file_ending: Optional[str] = FileMetadata.detect_file_ending(dataset_source.path)
        if file_ending is None:
            raise ValueError(f'Cannot detect file ending from path: "{dataset_source.path}"')
        dataset_params_file: FileMetadata = FileMetadata.of(
            StringUtil.remove_suffix(
                dataset_source.path,
                suffix=file_ending
            ) + DATASET_PARAMS_SAVE_FILE_ENDING,
            format=FileFormat.JSON,
        )
        dataset_params: Dict = json_reader.read_metadata(dataset_params_file)
        task: TaskOrStr = get_default(task, dataset_params.get('task'))
        return Dataset.of(**{
            **dataset_params,
            **dict(
                task=task,
                data=data,
            ),
        })
    else:
        ## There should be at least one .dataset-params.json file
        dataset_params_fpaths: List[str] = dataset_source.list(
            file_glob=f'*{DATASET_PARAMS_SAVE_FILE_ENDING}'
        )
        if len(dataset_params_fpaths) == 0:
            raise ValueError(
                f'No file ending in "{DATASET_PARAMS_SAVE_FILE_ENDING}" was found in "{dataset_source.path}"; '
                f'this file is required to create a {Dataset.class_name} object; please check the directory is '
                f'correct.'
            )
        if len(dataset_params_fpaths) > 1:
            ## Ensure all are the same:
            dataset_params_list: List[Dict] = accumulate([
                run_concurrent(
                    json_reader.read_metadata,
                    FileMetadata.of(
                        dataset_params_fpath,
                        format=FileFormat.JSON,
                    )
                )
                for dataset_params_fpath in dataset_params_fpaths
            ])
            dataset_params_list: List[Dict] = [
                remove_keys(dataset_params_d, ['data_idx', 'data_position', 'validated'])
                for dataset_params_d in dataset_params_list
            ]
            dataset_params_set: Set[str] = set([
                StringUtil.stringify(dataset_params_d)
                for dataset_params_d in dataset_params_list
            ])
            if len(dataset_params_set) > 1:
                raise ValueError(
                    f'Found {len(dataset_params_fpaths)} files ending with "{DATASET_PARAMS_SAVE_FILE_ENDING}" '
                    f'in directory "{dataset_source.path}", however these files have different parameters and thus '
                    f'the dataset cannot be merged. Different parameters found:'
                    f'\n{dataset_params_set}'
                )
        dataset_params_file: FileMetadata = FileMetadata.of(
            dataset_params_fpaths[0],
            format=FileFormat.JSON,
        )
        dataset_params: Dict = json_reader.read_metadata(dataset_params_file)
        file_endings: List[str] = as_list(FILE_FORMAT_TO_FILE_ENDING_MAP[dataset_source.format])
        if file_ending is None:
            for file_ending in file_endings:
                if len(dataset_source.list(file_glob=f'*{file_ending}')) > 0:
                    break
                file_ending: Optional[str] = None
        if file_ending is None:
            raise ValueError(
                f'No files ending with {StringUtil.join_human(file_endings, final_join="or")} '
                f'exist in directory: "{dataset_source.path}"'
            )
        data: ScalableDataFrame = reader.read_metadata(
            dataset_source,
            file_glob=f'*{file_ending}',
            **kwargs
        )
        return Dataset.of(
            **dataset_params,
            data=data,
        )


def save_dataset(
        dataset: Optional[DatasetSubclass],
        dataset_destination: Optional[Union[FileMetadata, Dict, str]],
        *,
        overwrite: bool = True,
        **kwargs
) -> NoReturn:
    from synthergent.base.data.writer import DataFrameWriter, JsonWriter

    if any_are_none(dataset, dataset_destination):
        return
    dataset_destination: FileMetadata = FileMetadata.of(dataset_destination)

    writer: DataFrameWriter = DataFrameWriter.of(
        dataset_destination.format,
        **kwargs,
    )
    json_writer: JsonWriter = JsonWriter()

    if dataset_destination.is_path_valid_dir():
        dataset_params_file: FileMetadata = dataset_destination.file_in_dir(
            DATASET_PARAMS_SAVE_FILE_NAME,
            file_ending=DATASET_PARAMS_SAVE_FILE_ENDING,
            return_metadata=True,
        )
        dataset_params_file: FileMetadata = dataset_params_file.update_params(format=FileFormat.JSON)
    else:
        kwargs['single_file']: bool = True  ## Passed to DataFrameWriter, writes a single file.
        file_endings: List[str] = as_list(FILE_FORMAT_TO_FILE_ENDING_MAP[dataset_destination.format])
        file_ending: Optional[str] = None
        for file_ending in file_endings:
            if dataset_destination.path.endswith(file_ending):
                break
            file_ending: Optional[str] = None
        if file_ending is None:
            file_ending: Optional[str] = FileMetadata.detect_file_ending(dataset_destination.path)
        if file_ending is None:
            raise ValueError(f'Cannot detect file ending from path: "{dataset_destination.path}"')
        dataset_params_file: FileMetadata = FileMetadata.of(
            StringUtil.remove_suffix(
                dataset_destination.path,
                suffix=file_ending
            ) + DATASET_PARAMS_SAVE_FILE_ENDING,
            format=FileFormat.JSON,
        )

    writer.write_metadata(
        file=dataset_destination,
        data=dataset.data,
        overwrite=overwrite,
        **kwargs
    )
    json_writer.write_metadata(
        file=dataset_params_file,
        data=dataset.dict(exclude={'data', 'data_idx', 'data_position', 'validated'}),
        overwrite=overwrite,
    )
