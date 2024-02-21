from typing import *
import os, math
from copy import deepcopy
from abc import ABC, abstractmethod
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.core.frame import Series as PandasSeries, DataFrame as PandasDataFrame
from dask.dataframe.core import Scalar as DaskScalar, Series as DaskSeries, DataFrame as DaskDataFrame
from synthesizrr.base.data import ScalableDataFrame, ScalableDataFrameOrRaw, ScalableSeries, ScalableOrRaw, \
    ScalableSeriesOrRaw, is_scalable, FileMetadata, Asset
from synthesizrr.base.data.reader import AssetReader, DataFrameReader
from synthesizrr.base.data.sdf import TensorScalableSeries
from synthesizrr.base.util import MutableParameters, Registry, FractionalBool, resolve_fractional_bool, Schema, SchemaTemplate, \
    StringUtil, random_sample, safe_validate_arguments, get_default, run_concurrent, accumulate, as_list, \
    str_normalize, optional_dependency, create_progress_bar, TqdmProgressBar, is_null, \
    format_exception_msg, is_abstract, only_item, is_list_or_set_like, accumulate, equal, as_set, \
    pd_partial_column_order, get_fn_args, ProgressBar
from synthesizrr.base.util.schema import Schema, SchemaTemplate
from synthesizrr.base.constants import Task, TaskOrStr, MLTypeSchema, MLType, DataLayout, MLTypeOrStr, DataPosition, \
    DataSplit, AVAILABLE_TENSOR_TYPES, SHORTHAND_TO_TENSOR_LAYOUT_MAP, Parallelize, FileFormat, ASSET_ML_TYPES, \
    TensorShortHand, Alias
from pydantic import Extra, root_validator, validator, conint, confloat
from pydantic.typing import Literal
from synthesizrr.base.framework.metric import Metric

InputOutputDataMixin = "InputOutputDataMixin"


class SchemaValidationError(ValueError):
    pass


class TaskRegistryMixin(MutableParameters, ABC):
    tasks: ClassVar[Tuple[TaskOrStr, ...]]  ## All supported tasks by the subclass
    task: Optional[TaskOrStr] = None  ## The specific task for an instance of the subclass

    @classmethod
    def get_subclass(
            cls,
            key: Optional[Any] = None,
            *,
            task: Optional[TaskOrStr] = None,
            name: Optional[str] = None,
            abstract: bool = False,
            **kwargs,
    ) -> Optional[Union[Type, List[Type]]]:
        if key is not None:
            if isinstance(key, tuple) and len(key) == 2:
                ## Key = (TaskOrStr, class name)
                if Task.matches_any(key[0]) and (isinstance(key[1], (str, type))):
                    key = (
                        str_normalize(Task.from_str(key[0])),
                        str_normalize(key[1].__name__ if isinstance(key[1], type) else key[1]),
                    )
                elif isinstance(key[0], str) and isinstance(key[1], (str, type)):
                    key = (
                        str_normalize(str_normalize(key[0])),
                        str_normalize(key[1].__name__ if isinstance(key[1], type) else key[1]),
                    )
            return super().get_subclass(key=key, **kwargs)
        elif task is not None:
            task: str = str_normalize(task)
            concrete_subclasses: Dict[str, Type] = {}
            for key, subclasses_dict in cls._registry.items():
                if not isinstance(key, tuple) or len(key) != 2:
                    continue
                for subclass_name, Subclass in subclasses_dict.items():
                    if task != key[0]:
                        continue
                    if is_abstract(Subclass) and abstract is False:
                        continue
                    concrete_subclasses[Subclass.__class__.__name__] = Subclass
            return only_item(list(concrete_subclasses.values()), raise_error=False)
        elif name is not None:
            name: str = str_normalize(name)
            concrete_subclasses: Dict[str, Type] = {}
            for key, subclasses_dict in cls._registry.items():
                for subclass_name, Subclass in subclasses_dict.items():
                    if name != key[1]:
                        continue
                    if is_abstract(Subclass) and abstract is False:
                        continue
                    concrete_subclasses[Subclass.__class__.__name__] = Subclass
            return only_item(list(concrete_subclasses.values()), raise_error=False)
        raise ValueError(
            f'Please pass either `key`, `task` or `name` arguments to retrieve subclasses of {cls.class_name}.'
        )


class InputOutputDataMixin(TaskRegistryMixin, ABC):
    class Config(TaskRegistryMixin.Config):
        extra = Extra.ignore
        allow_mutation = True  ## Mutable to allow caching
        ## Ref of validating set calls: https://docs.pydantic.dev/1.10/usage/model_config/
        validate_assignment = True

    schema_template: ClassVar[SchemaTemplate]

    name: Optional[str] = None
    description: Optional[str] = None

    data: Union[ScalableDataFrame, FileMetadata]
    data_split: Optional[DataSplit] = None
    data_idx: Optional[int] = None
    data_position: Optional[DataPosition] = None
    data_schema: Schema
    validated: bool = False

    map_apply: Literal['batch', 'column', 'item', 'none'] = 'batch'
    mapper: Optional[Union[Dict[Union[MLType, str], Callable], Callable]] = None
    asset_tensor_type: Optional[str] = None

    @classmethod
    @abstractmethod
    def of(
            cls,
            data: Union[InputOutputDataMixin, ScalableDataFrame, ScalableOrRaw, FileMetadata, str],
            task: Optional[TaskOrStr] = None,
            **kwargs,
    ) -> InputOutputDataMixin:
        pass

    @classmethod
    def _of(
            cls,
            data: Union[InputOutputDataMixin, ScalableDataFrame, ScalableOrRaw, FileMetadata, str],
            IOMixinBaseSubclass: Type[InputOutputDataMixin],
            task: Optional[TaskOrStr] = None,
            **kwargs,
    ) -> InputOutputDataMixin:
        if task is not None:
            IOMixinSubclass: Type[InputOutputDataMixin] = IOMixinBaseSubclass.get_subclass(task)
        else:
            IOMixinSubclass: Type[InputOutputDataMixin] = cls
        if IOMixinSubclass == IOMixinBaseSubclass:
            raise ValueError(
                f'"{IOMixinBaseSubclass.class_name}" is an abstract class. '
                f'To create an instance, please either pass `task`, '
                f'or call .of(...) on a subclass of {IOMixinBaseSubclass.class_name}'
            )
        if isinstance(data, InputOutputDataMixin):
            data: Union[ScalableDataFrame, ScalableOrRaw, FileMetadata, str] = data.data
        return IOMixinSubclass(
            task=task,
            data=data,
            **kwargs,
        )

    @root_validator(pre=True)
    def _set_data_params(cls, params: Dict) -> Dict:
        Alias.set_mapper(params)
        Alias.set_map_apply(params)
        Alias.set_map_apply(params)
        Alias.set_data_schema(params)
        Alias.set_data_split(params)

        ## Data:
        if params.get('data') is None:
            raise ValueError(
                f'Missing param: `data`. '
                f'Please pass this param to {cls.class_name}.of(...)'
            )
        data: Any = params['data']
        if isinstance(data, str):
            data: FileMetadata = FileMetadata.of(data)
        if not isinstance(data, (FileMetadata, ScalableDataFrame)):
            if isinstance(data, dict) and data.keys() == FileMetadata.param_names():
                data: FileMetadata = FileMetadata(**data)
            else:
                data: ScalableDataFrame = ScalableDataFrame.of(data)
        params['data'] = data

        ## Schema:
        if params.get('data_schema') is None:
            raise ValueError(
                f'Missing param: `data_schema`. '
                f'Please pass this param to {cls.class_name}.of(...)'
            )
        return params

    @root_validator(pre=False)  ## Runs after any @root_validator(pre=True) on Dataset and Predictions subclasses.
    def _validate_data_params(cls, params: Dict) -> Dict:
        if params.get('validated') is True and isinstance(params.get('data'), ScalableDataFrame):
            try:
                cls.validate_schema(params['data_schema'])
            except Exception as e:
                raise ValueError(
                    f"Invalid value for param `data_schema`:\n"
                    f"{params['data_schema']}:\nStacktrace:\n"
                    f"{format_exception_msg(e)}"
                )
            try:
                cls.validate_data(data=params['data'], data_schema=params['data_schema'])
            except Exception as e:
                raise ValueError(
                    f"Invalid value for param `data`:\n"
                    f"{params['data']}:\nStacktrace:\n"
                    f"{format_exception_msg(e)}"
                )
        return params

    @classmethod
    @abstractmethod
    def concat(
            cls,
            data_list: List[InputOutputDataMixin],
            **kwargs,
    ) -> InputOutputDataMixin:
        pass

    @classmethod
    def _concat(
            cls,
            io_data_list: Optional[Union[List[InputOutputDataMixin], InputOutputDataMixin]],
            *,
            IOMixinBaseSubclass: Type[InputOutputDataMixin],
            layout: Optional[DataLayout] = None,
            error_on_empty: bool = True,
    ) -> Optional[InputOutputDataMixin]:
        if isinstance(io_data_list, IOMixinBaseSubclass):
            return io_data_list
        if io_data_list is None:
            if error_on_empty:
                raise ValueError(f"Cannot concatenate None of {IOMixinBaseSubclass.class_name}.")
            return None
        io_data_list: List[InputOutputDataMixin] = as_list(io_data_list)
        if len(io_data_list) == 0:
            if error_on_empty:
                raise ValueError(f"Cannot concatenate empty list of {IOMixinBaseSubclass.class_name}.")
            return None
        subclasses: List[Type[InputOutputDataMixin]] = list(set(type(io_data) for io_data in io_data_list))
        if not equal(*subclasses):
            raise ValueError(
                f'Cannot concatenate {IOMixinBaseSubclass.class_name} instances of different subclasses. '
                f'Found following {len(subclasses)} unique subclasses: {subclasses}'
            )
        InputOutputDataMixinSubclass: Type[InputOutputDataMixin] = subclasses[0]

        schemas: List[Schema] = [io_data.data_schema for io_data in io_data_list]
        if not equal(*schemas):
            raise ValueError(
                f'Cannot concatenate {IOMixinBaseSubclass.class_name} subclasses with different schemas. '
                f'Found {len(schemas)} schemas in total:\n{schemas}'
            )
        data_schema: Schema = schemas[0]

        other_data: List[Dict] = [
            io_data.dict(exclude={'data', 'data_schema', 'data_idx', 'data_position'})
            for io_data in io_data_list
        ]
        if not equal(*other_data):
            raise ValueError(
                f'Cannot concatenate {IOMixinBaseSubclass.class_name} subclasses with different data fields. '
                f'Found {len(other_data)} data dicts in total:\n{other_data}'
            )
        other_data: Dict = other_data[0]

        data: ScalableDataFrame = ScalableDataFrame.concat(
            [io_data.data for io_data in io_data_list],
            reset_index=True,
            layout=layout,
        )
        return InputOutputDataMixinSubclass(
            data=data,
            data_schema=data_schema,
            **other_data,
        )

    def __len__(self) -> int:
        self.check_in_memory()
        return len(self.data)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.to_string()

    def to_string(self, *, data_schema: bool = True) -> str:
        if self.in_memory():
            out: str = f'{str(self.data_split).capitalize() if self.data_split is not None else ""}' \
                       f' {self.class_name}' \
                       f' (validated={self.validated})' \
                       f' with {len(self.data)} items (layout={self.data.layout})'
            if data_schema is True:
                out += f', having schema:\n{self.data_schema.json()}'

        else:
            out: str = f'{str(self.data_split).capitalize() if self.data_split is not None else ""}' \
                       f' {self.class_name}' \
                       f' on {self.data.storage} at "{self.data.path}"'
            if data_schema is True:
                out += f', having schema:\n{self.data_schema.json()}'
        return out.strip()

    @property
    def display_name(self) -> str:
        out: str = ''
        if self.name is not None:
            out += self.name
            if self.data_split is not None:
                out += StringUtil.SLASH
        if self.data_split is not None:
            out += self.data_split.capitalize()
        return out

    @classmethod
    def _registry_keys(cls) -> Optional[Union[List[Any], Any]]:
        tasks: List = as_list(cls.tasks)
        return tasks + [
            (task, str_normalize(cls.class_name))
            for task in tasks
        ]

    def index(self) -> ScalableSeries:
        self.check_in_memory()
        return self.data[self.data_schema.index_col]

    def has_index(self, raise_error: bool = True) -> bool:
        if self.in_memory(raise_error=raise_error) is False:
            return False
        if self.data_schema.index_col not in self.data:
            if raise_error:
                raise ValueError(f'Data does not contain index column "{self.data_schema.index_col}".')
            return False
        return True

    @safe_validate_arguments
    def set_layout(self, layout: DataLayout, **kwargs):
        self.data = self.data.to_layout(layout=layout, **kwargs)

    def to_layout(self, layout: DataLayout, **kwargs) -> Any:
        return self.update_params(
            data=self.data.to_layout(layout=layout, **kwargs)
        )

    def update_params(self, **new_params) -> Any:
        ## Since Parameters class is immutable, we create a new one:
        overidden_params: Dict = {
            **self.dict(exclude={'data', 'data_schema'}),
            'data': self.data,
            'data_schema': Schema(**self.data_schema.dict()),  ## Copy schema
            **new_params,
        }
        return self._constructor(**overidden_params)

    def dict(self, *args, **kwargs):
        ## Ensure we don't call ScalableDataFrame.dict()
        if kwargs.get('exclude') is None:
            kwargs['exclude'] = set()
        kwargs['exclude'] = as_set(kwargs['exclude'])
        orig_exclude: Set[str] = deepcopy(kwargs['exclude'])
        kwargs['exclude'].add('data')
        params: Dict = super(InputOutputDataMixin, self).dict(*args, **kwargs)
        if 'data' not in orig_exclude:
            ## Add `data` key back in the correct format:
            if self.in_memory():
                params['data'] = self.data  ## Do not dict-ify in-memory data (including ScalableDataFrame).
            else:
                params['data'] = self.data.dict()
        return params

    def copy(self, deep: bool = False) -> Any:
        """
        Perform a shallow copy operation.
        """
        if deep is not False:
            raise ValueError(f'We can only perform shallow copies.')
        return self.update_params()  ## With empty overrides, this just copies the Parameters object.

    def torch_dataloader(
            self,
            *,
            error: Literal['raise', 'warn', 'ignore'] = "raise",
            **kwargs
    ) -> Optional['PyTorchTaskDataDataset']:
        with optional_dependency('torch', error=error):
            from synthesizrr.base.framework.dl.torch import PyTorchTaskDataDataset
            return PyTorchTaskDataDataset(dataset=self, **kwargs).dataloader()
        return None

    @safe_validate_arguments
    def sample(self, n: conint(ge=1), **kwargs) -> InputOutputDataMixin:
        Alias.set_shuffle(kwargs, default=True)
        return next(self.iter(batch_size=n, **kwargs))

    @safe_validate_arguments
    def read(
            self,
            *,
            raw: bool = False,
            fetch_assets: bool = True,
            asset_tensor_type: Optional[str] = None,
            asset_readers: Optional[Dict[str, AssetReader]] = None,
            read_as: Optional[DataLayout] = None,
            read_params: Optional[Dict] = None,
            map: Optional[Union[Dict[Union[MLType, str], Callable], Callable]] = None,
            map_apply: Literal['batch', 'column', 'item', 'none'] = 'batch',
            map_kwargs: Optional[Dict] = None,
            validate: bool = False,
            rename_columns: Optional[Union[Dict, Callable]] = None,
            schema_columns_only: bool = False,
            persist: bool = False,
            **kwargs
    ) -> InputOutputDataMixin:
        if raw is True:
            raise ValueError(
                f'Cannot call {self.class_name}.read with `raw=True`; data must be a subclass of {self.class_name}'
            )
        map_kwargs: Dict = get_default(map_kwargs, {})
        asset_tensor_type: Optional[str] = get_default(asset_tensor_type, self.asset_tensor_type)
        data_schema: Schema = self.data_schema
        if not self.in_memory():
            file_metadata: FileMetadata = self.data
            reader: DataFrameReader = DataFrameReader.of(
                file_metadata.format,
                ## Filter based on columns:
                data_schema=data_schema.flatten() if schema_columns_only else None,
                params=read_params,
            )
            kwargs['raw'] = False
            data: ScalableDataFrame = reader.read_metadata(
                file_metadata,
                read_as=read_as,
                **kwargs
            )
            if persist:
                data: ScalableDataFrame = data.persist(wait=True)
        else:
            data: ScalableDataFrame = self.data
            if schema_columns_only:
                ## Filter based on columns:
                data: ScalableDataFrame = data[self.data_schema.columns]
            data: ScalableDataFrame = data.as_layout(layout=read_as, **kwargs)

        if rename_columns is not None:
            data: ScalableDataFrame = data.rename(columns=rename_columns)
            data_schema: Schema = data_schema.rename(columns=rename_columns)
        dataset: InputOutputDataMixin = self.update_params(
            data=data,
            data_schema=data_schema,
            validated=validate if map_apply != 'none' else False,  ## Validate post .map call
        )
        dataset.check_in_memory()
        if fetch_assets:
            dataset: InputOutputDataMixin = dataset.read_assets(
                asset_readers=asset_readers,
                tensor_type=asset_tensor_type,
            )
        if map_apply != 'none':
            dataset: InputOutputDataMixin = dataset.copy()
            dataset: InputOutputDataMixin = dataset.map(
                fn=map,
                apply=map_apply,
                validate=validate,
                **map_kwargs
            )
        return dataset

    def read_batches(self, **kwargs) -> Generator[Union[InputOutputDataMixin, Any], None, None]:
        return self.iter(**kwargs)

    @safe_validate_arguments
    def iter(
            self,
            *,
            batch_size: Optional[conint(ge=1)] = None,
            num_batches: Optional[conint(ge=1)] = None,
            steps: Optional[conint(ge=1)] = None,
            shuffle: bool = False,
            shard: Tuple[conint(ge=0), conint(ge=1)] = (0, 1),
            shard_shuffle: bool = False,
            shard_seed: Optional[int] = None,
            reverse_sharding: bool = False,
            drop_last: Optional[bool] = None,
            read_as: Optional[DataLayout] = None,
            stream_as: Optional[DataLayout] = None,
            read_params: Optional[Dict] = None,
            validate: Optional[FractionalBool] = None,
            data_split: Optional[DataSplit] = None,
            num_workers: conint(ge=1) = 1,
            parallelize: Parallelize = Parallelize.sync,
            raw: bool = False,
            fetch_assets: bool = True,
            asset_tensor_type: Optional[str] = None,
            asset_readers: Optional[Dict[str, AssetReader]] = None,
            map: Optional[Union[Dict[Union[MLType, str], Callable], Callable]] = None,
            map_apply: Literal['batch', 'column', 'item', 'none'] = 'batch',
            map_kwargs: Optional[Dict] = None,
            map_failure: Literal['raise', 'drop'] = 'raise',
            progress_bar: Optional[Union[Dict, bool]] = None,
            **kwargs
    ) -> Generator[Union[InputOutputDataMixin, Any], None, None]:
        if raw is True:
            raise ValueError(
                f'Cannot call {self.class_name}.read_batches with `raw=True`; '
                f'each batch must be a subclass of {self.class_name}'
            )
        batch_size: Optional[int] = get_default(batch_size, kwargs.pop('num_rows', None))
        if steps is not None and batch_size is None:
            raise ValueError(f'Cannot have empty `batch_size` when passing `steps`.')
        num_batches: Optional[int] = get_default(num_batches, kwargs.pop('num_chunks', None))
        kwargs['progress_bar'] = False
        dataset: InputOutputDataMixin = self.read(
            read_as=read_as,
            read_params=read_params,
            map_apply='none',  ## Do mapping during .stream(), so that it will happen in parallel.
            validate=False,  ## We will validate each batch separately.
            fetch_assets=False,  ## We will fetch assets separately for each batch.
            raw=False,
            **kwargs
        )
        progress_bar_total_num_rows: Optional[int] = None
        if dataset.in_memory() and not dataset.data.is_lazy():
            dataset_length: int = len(dataset)
            if steps is not None:
                progress_bar_total_num_rows: int = self._steps_iter_adjust_progress_bar_num_rows(
                    batch_size=batch_size,
                    steps=steps,
                    dataset_length=dataset_length,
                )
            else:
                progress_bar_total_num_rows: int = dataset_length
        map_kwargs: Dict = get_default(map_kwargs, {})
        asset_readers: Dict[str, AssetReader] = get_default(asset_readers, {})
        create_dataset_generator: Callable = lambda: dataset.data.stream(
            num_rows=batch_size,
            num_chunks=num_batches,
            stream_as=stream_as,
            map=self._process_batch,
            map_kwargs={
                **map_kwargs,
                **dict(
                    map=map,
                    map_apply=map_apply,
                    validate=validate,
                    fetch_assets=fetch_assets,
                    asset_readers=asset_readers,
                    data_split=data_split,
                    asset_tensor_type=asset_tensor_type,
                ),
            },
            num_workers=num_workers,
            map_failure=map_failure,
            parallelize=parallelize,
            shuffle=shuffle,
            shard=shard,
            shard_shuffle=shard_shuffle,
            shard_seed=shard_seed,
            reverse_sharding=reverse_sharding,
            drop_last=drop_last,
            raw=False,
            **kwargs,
        )
        if progress_bar is True:
            progress_bar: Optional[Dict] = dict()
        elif progress_bar is False:
            progress_bar: Optional[Dict] = None
        assert progress_bar is None or isinstance(progress_bar, dict)

        if isinstance(progress_bar, dict):
            progress_bar.setdefault('total', progress_bar_total_num_rows)
            progress_bar: TqdmProgressBar = create_progress_bar(**progress_bar)

        if steps is not None:
            return self._read_batches_steps(
                create_dataset_generator=create_dataset_generator,
                progress_bar=progress_bar,
                steps=steps,
            )
        else:
            return self._read_batches_epoch(
                create_dataset_generator=create_dataset_generator,
                progress_bar=progress_bar,
            )

    @classmethod
    def _steps_iter_adjust_progress_bar_num_rows(
            cls,
            batch_size: int,
            steps: int,
            dataset_length: int,
    ) -> int:
        if batch_size * steps <= dataset_length:
            progress_bar_total_num_rows: int = batch_size * steps
        else:
            ## No. steps it would take to iterate over the dataset once, i.e. number of batches in the dataset.
            steps_per_dataset: int = math.ceil(dataset_length / batch_size)
            ## How many complete passes of the dataset we should make:
            dataset_passes: int = math.floor(steps / steps_per_dataset)
            ## How many steps are left. This should be <= steps_per_dataset
            remaining_steps: int = steps - dataset_passes * steps_per_dataset
            progress_bar_total_num_rows: int = dataset_passes * dataset_length + remaining_steps * batch_size
        return progress_bar_total_num_rows

    def _read_batches_epoch(
            self,
            create_dataset_generator: Callable,
            progress_bar: Optional[TqdmProgressBar],
    ) -> Generator[Union[InputOutputDataMixin, Any], None, None]:
        dataset_generator: Generator[ScalableDataFrameOrRaw, None, None] = create_dataset_generator()
        ## Iterate through the dataset, setting batch index and batch position:
        batch: Union[InputOutputDataMixin, Any] = next(dataset_generator)
        batch.data_idx = 0
        batch.data_position = DataPosition.START
        stop: bool = False
        while not stop:
            try:
                next_batch: Union[InputOutputDataMixin, Any] = next(dataset_generator)
                next_batch.data_idx = batch.data_idx + 1
                next_batch.data_position = DataPosition.MIDDLE
            except StopIteration:  ## Once we run out of batches, mark the current batch as the last one.
                batch.data_position = DataPosition.END
                next_batch: Optional[InputOutputDataMixin] = None
                stop: bool = True
            if not isinstance(batch, InputOutputDataMixin):
                raise ValueError(
                    f'Expected batch to be an instance of {self.class_name}; '
                    f'found object of type: {type(batch)}'
                )
            yield batch
            if progress_bar is not None:
                progress_bar.update(len(batch))
            batch: InputOutputDataMixin = next_batch
        if progress_bar is not None:
            progress_bar.refresh()
            progress_bar.close()

    def _read_batches_steps(
            self,
            create_dataset_generator: Callable,
            progress_bar: Optional[TqdmProgressBar],
            steps: int,
    ) -> Generator[Union[InputOutputDataMixin, Any], None, None]:
        dataset_generator: Generator[ScalableDataFrameOrRaw, None, None] = create_dataset_generator()
        for step_i in range(steps):
            try:
                batch: Union[InputOutputDataMixin, Any] = next(dataset_generator)
            except StopIteration:
                ## We have reached the end of the dataset, but we still have more steps. Restart the generator:
                dataset_generator: Generator[ScalableDataFrameOrRaw, None, None] = create_dataset_generator()
                batch: Union[InputOutputDataMixin, Any] = next(dataset_generator)
            if not isinstance(batch, InputOutputDataMixin):
                raise ValueError(
                    f'Expected batch to be an instance of {self.class_name}; '
                    f'found object of type: {type(batch)}'
                )
            batch.data_idx = step_i
            if step_i == 0:
                batch.data_position = DataPosition.START
            elif step_i < steps - 1:
                batch.data_position = DataPosition.MIDDLE
            else:
                batch.data_position = DataPosition.END
            yield batch
            if progress_bar is not None:
                progress_bar.update(len(batch))
        if progress_bar is not None:
            progress_bar.refresh()
            progress_bar.close()

    def _process_batch(
            self,
            sdf_batch: ScalableDataFrame,
            validate: Optional[FractionalBool],
            fetch_assets: bool,
            asset_readers: Optional[Dict[str, AssetReader]],
            data_split: Optional[DataSplit],
            map: Optional[Union[Dict[Union[MLType, str], Callable], Callable]],
            map_apply: Literal['batch', 'column', 'item', 'none'],
            asset_tensor_type: Optional[str] = None,
            **map_kwargs
    ) -> InputOutputDataMixin:
        ## Convert from SDF batch to InputOutputDataMixin batch:
        should_validate: bool = resolve_fractional_bool(validate)  ## Do NOT pass "seed" here
        data_split: Optional[DataSplit] = get_default(data_split, self.data_split)
        asset_tensor_type: Optional[str] = get_default(asset_tensor_type, self.asset_tensor_type)
        data_batch: InputOutputDataMixin = self._constructor(
            data=sdf_batch,
            data_schema=self.data_schema.copy(deep=True),
            validated=should_validate if map_apply != 'none' else False,  ## Validate post .map call
            data_split=data_split,
            **self.dict(exclude={'data', 'data_schema', 'validated', 'data_split'})
        )
        if fetch_assets:
            data_batch = data_batch.read_assets(
                asset_readers=asset_readers,
                tensor_type=asset_tensor_type,
            )
        if map_apply != 'none':
            data_batch: InputOutputDataMixin = data_batch.copy()
            data_batch: InputOutputDataMixin = data_batch.map(
                fn=map,
                apply=map_apply,
                validate=should_validate,
                **map_kwargs
            )
        if not isinstance(data_batch, InputOutputDataMixin):
            raise ValueError(
                f'Expected batch to be an instance of {InputOutputDataMixin}, '
                f'found object of type: {type(data_batch)}'
            )
        return data_batch

    def iter_files(self, **kwargs) -> Generator[InputOutputDataMixin, None, None]:
        if self.in_memory():
            raise ValueError(f'Cannot iterate over files for data which is in memory')
        assert isinstance(self.data, FileMetadata)

        if not self.data.is_path_valid_dir():
            yield self  ## Return the current path
        else:
            files_metadata: List[FileMetadata] = self.data.list_metadata(**kwargs)
            progress_bar: Optional[Dict] = Alias.get_progress_bar(kwargs)
            pbar: ProgressBar = ProgressBar.of(
                progress_bar,
                total=len(files_metadata),
                unit='file',
            )
            for fmeta in files_metadata:
                pbar.update(1)
                yield self.update_params(data=fmeta)

    @safe_validate_arguments
    def read_assets(
            self,
            asset_readers: Optional[Dict[str, AssetReader]] = None,
            return_cols: bool = False,
            raise_error: bool = True,
            tensor_type: Optional[str] = None,
            **kwargs,
    ) -> Union[InputOutputDataMixin, Tuple[InputOutputDataMixin, Set[str]]]:
        self.in_memory(raise_error=raise_error)
        asset_readers: Dict[str, Optional[AssetReader]] = get_default(asset_readers, {})
        assets: Dict[str, ScalableSeries] = {}
        tensor_layout: Optional[DataLayout] = SHORTHAND_TO_TENSOR_LAYOUT_MAP.get(str_normalize(tensor_type), None)
        ## Create futures to read all assets:
        for asset_mltype in ASSET_ML_TYPES:
            for col, col_mltype in self.columns(mltypes=asset_mltype, return_mltypes=True):
                asset_paths: ScalableSeries = self.data[col]
                non_existent_asset_paths: List[str] = []
                undetectable_asset_paths: List[str] = []
                if col not in asset_readers:
                    ## Check if the data really is an asset path:
                    asset_paths_exists: Set[bool] = set()
                    asset_formats: Set[FileFormat] = set()
                    for asset_path in asset_paths:
                        if asset_path is not None:
                            asset_path_exists: bool = FileMetadata.path_exists(asset_path)
                            asset_format: Optional[FileFormat] = FileMetadata.detect_file_format(asset_path)
                            if asset_path_exists and asset_format is not None:
                                asset_formats.add(asset_format)
                            if not asset_path_exists:
                                non_existent_asset_paths.append(asset_path)
                            if asset_format is None:
                                undetectable_asset_paths.append(asset_path)
                    if len(non_existent_asset_paths) > 0:
                        raise ValueError(
                            f'Following {len(non_existent_asset_paths)} items in column "{col}" are either invalid '
                            f'paths for {col_mltype} assets, or do not exist (pwd: "{os.getcwd()}")'
                            f'\n{non_existent_asset_paths}.'
                        )
                    if len(undetectable_asset_paths) > 0:
                        raise ValueError(
                            f'Cannot detect file format for following {len(undetectable_asset_paths)} items '
                            f'in column "{col}": {undetectable_asset_paths}'
                        )
                    if len(asset_formats) > 1:
                        raise ValueError(
                            f'Detected multiple file formats for assets in column "{col}": '
                            f'detected {asset_formats} from following asset paths: {asset_paths.to_list()}'
                        )
                    if len(asset_formats) == 0:
                        raise ValueError(f'Cannot detect any file formats from asset paths: {asset_paths.to_list()}')
                    assert len(asset_formats) == 1 and len(non_existent_asset_paths) == 0
                    ## At least one item in this batch column was non-None:
                    asset_format: FileFormat = next(iter(asset_formats))
                    asset_readers[col] = AssetReader.of(
                        file_format=asset_format,
                        **kwargs
                    )
                if col in asset_readers:
                    ## Only read if we have a valid reader:
                    asset_reader: Optional[AssetReader] = asset_readers[col]
                    if asset_reader is not None:
                        assets[col] = asset_paths.apply(
                            lambda asset_path: run_concurrent(asset_reader.read, source=asset_path)
                        )
        ## Collect futures:
        for col in assets.keys():
            self.data[col] = assets[col].apply(self._accumulate_and_convert_asset, tensor_layout=tensor_layout)
        if return_cols:
            return self, set(assets.keys())
        return self

    @safe_validate_arguments
    def assets_to_tensor(
            self,
            tensor_type: TensorShortHand,
            stack: bool = False,
            raise_error: bool = True,
            **kwargs,
    ) -> InputOutputDataMixin:
        self.in_memory(raise_error=raise_error)
        tensor_layout: Optional[DataLayout] = SHORTHAND_TO_TENSOR_LAYOUT_MAP.get(str_normalize(tensor_type), None)
        if tensor_layout is None:
            raise ValueError(f'Must pass a valid value of `tensor_type`; found `tensor_type`={tensor_type}')
        for asset_mltype in ASSET_ML_TYPES:
            for col, col_mltype in self.columns(mltypes=asset_mltype, return_mltypes=True):
                data_col = self.data[col]
                TensorScalableSeriesSubclass: Type = TensorScalableSeries.get_subclass(tensor_layout)
                if not isinstance(data_col, TensorScalableSeriesSubclass):
                    data_col = data_col.apply(self._accumulate_and_convert_asset, tensor_layout=tensor_layout)
                    if stack:
                        data_col = data_col.as_layout(layout=tensor_layout, **kwargs)
                        if not isinstance(data_col, TensorScalableSeriesSubclass):
                            f'Expected {TensorScalableSeriesSubclass}; found: {type(data_col)}'
                self.data[col] = data_col
        return self

    @classmethod
    def _accumulate_and_convert_asset(
            cls,
            item: Any,
            tensor_layout: Optional[DataLayout] = None,
    ) -> Union[Asset, Any]:
        item: Asset = accumulate(item)
        if isinstance(item, Asset):
            if tensor_layout is not None:
                item: Any = item.as_tensor(tensor_layout)
        elif tensor_layout is not None:
            tensor_dtype: Type = AVAILABLE_TENSOR_TYPES[tensor_layout]
            if not isinstance(item, tensor_dtype):
                raise ValueError(
                    f'Value must be either a subclass of {Asset.class_name} or a tensor of type {tensor_dtype}; '
                    f'found item of type {type(item)} with value:\n{item}'
                )
        else:
            tensor_dtypes: Tuple[Type, ...] = tuple(AVAILABLE_TENSOR_TYPES.values())
            if not isinstance(item, tensor_dtypes):
                raise ValueError(
                    f'Value must be either a subclass of {Asset.class_name} or a tensor of one of the '
                    f'following types: {tensor_dtypes}; found item of type {type(item)} with value:\n{item}'
                )
        return item

    @safe_validate_arguments
    def map(
            self,
            fn: Optional[Union[Dict[Union[MLType, str], Callable], Callable]] = None,
            apply: Optional[Literal['batch', 'column', 'item', 'row']] = None,
            validate: bool = False,
            **kwargs
    ) -> InputOutputDataMixin:
        if fn is None and self.mapper is None:
            return self
        mapped_data_batch: InputOutputDataMixin = self._map_data(self, fn=self.mapper, apply=self.map_apply)
        mapped_data_batch: InputOutputDataMixin = self._map_data(mapped_data_batch, fn=fn, apply=apply, **kwargs)
        return mapped_data_batch.update_params(mapper=None, validated=validate)

    @classmethod
    def _map_data(
            cls,
            data_batch: InputOutputDataMixin,
            fn: Optional[Union[Dict[Union[MLType, str], Callable], Callable]],
            apply: Optional[Literal['batch', 'column', 'item', 'row']],
            **kwargs,
    ) -> InputOutputDataMixin:
        data_batch.check_in_memory()
        if fn is None:
            return data_batch
        if apply is None:
            raise ValueError(f'Must pass value for `apply` when calling .map()')
        if apply == 'batch':
            out: InputOutputDataMixin = fn(data_batch, **kwargs)
        elif apply in {'column', 'item'}:
            mappers: Dict[str, Callable] = {}
            if isinstance(fn, dict):
                for transform_key, mapper in fn.items():
                    if MLType.matches_any(transform_key):
                        cols: List[str] = data_batch.columns(mltypes=transform_key)
                    else:
                        cols: List[str] = as_list(transform_key)
                    for col in cols:
                        if col in mappers:
                            raise ValueError(
                                f'Column "{col}" has already been transformed; please ensure the following transforms '
                                f'are non-overlapping: {fn}'
                            )
                        mappers[col] = mapper
            else:
                for col in data_batch.data_schema.columns_set:
                    mappers[col]: Callable = fn
            data: ScalableDataFrame = data_batch.data
            for col, mapper in mappers.items():
                if apply == 'column':
                    data[col] = mapper(data[col], **kwargs)
                elif apply == 'item':
                    data[col] = data[col].apply(mapper, **kwargs)
            out: InputOutputDataMixin = data_batch.of(
                data=data,
                validated=False,
                **data_batch.dict(exclude={'data', 'validated'})  ## Do not retain "mapper" after mapping.
            )
        elif apply == 'row':
            data_schema: Schema = data_batch.data_schema.copy()  ## Might be updated by `fn`
            mapped_data: ScalableDataFrame = ScalableDataFrame.of(data_batch.data.apply(fn, args=(data_schema,)))
            out: InputOutputDataMixin = data_batch.of(
                data=mapped_data,
                data_schema=data_schema,
                validated=False,
                **data_batch.dict(exclude={'data', 'data_schema', 'validated'})
            )
        else:
            raise NotImplementedError(f'Unsupported value for argument `apply`: "{apply}".')
        if not isinstance(out, data_batch.__class__):
            raise ValueError(
                f'Expected mapper function to produce output of type {data_batch.class_name}; '
                f'found output of type {type(out)}'
            )
        return out

    def __iter__(self) -> Generator[InputOutputDataMixin, None, None]:
        for row in self.iter(num_rows=1, stream_as=DataLayout.LIST_OF_DICT):
            assert isinstance(row, InputOutputDataMixin)
            row.set_layout(DataLayout.RECORD)
            yield row

    @safe_validate_arguments
    def evaluate(
            self,
            metric: Optional[Union[Metric, Dict, str]] = None,
            *,
            rolling: bool = False,
            **kwargs
    ) -> Metric:
        if metric is None:
            return Metric.of(**kwargs).evaluate(self)
        if isinstance(metric, str):
            return Metric.of(name=metric, **kwargs).evaluate(self)
        if isinstance(metric, Metric):
            if rolling:
                return metric.evaluate(self, rolling=True)
            else:
                return metric.evaluate(self, inplace=False)
        raise NotImplementedError(f'Unsupported value for input `metric`: {type(metric)} with value:\n{metric}')

    @safe_validate_arguments
    def columns(
            self,
            mltypes: Union[Set[MLType], Tuple[MLType], List[MLType], MLType] = None,
            return_mltypes: bool = False,
            schema_portion: Literal['features', 'ground_truths', 'predictions', 'complete'] = 'complete',
            **kwargs
    ) -> Union[List[Tuple[str, MLType]], List[str]]:
        data_schema: Optional[MLTypeSchema] = None
        if schema_portion == 'complete':
            data_schema: MLTypeSchema = self.data_schema.flatten()
        elif schema_portion == 'features':
            data_schema: MLTypeSchema = self.data_schema.features_schema
        elif schema_portion == 'ground_truths':
            data_schema: MLTypeSchema = self.data_schema.ground_truths_schema
        elif schema_portion == 'predictions':
            data_schema: MLTypeSchema = self.data_schema.predictions_schema
        if mltypes is not None:
            data_schema: MLTypeSchema = Schema.filter_schema(
                data_schema=data_schema,
                mltypes=mltypes,
                **kwargs
            )
            if return_mltypes:
                return sorted(
                    [(col, mltype) for col, mltype in data_schema.items()],
                    key=lambda x: x[0]
                )
            else:
                return sorted(list(data_schema.keys()))
        else:
            if return_mltypes:
                return sorted(
                    [(col, mltype) for col, mltype in data_schema.items()],
                    key=lambda x: x[0]
                )
            else:
                return sorted(self.data_schema.columns)

    def filter(
            self,
            fn: Union[Callable[[Dict, Schema], bool], Callable[[Dict], bool]],
            *,
            yield_partial: bool = False,
            **kwargs
    ) -> Union[Generator[InputOutputDataMixin, None, None], InputOutputDataMixin]:
        Alias.set_num_rows(kwargs, default=int(1e3))
        if yield_partial:
            return self.__filter_yield(fn, **kwargs)
        filtered_batches: List[InputOutputDataMixin] = []
        for batch in self.iter(**kwargs):
            assert isinstance(batch, InputOutputDataMixin)
            if len(get_fn_args(fn)) == 1:
                args = ()
            elif len(get_fn_args(fn)) == 2:
                args = (self.data_schema.copy(),)
            else:
                raise NotImplementedError(
                    f'Passed function must have only one or two args; found: {len(get_fn_args(fn))}'
                )
            rows_to_keep: ScalableSeries = batch.data.apply(fn, args=args)
            if (rows_to_keep == False).all():
                continue
            filtered_batch: InputOutputDataMixin = batch.update_params(data=batch.data.loc[rows_to_keep])
            filtered_batches.append(filtered_batch)
        return self.concat(filtered_batches)

    def __filter_yield(
            self,
            fn: Union[Callable[[Dict, Schema], bool], Callable[[Dict], bool]],
            **kwargs
    ) -> Generator[InputOutputDataMixin, None, None]:
        for batch in self.iter(**kwargs):
            assert isinstance(batch, InputOutputDataMixin)
            if len(get_fn_args(fn)) == 1:
                args = ()
            elif len(get_fn_args(fn)) == 2:
                args = (self.data_schema.copy(),)
            else:
                raise NotImplementedError(
                    f'Passed function must have only one or two args; found: {len(get_fn_args(fn))}'
                )
            rows_to_keep: ScalableSeries = batch.data.apply(fn, args=args)
            if (rows_to_keep == False).all():
                continue
            filtered_batch: InputOutputDataMixin = batch.update_params(data=batch.data.loc[rows_to_keep])
            yield filtered_batch

    # def keep_columns(self, **kwargs) -> Union[ScalableDataFrame, ScalableSeries]:
    #     filtered_data_schema: MLTypeSchema = # TODO similar to .features()
    #     return self.update_params(
    #         data=Schema.filter_df(self.data, data_schema=filtered_data_schema, **kwargs),
    #         data_schema=filtered_data_schema,
    #     )
    #
    # def drop_columns(self, **kwargs) -> Union[ScalableDataFrame, ScalableSeries]:
    #     cols_to_remove: List[str] = # TODO similar to .features()
    #     filtered_data_schema: MLTypeSchema = {
    #         col: mltype
    #         for col, mltype in self.data_schema.flatten().items()
    #         if col not in cols_to_remove
    #     }
    #     return self.update_params(
    #         data=Schema.filter_df(self.data, data_schema=filtered_data_schema, **kwargs),
    #         data_schema=filtered_data_schema,
    #     )

    def __getattr__(self, attr_name: str):
        if attr_name in self.data.columns_set:
            return self.data[attr_name]
        raise AttributeError(
            f'`{attr_name}` is neither an attribute of {self.class_name} nor a data colum; '
            f'current data columns are: {self.data.columns}'
        )

    def __getitem__(self, attr_name: str):
        if attr_name in self.data.columns_set:
            return self.data[attr_name]
        raise KeyError(
            f'`{attr_name}` is not a data colum; '
            f'current data columns are: {self.data.columns}'
        )

    @safe_validate_arguments
    def features(
            self,
            *columns: Union[List[MLTypeOrStr], Set[MLTypeOrStr], Tuple[MLTypeOrStr, ...], MLTypeOrStr],
            raise_error: bool = True,
            return_series: bool = False,
            **kwargs
    ) -> Optional[Union[ScalableDataFrame, ScalableSeries]]:
        self.check_in_memory()
        data_schema: MLTypeSchema = self.data_schema.features_schema
        if len(columns) == 0:
            return self.data[sorted(data_schema.keys())]
        if len(columns) == 1 and is_list_or_set_like(columns[0]):
            columns = columns[0]
        columns: List = as_list(columns)
        filtered_cols: Set[str] = set()
        filtered_mltypes: Set[MLType] = set()
        for col in columns:
            col: str = str(col)
            if col in data_schema:
                filtered_cols.add(col)
            elif MLType.matches_any(col):
                filtered_mltypes.add(MLType.from_str(col))
        if len(filtered_cols) == len(filtered_mltypes) == 0:
            if raise_error:
                raise ValueError(
                    f'No columns returned when using filtering criterion {columns} '
                    f'on dataframe with schema: {data_schema}'
                )
            return None
        features_schema: MLTypeSchema = {
            col: mltype
            for col, mltype in data_schema.items()
            if mltype in filtered_mltypes or col in filtered_cols
        }
        return Schema.filter_df(
            self.data,
            data_schema=features_schema,
            return_series=return_series,
            sort_columns=True,
            **kwargs
        )

    def has_features(self, raise_error: bool = True) -> bool:
        if self.in_memory(raise_error=raise_error) is False:
            return False
        features_schema_cols: Set[str] = set(self.data_schema.features_schema.keys())
        if len(features_schema_cols) == 0:
            if raise_error:
                raise ValueError(
                    f'Data does not contain any feature columns as required by schema:\n'
                    f'Schema: {self.data_schema}'
                )
            return False
        if not features_schema_cols <= self.data.columns_set:
            if raise_error:
                raise ValueError(
                    f'Data does not contain all feature columns as per schema;\n'
                    f'Present feature columns: {self.data.columns_set}\n'
                    f'Missing feature columns: {features_schema_cols - self.data.columns_set}'
                )
            return False
        return True

    def ground_truths(
            self,
            *,
            allow_missing: bool = False,
            return_series: bool = True,
            return_index: bool = False,
            **kwargs
    ) -> Union[ScalableDataFrame, ScalableSeries]:
        self.check_in_memory()
        cols: List = list(self.data_schema.ground_truths_schema.keys())
        if return_index:
            cols.append(self.data_schema.index_col)
        return Schema.filter_df(
            self.data,
            cols,
            allow_missing=allow_missing,
            return_series=return_series,
            **kwargs
        )

    def has_ground_truths(self, raise_error: bool = True) -> bool:
        if self.in_memory(raise_error=raise_error) is False:
            return False
        ground_truths_schema_cols: Set[str] = set(self.data_schema.ground_truths_schema.keys())
        if len(ground_truths_schema_cols) == 0:
            if raise_error:
                raise ValueError(
                    f'Data does not contain any ground-truth columns as required by schema:\n'
                    f'Schema: {self.data_schema}'
                )
            return False
        if not ground_truths_schema_cols <= self.data.columns_set:
            if raise_error:
                raise ValueError(
                    f'Data does not contain all ground-truth columns as per schema;\n'
                    f'Present ground-truth columns: {self.data.columns_set}\n'
                    f'Missing ground-truth columns: {ground_truths_schema_cols - self.data.columns_set}'
                )
            return False
        return True

    def predictions(
            self,
            *,
            allow_missing: bool = False,
            return_series: bool = False,
            return_index: bool = True,
    ) -> Union[ScalableDataFrame, ScalableSeries]:
        self.check_in_memory()
        cols: List = list(self.data_schema.predictions_schema.keys())
        if return_index:
            cols.append(self.data_schema.index_col)
        return Schema.filter_df(
            self.data,
            cols,
            allow_missing=allow_missing,
            return_series=return_series,
        )

    def has_predictions(self, raise_error: bool = True) -> bool:
        if self.in_memory(raise_error=raise_error) is False:
            return False
        predictions_schema_cols: Set[str] = set(self.data_schema.predictions_schema.keys())
        if len(predictions_schema_cols) == 0:
            if raise_error:
                raise ValueError(
                    f'Data does not contain any predictions columns as required by schema:\n'
                    f'Schema: {self.data_schema}'
                )
            return False
        if not predictions_schema_cols <= self.data.columns_set:
            if raise_error:
                raise ValueError(
                    f'Data does not contain all predictions columns as per schema;\n'
                    f'Present predictions columns: {self.data.columns_set}\n'
                    f'Missing predictions columns: {predictions_schema_cols - self.data.columns_set}'
                )
            return False
        return True

    def in_memory(self, raise_error: bool = False) -> bool:
        error_msg: str = f'Cannot use data which is not in memory, ' \
                         f'please read it into memory using .read(...) or .iter(...)'
        if isinstance(self.data, FileMetadata):
            if raise_error:
                raise ValueError(error_msg)
            return False
        if not isinstance(self.data, ScalableDataFrame):
            if raise_error:
                raise ValueError(error_msg)
            return False
        return True

    def check_in_memory(self) -> NoReturn:
        self.in_memory(raise_error=True)

    @classmethod
    @abstractmethod
    def validate_schema(cls, data_schema: Schema):
        pass

    @classmethod
    @safe_validate_arguments
    def validate_data(cls, data: ScalableDataFrame, data_schema: Schema):
        d_c: Set = data.columns_set
        s_c: Set = data_schema.columns_set
        missing_cols: Set = s_c - d_c
        if len(missing_cols) != 0:
            raise SchemaValidationError(
                f'`data_schema` defines following columns which are missing from data: {missing_cols}\n'
                f'Data columns: {d_c}\n'
                f'Schema columns: {s_c}\n'
                f'Full schema:\n{data_schema}'
            )
        for col, mltype in data_schema.predictions_schema.items():
            if not data[col].valid(
                    validator=cls.validate_prediction,
                    sample_size=True,  ## Validate all rows.
                    col=col,
                    mltype=mltype,
            ):
                raise ValueError(f'Prediction column "{col}" with type {mltype} is not valid.')

        for col, mltype in data_schema.ground_truths_schema.items():
            if not data[col].valid(
                    validator=cls.validate_ground_truth,
                    sample_size=True,  ## Validate all rows.
                    col=col,
                    mltype=mltype,
            ):
                raise ValueError(f'Ground-truth column "{col}" with type {mltype} is not valid.')

    @classmethod
    def validate_ground_truth(cls, value: Any, col: str, mltype: MLType) -> bool:
        if is_null(value):
            return False
        return True

    @classmethod
    def validate_prediction(cls, value: Any, col: str, mltype: MLType) -> bool:
        if is_null(value):
            return False
        return True

    @classmethod
    def _create_index(
            cls,
            index: ScalableSeriesOrRaw,
            layout: Optional[DataLayout] = None,
            **kwargs,
    ) -> ScalableSeries:
        return ScalableSeries.of(index, layout=layout, **kwargs)

    @classmethod
    def _create_features(
            cls,
            features: ScalableOrRaw,
            index: ScalableSeries,
            index_col: str,
            **kwargs,
    ) -> ScalableDataFrame:
        features: ScalableDataFrame = ScalableDataFrame.of(features, **kwargs)
        features.loc[index_col] = index
        return features

    @classmethod
    def _create_ground_truths(
            cls,
            ground_truths: ScalableOrRaw,
            index: ScalableSeries,
            index_col: str,
            **kwargs,
    ) -> ScalableDataFrame:
        ground_truths: ScalableDataFrame = ScalableDataFrame.of(ground_truths, **kwargs)
        ground_truths.loc[index_col] = index
        return ground_truths

    @safe_validate_arguments
    def display(
            self,
            *,
            page_size: int = 5,
            readonly: bool = True,
            col_partial_order: Optional[List] = None,
            col_width_scale: confloat(gt=0.0, le=5.0) = 1.5,
            min_col_width: conint(ge=1) = 100,
            max_col_width: conint(ge=1) = 800,
            **kwargs,
    ):
        Alias.set_shuffle(kwargs)
        shuffle: bool = kwargs.pop('shuffle', False)
        Alias.set_seed(kwargs)
        seed: Optional[int] = kwargs.pop('seed', None)
        self.check_in_memory()
        df: pd.DataFrame = self.data.pandas()
        if col_partial_order is not None:
            df: pd.DataFrame = pd_partial_column_order(df, columns=col_partial_order)
        if shuffle:
            df: pd.DataFrame = df.sample(frac=1, replace=False, random_state=seed)
        with optional_dependency('panel'):
            import panel as pn
            pn.extension('tabulator')

            col_formatters: Dict[str, Dict] = dict()
            col_widths: Dict[str, int] = dict()
            col_header_filters: Dict[str, Any] = dict()
            for col, mltype in self.data_schema.flatten().items():
                if mltype in {MLType.INT, MLType.FLOAT, MLType.BOOL} or is_numeric_dtype(df[col]):
                    col_widths[col] = min_col_width
                    col_header_filters[col] = True
                elif mltype in {MLType.TEXT, MLType.INDEX, MLType.CATEGORICAL}:
                    col_widths[col]: int = self._display_get_text_col_width(
                        df,
                        col=col,
                        col_width_scale=col_width_scale,
                        min_col_width=min_col_width,
                        max_col_width=max_col_width,
                    )
                    col_formatters[col] = {'type': 'textarea'}
                    col_header_filters[col] = True
                elif mltype in {MLType.OBJECT}:
                    col_widths[col] = int(col_width_scale * min_col_width)
                    col_header_filters[col] = True
                else:
                    col_widths[col] = min_col_width
                    col_header_filters[col] = True
            for col in df.columns:
                if col not in col_widths:
                    if is_numeric_dtype(df[col]):
                        col_widths[col] = min_col_width
                    elif is_string_dtype(df[col]):
                        col_formatters[col] = {'type': 'textarea'}
                        col_widths[col]: int = self._display_get_text_col_width(
                            df,
                            col=col,
                            col_width_scale=col_width_scale,
                            min_col_width=min_col_width,
                            max_col_width=max_col_width,
                        )
                    else:
                        col_widths[col] = min_col_width
                if col not in col_header_filters:
                    col_header_filters[col] = True
                col_widths[col]: int = min(max(col_widths[col], min_col_width), max_col_width)
            df_widget = pn.widgets.Tabulator(
                df,
                name=str(self),
                layout='fit_data',
                formatters=col_formatters,
                widths=col_widths,
                pagination='remote',
                page_size=page_size,
                disabled=readonly,
                ## https://discourse.holoviz.org/t/tabulator-moveable-rows-and-columns/4007/6
                configuration={
                    'movableColumns': True,
                },
                header_filters=col_header_filters,
            )
            return df_widget
        from IPython.display import display
        display(df)

    @staticmethod
    def _display_get_text_col_width(
            df: pd.DataFrame,
            *,
            col: str,
            col_width_scale: float,
            min_col_width: int,
            max_col_width: int,
    ) -> int:
        text_lens: pd.Series = df[col].apply(lambda x: len(str(x)) if not is_null(x) else 0)
        col_width: int = int((text_lens.mean() + col_width_scale * text_lens.std()) / 2)
        col_width: int = min(max(col_width, min_col_width), max_col_width)
        return col_width
