import copy
from typing import *
import re
import numpy as np
## These must be imported separately from the file since we are in the same dir.
from synthesizrr.base.util.language import AutoEnum, as_list, auto, as_set, is_list_like, Parameters, get_default, \
    str_format_args, \
    safe_validate_arguments, format_exception_msg, keep_keys, is_empty_list_like, assert_not_empty_dict, remove_keys
from synthesizrr.base.constants.MLConstants import MLType, MLTypeSchema, PREDICTED_ML_TYPES, GROUND_TRUTH_ML_TYPES, \
    DATA_ML_TYPES
from synthesizrr.base.constants.FileConstants import FileContents
from pydantic import conint, constr, validator, root_validator, Field

ColTemplate = "ColTemplate"
SchemaTemplate = "Schema"
Schema = "Schema"

INDEX_COL_TEMPLATE_KEY: str = 'index_col'
INDEX_COL_NAME_TEMPLATE: str = '{' + INDEX_COL_TEMPLATE_KEY + '}'
INDEX_COL_DEFAULT_NAME: str = 'id'


class ColTemplate(Parameters):
    template: constr(min_length=1)
    args: Tuple[constr(min_length=1), ...]
    regex: re.Pattern

    def __hash__(self):
        return hash(str(self.template))

    def __str__(self):
        return str(self.template)

    @classmethod
    def of(cls, template: str, regex_fill: str = '.+?', regex_flags: int = re.IGNORECASE) -> ColTemplate:
        return ColTemplate(
            template=template,
            args=tuple(str_format_args(template)),
            regex=re.compile(cls.as_regex(template, fill=regex_fill), flags=regex_flags),
        )

    @classmethod
    def template_is_unfilled(cls, template: str) -> bool:
        return template.find('{') != -1 and template.find('}') != -1 and template.find('{') < template.find('}')

    @classmethod
    def as_regex(cls, template: str, fill: str = '.+?') -> str:
        return template.format(**{arg: fill for arg in str_format_args(template)})

    def populate(
            self,
            *,
            allow_unfilled: bool = False,
            **kwargs,
    ) -> Optional[Union[List[str], str]]:
        kwargs: Dict[str, Any] = keep_keys(kwargs, self.args)
        iterable_args: Set = set()
        non_iterable_args: Set = set()
        for arg, val in kwargs.items():
            if isinstance(val, (range, list, set, tuple, np.ndarray)):
                iterable_args.add(arg)
            else:
                non_iterable_args.add(arg)
        if len(iterable_args) == 0:
            col: str = self.template.format(**kwargs)
            if self.template_is_unfilled(col):
                if not allow_unfilled:
                    raise ValueError(
                        f'Column is templatized even after populating arguments. '
                        f'Column template: "{self.template}"; '
                        f'column after populating: "{col}"; '
                        f'detected args: {str_format_args(col)}; '
                        f'kwargs: {kwargs}'
                    )
                return None
            return col
        else:
            if len(non_iterable_args) > 0:
                partial_template: str = self.template.format(**{
                    arg: val for arg, val in kwargs
                    if arg in non_iterable_args
                })
            else:
                partial_template: str = self.template
            cols: List[str] = [partial_template]
            for arg in iterable_args:
                vals: Tuple = tuple(kwargs[arg])
                cols_temp = []
                for col in cols:
                    for val in vals:
                        cols_temp.append(col.format(**{arg: val}))
                cols: List[str] = cols_temp
            filtered_cols: List[str] = []
            for col in cols:
                if self.template_is_unfilled(col):
                    if not allow_unfilled:
                        raise ValueError(
                            f'Column is templatized even after populating arguments. '
                            f'Column template: "{self.template}"; '
                            f'column after populating: "{col}"; '
                            f'detected args: {str_format_args(col)}; '
                            f'kwargs: {kwargs}'
                        )
                else:
                    filtered_cols.append(col)
                cols = filtered_cols
            return cols

    def matches(self, cols: Union[List, Tuple, Set, Any]) -> Set[str]:
        cols: Set[str] = as_set(cols)
        return set(col for col in cols if self.regex.match(str(col)) is not None)


class SchemaTemplate(Parameters):
    index_col_template: ColTemplate
    predictions_schema_template: Dict[ColTemplate, MLType]
    ground_truths_schema_template: Dict[ColTemplate, MLType]
    features_schema_template: Dict[ColTemplate, MLType]

    @property
    def has_features(self) -> bool:
        return self.features_schema_template != {}

    @property
    def has_ground_truths(self) -> bool:
        return self.ground_truths_schema_template != {}

    @property
    def has_predictions(self) -> bool:
        return self.predictions_schema_template != {}

    @classmethod
    def from_parts(
            cls,
            index_col_template: Optional[str] = None,
            ground_truths_schema_template: Optional[MLTypeSchema] = None,
            predictions_schema_template: Optional[MLTypeSchema] = None,
            features_schema_template: Optional[MLTypeSchema] = None,
    ) -> Optional[SchemaTemplate]:
        def _to_schema_template(schema: MLTypeSchema) -> Dict[ColTemplate, MLType]:
            schema_template_part: Dict[ColTemplate, MLType] = {}
            for col, mltype in schema.items():
                if mltype in set.union(GROUND_TRUTH_ML_TYPES, PREDICTED_ML_TYPES).union({MLType.INDEX}):
                    raise ValueError(f'Schema template should have MLTypes like {DATA_ML_TYPES}, not {mltype}')
                schema_template_part[ColTemplate.of(col)] = mltype
            return schema_template_part

        ## Set index column:
        if index_col_template is None:
            index_col_template: str = INDEX_COL_NAME_TEMPLATE
        index_col_template: ColTemplate = ColTemplate.of(index_col_template)
        ## Set ground truths:
        ground_truths_schema_template: Dict[ColTemplate, MLType] = _to_schema_template(
            MLType.convert_values(get_default(ground_truths_schema_template, {}))
        )
        ## Set predictions:
        predictions_schema_template: Dict[ColTemplate, MLType] = _to_schema_template(
            MLType.convert_values(get_default(predictions_schema_template, {}))
        )
        ## Set features:
        features_schema_template: Dict[ColTemplate, MLType] = _to_schema_template(
            MLType.convert_values(get_default(features_schema_template, {}))
        )
        return cls(
            index_col_template=index_col_template,
            predictions_schema_template=predictions_schema_template,
            ground_truths_schema_template=ground_truths_schema_template,
            features_schema_template=features_schema_template,
        )

    @safe_validate_arguments
    def infer_from_mltype_schema(
            self,
            schema: Union[Dict, Any],
            *,
            index_col: Optional[constr(min_length=1)] = None,
            infer_features: bool = True,
            infer_ground_truths: bool = True,
            infer_predictions: bool = True,
            has_features: bool = False,
            has_ground_truths: bool = False,
            has_predictions: bool = False,
    ) -> Schema:
        if isinstance(schema, Schema):
            raise ValueError(f'Please call {Schema.class_name}.of(...)')
        if not isinstance(schema, dict):
            raise ValueError(
                f'Expected schema to be a dict of MLTypes; '
                f'found schema of type {type(schema)} with value: {schema}'
            )

        ## Might have MLType.GROUND_TRUTH, MLType.PREDICTED:
        mltype_schema: MLTypeSchema = MLType.convert_values(schema, raise_error=True)

        ## Set index column:
        if index_col is None:
            ## index_col must either be passed explicitly, or be present in the schema.
            index_col: Optional[str] = Schema.filter_index(schema, allow_missing=True)
            if index_col is None:
                raise ValueError(f'Passed schema must have exactly one index column, but None found. Schema:\n{schema}')
            if len(self.index_col_template.matches({index_col})) == 0:
                raise ValueError(
                    f'Passed schema has index column "{index_col}", '
                    f'which does not match index_col_template: {self.index_col_template}'
                )

        if infer_ground_truths is False:
            schema: MLTypeSchema = remove_keys(schema, GROUND_TRUTH_ML_TYPES)

        if infer_predictions is False:
            schema: MLTypeSchema = remove_keys(schema, PREDICTED_ML_TYPES)

        ## Will have MLType.CATEGORICAL, MLType.FLOAT, etc. instead of MLType.GROUND_TRUTH, MLType.PREDICTED:
        inferred_schema_from_cols: Schema = self.infer_from_columns(
            set(mltype_schema.keys()),
            index_col=index_col,
            infer_features=infer_features,
            infer_ground_truths=infer_ground_truths,
            infer_predictions=infer_predictions,
            has_features=has_features,
            has_predictions=has_predictions,
            has_ground_truths=has_ground_truths,
        )
        inferred_col_mltypes: MLTypeSchema = inferred_schema_from_cols.flatten()

        ## Set ground-truths:
        ground_truths_schema: MLTypeSchema = Schema.filter_schema(
            data_schema=schema,
            mltypes=GROUND_TRUTH_ML_TYPES,
        )
        if len(ground_truths_schema) == 0:
            ground_truths_schema: MLTypeSchema = inferred_schema_from_cols.ground_truths_schema
            if has_ground_truths and len(ground_truths_schema) == 0:
                raise ValueError(
                    f'Expected at least one ground-truth column (having MLType in {GROUND_TRUTH_ML_TYPES}), '
                    f'but none were found in schema: {schema}'
                )
        ground_truths_schema: MLTypeSchema = {
            col: inferred_col_mltypes.get(col, schema[col])
            for col, mltype in ground_truths_schema.items()
        }

        ## Set predictions:
        predictions_schema: MLTypeSchema = Schema.filter_schema(
            data_schema=schema,
            mltypes=PREDICTED_ML_TYPES,
        )
        if len(predictions_schema) == 0:
            predictions_schema: MLTypeSchema = inferred_schema_from_cols.predictions_schema
            if has_predictions and len(predictions_schema) == 0:
                raise ValueError(
                    f'Expected at least one predicted column (having MLType in {PREDICTED_ML_TYPES}), '
                    f'but none were found in schema: {schema}'
                )
        predictions_schema: MLTypeSchema = {
            col: inferred_col_mltypes.get(col, schema[col])
            for col, mltype in predictions_schema.items()
        }

        ## Set features:
        features_schema: MLTypeSchema = inferred_schema_from_cols.features_schema
        if has_features and len(features_schema) == 0:
            raise ValueError(
                f'Expected at least one feature column, '
                f'but none were found in schema: {schema}'
            )
        features_schema: MLTypeSchema = {
            col: inferred_col_mltypes.get(col, schema[col])
            for col, mltype in features_schema.items()
        }

        ## Merge remaining columns into features schema:
        cols_so_far: Set[str] = {index_col} \
            .union(set(features_schema.keys())) \
            .union(set(predictions_schema.keys())) \
            .union(set(ground_truths_schema.keys()))
        remaining_schema: MLTypeSchema = {
            col: mltype
            for col, mltype in schema.items()
            if col not in cols_so_far
        }
        features_schema: MLTypeSchema = {
            **remaining_schema,
            **features_schema
        }

        inferred_schema: Schema = Schema(
            index_col=index_col,
            features_schema=features_schema,
            predictions_schema=predictions_schema,
            ground_truths_schema=ground_truths_schema,
        )
        assert inferred_schema.columns_set == set(schema.keys())
        return inferred_schema

    @safe_validate_arguments
    def infer_from_columns(
            self,
            columns: Union[List, Tuple, Set],
            *,
            index_col: Optional[constr(min_length=1)] = None,
            infer_features: bool = True,
            infer_ground_truths: bool = True,
            infer_predictions: bool = True,
            has_features: bool = False,
            has_ground_truths: bool = False,
            has_predictions: bool = False,
    ) -> Schema:
        ## Note: it might not be possible to infer schema for all columns based on their name alone.
        columns_set: Set = as_set(columns)
        schema: Dict[str, Union[MLTypeSchema, str]] = {}
        flat_schema: MLTypeSchema = {}
        ## If infer_* is False, has_* should also become False:
        has_features: bool = has_features and infer_features
        has_ground_truths: bool = has_ground_truths and infer_ground_truths
        has_predictions: bool = has_predictions and infer_predictions

        schema_template_parts = []
        ## This ordering is important:
        if infer_predictions:
            schema_template_parts.append(('predictions_schema', self.predictions_schema_template))
        if infer_ground_truths:
            schema_template_parts.append(('ground_truths_schema', self.ground_truths_schema_template))
        if infer_features:
            schema_template_parts.append(('features_schema', self.features_schema_template))

        for schema_key, schema_template_part in schema_template_parts:
            schema_key_schema: MLTypeSchema = {}
            for col_template, mltype in schema_template_part.items():
                for col in col_template.matches(columns_set):
                    if col == index_col:
                        continue
                    if flat_schema.get(col, mltype) != mltype:
                        raise ValueError(
                            f'Conflict during schema inference; column "{col}" is assigned to MLType {flat_schema[col]}, '
                            f'but it also matches pattern {col_template.regex}, which is assigned to MLType {mltype} '
                            f'as per the following schema template:\n{schema_template_part}'
                        )
                    flat_schema[col] = mltype
                    schema_key_schema[col] = mltype
            if schema_key == 'features_schema' \
                    and has_features \
                    and len(schema_key_schema) == 0 \
                    and len(schema_template_part) > 0:
                raise ValueError(
                    f'Input columns {columns_set} did not match any feature column templates: '
                    f'{self.features_schema_template}'
                )
            if schema_key == 'ground_truths_schema' \
                    and has_ground_truths \
                    and len(schema_key_schema) == 0 \
                    and len(schema_template_part) > 0:
                raise ValueError(
                    f'Input columns {columns_set} did not match any ground-truth column templates: '
                    f'{self.ground_truths_schema_template}'
                )
            if schema_key == 'predictions_schema' \
                    and has_predictions \
                    and len(schema_key_schema) == 0 \
                    and len(schema_template_part) > 0:
                raise ValueError(
                    f'Input columns {columns_set} did not match any predicted column templates: '
                    f'{self.predictions_schema_template}'
                )
            schema[schema_key] = schema_key_schema

        if index_col is None:
            ## index_col must either be passed explicitly, or matchable.
            index_col: Set[str] = set(
                col for col in self.index_col_template.matches(columns_set)  ## Select matching columns...
                if col not in flat_schema  ## ...except those in "flat_schema".
            )
            if len(index_col) == 0:
                raise ValueError(
                    f'Did not match any index columns from {columns_set}; please explicitly pass `index_col`'
                )
            if len(index_col) > 1:
                raise ValueError(
                    f'Expected only one in {columns_set} to match index pattern {self.index_col_template.regex}; '
                    f'found {len(index_col)} matching index columns: {index_col}'
                )
            index_col: str = next(iter(index_col))
            flat_schema[index_col] = MLType.INDEX
        schema['index_col'] = index_col
        inferred_schema: Schema = Schema(**schema)
        assert inferred_schema.columns_set <= columns_set
        return inferred_schema

    def populate(
            self,
            allow_unfilled: bool = False,
            features: bool = True,
            ground_truths: bool = True,
            predictions: bool = True,
            **kwargs
    ) -> Schema:
        ## Populate index col:
        if self.index_col_template.template == INDEX_COL_NAME_TEMPLATE and INDEX_COL_TEMPLATE_KEY not in kwargs:
            kwargs[INDEX_COL_TEMPLATE_KEY] = INDEX_COL_DEFAULT_NAME
        index_col: str = self.index_col_template.populate(allow_unfilled=False, **kwargs)

        features_schema: MLTypeSchema = {}
        if features:
            features_schema: MLTypeSchema = self._populate_templates_dict(
                self.features_schema_template,
                allow_unfilled=allow_unfilled,
                **kwargs
            )
        ground_truths_schema: MLTypeSchema = {}
        if ground_truths:
            ground_truths_schema: MLTypeSchema = self._populate_templates_dict(
                self.ground_truths_schema_template,
                allow_unfilled=allow_unfilled,
                **kwargs
            )

        predictions_schema: MLTypeSchema = {}
        if predictions:
            predictions_schema: MLTypeSchema = self._populate_templates_dict(
                self.predictions_schema_template,
                allow_unfilled=allow_unfilled,
                **kwargs
            )
        return Schema(
            index_col=index_col,
            predictions_schema=predictions_schema,
            ground_truths_schema=ground_truths_schema,
            features_schema=features_schema,
        )

    @classmethod
    def _populate_templates_dict(
            cls,
            templates_dict: Dict[ColTemplate, MLType],
            allow_unfilled: bool,
            **kwargs
    ) -> MLTypeSchema:
        schema: MLTypeSchema = {}
        for col_template, mltype in templates_dict.items():
            col: Optional[Union[List[str], str]] = col_template.populate(
                allow_unfilled=allow_unfilled,
                **kwargs
            )
            if col is None or is_empty_list_like(col):
                continue
            if isinstance(col, str):
                schema[col] = mltype
            elif is_list_like(col):
                for c in col:
                    schema[c] = mltype
        return schema


class Schema(Parameters):
    index_col: str
    features_schema: MLTypeSchema = {}
    predictions_schema: MLTypeSchema = {}
    ground_truths_schema: MLTypeSchema = {}

    @root_validator(pre=True)
    def _set_schema_params(cls, params: Dict) -> Dict:
        try:
            ground_truths_schema: MLTypeSchema = MLType.convert_values(params.get('ground_truths_schema', {}))
            if len(set(ground_truths_schema.values()).intersection(GROUND_TRUTH_ML_TYPES)) > 0:
                raise ValueError(
                    f'Cannot have any of the following MLTypes in `ground_truths_schema`: {GROUND_TRUTH_ML_TYPES}; '
                    f'found following: {Schema.filter_schema(data_schema=ground_truths_schema, mltypes=GROUND_TRUTH_ML_TYPES)}. '
                    f'Please instead use the "data" MLTypes: {DATA_ML_TYPES}'
                )
            params['ground_truths_schema'] = ground_truths_schema

            predictions_schema: MLTypeSchema = MLType.convert_values(params.get('predictions_schema', {}))
            if len(set(predictions_schema.values()).intersection(PREDICTED_ML_TYPES)) > 0:
                raise ValueError(
                    f'Cannot have any of the following MLTypes in `predictions_schema`: {PREDICTED_ML_TYPES}; '
                    f'found following: {Schema.filter_schema(data_schema=predictions_schema, mltypes=PREDICTED_ML_TYPES)}. '
                    f'Please instead use the "data" MLTypes: {DATA_ML_TYPES}'
                )
            params['predictions_schema'] = predictions_schema

            params['features_schema'] = MLType.convert_values(params.get('features_schema', {}))
            return params
        except Exception as e:
            raise ValueError(format_exception_msg(e))

    def index(self) -> str:
        return self.index_col

    def features(self) -> MLTypeSchema:
        return self.features_schema

    def predictions(self) -> MLTypeSchema:
        return self.predictions_schema

    def ground_truths(self) -> MLTypeSchema:
        return self.ground_truths_schema

    def flatten(self) -> MLTypeSchema:
        return {
            self.index_col: MLType.INDEX,
            **self.features_schema,
            **self.ground_truths_schema,
            **self.predictions_schema,
        }

    @property
    def columns_set(self) -> Set[str]:
        return set(self.flatten().keys())

    @property
    def columns(self) -> List[str]:
        return sorted(list(self.columns_set))

    @property
    def has_features(self) -> bool:
        return self.features_schema != {}

    @property
    def has_ground_truths(self) -> bool:
        return self.ground_truths_schema != {}

    @property
    def has_predictions(self) -> bool:
        return self.predictions_schema != {}

    def rename(self, columns: Union[Dict, Callable]) -> Schema:
        if isinstance(columns, dict):
            col_mapper: Callable = lambda col: columns.get(col, col)
        else:
            col_mapper: Callable = columns
        return Schema(
            index_col=col_mapper(self.index_col),
            features_schema={
                col_mapper(col): mltype
                for col, mltype in self.features_schema.items()
            },
            predictions_schema={
                col_mapper(col): mltype
                for col, mltype in self.predictions_schema.items()
            },
            ground_truths_schema={
                col_mapper(col): mltype
                for col, mltype in self.ground_truths_schema.items()
            }
        )

    @staticmethod
    def of(
            schema: Union[Schema, MLTypeSchema],
            schema_template: SchemaTemplate,
            *,
            index_col: Optional[constr(min_length=1)] = None,
            infer_features: bool = True,
            infer_ground_truths: bool = True,
            infer_predictions: bool = True,
            has_features: bool = False,
            has_ground_truths: bool = False,
            has_predictions: bool = False,
    ) -> Optional[Schema]:
        if isinstance(schema, SchemaTemplate):
            raise ValueError(
                f'Cannot instantiate `{Schema.class_name} from an instance of `{SchemaTemplate.class_name}`.'
            )
        if isinstance(schema, Schema):
            return schema
        if isinstance(schema, dict) and set(schema.keys()) <= Schema.param_names():
            ## All keys from schema_template are match variable names in Schema class
            return Schema(**schema)
        if not isinstance(schema, dict):
            raise ValueError(f'Unsupported creation of {Schema.class_name} from data: {schema}')

        ## We have an MLTypeSchema dict:
        return schema_template.infer_from_mltype_schema(
            schema,
            index_col=index_col,
            infer_features=infer_features,
            infer_ground_truths=infer_ground_truths,
            infer_predictions=infer_predictions,
            has_features=has_features,
            has_ground_truths=has_ground_truths,
            has_predictions=has_predictions,
        )

    def set_features(self, features_schema: MLTypeSchema, override: bool = False) -> Schema:
        if self.has_features and override is False:
            raise ValueError(
                f'`features_schema` already set and cannot be overridden on {self.class_name}. '
                f'Current schema: \n{self}'
            )
        return Schema(**{**self.dict(), 'features_schema': features_schema})

    def drop_features(self) -> Schema:
        return Schema(**self.dict(exclude={'features_schema'}))

    def set_predictions(self, predictions_schema: MLTypeSchema, override: bool = False) -> Schema:
        if self.has_predictions and override is False:
            raise ValueError(
                f'`predictions_schema` already set and cannot be overridden on {self.class_name}. '
                f'Current schema: \n{self}'
            )
        return Schema(**{**self.dict(), 'predictions_schema': predictions_schema})

    def drop_predictions(self) -> Schema:
        return Schema(**self.dict(exclude={'predictions_schema'}))

    def predictions_to_features(self) -> Schema:
        return self.drop_predictions().set_features(
            {**self.features_schema, **self.predictions_schema},
            override=True,
        )

    def set_ground_truths(self, ground_truths_schema: MLTypeSchema, override: bool = False) -> Schema:
        if self.has_ground_truths and override is False:
            raise ValueError(
                f'`ground_truths_schema` already set and cannot be overridden on {self.class_name}. '
                f'Current schema: \n{self}'
            )
        return Schema(**{**self.dict(), 'ground_truths_schema': ground_truths_schema})

    def drop_ground_truths(self) -> Schema:
        return Schema(**self.dict(exclude={'ground_truths_schema'}))

    def ground_truths_to_features(self) -> Schema:
        return self.drop_ground_truths().set_features(
            {**self.features_schema, **self.ground_truths_schema},
            override=True,
        )

    def keep_columns(self, cols: Union[List, Tuple, Set]) -> Schema:
        cols: Set = as_set(cols)
        schema: Schema = self
        ## We always keep index column, so do not check that.
        schema: Schema = self.set_features(keep_keys(schema.features_schema, cols), override=True)
        schema: Schema = self.set_ground_truths(keep_keys(schema.ground_truths_schema, cols), override=True)
        schema: Schema = self.set_predictions(keep_keys(schema.predictions_schema, cols), override=True)
        return schema

    def remove_columns(self, cols: Union[List, Tuple, Set]) -> Schema:
        cols: Set = as_set(cols)
        schema: Schema = self
        if schema.index_col in cols:
            raise ValueError(f'Cannot drop index column "{schema.index_col}".')
        schema: Schema = self.set_features(remove_keys(schema.features_schema, cols), override=True)
        schema: Schema = self.set_ground_truths(remove_keys(schema.ground_truths_schema, cols), override=True)
        schema: Schema = self.set_predictions(remove_keys(schema.predictions_schema, cols), override=True)
        return schema

    @staticmethod
    @safe_validate_arguments
    def filter_df(
            df: Any,
            data_schema: Optional[Union[MLTypeSchema, List[str]]] = None,
            allow_missing: bool = False,
            return_series: bool = True,
            sort_columns: bool = True,
            **kwargs
    ) -> Optional[Any]:
        if data_schema is None:
            return df[sorted(list(df.columns))]
        if isinstance(data_schema, dict):
            cols_set: KeysView = data_schema.keys()
        else:
            cols_set: Set = as_set(data_schema)
        if allow_missing:
            cols: List = [
                col for col in df.columns
                if col in cols_set
            ]
        else:
            cols: List = [col for col in data_schema]
        if sort_columns:
            cols: List = sorted(cols)
        if return_series and len(cols) == 1:
            cols: str = cols[0]
        else:
            cols: List = as_list(cols)
        return df[cols]

    @staticmethod
    def remove_missing_columns(cols: List[str], data_schema: MLTypeSchema) -> MLTypeSchema:
        common_cols: Set[str] = set.intersection(as_set(cols), set(data_schema.keys()))
        return {col: data_schema[col] for col in common_cols}

    @classmethod
    def filter_index(cls, data_schema: Optional[MLTypeSchema], allow_missing: bool = False) -> Optional[str]:
        if data_schema is None:
            return None
        return cls.filter_single_column(
            data_schema,
            mltype=MLType.INDEX,
            allow_missing=allow_missing,
        )

    @classmethod
    @safe_validate_arguments
    def filter_single_column(
            cls,
            data_schema: MLTypeSchema,
            mltype: Union[Set[MLType], MLType],
            allow_missing: bool = False,
    ) -> Optional[str]:
        cols: MLTypeSchema = cls.filter_schema(
            data_schema=data_schema,
            mltypes={mltype},
            expected_num_cols=None,
        )
        if len(cols) == 0 and allow_missing:
            return None
        if len(cols) != 1:
            raise ValueError(
                f'Only expected one column with the following MLType(s): {mltype}; '
                f'found {len(cols)} columns: {cols}'
            )
        return next(iter(cols))

    @classmethod
    @safe_validate_arguments
    def filter_schema_columns(
            cls,
            data_schema: MLTypeSchema,
            mltypes: Union[Set[MLType], MLType],
            expected_num_cols: Optional[conint(ge=1)] = None,
    ) -> List[str]:
        cols: List[str] = list(cls.filter_schema(
            data_schema=data_schema,
            mltypes=mltypes,
            expected_num_cols=expected_num_cols
        ).keys())
        cols: List[str] = sorted(cols)
        return cols

    @classmethod
    @safe_validate_arguments
    def filter_schema(
            cls,
            data_schema: MLTypeSchema,
            mltypes: Union[Set[MLType], Tuple[MLType], List[MLType], MLType],
            expected_num_cols: Optional[conint(ge=1)] = None,
            **kwargs
    ) -> MLTypeSchema:
        assert_not_empty_dict(data_schema)
        mltypes: List[MLType] = as_list(mltypes)
        filtered_schema: MLTypeSchema = {col: mltype for col, mltype in data_schema.items() if mltype in mltypes}
        if expected_num_cols is not None:
            if len(filtered_schema) != expected_num_cols:
                raise ValueError(
                    f'Only expected {expected_num_cols} column(s) with the following MLType(s): {mltypes}; '
                    f'found {len(filtered_schema)} columns: {sorted(list(filtered_schema.keys()))}'
                )
        return filtered_schema
    