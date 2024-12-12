from typing import *
from synthergent.base.util import AutoEnum, auto, Utility, set_param_from_alias


class AliasMeta(type):

    def __getattr__(cls, attr_name: str) -> Callable:
        if attr_name.startswith('get_'):
            param_name: str = attr_name.replace('get_', '')
            setter_name: str = f"set_{param_name}"
            if not hasattr(cls, setter_name):
                raise AttributeError(
                    f'`{attr_name}` is does not have a corresponding setter function `{setter_name}`.'
                )
            setter: Callable = getattr(cls, setter_name)

            def getter(params: Dict, *args, pop: bool = True, **kwargs):
                setter(params, *args, **kwargs)
                if pop:
                    return params.pop(param_name, None)
                else:
                    return params.get(param_name, None)

            return getter
        raise AttributeError(
            f'`{attr_name}` is not an attribute of {cls.__name__}.'
        )


class Alias(Utility, metaclass=AliasMeta):

    @classmethod
    def set_AlgorithmClass(cls, params: Dict, param: str = 'AlgorithmClass', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['algorithm', 'AlgorithmClass'],
            **kwargs
        )

    @classmethod
    def set_retry(cls, params: Dict, param: str = 'retry', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['retries', 'num_retries', 'retry'],
            **kwargs
        )

    @classmethod
    def set_data_schema(cls, params: Dict, param: str = 'data_schema', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['schema', 'dataset_schema', 'data_schema'],
            **kwargs
        )

    @classmethod
    def set_data_split(cls, params: Dict, param: str = 'data_split', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['dataset_type', 'split', 'dataset_split', 'data_split', 'predictions_split'],
            **kwargs
        )

    @classmethod
    def set_stream_as(cls, params: Dict, param: str = 'stream_as', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['stream_as', 'stream_layout', 'iter_as'],
            **kwargs
        )

    @classmethod
    def set_num_rows(cls, params: Dict, param: str = 'num_rows', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['batch_size', 'nrows', 'num_rows'],
            **kwargs
        )

    @classmethod
    def set_predict_batch_size(cls, params: Dict, param: str = 'predict_batch_size', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['predict_batch_size', 'eval_batch_size', 'nrows', 'num_rows', 'batch_size'],
            **kwargs
        )

    @classmethod
    def set_num_chunks(cls, params: Dict, param: str = 'num_chunks', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['num_batches', 'nchunks', 'num_chunks'],
            **kwargs
        )

    @classmethod
    def set_shuffle(cls, params: Dict, param: str = 'shuffle', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['shuffle'],
            **kwargs
        )

    @classmethod
    def set_top_k(cls, params: Dict, param: str = 'top_k', **kwargs):
        set_param_from_alias(
            params=params,
            param=param,
            alias=['k', 'top_k'],
            **kwargs
        )

    @classmethod
    def set_seed(cls, params: Dict, param: str = 'seed', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['random_state', 'random_seed', 'seed'],
            **kwargs
        )

    @classmethod
    def set_shard_seed(cls, params: Dict, param: str = 'shard_seed', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['shard_random_state', 'shard_random_seed', 'shard_seed'],
            **kwargs
        )

    @classmethod
    def set_mapper(cls, params: Dict, param: str = 'mapper', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['mapping_fn', 'mapping', 'map', 'mapper'],
            **kwargs
        )

    @classmethod
    def set_map_apply(cls, params: Dict, param: str = 'map_apply', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['mapper_apply', 'mapping_apply', 'map_apply'],
            **kwargs
        )

    @classmethod
    def set_map_executor(cls, params: Dict, param: str = 'map_executor', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['executor', 'map_executor'],
            **kwargs
        )

    @classmethod
    def set_map_failure(cls, params: Dict, param: str = 'map_failure', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['mapper_failure', 'map_failure'],
            **kwargs
        )

    @classmethod
    def set_num_workers(cls, params: Dict, param: str = 'num_workers', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=[
                'num_processes', 'n_processes', 'n_process', 'n_proc', 'n_jobs', 'map_num_workers', 'num_workers',
            ],
            **kwargs
        )

    @classmethod
    def set_parallelize(cls, params: Dict, param: str = 'parallelize', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['map_parallelize', 'parallel', 'parallelize'],
            **kwargs
        )

    @classmethod
    def set_shard_rank(cls, params: Dict, param: str = 'shard_rank', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['shard_idx', 'shard_i', 'shard_rank'],
            **kwargs
        )

    @classmethod
    def set_num_shards(cls, params: Dict, param: str = 'num_shards', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['world_size', 'num_shards'],
            **kwargs
        )

    @classmethod
    def set_format(cls, params: Dict, param: str = 'format', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['file_format', 'format'],
            **kwargs
        )

    @classmethod
    def set_metrics(cls, params: Dict, param: str = 'metrics', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['metric', 'metrics_list', 'metric_list', 'metrics'],
            **kwargs
        )

    @classmethod
    def set_return_predictions(cls, params: Dict, param: str = 'return_predictions', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=[
                'return_preds', 'preds', 'predictions', 'return_predictions',
            ],
            **kwargs
        )

    @classmethod
    def set_predictions_destination(cls, params: Dict, param: str = 'predictions_destination', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=[
                'preds_destination', 'save_predictions', 'save_preds', 'save_preds_to', 'save_to',
                'predictions_destination',
            ],
            **kwargs
        )

    @classmethod
    def set_tracker(cls, params: Dict, param: str = 'tracker', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=[
                'experiment_tracker', 'experiment', 'experiment_name', 'trial', 'trial_name',
                'tracker',
            ],
            **kwargs
        )

    @classmethod
    def set_progress_bar(cls, params: Dict, param: str = 'progress_bar', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=[
                'progress', 'progress_bar', 'pbar',
            ],
            **kwargs
        )

    @classmethod
    def get_progress_bar(cls, params, *, default_progress_bar: bool = True, **kwargs) -> Optional[Dict]:
        Alias.set_progress_bar(params, **kwargs)
        progress_bar: Union[Dict, bool] = params.pop('progress_bar', default_progress_bar)
        if progress_bar is False:
            progress_bar: Optional[Dict] = None
        elif progress_bar is True:
            progress_bar: Optional[Dict] = dict()
        assert progress_bar is None or isinstance(progress_bar, dict)
        return progress_bar

    @classmethod
    def set_silent(cls, params: Dict, param: str = 'silent', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['quiet'],
            **kwargs
        )

    @classmethod
    def set_model_dir(cls, params: Dict, param: str = 'model_dir', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['load_model', 'pretrained_path', 'model_dir'],
            **kwargs
        )

    @classmethod
    def set_cache_dir(cls, params: Dict, param: str = 'cache_dir', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['model_cache_dir', 'model_cache', 'cache_dir'],
            **kwargs
        )

    @classmethod
    def set_cache_timeout(cls, params: Dict, param: str = 'cache_timeout', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=[
                'cache_timeout_sec', 'cache_timeout_seconds',
                'timeout', 'timeout_sec', 'timeout_seconds',
                'cache_model_timeout', 'cache_model_timeout_sec', 'cache_model_timeout_seconds',
                'model_timeout', 'model_timeout_sec', 'model_timeout_seconds',
                'model_cache_timeout', 'model_cache_timeout_sec', 'model_cache_timeout_seconds',
                'actor_timeout', 'actor_timeout_sec', 'actor_timeout_seconds',
                'keepalive', 'keepalive_timeout', 'keepalive_timeout_sec', 'keepalive_timeout_seconds',
            ],
            **kwargs
        )

    @classmethod
    def set_custom_definitions(cls, params: Dict, param: str = 'custom_definitions', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['udfs', 'custom', 'custom_classes', 'custom_definitions'],
            **kwargs
        )

    @classmethod
    def set_verbosity(cls, params: Dict, param: str = 'verbosity', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['verbose', 'verbosity'],
            **kwargs
        )

    @classmethod
    def set_log_file(cls, params: Dict, param: str = 'log_file', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=['log', 'file', 'path', 'fpath', 'log_fpath', ],
            **kwargs,
        )

    @classmethod
    def set_save_model(cls, params: Dict, param: str = 'save_model', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=[
                'save', 'save_to', 'model_save',
                'model_save_path', 'model_save_dir',
                'save_model_path', 'save_model_dir',
                'save_model',
            ],
            **kwargs
        )

    @classmethod
    def set_load_model(cls, params: Dict, param: str = 'load_model', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=[
                'load', 'load_from', 'model_load', 'model_dir',
                'model_load_path', 'model_load_dir',
                'load_model_path', 'load_model_dir',
                'load_model',
            ],
            **kwargs
        )

    @classmethod
    def failure_action(cls, params: Dict, param: str = 'failure_action', **kwargs):
        set_param_from_alias(
            params,
            param=param,
            alias=[
                'error_action', 'on_failure', 'on_error',
                'error_behavior', 'failure_behavior',
                'failure_action',
            ],
            **kwargs
        )
