from typing import *
from synthesizrr.base.util import type_str, optional_dependency, as_list, get_default, EnvUtil, set_param_from_alias, \
    FileSystemUtil
from synthesizrr.base.data import FileMetadata
from synthesizrr.base.framework.evaluator.LocalEvaluator import LocalEvaluator
from synthesizrr.base.framework.algorithm import Algorithm
from synthesizrr.base.framework.task import LanguageModelTaskMixin, GenerativeLM
from pydantic import root_validator, conint

with optional_dependency('accelerate', 'torch', 'transformers'):
    import torch
    from torch.nn import Module as TorchModule
    from synthesizrr.base.framework.dl.torch import PyTorch
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch


    class AccelerateEvaluator(LocalEvaluator):
        aliases = ['accelerate']

        num_devices: Optional[conint(ge=0)] = None
        device_map: Optional[Union[Dict, str]] = None
        model_weights_dtype: Union[torch.dtype, str] = torch.float32
        no_split_module_classes: Optional[List[str]] = None
        use_hf_from_pretrained: bool = False

        @root_validator(pre=True)
        def set_accelerate_evaluator_params(cls, params: Dict) -> Dict:
            set_param_from_alias(params, param='model_weights_dtype', alias=['weights_dtype', 'model_dtype'])
            set_param_from_alias(params, param='use_hf_from_pretrained', alias=[
                'hf_from_pretrained', 'use_hf_auto_model_class', 'hf_auto_model_class',
                'use_hf_AutoModelClass', 'hf_AutoModelClass', 'use_hf_AutoModel', 'hf_AutoModel',
            ])
            set_param_from_alias(params, param='num_devices', alias=['num_gpus', 'gpu_count'])
            if params.get('model_weights_dtype') is not None:
                if isinstance(params['model_weights_dtype'], str):
                    params['model_weights_dtype'] = getattr(torch, params['model_weights_dtype'])
                assert isinstance(params['model_weights_dtype'], torch.dtype)
            return params

        def _load_model(
                self,
                *,
                cache_dir: Optional[Union[FileMetadata, Dict, str]] = None,
                **kwargs,
        ) -> PyTorch:
            from base.algorithm import ALEXA_TM_SEQ2SEQ_MODEL_NAMES

            kwargs.pop('device', None)  ## We manage the device-allocation in the rest of this function.
            kwargs.pop('model_dir', None)  ## Do not allow overriding model_dir
            kwargs.pop('num_devices', None)  ## Use the one passed to evaluator.
            # print(f'Loading model copy with kwargs: {kwargs}')
            cache_dir: FileMetadata = FileMetadata.of(
                get_default(cache_dir, self.cache_dir)
            ).mkdir(return_metadata=True)
            # print(f'{self.class_name} using cache dir: "{cache_dir.path}"')

            num_devices: conint(ge=0) = get_default(
                self.num_devices,
                EnvUtil.num_gpus(),  ## By default, use all GPUs available.
            )
            # print(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
            # print(f'cuda_visible_devices: {EnvUtil.cuda_visible_devices()}')
            # print(f'num_devices: {num_devices}')

            alexa_tm_model_name: Optional[str] = self._create_hyperparams().dict().get('model_name')
            if alexa_tm_model_name is None:
                alexa_tm_model_name: Optional[str] = self._create_hyperparams().dict() \
                    .get('lm', {}).get('hyperparams', {}).get('model_name')
            # print(f'alexa_tm_model_name: {alexa_tm_model_name}')
            if alexa_tm_model_name in ALEXA_TM_SEQ2SEQ_MODEL_NAMES:
                return self._load_alexa_tm_model_copy(
                    cache_dir=cache_dir,
                    num_devices=num_devices,
                    **kwargs
                )
            if self.use_hf_from_pretrained:
                return self._load_hf_auto_model_class(
                    cache_dir=cache_dir,
                    num_devices=num_devices,
                    **kwargs
                )
            return self._load_model_copy_accelerate(
                cache_dir=cache_dir,
                num_devices=num_devices,
                **kwargs
            )

        def _load_hf_auto_model_class(
                self,
                num_devices: conint(ge=0),
                hyperparams: Optional[Dict] = None,
                **kwargs,
        ) -> Union[LanguageModelTaskMixin, PyTorch]:
            from base.algorithm.huggingface.transformers import HFPyTorchModel
            device_map: Union[Dict, str] = self._hf_accelerate_device_map(num_devices=num_devices)
            hyperparams: Dict = get_default(hyperparams, self.hyperparams)
            hyperparams: Dict = {
                **hyperparams,
                **dict(
                    device_map=device_map,
                )
            }
            model: Algorithm = Algorithm.of(**{
                **dict(
                    task=self.task,
                    algorithm=self.AlgorithmClass,
                    hyperparams=hyperparams,
                    model_dir=self.model_dir,
                ),
                **kwargs,
            })
            if isinstance(model, LanguageModelTaskMixin):
                hf_pt_model: GenerativeLM = model.lm
            else:
                hf_pt_model: Algorithm = model
            if not isinstance(hf_pt_model, HFPyTorchModel):
                raise ValueError(
                    f'Can only use {self.class_name} with subclasses of {HFPyTorchModel}; '
                    f'found model with type: {type_str(hf_pt_model)}'
                )
            return model

        def _load_model_copy_accelerate(
                self,
                cache_dir: FileMetadata,
                num_devices: conint(ge=0),
                **kwargs,
        ) -> PyTorch:
            ## If model directory is not empty, download the model to the cache dir:
            model_dir: Optional[FileMetadata] = self.download_remote_model_to_cache_dir(cache_dir, **kwargs)

            ## Load the model into empty weights (i.e. Torch's "meta" device):
            with init_empty_weights():
                model: Algorithm = Algorithm.of(**{
                    **dict(
                        task=self.task,
                        algorithm=self.AlgorithmClass,
                        hyperparams=self.hyperparams,
                    ),
                    **kwargs,
                    **dict(
                        cache_dir=cache_dir,
                        model_dir=model_dir,
                        post_init=False,  ## When using accelerate, first init an empty model, then split.
                        init_empty=True,  ## For HuggingFace models, calls .from_config() instead of .from_pretrained()
                    ),
                })
            if isinstance(model, LanguageModelTaskMixin):
                pt_model: GenerativeLM = model.lm
            else:
                pt_model: Algorithm = model

            if not isinstance(pt_model, PyTorch):
                raise ValueError(
                    f'Can only use {self.class_name} with subclasses of {PyTorch}; '
                    f'found model with type: {type_str(pt_model)}'
                )

            pt_model.model = self._accelerate_load_checkpoint_and_dispatch(
                pt_model=pt_model,
                num_devices=num_devices,
                model_dir=model_dir,
                cache_dir=cache_dir,
                **kwargs
            )
            pt_model.device = 'cuda'
            if isinstance(model, LanguageModelTaskMixin):
                model.lm = pt_model
            else:
                model: PyTorch = pt_model
            return model

        def _accelerate_load_checkpoint_and_dispatch(
                self,
                pt_model: PyTorch,
                num_devices: conint(ge=0),
                model_dir: Optional[FileMetadata],
                cache_dir: Union[FileMetadata, Dict, str],
                **kwargs
        ) -> TorchModule:
            from base.algorithm.huggingface.transformers import HFPyTorchModel
            if model_dir is None:
                if isinstance(pt_model, HFPyTorchModel):
                    model_dir: FileMetadata = pt_model.download_model_to_cache_dir(cache_dir=cache_dir, **kwargs)
                else:
                    raise ValueError(f'For non-HuggingFace PyTorch models, you must specify model_dir for loading.')
            assert isinstance(model_dir, FileMetadata)
            model_dir_path: str = model_dir.path

            ## Move to GPU:
            for file_glob in [
                ## HF Accelerate allows passing `checkpoint` as path to a ".json" file:
                'pytorch_model.bin.index.json',
                '*.index.json',
                ## HF Accelerate allows passing `checkpoint` as path to a "pytorch_model.bin" file.
                'pytorch_model.bin',
            ]:
                checkpoint_file_path: Optional[str] = self._checkpoint_file_path_in_pt_model_snapshot_dir(
                    model_dir_path,
                    file_glob=file_glob,
                )
                if checkpoint_file_path is not None:
                    break
            if checkpoint_file_path is not None:
                model_dir_path: str = checkpoint_file_path

            no_split_module_classes: Optional[List[str]] = self.no_split_module_classes
            if isinstance(pt_model, HFPyTorchModel) and no_split_module_classes is None:
                no_split_module_classes: List[str] = pt_model.model._no_split_modules  ## Param on PreTrainedModel
            if no_split_module_classes is None:
                raise ValueError(f'Expected `no_split_module_classes`; could not infer, please pass it explicitly.')

            device_map: Union[Dict, str] = self._hf_accelerate_device_map(num_devices=num_devices)
            ## Ref: https://huggingface.co/docs/accelerate/usage_guides/big_modeling
            ## "Note that loading the model with from_config in Transformers does not tie the weights, which may cause
            ## issue when loading a checkpoint that does not contain duplicate keys for the tied weights. So you
            ## should tie the weights before loading the checkpoint.
            pt_model.model.tie_weights()
            return load_checkpoint_and_dispatch(
                pt_model.model,
                checkpoint=model_dir_path,
                device_map=device_map,
                no_split_module_classes=as_list(no_split_module_classes),
                dtype=self.model_weights_dtype,
            )

        def _hf_accelerate_device_map(self, *, num_devices: int) -> Union[Dict, str]:
            device_map: Union[Dict, str] = get_default(
                self.device_map,
                "balanced_low_0" if num_devices > 2 else "auto",
            )
            return device_map

        @classmethod
        def _checkpoint_file_path_in_pt_model_snapshot_dir(
                cls,
                snapshot_model_dir_path: str,
                file_glob: str
        ) -> Optional[str]:
            snapshot_matching_files: List[str] = FileSystemUtil.list(
                snapshot_model_dir_path,
                file_glob=file_glob,
            )
            if len(snapshot_matching_files) == 0:
                return None
            if len(snapshot_matching_files) != 1:
                raise ValueError(
                    f'To load checkpoints, expected exactly one file in directory "snapshot_model_dir_path" which '
                    f'matches glob "{file_glob}"; instead found following {len(snapshot_matching_files)} files: '
                    f'{snapshot_matching_files}'
                )
            return snapshot_matching_files[0]

        def _load_alexa_tm_model_copy(
                self,
                cache_dir: FileMetadata,
                num_devices: conint(ge=0),
                **kwargs,
        ) -> PyTorch:
            from base.algorithm import AlexaTMSeq2Seq
            if num_devices % 2 != 0:
                raise ValueError(f'AlexaTM 20B can only be distributed across an even number of devices.')
            ## Load the model into CPU memory:
            assert cache_dir is not None
            model: Algorithm = Algorithm.of(**{
                **dict(
                    task=self.task,
                    algorithm=self.AlgorithmClass,
                    hyperparams=self.hyperparams,
                    model_dir=self.model_dir,
                ),
                **kwargs,
                **dict(
                    cache_dir=cache_dir,
                    post_init=False,  ## When using accelerate, first init an empty model, then split.
                ),
            })
            if isinstance(model, LanguageModelTaskMixin):
                pt_model: AlexaTMSeq2Seq = model.lm
            else:
                pt_model: AlexaTMSeq2Seq = model
            if not isinstance(pt_model, AlexaTMSeq2Seq):
                raise ValueError(f'Expected AlexaTM 20B model, found: {type_str(pt_model)}')
            ## Move to GPU:
            if self.model_weights_dtype == torch.float32:
                pt_model.model.float()
            elif self.model_weights_dtype == torch.float16:
                pt_model.model.half()
            elif self.model_weights_dtype == torch.bfloat16:
                pt_model.model.bfloat16()
            else:
                raise NotImplementedError(
                    f'Unsupported value for `model_weights_dtype`: {self.model_weights_dtype}'
                )

            pt_model.model.parallelize(num_devices)
            pt_model.device = 'cuda'
            if isinstance(model, LanguageModelTaskMixin):
                model.lm = pt_model
            else:
                model: AlexaTMSeq2Seq = pt_model
            return model
