from typing import *
import gc, os
from abc import ABC, abstractmethod
from synthesizrr.base.data.sdf import ScalableDataFrameOrRaw
from synthesizrr.base.framework.algorithm import Algorithm
from synthesizrr.base.framework.task_data import Dataset
from synthesizrr.base.framework.predictions import Predictions
from synthesizrr.base.constants import DataLayout, MLType, DataPosition, MLTypeSchema
from synthesizrr.base.util import optional_dependency, safe_validate_arguments, MappedParameters, get_default, check_isinstance
from pydantic import validator, root_validator, conint
from functools import partial

with optional_dependency('torch'):
    import torch
    from torch import Tensor
    from torch.nn import Module as TorchModule
    from torch.optim import Optimizer as TorchOptimizer
    from torch.optim.lr_scheduler import _LRScheduler as TorchLRScheduler
    from torch.nn.modules.loss import _Loss as TorchLoss
    from synthesizrr.base.data.sdf.TorchScalableSeries import TorchScalableSeries
    from torch.utils.data import IterableDataset as TorchIterableDataset, DataLoader as TorchDataLoader

    accelerate = None
    with optional_dependency('accelerate', error='ignore'):
        import accelerate


    def is_accelerator(device: Any) -> bool:
        return accelerate is not None and isinstance(device, accelerate.Accelerator)


    def move_tensor_to_device(x: Any, device: Any, **kwargs) -> Any:
        """Moves a torch.Tensor to a device. Supports HuggingFace Accelerate."""
        if not isinstance(x, Tensor):
            return x
        if is_accelerator(device):
            device = device.device
            # print(f'(pid={os.getpid()}) Moving data of shape {x.shape} to device {device} of accelerator {accelerate}')
        else:
            # print(f'(pid={os.getpid()}) Moving data of shape {x.shape} to device {device}')
            pass
        return x.to(device=device, **kwargs)


    def move_data_to_device(
            vals: Union[Tuple, List, Set, Dict, Any],
            device: Any,
            **kwargs
    ) -> Union[Tuple, List, Set, Dict, Any]:
        """Moves a collection of torch.Tensor or torch.nn.Module to a device"""
        if isinstance(vals, list):
            return [move_tensor_to_device(v, device=device, **kwargs) for v in vals]
        elif isinstance(vals, tuple):
            return tuple([move_tensor_to_device(v, device=device, **kwargs) for v in vals])
        elif isinstance(vals, set):
            return set([move_tensor_to_device(v, device=device, **kwargs) for v in vals])
        elif isinstance(vals, dict):
            return {k: move_tensor_to_device(v, device=device, **kwargs) for k, v in vals.items()}
        else:
            return move_tensor_to_device(vals, device=device, **kwargs)


    def get_model_device(model: TorchModule, allow_multiple: bool = False) -> Union[torch.device, Set[torch.device]]:
        device = {x.device for x in model.parameters()}
        if len(device) > 1:
            if allow_multiple:
                return device
            raise ValueError(f'Found multiple devices for model: ')
        elif len(device) == 0:
            raise ValueError(f'Did not find any parameters for model.')
        return next(iter(device))


    def models_are_equal(model_1: TorchModule, model_2: TorchModule, raise_error: bool = True) -> bool:
        mismatches = []

        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                continue
            ## Models are not equal at certain position:
            if key_item_1[0] == key_item_2[0]:
                mismatches.append(f'Mismtach found at: {key_item_1[0]}')
            else:
                if raise_error:
                    raise ValueError(f'Models do not have the same architecture.')
                return False
        if len(mismatches) > 0:
            if raise_error:
                mismatches_str: str = "\n".join(mismatches)
                raise ValueError(f'Models differ as follows:\n{mismatches_str}')
            return False
        return True


    def validate_data_on_device(
            tensor: Tensor,
            device: torch.device,
            raise_error: bool = True,
    ) -> bool:
        if not isinstance(tensor, Tensor):
            if raise_error:
                raise ValueError(f'Expected value to be PyTorch Tensor; found object of type: {type(tensor)}')
            return False
        if is_accelerator(device):
            device = device.device
        if tensor.device != device:
            if raise_error:
                raise ValueError(
                    f'Expected PyTorch Tensor be on device "{device}"; however it was on device "{tensor.device}"'
                )
            return False
        return True


    def clear_device_cache():
        for _ in range(2):
            ## Sometimes torch depends on objects in Python memory, and sometimes the other way around.
            ## Running this twice ensures cleanup of both torch and python objects.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


    class Optimizer(MappedParameters):
        _mapping: ClassVar[Dict[str, Type[TorchOptimizer]]] = {
            name: val
            for name, val in torch.optim.__dict__.items()
            if isinstance(val, type) and issubclass(val, TorchOptimizer)
        }


    class Loss(MappedParameters):
        _mapping: ClassVar[Dict[str, Type[TorchLoss]]] = {
            'KLDivLoss': torch.nn.KLDivLoss,
            'NLLLoss': torch.nn.NLLLoss,
            'SmoothL1Loss': torch.nn.SmoothL1Loss,
            'HuberLoss': torch.nn.HuberLoss,
            'MultiLabelMarginLoss': torch.nn.MultiLabelMarginLoss,
            'HingeEmbeddingLoss': torch.nn.HingeEmbeddingLoss,
            'SoftMarginLoss': torch.nn.SoftMarginLoss,
            'MultiMarginLoss': torch.nn.MultiMarginLoss,
            'CosineEmbeddingLoss': torch.nn.CosineEmbeddingLoss,
            'TripletMarginLoss': torch.nn.TripletMarginLoss,
            'MarginRankingLoss': torch.nn.MarginRankingLoss,
            'CTCLoss': torch.nn.CTCLoss,
            'CrossEntropyLoss': torch.nn.CrossEntropyLoss,
            'BCELoss': torch.nn.BCELoss,
            'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss,
            'L1Loss': torch.nn.L1Loss,
            'MSELoss': torch.nn.MSELoss,
        }


    ## Copied from `transformers`:
    @safe_validate_arguments
    def get_linear_schedule_with_warmup(
            optimizer: torch.optim.Optimizer,
            num_warmup_steps: conint(ge=1),
            num_training_steps: conint(ge=1),
            last_epoch: int = -1
    ):
        from torch.optim.lr_scheduler import LambdaLR
        """
        Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
        a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

        Args:
            optimizer ([`~torch.optim.Optimizer`]):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            num_training_steps (`int`):
                The total number of training steps.
            last_epoch (`int`, *optional*, defaults to -1):
                The index of the last epoch when resuming training.

        Return:
            `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """
        lr_lambda = partial(
            _get_linear_schedule_with_warmup_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return LambdaLR(optimizer, lr_lambda, last_epoch)


    def _get_linear_schedule_with_warmup_lr_lambda(
            current_step: int, *,
            num_warmup_steps: int,
            num_training_steps: int
    ):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


    class LRScheduler(MappedParameters):
        _mapping: ClassVar[Dict[str, Type[TorchLRScheduler]]] = {
            'LinearLR': torch.optim.lr_scheduler.LinearLR,
            'ConstantLR': torch.optim.lr_scheduler.ConstantLR,
            'LambdaLR': torch.optim.lr_scheduler.LambdaLR,
            'MultiplicativeLR': torch.optim.lr_scheduler.MultiplicativeLR,
            'StepLR': torch.optim.lr_scheduler.StepLR,
            'MultiStepLR': torch.optim.lr_scheduler.MultiStepLR,
            'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR,
            'SequentialLR': torch.optim.lr_scheduler.SequentialLR,
            'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
            'ChainedScheduler': torch.optim.lr_scheduler.ChainedScheduler,
            'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'CyclicLR': torch.optim.lr_scheduler.CyclicLR,
            'CosineAnnealingWarmRestarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            'OneCycleLR': torch.optim.lr_scheduler.OneCycleLR,
            'PolynomialLR': torch.optim.lr_scheduler.PolynomialLR,
            ## Equivalent to get_linear_schedule_with_warmup from transformers
            'linear_schedule_with_warmup': get_linear_schedule_with_warmup,
        }


    class PyTorch(Algorithm, ABC):
        model_file_name: ClassVar[str] = 'model.pt'

        default_batching_params: ClassVar[Dict[str, Any]] = {
            'stream_as': DataLayout.DICT,
        }
        batch_dim: ClassVar[int] = 0
        input_dtype: ClassVar[Optional[torch.dtype]] = None
        output_dtype: ClassVar[Optional[torch.dtype]] = None
        target_dtype: ClassVar[Optional[torch.dtype]] = None

        ## IMP! typing of these fields must be "Any" to avoid pydantic error: "Value not declarable with JSON Schema"
        model: Optional[Any] = None
        optimizer: Optional[Any] = None
        loss: Optional[Any] = None
        lr_scheduler: Optional[Any] = None
        device: Any = 'cpu'  ## Use CPU by default.

        class Hyperparameters(Algorithm.Hyperparameters):
            optimizer: Optional[Union[Optimizer, Dict, str]] = None
            loss: Optional[Union[Loss, Dict, str]] = None
            lr_scheduler: Union[LRScheduler, Dict, str] = {
                ## ConstantLR(factor=1, total_iters=0) will not change the learning rate at all.
                ## Note that the LRScheduler *assigns* the learning rate, the optimizer *uses* this learning rate.
                'name': 'ConstantLR',
                'factor': 1,
                'total_iters': 0,
            }
            gradient_accumulation_steps: conint(ge=1) = 1

            @root_validator(pre=False)  ## Run this post all values set by subclasses.
            def convert_hyperparams(cls, hyperparams: Dict) -> Dict:
                if hyperparams.get('optimizer', None) is not None:
                    hyperparams['optimizer'] = Optimizer.of(hyperparams['optimizer'])

                if hyperparams.get('loss', None) is not None:
                    hyperparams['loss'] = Loss.of(hyperparams['loss'])

                if hyperparams.get('lr_scheduler', None) is not None:
                    hyperparams['lr_scheduler'] = LRScheduler.of(hyperparams['lr_scheduler'])

                return hyperparams

        def __str__(self):
            params_str: str = self.json(indent=4, include={'hyperparams'})
            out: str = f'{self.class_name} running on {self.device} with params:\n{params_str}'
            return out

        def post_initialize(self):
            self.transfer_model()

        def init_training_components(self):
            if not isinstance(self.model, TorchModule):
                raise ValueError(
                    f'.initialize() should create or load a subclass of torch.nn.Module; '
                    f'found object of type: {type(self.model)}.'
                )
            if self.optimizer is None:
                if self.hyperparams.optimizer is None:
                    raise ValueError(
                        f'Please pass `optimizer` in hyperparams. '
                        f'`optimizer` should be a dict, where the "name" key is the name of a subclass of '
                        f'torch.optim.Optimizer (e.g. "AdamW"), and the other keys are constructor-args '
                        f'for this Optimizer (e.g. "lr", "weight_decay", "eps", etc.)'
                    )
                self.optimizer = self.init_optimizer()
            if self.loss is None:
                if self.hyperparams.loss is None:
                    raise ValueError(
                        f'Please pass `loss` in hyperparams. '
                        f'`loss` should be a dict, where the "name" key is the name of a subclass of '
                        f'torch.nn.modules.loss._Loss (e.g. "CrossEntropyLoss"), and the other keys are '
                        f'constructor-args for this Loss (e.g. "weight", "label_smoothing", etc.)'
                    )
                self.loss = self.init_loss()
            if self.lr_scheduler is None:
                if self.hyperparams.lr_scheduler is None:
                    raise ValueError(
                        f'Please pass `lr_scheduler` in hyperparams. '
                        f'`lr_scheduler` should be a dict, where the "name" key is the name of a subclass of '
                        f'torch.optim.lr_scheduler._LRScheduler (e.g. "ConstantLR"), and the other keys are '
                        f'constructor-args for this LRScheduler (e.g. "factor", "total_iters", etc.)'
                    )
                ## TODO: add warmup steps via `from transformers.optimization import get_scheduler`, or another way.
                self.lr_scheduler = self.init_lr_scheduler()

        def init_optimizer(self) -> torch.optim.Optimizer:
            return self.hyperparams.optimizer.initialize(params=self.model.parameters())

        def init_loss(self) -> torch.nn.modules.loss._Loss:
            return self.hyperparams.loss.initialize()

        def init_lr_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
            if self.hyperparams.lr_scheduler.mapped_callable() == get_linear_schedule_with_warmup:
                return self.hyperparams.lr_scheduler.initialize(
                    optimizer=self.optimizer,
                    num_training_steps=self.hyperparams.steps,
                    ## num_warmup_steps is expected to be a param of the LR Scheduler.
                )
            return self.hyperparams.lr_scheduler.initialize(optimizer=self.optimizer)

        def transfer_model(self):
            self.model = self.model.to(device=self.device)

        @staticmethod
        def preprocess(batch: Dataset, **kwargs) -> Dataset:
            if batch.data.layout is DataLayout.DICT:
                batch = batch.assets_to_tensor(tensor_type='pt', stack=True)
            else:
                batch = batch.assets_to_tensor(tensor_type='pt')
            return batch

        @safe_validate_arguments
        def train_iter(
                self,
                dataset: Any,
                **kwargs,
        ) -> Generator[Optional[Dict], None, None]:
            self.init_training_components()
            self.prepare_model_for_train(**kwargs)
            for train_step_metrics in super(PyTorch, self).train_iter(dataset, **kwargs):
                yield train_step_metrics

        def prepare_model_for_train(self, **kwargs):
            self.model.train()

        def train_step(
                self,
                batch: Dataset,
                input_dtype: Optional[torch.dtype] = None,
                output_dtype: Optional[torch.dtype] = None,
                target_dtype: Optional[torch.dtype] = None,
                **kwargs
        ):
            input_dtype: Optional[torch.dtype] = get_default(input_dtype, self.input_dtype)
            output_dtype: Optional[torch.dtype] = get_default(output_dtype, self.output_dtype)
            target_dtype: Optional[torch.dtype] = get_default(target_dtype, self.target_dtype)

            input: Any = self.prepare_input(batch, **kwargs)
            input: Any = move_data_to_device(input, device=self.device, dtype=input_dtype)

            output: Any = self.forward(input, **kwargs)
            output: Any = move_data_to_device(output, device=self.device, dtype=output_dtype)

            target: Any = self.prepare_target(batch, **kwargs)
            target: Any = move_data_to_device(target, device=self.device, dtype=target_dtype)

            loss: Tensor = self.prepare_training_loss(output=output, target=target, batch=batch, **kwargs)

            ## Ref: docs.ray.io/en/latest/train/examples/transformers/transformers_example.html
            loss = loss / self.hyperparams.gradient_accumulation_steps
            if is_accelerator(self.device):
                self.device.backward(loss)  ## accelerator.backward(loss)
            else:
                loss.backward()
            if (
                    batch.data_idx % self.hyperparams.gradient_accumulation_steps == 0
                    or batch.data_position is DataPosition.END  ## Last batch in epoch.
            ):
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            return {'loss': float(loss)}

        def prepare_prediction_dataset(
                self,
                dataset: Any,
                data_schema: Optional[MLTypeSchema] = None,
                **kwargs,
        ) -> Any:
            if isinstance(dataset, TorchDataLoader):
                return dataset
            else:
                return super(PyTorch, self).prepare_prediction_dataset(dataset, data_schema=data_schema, **kwargs)

        @safe_validate_arguments
        def predict_iter(
                self,
                dataset: Any,
                **kwargs,
        ) -> Generator[Optional[Predictions], None, None]:
            self.prepare_model_for_predict(**kwargs)
            predict_epoch_gen: Generator[Predictions, None, None] = super(PyTorch, self).predict_iter(
                dataset,
                **kwargs,
            )
            for predictions in predict_epoch_gen:
                yield predictions

        def prepare_model_for_predict(self, **kwargs):
            self.model.eval()

        def predict_step(
                self,
                batch: Dataset,
                input_dtype: Optional[torch.dtype] = None,
                output_dtype: Optional[torch.dtype] = None,
                return_tensors: bool = False,
                **kwargs
        ) -> Any:
            input_dtype: Optional[torch.dtype] = get_default(input_dtype, self.input_dtype)
            output_dtype: Optional[torch.dtype] = get_default(output_dtype, self.output_dtype)
            with torch.no_grad():
                input: Any = self.prepare_input(batch, **kwargs)
                input: Any = move_data_to_device(input, device=self.device, dtype=input_dtype)
                output: Any = self.forward(input, **kwargs)
                if return_tensors is True:
                    return output
                output: Any = move_data_to_device(output, device='cpu', dtype=output_dtype)
                # if is_accelerator(self.device):
                #     output: Any = self.device.gather(output)
                return self.prepare_predictions(output, input=input, **kwargs)

        def prepare_input(
                self,
                batch: Dataset,
                **kwargs,
        ) -> Any:
            return batch.features().torch()

        @abstractmethod
        def forward(self, input: Any, **kwargs) -> Any:
            pass

        def prepare_target(
                self,
                batch: Dataset,
                **kwargs,
        ) -> Any:
            return batch.ground_truths().torch()

        def prepare_training_loss(self, output: Any, target: Any, batch: Dataset, **kwargs) -> Tensor:
            # print(f'(pid={os.getpid()}) output.shape: {output.shape}, target.shape: {target.shape}')
            loss: Tensor = self.loss.forward(output, target, **kwargs)
            return loss

        @abstractmethod
        def prepare_predictions(self, output: Any, **kwargs) -> Any:
            """
            Convert output tensor to input to _create_predictions function.
            """
            pass

        def _validate_dims(
                self,
                tensor: Tensor,
                batch_size: int,
                raise_error: bool = True,
        ) -> bool:
            if not isinstance(tensor, Tensor):
                if raise_error:
                    raise ValueError(f'Expected value to be PyTorch Tensor; found object of type: {type(tensor)}')
                return False
            if tensor.shape[self.batch_dim] != batch_size:
                if raise_error:
                    raise ValueError(
                        f"Expected PyTorch Tensor to have dimension {self.batch_dim} as batch dimension, "
                        f"with length {batch_size}; found tensor of shape: {tensor.shape}"
                    )
                return False
            return True

        @classmethod
        def save_param_names(cls) -> Set[str]:
            return super(PyTorch, cls).save_param_names() - {
                ## Remove all params PyTorch-specific params which should not be saved in __model_params__.pkl file:
                'model', 'optimizer', 'loss', 'lr_scheduler', 'device'
            }

        def post_train_cleanup(self):
            optimizer, loss, lr_scheduler = self.model, self.optimizer, self.loss
            self.optimizer, self.loss, self.lr_scheduler = None, None, None
            del optimizer, loss, lr_scheduler
            clear_device_cache()

        def cleanup(self):
            super(PyTorch, self).cleanup()
            model = self.model
            self.model = None
            del model
            clear_device_cache()
