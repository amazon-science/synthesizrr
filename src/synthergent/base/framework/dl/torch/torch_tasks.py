from typing import *
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from synthergent.base.framework.algorithm import Algorithm
from synthergent.base.framework.task_data import Dataset
from synthergent.base.framework.task import Embedder, \
    EncodingRange, Classifier, ClassificationData, MultiLabelClassifier, MultiLabelClassificationData, \
    Regressor
from synthergent.base.constants import Task, TaskOrStr, MLType, Storage
from synthergent.base.data import FileMetadata
from synthergent.base.util import optional_dependency, FileSystemUtil
from pydantic import validator, root_validator, Extra, confloat
from pydantic.typing import Literal

with optional_dependency('torch'):
    import torch
    from torch import Tensor
    from torch.nn import Module as TorchModule
    from torch.optim import Optimizer as TorchOptimizer
    from torch.optim.lr_scheduler import _LRScheduler as TorchLRScheduler
    from torch.nn.modules.loss import _Loss as TorchLoss
    from torch.nn.functional import softmax

    from synthergent.base.framework.dl.torch.torch_base import PyTorch, Loss, Optimizer, LRScheduler, \
        move_data_to_device, get_model_device, is_accelerator


    class PyTorchBaseModel(Embedder, PyTorch, ABC):
        @classmethod
        def _pre_registration_hook(cls):
            super(PyTorchBaseModel, cls)._pre_registration_hook()
            if cls.embed_single == PyTorchBaseModel.embed_single and cls.embed_multi == PyTorchBaseModel.embed_multi:
                raise TypeError(
                    f'Class {cls} is a subclass of {PyTorchBaseModel} and must implement either '
                    f'.embed_single() or .embed_multi() functions to generate embeddings. '
                    f'You can also optionally implement ._prepare_input() to change the input Tensor sent to these '
                    f'function.'
                )

        def embed_single(self, input: Any, **kwargs) -> Tensor:
            ## Returns logits attached to the computational graph.
            ## IMP! do not call self.model.train() or self.model.eval() here...do that in train_step or predict_step.
            raise NotImplementedError()

        def embed_multi(self, input: Any, **kwargs) -> Tensor:
            ## Returns list of logits attached to the computational graph.
            ## IMP! do not call self.model.train() or self.model.eval() here...do that in train_step or predict_step.
            ## We expect the output to be a Tensor of shape: (batch_size, num_tokens, embedding_size)
            raise NotImplementedError()

        @abstractmethod
        def embedding_size(self) -> int:
            """Returns the embedding vector size for the current embedding model."""
            pass

        def forward(
                self,
                input: Any,
                *,
                required_embeddings: Literal['single', 'multi'] = 'single',
                **kwargs,
        ) -> Union[Tensor, List[Tensor]]:
            if required_embeddings == 'single':
                output: Tensor = self.embed_single(input, **kwargs)
            else:
                assert required_embeddings == 'multi'
                output: List[Tensor] = self.embed_multi(input, **kwargs)
            return output

        def prepare_predictions(self, output: Union[Tensor, List[Tensor]], **kwargs) -> List[np.ndarray]:
            if isinstance(output, Tensor):
                return list(output.cpu().detach().numpy())
            raise ValueError(
                f'{Embedder} does not support storing multiple embeddings at the moment; '
                f'found output of type {type(output)}'
            )


    class BaseModelWithHead(TorchModule):
        def __init__(
                self,
                base_model: PyTorchBaseModel,
                head: TorchModule,
                required_embeddings: Literal['single', 'multi'],
        ):
            super(BaseModelWithHead, self).__init__()
            self._base_model: PyTorchBaseModel = base_model
            self.base: TorchModule = base_model.model  ## Expose the raw model for .to(device) calls and printing.
            self.head: TorchModule = head
            self.required_embeddings: Literal['single', 'multi'] = required_embeddings

        def forward(self, input: Any, **kwargs) -> Any:
            embeddings: Union[Tensor, List[Tensor]] = self._base_model.forward(
                input,
                required_embeddings=self.required_embeddings,
                **kwargs,
            )
            ## Pass the embeddings to the task-head:
            output: Tensor = self.head.forward(embeddings, **kwargs)
            return output


    class PyTorchTaskHead(PyTorch, ABC):
        aliases = ['PyTorch']

        required_embeddings: ClassVar[Literal['single', 'multi']]
        prepare_input_fn: Optional[Callable] = None

        class Hyperparameters(PyTorch.Hyperparameters):
            base_model: Dict  ## Params for base model which generates embeddings.
            frozen_base: bool = False  ## Whether to freeze the base model when performing training.

        def initialize(self, model_dir: Optional[FileMetadata] = None):
            base_model: PyTorchBaseModel = PyTorchBaseModel.of(
                model_dir=model_dir,
                device=self.device,
                ## Do not setup optimizer etc. or move base_model to device, that will be done by the post_initialize
                ## call in the instance of PyTorchTaskHead.
                post_init=False,
                **{
                    **self.hyperparams.base_model,
                    'task': Task.EMBEDDING,
                },
            )
            ## Don't initialize optimizer, loss, etc for base_model...we will use those from the task-head class.
            ## Also don't move the model to the device, we will move it as part of the task-head class.
            self.prepare_input_fn = base_model.prepare_input
            if model_dir is None:
                head: TorchModule = self._create_head(base_model)
            else:
                ## Load model from disk:
                if model_dir.storage is not Storage.LOCAL_FILE_SYSTEM:
                    raise ValueError(
                        f'Can only load models from local disk, not the following {model_dir.storage} path: '
                        f'"{model_dir.path}"'
                    )
                head: TorchModule = torch.load(self._task_head_save_path(model_dir), map_location=self.device)
            self.model: BaseModelWithHead = BaseModelWithHead(
                base_model=base_model,
                head=head,
                required_embeddings=self.required_embeddings,
            )

        @abstractmethod
        def _create_head(self, base_model: PyTorchBaseModel) -> TorchModule:
            pass

        def save(self, model_dir: FileMetadata):
            model = self.model
            if is_accelerator(self.device):
                model = self.device.unwrap_model(model)
            FileSystemUtil.mkdir_if_does_not_exist(model_dir.path)
            model._base_model.save(model_dir)
            torch.save(model.head, self._task_head_save_path(model_dir))

        def _task_head_save_path(self, model_dir: FileMetadata) -> str:
            return model_dir.file_in_dir(self.model_file_name)

        def prepare_model_for_train(self, **kwargs):
            self.model.train()
            # if self.hyperparams.frozen_base:
            #     for param in self.model._base_model.model.parameters():
            #         param.requires_grad = False
            # else:
            #     self.model._base_model.model.train()
            #     for param in self.model.base.parameters():
            #         param.requires_grad = True
            # self.model.head.train()

        def prepare_model_for_predict(self, **kwargs):
            self.model.eval()
            # self.model._base_model.model.eval()
            # self.model.head.eval()

        def prepare_input(
                self,
                batch: Dataset,
                **kwargs,
        ) -> Any:
            return self.prepare_input_fn(batch, **kwargs)

        def forward(self, input: Any, **kwargs) -> Union[Tensor, List[Tensor]]:
            output: Tensor = self.model.forward(input, **kwargs)
            return output


    class PyTorchClassifierMixin(Classifier, PyTorch, ABC):
        label_encoding_range = EncodingRange.ZERO_TO_N_MINUS_ONE

        def prepare_target(
                self,
                batch: ClassificationData,
                **kwargs,
        ) -> Tensor:
            target = batch.ground_truths().torch().squeeze()
            if len(target.shape) == 0:  ## We accidentally converted it into a Scalar.
                target: Tensor = target.unsqueeze(0)
            return target

        def prepare_predictions(self, output: Tensor, **kwargs) -> Dict[str, np.ndarray]:
            if not isinstance(output, Tensor) or not output.ndim == 2:
                raise ValueError(
                    f'The output of {self.hyperparams.base_model["name"]}.embed_single() should be a 2D tensor; '
                    f'found tensor of shape: {output.shape}'
                )
            scores: np.ndarray = softmax(output, dim=1).detach().cpu().numpy()
            return {'scores': scores, 'labels': self.encoded_labelspace}


    class PyTorchClassifier(PyTorchClassifierMixin, PyTorchTaskHead):
        model_file_name = "classification_head.pt"
        required_embeddings = "single"

        class Hyperparameters(PyTorchTaskHead.Hyperparameters):
            dropout: float = 0.1
            loss: Union[Loss, Dict, str] = 'CrossEntropyLoss'
            optimizer = dict(
                name='AdamW',
                lr=5e-5,
                weight_decay=1e-7,
                eps=1e-8,
            )

        class ClassificationHead(TorchModule):
            def __init__(
                    self,
                    hyperparams: 'PyTorchClassifier.Hyperparameters',
                    embedding_size: int,
                    num_labels: int,
                    labelspace: Tuple[str, ...],
            ):
                super(self.__class__, self).__init__()
                self.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(hyperparams.dropout),
                    torch.nn.Linear(
                        in_features=embedding_size,
                        out_features=num_labels,
                    )
                )
                self.num_labels: int = num_labels
                self.labelspace: Tuple[str, ...] = labelspace

            def forward(self, input: Tensor, **kwargs) -> Tensor:
                return self.classifier(input)

        def _create_head(self, base_model: PyTorchBaseModel) -> TorchModule:
            return self.ClassificationHead(
                hyperparams=self.hyperparams,
                embedding_size=base_model.embedding_size(),
                num_labels=self.num_labels,
                labelspace=self.labelspace,
            )


    class PyTorchMultiLabelClassifierMixin(MultiLabelClassifier, PyTorch, ABC):
        label_encoding_range = EncodingRange.ZERO_TO_N_MINUS_ONE
        output_dtype = torch.float32
        target_dtype = torch.float32

        def prepare_target(
                self,
                batch: MultiLabelClassificationData,
                **kwargs,
        ) -> Tensor:
            return self._encode_multi_hot(batch)

        def _encode_multi_hot(self, batch: MultiLabelClassificationData) -> Tensor:
            labels = batch.ground_truths()
            batch_size: int = len(labels)
            multi_hot_arr: np.ndarray = np.zeros((batch_size, self.num_labels), dtype=np.int32)
            for i, lb_list in enumerate(labels):
                multi_hot_arr[i, lb_list] = 1
            return torch.from_numpy(multi_hot_arr)

        def prepare_predictions(self, output: Tensor, **kwargs) -> Dict[str, np.ndarray]:
            if not isinstance(output, Tensor) or not output.ndim == 2:
                raise ValueError(
                    f'The output of {self.hyperparams.base_model["name"]}.embed_single() should be a 2D tensor; '
                    f'found tensor of shape: {output.shape}'
                )
            scores: np.ndarray = torch.sigmoid(output).detach().cpu().numpy()
            return {'scores': scores, 'labels': self.encoded_labelspace}


    class PyTorchMultiLabelClassifier(PyTorchMultiLabelClassifierMixin, PyTorchClassifier):
        class Hyperparameters(PyTorchClassifier.Hyperparameters):
            loss: Union[Loss, Dict, str] = 'BCEWithLogitsLoss'


    class PyTorchRegressorMixin(Regressor, PyTorch, ABC):
        def prepare_target(
                self,
                batch: Dataset,
                **kwargs,
        ) -> Tensor:
            target: Tensor = batch.ground_truths().torch().squeeze()
            if len(target.shape) == 0:  ## We accidentally converted it into a Scalar.
                target: Tensor = target.unsqueeze(0)
            return target

        def prepare_predictions(self, output: Tensor, **kwargs) -> np.ndarray:
            if not isinstance(output, Tensor) or not output.ndim == 1:
                raise ValueError(
                    f'The output of {self.hyperparams.base_model["name"]}.embed_single() should be a 1D tensor; '
                    f'found tensor of shape: {output.shape}'
                )
            scores: np.ndarray = torch.squeeze(output, dim=1).detach().cpu().numpy()
            return scores


    class PyTorchRegressor(PyTorchRegressorMixin, PyTorchTaskHead):
        output_dtype = torch.float32
        target_dtype = torch.float32
        model_file_name = "regression_head.pt"
        required_embeddings = "single"

        class Hyperparameters(PyTorchTaskHead.Hyperparameters):
            dropout: float = 0.1
            loss = 'MSELoss'
            optimizer = dict(
                name='AdamW',
                lr=5e-5,
                weight_decay=1e-7,
                eps=1e-8,
            )

        class RegressionHead(TorchModule):
            def __init__(
                    self,
                    hyperparams: 'PyTorchRegressor.Hyperparameters',
                    embedding_size: int,
            ):
                super(self.__class__, self).__init__()
                self.regressor = torch.nn.Sequential(
                    torch.nn.Dropout(hyperparams.dropout),
                    torch.nn.Linear(
                        in_features=embedding_size,
                        out_features=1,
                    )
                )

            def forward(self, input: Tensor, **kwargs) -> Tensor:
                return self.regressor(input)

        def _create_head(self, base_model: PyTorchBaseModel) -> TorchModule:
            return self.RegressionHead(
                hyperparams=self.hyperparams,
                embedding_size=base_model.embedding_size(),
            )
