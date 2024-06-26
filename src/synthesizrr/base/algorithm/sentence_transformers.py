from typing import *
import os
from synthesizrr.base.util import optional_dependency, set_param_from_alias
from synthesizrr.base.data import FileMetadata
from synthesizrr.base.framework.task import EmbeddingData, Embedder
from synthesizrr.base.framework.dl.torch import PyTorchBaseModel
from synthesizrr.base.constants import Task, Storage, MLType
from pydantic import root_validator, conint, confloat

with optional_dependency('torch', 'transformers'):
    import torch
    from torch import Tensor
    from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerBase, PreTrainedModel, BatchEncoding

    ## Globally disable tokenizer parallelism. Instead, load and preprocess batches using a Processing queue (which is
    ## used by default in .stream())
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


    class SentenceTransformersBaseModel(PyTorchBaseModel):
        aliases = ['sentence_transformers']
        model: Optional[PreTrainedModel] = None
        tokenizer: Optional[PreTrainedTokenizerBase] = None
        _embedding_size: Optional[int] = None

        class Hyperparameters(PyTorchBaseModel.Hyperparameters):
            model_name: str
            max_length: int = 128
            padding: bool = True
            truncation: bool = True

            @root_validator(pre=True)
            def set_aliases(cls, params: Dict) -> Dict:
                set_param_from_alias(params, param='max_length', alias=[
                    'max_len', 'max_sequence_length', 'max_sequence_len',
                ])
                return params

        def initialize(self, model_dir: Optional[FileMetadata] = None):
            if model_dir is None:
                self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(self.hyperparams.model_name)
                self.model: PreTrainedModel = AutoModel.from_pretrained(self.hyperparams.model_name)
            else:
                assert model_dir.storage is Storage.LOCAL_FILE_SYSTEM, 'Can only load models from disk.'
                self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_dir.path)
                self.model: PreTrainedModel = AutoModel.from_pretrained(model_dir.path)

        def save(self, model_dir: FileMetadata):
            self.model.save_pretrained(model_dir.path)
            self.tokenizer.save_pretrained(model_dir.path)

        def embedding_size(self) -> int:
            if self._embedding_size is None:
                input: BatchEncoding = self.tokenizer(
                    ['test'],
                    truncation=True,
                    max_length=3,
                    return_tensors='pt',
                )
                with torch.no_grad():
                    model_output = self.model(**input)
                self._embedding_size: int = model_output[0].shape[2]
            return self._embedding_size

        def prepare_input(
                self,
                batch: EmbeddingData,
                **kwargs,
        ) -> Dict[str, Tensor]:
            sentences: List[str] = batch.features(MLType.TEXT, return_series=True).fillna('').to_list()
            input: BatchEncoding = self.tokenizer(
                sentences,
                padding=self.hyperparams.padding,
                truncation=self.hyperparams.truncation,
                max_length=self.hyperparams.max_length,
                return_tensors='pt'
            )
            return dict(input)

        def embed_single(self, input: Dict[str, Tensor], **kwargs) -> Tensor:
            # print(f'input: {input}')
            with torch.no_grad():
                model_output = self.model(**input)
            return self.mean_pooling(model_output, input['attention_mask'])

        def embed_multi(self, input: Dict[str, Tensor], **kwargs) -> Tensor:
            with torch.no_grad():
                model_output = self.model(**input)
            return model_output[0]  ## model_output[0] is a Tensor of shape (batch_size, num_tokens, embedding_size)

        @staticmethod
        def mean_pooling(model_output, attention_mask) -> Tensor:
            # print(f'model_output: {model_output}')
            ## Ref: https://www.sbert.net/examples/applications/computing-embeddings/README.html#sentence-embeddings-with-transformers
            ## Mean Pooling - Take attention mask into account for correct averaging
            token_embeddings = model_output[0]  ## First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings: Tensor = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask: Tensor = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
