from typing import *
from abc import abstractmethod, ABC
import os, time, logging, sys, shutil, numpy as np, pandas as pd, gc, warnings
from math import exp, log, inf
from contextlib import contextmanager
from synthesizrr.base.util import optional_dependency, set_param_from_alias, Parameters, get_default, safe_validate_arguments, \
    ignore_warnings, as_list, format_exception_msg
from synthesizrr.base.data import FileMetadata
from synthesizrr.base.data import ScalableDataFrame, ScalableSeries
from synthesizrr.base.constants import MLType, Storage, Alias
from collections import OrderedDict
from pydantic import root_validator, Extra
from pydantic.typing import Literal

with optional_dependency('torch', 'sentencepiece', 'transformers', 'tokenizers', 'huggingface_hub'):
    import torch
    from torch import Tensor
    from torch.nn.functional import softmax, sigmoid
    from synthesizrr.base.framework import Dataset, Classifier, EncodingRange
    from synthesizrr.base.framework.dl.torch import PyTorch, Loss, Optimizer, LRScheduler, PyTorchBaseModel, PyTorchClassifierMixin, \
        PyTorchMultiLabelClassifierMixin
    from transformers import AutoModel, AutoTokenizer, AutoConfig, \
        PreTrainedModel, PreTrainedTokenizerBase, PretrainedConfig, \
        AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForSeq2SeqLM, \
        BatchEncoding
    from transformers.models.auto.modeling_auto import _BaseAutoModelClass, MODEL_MAPPING_NAMES, \
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES, \
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
    from transformers.generation.utils import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput
    from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput
    import huggingface_hub
    from transformers.utils.logging import is_progress_bar_enabled, enable_progress_bar, disable_progress_bar
    from synthesizrr.base.framework.task.text_generation import GenerativeLM, Prompts, NextTokens, GenerationOutputScoresFormat, \
        TextGenerationParams, TextGenerationParamsMapper, \
        TEXT_PROMPT_COL, GENERATED_TEXTS_COL


    def mapping_to_auto_model_classes(mapping_names: Union[List, Dict, OrderedDict]) -> Dict[str, str]:
        if isinstance(mapping_names, (dict, OrderedDict)):
            mapping_names: List = list(mapping_names.items())
        mapping_names: List[Tuple[str, Union[Tuple, str]]] = list(mapping_names)
        return {
            k: as_list(v)[0]
            for k, v in mapping_names
        }


    @contextmanager
    def disable_hf_logging():
        should_reenable_progress_bar: bool = is_progress_bar_enabled()
        disable_progress_bar()
        with ignore_warnings():
            yield
        if should_reenable_progress_bar:
            enable_progress_bar()


    class HFPyTorchModel(PyTorch, ABC):
        aliases = ['huggingface']

        AutoModelClass: ClassVar[Type[_BaseAutoModelClass]]
        available_auto_model_classes: ClassVar[Dict[str, str]]
        cache_dir: Optional[Union[FileMetadata, Dict, str]] = None
        init_empty: bool = False

        @root_validator(pre=True)
        def set_aliases(cls, params: Dict) -> Dict:
            Alias.set_cache_dir(params)
            if params.get('cache_dir') is not None:
                params['cache_dir']: FileMetadata = FileMetadata.of(params['cache_dir'])
            return params

        def create_model(
                self,
                model_dir: Optional[FileMetadata] = None,
        ) -> PreTrainedModel:
            cache_dir: Optional[str] = self.cache_dir.path if self.cache_dir is not None else None
            if self.init_empty is True:
                ## Note: Loading a model from its configuration file (i.e. using "from_config") does not load the
                ## model weights. It only affects the model’s configuration. Use from_pretrained() to load the model
                ## weights.
                ## Ref: https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModel.from_config
                config: PretrainedConfig = self.create_model_config()
                with disable_hf_logging():
                    return self.AutoModelClass.from_config(
                        config,
                        trust_remote_code=True,
                    )
            elif model_dir is None:
                config: PretrainedConfig = self.create_model_config()
                with disable_hf_logging():
                    return self.AutoModelClass.from_pretrained(
                        config.name_or_path,
                        config=config,
                        cache_dir=cache_dir,
                        trust_remote_code=True,
                        device_map=self.hyperparams.device_map,
                    )
            else:
                # print(
                #     f"Loading tokenizer from: '{model_dir.path}' using params: "
                #     f"{self.hyperparams.model_config}"
                # )
                with disable_hf_logging():
                    return self.AutoModelClass.from_pretrained(
                        model_dir.path,
                        device_map=self.hyperparams.device_map,
                        **{
                            **self.hyperparams.model_config,
                            **dict(
                                cache_dir=cache_dir,
                                trust_remote_code=True,
                            ),
                        },
                    )

        def create_model_config(self) -> PretrainedConfig:
            return AutoConfig.from_pretrained(
                self.hyperparams.model_name,
                **{
                    **self.hyperparams.model_config,
                    **dict(
                        cache_dir=self.cache_dir.path if self.cache_dir is not None else None,
                        trust_remote_code=True,
                    ),
                },
            )

        @safe_validate_arguments
        def download_model_to_cache_dir(
                self,
                cache_dir: Optional[Union[FileMetadata, Dict, str]] = None,
                download_strategy: Literal['load_model', 'snapshot_download', 'none'] = 'snapshot_download',
                **kwargs
        ) -> FileMetadata:
            ## Download model if not already downloaded.
            cache_dir: FileMetadata = FileMetadata.of(get_default(cache_dir, self.cache_dir))
            # print(f'{self.class_name} using cache dir: {cache_dir}')

            with disable_hf_logging():
                if download_strategy == 'snapshot_download':
                    huggingface_hub.snapshot_download(
                        repo_id=self.hyperparams.model_name,
                        cache_dir=cache_dir.path,
                        ignore_patterns=[
                            ## Ignore HF-format files:
                            '*.safetensors', '*.safetensors.index.json',
                            ## Ignore Flax-format files:
                            '*.msgpack', '*.msgpack.index.json',
                            ## Ignore tensorflow-format files:
                            '*.h5', '*.h5.index.json',
                        ],
                    )
                    gc.collect()
                elif download_strategy == 'load_model':
                    hf_repos: Dict[str, Any] = {
                        repo.repo_id: repo
                        for repo in huggingface_hub.scan_cache_dir(cache_dir.path).repos
                    }
                    if self.hyperparams.model_name not in hf_repos:
                        model = self.AutoModelClass.from_pretrained(
                            self.hyperparams.model_name,
                            cache_dir=cache_dir.path,
                            trust_remote_code=True,
                        )
                        del model
                        gc.collect()
                elif download_strategy == 'none':
                    ## Do not download:
                    pass
                else:
                    raise NotImplementedError(f'Unsupported `download_strategy`: "{download_strategy}"')

            ## Pick the model dir from cache:
            exc = None
            for _ in range(3):
                try:
                    hf_repos: Dict[str, Any] = {
                        repo.repo_id: repo
                        for repo in huggingface_hub.scan_cache_dir(cache_dir.path).repos
                    }
                    model_cache_dir: str = str(
                        sorted(
                            [
                                (revision, revision.last_modified)
                                for revision in hf_repos[self.hyperparams.model_name].revisions
                            ],
                            key=lambda x: x[1],
                        )[-1][0].snapshot_path
                    )
                    return FileMetadata.of(model_cache_dir)
                except Exception as e:
                    exc = e
                    time.sleep(np.random.randint(1, 15))
            raise OSError(
                f'Failed to load HuggingFace model after 3 tries. '
                f'Error:\n{format_exception_msg(exc, short=False)}'
            )


    class HFTokenizerConfig(Parameters):
        class Config(Parameters.Config):
            extra = Extra.allow

        tokenizer_name: Optional[str] = None
        pad_token: Optional[str] = None
        truncation_side: Literal['left', 'right'] = 'right'  ## Keeps tokens at the start of the string

        @root_validator(pre=True)
        def set_aliases(cls, params: Dict) -> Dict:
            set_param_from_alias(params, param='tokenizer_name', alias=[
                'model_name', 'pretrained_model_name_or_path', 'model_name_or_path', 'tokenizer',
            ])
            return params


    class HFTokenizerEncode(Parameters):
        class Config(Parameters.Config):
            extra = Extra.allow

        max_length: Optional[int] = None
        padding: Literal['longest', 'max_length', 'do_not_pad'] = 'longest'  ## Same as padding="True"
        truncation: Literal['only_first', 'only_second',
                            'longest_first', 'do_not_truncate'] = 'longest_first'  ## Same as truncation="True"

        @root_validator(pre=True)
        def set_aliases(cls, params: Dict) -> Dict:
            set_param_from_alias(params, param='max_length', alias=[
                'max_len', 'max_sequence_length', 'max_sequence_len',
            ])
            set_param_from_alias(params, param='padding', alias=['pad'])
            return params


    class HFTokenizerDecode(Parameters):
        class Config(Parameters.Config):
            extra = Extra.allow

        skip_special_tokens: bool = True
        clean_up_tokenization_spaces: bool = True


    class HFPyTorchTextModel(HFPyTorchModel, ABC):
        AutoTokenizerClass: ClassVar[Type] = AutoTokenizer

        tokenizer: Optional[Any] = None
        token2id: Optional[Dict[str, int]] = None
        id2token: Optional[Dict[int, str]] = None
        id2token_np: Optional[Any] = None

        @classmethod
        def save_param_names(cls) -> Set[str]:
            return super(HFPyTorchTextModel, cls).save_param_names() - {
                ## Remove all params PyTorch-specific params which should not be saved in __model_params__.pkl file:
                'tokenizer', 'token2id', 'id2token', 'id2token_np',
            }

        class Hyperparameters(PyTorch.Hyperparameters):
            model_name: str
            tokenizer_config: HFTokenizerConfig = dict()
            tokenizer_encode: HFTokenizerEncode = dict()
            tokenizer_special_tokens: Optional[Dict] = None
            model_config: Dict[str, Any] = dict()
            device_map: Optional[Union[Dict, str]] = None
            optimizer: Optimizer = dict(
                name='AdamW',
                lr=5e-5,
                weight_decay=1e-7,
                eps=1e-8,
            )

            @root_validator(pre=True)
            def set_hf_text_model_params(cls, params: Dict) -> Dict:
                set_param_from_alias(params, param='model_name', alias=[
                    'model', 'pretrained_model_name_or_path', 'model_name_or_path',
                ])
                return params

        def initialize(self, model_dir: Optional[FileMetadata] = None):
            self.tokenizer: PreTrainedTokenizerBase = self.create_tokenizer(model_dir=model_dir)
            self.model: PreTrainedModel = self.create_model(model_dir=model_dir)
            if self.hyperparams.tokenizer_config.pad_token is not None:
                self.tokenizer.pad_token = self.hyperparams.tokenizer_config.pad_token
            if self.hyperparams.tokenizer_special_tokens is not None:
                self.tokenizer.add_special_tokens(self.hyperparams.tokenizer_special_tokens)

            self.token2id: Dict[str, int] = self.tokenizer.get_vocab()
            self.id2token: Dict[int, str] = {tok_id: tok for tok, tok_id in self.token2id.items()}
            max_vocab_value: int = max(self.tokenizer.get_vocab().values())
            id2token_np: List[str] = ['' for _ in range(max_vocab_value + 1)]
            for tok, tok_id in self.tokenizer.get_vocab().items():
                id2token_np[tok_id] = tok
            id2token_np: np.ndarray = np.array(id2token_np)
            self.id2token_np = id2token_np

        def create_tokenizer(self, model_dir: Optional[FileMetadata] = None) -> PreTrainedTokenizerBase:
            if model_dir is None:
                return self.AutoTokenizerClass.from_pretrained(
                    get_default(
                        ## Allow overriding the tokenizer:
                        self.hyperparams.tokenizer_config.tokenizer_name,
                        self.hyperparams.model_name,
                    ),
                    **self.hyperparams.tokenizer_config.dict(exclude={
                        'pad_token', 'cache_dir', 'tokenizer_name',
                    }),
                    cache_dir=self.cache_dir.path if self.cache_dir is not None else None,
                    trust_remote_code=True,
                )
            else:
                # print(
                #     f"Loading tokenizer from: '{model_dir.path}' using params: "
                #     f"{self.hyperparams.tokenizer_config.dict(exclude={'pad_token', 'cache_dir', 'tokenizer_name'})}"
                # )
                return self.AutoTokenizerClass.from_pretrained(
                    model_dir.path,
                    **self.hyperparams.tokenizer_config.dict(exclude={
                        'pad_token', 'cache_dir', 'tokenizer_name',
                    }),
                    trust_remote_code=True,
                )

        def save(self, model_dir: FileMetadata):
            model_dir.mkdir()
            self.tokenizer.save_pretrained(model_dir.path)
            self.model.save_pretrained(
                model_dir.path,
                # max_shard_size='9GiB',
            )

        def prepare_input(
                self,
                batch: Dataset,
                **kwargs,
        ) -> Dict[str, Tensor]:
            sentences: List[str] = batch.features(MLType.TEXT, return_series=True).fillna('').to_list()
            input: BatchEncoding = self.tokenizer(
                sentences,
                **{
                    **self.hyperparams.tokenizer_encode.dict(),
                    **dict(return_tensors='pt'),
                }
            )
            return dict(input)


    class HFPyTorchTextEmbedder(HFPyTorchTextModel, PyTorchBaseModel):
        aliases = ['huggingface', 'huggingface-TextEmbedder']
        AutoModelClass = AutoModel
        available_auto_model_classes = mapping_to_auto_model_classes(MODEL_MAPPING_NAMES)

        _embedding_size: Optional[int] = None

        class Hyperparameters(HFPyTorchTextModel.Hyperparameters):
            loss: Union[Loss, Dict, str] = 'CrossEntropyLoss'
            pooling: Literal['mean'] = 'mean'

        def embed_single(self, input: Dict[str, Tensor], **kwargs) -> Tensor:
            # print(f'input: {input}')
            with torch.no_grad():
                model_output = self.model(**input)
            if self.hyperparams.pooling == 'mean':
                return self.mean_pooling(model_output, input['attention_mask'])
            raise NotImplementedError(f'Unsupported pooling type: "{self.hyperparams.pooling}"')

        def embed_multi(self, input: Dict[str, Tensor], **kwargs) -> Tensor:
            with torch.no_grad():
                model_output = self.model(**input)
            return model_output[0]  ## model_output[0] is a Tensor of shape (batch_size, num_tokens, embedding_size)

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

        # @staticmethod
        # def mean_pooling(model_output, attention_mask) -> Tensor:
        #     # print(f'model_output: {model_output}')
        #     ## Ref: https://www.sbert.net/examples/applications/computing-embeddings/README.html#sentence-embeddings-with-transformers
        #     ## Mean Pooling - Take attention mask into account for correct averaging
        #     token_embeddings = model_output[0]  ## First element of model_output contains all token embeddings
        #     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        #     sum_embeddings: Tensor = torch.sum(token_embeddings * input_mask_expanded, 1)
        #     sum_mask: Tensor = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        #     return sum_embeddings / sum_mask

        @staticmethod
        def mean_pooling(model_output, attention_mask) -> Tensor:
            token_embeddings = model_output[0]  ## First element of model_output contains all token embeddings
            token_embeddings = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            return sentence_embeddings


    class HFPyTorchSequenceClassifier(HFPyTorchTextModel, PyTorchClassifierMixin):
        aliases = ['huggingface', 'huggingface-SequenceClassification']
        AutoModelClass = AutoModelForSequenceClassification
        available_auto_model_classes = mapping_to_auto_model_classes(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES)

        class Hyperparameters(HFPyTorchTextModel.Hyperparameters):
            loss: Union[Loss, Dict, str] = 'CrossEntropyLoss'

        def create_model_config(self) -> PretrainedConfig:
            return AutoConfig.from_pretrained(
                self.hyperparams.model_name,
                **{
                    **self.hyperparams.model_config,
                    **dict(
                        num_labels=self.num_labels,
                        cache_dir=self.cache_dir.path if self.cache_dir is not None else None,
                        trust_remote_code=True,
                    ),
                },
            )

        def forward(self, input: Dict, **kwargs) -> Any:
            ## Feed the input_ids and masks to the model:
            output: SequenceClassifierOutput = self.model(**input)
            # ## obtaining the last layer hidden states of the Transformer
            # last_hidden_state = output.last_hidden_state  ## shape: (batch_size, seq_length, bert_hidden_dim)
            # # As I said, the CLS token is in the beginning of the sequence. So, we grab its representation
            # # by indexing the tensor containing the hidden representations
            # CLS_token_state = last_hidden_state[:, 0, :]
            # # passing this representation through our custom head
            # logits = self.head(CLS_token_state)
            # return logits
            return output.logits


    class HFPyTorchMultiLabelSequenceClassifier(HFPyTorchTextModel, PyTorchMultiLabelClassifierMixin):
        aliases = ['huggingface', 'huggingface-MultiLabelSequenceClassification']
        AutoModelClass = AutoModelForSequenceClassification
        available_auto_model_classes = mapping_to_auto_model_classes(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES)

        class Hyperparameters(HFPyTorchTextModel.Hyperparameters):
            loss: Union[Loss, Dict, str] = 'BCEWithLogitsLoss'

        def create_model_config(self) -> PretrainedConfig:
            return AutoConfig.from_pretrained(
                self.hyperparams.model_name,
                **{
                    **self.hyperparams.model_config,
                    **dict(
                        num_labels=self.num_labels,
                        problem_type="multi_label_classification",
                        cache_dir=self.cache_dir.path if self.cache_dir is not None else None,
                        trust_remote_code=True,
                    ),
                },
            )

        def forward(self, input: Dict, **kwargs) -> Any:
            ## Feed the input_ids and masks to the model:
            output: SequenceClassifierOutput = self.model(**input)
            # ## obtaining the last layer hidden states of the Transformer
            # last_hidden_state = output.last_hidden_state  ## shape: (batch_size, seq_length, bert_hidden_dim)
            # # As I said, the CLS token is in the beginning of the sequence. So, we grab its representation
            # # by indexing the tensor containing the hidden representations
            # CLS_token_state = last_hidden_state[:, 0, :]
            # # passing this representation through our custom head
            # logits = self.head(CLS_token_state)
            # return logits
            return output.logits


    class HFGenerativeLMTokenizerConfig(HFTokenizerConfig):
        truncation_side: Literal['left', 'right'] = 'left'  ## Keeps tokens at the end of the string, useful for LLMs


    class HFPyTorchGenerativeLMMixin(GenerativeLM, HFPyTorchTextModel, ABC):
        class Hyperparameters(HFPyTorchTextModel.Hyperparameters):
            prompt_prefix: str = ''
            generation_params: Union[TextGenerationParams, Dict, str]

            tokenizer_config: HFGenerativeLMTokenizerConfig = dict(padding_side='left', truncation_side='left')
            tokenizer_decode: HFTokenizerDecode = dict()

            @root_validator(pre=True)
            def set_generative_lm_params(cls, params: Dict) -> Dict:
                set_param_from_alias(params, param='generation_params', alias=[
                    'text_generation_params',
                    'generation', 'text_generation',
                    'generation_strategy', 'text_generation_strategy',
                ])
                params['generation_params'] = TextGenerationParamsMapper.of(params['generation_params']).initialize()
                return params

        @property
        def max_num_generated_tokens(self) -> int:
            return self.hyperparams.generation_params.max_new_tokens

        def _task_preprocess(self, batch: Prompts, **kwargs) -> Prompts:
            batch: Prompts = super(HFPyTorchGenerativeLMMixin, self)._task_preprocess(
                batch,
                prompt_prefix=self.hyperparams.prompt_prefix,
            )
            return batch

        def forward(self, input: Dict, **kwargs) -> Dict:
            ## Feed the input_ids and masks to the model:
            input.pop('token_type_ids', None)
            with disable_hf_logging():
                gen_kwargs: Dict = {
                    **input,
                    **self.hyperparams.generation_params.hf_dict(),
                    **dict(return_dict_in_generate=True),  ## Always return a *DecoderOnlyOutput
                }
                out: Union[GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput] = self.model.generate(**gen_kwargs)
            return dict(out)

        def prepare_predictions(self, output: Dict, input: Dict, **kwargs) -> Any:
            input_ids: Tensor = input['input_ids']
            generated_sequences: Tensor = output['sequences']
            num_input_tokens: int = input_ids.shape[1]
            num_generated_tokens: int = generated_sequences.shape[1]
            if num_generated_tokens <= num_input_tokens:
                ## Tokenized generated sequences has fewer token timesteps than the input,
                ## so it probably does not include the input. Thus we do nothing.
                pass
            elif not bool((generated_sequences[:, 0:num_input_tokens].cpu() == input_ids.cpu()).all()):
                ## The generated sequences does not have the input ids as a prefix. So it does not include the input:
                ## so it probably does not include the input. Thus we do nothing.
                pass
            else:
                generated_sequences: Tensor = generated_sequences[:, num_input_tokens:]
                num_generated_tokens: int = generated_sequences.shape[1]
            generated_texts: List[str] = self.tokenizer.batch_decode(
                generated_sequences,
                **self.hyperparams.tokenizer_decode.dict(),
            )
            predictions: Dict = {
                GENERATED_TEXTS_COL: generated_texts
            }
            if self.hyperparams.generation_params.output_scores:
                ## We get a tensor of size (B, vocab_size) for each timestep
                token_probs: Tuple[Tensor, ...] = output['scores']
                token_probs: Tuple[Tensor, ...] = token_probs[:num_generated_tokens]
                predictions['generated_token_ids'], \
                predictions['generated_tokens'], \
                predictions['generated_token_scores'] = self._convert_hf_scores_to_score_arrays(
                    token_probs,
                    batch_size=len(generated_sequences),
                    token2id=self.token2id,
                    id2token_np=self.id2token_np,
                    top_k_output_scores=self.hyperparams.generation_params.top_k_output_scores,
                    output_scores_format=self.hyperparams.generation_params.output_scores_format,
                    min_possible_score=self.hyperparams.generation_params.min_possible_score,
                    renormalize_logits=self.hyperparams.generation_params.renormalize_logits,
                    tokens_to_keep=self.hyperparams.generation_params.tokens_to_keep,
                    force_vocab_size=self.hyperparams.generation_params.force_vocab_size,
                    output_scores_tolerance=self.hyperparams.generation_params.output_scores_tolerance,
                )
            return predictions

        @staticmethod
        def _convert_hf_scores_to_score_arrays(
                token_probs: Tuple[Tensor, ...],
                batch_size: int,
                token2id: Dict[str, int],
                id2token_np: np.ndarray,
                top_k_output_scores: Optional[int],
                output_scores_format: GenerationOutputScoresFormat,
                min_possible_score: float,
                output_scores_tolerance: Optional[float],
                renormalize_logits: bool,
                tokens_to_keep: Optional[List[str]],
                force_vocab_size: bool,
        ):
            num_generated_tokens: int = len(token_probs)
            token_probs: np.ndarray = np.array([
                token_probs_t.detach().cpu().float().numpy()
                for token_probs_t in token_probs
            ])

            token_probs: np.ndarray = np.swapaxes(token_probs, 0, 1)
            ## `token_probs` now has shape (B, num_generated_tokens, vocab_size).
            if output_scores_format == 'probabilities':
                ## Convert from log-softmax-probs to softmax-probs.
                assert renormalize_logits is True
                token_probs: np.ndarray = np.exp(token_probs)
                assert min_possible_score == 0.0
                assert 0.0 < output_scores_tolerance < 1.0
            elif output_scores_format == 'log-probabilities':
                ## `token_probs` is already in log-softmax form.
                assert renormalize_logits is True
                assert min_possible_score == -inf
                assert output_scores_tolerance < 0.0
            elif output_scores_format == 'logits':
                assert renormalize_logits is False
                assert min_possible_score == -inf
                assert output_scores_tolerance is None ## Do not filter out any tokens.
            else:
                raise NotImplementedError(f'Unsupported `output_scores_format`: "{output_scores_format}"')
            ## Check dim 0 has same size as number of examples:
            assert token_probs.shape[0] == batch_size
            ## Check dim 1 has same size as number of generated tokens:
            assert token_probs.shape[1] == num_generated_tokens
            ## Check dim 2 has same size as vocab:
            if force_vocab_size:
                ## Needed for flan-t5 model: https://github.com/huggingface/transformers/issues/21734#issuecomment-1510111894
                token_probs: np.ndarray = token_probs[:, :, :len(id2token_np)]
            assert token_probs.shape[2] == len(id2token_np)

            if tokens_to_keep is not None:
                ## filtered_tokens are the ones we should keep. We should remove the rest, i.e. set them to zero.
                filtered_out_token_ids_np: np.ndarray = np.array([
                    token2id[tok] for tok, tok_id in token2id.items()
                    if tok not in tokens_to_keep
                ])
                token_probs[:, :, filtered_out_token_ids_np] = min_possible_score

            ## Sort tokens based on probabilities. Here we are sorting along the vocab dimension.
            token_scores_ascending: np.ndarray = np.sort(token_probs, axis=2)
            ## Sort tokens based on probabilities and return sorted vocab indexes (i.e. token ids). Here again we
            ## are sorting along the vocab dimension.
            token_ids_ascending: np.ndarray = token_probs.argsort(axis=2)

            if top_k_output_scores is not None:
                token_scores_ascending: np.ndarray = token_scores_ascending[:, :, -top_k_output_scores:]
                token_ids_ascending: np.ndarray = token_ids_ascending[:, :, -top_k_output_scores:]
            tokens_ascending: np.ndarray = id2token_np[token_ids_ascending]
            if output_scores_tolerance is not None:
                token_scores_ascending[token_scores_ascending < output_scores_tolerance] = min_possible_score
            return token_ids_ascending, tokens_ascending, token_scores_ascending


    class HFPyTorchCausalLM(HFPyTorchGenerativeLMMixin):
        aliases = ['huggingface', 'huggingface-CausalLM']
        AutoModelClass = AutoModelForCausalLM
        available_auto_model_classes = mapping_to_auto_model_classes(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)


    class HFPyTorchSeq2SeqLM(HFPyTorchGenerativeLMMixin):
        aliases = ['huggingface', 'huggingface-Seq2SeqLM', 'huggingface-ConditionalGeneration']
        AutoModelClass = AutoModelForSeq2SeqLM
        available_auto_model_classes = mapping_to_auto_model_classes(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES)
