from typing import *
from abc import abstractmethod, ABC
import os, time, logging, sys, shutil, numpy as np, pandas as pd, gc, warnings, json
from contextlib import contextmanager
from synthesizrr.base.util import optional_dependency, set_param_from_alias, Parameters, get_default, safe_validate_arguments, \
    accumulate, dispatch, str_format_args, format_exception_msg, any_item, retry, Log, remove_values, as_list
from synthesizrr.base.framework import Dataset
from synthesizrr.base.framework.task.text_generation import GenerativeLM, Prompts, GENERATED_TEXTS_COL, TextGenerationParams, \
    TextGenerationParamsMapper
from synthesizrr.base.data import FileMetadata
from synthesizrr.base.data import ScalableDataFrame, ScalableSeries
from synthesizrr.base.constants import MLType, Parallelize
from collections import OrderedDict
from pydantic import root_validator, conint, confloat, constr

with optional_dependency('boto3'):
    import boto3


    def call_bedrock(
            prompt: str,
            *,
            model_name: str,
            generation_params: Dict,
            region_name: List[str],
    ) -> Dict:
        start = time.perf_counter()
        ## Note: creation of the bedrock client is fast.
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=any_item(region_name),
            # endpoint_url=f'https://bedrock.{region_name}.amazonaws.com',
        )
        bedrock_invoke_model_params = {
            "prompt": prompt,
            **generation_params
        }
        response = bedrock.invoke_model(
            body=json.dumps(bedrock_invoke_model_params),
            modelId=model_name,
            accept='application/json',
            contentType='application/json'
        )
        response_body = json.loads(response.get('body').read())
        end = time.perf_counter()
        time_taken_sec: float = end - start
        return response_body.get('completion')


    class BedrockPrompter(GenerativeLM):
        aliases = ['bedrock']

        class Hyperparameters(GenerativeLM.Hyperparameters):
            ALLOWED_TEXT_GENERATION_PARAMS: ClassVar[List[str]] = [
                'strategy',
                'name',
                'temperature',
                'top_k',
                'top_p',
                'max_new_tokens',
                'stop_sequences',
            ]

            region_name: List[str] = [
                'us-east-1',
                'us-west-2',
                'eu-central-1',
                'ap-northeast-1',
            ]
            model_name: constr(min_length=1)
            retries: conint(ge=0) = 3
            retry_wait: confloat(ge=0) = 1.0
            retry_jitter: confloat(ge=0) = 0.25
            parallelize: Parallelize = Parallelize.sync
            generation_params: Union[TextGenerationParams, Dict, str]

            @root_validator(pre=True)
            def set_bedrock_params(cls, params: Dict) -> Dict:
                set_param_from_alias(
                    params,
                    param='model_name',
                    alias=['model_id', 'modelId', 'model'],
                )
                set_param_from_alias(params, param='generation_params', alias=[
                    'text_generation_params',
                    'generation', 'text_generation',
                    'generation_strategy', 'text_generation_strategy',
                ])
                gen_params: Dict = params['generation_params']
                extra_gen_params: Set[str] = set(gen_params.keys()) - set(cls.ALLOWED_TEXT_GENERATION_PARAMS)
                if len(extra_gen_params) != 0:
                    raise ValueError(
                        f'Following extra parameters for text generation are not allowed: {list(extra_gen_params)}; '
                        f'allowed parameters: {cls.ALLOWED_TEXT_GENERATION_PARAMS}.'
                    )
                params['generation_params'] = TextGenerationParamsMapper.of(params['generation_params']).initialize()

                if params.get('region_name') is not None:
                    params['region_name']: List[str] = as_list(params['region_name'])
                return params

        @property
        def max_num_generated_tokens(self) -> int:
            return self.hyperparams.generation_params.max_new_tokens

        def initialize(self, model_dir: Optional[FileMetadata] = None):
            ## Ignore the model_dir.
            pass

        @property
        def bedrock_text_generation_params(self) -> Dict[str, Any]:
            ## https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html
            generation_params: TextGenerationParams = self.hyperparams.generation_params
            bedrock_params: Dict[str, Any] = {
                'max_tokens_to_sample': generation_params.max_new_tokens,
            }
            for param in remove_values(
                    self.hyperparams.ALLOWED_TEXT_GENERATION_PARAMS,
                    ['strategy', 'name', 'max_new_tokens'],
            ):
                if hasattr(generation_params, param) and getattr(generation_params, param) is not None:
                    bedrock_params[param] = getattr(generation_params, param)
            return bedrock_params

        def prompt_model_with_retries(self, prompt: str) -> str:
            try:
                return retry(
                    call_bedrock,
                    prompt=prompt,
                    region_name=self.hyperparams.region_name,
                    model_name=self.hyperparams.model_name,
                    generation_params=self.bedrock_text_generation_params,
                    retries=self.hyperparams.retries,
                    wait=self.hyperparams.retry_wait,
                    jitter=self.hyperparams.retry_jitter,
                    silent=True,
                )
            except Exception as e:
                Log.error(format_exception_msg(e))
                return ''

        def predict_step(self, batch: Prompts, **kwargs) -> Any:
            generated_texts: List = []
            for prompt in batch.prompts().tolist():  ## Template has already been applied
                generated_text: Any = dispatch(
                    self.prompt_model_with_retries,
                    prompt,
                    parallelize=self.hyperparams.parallelize,
                )
                generated_texts.append(generated_text)
            generated_texts: List[str] = accumulate(generated_texts)
            return {
                GENERATED_TEXTS_COL: generated_texts
            }
