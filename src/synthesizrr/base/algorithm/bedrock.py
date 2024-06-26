from typing import *
from abc import abstractmethod, ABC
import os, time, logging, sys, shutil, numpy as np, pandas as pd, gc, warnings, json
from contextlib import contextmanager
from synthesizrr.base.util import optional_dependency, set_param_from_alias, Parameters, get_default, safe_validate_arguments, \
    accumulate, dispatch, dispatch_executor, any_are_none, format_exception_msg, any_item, retry, Log, remove_values, as_list, \
    stop_executor
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


    def call_claude_v1_v2(
            bedrock,
            model_name: str,
            prompt: str,
            max_tokens_to_sample: int,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            stop_sequences: Optional[List[str]] = None,
            **kwargs,
    ) -> str:
        assert any_are_none(top_k, top_p), f'At least one of top_k, top_p must be None'
        bedrock_params = {
            "prompt": prompt,
            "max_tokens_to_sample": max_tokens_to_sample,
        }
        if top_p is not None and temperature is not None:
            raise ValueError(f'Cannot specify both top_p and temperature; at most one must be specified.')

        if top_k is not None:
            assert isinstance(top_k, int)
            bedrock_params["top_k"] = top_k
        elif temperature is not None:
            assert isinstance(temperature, (float, int)) and 0 <= temperature <= 1
            bedrock_params["temperature"] = temperature
        elif top_p is not None:
            assert isinstance(top_p, (float, int)) and 0 <= top_p <= 1
            bedrock_params["top_p"] = top_p

        if stop_sequences is not None:
            bedrock_params["stop_sequences"] = stop_sequences

        response = bedrock.invoke_model(
            body=json.dumps(bedrock_params),
            modelId=model_name,
            accept='application/json',
            contentType='application/json',
        )
        response_body: Dict = json.loads(response.get('body').read())
        return response_body.get('completion')


    def call_claude_v3(
            bedrock,
            *,
            model_name: str,
            prompt: str,
            max_tokens_to_sample: int,
            temperature: Optional[float] = None,
            system: Optional[str] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            stop_sequences: Optional[List[str]] = None,
            **kwargs,
    ) -> str:
        assert any_are_none(top_k, top_p), f'At least one of top_k, top_p must be None'
        bedrock_params = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens_to_sample,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }
        if system is not None:
            assert isinstance(system, str) and len(system) > 0
            bedrock_params["system"] = system

        if top_p is not None and temperature is not None:
            raise ValueError(f'Cannot specify both top_p and temperature; at most one must be specified.')

        if top_k is not None:
            assert isinstance(top_k, int) and len(system) >= 1
            bedrock_params["top_k"] = top_k
        elif top_p is not None:
            assert isinstance(top_p, (float, int)) and 0 <= top_p <= 1
            bedrock_params["top_p"] = top_p
        elif temperature is not None:
            assert isinstance(temperature, (float, int)) and 0 <= temperature <= 1
            bedrock_params["temperature"] = temperature

        if stop_sequences is not None:
            bedrock_params["stop_sequences"] = stop_sequences

        bedrock_params_json: str = json.dumps(bedrock_params)
        # print(f'\n\nbedrock_params_json:\n{json.dumps(bedrock_params, indent=4)}')
        response = bedrock.invoke_model(
            body=bedrock_params_json,
            modelId=model_name,
            accept='application/json',
            contentType='application/json',
        )
        response_body: Dict = json.loads(response.get('body').read())
        return '\n'.join([d['text'] for d in response_body.get("content")])


    def call_bedrock(
            prompt: str,
            *,
            model_name: str,
            generation_params: Dict,
            region_name: List[str],
    ) -> str:
        ## Note: creation of the bedrock client is fast.
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=any_item(region_name),
            # endpoint_url='https://bedrock.us-east-1.amazonaws.com',
        )
        if 'anthropic.claude-3' in model_name:
            generated_text: str = call_claude_v3(
                bedrock=bedrock,
                prompt=prompt,
                model_name=model_name,
                **generation_params
            )
        elif 'claude' in model_name:
            generated_text: str = call_claude_v1_v2(
                bedrock=bedrock,
                prompt=prompt,
                model_name=model_name,
                **generation_params
            )
        else:
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
            generated_text: str = response_body.get('completion')
        return generated_text


    class BedrockPrompter(GenerativeLM):
        aliases = ['bedrock']
        executor: Optional[Any] = None

        class Hyperparameters(GenerativeLM.Hyperparameters):
            ALLOWED_TEXT_GENERATION_PARAMS: ClassVar[List[str]] = [
                'strategy',
                'name',
                'temperature',
                'top_k',
                'top_p',
                'max_new_tokens',
                'stop_sequences',
                'system',
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
            retry_jitter: confloat(ge=0) = 0.5
            parallelize: Parallelize = Parallelize.sync
            max_workers: int = 1
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
            if self.executor is None:
                self.executor: Optional[Any] = dispatch_executor(
                    parallelize=self.hyperparams.parallelize,
                    max_workers=self.hyperparams.max_workers,
                )

        def cleanup(self):
            super(self.__class__, self).cleanup()
            stop_executor(self.executor)

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
                    executor=self.executor,
                )
                generated_texts.append(generated_text)
            generated_texts: List[str] = accumulate(generated_texts)
            return {
                GENERATED_TEXTS_COL: generated_texts
            }
