from typing import *
import os
from synthergent.base.util.string import StringUtil
from synthergent.base.util.language import Utility, get_default


class EnvUtil(Utility):
    ## KEY
    PROCESSING_JOB_NAME: ClassVar[str] = 'PROCESSING_JOB_NAME'
    SNS_TOPIC: ClassVar[str] = 'SNS_TOPIC_ARN'
    SNS_TOPIC_REGION: ClassVar[str] = 'SNS_TOPIC_REGION'
    DDB_TABLE_REGION: ClassVar[str] = 'DDB_TABLE_REGION'
    DDB_TABLE_NAME: ClassVar[str] = 'DDB_TABLE_NAME'
    LOG_LEVEL: ClassVar[str] = 'LOG_LEVEL'
    CHIME_WEBHOOK_URL: ClassVar[str] = 'CHIME_WEBHOOK_URL'
    CUDA_VISIBLE_DEVICES: ClassVar[str] = 'CUDA_VISIBLE_DEVICES'

    @classmethod
    def var_exists(cls, env_var_key) -> bool:
        env_var_key: str = StringUtil.assert_not_empty_and_strip(env_var_key)
        if os.environ.get(env_var_key) is not None:
            return True
        else:
            return False

    @classmethod
    def get_var(cls, env_var_key: str, check_cases: bool = True) -> Optional[str]:
        env_var_key: str = StringUtil.assert_not_empty_and_strip(env_var_key)
        if cls.var_exists(env_var_key):
            return os.environ.get(env_var_key)
        if check_cases and cls.var_exists(env_var_key.upper()):
            return os.environ.get(env_var_key.upper())
        if check_cases and cls.var_exists(env_var_key.lower()):
            return os.environ.get(env_var_key.lower())
        return None

    @classmethod
    def cuda_visible_devices(cls) -> List[int]:
        return [
            int(cuda_device_id)
            for cuda_device_id in get_default(cls.get_var(cls.CUDA_VISIBLE_DEVICES), '').split(',')
        ]

    @classmethod
    def num_gpus(cls, provider: str = 'cuda') -> int:
        if provider == 'cuda':
            if get_default(cls.get_var(cls.CUDA_VISIBLE_DEVICES), '') == '':
                return 0
            return len(cls.cuda_visible_devices())
        raise NotImplementedError(f'Unsupported GPU provider: "{provider}"')
