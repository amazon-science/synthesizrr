from typing import *
from synthergent.base.util import Parameters
from synthergent.base.constants import Parallelize
from pydantic import Extra, confloat, conint


class ScalingConfig(Parameters):
    parallelize: Parallelize
    max_workers: Optional[conint(ge=1)] = None
    batch_size: Optional[int] = None
    partition_size: Optional[str] = None

    class Config(Parameters):
        extra = Extra.ignore
