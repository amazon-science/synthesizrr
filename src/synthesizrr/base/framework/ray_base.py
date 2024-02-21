from typing import *
import warnings
import time, traceback, gc, os, math
from functools import partial
import time, os, gc, random, numpy as np, pandas as pd
import ray
from ray import tune, air
from ray.runtime_env import RuntimeEnv as RayRuntimeEnv
from synthesizrr.base.constants import Alias
from synthesizrr.base.util import Parameters, UserEnteredParameters, StringUtil, safe_validate_arguments, accumulate, \
    ProgressBar, as_list
from pydantic import Extra, conint, confloat, constr, root_validator, validator


def max_num_resource_actors(
        model_num_resources: Union[conint(ge=0), confloat(ge=0.0, lt=1.0)],
        ray_num_resources: int,
) -> Union[int, float]:
    ## Returns number of models possible, restricted by a particular resource; takes into account
    ## fractional resource requirements.
    ## Note: all resource-requirements are either 0, a float between 0 and 1, or an integer above 1.
    if model_num_resources == 0:
        return math.inf
    elif 0 < model_num_resources < 1:
        ## E.g. when a model needs <1 GPU, multiple models can occupy the same GPU.
        max_num_models_per_resource: int = math.floor(1 / model_num_resources)
        return ray_num_resources * max_num_models_per_resource
    else:
        ## E.g. when a model needs >1 GPU, it must be the only model occupying that GPU.
        return math.floor(ray_num_resources / model_num_resources)


class RayInitConfig(UserEnteredParameters):
    class Config(UserEnteredParameters.Config):
        extra = Extra.allow

    ## Default values:
    address: str = 'auto'
    temp_dir: Optional[str] = None
    include_dashboard: bool = False
    runtime_env: RayRuntimeEnv = {}


@ray.remote
class RequestCounter:
    def __init__(self):
        self.pending_requests: int = 0
        self.last_started: float = -1
        self.last_completed: float = -1

    def started_request(self):
        self.pending_requests += 1
        self.last_started: time.time()

    def completed_request(self):
        self.pending_requests -= 1
        self.last_completed: time.time()

    def num_pending_requests(self) -> int:
        return self.pending_requests

    def last_started_timestamp(self) -> float:
        return self.last_started

    def last_completed_timestamp(self) -> float:
        return self.last_completed


ActorComposite = "ActorComposite"


class ActorComposite(Parameters):
    actor_id: str
    actor: Any
    request_counter: Any

    def kill(self):
        accumulate(ray.kill(self.actor))
        accumulate(ray.kill(self.request_counter))
        actor: ray.actor.ActorHandle = self.actor
        request_counter: ray.actor.ActorHandle = self.request_counter
        del actor
        del request_counter

    @classmethod
    def create_actors(
            cls,
            actor_factory: Callable,
            *,
            num_actors: int,
            request_counter_num_cpus: float = 0.1,
            request_counter_max_concurrency: int = 1000,
            **kwargs
    ) -> List[ActorComposite]:
        progress_bar: Optional[Dict] = Alias.get_progress_bar(kwargs)
        actors_progress_bar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=num_actors,
            desc=f'Creating Ray actors',
            unit='actors',
        )
        actor_ids: List[str] = as_list(StringUtil.random_name(num_actors))
        actor_composites: List[ActorComposite] = []
        for actor_i, actor_id in zip(range(num_actors), actor_ids):
            request_counter: ray.actor.ActorHandle = RequestCounter.options(
                num_cpus=request_counter_num_cpus,
                max_concurrency=request_counter_max_concurrency,
            ).remote()
            actor: ray.actor.ActorHandle = actor_factory(
                request_counter=request_counter,
                actor_i=actor_i,
                actor_id=actor_id,
            )
            actor_composites.append(
                ActorComposite(
                    actor_id=actor_id,
                    actor=actor,
                    request_counter=request_counter,
                )
            )
            actors_progress_bar.update(1)
            time.sleep(0.100)
        if len(actor_composites) != num_actors:
            msg: str = f'Creation of {num_actors - len(actor_composites)} actors failed'
            actors_progress_bar.failed(msg)
            raise ValueError(msg)
        else:
            msg: str = f'Created {num_actors} actors'
            actors_progress_bar.success(msg)
        return actor_composites
