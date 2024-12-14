import io
import pathlib
from typing import *
import time, json, math, numpy as np, pandas as pd, os, logging
from abc import abstractmethod, ABC
from functools import partial
from synthergent.base.data import FileMetadata
from synthergent.base.util import Parameters, MutableParameters, Registry, FractionalBool, safe_validate_arguments, StringUtil, \
    all_are_true, all_are_false, any_are_not_none, any_are_none, Log, as_list, get_default, as_set, \
    set_param_from_alias, random_sample, format_exception_msg, optional_dependency, FileSystemUtil, JupyterNotebook
from synthergent.base.constants import Alias
from pydantic import root_validator, Extra, conint, constr
from synthergent.base.constants import _LIBRARY_NAME

Tracker = "Tracker"

_TRACKERS_CACHE: Dict[Tuple[str, str, str, str], Any] = {}

DEFAULT_TRACKER_PARAMS: Dict = dict(tracker='noop')
with optional_dependency('aim'):
    import aim

    DEFAULT_TRACKER_PARAMS: Dict = dict(tracker='aim')


class Tracker(MutableParameters, Registry, ABC):
    _allow_multiple_subclasses: ClassVar[bool] = False  ## Rejects multiple subclasses registered to the same name.
    _allow_subclass_override: ClassVar[bool] = True  ## Allows replacement of subclass with same name.

    class Config(Parameters.Config):
        extra = Extra.ignore

    tracker_name: ClassVar[str]
    DEFAULT_PROJECTS_BASE_DIR: ClassVar[str] = FileSystemUtil.expand_dir(f'~/{_LIBRARY_NAME}/tracker/')
    DEFAULT_PROJECT: ClassVar[str] = 'default'
    DEFAULT_EXPERIMENT: ClassVar[str] = 'logs'

    silent: bool = False
    projects_base_dir: constr(min_length=1)
    project: constr(min_length=1)
    experiment: constr(min_length=1)

    @classmethod
    def of(
            cls,
            tracker: Optional[Union[Tracker, Dict, str]] = None,
            **kwargs,
    ) -> Tracker:
        if isinstance(tracker, Tracker):
            return tracker
        if isinstance(tracker, dict):
            return cls.of(**tracker, **kwargs)
        if tracker is not None:
            TrackerClass: Type[Tracker] = Tracker.get_subclass(tracker)
        elif 'name' in kwargs:
            TrackerClass: Type[Tracker] = Tracker.get_subclass(kwargs.pop('name'))
        else:
            TrackerClass: Type[Tracker] = cls
        if TrackerClass == Tracker:
            subclasses: List[str] = random_sample(
                as_list(Tracker.subclasses),
                n=3,
                replacement=False
            )
            raise ValueError(
                f'"{Tracker.class_name}" is an abstract class. '
                f'To create an instance, please either pass `tracker`, '
                f'or call .of(...) on a subclass of {Tracker.class_name}, e.g. {", ".join(subclasses)}'
            )

        tracker: Tracker = TrackerClass(**kwargs)
        tracker.initialize(**kwargs)
        return tracker

    @root_validator(pre=True)
    def tracker_params(cls, params: Dict):
        set_param_from_alias(params, param='projects_base_dir', alias=[
            'projects_dir', 'projects_base_folder', 'projects_folder',
        ], default=cls.DEFAULT_PROJECTS_BASE_DIR)
        set_param_from_alias(params, param='project', alias=[
            'project_name', 'project_path', 'repo', 'repo_name', 'repository', 'repository_name',
        ], default=cls.DEFAULT_PROJECT)
        set_param_from_alias(params, param='experiment', alias=[
            'experiment_name', 'trial', 'trial_name',
        ], default=cls.DEFAULT_EXPERIMENT)
        Alias.set_silent(params, default=False)

        params['projects_base_dir']: str = FileMetadata.of(
            FileSystemUtil.expand_dir(params['projects_base_dir'])
        ).mkdir(return_metadata=False)  ## Create the directory
        params['project']: str = cls._normalize_name(params['project'])
        params['experiment']: str = cls._normalize_name(params['experiment'])

        return params

    @abstractmethod
    def initialize(self, init_msg: bool = True, **kwargs):
        """
        Should initialize a new tracker.
        """
        pass

    @property
    @abstractmethod
    def id(self) -> str:
        pass

    @property
    @abstractmethod
    def log_dir(self) -> Optional[str]:
        pass

    @classmethod
    def _normalize_name(cls, experiment: str) -> str:
        return str(experiment) \
            .replace('  ', ' ').replace('  ', ' ').strip() \
            .replace(' ', '-').replace('_', '-').replace('--', '-').replace('--', '-') \
            .lower()

    def dict(self, *args, exclude: Optional[Any] = None, **kwargs) -> Dict:
        exclude: Set[str] = as_set(get_default(exclude, [])).union(as_set(self.dict_exclude))
        dict: Dict = super(Tracker, self).dict(*args, exclude=exclude, **kwargs)
        dict['tracker'] = self.tracker_name
        return dict

    def log(self, *data, level: int, silent: Optional[bool] = None, **kwargs):
        self._log_at_level(
            *data,
            sysout_fn=partial(Log.log, level=level),
            tracker_fn=partial(self._tracker_log, level=level),
            silent=silent,
            **kwargs,
        )

    @abstractmethod
    def _tracker_log(self, *data, level: int, **kwargs):
        pass

    def error(self, *data, **kwargs):
        self._log_at_level(*data, sysout_fn=Log.error, tracker_fn=self._tracker_error, **kwargs)

    @abstractmethod
    def _tracker_error(self, *data, **kwargs):
        pass

    def warn(self, *data, **kwargs):
        return self.warning(*data, **kwargs)

    def warning(self, *data, **kwargs):
        self._log_at_level(*data, sysout_fn=Log.warning, tracker_fn=self._tracker_warning, **kwargs)

    @abstractmethod
    def _tracker_warning(self, *data, **kwargs):
        pass

    def info(self, *data, **kwargs):
        self._log_at_level(*data, sysout_fn=Log.info, tracker_fn=self._tracker_info, **kwargs)

    @abstractmethod
    def _tracker_info(self, *data, **kwargs):
        pass

    def debug(self, *data, **kwargs):
        self._log_at_level(*data, sysout_fn=Log.debug, tracker_fn=self._tracker_debug, **kwargs)

    @abstractmethod
    def _tracker_debug(self, *data, **kwargs):
        pass

    def _log_at_level(
            self,
            *data,
            sysout_fn: Optional[Callable],
            tracker_fn: Optional[Callable],
            silent: Optional[bool] = None,
            **kwargs,
    ):
        ## Log to sysout:
        if (sysout_fn is not None) and (self.silent is not True) and (silent is not True):
            sysout_fn(*data, **kwargs)
        ## Log to tracker's data store:
        if tracker_fn is not None:
            tracker_fn(*data, **kwargs)

    @classmethod
    def _to_tracker_str(cls, *data, level: int, prefix: Optional[str] = None, **kwargs):
        if prefix is None:
            prefix: str = {
                logging.FATAL: '[FATAL] ',
                logging.ERROR: '[ERROR] ',
                logging.WARNING: '[ERROR] ',
                logging.WARN: '[WARN] ',
                logging.INFO: '',
                logging.DEBUG: '[DEBUG] ',
            }[level]
        data_str: str = prefix + ' '.join([Log.to_log_str(x) for x in data])
        return data_str

    @abstractmethod
    def tail(self, n: int = 10, return_logs: bool = False) -> Optional[List[Dict]]:
        pass

    @classmethod
    def default(cls) -> Tracker:
        return cls.of(DEFAULT_TRACKER_PARAMS)

    @classmethod
    def noop_tracker(cls) -> Tracker:
        return cls.of(tracker='noop')


class NoopTracker(Tracker):
    """Tracker which does nothing."""
    aliases = ['noop']

    tracker_name = 'noop'

    def initialize(self, init_msg: bool = False, **kwargs):
        if init_msg:
            self.info(f'Created sysout logger at: {StringUtil.now()}')

    @property
    def id(self) -> str:
        return 'None'

    @property
    def log_dir(self) -> Optional[str]:
        return None

    def _tracker_log(self, *data, level: int, **kwargs):
        pass

    def _tracker_error(self, *data, **kwargs):
        pass

    def _tracker_warning(self, *data, **kwargs):
        pass

    def _tracker_info(self, *data, **kwargs):
        pass

    def _tracker_debug(self, *data, **kwargs):
        pass

    def tail(self, n: int = 10, return_logs: bool = False) -> Optional[List[Dict]]:
        return None
