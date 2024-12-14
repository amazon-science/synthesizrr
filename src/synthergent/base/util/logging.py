from typing import *
from abc import ABC, abstractmethod

import pandas as pd

from synthergent.base.util.language import MutableParameters, safe_validate_arguments, binary_search
from synthergent.base.util.jupyter import JupyterNotebook
from synthergent.base.util.string import StringUtil
import logging, sys
from pydantic import constr, Field, FilePath, validator, conint
from pydantic.typing import Literal


class Log(MutableParameters):
    LOG_LEVEL_SUFFIX: ClassVar[str] = ''
    DEBUG: ClassVar[str] = 'DEBUG'
    INFO: ClassVar[str] = 'INFO'
    WARNING: ClassVar[str] = 'WARNING'
    ERROR: ClassVar[str] = 'ERROR'
    FATAL: ClassVar[str] = 'FATAL'

    LOG_LEVELS: ClassVar[Dict[str, int]] = {
        f'{DEBUG}{LOG_LEVEL_SUFFIX}': logging.DEBUG,
        f'{INFO}{LOG_LEVEL_SUFFIX}': logging.INFO,
        f'{WARNING}{LOG_LEVEL_SUFFIX}': logging.WARNING,
        f'{ERROR}{LOG_LEVEL_SUFFIX}': logging.ERROR,
        f'{FATAL}{LOG_LEVEL_SUFFIX}': logging.FATAL,
    }
    LOG_LEVELS_REVERSE: ClassVar[Dict[int, str]] = {
        logging.DEBUG: f'{DEBUG}{LOG_LEVEL_SUFFIX}',
        logging.INFO: f'{INFO}{LOG_LEVEL_SUFFIX}',
        logging.WARNING: f'{WARNING}{LOG_LEVEL_SUFFIX}',
        logging.ERROR: f'{ERROR}{LOG_LEVEL_SUFFIX}',
        logging.FATAL: f'{FATAL}{LOG_LEVEL_SUFFIX}',
    }
    ## Add new level names for our purposes to avoid getting logs from other libraries.
    for custom_log_level_name, custom_log_level in LOG_LEVELS.items():
        logging.addLevelName(level=custom_log_level, levelName=custom_log_level_name)

    LOG_LEVEL: Literal[
        f'{DEBUG}{LOG_LEVEL_SUFFIX}',
        f'{INFO}{LOG_LEVEL_SUFFIX}',
        f'{WARNING}{LOG_LEVEL_SUFFIX}',
        f'{ERROR}{LOG_LEVEL_SUFFIX}',
        f'{FATAL}{LOG_LEVEL_SUFFIX}',
    ] = f'{INFO}{LOG_LEVEL_SUFFIX}'
    FILE_LOG_LEVEL: Literal[
        f'{DEBUG}{LOG_LEVEL_SUFFIX}',
        f'{INFO}{LOG_LEVEL_SUFFIX}',
        f'{WARNING}{LOG_LEVEL_SUFFIX}',
        f'{ERROR}{LOG_LEVEL_SUFFIX}',
        f'{FATAL}{LOG_LEVEL_SUFFIX}',
    ] = f'{DEBUG}{LOG_LEVEL_SUFFIX}'
    LOG_FILE_PATH: FilePath = None
    LOG_FILE_LOGGER: Optional[logging.Logger] = None
    IS_JUPYTER_NOTEBOOK: bool = JupyterNotebook.is_notebook()

    class Config(MutableParameters.Config):
        arbitrary_types_allowed = True

    @safe_validate_arguments
    def set_log_file(
            self,
            file_path: FilePath,
            actor_name: Optional[constr(min_length=1, max_length=64)] = None,
    ):
        if self.LOG_FILE_LOGGER is not None:
            raise RuntimeError(f'Cannot set log file multiple times; already logging to "{self.LOG_FILE_PATH}"')
        if actor_name is not None:
            formatter = logging.Formatter(
                f'[{actor_name} @ %(asctime)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S UTC%z'
            )
        else:
            formatter = logging.Formatter(
                f'[%(asctime)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S UTC%z'
            )
        root_logger: logging.Logger = logging.getLogger()  ## Gets root logger
        root_logger.handlers[:] = []  ## Removes all existing handlers
        file_handler: logging.Handler = logging.FileHandler(file_path, mode='a+')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.setLevel(self.LOG_LEVELS[f'{self.DEBUG}{self.LOG_LEVEL_SUFFIX}'])
        self.LOG_FILE_LOGGER = root_logger
        self.LOG_FILE_PATH = file_path

    @safe_validate_arguments
    def set_log_level(self, log_level: Literal[DEBUG, INFO, WARNING, ERROR, FATAL]):
        log_level: str = StringUtil.assert_not_empty_and_strip(log_level).upper() + self.LOG_LEVEL_SUFFIX
        self.LOG_LEVEL = log_level

    @safe_validate_arguments
    def set_file_log_level(self, log_level: Literal[DEBUG, INFO, WARNING, ERROR, FATAL]):
        log_level: str = StringUtil.assert_not_empty_and_strip(log_level).upper() + self.LOG_LEVEL_SUFFIX
        self.FILE_LOG_LEVEL = log_level

    def log(self, *data, level: Union[str, int, float], flush: bool = False, **kwargs):
        if isinstance(level, (int, float)):
            ## Translate to our log level:
            level: str = self.LOG_LEVELS_REVERSE[binary_search(
                list(self.LOG_LEVELS_REVERSE.keys()),
                target=level,
                return_tuple=True,
            )[0]]  ## E.g. level=23 returns (DEBUG=20, WARN=30), we should pick DEBUG (lower of the two).
        data_str: str = ' '.join([self.to_log_str(x) for x in data])
        ## print at the appropriate level:
        if self.LOG_LEVELS[self.LOG_LEVEL] <= self.LOG_LEVELS[level]:
            ## Logs to both stdout and file logger if setup:
            if self.IS_JUPYTER_NOTEBOOK:
                from IPython.display import display
                for x in data:
                    if isinstance(x, (pd.DataFrame, pd.Series)):
                        display(x)
                    else:
                        print(self.to_log_str(x), end='', flush=flush)
                print('', flush=flush)
            else:
                print(data_str, flush=flush)

        if self.LOG_FILE_LOGGER is not None \
                and self.LOG_LEVELS[self.FILE_LOG_LEVEL] <= self.LOG_LEVELS[level]:
            self.LOG_FILE_LOGGER.log(
                ## We log to file at debug level:
                level=self.LOG_LEVELS[f'{self.DEBUG}{self.LOG_LEVEL_SUFFIX}'],
                msg=data_str,
            )

    def debug(self, *data, **kwargs):
        self.log(*data, level=f'{self.DEBUG}{self.LOG_LEVEL_SUFFIX}', **kwargs)

    def info(self, *data, **kwargs):
        self.log(*data, level=f'{self.INFO}{self.LOG_LEVEL_SUFFIX}', **kwargs)

    def warning(self, *data, **kwargs):
        self.log(*data, level=f'{self.WARNING}{self.LOG_LEVEL_SUFFIX}', **kwargs)

    def error(self, *data, **kwargs):
        self.log(*data, level=f'{self.ERROR}{self.LOG_LEVEL_SUFFIX}', **kwargs)

    def fatal(self, *data, **kwargs):
        self.log(*data, level=f'{self.FATAL}{self.LOG_LEVEL_SUFFIX}', **kwargs)

    @classmethod
    def to_log_str(cls, data: Any, *, df_num_rows: conint(ge=1) = 10) -> str:
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            return '\n' + StringUtil.jsonify(data)
        if isinstance(data, (list, set, frozenset, tuple)):
            return '\n' + StringUtil.pretty(data, max_width=int(1e6))
        if isinstance(data, (pd.Series, pd.DataFrame)):
            if len(data) <= df_num_rows:
                return '\n' + str(data.to_markdown())
            return '\n' \
                   + str(data.head(df_num_rows // 2).to_markdown()) \
                   + f'\n...({len(data) - df_num_rows} more rows)...\n' \
                   + str(data.tail(df_num_rows // 2).to_markdown())
        return StringUtil.pretty(data, max_width=int(1e6))


Log: Log = Log()  ## Creates a singleton
