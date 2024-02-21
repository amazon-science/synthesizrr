import io, threading
import os.path
from pathlib import Path
from typing import *
import time, json, math, numpy as np, pandas as pd, random
from datetime import datetime
from abc import abstractmethod, ABC
from synthesizrr.base.data import FileMetadata
from synthesizrr.base.framework.tracker.Tracker import Tracker
from synthesizrr.base.util import optional_dependency, only_item, safe_validate_arguments, StringUtil, \
    all_are_true, all_are_false, any_are_not_none, any_are_none, FileSystemUtil, as_list, get_default, \
    set_param_from_alias, Log, format_exception_msg, JupyterNotebook, unset, keep_keys, get_fn_args
from synthesizrr.base.constants import Storage, Alias
from collections import deque
from pydantic import root_validator, Extra, conint
import logging


class LogTrackerFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return StringUtil.readable_datetime(
            datetime.fromtimestamp(record.created),
            human=False,
            microsec=True,
            tz=True,
        )


_ALL_LOG_FILE_HANDLERS: Dict[str, Any] = {}


class LogFileTracker(Tracker):
    aliases = ['log', 'log_file', 'logger', 'file']
    tracker_name = 'log'

    log_file: Optional[FileMetadata] = None
    name: Optional[str] = None
    logger: Optional[logging.Logger] = None  ## File logger
    logger_handler: Optional[logging.Handler] = None
    dict_exclude = ('logger', 'logger_handler')

    def initialize(
            self,
            *,
            init_msg: bool = True,
            **kwargs
    ):
        log_fpath: str = Alias.get_log_file(kwargs)
        log_file: FileMetadata = FileMetadata.of(log_fpath)
        log_file.get_dir(return_metadata=True).mkdir()
        assert log_file.storage is Storage.LOCAL_FILE_SYSTEM
        assert not log_file.is_path_valid_dir()
        process_id: int = os.getpid()
        thread_id: int = threading.get_ident()
        self.log_file: FileMetadata = log_file
        if self.name is None:
            if JupyterNotebook.is_notebook():
                self.name = f'{JupyterNotebook.name(extension=True)} (pid={process_id})'
            else:
                self.name = f'(pid={process_id})'
        self.logger, self.logger_handler = self.create_logger(
            logger_name=self.name,
            log_file=self.log_file,
        )
        assert self.logger is not None
        assert self.logger_handler is not None

        ## Log initial message:
        if init_msg:
            init_msg: str = f'{self.name} logging to "{self.log_file.path}" started at: {StringUtil.now()}'
            div_len: int = len(init_msg) + 8
            upper_div: str = (f'▔' * div_len) + '\n' + (f'▔' * div_len)
            lower_div: str = (f'▁' * div_len) + '\n' + (f'▁' * div_len)
            init_msg: str = f'\n{upper_div}\n    {init_msg}    \n{lower_div}'
            self.info(init_msg)

    @staticmethod
    def create_logger(
            logger_name: str,
            log_file: FileMetadata,
    ) -> Tuple[logging.Logger, logging.FileHandler]:
        global _ALL_LOG_FILE_HANDLERS
        ## Create a logger instance
        logger: logging.Logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        if log_file.path not in _ALL_LOG_FILE_HANDLERS:
            # print(f'Creating logger handler for path: "{log_file.path}"', flush=True)
            file_handler: logging.FileHandler = logging.FileHandler(log_file.path)
            ## Set the log level for the file handler (if different from the logger level).
            file_handler.setLevel(logging.DEBUG)
            ## Disable printing to sysout, we will do that separately:
            file_handler.stream = None
            formatter = LogTrackerFormatter(f'[{logger_name} @ %(asctime)s] %(message)s')
            file_handler.setFormatter(formatter)
            _ALL_LOG_FILE_HANDLERS[log_file.path]: logging.FileHandler = file_handler
        file_handler: logging.FileHandler = _ALL_LOG_FILE_HANDLERS[log_file.path]
        logger.addHandler(file_handler)
        logger.propagate = False
        return logger, file_handler

    def _tracker_error(self, *data, **kwargs):
        self._tracker_log(*data, level=logging.ERROR, **kwargs)

    def _tracker_warning(self, *data, **kwargs):
        self._tracker_log(*data, level=logging.WARNING, **kwargs)

    def _tracker_info(self, *data, **kwargs):
        self._tracker_log(*data, level=logging.INFO, **kwargs)

    def _tracker_debug(self, *data, **kwargs):
        self._tracker_log(*data, level=logging.DEBUG, **kwargs)

    def _tracker_log(self, *data, level: int, **kwargs):
        data_str: str = self._to_tracker_str(*data, level=level, **kwargs)

        self.logger.log(
            msg=data_str,
            level=level,
            **keep_keys(kwargs, get_fn_args(self.logger._log)),
        )

    @safe_validate_arguments
    def tail(self, n: conint(ge=1) = 10, *, last_start: bool = True, return_logs: bool = False) -> List[str]:
        with io.open(self.log_file.path, 'r', encoding='utf-8') as file:
            log_lines: List[str] = []
            for line_i, line in enumerate(reversed(list(file))):
                log_lines.append(line.removesuffix('\n'))
                if last_start is True and len(log_lines) >= 2 and '▔▔' in log_lines[-1] and '▔▔' in log_lines[-2]:
                    break
                elif len(log_lines) == n:
                    break
            log_lines: List[str] = log_lines[::-1]
            if return_logs:
                return log_lines
            for log in log_lines:
                print(log)

    def __del__(self):
        if self.logger_handler is not None:
            self.logger_handler.close()
            if self.logger is not None:
                self.logger.removeHandler(self.logger_handler)
        unset(self, 'logger')
        unset(self, 'logger_handler')

    @property
    def id(self) -> str:
        return StringUtil.hash(str(self.log_file.path))

    @property
    def log_dir(self) -> Optional[str]:
        return str(self.log_file.get_dir())
