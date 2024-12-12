from typing import *
import io, logging
from pathlib import Path
import time, json, math, numpy as np, pandas as pd, random
from datetime import datetime
from abc import abstractmethod, ABC
from synthergent.base.data import FileMetadata
from synthergent.base.framework.tracker.Tracker import Tracker
from synthergent.base.util import optional_dependency, only_item, safe_validate_arguments, StringUtil, \
    all_are_true, all_are_false, any_are_not_none, any_are_none, FileSystemUtil, as_list, get_default, \
    set_param_from_alias, Log, format_exception_msg, JupyterNotebook, get_fn_args, keep_keys
from collections import deque
from pydantic import root_validator, Extra, conint

with optional_dependency('aim'):
    import aim
    from aim import Run, Repo, Text as AimText
    from aim.sdk.errors import RepoIntegrityError


    class AimTracker(Tracker):
        aliases = ['aim', 'aimhub', 'aimstack']

        dict_exclude = ('repo', 'run')
        tracker_name = 'aim'

        repo: Optional[Repo] = None
        run: Optional[Run] = None
        experiment_base: Optional[str] = None

        @property
        def repo_dir(self) -> str:
            repo_dir: FileMetadata = FileMetadata.of(self.projects_base_dir) / self.tracker_name / self.project
            return str(repo_dir.path)

        @property
        def is_default(self) -> bool:
            return self.project == self.default_project and self.experiment == self.default_experiment

        @property
        def id(self) -> str:
            return str(self.run.hash)

        @property
        def log_dir(self) -> Optional[str]:
            return str(self.run.repo.path)

        def initialize(
                self,
                *,
                init_msg: bool = True,
                resume: bool = True,
                experiment_base: Optional[str] = None,
                capture_terminal_logs: Optional[bool] = None,
                **kwargs
        ):
            self.repo: Repo = Repo(self.repo_dir, init=True)
            assert isinstance(self.repo, Repo)

            notebook_name: str = get_default(JupyterNotebook.name(), 'kernel')
            kernel_start_time: str = StringUtil.kernel_start_time(human=True)
            now: str = StringUtil.now(human=True)

            self.experiment_base: str = get_default(
                experiment_base,
                f'{notebook_name}/{kernel_start_time}'
            )
            experiment: str = f'{self.experiment_base}/{self.experiment}'
            capture_terminal_logs: bool = get_default(
                capture_terminal_logs,
                True if self.is_default else False
            )

            run_hash: Optional[str] = None
            if resume:
                for _run in self.repo.iter_runs():
                    try:
                        if _run.experiment == experiment:
                            run_hash: str = _run.hash
                            break
                    except RepoIntegrityError as e:
                        pass
            self.run = Run(
                run_hash=run_hash,
                experiment=experiment,
                repo=self.repo,
                system_tracking_interval=None,
                log_system_params=False,
                capture_terminal_logs=capture_terminal_logs,  ## Allows logging with tracker.info('...'), etc.
            )
            ## Log initial message:
            if init_msg:
                if self.is_default:
                    init_msg: str = f'Automatic logging for {notebook_name}'
                    init_msg: str = f'{init_msg} started at: {kernel_start_time}'
                else:
                    init_msg: str = f'Tracking experiment "{self.experiment}"'
                    init_msg: str = f'{init_msg} started at: {now}'

                init_msg: str = f'{init_msg} (id={self.run.hash})'
                div_len: int = len(init_msg) + 8
                upper_div: str = (f'▔' * div_len) + '\n' + (f'▔' * div_len)
                lower_div: str = (f'▁' * div_len) + '\n' + (f'▁' * div_len)
                init_msg: str = f'\n{upper_div}\n    {init_msg}    \n{lower_div}'
                self.info(init_msg)

        def _tracker_log(self, *data, level: int, **kwargs):
            data_str: str = self._to_tracker_str(*data, level=level, **kwargs)
            self.run.track(
                value=AimText(data_str),
                **keep_keys(kwargs, get_fn_args(self.run.track))
            )

        def _tracker_error(self, *data, **kwargs):
            self.run.log_error(
                self._to_tracker_str(*data, level=logging.ERROR, **kwargs),
                **keep_keys(kwargs, get_fn_args(self.run.log_error))
            )

        def _tracker_warning(self, *data, **kwargs):
            self.run.log_warning(
                self._to_tracker_str(*data, level=logging.WARNING, **kwargs),
                **keep_keys(kwargs, get_fn_args(self.run.log_warning))
            )

        def _tracker_info(self, *data, **kwargs):
            self.run.log_info(
                self._to_tracker_str(*data, level=logging.INFO, **kwargs),
                **keep_keys(kwargs, get_fn_args(self.run.log_info))
            )

        def _tracker_debug(self, *data, **kwargs):
            self.run.log_debug(
                self._to_tracker_str(*data, level=logging.DEBUG, **kwargs),
                **keep_keys(kwargs, get_fn_args(self.run.log_debug))
            )

        def tail(self, n: int = 10, return_logs: bool = False) -> Optional[List[Dict]]:
            logs_queue = deque(maxlen=n)  ## Only keep last N elements in dequeue
            for log in self.run.get_log_records().values:
                logs_queue.append(log.json())
            if return_logs:
                return list(logs_queue)
            ## Otherwise,
            for log in logs_queue:
                print(
                    f'[{StringUtil.readable_datetime(datetime.fromtimestamp(log["timestamp"]), human=True)}]\n{log["message"]}'
                )

        def __del__(self):
            self.run.finalize()
            # stop_daemon(self._daemon_id)

    # for hparam, val in {'lr': 0.1, 'seed': 42}.items():
    #     trial[f'hyperparams.{hparam}'] = val
    # for epoch_num in range(1, 10):
    #     for batch_i in range(1000):
    #         trial.track(
    #             name='train/loss',
    #             # context={'subset': 'train'},
    #             value=1000 - random.randint(batch_i, batch_i + 10),
    #             step=batch_i,
    #             epoch=epoch_num,
    #         )
    #         trial.track(
    #             name='val/loss',
    #             # context={'subset':'validation'},
    #             value=1000 - 0.9 * random.randint(batch_i - 5, batch_i + 5),
    #             step=batch_i,
    #             epoch=epoch_num
    #         )
    # trial.finalize()
