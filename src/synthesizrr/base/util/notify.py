from typing import *
import requests, json
from abc import abstractmethod, ABC
from synthesizrr.base.util.language import Parameters, Registry, safe_validate_arguments, get_default, format_exception_msg
from synthesizrr.base.util.string import StringUtil
from synthesizrr.base.constants.DataProcessingConstants import Status
from pydantic import constr, BaseModel, root_validator
from datetime import datetime

Notifier = "Notifier"
NotifierSubclass = TypeVar("NotifierSubclass", bound="Notifier")


class Notifier(Parameters, Registry, ABC):
    name: constr(min_length=1)

    @root_validator(pre=True)
    def convert_params(cls, params: Dict):
        params['name'] = cls.class_name
        return params

    @abstractmethod
    def send(
            self,
            msg: Union[constr(min_length=1), int, float, BaseModel],
            **kwargs
    ) -> bool:
        pass

    @classmethod
    def of(
            cls,
            notifier: Optional[Union[Notifier, Dict, str]] = None,
            **kwargs,
    ) -> NotifierSubclass:
        if isinstance(notifier, Notifier):
            return notifier
        if notifier is None and 'name' in kwargs:
            notifier = kwargs.pop('name')
        if isinstance(notifier, dict):
            return cls.of(**notifier)
        if isinstance(notifier, str):
            if notifier is not None:
                NotifierClass: Type[Notifier] = Notifier.get_subclass(notifier)
            else:
                NotifierClass: Type[Notifier] = cls
            if NotifierClass == Notifier:
                raise ValueError(
                    f'"{Notifier.class_name}" is an abstract class. '
                    f'To create an instance, please either pass `notifier`, '
                    f'or call .of(...) on a subclass of "{Notifier.class_name}".'
                )
            try:
                return NotifierClass(**kwargs)
            except Exception as e:
                raise ValueError(f'Cannot create notifier with kwargs:\n{kwargs}\nError: {format_exception_msg(e)}')
        raise NotImplementedError(
            f'Unsupported value for `notifier`; '
            f'found {type(notifier)} with following value:\n{notifier}'
        )

    def pending(self, msg: Optional[str] = None, **kwargs) -> bool:
        msg: str = self._create_msg(status=Status.PENDING, msg=msg, **kwargs)
        return self.send(msg, **kwargs)

    def running(self, msg: Optional[str] = None, **kwargs) -> bool:
        msg: str = self._create_msg(status=Status.RUNNING, msg=msg, **kwargs)
        return self.send(msg, **kwargs)

    def success(self, msg: Optional[str] = None, **kwargs) -> bool:
        msg: str = self._create_msg(status=Status.SUCCEEDED, msg=msg, **kwargs)
        return self.send(msg, **kwargs)

    def failed(self, msg: Optional[str] = None, **kwargs) -> bool:
        msg: str = self._create_msg(status=Status.FAILED, msg=msg, **kwargs)
        return self.send(msg, **kwargs)

    def stopped(self, msg: Optional[str] = None, **kwargs) -> bool:
        msg: str = self._create_msg(status=Status.STOPPED, msg=msg, **kwargs)
        return self.send(msg, **kwargs)

    @classmethod
    def _create_msg(
            cls,
            status: Status,
            *,
            msg: Optional[str] = None,
            start_dt: Optional[datetime] = None,
            now: Optional[datetime] = None,
            raise_error: bool = False,
            **kwargs
    ) -> str:
        if status is Status.SUCCEEDED:
            out: str = f'Succeeded'
        elif status is Status.FAILED:
            out: str = f'Failed'
        elif status is Status.STOPPED:
            out: str = f'Stopped'
        elif status is Status.RUNNING:
            out: str = f'Running'
        elif status is Status.PENDING:
            out: str = f'Pending'
        else:
            raise NotImplementedError(f'Unsupported status: {status}')
        now: datetime = get_default(now, datetime.now())
        now: datetime = now.replace(tzinfo=now.astimezone().tzinfo)
        out += f' at {StringUtil.readable_datetime(now, human=True)}'
        if msg is not None:
            out = f'[{out}] {msg}'
        if start_dt is not None:
            start_dt: datetime = start_dt.replace(tzinfo=start_dt.astimezone().tzinfo)
            try:
                out += f' ({StringUtil.readable_seconds((now - start_dt))} elapsed)'
            except Exception as e:
                if raise_error:
                    raise e
                pass
        out: str = out.strip() + '.'
        return out


class NoopNotifier(Notifier):
    aliases = ['noop']

    @safe_validate_arguments
    def send(
            self,
            msg: Union[constr(min_length=1), int, float, BaseModel],
            **kwargs
    ) -> bool:
        return True  ## Do nothing


class ChimeNotifier(Notifier):
    aliases = ['chime']

    webhook: constr(min_length=10, max_length=1024, regex='^.*hooks.chime.aws.*$', strip_whitespace=True)

    @safe_validate_arguments
    def send(
            self,
            msg: Union[constr(min_length=1), int, float, BaseModel],
            priority: bool = True,
            markdown: bool = True,
            **kwargs
    ) -> bool:
        if isinstance(msg, BaseModel):
            msg: str = f'```\n{msg.json(indent=4)}\n```'
            markdown: bool = True
        msg: str = str(msg)
        if priority:
            msg = '@All ' + msg  ## Will notify those with "Normal" and "Full" notification settings for a room.
        if markdown:
            msg = '/md\n\n' + msg
        response = requests.post(url=self.webhook, json={"Content": msg})
        return str(response.status_code).startswith('2')


class DiscordNotifier(Notifier):
    aliases = ['discord']

    webhook: constr(min_length=10, max_length=1024, regex='^.*discord.com/api/webhooks/.*$', strip_whitespace=True)

    @safe_validate_arguments
    def send(
            self,
            msg: Union[constr(min_length=1), int, float, BaseModel],
            **kwargs
    ) -> bool:
        if isinstance(msg, BaseModel):
            msg: str = f'```\n{msg.json(indent=4)}\n```'
        msg: str = str(msg)
        response = requests.post(self.webhook, json={"content": msg}, headers={'Content-Type': 'application/json'})
        return str(response.status_code).startswith('2')
