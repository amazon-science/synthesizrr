from typing import *
from abc import abstractmethod, ABC
import io, requests
from requests import Response
from synthesizrr.base.constants import FileContents, MLType, MLTypeSchema
from synthesizrr.base.util.language import is_list_like, as_list, optional_dependency
from synthesizrr.base.util import AutoEnum, auto, StringUtil, FileSystemUtil, StructuredBlob, safe_validate_arguments
from synthesizrr.base.data.reader.Reader import Reader
from synthesizrr.base.util.aws import S3Util

StructuredBlob = Union[List, Dict, List[Dict]]


class ConfigReader(Reader, ABC):
    file_contents = [
        FileContents.CONFIG,
        FileContents.SCHEMA,
        FileContents.AIW_SCHEMA,
    ]
    streams = [io.TextIOBase]

    @safe_validate_arguments
    def _read_stream(
            self,
            stream: io.TextIOBase,
            file_contents: Optional[FileContents] = None,
            data_schema: Optional[MLTypeSchema] = None,
            **kwargs,
    ) -> StructuredBlob:
        error_to_raise: Optional[Exception] = None
        for _ in range(self.retry):
            try:
                return self._process_config_str(
                    stream.read(),
                    file_contents=file_contents,
                    **kwargs,
                )
            except Exception as e:
                error_to_raise = e
        raise error_to_raise

    @safe_validate_arguments
    def _read_url(
            self,
            url: Union[str, List[str]],
            file_contents: Optional[FileContents] = None,
            data_schema: Optional[MLTypeSchema] = None,
            **kwargs,
    ) -> StructuredBlob:
        error_to_raise: Optional[Exception] = None
        for _ in range(self.retry):
            try:
                if is_list_like(url):
                    if len(url) > 1:
                        raise IOError(f'More than one config file found:\n"{url}"')
                    url: str = url[0]
                text: Optional[str] = None
                with optional_dependency('smart_open'):
                    import smart_open
                    with smart_open.open(url, mode='r') as inp:
                        text: str = inp.read()
                if text is None:
                    response: Response = requests.get(url)
                    text: str = response.text
                return self._process_config_str(
                    text,
                    file_contents=file_contents,
                    **kwargs,
                )
            except Exception as e:
                error_to_raise = e
        raise error_to_raise

    @safe_validate_arguments
    def _read_local(
            self,
            local_path: Union[str, List[str]],
            file_contents: Optional[FileContents] = None,
            data_schema: Optional[MLTypeSchema] = None,
            **kwargs,
    ) -> StructuredBlob:
        error_to_raise: Optional[Exception] = None
        for _ in range(self.retry):
            try:
                if is_list_like(local_path):
                    if len(local_path) > 1:
                        raise IOError(f'More than one config file found:\n"{local_path}"')
                    local_path: str = local_path[0]
                return self._process_config_str(
                    FileSystemUtil.get_file_str(local_path),
                    file_contents=file_contents,
                    **kwargs,
                )
            except Exception as e:
                error_to_raise = e
        raise error_to_raise

    @safe_validate_arguments
    def _read_s3(
            self,
            s3_path: Union[str, List[str]],
            file_contents: Optional[FileContents] = None,
            data_schema: Optional[MLTypeSchema] = None,
            files_to_ignore: List[str] = StringUtil.FILES_TO_IGNORE,
            **kwargs,
    ) -> StructuredBlob:
        error_to_raise: Optional[Exception] = None
        for _ in range(self.retry):
            try:
                if is_list_like(s3_path):
                    if len(s3_path) > 1:
                        raise IOError(f'More than one config file found:\n"{s3_path}"')
                    s3_path: str = s3_path[0]
                return self._process_config_str(
                    S3Util.get_s3_object_str(s3_path),
                    file_contents=file_contents,
                    **kwargs,
                )
            except Exception as e:
                error_to_raise = e
        raise error_to_raise

    def _process_config_str(
            self,
            string: str,
            file_contents: Optional[FileContents] = None,
            **kwargs,
    ) -> StructuredBlob:
        structured_blob: StructuredBlob = self._from_str(string, **kwargs)
        if file_contents is FileContents.SCHEMA:
            structured_blob = MLType.convert_values(structured_blob)
        return structured_blob

    @abstractmethod
    def _from_str(self, string: str, **kwargs) -> StructuredBlob:
        pass
