from typing import *
from abc import abstractmethod, ABC
import io, requests, pickle
from requests import Response
from synthergent.base.constants import FileContents, FileFormat, MLTypeSchema
from synthergent.base.util.language import is_list_like, as_list, optional_dependency
from synthergent.base.util import AutoEnum, auto, StringUtil, FileSystemUtil, StructuredBlob, safe_validate_arguments
from synthergent.base.data.reader.Reader import Reader
from synthergent.base.util.aws import S3Util


class PickleReader(Reader):
    file_contents = [
        FileContents.PICKLED_OBJECT,
    ]
    streams = [io.BytesIO]
    file_formats = [FileFormat.PICKLE]

    @safe_validate_arguments
    def _read_stream(
            self,
            stream: io.BytesIO,
            file_contents: Optional[FileContents] = None,
            data_schema: Optional[MLTypeSchema] = None,
            **kwargs,
    ) -> StructuredBlob:
        error_to_raise: Optional[Exception] = None
        for _ in range(self.retry):
            try:
                stream.seek(0)
                return pickle.load(stream)
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
        raise NotImplementedError()

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
                        raise IOError(f'More than one pickle file found:\n"{local_path}"')
                    local_path: str = local_path[0]
                return FileSystemUtil.get_file_pickle(
                    local_path,
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
                return S3Util.get_s3_object_pickle(
                    s3_path,
                )
            except Exception as e:
                error_to_raise = e
        raise error_to_raise
