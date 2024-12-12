from typing import *
import io
from abc import abstractmethod, ABC
from synthergent.base.data.writer.Writer import Writer
from synthergent.base.constants import FileContents, MLType, MLTypeSchema
from synthergent.base.util import FileSystemUtil, StringUtil
from synthergent.base.util.aws import S3Util
from pydantic import *


class ConfigWriter(Writer, ABC):
    file_contents = [
        FileContents.CONFIG,
        FileContents.SCHEMA,
        FileContents.AIW_SCHEMA,
    ]
    streams = [io.TextIOBase]

    def _write_stream(
            self,
            stream: io.IOBase,
            data: Any,
            file_contents: FileContents,
            **kwargs,
    ) -> NoReturn:
        stream.write(self._get_obj_str(data, file_contents=file_contents, **kwargs))

    def _write_local(
            self,
            local_path: str,
            data: Any,
            file_contents: FileContents,
            file_name: Optional[constr(min_length=1)] = None,
            **kwargs,
    ) -> str:
        if FileSystemUtil.is_path_valid_dir(local_path):
            if file_name is None:
                raise ValueError(f'You must pass `file_name` when writing to local directory "{local_path}".')
            local_path: str = FileSystemUtil.construct_file_path_in_dir(
                path=local_path,
                name=file_name,
                file_ending=self.file_ending,
            )
        FileSystemUtil.put_file_str(
            local_path,
            file_str=self._get_obj_str(
                data,
                file_contents=file_contents,
                **kwargs,
            ),
            overwrite=True,
        )
        return local_path

    def _write_s3(
            self,
            s3_path: str,
            data: Any,
            file_contents: FileContents,
            **kwargs,
    ) -> str:
        S3Util.put_s3_object_str(
            s3_path,
            obj_str=self._get_obj_str(
                data,
                file_contents=file_contents,
                **kwargs,
            ),
            overwrite=True,
        )
        return s3_path

    def _get_obj_str(self, data: Any, file_contents: FileContents, **kwargs) -> Any:
        if file_contents is FileContents.SCHEMA:
            data = MLType.convert_values_to_str(data)
        obj_str: str = self.to_str(data, **kwargs)
        StringUtil.assert_not_empty(obj_str)
        return obj_str

    @abstractmethod
    def to_str(self, content: Any, **kwargs) -> str:
        pass
