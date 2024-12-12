from typing import *
import io, yaml
from synthergent.base.data.writer.config.ConfigWriter import ConfigWriter
from synthergent.base.util import StructuredBlob
from synthergent.base.constants import FileFormat, FileContents


class YamlWriter(ConfigWriter):
    file_formats = [FileFormat.YAML]

    def to_str(self, content: Any, **kwargs) -> str:
        stream = io.StringIO()
        yaml.safe_dump(content, stream, **self.filtered_params(yaml.safe_dump))
        return stream.getvalue()
