from typing import *
import io, json
from synthergent.base.data.writer.config.ConfigWriter import ConfigWriter
from synthergent.base.util import StructuredBlob
from synthergent.base.constants import FileFormat, FileContents


class JsonWriter(ConfigWriter):
    file_formats = [FileFormat.JSON]

    class Params(ConfigWriter.Params):
        indent: int = 4

    def to_str(self, content: StructuredBlob, **kwargs) -> str:
        return json.dumps(content, **self.filtered_params(json.dumps))
