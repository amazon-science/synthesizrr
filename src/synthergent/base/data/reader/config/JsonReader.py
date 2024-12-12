import json
from synthergent.base.data.reader.config.ConfigReader import StructuredBlob, ConfigReader
from synthergent.base.constants import FileFormat


class JsonReader(ConfigReader):
    file_formats = [FileFormat.JSON]

    def _from_str(self, string: str, **kwargs) -> StructuredBlob:
        return json.loads(string, **(self.filtered_params(json.loads)))
