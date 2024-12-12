from typing import *
from synthergent.base.util import AutoEnum, auto, as_list

UNKNOWN_LABEL_FILL: str = '__UNKNOWN__LABEL__'


class FileFormat(AutoEnum):
    ## Config:
    YAML = auto()
    JSON = auto()
    ## Dataframe:
    CSV = auto()
    TSV = auto()
    PARQUET = auto()
    METRICS_JSONLINES = auto()
    JSONLINES = auto()
    EXCEL = auto()
    LIBSVM = auto()  ## Example datasets: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
    ## Binary
    PICKLE = auto()
    BIN = auto()
    ## Algorithm:
    FASTTEXT = auto()
    BLAZINGTEXT = auto()
    VOWPALWABBIT = auto()
    XGBOOST = auto()
    TFRECORD = auto()
    ## Image:
    PNG = auto()
    JPEG = auto()
    TIFF = auto()
    BMP = auto()
    GIF = auto()
    ICO = auto()
    WEBP = auto()
    SVG = auto()
    ## Document:
    PDF = auto()
    # Embedding:
    NPZ = auto()
    PSD = auto()  ## Adobe Photoshop
    ## Compressed formats:
    ZIP = auto()
    ## Other:
    PLAIN_TEXT = auto()
    CUSTOM = auto()

    def is_binary_format(self):
        return self in BINARY_FILE_FORMATS


class Storage(AutoEnum):
    STREAM = auto()  ## io.StringIO and io.BytesIO
    LOCAL_FILE_SYSTEM = auto()  ## /whatever/the/path
    S3 = auto()  ## s3://
    URL = auto()  ## http://, https://, etc.


REMOTE_STORAGES: Set[Storage] = {Storage.S3, Storage.URL}


class FileContents(AutoEnum):
    CONFIG = auto()
    SCHEMA = auto()
    AIW_SCHEMA = auto()
    PICKLED_OBJECT = auto()
    DATAFRAME = auto()
    ASSET = auto()
    LABEL_ENCODING_DATAFRAME = auto()
    TRANSFORMATION_PIPELINE_ARTIFACTS_DIR = auto()
    ALGORITHM_TRAIN_DATASET = auto()
    ALGORITHM_INFERENCE_DATASET = auto()
    ALGORITHM_PREDICTIONS_DATASET = auto()  ## Serialized Predictions Format
    METRICS_DATAFRAME = auto()
    MODEL = auto()
    TENSORFLOW_MODEL = auto()
    PYTORCH_MODEL = auto()


FILE_FORMAT_TO_FILE_ENDING_MAP: Dict[FileFormat, Union[str, List[str]]] = {
    ## Map of file formats to file endings.
    ## If multiple valid file-endings exist for a format, mention them in decreasing order of preference.

    ## Data formats:
    ### CSV and TSV:
    FileFormat.CSV: ['.csv', '.csv.part'],
    FileFormat.TSV: ['.tsv', '.tsv.part'],
    ### JSON and JSONLINES:
    FileFormat.JSON: ['.json', '.aiw_schema.json'],
    FileFormat.JSONLINES: ['.jsonl', '.jsonl.part', '.jsonlines.json', '.jsonlines'],
    FileFormat.METRICS_JSONLINES: '.metrics.json',
    ### YAML:
    FileFormat.YAML: ['.yaml', '.yml'],
    ### Plain text:
    FileFormat.PLAIN_TEXT: '.txt',
    ### Parquet:
    FileFormat.PARQUET: '.parquet',
    ### Pickled Python objects:
    FileFormat.PICKLE: '.pickle',  ## Ref: https://docs.python.org/3.7/library/pickle.html#examples
    ### Excel:
    FileFormat.EXCEL: '.xlsx',
    ### LIBSVM:
    FileFormat.LIBSVM: '.libsvm',
    ### Compressed:
    FileFormat.ZIP: '.zip',
    ## Image:
    FileFormat.PNG: '.png',
    FileFormat.JPEG: ['.jpg', '.jpeg'],
    FileFormat.TIFF: ['.tif', '.tiff'],
    FileFormat.BMP: '.bmp',
    FileFormat.GIF: '.gif',
    FileFormat.ICO: '.ico',
    FileFormat.WEBP: '.webp',
    FileFormat.SVG: '.svg',

    ## Algorithm formats:
    ### BlazingText:
    FileFormat.BLAZINGTEXT: '.blazingtext.txt',
    ### FastText:
    FileFormat.FASTTEXT: '.fasttext.txt',
    ### VowpalWabbit:
    FileFormat.VOWPALWABBIT: '.vw.txt',
    ### XGBoost:
    FileFormat.XGBOOST: '.xgboost.libsvm',  ## LIBSVM is used for XGB, CatBoost, LightGBM, etc.
    ### TFRecord:
    FileFormat.TFRECORD: '.tfrecord',

    ## EmbeddingFormats:
    ### NPZ:
    FileFormat.NPZ: '.npz'
}

FILE_ENDING_TO_FILE_FORMAT_MAP: Dict[str, FileFormat] = {}
for file_format, file_ending in FILE_FORMAT_TO_FILE_ENDING_MAP.items():
    for fe in as_list(file_ending):
        if fe in FILE_ENDING_TO_FILE_FORMAT_MAP:
            raise ValueError(f'Cannot have duplicate file-ending keys: {fe}')
        FILE_ENDING_TO_FILE_FORMAT_MAP[fe] = file_format

FILE_FORMAT_TO_CONTENT_TYPE_MAP: Dict[FileFormat, str] = {
    FileFormat.CSV: 'text/csv',
    FileFormat.TSV: 'text/tsv',
    FileFormat.PARQUET: 'application/parquet',
    FileFormat.EXCEL: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    ## Ref: https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types/Common_types
    FileFormat.JSON: 'application/json',
    FileFormat.JSONLINES: 'application/jsonlines',
    FileFormat.YAML: 'application/x-yaml',
    FileFormat.LIBSVM: 'text/libsvm',
    FileFormat.PICKLE: 'application/octet-stream',  ## Ref: https://stackoverflow.com/a/40433504
    FileFormat.TFRECORD: 'application/x-tfexample',
    FileFormat.PLAIN_TEXT: 'text/plain',
    FileFormat.ZIP: 'application/zip',

    FileFormat.PNG: 'image/png',
    FileFormat.JPEG: 'image/jpeg',
    FileFormat.TIFF: 'image/tiff',
    FileFormat.BMP: 'image/bmp',
    FileFormat.GIF: 'image/gif',
    FileFormat.ICO: 'image/vnd.microsoft.icon',
    FileFormat.WEBP: 'image/webp',
    FileFormat.SVG: 'image/svg+xml',

    ## Made-up algorithm content types:
    FileFormat.BLAZINGTEXT: 'application/blazingtext',
    FileFormat.XGBOOST: 'application/xgboost',
    FileFormat.FASTTEXT: 'application/fasttext',
    FileFormat.VOWPALWABBIT: 'application/vw',
}
CONTENT_TYPE_TO_FILE_FORMAT_MAP: Dict[str, FileFormat] = {}
for file_format, content_type in FILE_FORMAT_TO_CONTENT_TYPE_MAP.items():
    if content_type in CONTENT_TYPE_TO_FILE_FORMAT_MAP:
        raise ValueError(f'Cannot have duplicate content-type keys: {content_type}')
    CONTENT_TYPE_TO_FILE_FORMAT_MAP[content_type] = file_format

BINARY_FILE_FORMATS: List[FileFormat] = [
    FileFormat.BIN,
    FileFormat.PARQUET,
    FileFormat.EXCEL,
    FileFormat.PICKLE,
    FileFormat.TFRECORD,
]

CONFIG_FILE_FORMATS: List[FileFormat] = [
    FileFormat.JSON,
    FileFormat.YAML,
]

DATAFRAME_FILE_FORMATS: List[FileFormat] = [
    FileFormat.CSV,
    FileFormat.TSV,
    FileFormat.PARQUET,
    FileFormat.EXCEL,
    FileFormat.JSONLINES,
]
