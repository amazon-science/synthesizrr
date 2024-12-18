from typing import *
import math, ray, re, multiprocessing as mp
from abc import ABC, abstractmethod
import pandas as pd
from nltk import word_tokenize
from pydantic import root_validator, conint, confloat, constr, Extra
from pydantic.typing import Literal
from sklearn.model_selection import train_test_split
from synthergent.base.constants import FileFormat, DataLayout, DataSplit, Parallelize
from synthergent.base.data import FileMetadata, Reader, Writer
from synthergent.base.data.reader import DataFrameReader
from synthergent.base.framework.ray_base import ActorComposite
from synthergent.base.framework.chain.Chain import Step
from synthergent.base.data.sdf import ScalableDataFrame, ScalableDataFrameRawType, DaskDataFrame
from synthergent.base.framework.metric import Metric, Metrics
from synthergent.base.framework.task.classification import ClassificationData
from synthergent.base.framework.task.text_generation import TextGenerationsPredictionsBase, GENERATED_TEXTS_COL
from synthergent.base.framework.task_data import Dataset
from synthergent.base.util import Parameters, as_list, as_set, \
    safe_validate_arguments, get_default, StringUtil, AutoEnum, auto, alias, not_impl, \
    binary_search, ProgressBar, shuffle_items, only_item, irange, punct_normalize, remove_nulls, all_are_none, \
    dispatch, Timer
from synthergent.base.util.concurrency import accumulate
from bs4 import BeautifulSoup
from synthergent.cleaner.Cleaner import Cleaner


class XMLParser(Cleaner):
    class Params(Cleaner.Params):
        string_cleaner: Callable
        col: constr(min_length=1) = 'generations'
        tags: Union[List[constr(min_length=1)], constr(min_length=1)]
        path_sep: constr(min_length=1) = '.'

        @root_validator(pre=True)
        def _set_xml_parser_params(cls, params: Dict) -> Dict:
            params['tags'] = as_list(params['tags'])
            return params

    def clean(
            self,
            data: pd.DataFrame,
            **kwargs,
    ) -> pd.DataFrame:
        data: pd.DataFrame = data.apply(self._set_xml_tags, axis=1)
        return data

    def _set_xml_tags(self, row: pd.Series) -> pd.Series:
        text: str = row[self.params.col]
        extracted_tag_vals: Dict[str, Optional[List[str]]] = extract_xpath_values(
            text,
            optional_tags=self.params.tags,
            path_sep=self.params.path_sep,
        )
        for tag_path, tag_vals in extracted_tag_vals.items():
            row[tag_path]: List[str] = get_default(tag_vals, [])
        return row


def extract_xpath_values(
        text: str,
        *,
        mandatory_tags: Optional[List[str]] = None,
        optional_tags: Optional[List[str]] = None,
        path_sep: str = '.',
        parser: str = "html.parser",
        extract: Literal["text", "xml"] = "text",
        exception_cls: Type[Exception] = ValueError,
) -> Dict[str, Optional[List[str]]]:
    """
    Extracts tag values as a list of strings, allowing passing xpath-like tags.
    E.g.
    extract_xpath_values(
        '''
        <thinking>
        <step>Do thing</step>
        <step>Do another thing</step>
        <step>Do third thing</step>
        </thinking>

        <grocery>
        <item>item 1</item>
        <item>item 2</item>
        <item>item 3</item>
        </grocery>
        ''',
        mandatory_tags=['thinking.step', 'grocery'],
        optional_tags=['missing-tag'],
        extract='text'
    )

    Returns:
    {
        'thinking.step': ['Do thing', 'Do another thing', 'Do third thing'],
        'grocery': ['item 1\nitem 2\nitem 3'],
        'missing-tag': None
    }
    :param text: text to be parsed. Assumed to be XML-like.
    :param mandatory_tags: one or more xpath-like tags to be extracted. If these do not exist, they are set to be None in the
    :param optional_tags: one or more xpath-like tags to be extracted. If any of these do not exist, they are set to be None in
        the returned dict (see example above).
    :param path_sep: separator between paths. Default '.' for jsonpath, or '/' for XMLPath.
    :param parser: which BeautifulSoup parser to use. Defaults to 'html.parser'.
    :param extract: what to extract from the tag. 'text' extracts only the nested text, 'xml' keeps nested tags as well.
     be excluded from the output dict. If False (default) an exception is thrown.
    :param exception_cls: which exception two throw in the above two cases.

    :return A dict with keys (tags) and values (list of content of that tag).
    - For each tag, we  return a list because in some cases, there might be multiple nested tags within a tag. For example, see
        "thinking.step" in the example above: there are multiple <step> tags, so we output each of them in a list entry.
        If you expect one instance of a tag to be present (e.g. <status>) use first_item_of_extracted_tags() function to pick it
        from the list e.g. first_item_of_extracted_tags(parsed_dict, 'status'). This will pick the first item.
    - The returned list for optional tag can have None values. This indicates that the tag was empty.
        For mandatory tags, an error will be thrown if any value is empty.
        e.g. For string
            <thinking>
            <step>Do thing</step>
            <step></step>       <----- EMPTY TAG
            <step>Do third thing</step>
            </thinking>
        When running
            extract_xpath_values(
                string,
                optional_tags=['thinking.step']
            )
        we should get:
            {
                'thinking.step': ['Do thing', None, 'Do third thing']
            }

    - For optional tags, if the tag itself does not exist in the string, then we will output a key with "None" value (rather
        than a list of None).
         e.g. For string
            <thinking>
            <step>Do thing</step>
            <step></step>       <----- EMPTY TAG
            <step>Do third thing</step>
            </thinking>
        extract_xpath_values(
            string,
            optional_tags=['missing-tag']
        )
        we should get:
            {
                'missing-tag': None
            }
    """
    if mandatory_tags is None and optional_tags is None:
        raise exception_cls(
            f"At least one of `mandatory_tags` and `optional_tags` should be non-None."
        )
    if mandatory_tags is not None:
        assert isinstance(mandatory_tags, list)
        assert len(mandatory_tags) > 0
    else:
        mandatory_tags = []
    if optional_tags is not None:
        assert isinstance(optional_tags, list)
        assert len(optional_tags) > 0
    else:
        optional_tags = []
    assert (
            len(set(mandatory_tags).intersection(set(optional_tags))) == 0
    ), "Unsupported: common tags in mandatory and optional."

    soup = BeautifulSoup(text, parser)
    parsed_dict: Dict[str, Optional[List[str]]] = {}
    invalid_tags: List[str] = []
    for tag_xpath in mandatory_tags + optional_tags:
        tag_values = _extract_xpath(
            soup=soup,
            tag_xpath_parts=tag_xpath.strip().split(
                path_sep,
            ),  ## Allow Xpath-like selection e.g. "thinking.step"
            extract=extract,
        )
        if tag_values is None:  ## Only occurs when one of the tags in our xpath is missing.
            if tag_xpath in mandatory_tags:
                invalid_tags.append(tag_xpath)
            else:  ## It is an optional tag, set it to a list with None.
                parsed_dict[tag_xpath] = None
        else:
            clean_tag_values: List[str] = []
            for tag_value in tag_values:
                cleaned_tag_value = _clean_tag_value(
                    tag_value,
                    allow_empty_tag_value=False,
                )
                if cleaned_tag_value is not None:
                    clean_tag_values.append(cleaned_tag_value)

            if not clean_tag_values:
                if tag_xpath in mandatory_tags:  ## Mandatory tag value is invalid
                    invalid_tags.append(tag_xpath)
                else:
                    parsed_dict[tag_xpath] = None
            else:
                parsed_dict[tag_xpath] = clean_tag_values
    if len(invalid_tags) > 0:
        raise exception_cls(f'Could not extract tags {invalid_tags} from the text:\n"""{text}"""')
    return parsed_dict


def _extract_xpath(
        soup: Any,
        *,
        tag_xpath_parts: List[str],
        extract: Literal["text", "xml"],
) -> Optional[List[str]]:
    if soup is None:
        return None
    if len(tag_xpath_parts) == 1:
        ## Base case
        if soup.find(tag_xpath_parts[0], recursive=False) is None:
            return None
        tag_values = []
        for tag_soup in soup.find_all(tag_xpath_parts[0], recursive=False):
            if extract == "text":
                tag_values.append(tag_soup.text)
            elif extract == "xml":
                tag_values.append(tag_soup.decode_contents())
            else:
                raise NotImplementedError(f'Unsupported value: extract="{extract}"')
        return tag_values
    else:
        ## Recurse:
        return _extract_xpath(
            soup=soup.find(tag_xpath_parts[0], recursive=False),  ## Go 1 level deeper
            tag_xpath_parts=tag_xpath_parts[1:],
            extract=extract,
        )


def _clean_tag_value(original_tag_value: str, allow_empty_tag_value: bool) -> Optional[str]:
    tag_value: Optional[str] = str_strip_if_not_empty(original_tag_value)
    if tag_value is None:
        if allow_empty_tag_value:
            return ""
        else:
            return None

    assert isinstance(tag_value, str)
    return tag_value


def str_is_empty(potential_string: Optional[Union[str, Any]]) -> bool:
    if potential_string is None:
        return True
    return str(potential_string).strip() in {"{}", "[]", "set()", "dict()", ""}


def str_strip_if_not_empty(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    if not isinstance(text, str):
        raise ValueError(f"Expected a string or None, found: {type(text)}")
    text = text.strip()
    if len(text) == 0:
        return None
    return text
