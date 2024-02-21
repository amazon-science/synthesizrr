from typing import *
from abc import ABC
import numpy as np, pandas as pd
from synthesizrr.base.util import as_list, optional_dependency
from synthesizrr.base.framework import Visualization, Dataset, Predictions
from synthesizrr.base.framework.task.text_generation import NextTokens, TextGenerations, TEXT_PROMPT_COL, GENERATED_TEXTS_COL
from synthesizrr.base.constants import MLType, DataLayout, VisualizationBackend, VISUALIZATION_BACKEND_DEPENDENCIES
from pandas.core.frame import DataFrame as PandasDataFrame, Series as PandasSeries


class TextGenerationsViz(Visualization, ABC):
    aliases = ['text-generation', 'text-generations', 'generated-text']
    data_classes = [NextTokens, TextGenerations]

    class Params(Visualization.Params):
        prompts_color: str = 'grey'
        generations_color: str = 'blue'
        ground_truth_color: str = 'green'
        divider_color: str = 'light_grey'
        divider_length: int = 100


with optional_dependency(*VISUALIZATION_BACKEND_DEPENDENCIES[VisualizationBackend.TERMCOLOR]):
    from termcolor import cprint


    class TextGenerationsViz(TextGenerationsViz):
        backend = VisualizationBackend.TERMCOLOR

        def plot_figure(self, data: Union[NextTokens, TextGenerations], **kwargs) -> Any:
            div: str = (f'═' * self.params.divider_length) + '\n' + (f'═' * self.params.divider_length)
            for d in data.data.to_list_of_dict():
                cprint(div, color=self.params.divider_color)
                cprint(d[TEXT_PROMPT_COL], color=self.params.prompts_color, end='', attrs=['concealed'])
                cprint(d[GENERATED_TEXTS_COL], color=self.params.generations_color)
            return True
