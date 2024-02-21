from typing import *
from abc import ABC
import numpy as np, pandas as pd
from synthesizrr.base.util import as_list, optional_dependency
from synthesizrr.base.framework import Visualization, Dataset, Predictions
from synthesizrr.base.constants import MLType, DataLayout, VisualizationBackend, VISUALIZATION_BACKEND_DEPENDENCIES
from pandas.core.frame import DataFrame as PandasDataFrame, Series as PandasSeries


class Waffle(Visualization, ABC):
    aliases = ['waffle_chart', 'waffle']
    data_classes = [Dataset, Predictions]

    class Params(Visualization.Params):
        vertical_boxes: int = 10
        horizontal_boxes: int = 10
        width: int = 400
        height: int = 300

        ## Default list of colors was generated from Bokeh's Category20, after removing the greys, and moving the
        ## lighter shades to the end. We use greys for "Other" category in the waffle plot.
        ## Code to generate list of colors:
        ## from bokeh import palettes
        ## swap_alternate = lambda l: l[0::2] + l[1::2]
        ## colors = tuple(swap_alternate(palettes.Category20[20][:14] + palettes.Category20[20][16:]))
        colors: Tuple[str, ...] = (
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d', '#9edae5',
        )
        other_color: str = 'whitesmoke'


with optional_dependency(*VISUALIZATION_BACKEND_DEPENDENCIES[VisualizationBackend.HVPLOT_BOKEH]):
    import bokeh, holoviews as hv
    from holoviews import opts
    from bokeh.models.tickers import FixedTicker


    class WaffleHvplotBokeh(Waffle):
        backend = VisualizationBackend.HVPLOT_BOKEH

        def plot_figure(self, data: Union[Dataset, Predictions], column: Optional[str] = None, **kwargs) -> Any:
            if column is None and 'col' in kwargs:
                column: str = kwargs['col']
            if not isinstance(column, str):
                raise ValueError(f'Must pass a string value for `column` which should be a column from the data.')
            num_squares: int = self.params.vertical_boxes * self.params.horizontal_boxes
            data_col = data.data[column]
            data_counts: PandasSeries = data_col.pandas().value_counts()
            data_counts_num_squares: PandasSeries = num_squares * data_counts / data_col.shape[0]
            data_num_squares: PandasSeries = data_counts_num_squares[data_counts_num_squares >= 1].round().astype(int)
            data_num_squares: Dict[str, int] = data_num_squares.to_dict()

            data_vals: Dict[str, int] = {}
            vals = []
            i = None
            for i, (val, count) in enumerate(data_num_squares.items()):
                data_vals[val] = i
                vals.extend([
                    i
                    for _ in range(count)
                ])
            for others_key in ['Other', 'OTHER VALUES', None]:
                if others_key is not None and others_key not in data_vals:
                    data_vals[others_key] = i + 1
                    break
            if others_key is None:
                raise ValueError(f'We have exhaused the list of possible "Other" keys, please rename a column.')

            vals.extend([data_vals[others_key] for _ in range(num_squares - len(vals))])
            vals: PandasDataFrame = pd.DataFrame(
                np.array(vals).reshape((self.params.vertical_boxes, self.params.horizontal_boxes)),
                columns=[str(i) for i in range(self.params.horizontal_boxes)][::-1],
                index=[str(i) for i in range(self.params.vertical_boxes)]
            )
            listfloat_color_levels = sorted(list(data_vals.values()))
            liststr_cmap = self.params.colors[:len(listfloat_color_levels) - 2] + (self.params.other_color,)
            fig = vals.hvplot.heatmap(
                shared_axes=False,
            ).opts(
                title=f'Waffle of "{column}" (count≥{data_counts[data_counts.index.isin(data_vals.keys())].iloc[-1]})',
                ## Ref: https://discourse.holoviz.org/t/how-to-control-heatmap-colorbar-ticks/4232/4
                clim=(min(listfloat_color_levels), max(listfloat_color_levels) + 1),
                colorbar_opts={
                    ## Ref: https://docs.bokeh.org/en/latest/docs/reference/models/annotations.html#bokeh.models.ColorBar
                    "major_label_overrides": {v: str(k) for k, v in data_vals.items()},
                    # 'title': column,
                    'major_label_text_baseline': 'top',
                    'width': 10,
                    'ticker': FixedTicker(ticks=listfloat_color_levels)
                },
                xaxis=None,
                yaxis=None,
                color_levels=listfloat_color_levels,
                cmap=liststr_cmap,
            )
            return fig
