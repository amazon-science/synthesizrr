from typing import *
from abc import ABC
from synthesizrr.base.util import optional_dependency
from synthesizrr.base.framework import Visualization, Dataset, Predictions
from synthesizrr.base.constants import MLType, DataLayout, VisualizationBackend, VISUALIZATION_BACKEND_DEPENDENCIES
from pandas.core.frame import DataFrame as PandasDataFrame


class CorrelationMatrix(Visualization, ABC):
    aliases = ['feature_correlation', 'correlation_matrix']
    data_classes = [Dataset, Predictions]

    class Params(Visualization.Params):
        xaxis_position: str = 'top'
        decimals: int = 2
        height: int = 600
        width: int = 650
        method: Literal['pearson', 'kendall', 'spearman'] = 'pearson'


with optional_dependency(*VISUALIZATION_BACKEND_DEPENDENCIES[VisualizationBackend.HVPLOT_BOKEH]):
    import bokeh, holoviews as hv


    class CorrelationMatrixHvplotBokeh(CorrelationMatrix):
        backend = VisualizationBackend.HVPLOT_BOKEH

        def plot_figure(self, data: Union[Dataset, Predictions], **kwargs) -> Any:
            corr_mat: PandasDataFrame = data.features(
                MLType.INT,
                MLType.FLOAT,
            ).pandas().corr(method=self.params.method).round(self.params.display_decimals)
            negative_corr_palette = bokeh.palettes.Reds[256][48:]
            postivie_corr_palette = bokeh.palettes.Blues[256][48:]
            heatmap = corr_mat.hvplot.heatmap(
                cmap=bokeh.palettes.diverging_palette(
                    negative_corr_palette,
                    postivie_corr_palette,
                    n=len(negative_corr_palette) + len(postivie_corr_palette)
                ),
                clim=(-1, 1)
            )
            labels = hv.Labels(heatmap).opts(
                text_font_size=self.params.data_font_size,
            )
            return (heatmap * labels).opts(
                title=f"{self.params.method.capitalize()} Correlation between features",
                invert_xaxis=True,
            )
