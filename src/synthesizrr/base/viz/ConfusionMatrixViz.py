from typing import *
from abc import ABC
import numpy as np, pandas as pd
from pandas.core.frame import DataFrame as PandasDataFrame
from synthesizrr.base.util import optional_dependency
from synthesizrr.base.framework import Visualization, Dataset, Predictions, ClassificationPredictions
from synthesizrr.base.metric.classification_metrics import ConfusionMatrix
from synthesizrr.base.constants import MLType, DataLayout, VisualizationBackend, VISUALIZATION_BACKEND_DEPENDENCIES


class ConfusionMatrixViz(Visualization, ABC):
    aliases = ['confusion', 'confusion_matrix']
    data_classes = [Dataset, Predictions]

    class Params(Visualization.Params):
        xaxis_position: str = 'top'
        decimals: int = 2
        height: int = 600
        width: int = 650
        method: Literal['pearson', 'kendall', 'spearman'] = 'pearson'


with optional_dependency(*VISUALIZATION_BACKEND_DEPENDENCIES[VisualizationBackend.SEABORN]):
    import seaborn as sns
    import matplotlib.pyplot as plt


    class ConfusionMatrixVizSeaborn(ConfusionMatrixViz):
        backend = VisualizationBackend.SEABORN

        def plot_figure(self, data: ClassificationPredictions, **kwargs) -> Any:
            cf_mat_metric: ConfusionMatrix = data.evaluate(ConfusionMatrix())
            cf_mat: PandasDataFrame = cf_mat_metric.pandas()
            accuracy: float = cf_mat.values.diagonal().sum() / cf_mat.sum().sum()
            plt.figure(figsize=np.array((10, 7)) * 1.10)
            plt.title(
                f'Confusion matrix across {len(data.labelspace)} labels '
                f'(accuracy={100 * accuracy:.1f}%)',
                fontsize=12
            )
            return sns.heatmap(cf_mat, annot=True, cmap=sns.color_palette("Blues", as_cmap=True))
