from typing import *
from importlib import import_module
from synthesizrr.base.util import AutoEnum, auto, optional_dependency


class VisualizationBackend(AutoEnum):
    HVPLOT_BOKEH = auto()
    HVPLOT_MATPLOTLIB = auto()
    BOKEH = auto()
    PLOTLY_EXPRESS = auto()
    SEABORN = auto()
    MATPLOTLIB = auto()
    ALTAIR = auto()
    TERMCOLOR = auto()


VISUALIZATION_BACKEND_DEPENDENCIES: Dict[VisualizationBackend, List[str]] = {
    VisualizationBackend.HVPLOT_BOKEH: ['hvplot', 'holoviews', 'bokeh'],
    VisualizationBackend.HVPLOT_MATPLOTLIB: ['hvplot', 'holoviews', 'matplotlib'],
    VisualizationBackend.BOKEH: ['bokeh'],
    VisualizationBackend.PLOTLY_EXPRESS: ['plotly', 'plotly-express'],
    VisualizationBackend.SEABORN: ['seaborn'],
    VisualizationBackend.MATPLOTLIB: ['matplotlib'],
    VisualizationBackend.ALTAIR: ['altair', 'vega_datasets'],
    VisualizationBackend.TERMCOLOR: ['termcolor'],
}

AVAILABLE_VISUALIZATION_BACKENDS: Set[VisualizationBackend] = set()
for viz_backend, deps in VISUALIZATION_BACKEND_DEPENDENCIES.items():
    with optional_dependency(*deps):
        for dep in deps:
            import_module(dep)
        AVAILABLE_VISUALIZATION_BACKENDS.add(viz_backend)
