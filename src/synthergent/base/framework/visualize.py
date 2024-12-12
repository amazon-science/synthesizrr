from typing import *
from abc import ABC, abstractmethod
from synthergent.base.constants import VisualizationBackend, AVAILABLE_VISUALIZATION_BACKENDS, VISUALIZATION_BACKEND_DEPENDENCIES
from synthergent.base.util import Parameters, Registry, optional_dependency, safe_validate_arguments, as_list, str_normalize, \
    is_abstract, all_are_not_none, all_are_none, get_default
from synthergent.base.framework.task_data import Dataset
from synthergent.base.framework.predictions import Predictions
from pydantic import constr, root_validator, Extra
from functools import singledispatchmethod


class Visualization(Parameters, Registry, ABC):
    ## TODO: split this into multiple classes:
    ## Visualization
    ## 1) Library-specific subclasses (these setup, run post-processing code, hold theme, etc):
    ## - Altair
    ## - Plotly
    ## - Matplotlib
    ## - Seaborn
    ## - Holoviews / Hvplot
    ## - etc.
    ## 2) Implementation-specific base subclasses (these implement get_data function which should get the data in a
    ## format that is easy to plot. Each kind of visualization will return different data format, but they will all
    ## be used by the plotting library.
    ## - ConfusionMatrix
    ## - FeatureCorrelation
    ## - etc.
    ## 3) Concrete implementations (should subclass one library-specific subclass and one implementation-specific base
    ## subclass).
    ## - class SeabornConfusionMatrix(Seaborn, ConfusionMatrix)
    ## - class PlotlyConfusionMatrix(Plotly, ConfusionMatrix)
    ## - class HoloviewsConfusionMatrix(Holoviews, ConfusionMatrix)
    ## - class MatplotlibFeatureCorrelation(Matplotlib, FeatureCorrelation)
    ## - class HoloviewsFeatureCorrelation(Holoviews, FeatureCorrelation)
    ## - etc.
    ## Let people implement whatever they want or can find on the internet.
    ## E.g. if I want to plot a Bump chart, and I found a convenient implementation in Altair, let me create a class
    ## as follows:
    ## class SeabornBumpChart(Seaborn):
    ##      def get_data(self, data: Any) -> Any:
    ##          ...
    ##      def plot(self, data: Any) -> Any:
    ##          # Use the data retrieved from .get_data() to plot the figure using the corresponding library.
    ##          ...
    _allow_multiple_subclasses = False
    _allow_subclass_override = True
    initialized_backends: ClassVar[Set[VisualizationBackend]] = set()
    data_classes: ClassVar[List[Type]]
    backend: ClassVar[VisualizationBackend]

    @classmethod
    def _pre_registration_hook(cls):
        cls.data_classes: ClassVar[List[Type]] = cls._validate_data_classes(cls.data_classes)

    @staticmethod
    def _validate_data_classes(data_classes: List[Type]) -> List[Type]:
        if data_classes is None:
            raise ValueError(
                f'You must specify class variable `data_classes` on a subclass of {Visualization}. '
                f'This must be a list of classes which the visualization can take as an input data, '
                f'e.g Dataset, ClassificationTaskData, Predictions, RegressionPredictions, pd.DataFrame, etc.'
            )
        data_classes: List = as_list(data_classes)
        for input_type in as_list(data_classes):
            if not isinstance(input_type, type):
                raise ValueError(
                    f'Class variable `data_classes` must be a list of classes which the visualization can take as an '
                    f'input data, e.g Dataset, ClassificationTaskData, Predictions, RegressionPredictions, '
                    f'pd.DataFrame, etc; found unsupported value: "{input_type}"'
                )
        return data_classes

    @classmethod
    def _registry_keys(cls) -> Optional[Union[List[Any], Any]]:
        return [
            (str_normalize(name), cls.backend)
            for name in (cls.class_name,) + cls.aliases
        ]

    @classmethod
    def get_subclass(
            cls,
            key: Any,
            **kwargs,
    ) -> Optional[Union[Type, List[Type]]]:
        if isinstance(key, tuple) and len(key) == 2:
            if (isinstance(key[0], (str, type))) and VisualizationBackend.matches_any(key[1]):
                key = (
                    str_normalize(key[0].__name__ if isinstance(key[0], type) else key[0]),
                    str_normalize(VisualizationBackend.from_str(key[1])),
                )
        elif isinstance(key, str):
            ## Assume it is the class name:
            Subclasses: Set[Type] = set()
            for registry_key, class_dict in cls._registry.items():
                if str_normalize(key) == str_normalize(registry_key[0]):
                    for Subclass in class_dict.values():
                        Subclasses.add(Subclass)
            if len(Subclasses) == 1:
                return next(iter(Subclasses))
            return list(Subclasses)
        return super().get_subclass(key=key, **kwargs)

    @classmethod
    def available_backends(cls, name: str) -> Set[VisualizationBackend]:
        available_backends: Set[VisualizationBackend] = set()
        for Subclass in as_list(cls.get_subclass(key=name)):
            available_backends.add(Subclass.backend)
        return available_backends

    class Config(Parameters.Config):
        keep_untouched = (singledispatchmethod,)

    class Params(Parameters):
        """
        BaseModel for parameters. Expected to be overridden by subclasses.
        E.g.
            class PrecisionRecallCurve(Visualization):
                class Params(Visualization.Params):
                    precision: confloat(ge=0.0, le=1.0)
                ...
        """
        width: int = None
        height: int = None
        title: Optional[str] = None
        title_font_size: Union[int, str] = '12pt'
        data_font_size: Union[int, str] = '8pt'
        xaxis_ticks_font_size: Union[int, str] = '8pt'
        yaxis_ticks_font_size: Union[int, str] = '8pt'
        xaxis_label_font_size: str = '11pt'
        yaxis_label_font_size: str = '11pt'
        xaxis_label_font_style: str = 'normal'
        yaxis_label_font_style: str = 'normal'
        xaxis_position: str = 'bottom'
        yaxis_position: str = 'left'
        decimals: int = 3

        class Config(Parameters.Config):
            extra = Extra.ignore

    name: constr(min_length=1)
    params: Params = {}

    @root_validator(pre=True)
    def convert_params(cls, params: Dict):
        params['params'] = super(Visualization, cls)._convert_params(cls.Params, params.get('params'))
        params['name'] = cls.class_name
        return params

    def __str__(self):
        params_str: str = self.json(indent=4, exclude={'figure'})
        out: str = f'{self.class_name} with params:\n{params_str}'
        return out

    @classmethod
    def of(
            cls,
            name: Optional[Union['Visualization', Dict, str]] = None,
            backend: Optional[VisualizationBackend] = None,
            **kwargs,
    ) -> 'Visualization':
        if isinstance(name, Visualization):
            return name
        if isinstance(name, dict):
            return cls.of(**name)
        if all_are_not_none(name, backend):
            backend: VisualizationBackend = VisualizationBackend.from_str(backend)
            VisualizationClass: Type[Visualization] = Visualization.get_subclass((name, backend))
        elif name is not None:
            if not isinstance(name, str):
                raise NotImplementedError(
                    f'Unsupported value for `name`; expected a string, but '
                    f'found {type(name)} with value:\n{name}'
                )
            backend: VisualizationBackend = cls._initialize_backend_for_viz(name=name, **kwargs)
            VisualizationClass: Type[Visualization] = Visualization.get_subclass((name, backend))
        else:
            ## name is None
            VisualizationClass: Type[Visualization] = cls
        if VisualizationClass == Visualization or is_abstract(VisualizationClass):
            raise ValueError(
                f'"{VisualizationClass.class_name}" is an abstract class. '
                f'To create an instance, please pass name="..." (and optionally backend="...")'
            )
        return VisualizationClass(params=kwargs)

    @classmethod
    @safe_validate_arguments
    def init(cls, backend: VisualizationBackend, **kwargs) -> bool:
        init_fn = getattr(cls, f'_initialize_{backend.lower().strip()}')
        return init_fn(**kwargs)

    @classmethod
    @safe_validate_arguments
    def plot(
            cls,
            data: Any,
            name: Optional[str] = None,
            backend: Optional[VisualizationBackend] = None,
            **kwargs
    ) -> Any:
        viz = Visualization.of(name=name, backend=backend, **kwargs)
        if not isinstance(data, tuple(viz.data_classes)):
            raise ValueError(
                f'Visualization "{viz.name if name is None else name}" only takes in data of the following types: '
                f'{list(cls.data_classes)}; found input data of type "{type(data)}".'
            )
        if isinstance(data, (Dataset, Predictions)):
            if not data.in_memory():
                raise ValueError(f'Can only visualize data which is in memory. Please call .read() beforehand.')
        figure: Optional[Any] = viz.plot_figure(data=data, **kwargs)
        return viz._postprocess_figure(figure=figure, backend=viz.backend, **kwargs)

    @classmethod
    @safe_validate_arguments
    def _initialize_backend_for_viz(
            cls,
            name: str,
            backend: Optional[VisualizationBackend] = None,
            **kwargs
    ) -> VisualizationBackend:
        def _create_import_error(available_backends: Set[VisualizationBackend]):
            out_str: str = f'Could not initialize dependencies for any of the implemented backends; ' \
                           f'please ensure the at least one of the following dependencies is installed:\n'
            for available_backend in available_backends:
                deps: List[str] = VISUALIZATION_BACKEND_DEPENDENCIES[available_backend]
                out_str += f'To plot using {available_backend}: {deps}'
            return ImportError(out_str)

        available_backends: Set[VisualizationBackend] = cls.available_backends(name)
        if backend is None:
            ## Pick one according to which dependencies exist.
            for available_backend in available_backends:
                if cls.init(available_backend):
                    backend: VisualizationBackend = available_backend
                    break
            if backend is None:
                raise _create_import_error(available_backends)
        else:
            init_fn = getattr(cls, f'_initialize_{backend.lower()}')
            if not init_fn(**kwargs):
                raise _create_import_error(available_backends)
        return backend

    @safe_validate_arguments
    def _postprocess_figure(
            self,
            figure: Any,
            backend: Optional[VisualizationBackend],
            **kwargs
    ):
        if figure is None:
            raise ValueError(
                f'Plotting using `{self.class_name}.plot_figure` must return a figure object; '
                f'instead, found {type(figure)}.'
            )
        postprocess_fn_name: str = f'_postprocess_{backend.lower().strip()}'
        postproces_fn: Callable = getattr(self, postprocess_fn_name)
        if figure is None:
            raise ValueError(
                f'Postprocessing function `{self.class_name}.{postprocess_fn_name}` must return a '
                f'figure object; instead, found {type(figure)}.'
            )
        return postproces_fn(figure, **kwargs)

    @classmethod
    def _initialize_hvplot_bokeh(cls, reinit: bool = False, set_theme: bool = True, **kwargs) -> bool:
        backend: VisualizationBackend = VisualizationBackend.HVPLOT_BOKEH
        with optional_dependency(
                *VISUALIZATION_BACKEND_DEPENDENCIES[backend],
                error='ignore'
        ):
            import bokeh
            import holoviews as hv
            import hvplot.pandas
            if reinit or backend not in cls.initialized_backends:
                hv.extension('bokeh')
            cls.initialized_backends.add(backend)
            return True
        return False

    def _postprocess_hvplot_bokeh(self, figure: Any, **kwargs) -> Any:
        ## Ref: https://holoviews.org/FAQ.html, see the following answer:
        ## "Q: What if I need to do more complex customization supported by the backend but not exposed in HoloViews?"
        from bokeh.plotting import Figure as BokehFigure
        def hook(plot, element):
            # Allows accessing the backends figure object
            p: BokehFigure = plot.state
            p.xaxis.axis_label_text_font_size = self.params.xaxis_label_font_size
            p.yaxis.axis_label_text_font_size = self.params.yaxis_label_font_size
            p.xaxis.axis_label_text_font_style = self.params.xaxis_label_font_style
            p.yaxis.axis_label_text_font_style = self.params.yaxis_label_font_style

        if self.params.title is not None:
            figure = figure.opts(title=self.params.title)
        if self.params.width is not None:
            figure = figure.opts(width=self.params.width)
        if self.params.height is not None:
            figure = figure.opts(height=self.params.height)
        figure = figure.opts(
            xaxis=self.params.xaxis_position,
            yaxis=self.params.yaxis_position,
            fontsize={
                'title': self.params.title_font_size,
                'xticks': self.params.xaxis_ticks_font_size,
                'yticks': self.params.yaxis_ticks_font_size,
            },
            hooks=[hook]
        )
        return figure

    @classmethod
    def _initialize_hvplot_matplotlib(cls, reinit: bool = False, **kwargs) -> bool:
        backend: VisualizationBackend = VisualizationBackend.HVPLOT_MATPLOTLIB
        with optional_dependency(
                *VISUALIZATION_BACKEND_DEPENDENCIES[backend],
                error='ignore'
        ):
            import bokeh
            import holoviews as hv
            import hvplot.pandas
            if reinit or backend not in cls.initialized_backends:
                hv.extension('matplotlib')
            cls.initialized_backends.add(backend)
            return True
        return False

    def _postprocess_hvplot_matplotlib(self, figure: Any, **kwargs) -> Any:
        return figure

    @classmethod
    def _initialize_bokeh(cls, reinit: bool = False, **kwargs) -> bool:
        backend: VisualizationBackend = VisualizationBackend.BOKEH
        with optional_dependency(
                *VISUALIZATION_BACKEND_DEPENDENCIES[backend],
                error='ignore'
        ):
            import bokeh
            cls.initialized_backends.add(backend)
            return True
        return False

    def _postprocess_bokeh(self, figure: Any, **kwargs) -> Any:
        return figure

    @classmethod
    def _initialize_plotly_express(cls, reinit: bool = False, **kwargs) -> bool:
        backend: VisualizationBackend = VisualizationBackend.PLOTLY_EXPRESS
        with optional_dependency(
                *VISUALIZATION_BACKEND_DEPENDENCIES[backend],
                error='ignore'
        ):
            import hvplot.pandas
            import plotly.express as px
            import plotly.io as pio
            pio.templates.default = 'plotly_white'
            if reinit or backend not in cls.initialized_backends:
                hvplot.extension('plotly')
            cls.initialized_backends.add(backend)
            return True
        return False

    def _postprocess_plotly_express(self, figure: Any, **kwargs) -> Any:
        return figure

    @classmethod
    def _initialize_seaborn(cls, reinit: bool = False, **kwargs) -> bool:
        backend: VisualizationBackend = VisualizationBackend.SEABORN
        with optional_dependency(
                *VISUALIZATION_BACKEND_DEPENDENCIES[backend],
                error='ignore'
        ):
            import seaborn as sns
            return True
        return False

    def _postprocess_seaborn(self, figure: Any, **kwargs) -> Any:
        return figure

    @classmethod
    def _initialize_matplotlib(cls, reinit: bool = False, **kwargs) -> bool:
        backend: VisualizationBackend = VisualizationBackend.MATPLOTLIB
        with optional_dependency(
                *VISUALIZATION_BACKEND_DEPENDENCIES[backend],
                error='ignore'
        ):
            import matplotlib as plt
            cls.initialized_backends.add(backend)
            return True
        return False

    def _postprocess_matplotlib(self, figure: Any, **kwargs) -> Any:
        return figure

    @classmethod
    def _initialize_altair(cls, reinit: bool = False, **kwargs) -> bool:
        backend: VisualizationBackend = VisualizationBackend.ALTAIR
        with optional_dependency(
                *VISUALIZATION_BACKEND_DEPENDENCIES[backend],
                error='ignore'
        ):
            import altair as alt
            alt.renderers.enable('notebook')
            alt.data_transformers.disable_max_rows()
            cls.initialized_backends.add(backend)
            return True
        return False

    def _postprocess_altair(self, figure: Any, **kwargs) -> Any:
        return figure

    @classmethod
    def _initialize_termcolor(cls, reinit: bool = False, **kwargs) -> bool:
        backend: VisualizationBackend = VisualizationBackend.TERMCOLOR
        with optional_dependency(
                *VISUALIZATION_BACKEND_DEPENDENCIES[backend],
                error='ignore'
        ):
            import termcolor
            cls.initialized_backends.add(backend)
            return True
        return False

    def _postprocess_termcolor(self, figure: Any, **kwargs) -> Any:
        return figure

    @abstractmethod
    def plot_figure(self, data: Any, **kwargs) -> Any:
        pass


# class Visualization(Parameters, Registry, ABC):
#     _allow_multiple_subclasses = False
#     _allow_subclass_override = True





Viz = Visualization
visualize = Visualization.plot
