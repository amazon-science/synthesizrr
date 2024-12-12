import warnings
from synthergent.base.framework.tracker.Tracker import *
from synthergent.base.framework.tracker.AimTracker import *
from synthergent.base.framework.tracker.LogFileTracker import *

DEFAULT_TRACKER: Optional[Tracker] = None

try:
    from synthergent.base.util.language import get_default
    from synthergent.base.util.jupyter import JupyterNotebook
    from synthergent.base.util.environment import EnvUtil

    if JupyterNotebook.is_notebook() and bool(get_default(EnvUtil.get_var('ENABLE_DEFAULT_TRACKER', False))):
        DEFAULT_TRACKER: Tracker = Tracker.default()
except Exception as e:
    warnings.warn(
        f'Cannot capture automatic logs using tracker: {DEFAULT_TRACKER_PARAMS}.'
        f'\nFollowing error was thrown: {str(e)}'
    )
