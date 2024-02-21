import warnings
from synthesizrr.base.framework.tracker.Tracker import *
from synthesizrr.base.framework.tracker.AimTracker import *
from synthesizrr.base.framework.tracker.LogFileTracker import *

DEFAULT_TRACKER: Optional[Tracker] = None

try:
    from synthesizrr.base.util.language import get_default
    from synthesizrr.base.util.jupyter import JupyterNotebook
    from synthesizrr.base.util.environment import EnvUtil

    if JupyterNotebook.is_notebook() and bool(get_default(EnvUtil.get_var('ENABLE_DEFAULT_TRACKER', False))):
        DEFAULT_TRACKER: Tracker = Tracker.default()
except Exception as e:
    warnings.warn(
        f'Cannot capture automatic logs using tracker: {DEFAULT_TRACKER_PARAMS}.'
        f'\nFollowing error was thrown: {str(e)}'
    )
