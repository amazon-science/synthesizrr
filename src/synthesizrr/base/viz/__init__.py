from importlib import import_module
import os
from pathlib import Path
from synthesizrr.base.util.string import StringUtil

__THIS_FILE__ = __file__  ## Needed when calling reload() from outside this file.


def reload():
    ## Imports all .py files in this dir to register visuals. Ref: https://stackoverflow.com/a/59054776
    for f in Path(__THIS_FILE__).parent.glob("**/*.py"):
        if str(f) == __THIS_FILE__:
            continue
        ## E.g. ".sklearn"
        import_dir: str = '.'.join(
            StringUtil.remove_prefix(
                str(f),
                prefix=os.path.commonpath([__THIS_FILE__, str(f)])
            ).split(os.path.sep)[:-1])
        import_path: str = f"{import_dir}.{f.stem}"
        if 'ipynb_checkpoints' in import_path:
            continue
        import_module(import_path, __package__)


reload()
