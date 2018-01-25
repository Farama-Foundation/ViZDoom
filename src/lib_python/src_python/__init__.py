from .vizdoom import __version__ as __version__
from .vizdoom import *

import os as _os

scenarios_path = _os.path.join(__path__[0], "scenarios")
wads = [wad for wad in sorted(_os.listdir(scenarios_path)) if wad.endswith(".wad")]
configs = [cfg for cfg in sorted(_os.listdir(scenarios_path)) if cfg.endswith(".cfg")]
