from typing import Tuple, Dict

import numpy as np
from vizdoom import ScreenResolution

RESOLUTIONS: Dict[str, ScreenResolution] = {
    "1920x1080": ScreenResolution.RES_1920X1080,
    "1600x1200": ScreenResolution.RES_1600X1200,
    "1280x720": ScreenResolution.RES_1280X720,
    "800x600": ScreenResolution.RES_800X600,
    "640x480": ScreenResolution.RES_640X480,
    "320x240": ScreenResolution.RES_320X240,
    "160x120": ScreenResolution.RES_160X120,
}


def parse_hw(res: str) -> Tuple[int, int]:
    w, h = res.lower().split("x")
    return int(w), int(h)


def screen_res(res: str) -> ScreenResolution:
    if res not in RESOLUTIONS:
        raise ValueError(f"Invalid resolution: {res}")
    return RESOLUTIONS[res]


def get_flat_game_vars(state, available_game_vars) -> Dict[str, float]:
    """Return game variables as flat scalars suitable for info (no nested dict)."""
    if state is None or state.game_variables is None:
        return {}
    game_variables = state.game_variables
    out: Dict[str, float] = {}
    n = min(len(available_game_vars), len(game_variables))
    for i in range(n):
        name = available_game_vars[i].name
        val = game_variables[i]
        out[name] = float(val)
    return out


def read_frame(state, resolution) -> np.ndarray:
    if state is not None and state.screen_buffer is not None:
        sb = state.screen_buffer  # (C,H,W) or (H,W)
        return np.transpose(sb, (1, 2, 0)) if sb.ndim == 3 else sb[..., None]
    # fallback to zeros if no frame
    h, w = parse_hw(resolution)[1], parse_hw(resolution)[0]
    return np.zeros((h, w, 3), dtype=np.uint8)
