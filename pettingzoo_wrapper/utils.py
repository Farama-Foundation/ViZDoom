from typing import Tuple, Dict

import numpy as np
import vizdoom as vzd


def parse_hw(res: str) -> Tuple[int, int]:
    w, h = res.lower().split("x")
    return int(w), int(h)


def get_screen_resolution(resolution: str) -> vzd.ScreenResolution:
    try:
        return getattr(vzd.ScreenResolution, f"RES_{resolution}")
    except AttributeError as e:
        raise ValueError(f"Invalid resolution: {resolution}")


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


def discover_buttons(config_path: str) -> Tuple[int, int]:
    game = vzd.DoomGame()
    game.load_config(config_path)
    game.set_window_visible(False)
    delta, binary = [], []
    for b in game.get_available_buttons():
        if vzd.is_delta_button(b) and b not in delta:
            delta.append(b)
        else:
            binary.append(b)
    game.close()
    return len(delta), len(binary)