from typing import Tuple, Dict

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
