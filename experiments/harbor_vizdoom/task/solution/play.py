"""Simple visual reference policy for Harbor's oracle agent."""

from __future__ import annotations

import asyncio
import io
import json
import os
from typing import Any

import numpy as np
from fastmcp import Client
from mcp.types import ImageContent, TextContent
from PIL import Image

MCP_URL = os.environ.get("VIZDOOM_MCP_URL", "http://vizdoom:8000/mcp")


def _tool_payload(result: Any) -> tuple[dict[str, Any], np.ndarray]:
    metadata: dict[str, Any] | None = result.structured_content
    image_bytes: bytes | None = None
    for block in result.content:
        if isinstance(block, TextContent) and metadata is None:
            _, _, raw_json = block.text.partition("\n")
            metadata = json.loads(raw_json)
        elif isinstance(block, ImageContent):
            import base64

            image_bytes = base64.b64decode(block.data)
    if metadata is None or image_bytes is None:
        raise RuntimeError("ViZDoom tool result lacked metadata or an image")
    image = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    return metadata, image


def _choose_action(image: np.ndarray) -> list[str]:
    red = image[..., 0].astype(int)
    green = image[..., 1].astype(int)
    blue = image[..., 2].astype(int)
    monster_pixels = (red > 100) & (red > green * 1.45) & (red > blue * 1.35)
    # Exclude the player's weapon and muzzle flash at the bottom of the view.
    monster_pixels[int(image.shape[0] * 0.68) :, :] = False
    ys, xs = np.nonzero(monster_pixels)
    if len(xs) < 8:
        return ["MOVE_LEFT"]

    center_x = float(np.median(xs))
    target_x = (image.shape[1] - 1) / 2
    tolerance = image.shape[1] * 0.035
    if center_x < target_x - tolerance:
        return ["MOVE_LEFT"]
    if center_x > target_x + tolerance:
        return ["MOVE_RIGHT"]
    return ["ATTACK"]


async def play() -> None:
    async with Client(MCP_URL) as client:
        result = await client.call_tool("get_current_state", {})
        metadata, image = _tool_payload(result)
        while not metadata["done"]:
            action = _choose_action(image)
            result = await client.call_tool("make_action", {"buttons": action})
            metadata, image = _tool_payload(result)
        if not metadata["success"]:
            raise RuntimeError(f"Reference policy failed: {metadata['status']}")


if __name__ == "__main__":
    asyncio.run(play())
