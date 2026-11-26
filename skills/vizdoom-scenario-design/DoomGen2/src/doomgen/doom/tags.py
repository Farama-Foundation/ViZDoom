"""Unique sector tag allocation."""

from __future__ import annotations

from typing import Optional


class TagRegistry:
    """Allocates unique Doom sector tags and optional names."""

    def __init__(self, start_tag: int = 1):
        """Create a registry starting at `start_tag`."""
        self._next_tag = start_tag
        self._used_tags: set[int] = set()
        self._named_tags: dict[str, int] = {}

    def allocate(self, name: Optional[str] = None) -> int:
        """Allocate a new tag and optionally bind it to `name`."""
        tag = self._next_tag
        self._next_tag += 1
        self._used_tags.add(tag)

        if name is not None:
            self._named_tags[name] = tag

        return tag

    def reserve(self, tag: int, name: Optional[str] = None) -> int:
        """Reserve a specific tag value."""
        if tag in self._used_tags:
            raise ValueError(f"Tag {tag} is already in use")

        self._used_tags.add(tag)

        if name is not None:
            self._named_tags[name] = tag

        if tag >= self._next_tag:
            self._next_tag = tag + 1

        return tag

    def get(self, name: str) -> int:
        """Return a previously named tag."""
        if name not in self._named_tags:
            raise KeyError(f"No tag named '{name}'")
        return self._named_tags[name]

    def get_or_allocate(self, name: str) -> int:
        """Return an existing named tag or allocate a new one."""
        if name in self._named_tags:
            return self._named_tags[name]
        return self.allocate(name)

    def is_used(self, tag: int) -> bool:
        """Return whether `tag` has already been allocated."""
        return tag in self._used_tags

    def list_named_tags(self) -> dict[str, int]:
        """Return a copy of the named-tag mapping."""
        return self._named_tags.copy()

    def reset(self, start_tag: int = 1) -> None:
        """Reset the registry and start allocating from `start_tag`."""
        self._next_tag = start_tag
        self._used_tags.clear()
        self._named_tags.clear()
