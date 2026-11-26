from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import os


class Scenario(ABC):
    """Base class for a generated DoomGen scenario."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: str = "scenario"):
        """Initialize the scenario config and validate it."""
        self.config = self.get_default_config()
        if config:
            self.config.update(config)
        self.name = name
        self.validate_config()

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Return the default configuration for the scenario."""
        pass

    @abstractmethod
    def generate(self, output_path: str) -> None:
        """Generate the scenario WAD at `output_path`."""
        pass

    def validate_config(self) -> None:
        """Validate the current configuration."""
        pass

    def get_param(self, key: str) -> Any:
        """Return a config parameter if present."""
        return self.config.get(key)
