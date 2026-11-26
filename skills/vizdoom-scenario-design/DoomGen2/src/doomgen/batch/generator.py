from typing import Any, Dict, List, Type, Iterator
import itertools
from copy import deepcopy
from .scenario import Scenario


class BatchGenerator:
    """Generates scenario instances from a parameter grid."""

    def __init__(self, scenario_cls: Type[Scenario], param_grid: Dict[str, List[Any]]):
        """Create a batch generator for `scenario_cls` and `param_grid`."""
        self.scenario_cls = scenario_cls
        self.param_grid = param_grid

    def generate_configs(self) -> Iterator[Dict[str, Any]]:
        """Yield one config for each cartesian-product combination."""
        keys = self.param_grid.keys()
        values = self.param_grid.values()

        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

    def generate_scenarios(self, base_name: str = "batch") -> Iterator[Scenario]:
        """Yield initialized scenarios for each generated config."""
        for i, config in enumerate(self.generate_configs()):
            name = f"{base_name}_{i}"
            scenario = self.scenario_cls(config, name=name)
            yield scenario

    def count_combinations(self) -> int:
        """Returns the total number of combinations."""
        count = 1
        for v in self.param_grid.values():
            count *= len(v)
        return count
