import os
import sys
import json
import logging

# Ensure project root and src are in path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

from doomgen.batch import BatchGenerator
from examples.benchmark.armory_siege import ArmorySiegeScenario


SCENARIO_CLS = ArmorySiegeScenario
EXPECTED_COUNT = 162
# Axis names match config keys; option order defines the generation order.
AXIS_OPTIONS = {
    "distance": [
        {"_label": "close", "distance": 600},
        {"_label": "mid", "distance": 800},
        {"_label": "far", "distance": 1100},
    ],
    "corridor_width": [
        {"_label": "wide", "corridor_width": 256},
        {"_label": "tight", "corridor_width": 128},
    ],
    "door_timer": [
        {"_label": "long", "door_timer": 600},
        {"_label": "standard", "door_timer": 250},
        {"_label": "brief", "door_timer": 100},
    ],
    "core_health": [
        {"_label": "durable", "core_health": 1500},
        {"_label": "baseline", "core_health": 800},
        {"_label": "fragile", "core_health": 300},
    ],
    "enemy_difficulty": [
        {"_label": "light", "enemy_difficulty": 0.25},
        {"_label": "standard", "enemy_difficulty": 0.5},
        {"_label": "heavy", "enemy_difficulty": 1},
    ],
}


def main():
    print("Initializing Batch Generation for Armory Siege...")

    param_grid = {
        key: [option[key] for option in options]
        for key, options in AXIS_OPTIONS.items()
    }

    demo_grid = {
        'distance': [600, 800, 1100],
        'corridor_width': [192],
        'door_timer': [600, 250],
        'core_health': [1500, 500],
        'enemy_difficulty': [0.5, 1.25]
    }

    generator = BatchGenerator(ArmorySiegeScenario, param_grid=param_grid)

    print(f"Total scenarios to generate: {generator.count_combinations()}")

    output_dir = "examples/benchmark/output/batch_armory"
    os.makedirs(output_dir, exist_ok=True)

    batch_metadata = []

    for i, scenario in enumerate(generator.generate_scenarios(base_name="armory")):
        filename = f"{scenario.name}.wad"
        path = os.path.join(output_dir, filename)

        print(f"[{i+1}/{generator.count_combinations()}] Generating {filename}...")
        try:
            scenario.generate(path)

            # Record metadata
            meta = {
                'id': scenario.name,
                'filename': filename,
                'config': scenario.config
            }
            batch_metadata.append(meta)

        except Exception as e:
            print(f"Failed to generate {filename}: {e}")
            # traceback.print_exc()

    #Central JSON
    central_json_path = os.path.join(output_dir, "batch_registry.json")
    with open(central_json_path, 'w') as f:
        json.dump(batch_metadata, f, indent=4)

    print(f"Batch generation complete. Registry saved to {central_json_path}")

if __name__ == "__main__":
    main()
