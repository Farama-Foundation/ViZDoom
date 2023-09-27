# Scenarios

This directory contains the files that define default ViZDoom scenarios/environments (Gymnasium/Open AI Gym nomenclature)nomenclature).
Usually, a scenario consists of two files - .wad and .cfg. The .wad file contains the map and script, and the .cfg file contains additional settings. The maps contained in .wad files (Doom's engine format for storing maps and assets) usually do not implement action constraints, the death penalty, and living rewards (however it is possible). To make it easier, this can be specified in ViZDoom .cfg files as well as other options like access to additional information.

You can read more about the default scenarios in the [documentation](https://vizdoom.farama.org/environments/default/). Some of these files are also used in the [examples](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python).
