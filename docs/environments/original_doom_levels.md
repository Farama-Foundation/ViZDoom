# Playing original Doom levels

Beside the custom scenarios (original ViZDoom nomenclature)/environments (Gymnasium/Open AI Gym nomenclature) original introduced with ViZDoom and many user-created. It is possible to play the original Doom/Doom 2/Freedoom/Freedoom 2 levels as well.

Because we cannot provide original Doom's and Doom's 2 levels, in order to play them you need to have original Doom or Doom 2 WAD files.
You can get them by purchasing the original game from [Steam](https://store.steampowered.com/app/2280/DOOM__DOOM_II/) or [GOG](https://www.gog.com/game/doom_doom_ii).
You can then place the doom2.wad and doom.wad files into your vizdoom package directory (same directory as vizdoom(.exe)).

## Convention of environment names in Gymnasium

The naming convention of original Doom levels environments is as follows:

`"Vizdoom<Game><Map>-S<X>-v0"`.

Where `<Game>` is Freedoom, Freedoom2, Doom or Doom2, and S<X> is a skill level, with X being number from 1 to 5:
- `S1` - VERY EASY, “I'm Too Young to Die”
- `S2` - EASY, “Hey, Not Too Rough"
- `S3` - NORMAL, “Hurt Me Plenty”
- `S4` - HARD, “Ultra-Violence”
- `S5` - VERY HARD, “Nightmare!”

For example:
- `"VizdoomDoomE1M1-S1-v0"` environment uses original Doom 1 E1M1 and skill (difficulty) level 1 (VERY EASY, “I'm Too Young to Die”)
- `"VizdoomDoom2MAP01-S3-v0"` environment uses original Doom 2 MAP01 level with skill level 3


## Using original Doom levels with Original ViZDoom API

To play original Doom levels using the original ViZDoom API, you can use the following configuration file as a base:

Python example:
```{code-block} python
import os
import vizdoom as vzd
game = vzd.DoomGame()
game.load_config(os.path.join(vzd.scenarios_path, "doom.cfg")) # or doom2.cfg, freedoom1.cfg, freedoom2.cfg
game.set_doom_map("E1M1")  # see list of map IDs below
game.set_skill_level(1)  # or 2, 3, 4, 5
```

Configuration file:
- Doom: [doom.cfg](https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios/doom.cfg)
- Doom 2: [doom2.cfg](https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios/doom2.cfg)
- Freedoom: [freedoom1.cfg](https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios/freedoom.cfg)
- Freedoom 2: [freedoom2.cfg](https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios/freedoom2.cfg)


## Original Doom 1 levels

The list of all original Doom 1 levels and their corresponding ViZDoom environment names:

### Episode 1: Knee-Deep in the Dead

| Map ID   | Gymnasium Environment Name  | Level Name                                      |
| :--      | :--                         | :--                                             |
| `"E1M1"` | `"VizdoomDoomE1M1-S<X>-v0"` | Doom 1 E1M1: Hangar                             |
| `"E1M2"` | `"VizdoomDoomE1M2-S<X>-v0"` | Doom 1 E1M2: Nuclear Plant                      |
| `"E1M3"` | `"VizdoomDoomE1M3-S<X>-v0"` | Doom 1 E1M3: Toxin Refinery                     |
| `"E1M4"` | `"VizdoomDoomE1M4-S<X>-v0"` | Doom 1 E1M4: Command Control                    |
| `"E1M5"` | `"VizdoomDoomE1M5-S<X>-v0"` | Doom 1 E1M5: Phobos Lab                         |
| `"E1M6"` | `"VizdoomDoomE1M6-S<X>-v0"` | Doom 1 E1M6: Central Processing                 |
| `"E1M7"` | `"VizdoomDoomE1M7-S<X>-v0"` | Doom 1 E1M7: Computer Station                   |
| `"E1M8"` | `"VizdoomDoomE1M8-S<X>-v0"` | Doom 1 E1M8: Phobos Anomaly                     |
| `"E1M9"` | `"VizdoomDoomE1M9-S<X>-v0"` | Doom 1 E1M9: Military Base (Secret level)       |

### Episode 2: The Shores of Hell

| Map ID   | Gymnasium Environment Name  | Level Name                                      |
| :--      | :--                         | :--                                             |
| `"E2M1"` | `"VizdoomDoomE2M1-S<X>-v0"` | Doom 1 E2M1: Deimos Anomaly                     |
| `"E2M2"` | `"VizdoomDoomE2M2-S<X>-v0"` | Doom 1 E2M2: Containment Area                   |
| `"E2M3"` | `"VizdoomDoomE2M3-S<X>-v0"` | Doom 1 E2M3: Refinery                           |
| `"E2M4"` | `"VizdoomDoomE2M4-S<X>-v0"` | Doom 1 E2M4: Deimos Lab                         |
| `"E2M5"` | `"VizdoomDoomE2M5-S<X>-v0"` | Doom 1 E2M5: Command Center                     |
| `"E2M6"` | `"VizdoomDoomE2M6-S<X>-v0"` | Doom 1 E2M6: Halls of the Damned                |
| `"E2M7"` | `"VizdoomDoomE2M7-S<X>-v0"` | Doom 1 E2M7: Spawning Vats                      |
| `"E2M8"` | `"VizdoomDoomE2M8-S<X>-v0"` | Doom 1 E2M8: Tower of Babel                     |
| `"E2M9"` | `"VizdoomDoomE2M9-S<X>-v0"` | Doom 1 E2M9: Fortress of Mystery (Secret level) |


### Episode 3: Inferno

| Map ID   | Gymnasium Environment Name  | Level Name                                      |
| :--      | :--                         | :--                                             |
| `"E3M1"` | `"VizdoomDoomE3M1-S<X>-v0"` | Doom 1 E3M1: Hell Keep                          |
| `"E3M2"` | `"VizdoomDoomE3M2-S<X>-v0"` | Doom 1 E3M2: Slough of Despair                  |
| `"E3M3"` | `"VizdoomDoomE3M3-S<X>-v0"` | Doom 1 E3M3: Pandemonium                        |
| `"E3M4"` | `"VizdoomDoomE3M4-S<X>-v0"` | Doom 1 E3M4: House of Pain                      |
| `"E3M5"` | `"VizdoomDoomE3M5-S<X>-v0"` | Doom 1 E3M5: Unholy Cathedral                   |
| `"E3M6"` | `"VizdoomDoomE3M6-S<X>-v0"` | Doom 1 E3M6: Mt. Erebus                         |
| `"E3M7"` | `"VizdoomDoomE3M7-S<X>-v0"` | Doom 1 E3M7: Limbo                              |
| `"E3M8"` | `"VizdoomDoomE3M8-S<X>-v0"` | Doom 1 E3M8: Dis                                |
| `"E3M9"` | `"VizdoomDoomE3M9-S<X>-v0"` | Doom 1 E3M9: Warrens (Secret level)             |


### Episode 4: Thy Flesh Consumed

| Map ID   | Gymnasium Environment Name  | Level Name                                      |
| :--      | :--                         | :--                                             |
| `"E4M1"` | `"VizdoomDoomE4M1-S<X>-v0"` | Doom 1 E4M1: Hell Beneath                       |
| `"E4M2"` | `"VizdoomDoomE4M2-S<X>-v0"` | Doom 1 E4M2: Perfect Hatred                     |
| `"E4M3"` | `"VizdoomDoomE4M3-S<X>-v0"` | Doom 1 E4M3: Sever the Wicked                   |
| `"E4M4"` | `"VizdoomDoomE4M4-S<X>-v0"` | Doom 1 E4M4: Unruly Evil                        |
| `"E4M5"` | `"VizdoomDoomE4M5-S<X>-v0"` | Doom 1 E4M5: They Will Repent                   |
| `"E4M6"` | `"VizdoomDoomE4M6-S<X>-v0"` | Doom 1 E4M6: Against Thee Wickedly              |
| `"E4M7"` | `"VizdoomDoomE4M7-S<X>-v0"` | Doom 1 E4M7: And Hell Followed                  |
| `"E4M8"` | `"VizdoomDoomE4M8-S<X>-v0"` | Doom 1 E4M8: Unto the Cruel                     |
| `"E4M9"` | `"VizdoomDoomE4M9-S<X>-v0"` | Doom 1 E4M9: Fear (Secret level)                |


## Original Doom 2 levels

The list of all original Doom 2 levels and their corresponding ViZDoom environment names:

### Episode 1: The Space Station

| Map ID    | Gymnasium Environment Name    | Level Name                                   |
| :--       | :--                           | :--                                          |
| `"MAP01"` | `"VizdoomDoom2MAP01-S<X>-v0"` | Doom 2 MAP01: Entryway                       |
| `"MAP02"` | `"VizdoomDoom2MAP02-S<X>-v0"` | Doom 2 MAP02: Underhalls                     |
| `"MAP03"` | `"VizdoomDoom2MAP03-S<X>-v0"` | Doom 2 MAP03: The Gantlet                    |
| `"MAP04"` | `"VizdoomDoom2MAP04-S<X>-v0"` | Doom 2 MAP04: The Focus                      |
| `"MAP05"` | `"VizdoomDoom2MAP05-S<X>-v0"` | Doom 2 MAP05: The Waste Tunnels              |
| `"MAP06"` | `"VizdoomDoom2MAP06-S<X>-v0"` | Doom 2 MAP06: The Crusher                    |
| `"MAP07"` | `"VizdoomDoom2MAP07-S<X>-v0"` | Doom 2 MAP07: Dead Simple                    |
| `"MAP08"` | `"VizdoomDoom2MAP08-S<X>-v0"` | Doom 2 MAP08: Tricks and Traps               |
| `"MAP09"` | `"VizdoomDoom2MAP09-S<X>-v0"` | Doom 2 MAP09: The Pit                        |
| `"MAP10"` | `"VizdoomDoom2MAP10-S<X>-v0"` | Doom 2 MAP10: Refueling Base                 |
| `"MAP11"` | `"VizdoomDoom2MAP11-S<X>-v0"` | Doom 2 MAP11: 'O' of Destruction!            |


### Episode 2: The City

| Map ID    | Gymnasium Environment Name    | Level Name                                   |
| :--       | :--                           | :--                                          |
| `"MAP12"` | `"VizdoomDoom2MAP12-S<X>-v0"` | Doom 2 MAP12: The Factory                    |
| `"MAP13"` | `"VizdoomDoom2MAP13-S<X>-v0"` | Doom 2 MAP13: Downtown                       |
| `"MAP14"` | `"VizdoomDoom2MAP14-S<X>-v0"` | Doom 2 MAP14: The Inmost Dens                |
| `"MAP15"` | `"VizdoomDoom2MAP15-S<X>-v0"` | Doom 2 MAP15: Industrial Zone                |
| `"MAP16"` | `"VizdoomDoom2MAP16-S<X>-v0"` | Doom 2 MAP16: Suburbs                        |
| `"MAP17"` | `"VizdoomDoom2MAP17-S<X>-v0"` | Doom 2 MAP17: Tenements                      |
| `"MAP18"` | `"VizdoomDoom2MAP18-S<X>-v0"` | Doom 2 MAP18: The Courtyard                  |
| `"MAP19"` | `"VizdoomDoom2MAP19-S<X>-v0"` | Doom 2 MAP19: The Citadel                    |
| `"MAP20"` | `"VizdoomDoom2MAP20-S<X>-v0"` | Doom 2 MAP20: Gotcha!                        |

### Episode 3: Hell

| Map ID    | Gymnasium Environment Name    | Level Name                                   |
| :--       | :--                           | :--                                          |
| `"MAP21"` | `"VizdoomDoom2MAP21-S<X>-v0"` | Doom 2 MAP21: Nirvana                        |
| `"MAP22"` | `"VizdoomDoom2MAP22-S<X>-v0"` | Doom 2 MAP22: The Catacombs                  |
| `"MAP23"` | `"VizdoomDoom2MAP23-S<X>-v0"` | Doom 2 MAP23: Barrels o' Fun                 |
| `"MAP24"` | `"VizdoomDoom2MAP24-S<X>-v0"` | Doom 2 MAP24: The Chasm                      |
| `"MAP25"` | `"VizdoomDoom2MAP25-S<X>-v0"` | Doom 2 MAP25: Bloodfalls                     |
| `"MAP26"` | `"VizdoomDoom2MAP26-S<X>-v0"` | Doom 2 MAP26: The Abandoned Mines            |
| `"MAP27"` | `"VizdoomDoom2MAP27-S<X>-v0"` | Doom 2 MAP27: Monster Condo                  |
| `"MAP28"` | `"VizdoomDoom2MAP28-S<X>-v0"` | Doom 2 MAP28: The Spirit World               |
| `"MAP29"` | `"VizdoomDoom2MAP29-S<X>-v0"` | Doom 2 MAP29: The Living End                 |
| `"MAP30"` | `"VizdoomDoom2MAP30-S<X>-v0"` | Doom 2 MAP30: Icon of Sin                    |


### Secret levels:

| Map ID    | Gymnasium Environment Name    | Level Name                                   |
| :--       | :--                           | :--                                          |
| `"MAP31"` | `"VizdoomDoom2MAP31-S<X>-v0"` | Doom 2 MAP31: Wolfenstein2                   |
| `"MAP32"` | `"VizdoomDoom2MAP32-S<X>-v0"` | Doom 2 MAP32: Grosse2                        |


## Freedoom 1 levels

The list of all Freedoom 1 (Freedoom: Phase 1) levels and their corresponding ViZDoom environment names:

### Episode 1: Outpost Outbreak

| Map ID   | Gymnasium Environment Name       | Level Name                                 |
| :--      | :--                              | :--                                        |
| `"E1M1"` | `"VizdoomFreedoom1E1M1-S<X>-v0"` | Freedoom E1M1: Outer Prison                |
| `"E1M2"` | `"VizdoomFreedoom1E1M2-S<X>-v0"` | Freedoom E1M2: Communications Center       |
| `"E1M3"` | `"VizdoomFreedoom1E1M3-S<X>-v0"` | Freedoom E1M3: Waste Disposal              |
| `"E1M4"` | `"VizdoomFreedoom1E1M4-S<X>-v0"` | Freedoom E1M4: Supply Depot                |
| `"E1M5"` | `"VizdoomFreedoom1E1M5-S<X>-v0"` | Freedoom E1M5: Armory                      |
| `"E1M6"` | `"VizdoomFreedoom1E1M6-S<X>-v0"` | Freedoom E1M6: Training Facility           |
| `"E1M7"` | `"VizdoomFreedoom1E1M7-S<X>-v0"` | Freedoom E1M7: Xenobiotic Materials Lab    |
| `"E1M8"` | `"VizdoomFreedoom1E1M8-S<X>-v0"` | Freedoom E1M8: Outpost Quarry              |
| `"E1M9"` | `"VizdoomFreedoom1E1M9-S<X>-v0"` | Freedoom E1M9: Nutrient Recycling          |


### Episode 2: Military Labs

| Map ID   | Gymnasium Environment Name       | Level Name                                 |
| :--      | :--                              | :--                                        |
| `"E2M1"` | `"VizdoomFreedoom1E2M1-S<X>-v0"` | Freedoom E2M1: Elemental Gate              |
| `"E2M2"` | `"VizdoomFreedoom1E2M2-S<X>-v0"` | Freedoom E2M2: Shifter                     |
| `"E2M3"` | `"VizdoomFreedoom1E2M3-S<X>-v0"` | Freedoom E2M3: Reclaimed Facilities        |
| `"E2M4"` | `"VizdoomFreedoom1E2M4-S<X>-v0"` | Freedoom E2M4: Flooded Installation        |
| `"E2M5"` | `"VizdoomFreedoom1E2M5-S<X>-v0"` | Freedoom E2M5: Underground Hub             |
| `"E2M6"` | `"VizdoomFreedoom1E2M6-S<X>-v0"` | Freedoom E2M6: Hidden Sector               |
| `"E2M7"` | `"VizdoomFreedoom1E2M7-S<X>-v0"` | Freedoom E2M7: Control Complex             |
| `"E2M8"` | `"VizdoomFreedoom1E2M8-S<X>-v0"` | Freedoom E2M8: Containment Cell            |
| `"E2M9"` | `"VizdoomFreedoom1E2M9-S<X>-v0"` | Freedoom E2M9: Fortress 31                 |


### Episode 3: Event Horizon

| Map ID   | Gymnasium Environment Name       | Level Name                                 |
| :--      | :--                              | :--                                        |
| `"E3M1"` | `"VizdoomFreedoom1E3M1-S<X>-v0"` | Freedoom E3M1: Land of the Lost            |
| `"E3M2"` | `"VizdoomFreedoom1E3M2-S<X>-v0"` | Freedoom E3M2: Geothermal Tunnels          |
| `"E3M3"` | `"VizdoomFreedoom1E3M3-S<X>-v0"` | Freedoom E3M3: Sacrificial Bastion         |
| `"E3M4"` | `"VizdoomFreedoom1E3M4-S<X>-v0"` | Freedoom E3M4: Oblation Temple             |
| `"E3M5"` | `"VizdoomFreedoom1E3M5-S<X>-v0"` | Freedoom E3M5: Infernal Hallows            |
| `"E3M6"` | `"VizdoomFreedoom1E3M6-S<X>-v0"` | Freedoom E3M6: Igneous Intrusion           |
| `"E3M7"` | `"VizdoomFreedoom1E3M7-S<X>-v0"` | Freedoom E3M7: No Regrets                  |
| `"E3M8"` | `"VizdoomFreedoom1E3M8-S<X>-v0"` | Freedoom E3M8: Ancient Lair                |
| `"E3M9"` | `"VizdoomFreedoom1E3M9-S<X>-v0"` | Freedoom E3M9: Acquainted With Grief       |


### Episode 4: Double Impact

| Map ID   | Gymnasium Environment Name       | Level Name                                 |
| :--      | :--                              | :--                                        |
| `"E4M1"` | `"VizdoomFreedoom1E4M1-S<X>-v0"` | Freedoom E4M1: Maintenance Area            |
| `"E4M2"` | `"VizdoomFreedoom1E4M2-S<X>-v0"` | Freedoom E4M2: Research Complex            |
| `"E4M3"` | `"VizdoomFreedoom1E4M3-S<X>-v0"` | Freedoom E4M3: Central Computing           |
| `"E4M4"` | `"VizdoomFreedoom1E4M4-S<X>-v0"` | Freedoom E4M4: Hydroponic Facility         |
| `"E4M5"` | `"VizdoomFreedoom1E4M5-S<X>-v0"` | Freedoom E4M5: Engineering Station         |
| `"E4M6"` | `"VizdoomFreedoom1E4M6-S<X>-v0"` | Freedoom E4M6: Command Center              |
| `"E4M7"` | `"VizdoomFreedoom1E4M7-S<X>-v0"` | Freedoom E4M7: Waste Treatment             |
| `"E4M8"` | `"VizdoomFreedoom1E4M8-S<X>-v0"` | Freedoom E4M8: Launch Bay                  |
| `"E4M9"` | `"VizdoomFreedoom1E4M9-S<X>-v0"` | Freedoom E4M9: Operations                  |


## Freedoom 2 (Freedoom: Phase 2) levels

The list of all Freedoom 2 (Freedoom: Phase 2) levels and their corresponding ViZDoom environment names:

| Map ID    | Gymnasium Environment Name        | Level Name                                     |
| :--       | :--                               | :--                                            |
| `"MAP01"` | `"VizdoomFreedoom2MAP01-S<X>-v0"` | Freedoom 2 MAP01: Hydroelectric Plant          |
| `"MAP02"` | `"VizdoomFreedoom2MAP02-S<X>-v0"` | Freedoom 2 MAP02: Filtration Tunnels           |
| `"MAP03"` | `"VizdoomFreedoom2MAP03-S<X>-v0"` | Freedoom 2 MAP03: Crude Processing Center      |
| `"MAP04"` | `"VizdoomFreedoom2MAP04-S<X>-v0"` | Freedoom 2 MAP04: Containment Bay              |
| `"MAP05"` | `"VizdoomFreedoom2MAP05-S<X>-v0"` | Freedoom 2 MAP05: Sludge Burrow                |
| `"MAP06"` | `"VizdoomFreedoom2MAP06-S<X>-v0"` | Freedoom 2 MAP06: Janus Terminal               |
| `"MAP07"` | `"VizdoomFreedoom2MAP07-S<X>-v0"` | Freedoom 2 MAP07: Logic Gate                   |
| `"MAP08"` | `"VizdoomFreedoom2MAP08-S<X>-v0"` | Freedoom 2 MAP08: Astronomy Complex            |
| `"MAP09"` | `"VizdoomFreedoom2MAP09-S<X>-v0"` | Freedoom 2 MAP09: Datacenter                   |
| `"MAP10"` | `"VizdoomFreedoom2MAP10-S<X>-v0"` | Freedoom 2 MAP10: Deadly Outlands              |
| `"MAP11"` | `"VizdoomFreedoom2MAP11-S<X>-v0"` | Freedoom 2 MAP11: Dimensional Rift Observatory |
| `"MAP12"` | `"VizdoomFreedoom2MAP12-S<X>-v0"` | Freedoom 2 MAP12: Railroads                    |
| `"MAP13"` | `"VizdoomFreedoom2MAP13-S<X>-v0"` | Freedoom 2 MAP13: Station Earth                |
| `"MAP14"` | `"VizdoomFreedoom2MAP14-S<X>-v0"` | Freedoom 2 MAP14: Nuclear Zone                 |
| `"MAP15"` | `"VizdoomFreedoom2MAP15-S<X>-v0"` | Freedoom 2 MAP15: Hostile Takeover             |
| `"MAP16"` | `"VizdoomFreedoom2MAP16-S<X>-v0"` | Freedoom 2 MAP16: Urban Jungle                 |
| `"MAP17"` | `"VizdoomFreedoom2MAP17-S<X>-v0"` | Freedoom 2 MAP17: City Capitol                 |
| `"MAP18"` | `"VizdoomFreedoom2MAP18-S<X>-v0"` | Freedoom 2 MAP18: Aquatics Lab                 |
| `"MAP19"` | `"VizdoomFreedoom2MAP19-S<X>-v0"` | Freedoom 2 MAP19: Sewage Control               |
| `"MAP20"` | `"VizdoomFreedoom2MAP20-S<X>-v0"` | Freedoom 2 MAP20: Blood Ember Fortress         |
| `"MAP21"` | `"VizdoomFreedoom2MAP21-S<X>-v0"` | Freedoom 2 MAP21: Under Realm                  |
| `"MAP22"` | `"VizdoomFreedoom2MAP22-S<X>-v0"` | Freedoom 2 MAP22: Remanasu                     |
| `"MAP23"` | `"VizdoomFreedoom2MAP23-S<X>-v0"` | Freedoom 2 MAP23: Underground Facility         |
| `"MAP24"` | `"VizdoomFreedoom2MAP24-S<X>-v0"` | Freedoom 2 MAP24: Abandoned Teleporter Lab     |
| `"MAP25"` | `"VizdoomFreedoom2MAP25-S<X>-v0"` | Freedoom 2 MAP25: Persistence of Memory        |
| `"MAP26"` | `"VizdoomFreedoom2MAP26-S<X>-v0"` | Freedoom 2 MAP26: Dark Depths                  |
| `"MAP27"` | `"VizdoomFreedoom2MAP27-S<X>-v0"` | Freedoom 2 MAP27: Palace of Red                |
| `"MAP28"` | `"VizdoomFreedoom2MAP28-S<X>-v0"` | Freedoom 2 MAP28: Grim Redoubt                 |
| `"MAP29"` | `"VizdoomFreedoom2MAP29-S<X>-v0"` | Freedoom 2 MAP29: Melting Point                |
| `"MAP30"` | `"VizdoomFreedoom2MAP30-S<X>-v0"` | Freedoom 2 MAP30: Jaws of Defeat               |
| `"MAP31"` | `"VizdoomFreedoom2MAP31-S<X>-v0"` | Freedoom 2 MAP31: Be Quiet (secret level)      |
| `"MAP32"` | `"VizdoomFreedoom2MAP32-S<X>-v0"` | Freedoom 2 MAP32: Not Sure (secret level)      |
