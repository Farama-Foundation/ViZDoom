# Tooling: building, scripting and inspecting a scenario

| Component | Role | Where it comes from | Ships with ViZDoom |
|---|---|---|---|
| ViZDoom engine + Python package | runs and tests the scenario | `pip install vizdoom` / editable install | yes |
| `freedoom2.wad` | IWAD: textures, flats, sprites, monsters, sounds | `vizdoom` package dir | yes |
| `vizdoom.pk3` | engine resources and built-in DECORATE actors | `vizdoom` package dir | yes |
| ACC 1.60 (`acc`, `zcommon.acs`, `zspecial.acs`, ...) | compiles ACS to the `BEHAVIOR` lump | `DoomGen2/acc/` in this skill (macOS universal `acc`, Windows `acc.exe`); upstream <https://github.com/rheit/acc> (build from source on Linux) | no (ships with the skill) |
| DoomGen | procedural map generator + Pythonic ACS builder | `DoomGen2/` in this skill, vendored from <https://github.com/TTomilin/COMRAD/tree/main/DoomGen2> | no (ships with the skill) |
| omgifol | read/write WAD lumps and maps from Python | `pip install omgifol` (DoomGen dependency) | no |
| SLADE / Ultimate Doom Builder | GUI inspection and hand editing (UDMF) | <https://slade.mancubus.net>, <https://ultimatedoombuilder.github.io> | no (optional) |

Find the shipped files with
`python -c "import vizdoom, os; print(os.path.dirname(vizdoom.__file__))"`.

## Route A: generate a map with DoomGen2

DoomGen2 builds Hexen-format maps and ACS scripts from Python. It is seedable and supports
parameter-grid batches via `BatchGenerator`. It discovers the bundled `acc` and `zdbsp`
automatically from any working directory (and skips `zdbsp` cleanly if unavailable on Linux):

```sh
PYTHONPATH=$SKILL/DoomGen2/src python $SKILL/DoomGen2/examples/simple_tag.py --audio --output /tmp/simple_tag_audio.wad
PYTHONPATH=$SKILL/DoomGen2/src python $SKILL/DoomGen2/examples/deathmatch_arena.py --batch-dir /tmp/arena_batch
```

See `DoomGen2/README.md` for the API and `DoomGen2/examples/README.md` for all five worked
examples. Run the DoomGen2 tests with `pytest $SKILL/DoomGen2/tests`.

## Route B: patch an existing WAD

When the geometry already exists, edit lumps in place with `omgifol` (`omg.WAD`,
`omg.UMapEditor` for UDMF, `omg.MapEditor` for Doom/Hexen format) or read/rewrite lumps
directly with `struct`: edit the map geometry or `TEXTMAP`, compile a standalone `.acs`
with `acc`, and replace the `BEHAVIOR` and `SCRIPTS` lumps.

## Compiling ACS with `acc`

```sh
$SKILL/DoomGen2/acc/acc -i $SKILL/DoomGen2/acc my_scenario.acs my_scenario.o   # -i: directory with zcommon.acs
```

The output object is the `BEHAVIOR` lump. Start scripts with `#include "zcommon.acs"`.
A clean compile proves only that ACC knows the names: ACC 1.60 also declares GZDoom,
Zandronum, Eternity and ZDaemon functions that ViZDoom does not run. Check every call
against `acs_functions.md` / `line_specials.md` and run `scripts/validate_scenario.py`.

## Route C: inspect or edit by hand in SLADE / Ultimate Doom Builder

Agents cannot use the GUI, but a human can open a generated WAD to check geometry, thing
placement and tags, or draw a map from scratch. Use the **ZDoom (UDMF)** map format, set
`freedoom2.wad` as the base resource and `vizdoom.pk3` as the ZDoom PK3, and point the
ACS compiler setting at `acc` (`docs/environments/slade_config.md`). No node builder is
needed. In SLADE choose `ZDoom` as the source port. Ultimate Doom Builder is Windows-first;
SLADE runs on Linux, macOS and Windows.
