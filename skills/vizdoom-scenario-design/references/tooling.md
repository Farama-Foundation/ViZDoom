# Tooling: building, scripting and inspecting a scenario

| Component | Role | Where it comes from | Ships with ViZDoom |
|---|---|---|---|
| ViZDoom engine + Python package | runs and tests the scenario | `pip install vizdoom` / editable install | yes |
| `freedoom2.wad` | IWAD: textures, flats, sprites, monsters, sounds | `vizdoom` package dir | yes |
| `vizdoom.pk3` | engine resources and built-in DECORATE actors | `vizdoom` package dir | yes |
| ACC 1.60 (`acc`, `zcommon.acs`, `zspecial.acs`, ...) | compiles ACS to the `BEHAVIOR` lump | COMRAD `DoomGen/acc/` (macOS universal `acc`, Windows `acc.exe`); upstream <https://github.com/rheit/acc> (build from source on Linux) | no |
| DoomGen | procedural map generator + Pythonic ACS builder | COMRAD `DoomGen2/` (<https://github.com/TTomilin/COMRAD/tree/main/DoomGen2>) | no |
| omgifol | read/write WAD lumps and maps from Python | `pip install omgifol` (DoomGen dependency) | no |
| SLADE / Ultimate Doom Builder | GUI inspection and hand editing (UDMF) | <https://slade.mancubus.net>, <https://ultimatedoombuilder.github.io> | no (optional) |

Find the shipped files with
`python -c "import vizdoom, os; print(os.path.dirname(vizdoom.__file__))"`.

## Route A: generate a map with DoomGen

If DoomGen is not installed, get it at `https://github.com/TTomilin/COMRAD/tree/main/DoomGen2`. DoomGen builds maps and ACS scripts from a Python file. It is seedable, so one script can emit a batch of layout variants.

Run command:
```sh
PYTHONPATH=DoomGen2/src python DoomGen2/examples/benchmark/file_name.py --acc DoomGen2/acc/acc
make -j # refresh the package copy of scenarios/
```

Skeleton/API on how to build a map can be checked from examples in the library.

## Route B: patch an existing WAD

When the geometry already exists, edit it instead of regenerating it.
`scripts/wad/build_multi_duel_hide_and_seek_wad.py` builds Hide and Seek from
`multi_duel.wad` with plain Python: it reads the lumps with `struct`, edits the UDMF
`TEXTMAP` text (shortens the pit, adds two gate lines), compiles a standalone `.acs` with
`acc`, and replaces the `BEHAVIOR`/`SCRIPTS` lumps. omgifol offers the same operations as an
API (`omg.WAD`, `omg.UMapEditor` for UDMF, `omg.MapEditor` for Doom/Hexen format).

## Compiling ACS with `acc`

```sh
acc -i DoomGen2/acc my_scenario.acs my_scenario.o     # -i: directory with zcommon.acs
```

The output object is the `BEHAVIOR` lump. Start scripts with `#include "zcommon.acs"`.
A clean compile proves only that ACC knows the names: ACC 1.60 also declares GZDoom,
Zandronum, Eternity and ZDaemon functions that ViZDoom does not run. Check every call
against `acs_functions.md` / `line_specials.md` (see `acs_unsupported.md` for symptoms).

## Route C: inspect or edit by hand in SLADE / Ultimate Doom Builder

Agents cannot use the GUI, but a human can open a generated WAD to check geometry, thing
placement and tags, or draw a map from scratch. Use the **ZDoom (UDMF)** map format, set
`freedoom2.wad` as the base resource and `vizdoom.pk3` as the ZDoom PK3, and point the
ACS compiler setting at `acc` (`docs/environments/slade_config.md`). No node builder is
needed. In SLADE choose `ZDoom` as the source port. Ultimate Doom Builder is Windows-first;
SLADE runs on Linux, macOS and Windows.

## Testing tools in this skill

| Script | Level |
|---|---|
| `scripts/validate_scenario.py` | engine load + `USER` contract probe, 1-16 players |
| `scripts/capture_multiagent_scenario.py` | per-player frames, poses, notifications, audio and a top-down map for a multimodal agent |
| `scripts/play_netgame.sh` | N-player human playtest (host + joins) |
