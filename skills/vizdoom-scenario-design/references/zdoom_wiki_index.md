# Curated ZDoom wiki index (for ViZDoom's ZDoom 2.8.1 engine)

The ZDoom wiki (<https://zdoom.org/wiki>) is the best description of what ACS functions,
line specials, DECORATE and MAPINFO do, but it documents the **current GZDoom**. ViZDoom runs
a ZDoom 2.8.1 fork, so:

- use a page only for a name that is in `acs_functions.md`, in `line_specials.md` with
  `Works = yes`, or in `actor_classes.md`;
- ignore arguments, flags and properties that the page marks as added in a GZDoom version;
- ZScript (introduced by GZDoom 2.4) is not available: write actors in DECORATE and logic in ACS.

Tags: **works** = name checked against the generated tables (all links checked on
2026-10-26); **not in ViZDoom** = compiles or exists in GZDoom but does nothing here;
**reference** = overview page, apply the version filter above.

## Language and structure

| Page | Tag | Why it matters |
|---|---|---|
| [ACS](https://zdoom.org/wiki/ACS) | reference | language overview |
| [Script types](https://zdoom.org/wiki/Script_types) | reference | `OPEN`, `ENTER`, `RESPAWN`, `DEATH`, `NET` scripts |
| [Scripts](https://zdoom.org/wiki/Scripts) | reference | script syntax, arguments, `Delay` and termination |
| [Built-in ACS functions](https://zdoom.org/wiki/Built-in_ACS_functions) | reference | full GZDoom list: filter through `acs_functions.md` |
| [Action specials](https://zdoom.org/wiki/Action_specials) | reference | full list: filter through `line_specials.md` |
| [Fixed point number](https://zdoom.org/wiki/Fixed_point_number) | reference | `1.0` vs `1`; the ViZDoom reward is fixed point |
| [UDMF](https://zdoom.org/wiki/UDMF) | reference | the text map format DoomGen writes |
| [MAPINFO](https://zdoom.org/wiki/MAPINFO) | reference | map definitions, DoomEdNums and SpawnNums |
| [SNDINFO](https://zdoom.org/wiki/SNDINFO) | reference | registering custom sounds (Simple Tag Audio's chirp) |
| [Editor number](https://zdoom.org/wiki/Editor_number) | reference | thing types in maps (use `actor_classes.md` numbers) |
| [Spawn number](https://zdoom.org/wiki/Spawn_number) | reference | ids for `Thing_Spawn` (use `actor_classes.md` numbers) |

## Actors (DECORATE)

| Page | Tag | Why it matters |
|---|---|---|
| [DECORATE](https://zdoom.org/wiki/DECORATE) | reference | `ACTOR Name : Parent` definitions |
| [Actor properties](https://zdoom.org/wiki/Actor_properties) | reference | speed, health, render style; check version notes |
| [Actor flags](https://zdoom.org/wiki/Actor_flags) | reference | e.g. `NODAMAGE`, `NOTELEFRAG`; check version notes |
| [Actor states](https://zdoom.org/wiki/Actor_states) | reference | state labels used from ACS (`SetActorState`) |
| [Inventory](https://zdoom.org/wiki/Inventory) | reference | base class of items and powerups |
| [Classes:Doom](https://zdoom.org/wiki/Classes:Doom) | reference | Doom classes; the authoritative list is `actor_classes.md` |

## ACS functions used by the repo's scenarios

| Page | Tag | Typical use |
|---|---|---|
| [SpawnSpot](https://zdoom.org/wiki/SpawnSpot) | works | spawn an actor at a `MapSpot` |
| [SpawnForced](https://zdoom.org/wiki/SpawnForced) | works | spawn even if blocked |
| [Thing_Spawn](https://zdoom.org/wiki/Thing_Spawn) | works | spawn by spawn number |
| [Thing_Remove](https://zdoom.org/wiki/Thing_Remove) | works | remove a tagged thing |
| [Thing_ChangeTID](https://zdoom.org/wiki/Thing_ChangeTID) | works | retag spawned actors |
| [Thing_SetTranslation](https://zdoom.org/wiki/Thing_SetTranslation) | works | recolour players/actors by role |
| [CreateTranslation](https://zdoom.org/wiki/CreateTranslation) | works | define the colour translation |
| [Thing_Damage](https://zdoom.org/wiki/Thing_Damage) | works | damage or kill tagged things |
| [SpawnProjectile](https://zdoom.org/wiki/SpawnProjectile) | works | scripted projectiles fired by an actor (Hide and Seek's evaluation turret) |
| [SetActorVelocity](https://zdoom.org/wiki/SetActorVelocity) | works | exact projectile velocity (`SpawnProjectile` only takes a byte angle) |
| [SetActorProperty](https://zdoom.org/wiki/SetActorProperty) | works | speed, render style, alpha per role |
| [GetActorProperty](https://zdoom.org/wiki/GetActorProperty) | works | read health, speed |
| [GetActorX](https://zdoom.org/wiki/GetActorX) | works | positions (fixed point) for distance checks |
| [CheckSight](https://zdoom.org/wiki/CheckSight) | works | line-of-sight conditions (Simple Tag capture) |
| [SetActivator](https://zdoom.org/wiki/SetActivator) | works | make a script act on a specific actor (use instead of `SetActivatorToPlayer`) |
| [ActivatorTID](https://zdoom.org/wiki/ActivatorTID) | works | tag of the current activator |
| [PlayerNumber](https://zdoom.org/wiki/PlayerNumber) | works | network slot of the activator |
| [SetPlayerProperty](https://zdoom.org/wiki/SetPlayerProperty) | works | freeze players (`PROP_TOTALLYFROZEN`) |
| [ClearInventory](https://zdoom.org/wiki/ClearInventory) | works | strip weapons in `ENTER`/`RESPAWN` |
| [SetThingSpecial](https://zdoom.org/wiki/SetThingSpecial) | works | attach a special to a thing |
| [HudMessage](https://zdoom.org/wiki/HudMessage) | works | private on-screen text (`HUDMSG_LOG` also feeds the notifications buffer) |
| [SetHudSize](https://zdoom.org/wiki/SetHudSize) | works | legible HUD text at low resolutions |
| [PlaySound](https://zdoom.org/wiki/PlaySound) | works | positional sounds; argument order `tid, sound, channel, volume, looping, attenuation` |
| [Random](https://zdoom.org/wiki/Random) | works | per-episode randomisation (no re-seed between netgame episodes) |
| [Delay](https://zdoom.org/wiki/Delay) | works | wait in tics |
| [Floor_MoveToValue](https://zdoom.org/wiki/Floor_MoveToValue) | works | raise/lower cover or doors per episode |
| [Teleport_NoFog](https://zdoom.org/wiki/Teleport_NoFog) | works | place players without fog |
| [Sector_SetDamage](https://zdoom.org/wiki/Sector_SetDamage) | works | make a floor damage players from a script |
| [ACS_Execute](https://zdoom.org/wiki/ACS_Execute) | works | start another script |
| [Exit_Normal](https://zdoom.org/wiki/Exit_Normal) | works | end the map (single player); multi-agent scenarios usually terminate from Python on a `USER` flag instead |

## Not in ViZDoom

| Page | Tag | Note |
|---|---|---|
| [ZScript](https://zdoom.org/wiki/ZScript) | not in ViZDoom | GZDoom 2.4+ scripting language |
| [SetSectorGlow](https://zdoom.org/wiki/SetSectorGlow) | not in ViZDoom | GZDoom (OpenGL) extension function; silently returns 0 |
| `SetActivatorToPlayer` | not in ViZDoom | Zandronum; no wiki page, silently returns 0, use `SetActivator(tid)` |
| `GetLineX` / `GetLineY` | not in ViZDoom | Eternity; no wiki page |

The complete list of compile-but-do-not-work names is `acs_unsupported.md`.

## Editors

| Page | Note |
|---|---|
| [SLADE](https://zdoom.org/wiki/SLADE) | cross-platform WAD and map editor |
| [Ultimate Doom Builder](https://zdoom.org/wiki/Ultimate_Doom_Builder) | map editor (Windows-first) |
| [Doom Builder](https://zdoom.org/wiki/Doom_Builder) | the original editor UDB descends from |
| [ZDBSP](https://zdoom.org/wiki/ZDBSP) | node builder (DoomGen runs it; ViZDoom also builds nodes at load time) |
