# Changelog

## Changes in 1.1

#### Buffers

- Depth buffer is now separate buffer in state and `ScreenFormat` values with it was removed - `is/setDepthBufferEnabled` added.
- Added in frame actors labeling feature -`is/setLabelsBufferEnabled` added.
- Added buffer with in game automap - `is/setAutomapBufferEnabled`, `setAutomapMode`, `setAutomapRoate`, `setAutomapRenderTextures`, `AutomapMode` enum added.


#### GameState

- `getState` will now return `nullptr/null/None` if game is in the terminal state.
- `imageBuffer` renamed to `screenBuffer`.
- Added `depthBuffer`, `labelsBuffer` and `automapBuffer` and `labels` fields.


#### Rendering options

- The option to use minimal hud instead of default full hud - `setRenderMinimalHud` added.
- The option to enable/disable effects that use sprites - `setRenderEffectsSprites` added.
- The option to enable/disable in game messages independently of the console output - `setRenderMessages` added.
- The option to enable/disable corpses - `setRenderCorpses` added.


#### Episode recording and replaying

- The option to record and replaying episodes, based on adapted ZDoom's demo mechanism - 
recording `filePath` argument added to `newEpisode`, `replayEpisode` added.
- The option to replay demo from other player perspective.


#### Ticrate

- The option to set number of tics executed per second in ASNYC Modes.
- New `ticrate` optional argument in `doomTicsToMs`, `msToDoomTics`.
- `doomTicsToSec` and `secToDoomTics` added.


#### Paths

- **Paths in config files are now relative to config file.**
- setting **vizdoom_path** and **doom_game_path** is no longer needed - they default to location(installation) of vizdoom.so.


#### Others

- ZDoom engine updated to 2.8.1.
- **Basic support for multiplayer in PLAYER and SPECTATOR Modes.**
- Improved exceptions messages.
- Bugs associated with paths handling fixed.
- Many minor bugs fixed.
- Possibility to change scenario wad during runtime (only first map from WAD file).


#### C++ specific

- A lot of overloaded methods turned into a methods with default arguments.
- `getState()` now returns `GameStatePtr (std::shared_ptr<GameState>)` instead of `GameState`.
- Buffers are now copied.
- GameState's buffer has now `BufferPtr (std::shared_ptr<Buffer>)` type - `Buffer (std::vector<uint8_t>)`.
- GameState's gameVariables are now vector of doubles instead of ints.


#### Lua specific

- Lua binding added.
- Support for LuaRocks installation for Linux and MacOS.


#### Java specific

- GameState buffers type changed to byte[].
- Performance improved.
- Java exceptions handling fixed.
- Few functions fixed.


#### Python specific

- Consts added to Python.
- Aliases for `doom_fixed_to_double` - `doom_fixed_to_float` added.
- Support for pip installation for Linux, MacOS.



 

