# Changelog

## Changes in 1.1

#### Buffers

* Depth buffer is now separate buffer in state and ScreenFormat values with it was removed - `is/setDepthBufferEnabled` added.
* Added in frame actors labeling feature -`is/setLabelsBufferEnabled` added.
* Added buffer with in game automap - `is/setAutomapBufferEnabled`, `setAutomapMode`, `setAutomapRoate`, `setAutomapRenderTextures`, AutomapMode enum added.

#### GameState

* `getState` will now return nullptr/null/None if game is in the terminal state.
* `imageBuffer` renamed to `screenBuffer`.
* Added `depthBuffer`, `labelsBuffer` and `automapBuffer` and `labels` fields.

#### Rendering options

* The option to use minimal hud instead of default - `setRenderMinimalHud` added.
* The option to enable/disable effects that use sprites - `setRenderEffectsSprites` added.
* The option to render ingame messages independently of the console output - `setRenderMessages` added.

#### Episode recording and replaying
...

#### Ticrate
* The option to set number of tics executed per second in ASNYC Modes.

#### Config loading
...

#### Others
* Improved exceptions messages.
* Java exceptions handling fixed.
* Bugs associated with paths handling fixed.
* Many minor bugs fixed.

#### C++ specific

* A lot of overloaded methods turned into a methods with default arguments.
* `getState()` now returns GameStatePtr (std::shared_ptr<GameState>) instead of GameState.
* Buffers are now copied.
* GameState's buffer has now BufferPtr (std::shared_ptr<Buffer>) type - Buffer (std::vector<uint8_t>).
