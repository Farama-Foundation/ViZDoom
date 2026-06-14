#ifndef __VIZDOOM_OBJECTS_DOCSTRINGS_H__
#define __VIZDOOM_OBJECTS_DOCSTRINGS_H__

namespace vizdoom {
namespace docstrings {
    
    const char *Mode = R"DOCSTRING(Defines the mode for controlling the game.)DOCSTRING";
    const char *ScreenFormat = R"DOCSTRING(Defines the format of the screen buffer.)DOCSTRING";
    const char *ScreenResolution = R"DOCSTRING(Defines the resolution of the screen buffer. Available resolutions include various predefined sizes like RES_320x240, etc.)DOCSTRING";
    const char *AutomapMode = R"DOCSTRING(Defines the automap rendering mode.)DOCSTRING";
    const char *Button = R"DOCSTRING(Defines available game buttons/actions that can be used to control the game.)DOCSTRING";
    const char *GameVariable = R"DOCSTRING(Defines available game variables that can be accessed to get information about the game state.)DOCSTRING";
    const char *SamplingRate = R"DOCSTRING(Defines available audio sampling rates.)DOCSTRING";
    const char *Label = R"DOCSTRING(Represents object labels in the game world with associated properties.)DOCSTRING";
    const char *Line = R"DOCSTRING(Represents line segments in the game world geometry.)DOCSTRING";
    const char *Sector = R"DOCSTRING(Represents sectors (floor/ceiling areas) in the game world geometry.)DOCSTRING";
    const char *Object = R"DOCSTRING(Represents objects in the game world with position and other properties.)DOCSTRING";
    const char *ServerState = R"DOCSTRING(Contains the state of the multiplayer server.)DOCSTRING";
    const char *GameState = R"DOCSTRING(Contains the state of the game including screen buffer, game variables, and world geometry, available information depends on the configuration of the game instance.)DOCSTRING";
    const char *DoomGame = R"DOCSTRING(DoomGame is the main object of the ViZDoom library, representing a single instance of the Doom game and providing the interface for a single agent/player to interact with the game. The object allows sending actions to the game, getting the game state, etc.)DOCSTRING";

} // namespace docstrings
} // namespace vizdoom

#endif
