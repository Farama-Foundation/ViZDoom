#ifndef VIZDOOMGAME_LUA_H
#define VIZDOOMGAME_LUA_H


#include "ViZDoom.h"
#include "THStorage.h"
#include "THTensor.h"

#include <cstdlib>
#include <vector>


typedef vizdoom::DoomGame DoomGame;
typedef vizdoom::GameState GameState;
typedef vizdoom::ScreenResolution ScreenResolution;
typedef vizdoom::ScreenFormat ScreenFormat;
typedef vizdoom::GameVariable GameVariable;
typedef vizdoom::Button Button;
typedef vizdoom::Mode Mode;

extern "C" {
#include "vizdoom.inl"
}



#endif
