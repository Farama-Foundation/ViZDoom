#ifndef __VIZIA_DOOM_UTILITIES_H__
#define __VIZIA_DOOM_UTILITIES_H__

#include "ViziaDoomDefines.h"

namespace Vizia {

    double DoomTics2Ms(double tics);

    double Ms2DoomTics(double ms);

    double DoomFixedToDouble(int doomFixed);

    bool isBinaryButton(Button button);

    bool isDeltaButton(Button button);
}

#endif