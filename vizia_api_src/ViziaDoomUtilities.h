#ifndef __VIZIA_DOOM_UTILITIES_H__
#define __VIZIA_DOOM_UTILITIES_H__

#include "ViziaDoomDefines.h"

namespace Vizia {

    double DoomTicsToMs(double tics);

    double MsToDoomTics(double ms);

    double DoomFixedToDouble(int doomFixed);

    bool isBinaryButton(Button button);

    bool isDeltaButton(Button button);
}

#endif