#ifndef __VIZDOOM_UTILITIES_H__
#define __VIZDOOM_UTILITIES_H__

#include "ViZDoomDefines.h"

namespace vizdoom {

    double DoomTicsToMs(double tics);

    double MsToDoomTics(double ms);

    double DoomFixedToDouble(int doomFixed);

    bool isBinaryButton(Button button);

    bool isDeltaButton(Button button);
}

#endif