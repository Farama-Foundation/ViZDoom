#include "ViziaDoomUtilities.h"

namespace Vizia {

    double DoomTics2Ms(double tics) {
        return (double) 1000 / 35 * tics;
    }

    double Ms2DoomTics(double ms) {
        return (double) 35 / 1000 * ms;
    }

    double DoomFixedToDouble(int doomFixed) {
        double res = double(doomFixed) / 65536.0;
        return res;
    }

    bool isBinaryButton(Button button){
        return button < BinaryButtonsNumber;
    }

    bool isDeltaButton(Button button){
        return button >= BinaryButtonsNumber && button < BinaryButtonsNumber + DeltaButtonsNumber;
    }
}