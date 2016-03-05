#include "ViZDoomUtilities.h"

namespace vizdoom {

    double DoomTicsToMs(double tics) {
        return (double) 1000 / 35 * tics;
    }

    double MsToDoomTics(double ms) {
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