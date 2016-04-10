#include "ViZDoomUtilities.h"

namespace vizdoom {

    double DoomTicsToMs(double tics) {
        return static_cast<double>(1000) / 35 * tics;
    }

    double MsToDoomTics(double ms) {
        return static_cast<double>(35) / 1000 * ms;
    }

    double DoomFixedToDouble(int doomFixed) {
        double res = static_cast<double>(doomFixed) / 65536.0;
        return res;
    }

    bool isBinaryButton(Button button){
        return button < BinaryButtonsNumber;
    }

    bool isDeltaButton(Button button){
        // return button >= BinaryButtonsNumber && button < (BinaryButtonsNumber + DeltaButtonsNumber);
        return button >= BinaryButtonsNumber && button < ButtonsNumber;
    }
}