/*
 Copyright (C) 2016 by Wojciech Jaśkowski, Michał Kempka, Grzegorz Runc, Jakub Toczek, Marek Wydmuch

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

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