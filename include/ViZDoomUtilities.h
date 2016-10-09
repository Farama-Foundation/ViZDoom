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

#ifndef __VIZDOOM_UTILITIES_H__
#define __VIZDOOM_UTILITIES_H__

#include "ViZDoomConsts.h"
#include "ViZDoomTypes.h"

namespace vizdoom {

    /*
     * Calculates how many tics will be made during given number of milliseconds.
     */
    double doomTicsToMs(double tics, unsigned int ticrate = DEFAULT_TICRATE);

    /*
     * Calculates the number of milliseconds that will pass during specified number of tics.
     */
    double msToDoomTics(double ms, unsigned int ticrate = DEFAULT_TICRATE);

    /*
     * Calculates how many tics will be made during given number of seconds.
     */
    double doomTicsToSec(double tics, unsigned int ticrate = DEFAULT_TICRATE);

    /*
     * Calculates the number of seconds that will pass during specified number of tics.
     */
    double secToDoomTics(double sec, unsigned int ticrate = DEFAULT_TICRATE);

    /*
     * Converts Doom's fixed point numeral to a floating point value.
     */
    double doomFixedToDouble(int doomFixed);
    double doomFixedToDouble(double doomFixed);

    /*
     * Returns true if button is binary button.
     */
    bool isBinaryButton(Button button);

    /*
     * Returns true if button is delta button.
     */
    bool isDeltaButton(Button button);
}

#endif