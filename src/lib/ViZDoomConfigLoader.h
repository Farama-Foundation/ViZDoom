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

#ifndef __VIZDOOM_CONFIGLOADER_H__
#define __VIZDOOM_CONFIGLOADER_H__

#include "ViZDoomTypes.h"
#include "ViZDoomGame.h"

#include <string>

namespace vizdoom {

    class ConfigLoader {
    public:

        ConfigLoader(DoomGame *game);

        ~ConfigLoader();

        bool load(std::string filePath);

    protected:

        /* Load config helpers */
        /*------------------------------------------------------------------------------------------------------------*/

        static bool stringToBool(std::string boolString);

        static int stringToInt(std::string str);

        static unsigned int stringToUint(std::string str);

        static ScreenResolution stringToResolution(std::string str);

        static ScreenFormat stringToFormat(std::string str);

        static Button stringToButton(std::string str);

        static GameVariable stringToGameVariable(std::string str);

        static bool
        parseListProperty(int &line_number, std::string &value, std::ifstream &input, std::vector<std::string> &output);

    private:
        DoomGame *game;
        std::string filePath;
    };
}

#endif
