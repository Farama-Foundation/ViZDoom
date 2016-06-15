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

#ifndef __VIZDOOM_GAME_PYTHON_H__
#define __VIZDOOM_GAME_PYTHON_H__

#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION

#include "ViZDoomGame.h"

#include <iostream>
#include <Python.h>
#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>
#include <numpy/npy_math.h>
#include <vector>

namespace vizdoom {

    namespace bpy       = boost::python;
    namespace bpya      = bpy::api;
    namespace bpyn      = bpy::numeric;

    struct GameStatePython {
        unsigned int number;
        bpya::object imageBuffer;
        bpya::object gameVariables;
        GameStatePython(int n, bpya::object buf, bpya::object v ):number(n),imageBuffer(buf),gameVariables(v){}
        GameStatePython(int n, bpya::object buf):number(n),imageBuffer(buf){}
        GameStatePython(int n):number(n){}
    };

    class DoomGamePython : public DoomGame {
        
    public:        
        DoomGamePython();
        bool init();
        
        GameStatePython getState();
        bpy::list getLastAction();
        bpya::object getGameScreen();
        void setAction(bpy::list const &action);
        double makeAction(bpy::list const &action);
        double makeAction(bpy::list const &action, unsigned int tics);

        // These functions are workaround for
        // "TypeError: No registered converter was able to produce a C++ rvalue of type std::string from this Python object of type str"
        // on GCC versions lower then 5
        bool loadConfig(bpy::str const &pyPath);

        void newEpisode();
        void newEpisode(bpy::str const &pyPath);
        void replayEpisode(bpy::str const &pyPath);

        void setViZDoomPath(bpy::str const &pyPath);
        void setDoomGamePath(bpy::str const &pyPath);
        void setDoomScenarioPath(bpy::str const &pyPath);
        void setDoomMap(bpy::str const &pyMap);
        void setDoomConfigPath(bpy::str const &pyPath);
        void addGameArgs(bpy::str const &pyArgs);
        void sendGameCommand(bpy::str const &pyCmd);

    private:
        npy_intp imageShape[3];
        static std::vector<int> pyListToIntVector(bpy::list const &action);

    };

}

#endif
