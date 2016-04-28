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
#include <vector>
#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <boost/python/tuple.hpp>
#include <numpy/ndarrayobject.h>
#include <numpy/npy_math.h>

namespace vizdoom {
    using boost::python::api::object;
/* C++ code to expose C arrays as python objects */

    struct GameStatePython {
        unsigned int number;
        object imageBuffer;
        object gameVariables;
        GameStatePython(int n, object buf, object v ):number(n),imageBuffer(buf),gameVariables(v){}
        GameStatePython(int n, object buf):number(n),imageBuffer(buf){}
        GameStatePython(int n):number(n){}
    };

    class DoomGamePython : public DoomGame {
        
    public:        
        DoomGamePython();
        bool init();
        
        GameStatePython getState();
        boost::python::list getLastAction();
        object getGameScreen();
        void setAction(boost::python::list const &action);
        double makeAction(boost::python::list const &action);
        double makeAction(boost::python::list const &action, unsigned int tics);

    private:

        npy_intp imageShape[3];
        static std::vector<int> pyListToIntVector(boost::python::list const &action);

    };

}

#endif
