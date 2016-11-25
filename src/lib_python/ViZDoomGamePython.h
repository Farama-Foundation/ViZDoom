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

    // For GCC versions lower then 5 compatibility
    // Python version of Label struct with Python string instead C++ string type.
    struct LabelPython{
        unsigned int objectId;
        bpy::str objectName;
        uint8_t value;
        double objectPositionX;
        double objectPositionY;
        double objectPositionZ;
    };

    struct GameStatePython {
        unsigned int number;

        bpya::object gameVariables;
        //bpy::list gameVariables;

        bpya::object screenBuffer;
        bpya::object depthBuffer;
        bpya::object labelsBuffer;
        bpya::object automapBuffer;

        bpy::list labels;
    };

    class DoomGamePython : public DoomGame {
        
    public:        
        DoomGamePython();

        void setAction(bpy::list const &pyAction);
        double makeAction(bpy::list const &pyAction, unsigned int tics = 1);

        GameStatePython getState();
        bpy::list getLastAction();

        bpy::list getAvailableButtons();
        void setAvailableButtons(bpy::list const &pyButtons);

        bpy::list getAvailableGameVariables();
        void setAvailableGameVariables(bpy::list const &pyGameVariables);


        // These functions are workaround for
        // "TypeError: No registered converter was able to produce a C++ rvalue of type std::string from this Python object of type str"
        // on GCC versions lower then 5
        bool loadConfig(bpy::str const &pyPath);

        void newEpisode();
        void newEpisode(bpy::str const &pyPath);
        void replayEpisode(bpy::str const &pyPath, unsigned int player = 0);

        void setViZDoomPath(bpy::str const &pyPath);
        void setDoomGamePath(bpy::str const &pyPath);
        void setDoomScenarioPath(bpy::str const &pyPath);
        void setDoomMap(bpy::str const &pyMap);
        void setDoomConfigPath(bpy::str const &pyPath);
        void addGameArgs(bpy::str const &pyArgs);
        void sendGameCommand(bpy::str const &pyCmd);

    private:
        npy_intp colorShape[3];
        npy_intp grayShape[2];

        void updateBuffersShapes();

        template<class T> static bpy::list vectorToPyList(const std::vector<T>& vector);
        template<class T> static std::vector<T> pyListToVector(bpy::list const &pyList);

        static bpy::object dataToNumpyArray(int dims, npy_intp * shape, int type, void * data);

    };

}

#endif
