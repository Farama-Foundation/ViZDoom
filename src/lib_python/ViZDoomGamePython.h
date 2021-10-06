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

#include "ViZDoomGame.h"

#include <iostream>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

namespace vizdoom {

    namespace pyb = pybind11;

    class ReleaseGIL {
    public:
        inline ReleaseGIL(){
            state = PyEval_SaveThread();
        }

        inline ~ReleaseGIL(){
            PyEval_RestoreThread(state);
        }
    private:
        PyThreadState *state;
    };

    class AcquireGIL {
    public:
        inline AcquireGIL(){
            state = PyGILState_Ensure();
        }

        inline ~AcquireGIL(){
            PyGILState_Release(state);
        }
    private:
        PyGILState_STATE state;
    };

    struct SectorPython {
        double floorHeight;
        double ceilingHeight;
        pyb::list lines;
    };

    struct GameStatePython {
        unsigned int number;
        unsigned int tic;

        pyb::object gameVariables;
        //pyb::list gameVariables;

        pyb::object screenBuffer;
        pyb::object depthBuffer;
        pyb::object labelsBuffer;
        pyb::object automapBuffer;
        pyb::object audioBuffer;

        pyb::list labels;
        pyb::list objects;
        pyb::list sectors;
    };

    struct ServerStatePython {
        unsigned int tic;
        unsigned int playerCount;
        pyb::list playersInGame;
        pyb::list playersFrags;
        pyb::list playersNames;
        pyb::list playersAfk;
        pyb::list playersLastActionTic;
        pyb::list playersLastKillTic;
    };

    class DoomGamePython : public DoomGame {

    public:
        DoomGamePython();

        void setAction(pyb::object const &pyAction);
        double makeAction(pyb::object const &pyAction, unsigned int tics = 1);

        GameStatePython* getState();
        ServerStatePython* getServerState();
        pyb::list getLastAction();

        pyb::list getAvailableButtons();
        void setAvailableButtons(pyb::list const &pyButtons);

        pyb::list getAvailableGameVariables();
        void setAvailableGameVariables(pyb::list const &pyGameVariables);

        // These functions are wrapped for manual GIL management
        void init();
        void newEpisode(std::string filePath = "");
        void advanceAction(unsigned int tics = 1, bool updateState = true);
        void respawnPlayer();

        // Overloaded functions instead of default arguments for pybind11

        void newEpisode_() { this->newEpisode(); };
        void newEpisode_str(std::string _str) { this->newEpisode(_str); };

//        double makeAction_list(pyb::list const &_list){ return this->makeAction(_list); }
//        double makeAction_list_int(pyb::list const &_list, unsigned int _int){ return this->makeAction(_list, _int); }

        double makeAction_list(pyb::object const &_list){ return this->makeAction(_list); }
        double makeAction_list_int(pyb::object const &_list, unsigned int _int){ return this->makeAction(_list, _int); }

        void advanceAction_() { this->advanceAction(); }
        void advanceAction_int(unsigned int _int) { this->advanceAction(_int); }
        void advanceAction_int_bool(unsigned int _int, bool _bool) { this->advanceAction(_int, _bool); }

        void addAvailableButton_btn(Button _btn) { this->addAvailableButton(_btn); }
        void addAvailableButton_btn_int(Button _btn, double _double) { this->addAvailableButton(_btn, _double); }

        void replayEpisode_str(std::string _str) { this->replayEpisode(_str); }
        void replayEpisode_str_int(std::string _str, unsigned int _int) { this->replayEpisode(_str, _int); }

    private:
        GameStatePython* pyState;

        std::vector<pyb::ssize_t> colorShape;
        std::vector<pyb::ssize_t> grayShape;
        std::vector<pyb::ssize_t> audioShape;
        std::vector<pyb::ssize_t> variablesShape;

        void updateBuffersShapes();

        template<class T> static std::vector<T> pyListToVector(pyb::list const &pyList);
        template<class T> static std::vector<T> pyArrayToVector(pyb::array_t<T> const &pyArray);
        template<class T> static std::vector<T> pyObjectToVector(pyb::object const &pyObject);

        template<class T> static pyb::list vectorToPyList(const std::vector<T>& vector);
        template<class T> static pyb::array_t<T> dataToNumpyArray(std::vector<pyb::ssize_t> dims, T *data);
    };

}

#endif
