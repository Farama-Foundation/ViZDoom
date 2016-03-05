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
